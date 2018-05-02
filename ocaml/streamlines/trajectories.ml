(** {v Copyright (C) 2017-2018,  Colin P Stark and Gavin J Stark.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * @file   trajectories.ml
 * @brief  Integration of trajectories using GPU for trace
 *
 * Up to date with python of git CS 189bfccdabc3371eafe8bcafa3bdfa8c241e56e4
 * Except  max_time_per_kernel and initial_size_factor are hardwired
 *
 * Note that integrate_trajectories does NOT create seeds - that is up to the client
 *
 * v}
 *)

(*a Module abbreviations *)
open Globals
open Core
open Properties
module ODM = Owl.Dense.Matrix.Generic
module ODN = Owl.Dense.Ndarray.Generic
module OS  = Owl.Stats

(** {1 pv_verbosity functions} *)

(**  [pv_noisy t]

  Shortcut to use {!type:Properties.t_props_trace} verbosity for {!val:Properties.pv_noisy}

 *)
let pv_noisy   data = Workflow.pv_noisy data.properties.trace.workflow

(**  [pv_debug t]

  Shortcut to use {!type:Properties.t_props_trace} verbosity for {!val:Properties.pv_debug}

 *)
let pv_debug   data = Workflow.pv_noisy data.properties.trace.workflow

(**  [pv_info t]

  Shortcut to use {!type:Properties.t_props_trace} verbosity for {!val:Properties.pv_info}

 *)
let pv_info    data = Workflow.pv_info data.properties.trace.workflow

(**  [pv_verbose t]

  Shortcut to use {!type:Properties.t_props_trace} verbosity for {!val:Properties.pv_verbose}

 *)
let pv_verbose data = Workflow.pv_verbose data.properties.trace.workflow

(** {1 Statics} *)

let cl_files = ["rng.cl";"essentials.cl";
                "writearray.cl";"trajectoryfns.cl";"computestep.cl";
                "integrationfns.cl";"trajectory.cl";"integratetraj.cl"]
let cl_src_path = ["opencl"]

(** {1 Chunks} *)

(**  [t_chunk] *)
type t_chunk = {
    required           : bool; (* false if not required to be calculated *)
    direction          : string;
    downup_index       : int;   (* 0 for downstream, 1 for upstream - last index into arrays to put results *)
    downup_sign        : float; (* -1. for upstream, +1. for downstream *)
    chunk_index        : int; (* Which chunk number *)
    seed_offset        : int; (* Offset to first seed in chunk *)
    num_seeds          : int; (* Number of seeds in chunk *)
  }

(**  [chunk data downstream (n, seed_start, seed_end)]

  Create a chunk (index {i n}) with the given seeds, for upstream or downstream

 *)
let chunk data downstream nse =
  let (n, seed_start, seed_end) = nse in
  let (required, direction, downup_index, downup_sign) = 
    if downstream then
      (data.properties.trace.do_trace_downstream, "Downstream", 0,(+1.))
    else
      data.properties.trace.do_trace_upstream,   "Upstream",   1,(-1.)
  in
  {
    required; direction; downup_index; downup_sign; chunk_index=n; seed_offset=seed_start; num_seeds=(seed_end-seed_start)
  }

(**  [show_chunk t] 

  Not yet implemented

*)
let show_chunk t =
  Printf.printf "Chunk %d\n" t.chunk_index;
  Printf.printf "  %s (required %b, index %d, sign %f)" t.direction t.required t.downup_index t.downup_sign;
  Printf.printf "  from seed %d for %d seeds\n%!" t.seed_offset t.num_seeds

(**  [generate_chunks data num_seeds chunk_size]

  Generate a to_do_list of chunks

 *)
let generate_chunks data num_seeds chunk_size =
  let num_chunks_required = required_units num_seeds chunk_size in
  let chunks = List.init num_chunks_required (fun i -> (i, i*chunk_size, min ((i+1)*chunk_size) num_seeds)) in
  let to_do_list = List.fold_left (fun acc nse -> (chunk data true nse)::(chunk data false nse)::acc) [] chunks in
  to_do_list

(** {1 Memory} *)

(**  t_memory

  Structure containing the memory and OpenCL buffers for interacting
  with the integrate_trajectories kernel.

 *)
type t_memory = {
    uv_array           : t_ba_floats;  (* padded ROI - vector field array input *)
    mapping_array      : t_ba_ints;    (* padded ROI - #streamlines crossing pixel *)
    chunk_nsteps_array : t_ba_int16s;  (* chunk_size - number of steps taken for a streamline from the seed *)
    chunk_length_array : t_ba_floats;  (* chunk_size - streamline length from the seed *)
    chunk_trajcs_array : t_ba_chars;   (* chunk_size * max_traj_length*2 - char path of streamline from the seed *)

    seeds_buffer        : Pocl.t_buffer;
    uv_buffer           : Pocl.t_buffer;
    mask_buffer         : Pocl.t_buffer;
    mapping_buffer      : Pocl.t_buffer;
    chunk_trajcs_buffer : Pocl.t_buffer;
    chunk_nsteps_buffer : Pocl.t_buffer;
    chunk_length_buffer : Pocl.t_buffer;
  }

(**  [copy_read pocl]

  Shortcut for creating an OpenCL buffer that is read-only by the
  kernel and copied from a big array before execution

  @return OpenCL buffer

 *)
let copy_read       pocl = Pocl.buffer_of_array pocl ~copy:true true false

(**  [copy_read_write pocl]

  Shortcut for creating an OpenCL buffer that is read-write by the
  kernel and copied from a big array before execution

  @return OpenCL buffer

 *)
let copy_read_write pocl = Pocl.buffer_of_array pocl ~copy:true true true

(**  [write_only pocl]

  Shortcut for creating an OpenCL buffer that is write-only by the
  kernel

  @return OpenCL buffer

 *)
let write_only      pocl = Pocl.buffer_of_array pocl false true

(**  [memory_create_buffers pocl data seeds chunk_size]

  Create a t_memory strucure with big arrays and PyOpenCL buffers to
  allow CPU-GPU data transfer for the integrate_trajectories kernel

 *)
let memory_create_buffers pocl data seeds chunk_size =
  let max_traj_length = Info.int_of data.info "max_n_steps" in
  let roi_nx = data.pad_width*2+data.roi_nx in
  let roi_ny = data.pad_width*2+data.roi_ny in

  let uv_array = ba_float2d (roi_nx) (roi_ny*2) in
  let fill_uv x y u v = 
    ODM.set uv_array x (y*2+0) u;
    ODM.set uv_array x (y*2+1) v;
  in
  ODM.iter2i_2d fill_uv data.u_array data.v_array;
  let mapping_array = ba_int2d (roi_nx) (roi_ny) in
  ODM.fill mapping_array 0;

  (* Chunk-sized temporary arrays - use for a work group, then copy to traj_* *)
  (* Use "bag o' bytes" buffer for huge trajectories array. Write (by GPU) only. *)
  let chunk_trajcs_array = ba_char2d chunk_size (max_traj_length*2) in
  let chunk_nsteps_array = ba_int16s chunk_size in
  let chunk_length_array = ba_floats chunk_size in
  
  (* Create OpenCL buffers of the arrays *)
  let seeds_buffer        = copy_read pocl seeds in
  let mask_buffer         = copy_read pocl data.basin_mask_array in
  let uv_buffer           = copy_read pocl uv_array in
  let mapping_buffer      = copy_read_write pocl mapping_array in
  let chunk_nsteps_buffer = write_only pocl chunk_nsteps_array in
  let chunk_length_buffer = write_only pocl chunk_length_array in
  let chunk_trajcs_buffer = write_only pocl chunk_trajcs_array in

  let show_sizes _ =
    Printf.printf "Array sizes:\n";
    Printf.printf "ROI-type = %d,%d\n" roi_nx roi_ny ;
    let (x,y) = ODM.shape uv_array in
    Printf.printf "uv = %d,%d\n" x y;
    let (x,y) = ODM.shape chunk_trajcs_array in
    let (num_seeds,_) = ODM.shape seeds in
    Printf.printf "Streamlines virtual array allocation: %d,%d size %d\n" num_seeds y (num_seeds*y*max_traj_length*2);
    Printf.printf "Streamlines array allocation per chunk: %d,%d size %d\n%!" x y (ODM.size_in_bytes chunk_trajcs_array)
  in
  pv_verbose data show_sizes;
  {
    uv_array;
    mapping_array;
    chunk_nsteps_array;
    chunk_length_array;
    chunk_trajcs_array;

    seeds_buffer;
    mask_buffer;
    uv_buffer;
    mapping_buffer;
    chunk_nsteps_buffer;
    chunk_length_buffer;
    chunk_trajcs_buffer;
  }

(**  [memory_copyback t pocl]

  Copy back the chunk buffers from the GPU to the big arrays after execution
  of the integrate_trajectories kernel

 *)
let memory_copyback t pocl = 
  Pocl.copy_buffer_from_gpu pocl ~src:t.chunk_trajcs_buffer ~dst:t.chunk_trajcs_array;
  Pocl.copy_buffer_from_gpu pocl ~src:t.chunk_nsteps_buffer ~dst:t.chunk_nsteps_array;
  Pocl.copy_buffer_from_gpu pocl ~src:t.chunk_length_buffer ~dst:t.chunk_length_array;
  Pocl.finish_queue pocl

(** {1 GPU functions} *)

(**  [gpu_integrate_chunk pocl data memory streamline_lists results cl_kernel_source t]

  Integrate a chunk of seeds on the GPU - this is a set of consecutive
  seeds in either upstream or downstream - and aggregate the results

 *)
let gpu_integrate_chunk pocl data memory streamline_lists results cl_kernel_source t =
  if t.required then (
    pv_verbose data (fun _ -> show_chunk t);

    (* Specify this integration job's parameters and compile *)
    let grid_scale      = Info.float_of data.info "grid_scale" in
    Info.set data.info "downup_sign" (Info.Float32 t.downup_sign);
    Info.set data.info "seeds_chunk_offset" (Info.Int t.seed_offset);
    Info.set_float32 data.info "combo_factor" ((grid_scale *. data.properties.trace.integrator_step_factor) *. t.downup_sign);
    let compile_options = Pocl.compile_options pocl data.info "integrate_trajectory" in
    let program = Pocl.compile_program pocl cl_kernel_source compile_options in
    let kernel = Pocl.get_kernel pocl program "integrate_trajectory" in

    (* Execute the kernel *)
    Pocl.kernel_set_arg_buffer pocl kernel 0 memory.seeds_buffer;
    Pocl.kernel_set_arg_buffer pocl kernel 1 memory.mask_buffer;
    Pocl.kernel_set_arg_buffer pocl kernel 2 memory.uv_buffer;
    Pocl.kernel_set_arg_buffer pocl kernel 3 memory.mapping_buffer;
    Pocl.kernel_set_arg_buffer pocl kernel 4 memory.chunk_trajcs_buffer;
    Pocl.kernel_set_arg_buffer pocl kernel 5 memory.chunk_nsteps_buffer;
    Pocl.kernel_set_arg_buffer pocl kernel 6 memory.chunk_length_buffer;

    let n_work_items    = Info.int_of   data.info "n_work_items" in
    let global_size     = round_up_to_unit_size t.num_seeds n_work_items in
    let time_taken = Pocl.adaptive_enqueue_kernel pocl kernel global_size n_work_items in
    Printf.printf "\n##### Kernel lapsed time: %0.3f secs #####\n" time_taken;
    
    memory_copyback memory pocl;

    (* Compile streamline results *)
    pv_noisy data (fun _ -> Printf.printf "Building streamlines compressed array for chunk\n%!");
    for i=0 to t.num_seeds-1 do (* i is in range 0 to chunk_size-1 *)
      let seed_downup_index = 2*(i + t.seed_offset) + t.downup_index in
      let traj_nsteps = ODN.get memory.chunk_nsteps_array [|i|] in
      let traj_length = ODN.get memory.chunk_length_array [|i|] in
      let traj_vector n = ODN.get memory.chunk_trajcs_array [|i; n|] in
      ODN.set results.traj_nsteps_array  [|seed_downup_index|] traj_nsteps;
      ODN.set results.traj_lengths_array [|seed_downup_index|] traj_length;
      let streamlines_of_seed = Bytes.init (traj_nsteps*2) traj_vector in
      streamline_lists.(t.downup_index) <- streamlines_of_seed :: ((streamline_lists.(t.downup_index)));
    done;
    
  ) else (
     (* chunk was not required *)
  )

(**  [gpu_integrate_trajectories pocl data seeds chunk_size to_do_list]

  Carry out GPU computations in the chunks on {i to_do_list}.

  Each chunk is handled with a separate compilation, and results are
  aggregated.

 *)
let gpu_integrate_trajectories pocl data results seeds chunk_size to_do_list =

  let memory = memory_create_buffers pocl data seeds chunk_size in
  let streamline_lists = [| []; []; |] in
  let cl_kernel_source = Pocl.read_source cl_src_path cl_files in

  List.iter (fun chunk -> gpu_integrate_chunk pocl data memory streamline_lists results cl_kernel_source chunk) to_do_list;

  results.streamline_arrays <- [| Array.of_list (List.rev streamline_lists.(0)); Array.of_list (List.rev streamline_lists.(1)); |];

  let total_steps = Array.fold_left (fun acc t -> acc+(Bytes.length t)) 0           results.streamline_arrays.(0) in
  let total_steps = Array.fold_left (fun acc t -> acc+(Bytes.length t)) total_steps results.streamline_arrays.(1) in
  pv_verbose data (fun _ -> Printf.printf "Total steps in all streamlines %d\n%!" (total_steps/2));
  results

(**  [integrate_trajectories tprops pocl data results seeds]

  Integrate trajectories both upstream and downstream from the {i
  seeds} array of unpadded ROI coordinates.

  Trace each streamline from its corresponding seed point using 2nd-order Runge-Kutta 
  integration of the topographic gradient vector field.

  @return trace_results

 *)
  let get_traj_stats data results num_seeds downup_index =
    let pixel_size = Info.float_of data.info "pixel_size" in

    let data_of_seed seed =
      let seed_downup_index = 2*seed + downup_index in
      let ln0 = (ODN.get results.traj_lengths_array [|seed_downup_index|]) *. pixel_size in
      let ln1 = float (ODN.get results.traj_nsteps_array  [|seed_downup_index|]) in
      (ln0, ln1, ln0 /. ln1)
    in
    let line_stats = Array.init num_seeds data_of_seed in
    let lengths = Array.map (fun (x,_,_) -> x) line_stats in
    let counts  = Array.map (fun (_,x,_) -> x) line_stats in
    let dses    = Array.map (fun (_,_,x) -> x) line_stats in
    {l_mean=OS.mean lengths; l_min=OS.min lengths; l_max=OS.max lengths;
     c_mean=OS.mean counts;  c_min=OS.min counts;  c_max=OS.max counts;
     d_mean=OS.mean dses;    d_min=OS.min dses;    d_max=OS.max dses;
    }
  let show_stats results =
    let ds_stats = Option.get results.ds_stats in
    let us_stats = Option.get results.us_stats in
    Printf.printf "   downstream                            upstream\n";
    Printf.printf "          min        mean         max         min        mean         max\n";
    Printf.printf "l %11.6f %11.6f %11.6f %11.6f %11.6f %11.6f\n"    ds_stats.l_min ds_stats.l_mean ds_stats.l_max us_stats.l_min us_stats.l_mean us_stats.l_max ;
    Printf.printf "n %11.6f %11.6f %11.6f %11.6f %11.6f %11.6f\n"    ds_stats.c_min ds_stats.c_mean ds_stats.c_max us_stats.c_min us_stats.c_mean us_stats.c_max ;
    Printf.printf "ds%11.6f %11.6f %11.6f %11.6f %11.6f %11.6f\n%!"  ds_stats.d_min ds_stats.d_mean ds_stats.d_max us_stats.d_min us_stats.d_mean us_stats.d_max ;
    ()

let integrate_trajectories (tprops:t_props_trace) pocl data results seeds =
  Workflow.workflow_start ~subflow:"integrating trajectories" tprops.workflow;

  let gpu_traj_memory_limit = Pocl.get_memory_limit pocl in (* max memory permitted to use *)
  let (num_seeds,_) = ODM.shape seeds in
  let mem_per_seed = (Info.int_of data.info "max_n_steps") * 2 in (* an approximation *)
  let work_items_per_warp = Info.int_of data.info "n_work_items" in
  let max_chunk_size = gpu_traj_memory_limit / mem_per_seed in
  let max_chunk_size = round_up_to_unit_size max_chunk_size work_items_per_warp in
  let chunk_size = min (round_up_to_unit_size num_seeds work_items_per_warp) max_chunk_size in
  let full_traj_memory_request = chunk_size * mem_per_seed in
  let to_do_list = generate_chunks data num_seeds chunk_size  in
  let num_chunks_required = List.length to_do_list in

  let show_memory _ =
    Printf.printf "GPU/OpenCL device global memory limit for streamline trajectories: %d\n" gpu_traj_memory_limit;
    Printf.printf "GPU/OpenCL device memory required for streamline trajectories: %d\n" full_traj_memory_request;
    (if num_chunks_required=1 then (
      Printf.printf "no need to chunkify\n"
     ) else (
      Printf.printf "need to split into %d chunks (note separation of up/down may impact this)\n" num_chunks_required
     )
    );
    Printf.printf "Number of seed points = total number of kernel instances: %d\n" num_seeds;
    Printf.printf "Max chunk size (given memory constraint, rounding up to num work items): %d\n" max_chunk_size;
    Printf.printf "Actual chunk size = number of kernel instances per chunk: %d\n" chunk_size;
    Printf.printf "%!"
  in
  pv_verbose data show_memory;

  gpu_integrate_trajectories pocl data results seeds chunk_size to_do_list;
    
  (* Streamline stats *)
  pv_verbose data (fun _ -> Printf.printf "Computing streamlines statistics\n");
  results.ds_stats <- Some (get_traj_stats data results num_seeds 0);
  results.us_stats <- Some (get_traj_stats data results num_seeds 1);
  pv_verbose data (fun _ -> show_stats results);

  Workflow.workflow_end tprops.workflow;
  results

