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
 * @file   integration.ml
 * @brief  Integration using GPU for trace
 * v}
 *)

(*a Module abbreviations *)
open Globals
open Core
open Properties
module ODM = Owl.Dense.Matrix.Generic
module ODN = Owl.Dense.Ndarray.Generic

(** {1 pv_verbosity functions} *)

(**  [pv_noisy t]

  Shortcut to use {!type:Properties.t_props_trace} verbosity for {!val:Properties.pv_noisy}

 *)
let pv_noisy   (props:t_props_trace) = Workflow.pv_noisy props.workflow

(**  [pv_debug t]

  Shortcut to use {!type:Properties.t_props_trace} verbosity for {!val:Properties.pv_debug}

 *)
let pv_debug   (props:t_props_trace) = Workflow.pv_noisy props.workflow

(**  [pv_info t]

  Shortcut to use {!type:Properties.t_props_trace} verbosity for {!val:Properties.pv_info}

 *)
let pv_info    (props:t_props_trace) = Workflow.pv_info props.workflow

(**  [pv_verbose t]

  Shortcut to use {!type:Properties.t_props_trace} verbosity for {!val:Properties.pv_verbose}

 *)
let pv_verbose (props:t_props_trace) = Workflow.pv_verbose props.workflow

(** {1 Statics} *)

let cl_files = ["rng.cl";"essentials.cl";
                "writearray.cl";"trajectoryfns.cl";"computestep.cl";
                "integrationfns.cl";"trajectory.cl";"integration.cl"]
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
    ()

(**  [generate_chunks num_seeds data num_chunks_required]

  Generate a to_do_list of chunks

 *)
let generate_chunks num_seeds data num_chunks_required =
  let chunk_size = (num_seeds + num_chunks_required - 1) / num_chunks_required in
  let chunks = List.init num_chunks_required (fun i -> (i, i*chunk_size, min ((i+1)*chunk_size) num_seeds)) in
  let to_do_list = List.fold_left (fun acc nse -> (chunk data true nse)::(chunk data false nse)::acc) [] chunks in
  (chunk_size, to_do_list)

(** {1 Memory} *)

(**  t_memory

  Structure containing the memory and OpenCL buffers for interacting
  with the integrate_trajectories kernel.

 *)
type t_memory = {
    uv_array           : t_ba_floats;  (* padded ROI - vector field array input *)
    chunk_slc_array    : t_ba_ints;    (* padded ROI - #streamlines crossing pixel *)
    chunk_slt_array    : t_ba_ints;    (* padded ROI - streamline length total of all crossing pixel *)
    chunk_nsteps_array : t_ba_int16s;  (* chunk_size - number of steps taken for a streamline from the seed *)
    chunk_length_array : t_ba_floats;  (* chunk_size - streamline length from the seed *)
    chunk_trajcs_array : t_ba_chars;   (* chunk_size * max_traj_length*2 - char path of streamline from the seed *)

    seeds_buffer        : Pocl.t_buffer;
    mask_buffer         : Pocl.t_buffer;
    uv_buffer           : Pocl.t_buffer;
    chunk_slc_buffer    : Pocl.t_buffer;
    chunk_slt_buffer    : Pocl.t_buffer;
    chunk_nsteps_buffer : Pocl.t_buffer;
    chunk_length_buffer : Pocl.t_buffer;
    chunk_trajcs_buffer : Pocl.t_buffer;
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
  let (num_seeds,_) = ODM.shape seeds in
  let roi_nx = data.pad_width*2+data.roi_nx in
  let roi_ny = data.pad_width*2+data.roi_ny in

  let uv_array = ba_float2d (roi_nx) (roi_ny*2) in
  let fill_uv x y u v = 
    ODM.set uv_array x (y*2+0) u;
    ODM.set uv_array x (y*2+1) v;
  in
  ODM.iter2i_2d fill_uv data.u_array data.v_array;

  (* fill with zeros *)
  let chunk_slc_array = ba_int2d roi_nx roi_ny  in
  let chunk_slt_array = ba_int2d roi_nx roi_ny  in

  (* Chunk-sized temporary arrays - use for a work group, then copy to traj_* *)
  (* Use "bag o' bytes" buffer for huge trajectories array. Write (by GPU) only. *)
  let chunk_trajcs_array = ba_char2d chunk_size (max_traj_length*2) in
  let chunk_nsteps_array = ba_int16s chunk_size in
  let chunk_length_array = ba_floats chunk_size in
  
  (* Create OpenCL buffers of the arrays *)
  let seeds_buffer        = copy_read pocl seeds in
  let mask_buffer         = copy_read pocl data.basin_mask_array in
  let uv_buffer           = copy_read pocl uv_array in
  let chunk_slc_buffer    = copy_read_write pocl chunk_slc_array in (* no need to copy - filled with zero before run *)
  let chunk_slt_buffer    = copy_read_write pocl chunk_slt_array in (* no need to copy - filled with zero before run *)
  let chunk_nsteps_buffer = write_only pocl chunk_nsteps_array in
  let chunk_length_buffer = write_only pocl chunk_length_array in
  let chunk_trajcs_buffer = write_only pocl chunk_trajcs_array in

  let show_sizes _ =
    Printf.printf "Array sizes:\n";
    Printf.printf "ROI-type = %d,%d\n" roi_nx roi_ny ;
    let (x,y) = ODM.shape uv_array in
    Printf.printf "uv = %d,%d\n" x y;
    let (x,y) = ODM.shape chunk_slc_array in
    Printf.printf "chunk slc-type = %d,%d\n" x y;
    let (x,y) = ODM.shape chunk_trajcs_array in
    Printf.printf "Streamlines virtual array allocation: %d,%d size %d\n" num_seeds y (num_seeds*y*max_traj_length*2);
    Printf.printf "Streamlines array allocation per chunk: %d,%d size %d\n%!" x y (ODM.size_in_bytes chunk_trajcs_array)
  in
  pv_verbose data.properties.trace show_sizes;
  {
    uv_array;
    chunk_slc_array;
    chunk_slt_array;
    chunk_nsteps_array;
    chunk_length_array;
    chunk_trajcs_array;

    seeds_buffer;
    mask_buffer;
    uv_buffer;
    chunk_slc_buffer;
    chunk_slt_buffer;
    chunk_nsteps_buffer;
    chunk_length_buffer;
    chunk_trajcs_buffer;
  }

(**  [memory_clear t pocl]

  Clear the chunk buffers and copy them to the GPU, prior to execution
  of the integrate_trajectories kernel

 *)
let memory_clear t pocl = 
  ODN.fill t.chunk_slc_array 0;
  ODN.fill t.chunk_slt_array 0;
  Pocl.copy_buffer_to_gpu pocl ~dst:t.chunk_slc_buffer ~src:t.chunk_slc_array;
  Pocl.copy_buffer_to_gpu pocl ~dst:t.chunk_slt_buffer ~src:t.chunk_slt_array;
  Pocl.finish_queue pocl

(**  [memory_copyback t pocl]

  Copy back the chunk buffers from the GPU to the big arrays after execution
  of the integrate_trajectories kernel

 *)
let memory_copyback t pocl = 
  Pocl.copy_buffer_from_gpu pocl ~src:t.chunk_trajcs_buffer ~dst:t.chunk_trajcs_array;
  Pocl.copy_buffer_from_gpu pocl ~src:t.chunk_nsteps_buffer ~dst:t.chunk_nsteps_array;
  Pocl.copy_buffer_from_gpu pocl ~src:t.chunk_length_buffer ~dst:t.chunk_length_array;
  Pocl.copy_buffer_from_gpu pocl ~src:t.chunk_slc_buffer    ~dst:t.chunk_slc_array;
  Pocl.copy_buffer_from_gpu pocl ~src:t.chunk_slt_buffer    ~dst:t.chunk_slt_array;
  Pocl.finish_queue pocl

(** {1 Results} *)

(**  [results_create data seeds]

  Create the trace results structure given the seeds

 *)
let results_create data seeds =
  let (num_seeds,_) = ODM.shape seeds in
  let roi_nx = data.roi_nx + data.pad_width*2 in
  let roi_ny = data.roi_ny + data.pad_width*2 in

  (* Result arrays *)
  let traj_nsteps_array  = ba_int16s (num_seeds*2) in (* *2 as there are 2 directions ? *)
  let traj_lengths_array = ba_floats (num_seeds*2) in (* *2 as there are 2 directions ? *)
  let slc_array          = ba_int3d   2 roi_nx roi_ny in
  let slt_array          = ba_float3d 2 roi_nx roi_ny in
  let sla_array          = ba_float3d 2 roi_nx roi_ny in
  ODN.fill traj_nsteps_array  0;
  ODN.fill traj_lengths_array 0.;
  ODN.fill slc_array 0;
  ODN.fill slt_array 0.;
  ODN.fill sla_array 0.;
  {
    streamline_arrays= Array.make 2 (Array.make 0 (Bytes.make 0 ' '));
    traj_nsteps_array;
    traj_lengths_array;
    slc_array;
    slt_array;
    sla_array;
  }

(** {1 GPU functions} *)

(**  [gpu_integrate_chunk pocl data memory streamline_lists results cl_kernel_source t]

  Integrate a chunk of seeds on the GPU - this is a set of consecutive
  seeds in either upstream or downstream - and aggregate the results

 *)
let gpu_integrate_chunk pocl data memory streamline_lists results cl_kernel_source t =
  if t.required then (
    pv_verbose data.properties.trace (fun _ -> show_chunk t);

    (* Specify this integration job's parameters and compile *)
    let global_size = [t.num_seeds; 1] in
    let info = data.info in
    Info.set info "downup_sign" (Info.Float32 t.downup_sign);
    Info.set info "seeds_chunk_offset" (Info.Int t.seed_offset);
    let compile_options = Pocl.compile_options pocl data info "INTEGRATE_TRAJECTORY" in
    let program = Pocl.compile_program pocl cl_kernel_source compile_options in
    let kernel = Pocl.get_kernel pocl program "integrate_trajectory" in

    (* Execute the kernel *)
    memory_clear memory pocl;

    Pocl.kernel_set_arg_buffer pocl kernel 0 memory.seeds_buffer;
    Pocl.kernel_set_arg_buffer pocl kernel 1 memory.mask_buffer;
    Pocl.kernel_set_arg_buffer pocl kernel 2 memory.uv_buffer;
    Pocl.kernel_set_arg_buffer pocl kernel 3 memory.chunk_trajcs_buffer;
    Pocl.kernel_set_arg_buffer pocl kernel 4 memory.chunk_nsteps_buffer;
    Pocl.kernel_set_arg_buffer pocl kernel 5 memory.chunk_length_buffer;
    Pocl.kernel_set_arg_buffer pocl kernel 6 memory.chunk_slc_buffer;
    Pocl.kernel_set_arg_buffer pocl kernel 7 memory.chunk_slt_buffer;

    let event = Pocl.enqueue_kernel pocl kernel global_size in
    Pocl.event_wait pocl event;
    let elapsed = Owl_enhance.event__get_duration event in
    let elapsed = (Int64.to_float elapsed) *. 1E-9 in
    Printf.printf "\n##### Kernel lapsed time: %0.3f secs #####\n" elapsed;
    
    memory_copyback memory pocl;

    (* Compile streamline results *)
    pv_noisy data.properties.trace (fun _ -> Printf.printf "Building streamlines compressed array for chunk\n%!");
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
    
    (* Compile slc/slt array results *)
    let sum_to_total x y slc slt =
      let tslc = ODN.get results.slc_array [|t.downup_index; x; y|] in
      let tslt = ODN.get results.slt_array [|t.downup_index; x; y|] in
      ODN.set results.slc_array [|t.downup_index; x; y|] (tslc + slc);
      ODN.set results.slt_array [|t.downup_index; x; y|] (tslt +. (float slt));
    in
    ODM.iter2i_2d sum_to_total memory.chunk_slc_array memory.chunk_slt_array;

  ) else (
     (* chunk was not required *)
  )

(**  [gpu_integrate_trajectories pocl data seeds chunk_size to_do_list]

  Carry out GPU computations in the chunks on {i to_do_list}.

  Each chunk is handled with a separate compilation, and results are
  aggregated.

 *)
let gpu_integrate_trajectories pocl data seeds chunk_size to_do_list =

  let memory = memory_create_buffers pocl data seeds chunk_size in
  let streamline_lists = [| []; []; |] in
  let results = results_create data seeds in
  let cl_kernel_source = Pocl.read_source cl_src_path cl_files in

  List.iter (fun chunk -> gpu_integrate_chunk pocl data memory streamline_lists results cl_kernel_source chunk) to_do_list;

  (* Compute average streamline lengths (sla) from total lengths (slt) and counts (slc)
  *)
  let sspd2 = (Info.float_of data.info "subpixel_seed_point_density") ** 2. in
  let map_slc_slt ind slt =
    let slc=ODN.get results.slc_array ind in
    ODN.set results.slt_array ind (slt /. sspd2);
    if (slc=0) then (
      0. 
    ) else  (
      slt /. (float slc)
    )
  in
  let sla = ODN.mapi_nd map_slc_slt results.slt_array in

  results.streamline_arrays <- [| Array.of_list (List.rev streamline_lists.(0)); Array.of_list (List.rev streamline_lists.(1)); |];

  let total_steps = Array.fold_left (fun acc t -> acc+(Bytes.length t)) 0           results.streamline_arrays.(0) in
  let total_steps = Array.fold_left (fun acc t -> acc+(Bytes.length t)) total_steps results.streamline_arrays.(1) in
  pv_verbose data.properties.trace (fun _ -> Printf.printf "Total steps in all streamlines %d\n%!" (total_steps/2));
  results

(**  [integrate_trajectories tprops pocl data seeds]

  Integrate trajectories both upstream and downstream from the {i
  seeds} array of unpadded ROI coordinates.

  Trace each streamline from its corresponding seed point using 2nd-order Runge-Kutta 
  integration of the topographic gradient vector field.

  @return trace_results

 *)
let integrate_trajectories (tprops:t_props_trace) pocl data seeds =
  Workflow.workflow_start ~subflow:"integrating streamlines" tprops.workflow;
  Pocl.prepare_cl_context_queue pocl;

  let gpu_traj_memory_limit = Pocl.get_memory_limit pocl in (* max memory permitted to use *)
  let (num_seeds,_) = ODM.shape seeds in
  let full_traj_memory_request = num_seeds * (Info.int_of data.info "max_n_steps") * 2 in
  let num_chunks_required = (full_traj_memory_request+gpu_traj_memory_limit-1) / gpu_traj_memory_limit in
  let (chunk_size, to_do_list) = generate_chunks num_seeds data num_chunks_required  in

  let show_memory _ =
    Printf.printf "GPU/OpenCL device global memory limit for streamline trajectories: %d\n" gpu_traj_memory_limit;
    Printf.printf "GPU/OpenCL device memory required for streamline trajectories: %d\n" full_traj_memory_request;
    (if num_chunks_required=1 then (
      Printf.printf "no need to chunkify\n"
     ) else (
      Printf.printf "need to split into %d chunks\n" num_chunks_required
     )
    );
    Printf.printf "Number of seed points = total number of kernel instances: %d\n" num_seeds;
    Printf.printf "Chunk size = number of kernel instances per chunk: %d\n" chunk_size;
    Printf.printf "%!"
  in
  pv_verbose data.properties.trace show_memory;

  let results = gpu_integrate_trajectories pocl data seeds chunk_size to_do_list in
    
  (* Streamline stats *)
  let pixel_size = Info.float_of data.info "pixel_size" in

(*
    traj_stats_df = compute_stats(traj_length_array,traj_nsteps_array,pixel_size,verbose)
    dds =  traj_stats_df['ds']['downstream','mean']
    uds =  traj_stats_df['ds']['upstream','mean']
    # slt: sum of line lengths crossing a pixel * number of line-points per pixel
    # slt: <=> sum of line-points per pixel
    rtn_slt_array[:,:,0] = rtn_slt_array[:,:,0]*(dds/pixel_size)
    rtn_slt_array[:,:,1] = rtn_slt_array[:,:,1]*(uds/pixel_size)
    # slt:  => sum of line-points per meter
    rtn_slt_array = np.sqrt(rtn_slt_array) #*np.exp(-1/4) # *(3/4)
    # slt:  =>  sqrt(area)

*)
  Workflow.workflow_end tprops.workflow;
  results

(*a Post-GPU functions *)
(*f compute_stats
    Compute streamline point density and trajectory length min, mean, max.

    Returns:
        pandas.DataFrame:  lnds_stats_df
let compute_stats trace traj_length_array traj_nsteps_array  pixel_size =
    vprint(verbose,'Computing streamlines statistics')
    lnds_stats = []
    for downup_idx in [0,1]:
(
        ln0(xy) = traj_length_array[:,downup_idx]*pixel_size
        ln1(xy) = traj_nsteps_array[:,downup_idx])).T)])
        lnds = array_of (ln0; ln1; ln0/ln1;) for all xy
        lnds_stats.append( min(lnds,axis=0); mean(lnds,axis=0); np.max(lnds,axis=0) )
)
    lnds_stats_array = np.array(lnds_stats,dtype=np.float32)
    lnds_indexes = [np.array(['downstream', 'downstream', 'downstream', 
                              'upstream', 'upstream', 'upstream']),
                         np.array(['min','mean','max','min','mean','max'])]
    lnds_stats_df = pd.DataFrame(data=lnds_stats_array, 
                                 columns=['l','n','ds'],
                                 index=lnds_indexes)
    vprint(verbose,lnds_stats_df.T)
    return lnds_stats_df
 *)

