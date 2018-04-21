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
 * @file   pocl.ml
 * @brief  OpenCL support library
 * v}
 *)

(*a Module abbreviations *)
module ODM = Owl.Dense.Matrix.Generic
module ODN = Owl.Dense.Ndarray.Generic

open Owl_opencl.Base
open Owl_enhance
open Globals
open Core
open Properties
module Option=Batteries.Option

(** {1 Types} *)

(**  [t] - structure for the Pocl processing *)
type t = {
    props : t_props_pocl;
    mutable platform : Owl_opencl_generated.cl_platform_id option;
    mutable device   : Owl_opencl_generated.cl_device_id option;
    mutable context  : Owl_opencl_generated.cl_context option;
    mutable queue    : Owl_opencl_generated.cl_command_queue option;
  }

(**  [cl_platform t] - retrieve OpenCL platform id from a Pocl.t *)
let cl_platform t = Option.get t.platform

(**  [cl_device t] - retrieve OpenCL device id from a Pocl.t *)
let cl_device   t = Option.get t.device

(**  [cl_context t] - retrieve OpenCL context from a Pocl.t *)
let cl_context  t = Option.get t.context

(**  [cl_context t] - retrieve OpenCL queue from a Pocl.t *)
let cl_queue    t = Option.get t.queue

(** {1 pv_verbosity functions} *)

(**  [pv_noisy t] - Shortcut to use Pocl.t verbosity for pv_noisy *)
let pv_noisy   t = Workflow.pv_noisy t.props.workflow

(**  [pv_debug t] - Shortcut to use Pocl.t verbosity for pv_debug *)
let pv_debug   t = Workflow.pv_noisy t.props.workflow

(**  [pv_info t] - Shortcut to use Pocl.t verbosity for pv_info *)
let pv_info    t = Workflow.pv_info t.props.workflow

(**  [pv_verbose t] - Shortcut to use Pocl.t verbosity for pv_verbose *)
let pv_verbose t = Workflow.pv_verbose t.props.workflow

(** {1 Show functions} *)

(**  [show_device pi p di d] - show data for platform p and device d *)
let show_device pi p di d =
    Printf.printf "Device %d.%d %s\n" pi di Device.(to_string d);
    Printf.printf "\n%!"

(**  [show_platform pi p] - show data and all devices for platform p *)
let show_platform i p =
    Printf.printf "Platform %d %s\n" i Platform.(to_string p);
    let d = Device.get_devices p in
    Array.iteri (show_device i p) d;
    Printf.printf "\n%!"

(**  [show_system _] - show data for all platforms and all devices *)
let show_system _ =
  let p = Platform.get_platforms () in
  Array.iteri show_platform p

(**  [show_context ctxt] - show data for a context *)
let show_context ctxt =
    Printf.printf "Context %s\n%!" (Context.to_string ctxt)

(**  [show_program program] - show data for a program *)
let show_program program =
    Printf.printf "Program %s\n%!" (Program.to_string program)

(**  [show_kernel kernel] - show data for a kernel *)
let show_kernel kernel =
    Printf.printf "Kernel %s\n%!" (Kernel.to_string kernel)

(**  [show_queue queue] - show data for a queue *)
let show_queue queue =
    Printf.printf "Queue %s\n%!" (CommandQueue.to_string queue)

(**  [debug_build program device] - display debug output for a built program on a device *)
let debug_build program device =
    let options = program__get_build_options_str program device in
    Printf.printf "Build options\n%s\n%!" options;
    let source = program__get_info_source_str program in
    Printf.printf "Program source\n%s\n%!" source;
    let log = program__get_build_log_str program device in
    Printf.printf "Build log\n%s\n%!" log;
    ()

(**  [make_program t source compile_options]

  Compile source with compile_options using the platform, device, and context specified by the Pocl.t t.

  Provide verbose information if required.

  On failure, output debug information for the program and raise an exception

    @returns the compiled program

 *)
let make_program t source compile_options =
    let program = Program.create_with_source (cl_context t) [|source|] in
    let d = (cl_device t) in
    (try (Program.build ~options:compile_options program [|d|]) with
     | e -> (
       Printf.printf "Compile options in : %s\n%!" compile_options; (* if they are bad they do not come in debug_build *)
       debug_build program d;
       raise e
     )
    );
    let status = program__get_build_status program d in
    pv_verbose t (fun _ -> Printf.printf "Opencl build okay %d\n%!" status);
    pv_debug   t (fun _ -> debug_build program d);
    program

let get_kernel pocl program name =
    let kernel = Kernel.create program name in
    kernel

let make_queue t ctxt d =
    let queue = CommandQueue.create ~properties:[Owl_opencl.G.cl_QUEUE_PROFILING_ENABLE] ctxt d in
    queue

type t_buffer = Owl_opencl_generated.cl_mem
let buffer_of_array ?copy:(copy=false) t kernel_read kernel_write (ba:('a, 'b, 'c) Bigarray.Genarray.t) =
  let flags =
    (
      if kernel_read && kernel_read then [Owl_opencl.G.cl_MEM_READ_WRITE]
      else if kernel_read then [Owl_opencl.G.cl_MEM_READ_ONLY]
      else [Owl_opencl.G.cl_MEM_WRITE_ONLY]
    )
  in
  let flags = if copy then (Owl_opencl.G.cl_MEM_COPY_HOST_PTR::flags) else (Owl_opencl.G.cl_MEM_USE_HOST_PTR::flags) in
  Buffer.create ~flags (cl_context t) ba (*(owl_ba2d ba)*)

let copy_buffer_to_gpu t ~src ~dst =
  pv_verbose t (fun _ -> Printf.printf "Copy buffer size %d\n%!" (buffer__size dst));
  ignore (buffer__enqueue_write (cl_queue t) src dst) (* ignore the event *)

let copy_buffer_from_gpu t ~src ~dst =
  pv_verbose t (fun _ -> Printf.printf "Copy buffer size %d\n%!" (buffer__size src));
  ignore (buffer__enqueue_read (cl_queue t) src dst) (* ignore the event *)

let finish_queue t =
  pv_verbose t (fun _ -> Printf.printf "Wait for queue\n%!");
  CommandQueue.finish (cl_queue t)

let kernel_set_arg_buffer t kernel index buffer =
  let open Ctypes in
  let _len = buffer__size buffer in
  let _buffer = allocate Owl_opencl.G.cl_mem buffer in
  pv_verbose t  (fun _ -> Printf.printf "Set arg %d to buffer of size %d\n%!" index _len);
  Kernel.set_arg kernel index (Ctypes.sizeof Owl_opencl.G.cl_mem) _buffer

let enqueue_kernel t kernel global_work_size =
(*Kernel.enqueue_ndrange ?(wait_for=[]) ?(global_work_ofs=[]) ?(local_work_size=[]) cmdq kernel work_dim global_work_size =*)
  let work_dim = List.length global_work_size in
  let event = Kernel.enqueue_ndrange (cl_queue t) kernel work_dim global_work_size in
  event

let event_wait t event =
  ignore (Event.wait_for [event])

(*a Pocl stuff *)
(*f prepare_cl_context_queue *)
let prepare_cl_context_queue t =
  pv_debug t (show_system);
  let pa = Platform.get_platforms () in
  let pai = t.props.cl_platform in
  let pai = if (pai>=Array.length pa) then 0 else pai in
  let da = Device.get_devices pa.(pai) in
  let dai = t.props.cl_device in
  let dai = if (dai>=Array.length da) then 0 else dai in
  let platform = pa.(pai) in
  let device   = da.(dai) in
  pv_verbose t (fun _ -> show_device pai platform dai device);
  let ctxt = Context.create [|device|] in
  let queue   = make_queue t ctxt device in
  pv_verbose t (fun _ -> show_context ctxt);
  t.platform <- Some platform;
  t.device <- Some device;
  t.context <- Some ctxt;
  t.queue <- Some queue;
  ()

(*f compile_program *)
let compile_program t source compile_options =
  make_program t source compile_options

(*f get_memory_limit *)
let get_memory_limit t =
  let global_mem_size = (float (Device.get_info (cl_device t)).global_mem_size) in
  int_of_float (global_mem_size *. t.props.gpu_memory_limit_pc /. 100.)

(*f compile_options

    Convert the info struct into a list of '-D' compiler macros.
    
    Args:
        info_struct (Core.Info)
        
    Returns:
        string:
            compile options
 *)
let compile_options t data info_struct kernel_name =
  let grid_scale  = Info.float_of info_struct "grid_scale" in
  let downup_sign = Info.float_of info_struct "downup_sign" in
  let array_order = Info.str_of   info_struct "array_order" in
  let kernel_name = String.uppercase_ascii kernel_name in
  Info.add_float32 info_struct "combo_factor" ((grid_scale *. data.properties.trace.integrator_step_factor) *. downup_sign);
  let add_option acc nv = 
    let option = sfmt "%s " (Info.define_str nv) in
    acc ^ option
  in
  let base_options = sfmt "-D KERNEL_%s -D%s_ORDER " kernel_name array_order in
  Info.fold_left add_option base_options info_struct

(*f append_in_file *)
let append_in_file (source, source_lines) filename f =
  let source_lines = source_lines+1 in
  let source = source ^ (sfmt "// %d : File '%s'\n" source_lines filename) in
  let rec add_source_line source source_lines line = 
    match (
      try (Some ((input_line f) ^ "\n")) with
      | _ -> None
    ) with
    | Some l -> (
      let source_lines = source_lines+1 in
      let line = line+1 in
      let l,source_lines =
        if ((String.length l)=1) then ((sfmt "// %d: '%s' Line %d\n\n" source_lines filename line), source_lines+1) else (l, source_lines)
      in
      add_source_line (source ^ l) source_lines line
    )
    | _ -> (source, source_lines)
  in
  add_source_line source source_lines 0
    
(*f read_source *)
let read_source path cl_files =
  let read_file acc f =
    let source_filename = filename_from_path path f in
    let src_f = open_in source_filename in
    append_in_file acc source_filename src_f
  in
  let (source, _) = List.fold_left read_file ("",0) cl_files in
  source

(*f rebuild_info_struct *)
let rebuild_info_struct pocl (data:t_core_data) info_struct =
  let properties = data.properties in
  let s_props = properties.state in
  let t_props = properties.trace in
  let max_length = if (t_props.max_length=0.) then infinity else t_props.max_length in
  let max_n_steps = int_of_float (t_props.max_length /. t_props.integrator_step_factor) in
  let interchannel_max_n_steps = if t_props.interchannel_max_n_steps=0 then max_n_steps else t_props.interchannel_max_n_steps in
  let nxf = (float data.roi_nx) in
  let nyf = (float data.roi_ny) in
  let grid_scale = sqrt (nxf *. nyf) in
  let dt_max = min 0.1 (min (1.0 /. nxf) (1.0 /. nyf)) in
  let sspd_f = float t_props.subpixel_seed_point_density in
  let subpixel_seed_span = 1.0 -. ( 1.0 /. sspd_f) in
  let subpixel_seed_step = subpixel_seed_span /. (sspd_f -. 1.0) in

  Info.add_str     info_struct "array_order"                   s_props.array_order;
  Info.add_float32 info_struct "max_integration_step_error"    t_props.max_integration_step_error;
  Info.add_float32 info_struct "integration_halt_threshold"    t_props.integration_halt_threshold;
  Info.add_uint    info_struct "trajectory_resolution"         t_props.trajectory_resolution;
  Info.add_uint    info_struct "subpixel_seed_point_density"   t_props.subpixel_seed_point_density;
  Info.add_float32 info_struct "jitter_magnitude"              t_props.jitter_magnitude;
  Info.add_uint    info_struct "interchannel_max_n_steps"      interchannel_max_n_steps;
  Info.add_uint    info_struct "segmentation_threshold"        t_props.segmentation_threshold;

  Info.add_float32 info_struct "adjusted_max_error"            (0.85 *. (sqrt t_props.max_integration_step_error));
  Info.add_float32 info_struct "max_length"                    (max_length /. data.roi_pixel_size);
  Info.add_float32 info_struct "pixel_size"                    data.roi_pixel_size;
  Info.add_uint    info_struct "pad_width"                     data.pad_width;
  Info.add_float32 info_struct "pad_width_pp5"                 ((float data.pad_width) +. 0.5);

  Info.add_float32 info_struct "gpu_memory_limit_pc"           s_props.gpu_memory_limit_pc;

  Info.add_uint info_struct "nx"                               data.roi_nx;
  Info.add_uint info_struct "ny"                               data.roi_ny;
  Info.add_float32 info_struct "nxf"                           nxf;
  Info.add_float32 info_struct "nyf"                           nyf;
  Info.add_uint info_struct "nx_padded"                        (data.roi_nx + 2*data.pad_width);
  Info.add_uint info_struct "ny_padded"                        (data.roi_ny + 2*data.pad_width);
  Info.add_float32 info_struct "x_max"                         ((float data.roi_nx) -. 0.5);
  Info.add_float32 info_struct "y_max"                         ((float data.roi_ny) -. 0.5);
  Info.add_float32 info_struct "grid_scale"                    grid_scale;
  Info.add_float32 info_struct "combo_factor"                  (grid_scale *. t_props.integrator_step_factor);
  Info.add_float32 info_struct "dt_max"                        dt_max;
  Info.add_uint    info_struct "max_n_steps"                   max_n_steps;
  Info.add_uint    info_struct "seeds_chunk_offset"            0;
  Info.add_float32 info_struct "subpixel_seed_halfspan"        (subpixel_seed_span /. 2.0);
  Info.add_float32 info_struct "subpixel_seed_step"            subpixel_seed_step;
  Info.add_uint32  info_struct "left_flank_addition"           t_props.left_flank_addition;

  Info.add_uint info_struct "is_channel"              0x1;
  Info.add_uint info_struct "is_thinchannel"          0x2;
  Info.add_uint info_struct "is_interchannel"         0x4;
  Info.add_uint info_struct "is_channelhead"          0x8;
  Info.add_uint info_struct "is_channeltail"          0x10;
  Info.add_uint info_struct "is_majorconfluence"      0x20;
  Info.add_uint info_struct "is_minorconfluence"      0x40;
  Info.add_uint info_struct "is_majorinflow"          0x80;
  Info.add_uint info_struct "is_minorinflow"          0x100;
  Info.add_uint info_struct "is_leftflank"            0x200;
  Info.add_uint info_struct "is_rightflank"           0x400;
  Info.add_uint info_struct "is_midslope"             0x800;
  Info.add_uint info_struct "is_ridge"                0x1000;
  Info.add_uint info_struct "is_stuck"                0x2000;
  Info.add_uint info_struct "is_loop"                 0x4000;
  Info.add_uint info_struct "is_blockage"             0x8000;

  pv_debug pocl(fun _ -> Info.display info_struct);
  ()

(*f [create props] -  *)
let create props =
    { props=props.pocl;
      platform=None;
      device=None;
      context=None;
      queue=None;
    }
