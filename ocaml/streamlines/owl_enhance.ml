(*a Owl OpenCL extensions *)
open Ctypes
open Owl_opencl.G
open Owl_opencl.U
open Owl_opencl.Base

(* Questions for Liang

let work_dim = List.length global_work_size in
let event = Kernel.enqueue_ndrange local_size queue kernel work_dim global_work_size in

Ulong or Uint64 for cl_ulong?

Change Buffer enqueue_read/write to use buffer / array rather than src/dst

Expose area_of

 *)


let char_ptr_to_uint64_ptr x = coerce (ptr char) (ptr uint64_t) x

(*cl_PROGRAM_BUILD_STATUS*)

let opencl_read_param_str opencl_fn param_name =
  let param_name = Unsigned.UInt32.of_int param_name in
  let param_value_size_ret = allocate size_t size_0 in
  opencl_fn param_name size_0 null param_value_size_ret |> cl_check_err;

  let _param_value_size = Unsigned.Size_t.to_int !@param_value_size_ret in
  let param_value = allocate_n char ~count:_param_value_size |> Obj.magic in
  opencl_fn param_name !@param_value_size_ret param_value magic_null |> cl_check_err;
  (* null terminated string, so minus 1 *)
  string_from_ptr param_value (_param_value_size - 1)

let opencl_read_param_int opencl_fn param_name =
  let param_name = Unsigned.UInt32.of_int param_name in
  let param_value_size_ret = allocate size_t size_0 in
  opencl_fn param_name size_0 null param_value_size_ret |> cl_check_err;

  let _param_value_size = Unsigned.Size_t.to_int !@param_value_size_ret in
  let param_value = allocate_n char ~count:_param_value_size |> Obj.magic in
  opencl_fn param_name !@param_value_size_ret param_value magic_null |> cl_check_err;
  (* null terminated string, so minus 1 *)
  Ctypes.(!@ (from_voidp Ctypes.int (to_voidp param_value)))

let program__get_build_status program device  = opencl_read_param_int (clGetProgramBuildInfo program device) cl_PROGRAM_BUILD_STATUS
let program__get_build_options_str program device = opencl_read_param_str (clGetProgramBuildInfo program device) cl_PROGRAM_BUILD_OPTIONS
let program__get_build_log_str program device     = opencl_read_param_str (clGetProgramBuildInfo program device) cl_PROGRAM_BUILD_LOG
let program__get_info_source_str program    = opencl_read_param_str (clGetProgramInfo program) cl_PROGRAM_SOURCE

  let event__get_profiling_info event param_name =
    let param_name = Unsigned.UInt32.of_int param_name in
    let param_value_size_ret = allocate size_t size_0 in
    clGetEventProfilingInfo event param_name size_0 null param_value_size_ret |> cl_check_err;

    let _param_value_size = Unsigned.Size_t.to_int !@param_value_size_ret in
    let param_value = allocate_n char ~count:_param_value_size |> Obj.magic in
    clGetEventProfilingInfo event param_name !@param_value_size_ret param_value magic_null |> cl_check_err;
    let (p, l) = param_value, _param_value_size in
    !@(char_ptr_to_uint64_ptr p) |> Unsigned.UInt64.to_int64

let event__get_duration event = 
    let prof_start = event__get_profiling_info event cl_PROFILING_COMMAND_START in
    let prof_end   = event__get_profiling_info event cl_PROFILING_COMMAND_END in
    Int64.(sub prof_end prof_start)

  let get_buffer_info buf param_name =
    let param_name = Unsigned.UInt32.of_int param_name in
    let param_value_size_ret = allocate size_t size_0 in
    clGetMemObjectInfo buf param_name size_0 null param_value_size_ret |> cl_check_err;

    let _param_value_size = Unsigned.Size_t.to_int !@param_value_size_ret in
    let param_value = allocate_n char ~count:_param_value_size |> Obj.magic in
    clGetMemObjectInfo buf param_name !@param_value_size_ret param_value magic_null |> cl_check_err;
    param_value, _param_value_size

let buffer__size buf =
  ( let p, l = get_buffer_info buf cl_MEM_SIZE in !@(char_ptr_to_size_t_ptr p) |> Unsigned.Size_t.to_int )

let buffer__enqueue_read ?ofs:(ofs=0) ?len ?(blocking=true) ?(wait_for=[]) cmdq src dst =
  let len = Batteries.Option.default (buffer__size src) len in
  let _dst = bigarray_to_void_ptr dst in
  Buffer.enqueue_read ~wait_for ~blocking cmdq src ofs len _dst

let buffer__enqueue_write ?ofs:(ofs=0) ?len ?(blocking=true) ?(wait_for=[]) cmdq src dst =
  let len = Batteries.Option.default (buffer__size dst) len in
  let _src = bigarray_to_void_ptr src in
  Buffer.enqueue_write ~wait_for ~blocking cmdq dst ofs len _src


