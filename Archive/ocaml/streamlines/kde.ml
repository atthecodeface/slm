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
 * @file   kde.ml
 * @brief  Kernel density estimation
 *
 * Up to date with python of git CS 189bfccdabc3371eafe8bcafa3bdfa8c241e56e4
 * Except  max_time_per_kernel and initial_size_factor are hardwired
 * except for bivariate which is not there yet
 *
 * v}
 *)

(*f  Module abbreviations *)
open Globals
open Properties
open Core
module ODM = Owl.Dense.Matrix.Generic
module ODN = Owl.Dense.Ndarray.Generic
module BA=Bigarray

(** {1 Types } *)

(**  [t]

  Structure for the Preprocess workflow

 *)
type t = {
    props : t_props_preprocess;
    mutable roi_gradx_array : t_ba_floats;
    mutable roi_grady_array : t_ba_floats;
    mutable where_looped : (int * int) list;
    pad_width : int;
  }

(** {1 pv_verbosity functions} *)

(**  [pv_noisy t]

  Shortcut to use {!type:Properties.t_props_trace} verbosity for {!val:Properties.pv_noisy}

 *)
let pv_noisy   (props:t_props_analysis) = Workflow.pv_noisy props.workflow

(**  [pv_debug t]

  Shortcut to use {!type:Properties.t_props_trace} verbosity for {!val:Properties.pv_debug}

 *)
let pv_debug   (props:t_props_analysis) = Workflow.pv_noisy props.workflow

(**  [pv_info t]

  Shortcut to use {!type:Properties.t_props_trace} verbosity for {!val:Properties.pv_info}

 *)
let pv_info    (props:t_props_analysis) = Workflow.pv_info props.workflow

(**  [pv_verbose t]

  Shortcut to use {!type:Properties.t_props_trace} verbosity for {!val:Properties.pv_verbose}

 *)
let pv_verbose (props:t_props_analysis) = Workflow.pv_verbose props.workflow

(** {1 Memory functions} *)

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

(** {1 GPU compute functions} *)

(**  [gpu_compute_histogram pocl info program sl_array is_bivariate]
 *)
let gpu_compute_histogram pocl info program sl_array is_bivariate =
  let n_hist_bins  = Info.int_of info "n_hist_bins" in
  let n_data       = Info.int_of info "n_data" in

  let cl_kernel_name = if is_bivariate then "histogram_bivariate" else  "histogram_univariate" in
  let kernel = Pocl.get_kernel pocl program cl_kernel_name in

  let nx = n_hist_bins in
  let ny = if is_bivariate then n_hist_bins else 1 in
  let histogram_array    = ba_int2d nx ny in
  ODM.fill histogram_array 0;
  let sl_buffer          = copy_read       pocl sl_array in
  let histogram_buffer   = copy_read_write pocl histogram_array in
  Pocl.kernel_set_arg_buffer pocl kernel 0 sl_buffer;
  Pocl.kernel_set_arg_buffer pocl kernel 1 histogram_buffer;

  let time_taken = Pocl.adaptive_enqueue_kernel pocl kernel n_data 64 in
  Printf.printf "\n##### Kernel lapsed time: %0.3f secs #####\n" time_taken;

  Pocl.copy_buffer_from_gpu pocl ~src:histogram_buffer    ~dst:histogram_array;
  Pocl.finish_queue pocl;
  histogram_array

(**  [gpu_compute_partial_pdf pocl info program histogram_array]
 *)
let gpu_compute_partial_pdf pocl info program histogram_array =
  let n_hist_bins  = Info.int_of info "n_hist_bins" in

  let cl_kernel_name = "pdf_bivariate_rows" in
  let kernel = Pocl.get_kernel pocl program cl_kernel_name in

  let partial_pdf_array  = ba_float2d n_hist_bins n_hist_bins in
  ODM.fill partial_pdf_array 0.;
  let histogram_buffer   = copy_read       pocl histogram_array in
  let partial_pdf_buffer = copy_read_write pocl partial_pdf_array in

  Pocl.kernel_set_arg_buffer pocl kernel 0 histogram_buffer;
  Pocl.kernel_set_arg_buffer pocl kernel 1 partial_pdf_buffer;

  let global_size = n_hist_bins*n_hist_bins in
  let time_taken = Pocl.adaptive_enqueue_kernel pocl kernel global_size 64 in
  Printf.printf "\n##### Kernel lapsed time: %0.3f secs #####\n" time_taken;
  Pocl.copy_buffer_from_gpu pocl ~src:partial_pdf_buffer    ~dst:partial_pdf_array;
  Pocl.finish_queue pocl;
  partial_pdf_array

(**  [gpu_compute_full_bivariate_pdf pocl info program partial_pdf_array]
 *)
let gpu_compute_full_bivariate_pdf pocl info program partial_pdf_array =
  let n_pdf_points  = Info.int_of info "n_pdf_points" in

  let cl_kernel_name = "pdf_bivariate_cols" in
  let kernel = Pocl.get_kernel pocl program cl_kernel_name in

  let pdf_array          = ba_float2d n_pdf_points n_pdf_points in
  ODM.fill pdf_array 0.;
  let partial_pdf_buffer = copy_read       pocl partial_pdf_array in
  let pdf_buffer         = copy_read_write pocl pdf_array in

  Pocl.kernel_set_arg_buffer pocl kernel 0 partial_pdf_buffer;
  Pocl.kernel_set_arg_buffer pocl kernel 1 pdf_buffer;

  let global_size = n_pdf_points*n_pdf_points in
  let time_taken = Pocl.adaptive_enqueue_kernel pocl kernel global_size 64 in
  Printf.printf "\n##### Kernel lapsed time: %0.3f secs #####\n" time_taken;
  Pocl.copy_buffer_from_gpu pocl ~src:pdf_buffer    ~dst:pdf_array;
  Pocl.finish_queue pocl;
  pdf_array

(**  [gpu_compute_full_univariate_pdf pocl info program histogram_array]
 *)
let gpu_compute_full_univariate_pdf pocl info program histogram_array =
  let n_pdf_points  = Info.int_of info "n_pdf_points" in

  let cl_kernel_name = "pdf_univariate" in
  let kernel = Pocl.get_kernel pocl program cl_kernel_name in

  let pdf_array          = ba_float2d n_pdf_points 1 in
  ODM.fill pdf_array 0.;
  let histogram_buffer   = copy_read       pocl histogram_array in
  let pdf_buffer         = copy_read_write pocl pdf_array in

  Pocl.kernel_set_arg_buffer pocl kernel 0 histogram_buffer;
  Pocl.kernel_set_arg_buffer pocl kernel 1 pdf_buffer;

  let global_size = n_pdf_points in
  let time_taken = Pocl.adaptive_enqueue_kernel pocl kernel global_size 64 in
  Printf.printf "\n##### Kernel lapsed time: %0.3f secs #####\n" time_taken;
  Pocl.copy_buffer_from_gpu pocl ~src:pdf_buffer    ~dst:pdf_array;
  Pocl.finish_queue pocl;
  pdf_array

(** {1 Statics} *)

let cl_files = ["kde.cl";
    ]
let cl_src_path = ["opencl"]

(** {1 PDF estimation }
 *)

(**  [estimate_univariate_pdf t pocl info sl_array]

For KDE we need to set
            '-D','KERNEL_{}'.format(kernel_def.upper()),
            '-D','KDF_BANDWIDTH={}f'.format(info_dict['kdf_bandwidth']),
            '-D','KDF_IS_{}'.format(info_dict['kdf_kernel'].upper()),
            '-D','N_DATA={}u'.format(info_dict['n_data']),
            '-D','N_HIST_BINS={}u'.format(info_dict['n_hist_bins']),
            '-D','N_PDF_POINTS={}u'.format(info_dict['n_pdf_points']),
            '-D','X_MIN={}f'.format(info_dict['x_min']),
            '-D','X_MAX={}f'.format(info_dict['x_max']),
            '-D','X_RANGE={}f'.format(info_dict['x_range']),
            '-D','BIN_DX={}f'.format(info_dict['bin_dx']),
            '-D','PDF_DX={}f'.format(info_dict['pdf_dx']),
            '-D','KDF_WIDTH_X={}f'.format(info_dict['kdf_width_x']),
            '-D','N_KDF_PART_POINTS_X={}u'.format(info_dict['n_kdf_part_points_x']),
            '-D','Y_MIN={}f'.format(info_dict['y_min']),
            '-D','Y_MAX={}f'.format(info_dict['y_max']),
            '-D','Y_RANGE={}f'.format(info_dict['y_range']),
            '-D','BIN_DY={}f'.format(info_dict['bin_dy']),
            '-D','PDF_DY={}f'.format(info_dict['pdf_dy']),
            '-D','KDF_WIDTH_Y={}f'.format(info_dict['kdf_width_y']),
            '-D','N_KDF_PART_POINTS_Y={}u'.format(info_dict['n_kdf_part_points_y'])

 *)
let estimate_univariate_pdf t pocl info sl_array =

  let cl_kernel_source = Pocl.read_source cl_src_path cl_files in
            
  let n_data   = Info.int_of   info "n_data" in
  let bin_dx   = Info.float_of info "bin_dx" in
  let pdf_dx   = Info.float_of info "pdf_dx" in

  let stddev=1.0 in (*    stddev       = np.std(sl_array)*)
    
  (* Set up kernel filter -  Silverman hack for now *)
  let kdf_width_x = 1.06 *. stddev *. ((float n_data) ** (-0.2)) *. 8. in
  let n_kdf_part_points_x = (truncate (kdf_width_x /. pdf_dx)) / 2 in
  Info.set info "kdf_width_x"         (Info.Float32 kdf_width_x);
  Info.set info "n_kdf_part_points_x" (Info.Int     n_kdf_part_points_x);

  Info.set_int info "kernel_histogram_univariate" 1;
  Info.set_int info "kernel_pdf_univariate" 1;
  Info.set_int info (sfmt "kdf_is_%s" (Info.str_of info "kernel")) 1;

  let compile_options = Pocl.compile_options pocl info "KDE" in
  let program = Pocl.compile_program pocl cl_kernel_source compile_options in

  pv_verbose t (fun _ -> Printf.printf "histogram...%!");
  let histogram_array = gpu_compute_histogram pocl info program sl_array false in
  pv_verbose t (fun _ -> Printf.printf "done\n%!");

  pv_verbose t (fun _ -> Printf.printf "kernel filtering...%!");
  let pdf_array = gpu_compute_full_univariate_pdf pocl info program histogram_array in
  pv_verbose t (fun _ -> Printf.printf "done\n%!");

  let weighted_total = (ODM.sum' pdf_array) *. bin_dx in
  ODM.div_scalar pdf_array weighted_total

(**  [estimate_bivariate_pdf pocl data]

    Compute bivariate histogram and subsequent kernel-density smoothed pdf.

 *)
let estimate_bivariate_pdf t pocl info sl_array =

  let cl_kernel_source = Pocl.read_source cl_src_path cl_files in
            
  let bandwidth   = Info.float_of info  "kdf_bandwidth" in

  let x_range  = Info.float_of info "x_range" in
  let stddev_x  = 1.0 in
  let bin_dx   = Info.float_of info "bin_dx" in
  let pdf_dx   = Info.float_of info "pdf_dx" in

  let y_range  = Info.float_of info "y_range" in
  let stddev_y  = 1.0 in
  let bin_dy   = Info.float_of info "bin_dy" in
  let pdf_dy   = Info.float_of info "pdf_dy" in

  (* Set up kernel filter -  Silverman hack for now *)
  let kdf_width_x = stddev_x *. bandwidth *. 3. in
  let kdf_width_y = stddev_y *. bandwidth *. 3. *. 2. in
  let n_kdf_part_points_x = ((truncate (kdf_width_x /. bin_dx)) / 2) * 2 in
  let n_kdf_part_points_y = ((truncate (kdf_width_y /. bin_dy)) / 2) * 2 in
  Info.set info "kdf_width_x"         (Info.Float32 kdf_width_x);
  Info.set info "kdf_width_y"         (Info.Float32 kdf_width_y);
  Info.set info "n_kdf_part_points_x" (Info.Int     n_kdf_part_points_x);
  Info.set info "n_kdf_part_points_y" (Info.Int     n_kdf_part_points_y);

  let show_parameters _ =
    Printf.printf " stddev_x %f\n" stddev_x;
    Printf.printf " stddev_y %f\n" stddev_y;
    Printf.printf " kdf_width_x %f\n" (Info.float_of info "kdf_width_x");
    Printf.printf " kdf_width_y %f\n" (Info.float_of info "kdf_width_y");
    Printf.printf " x_min %f\n" (Info.float_of info "x_min");
    Printf.printf " x_max %f\n" (Info.float_of info "x_max");
    Printf.printf " y_min %f\n" (Info.float_of info "y_min");
    Printf.printf " y_max %f\n" (Info.float_of info "y_max");
    Printf.printf " n_kdf_part_points_x %f\n" (Info.float_of info "n_kdf_part_points_x");
    Printf.printf " n_kdf_part_points_y %f\n" (Info.float_of info "n_kdf_part_points_y");
    Printf.printf " n_data %f\n" (Info.float_of info "n_data");
    Printf.printf " n_hist_bins/n_pdf_points %d\n" (Info.int_of info "n_hist_bins" / Info.int_of info "n_pdf_points");
  in
  pv_debug t show_parameters;
            
  let compile_options = Pocl.compile_options pocl info "KDE" in
  let program = Pocl.compile_program pocl cl_kernel_source compile_options in

  pv_verbose t (fun _ -> Printf.printf "histogram...%!");
  let histogram_array = gpu_compute_histogram pocl info program sl_array true in
  pv_verbose t (fun _ -> Printf.printf "done\n%!");

  pv_verbose t (fun _ -> Printf.printf "kernel filtering rows...%!");
  let partial_pdf_array = gpu_compute_partial_pdf pocl info program histogram_array in
  pv_verbose t (fun _ -> Printf.printf "done\n%!");

  pv_verbose t (fun _ -> Printf.printf "kernel filtering columns...%!");
  let pdf_array = gpu_compute_full_bivariate_pdf pocl info program partial_pdf_array in
  pv_verbose t (fun _ -> Printf.printf "done\n%!");

  let weighted_total = (ODM.sum' pdf_array) *. bin_dx *. bin_dy in
  ODM.div_scalar pdf_array weighted_total
    
