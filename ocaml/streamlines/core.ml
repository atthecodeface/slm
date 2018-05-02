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
 * @file   core.ml
 * @brief  Core data, Info, and Workflow handling
 *
 * Up to date with python of git CS 189bfccdabc3371eafe8bcafa3bdfa8c241e56e4
 *
 * v}
 *)

(*f Module abbreviations *)
open Globals
module ODM = Owl.Dense.Matrix.Generic
module ODN = Owl.Dense.Ndarray.Generic

(** {1 Info module}

The Info module provides name/value pairs, with the value being of a
variant type permitting strings, ints and floats. The module is used
to create the OpenCL compiler '#define' options.

 *)

module Info =
struct

  (**  [Bad_value why] exception *)
  exception Bad_value of string

  (**  [t_value] Variant type to handle the values potentially required
  to be passed to OpenCL - ths info struct is used to generate the
  compiler options

    *)
  type t_value = 
    | Str     of string
    | Int     of int
    | Int32   of int32
    | Uint    of int
    | Uint32  of int32
    | Float32 of float

  (**  [t_named_value] - A named value that is the basic element of the Info structure
    *)
  type t_named_value = string * t_value

  (**  [t] - structure type of an Info module, just a list of named values *)
  type t = {
      mutable info : t_named_value list;
    }

  (**  [value_str v] - return string representation of a value

    The string returned is suitable for inclusion in a '-Dfred=<x>'
  *)
  let value_str v =
    match v with 
    | Str     x -> sfmt "'%s'" x
    | Int     x -> sfmt "%d" x
    | Int32   x -> sfmt "%ld" x
    | Uint    x -> sfmt "%uu" x
    | Uint32  x -> sfmt "%luu" x
    | Float32 x -> sfmt "%ff" x

  (**  [type_str v] - return string representation of the type of a value

    This is used for display purposes, not for communication with the
    OpenCL compiler.

  *)
  let type_str v =
    match v with 
    | Str     x -> "Str"
    | Int     x -> "Int"
    | Int32   x -> "Int32"
    | Uint    x -> "Uint"
    | Uint32  x -> "Uint32"
    | Float32 x -> "Float32"

  (**  [define_str nv] - generate '-D<name>=<value>' string for the OpenCL compiler *)
  let define_str (name,value) =
    sfmt "-D%s=%s" (String.uppercase_ascii name) (value_str value)

  (**  [create _] - create an Info block *)
  let create _ =
    { info=[]; }

  (**  [set t name value]

    This is for internal use; externally functions such as
    {!val:set_float32} should be used. so that {!type:t_value}
    constructors need not be used.

     *)
  let set t name value =
    t.info <- list_assoc_replace t.info name value

  (**  [add_uint t name x]

    Set name/(unsigned int x) as a name/value pair in the Info block

     *)
  let set_uint t name x =
    set t name (Uint x)

  (**  [set_uint32 t name x] 

     Set name/(unsigned int32 x) as a name/value pair in the Info block

     *)
  let set_uint32 t name x =
    set t name (Uint32 x)

  (**  [set_int t name x] 

     Set name/(int x) as a name/value pair in the Info block

     *)
  let set_int t name x =
    set t name (Int x)

  (**  [set_float32 t name x]

     Set name/(float32 x) as a name/value pair in the Info block

     *)
  let set_float32 t name x =
    set t name (Float32 x)

  (**  [set_str t name x] 

     Set name/(string x) as a name/value pair in the Info block.

     *)
  let set_str t name x =
    set t name (Str x)

  (**  [int_of t name] 

     Get int of the value of name in the Info block; this works for
    values that are int32s, or similar, but if an int cannot be created
    than a Bad_value exception is raised.

     *)
  let int_of t name =
    match List.assoc_opt name t.info with
    | Some Int x   -> x
    | Some Int32 x -> Int32.to_int x
    | Some Uint x   -> x
    | Some Uint32 x -> Int32.to_int x
    | _ -> raise (Bad_value (sfmt "int_of of '%s'" name))

  (**  [str_of t name] 

     Get string of the value of name in the Info block. If it is not a
     string then a Bad_value is raised.

     *)
  let str_of t name =
    match List.assoc_opt name t.info with
    | Some Str s -> s
    | _ -> raise (Bad_value (sfmt "str_of of '%s'" name))

  (**  [float_of t name] 

    Get float of the value of name in the Info block; this works for
    values that are ints, or similar, but if a float cannot be created
    than a Bad_value exception is raised.

     *)
  let float_of t name =
    match List.assoc_opt name t.info with
    | Some Int     x -> float x
    | Some Int32   x -> Int32.to_float x
    | Some Uint    x -> float x
    | Some Uint32  x -> Int32.to_float x
    | Some Float32 x   -> x
    | _ -> raise (Bad_value (sfmt "float_of of '%s'" name))

  (**  [iter f t]

    Apply f to every name/value pair in the Info block

     *)
  let iter f t =
    List.iter f t.info

  (**  [fold_left f acc t] 

    Fold f over every name/value pair in the Info block

   *)
  let fold_left f acc t =
    List.fold_left f acc t.info

  (**  [display t]

    Print out the info block contents prettily for debug, for example

     *)
  let display t =
    Printf.printf "Info block:\n";
    iter (fun (name,value) -> Printf.printf "  %30s : %9s %s\n" name (type_str value) (value_str value)) t;

  (*f All done *)
end

(** {1 Data and results structures}

Core data structures and results, used throughout the processing stages.

 *)

(**  t_core_data

  This is the data structure used to contain the data used throughout
  the processing stages.

  The properties covers all the workflows; the
  info is built from those properties with some short-term adaptations
  (since it is used for OpenCL compiler options, which vary slightly
  on invocation).

  The arrays are filled in by individual workflows; they are
  initialized to small arrays, before they are modified at the
  appropriate point in the workflow.

 *)
type t_core_data = {
    properties : Properties.t_props;
    info : Info.t;
    mutable roi_nx : int; (* roi_x_bounds.(1)-.(0) - the width of the ROI in samples *)
    mutable roi_ny : int; (* roi_y_bounds.(1)-.(0) - the height of the ROI in samples *)
    mutable roi_pixel_size : float; (* Used for general GPU and so on, from geodata file *)
    mutable roi_region : float array; (* Region LX, BY, RX, TY *)
    mutable pad_width : int; (* Used all over, copied from geodata state *)
    mutable roi_array             : t_ba_floats; (* roi_nx*roi_ny of float32 for region of interest extracted from dtm_array *)
    mutable basin_fatmask_array   : t_ba_chars; (* *padded* roi_nx*roi_ny of char with 255=ignore, 0=use *)
    mutable basin_mask_array      : t_ba_chars; (* *padded* roi_nx*roi_ny of char with 255=ignore, 0=use *)
    mutable x_roi_n_pixel_centers : t_ba_floats; (* roi_nx of float32 of pixel center xs *)
    mutable y_roi_n_pixel_centers : t_ba_floats; (* roi_ny of float32 of pixel center ys *)
    mutable u_array               : t_ba_floats; (* *padded* roi_nx*roi_ny of float32 for region of interest extracted from dtm_array *)
    mutable v_array               : t_ba_floats; (* *padded* roi_nx*roi_ny of float32 for region of interest extracted from dtm_array *)
    mutable seeds                 : t_ba_floats; (* *)
  }


(**  t_stats *)
type t_stats = {
  l_mean : float;
  l_min  : float;
  l_max  : float;

  c_mean : float;
  c_min  : float;
  c_max  : float;

  d_mean : float;
  d_min  : float;
  d_max  : float;
  }

(**  t_trace_results

  The trace results are produced by the trace workflow, and are
  consumed by later workflow stages.

 *)
type t_trace_results = {
    mutable streamline_arrays : bytes array array; (* .(downup) array of array of streamlines, each encoded as a stream of byte-pairs *)
    traj_nsteps_array  : t_ba_int16s; (* num_seeds*2 - number of steps for streamline from seed in direction *)
    traj_lengths_array : t_ba_floats; (* num_seeds*2 - streamline length from seed in direction *)
    slc_array          : t_ba_ints;   (* 2:downup * padded roi_nx * padded roi_ny : count of streamlines per pixel *)
    slt_array          : t_ba_floats; (* 2:downup * padded roi_nx * padded roi_ny : length of streamlines per pixel *)
    mutable sla_array  : t_ba_floats; (* 2:downup * padded roi_nx * padded roi_ny : area of streamlines per pixel *)
    mutable ds_stats   : t_stats option; (* Stats (downstream), only available once trajectories are found *)
    mutable us_stats   : t_stats option; (* Stats (upstream), only available once trajectories are found *)
  }

(** {1 Core toplevel functions}

 *)

(**  [create properties] 

  Create the core data structure from the given properties,
  initializing arrays minimally.

 *)
let create properties =
  let roi_pixel_size = 1. in
  let roi_nx = 1 in
  let roi_ny = 1 in
  let roi_region = [|0.;0.;0.;0.|] in
  let pad_width = 0 in
  let roi_array             = ba_float2d 1 1 in
  let basin_fatmask_array   = ba_char2d 1 1 in
  let basin_mask_array      = ba_char2d 1 1 in
  let x_roi_n_pixel_centers = ba_floats 1 in
  let y_roi_n_pixel_centers = ba_floats 1 in
  let u_array               = ba_float2d 1 1 in
  let v_array               = ba_float2d 1 1 in
  let seeds                 = ba_float2d 1 1 in
  let info                  = Info.create () in
  {
    properties;
    info;
    roi_pixel_size;
    roi_nx;
    roi_ny;
    roi_region;
    pad_width;
    roi_array;
    basin_mask_array;
    basin_fatmask_array;
    x_roi_n_pixel_centers;
    y_roi_n_pixel_centers;
    u_array;
    v_array;
    seeds;
  }

(**  [set_roi t roi]

  Allocate empty arrays for the Region-of-Interest

 *)
let set_roi t roi =
  let roi_nx = roi.(2) - roi.(0) in (* could downsample if roi_dx were 2 *)
  let roi_ny = roi.(3) - roi.(1) in (* could downsample if roi_dy were 2 *)

  t.roi_nx <- roi_nx;
  t.roi_ny <- roi_ny;
  t.roi_region <- [| float roi.(0); float roi.(1); float roi.(2); float roi.(3) |];

  t.roi_array             <- ba_float2d roi_nx roi_ny;
  t.basin_fatmask_array   <- ba_char2d  roi_nx roi_ny;
  t.basin_mask_array      <- ba_char2d  roi_nx roi_ny;
  t.x_roi_n_pixel_centers <- ba_floats  roi_nx;
  t.y_roi_n_pixel_centers <- ba_floats  roi_ny;
  t.u_array <- ba_float2d 1 1;
  t.v_array <- ba_float2d 1 1;
  ODN.fill t.roi_array nan;
  ODN.fill t.basin_mask_array    '\255';
  ODN.fill t.basin_fatmask_array '\255';
  ()

