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
 * v}
 *)

(*a Module abbreviations *)
open Globals
module ODM = Owl.Dense.Matrix.Generic
module ODN = Owl.Dense.Ndarray.Generic

(*a Info module *)
module Info =
struct
  exception Bad_value of string

  (*t t_value *)
  type t_value = 
    | Str     of string
    | Int     of int
    | Int32   of int32
    | Uint    of int
    | Uint32  of int32
    | Float32 of float

  (*t t_named_value - internal type *)
  type t_named_value = string * t_value

  (*t t - type of an Info block *)
  type t = {
      mutable info : t_named_value list;
    }

  (*f [value_str v] - return string representation of a value *)
  let value_str v =
    match v with 
    | Str     x -> sfmt "'%s'" x
    | Int     x -> sfmt "%d" x
    | Int32   x -> sfmt "%ld" x
    | Uint    x -> sfmt "%uu" x
    | Uint32  x -> sfmt "%luu" x
    | Float32 x -> sfmt "%ff" x

  (*f [type_str v] - return string representation of the type of a value *)
  let type_str v =
    match v with 
    | Str     x -> "Str"
    | Int     x -> "Int"
    | Int32   x -> "Int32"
    | Uint    x -> "Uint"
    | Uint32  x -> "Uint32"
    | Float32 x -> "Float32"

  (*f [define_str nv] - return '-D<name>=<value>' compilation 'DEFINE' string *)
  let define_str (name,value) =
    sfmt "-D%s=%s" (String.uppercase_ascii name) (value_str value)

  (*f [create _] - create an Info block *)
  let create _ =
    { info=[]; }

  (*f [add_value t name value] - internal use - add a name/value pair to the Info block  *)
  let add_value t name value =
    t.info <- t.info @ [(name,value)]

  (*f [set t name value] - set name/value pair in the Info block *)
  let set t name value =
    t.info <- list_assoc_replace t.info name value

  (*f [add_uint t name x] - add name/(unsigned int x) as a name/value pair in the Info block *)
  let add_uint t name x =
    add_value t name (Uint x)

  (*f [add_uint32 t name x] - add name/(unsigned int32 x) as a name/value pair in the Info block *)
  let add_uint32 t name x =
    add_value t name (Uint32 x)

  (*f [add_float32 t name x] - add name/(float32 x) as a name/value pair in the Info block *)
  let add_float32 t name x =
    add_value t name (Float32 x)

  (*f [add_str t name x] - add name/(string x) as a name/value pair in the Info block *)
  let add_str t name x =
    add_value t name (Str x)

  (*f [int_of t name] - get int of the value of name in the Info block *)
  let int_of t name =
    match List.assoc_opt name t.info with
    | Some Int x   -> x
    | Some Int32 x -> Int32.to_int x
    | Some Uint x   -> x
    | Some Uint32 x -> Int32.to_int x
    | _ -> raise (Bad_value (sfmt "int_of of '%s'" name))

  (*f [str_of t name] - get int of the value of name in the Info block *)
  let str_of t name =
    match List.assoc_opt name t.info with
    | Some Str s -> s
    | _ -> raise (Bad_value (sfmt "str_of of '%s'" name))

  (*f [float_of t name] - get float of the value of name in the Info block *)
  let float_of t name =
    match List.assoc_opt name t.info with
    | Some Int     x -> float x
    | Some Int32   x -> Int32.to_float x
    | Some Uint    x -> float x
    | Some Uint32  x -> Int32.to_float x
    | Some Float32 x   -> x
    | _ -> raise (Bad_value (sfmt "float_of of '%s'" name))

  (*f [iter f t] - apply f:nv->unit to every name/value pair in the Info block *)
  let iter f t =
    List.iter f t.info

  (*f [fold_left f acc t] - fold f:'a->nv->'a over every name/value pair in the Info block *)
  let fold_left f acc t =
    List.fold_left f acc t.info

  (*f display *)
  let display t =
    Printf.printf "Info block:\n";
    iter (fun (name,value) -> Printf.printf "  %30s : %9s %s\n" name (type_str value) (value_str value)) t;

  (*f All done *)
end

(*a Data and results structures *)
(*t t_core_data *)
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

(*t t_trace_results *)
type t_trace_results = {
    mutable streamline_arrays : bytes array array; (* .(downup) array of array of streamlines, each encoded as a stream of byte-pairs *)
    traj_nsteps_array  : t_ba_int16s; (* num_seeds*2 - number of steps for streamline from seed in direction *)
    traj_lengths_array : t_ba_floats; (* num_seeds*2 - streamline length from seed in direction *)
    slc_array          : t_ba_ints;   (* 2:downup * padded roi_nx * padded roi_ny : count of streamlines per pixel *)
    slt_array          : t_ba_floats; (* 2:downup * padded roi_nx * padded roi_ny : length of streamlines per pixel *)
    sla_array          : t_ba_floats; (* 2:downup * padded roi_nx * padded roi_ny : area of streamlines per pixel *)
  }

(*a Core toplevel *)
(*f [create properties] - Create the core data structure from the given properties *)
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

(*f [set_roi t roi] - Create Bigarrays for the core data given a region of interest and padding *)
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
  ODN.fill t.basin_mask_array '\255';
  ODN.fill t.basin_fatmask_array '\255';
  ()

