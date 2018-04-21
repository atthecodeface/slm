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
 * @file   preprocess.ml
 * @brief  Preprocessing
 * v}
 *)

(*a Module abbreviations *)
(*	"preprocess": {	
		"do_simple_gradient_vector_field" : true,
		"do_normalize_speed" : true,
		
		"vecsum_threshold" : 0.95,
		"divergence_threshold" : -0.5,
		"curl_threshold" : 0.0
	},
 *)
open Globals
open Properties
open Core
module ODM = Owl.Dense.Matrix.Generic
module ODN = Owl.Dense.Ndarray.Generic
module BA=Bigarray

(*a Types *)
type t = {
    props : t_props_preprocess;
    mutable roi_gradx_array : t_ba_floats;
    mutable roi_grady_array : t_ba_floats;
    mutable where_looped : (int * int) list;
    pad_width : int;
  }

let pv_noisy   t = pv_noisy   t.props.verbosity
let pv_debug   t = pv_debug   t.props.verbosity
let pv_info    t = pv_info    t.props.verbosity
let pv_verbose t = pv_verbose t.props.verbosity

(*a Basics *)
(*f [create props] *)
let create props =
  let pad_width = props.geodata.pad_width in
  let roi_gradx_array = ba_float2d 1 1 in
  let roi_grady_array = ba_float2d 1 1 in
  {
    props=props.preprocess;
    roi_gradx_array;
    roi_grady_array;
    where_looped=[];
    pad_width;
  }

(*a Stuff *)
(*f vector_length *)
let vector_length a b = sqrt (a*.a +. b*.b)

(*f fold2d *)
let fold2d f acc src =
  let acc_r = ref acc in
  let f_acc x y v =
    acc_r := f x y (!acc_r) v
  in
  ODM.iteri_2d f_acc src;
  !acc_r

(*f get_flow_vector *)
let get_flow_vector nn = 
  let nn = Char.code nn in
  if      ((nn land  16)<>0) then (-1,-1) (* SW *)
  else if ((nn land  32)<>0) then (+1,-1) (* SE *)
  else if ((nn land  64)<>0) then (-1,+1) (* NW *)
  else if ((nn land 128)<>0) then (+1,+1) (* NE *)
  else if ((nn land   1)<>0) then (-1, 0) (* W *)
  else if ((nn land   2)<>0) then (+1, 0) (* E *)
  else if ((nn land   4)<>0) then ( 0,-1) (* S *)
  else if ((nn land   8)<>0) then ( 0,+1) (* N *)
  else (0,0)

(*f get_unit_flow_vector *)
let get_unit_flow_vector nn = 
  let (dx,dy) = get_flow_vector nn in
  let l = dx*dx+dy*dy in
  if l=0 then (0.,0.) else (
    let scale = sqrt (float l) in
    ((float dx) /. scale, (float dy) /. scale)
  )

(*f has_one_diagonal_outflow filter
  Return mask of w,x,y,z otherwise for SW, SE, NW, NE outflows
    but 0 if there are any N, S, E or W outflows.

  Consider a gradient field of the topology; it is composed using
    dh/dx and dh/dy. Hence diagonal outflows are missed at the
    vector flow, which is generated from that gradient field

  Hence if a point only has diagonal outflows it is a 'blockage'.

  Consider in particular:
  roi -> roi_gradx_array -> roi_grady_array -> normalized velocity
  6 6 6 6 6     
  6 7 9 9 6    +1.5 +1.0 -1.5   -1.5 -1.0 -1.5    -,+ -,+ +,+
  6 9 8 9 6 -> +1.0    0 -1.0 ->-1.0    0    0 -> -,+ 0,0 +,0
  6 9 9 9 6    +1.5    0 -1.5   +1.5 +1.0 +1.5    -,- 0,- +,-
  6 6 6 6 6     

  This is mapped to blockages and blocked_neighbors

  6 6 6 6 6     
  6 7 9 9 6    0  0  0    0  0  0    NW NW NE
  6 9 8 9 6 -> 0 64  0 -> 0  0  1 -> NW NW  W
  6 9 9 9 6    0  0  0    0  8 64    SW  N NW
  6 6 6 6 6     

  The DTM will have been preconditioned by a GIS to drain to the edge
  using eight different directions

 *)
let has_just_diagonal_outflows get x y =
  let h = get x y in
  let nn = ref 0 in
  if (get (x-1) (y-1)) < h then nn := !nn + 16;  (* SW *)
  if (get (x  ) (y-1)) < h then nn := !nn + 4;   (* S  *)
  if (get (x+1) (y-1)) < h then nn := !nn + 32;  (* SE *)
  if (get (x-1) (y  )) < h then nn := !nn + 1;   (* W *)
  if (get (x+1) (y  )) < h then nn := !nn + 2;   (* E *)
  if (get (x-1) (y+1)) < h then nn := !nn + 64;  (* NW *)
  if (get (x  ) (y+1)) < h then nn := !nn + 8;   (* N  *)
  if (get (x+1) (y+1)) < h then nn := !nn + 128; (* NE *)
  if (!nn land (16+32+64+128)) = !nn then (Char.chr !nn) else (Char.chr 0)

(*f upstream_of_diagonal_outflow

  The upstream pixels of a blockage need to be found and fixed; why is not
    quite clear.

  This helps to remove loops introduced by tweaking the vector field at the
    blockages

 *)
let upstream_of_diagonal_outflow get x y =
  let nn = ref 0 in
  if (get (x-1) (y-1)) = '\016' then nn := !nn + 16;   (* SW has a single SW diagonal *)
  if (get (x  ) (y-1)) = '\016' then nn := !nn + 4;    (* S has a single SW diagonal *)
  if (get (x  ) (y-1)) = '\032' then nn := !nn + 4;    (* S has a single SE diagonal *)
  if (get (x+1) (y-1)) = '\032' then nn := !nn + 32;   (* SE has a single SE diagonal *)

  if (get (x-1) (y  )) = '\016' then nn := !nn + 1;    (* W has a single SW diagonal *)
  if (get (x-1) (y  )) = '\064' then nn := !nn + 1;    (* W has a single NW diagonal *)

  if (get (x+1) (y  )) = '\032' then nn := !nn + 2;    (* E has a single SE diagonal *)
  if (get (x+1) (y  )) = '\128' then nn := !nn + 2;   (* E has a single NE diagonal *)

  if (get (x-1) (y+1)) = '\064' then nn := !nn + 64;   (* NW has a single NW diagonal *)
  if (get (x  ) (y+1)) = '\064' then nn := !nn + 8;    (* N has a single NW diagonal *)
  if (get (x  ) (y+1)) = '\128' then nn := !nn + 8;   (* N has a single NE diagonal *)
  if (get (x+1) (y+1)) = '\128' then nn := !nn + 128; (* NE has a single NE diagonal *)

  Char.chr !nn

(*f find_blockages *)
let find_blockages t data =
    let roi_array = data.roi_array in
    pv_info t (fun _ -> Printf.printf "Finding blockages...\n%!");
    let blockages_array = ODN.(empty Bigarray.char (shape roi_array)) in
    ba_filter 3 3 has_just_diagonal_outflows '\000' data.roi_array blockages_array;
    let blocked_neighbors_array = ODN.(empty Bigarray.char (shape roi_array)) in
    ba_filter 3 3 upstream_of_diagonal_outflow '\000' blockages_array blocked_neighbors_array;
    (blockages_array, blocked_neighbors_array)

(*f get_blockages_lists *)
let get_blockages_lists blockages =
  let (blockages_array, blocked_neighbors_array) = blockages in
  let acc_non_zero x y acc v = if v='\000' then acc else ((x,y)::acc) in
  let where_blockages = fold2d acc_non_zero [] blockages_array in
  let where_blocked_neighbors = fold2d acc_non_zero [] blocked_neighbors_array in
  (where_blockages, where_blocked_neighbors)

(*f show_blockages *)
let show_blockages blockages =
    Printf.printf "Finding blockages\n%!";
    let (where_blockages, _) = get_blockages_lists blockages in
    Printf.printf "found %d blockages" (List.length where_blockages);
    if true then ( (* self.state.noisy: *)
      if where_blockages<>[] then (
        Printf.printf "Blockages at:\n";
        let (blockages_array, _) = blockages in
        let show_blockage (x,y) =
          let dx,dy = get_flow_vector (ODM.get blockages_array x y) in
          Printf.printf "[%d, %d] @ [%d, %d] => [%d, %d]\n" dx dy x y (x+dx) (y+dy)
        in
        let x = Array.of_list where_blockages in
        Array.sort (fun (ax,ay) (bx,by) -> let r = compare ax bx in (if r=0 then compare ay by else r)) x;
        (*List.iter show_blockage where_blockages*)
        Array.iter show_blockage x
      ) else (
        Printf.printf "Blockages at.. none:\n"
      );
    );
    Printf.printf "Done\n%!";
    ()

(*f compute_topo_gradient_field
    Differentiate ROI topo in x and y directions using simple numpy method.
    
    Args:
        roi_array (numpy.array):
        
    Returns:
        numpy.array,numpy.array: ROI topo x gradient, y gradient
 *)
let compute_topo_gradient_field roi_array =
  let gradient_x get x y = ((get (x+1) y) -. (get (x-1) y)) *. 0.5 in
  let gradient_y get x y = ((get x (y+1)) -. (get x (y-1))) *. 0.5 in
  let (w,h)   = ODM.shape roi_array in
  let u_array = ODM.(empty BA.float32 w h) in
  let v_array = ODM.(empty BA.float32 w h) in
  ba_filter 3 1 gradient_x 0. roi_array u_array;
  ba_filter 1 3 gradient_y 0. roi_array v_array;
  (u_array, v_array)

(*f normalize_arrays u_array v_array *)
let normalize_arrays u_array v_array =
  let speed_array = ODM.map2 vector_length u_array v_array in
  ODM.(u_array /= speed_array);
  ODM.(v_array /= speed_array);
  (u_array, v_array)

(*f compute_gradient_velocity_field
    Compute normalized gradient velocity vector field from ROI topo grid.

  This is effectively an array of unit (dh/dx, dh/dy) vectors,
    but held as two scalar arrays dh/dx = u and dh/dy = v

 *)
let compute_gradient_velocity_field roi_gradx_array roi_grady_array =
trace __POS__;
  let u_array = ODM.neg roi_gradx_array in
  let v_array = ODM.neg roi_grady_array in
trace __POS__;
  normalize_arrays u_array v_array

(*f check_has_loop
Get UV of x,y and its neighbors, the speed=mod(avg uv) divergence and curl

Geograpically:
 uv01 uv11
 uv00 uv10

speed = sqrt((+u00+u01+u10+u11)^2 + (+v00+v01+v10+v11)^2 )/4
div   = -u00-u01+u10+u11 -v00+v01-v10+v11a
curl  = +u00-u01+u10-u11 -v00-v01+v10+v11

Another way to look at uv is uvNM = (dh/dx (x+N,y+M), dh/dy (x+N,y+M))

Hence:
 uv00 . (-1,-1) = -dh/dx(x,y) - dh/dy(x,y)
etc

Hence div is
   uv01.NW + uv11.NE
 + uv00.SW + uv10.SE

And curl is
   uv01.SW + uv11.NW
 + uv00.SE + uv10.NE

As the vector field is a gradient vector field it ought to have a curl
of zero (it would in a perfect world) and divergence should be greater
than zero (as there is always a downhill); 

 *)
let calc_speed_div_curl x y u v  =
  let u = Bigarray.array2_of_genarray u in
  let v = Bigarray.array2_of_genarray v in
  let u00 = u.{x+0,y+0} in
  let u01 = u.{x+0,y+1} in
  let u10 = u.{x+1,y+0} in
  let u11 = u.{x+1,y+1} in
  let v00 = v.{x+0,y+0} in
  let v01 = v.{x+0,y+1} in
  let v10 = v.{x+1,y+0} in
  let v11 = v.{x+1,y+1} in
  let velocity_u = u00 +. u01 +. u10 +. u11 in
  let velocity_v = v00 +. v01 +. v10 +. v11 in
  let speed = (sqrt (velocity_u*.velocity_u +. velocity_v*.velocity_v)) /. 4. in
  let div  = 0. -. u00 -. u01 +. u10 +. u11 -. v00 +. v01 -. v10 +. v11 in
  let curl = 0. +. u00 -. u01 +. u10 -. u11 -. v00 -. v01 +. v10 +. v11 in
  (speed, div, curl)

(*f break_out_of_loop *)
let break_out_of_loop data (x,y) =
  let roi_nx = data.roi_nx in
  let roi_ny = data.roi_ny in
  let well_within_bounds xi yi = (xi>=1) && (xi<roi_nx-1) && (yi>1) && (yi<roi_ny-1) in
    (*  let well_within_bounds xi yi = false in*)
  if not (well_within_bounds x y) then (
  ) else (
    let lowest_neighbour acc nn =
      let (min_nn, min_h) = acc in
      let (dx,dy) = get_flow_vector nn in
      let h = ODM.get data.roi_array (x+dx) (y+dy) in
      if (h < min_h) then (nn, h) else acc
    in
    let nns = [|'\001';'\002';'\004';'\008';'\016';'\032';'\064';'\128';|] in
    let (nn_min,_) = Array.fold_left lowest_neighbour ('\000', infinity) nns in
    let (dx,dy) = get_unit_flow_vector nn_min in
    ODM.set data.u_array x y dx;
    ODM.set data.v_array x y dy;
  )

(*f find_and_fix_loops *)
let find_and_fix_loops t data =
    pv_info t (fun _ -> Printf.printf "Finding and fixing loops...\n");
    let roi_nx = data.roi_nx in
    let roi_ny = data.roi_ny in
    let where_looped = ref [] in
    for xi=0 to roi_nx-2 do
      for yi=0 to roi_ny-2 do
        let (vecsum,divergence,curl) = calc_speed_div_curl xi yi data.u_array data.v_array in
        if ((vecsum < t.props.vecsum_threshold) && 
              (divergence <= t.props.divergence_threshold) && 
                ((abs_float curl) >= t.props.curl_threshold)) then (
          where_looped := (xi,yi) :: (xi+1,yi) :: (xi+1,yi+1) :: (xi,yi+1) :: !where_looped
        )
      done;
    done;
    pv_info t (fun _ -> Printf.printf "...found %d...\n" (List.length !where_looped));
    List.iter (break_out_of_loop data) !where_looped;
    pv_info t (fun _ -> Printf.printf "...done\n");
    !where_looped

(*f fix_blockages *)
let fix_blockages t blockages data =
  pv_info t (fun _ -> Printf.printf "Fixing blockages...\n%!");
  let (blockages_array, blocked_neighbors_array) = blockages in
  let (where_blockages, where_blocked_neighbors) = get_blockages_lists blockages in
  if where_blockages<>[] then (
    let set_uv_from_flow_vector x y nn =
      let (dx,dy) = get_unit_flow_vector nn in
      ODM.set data.u_array x y dx;
      ODM.set data.v_array x y dy;
    in
    let fix_blockage (x,y) = set_uv_from_flow_vector x y (ODM.get blockages_array x y) in
    let fix_blocked_neighbor (x,y) = set_uv_from_flow_vector x y (ODM.get blocked_neighbors_array x y) in
    pv_info t (fun _ -> Printf.printf "...%d blockages...\n%!" (List.length where_blockages));
    List.iter fix_blockage where_blockages;
    pv_info t (fun _ -> Printf.printf "...%d blocked neighbors...\n%!" (List.length where_blocked_neighbors));
    List.iter fix_blocked_neighbor where_blocked_neighbors;
    pv_info t (fun _ -> Printf.printf "...done\n%!");
  ) else (
    pv_info t (fun _ -> Printf.printf "...none to fix\n%!");
  )

(*f conditioned_gradient_vector_field t data
        Compute topographic gradient vector field on a preconditioned DTM.
        
        The preconditioning steps are:
        
        1. Find blockages in gradient vector field 
        2. Calculate surface derivatives (gradient & 'curvature')
        3. Set gradient vector field magnitudes ('speeds')
        4. Find and fix loops in gradient vector field
        5. Fix blockages in gradient vector field 
        6. Set initial streamline points ('seeds')
 *)
let conditioned_gradient_vector_field t data =
    pv_info t (fun _ -> Printf.printf "Precondition gradient vector field by fixing loops & blockages\n%!");

    let (roi_gradx_array, roi_grady_array) = compute_topo_gradient_field data.roi_array in
    let (u_array, v_array) = compute_gradient_velocity_field roi_gradx_array roi_grady_array in
    data.u_array <- u_array;
    data.v_array <- v_array;
    t.roi_gradx_array <- roi_gradx_array;
    t.roi_grady_array <- roi_grady_array;
    t.where_looped <- [];

    let do_fixes = true in
    if do_fixes then (
      let blockages = find_blockages t data in
      pv_noisy t (fun _ -> show_blockages blockages);
      t.where_looped <- find_and_fix_loops t data;
      fix_blockages t blockages data;
    );
    let (u_array, v_array) = 
      if t.props.do_normalize_speed then (
        normalize_arrays u_array v_array
      ) else (
    u_array,v_array
        (* Cannot do the next as speed_array is not defined on this code path
    let (u_array, v_array) = unnormalize_velocity_field u_array v_array speed_array in *)
      )
    in
    data.u_array <- get_padded_array u_array t.pad_width 0.;
    data.v_array <- get_padded_array v_array t.pad_width 0.;
    ()
        
(*f mask_nan_uv *)
let mask_nan_uv data =
    Printf.printf "Mask out bad uv pixels...\n%!";
    let bma   = data.basin_mask_array in
    let bfma  = data.basin_fatmask_array in
    let ua    = data.u_array in
    let va    = data.v_array in
    let bma'  = BA.array1_of_genarray (ODN.flatten bma) in
    let bfma' = BA.array1_of_genarray (ODN.flatten bfma) in
    let ua'   = BA.array1_of_genarray (ODN.flatten ua) in
    let va'   = BA.array1_of_genarray (ODN.flatten va) in
    let masked = ref 0 in
    ODN.iter2i (fun i u v -> if (is_nan u) || (is_nan v) then (bma'.{i}<-'\255';bfma'.{i}<-'\255';)) ua va;
    ODN.iteri  (fun i b   -> if b<>'\000' then (masked:=!masked+1;ua'.{i}<-0.;va'.{i}<-0.;)) bma;
    if (!masked>0) then Printf.printf "...masked %d" !masked;
    Printf.printf "...done\n%!"
        
(*f raw_gradient_vector_field
        Compute topographic gradient vector field without preconditioning the DTM.
let raw_gradient_vector_field data = 
        self.print('Compute raw gradient vector field')  
        (self.roi_gradx_array,self.roi_grady_array) = derivatives(self.geodata.roi_array)
        self.speed_array = set_speed_field(self.u_array, self.v_array)
 *)


(*f process t data *)
let process t data = 
  let w = workflow_start "preprocess" t.props.verbosity in
  (* if self.state.do_condition: *)
  conditioned_gradient_vector_field t data;
  (*        else:
            self.raw_gradient_vector_field()
   *)
  mask_nan_uv data;
  workflow_end w

