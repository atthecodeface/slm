(** Copyright (C) 2017-2018,  Colin P Stark and Gavin J Stark.  All rights reserved.
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
 * @file   geodata.ml
 * @brief  Module to provide loading of Geotiff data and filling out the core data
 *
 *)

(*a Libraries *)
open Globals
open Core
open Properties
module Option = Batteries.Option
module Owl_plot=Local_owl_plot
module ODM = Owl.Dense.Matrix.Generic
module ODN = Owl.Dense.Ndarray.Generic

(*a Types *)
(*t t_plot *)
type t_plot = {
    props : t_props_plot;
    region : int list list; (* min, max, stride list for x and y *)
    plot : Owl_plot.handle;
    subplots : int * int;
    mutable current_sub : int;
    filename : string;
  }

(*a Useful functions *)
let pv_noisy   t = pv_noisy   t.props.verbosity
let pv_debug   t = pv_debug   t.props.verbosity
let pv_info    t = pv_info    t.props.verbosity
let pv_verbose t = pv_verbose t.props.verbosity

let matrix_for_plot_float32 ~region ba = 
  let z = ODN.(cast_s2d (get_slice region ba)) in
  let z = ODM.map (fun a->if is_nan a then 0. else a) z in
  let z = ODM.flip z in
  z

let matrix_for_plot_byte ~region ba = 
  let (ny,nx) = ODM.shape ba in
  let z = ODM.init_2d (Bigarray.float64) ny nx (fun y x -> float (Char.code (ODM.get ba y x))) in
  let z = ODN.(get_slice region z) in
  let z = ODM.flip z in
  z

(*f [generate_hillshade ba azimuth_degrees angle_altitude_degrees]

        Hillshade render DTM topo with light from direction azimuth_degrees and tilt
        angle_attitude_degrees

    Generate a [0,1] matrix with each pixel being a shaded relief
 *)
let generate_hillshade ba azimuth_degrees altitude_degrees =
    let pi = 4. *. atan 1. in
    let half_pi = pi /. 2. in
    let gradient_x get x y = ((get (x+1) y) -. (get (x-1) y)) *. 0.5 in
    let gradient_y get x y = ((get x (y+1)) -. (get x (y-1))) *. 0.5 in
    let x_grad = ba_filter 3 1 gradient_x 0. ba (ODN.copy ba) in
    let y_grad = ba_filter 1 3 gradient_y 0. ba (ODN.copy ba) in
    let slope  = ODM.map2 (fun x' y' -> half_pi -. (atan (sqrt ((x' ** 2.) +. (y' ** 2.))))) x_grad y_grad in
    let aspect = ODM.map2 (fun x' y' -> atan2 (-. x') y')                                    x_grad y_grad in
    let azimuth  = (azimuth_degrees +. 180.) *. pi /. 180. in
    let altitude = altitude_degrees          *. pi /. 180. in
    let alt_sin = sin altitude in
    let alt_cos = cos altitude in
    let shade sl asp = ((alt_sin*.(sin sl) +. alt_cos *. (cos sl) *. (cos (azimuth -. asp)))+. 1.) /. 2. in
    ODM.map2 shade slope aspect

(*f [generate_masked_roi ba mask]

    Generate a [0,0.99] matrix with each pixel being a height, with masked out as 1.0
 *)
let generate_masked_roi ba mask =
  let mask' = ODN.flatten mask |> Bigarray.array1_of_genarray in
  let a_height = ba_foldi (fun i acc v -> if mask'.{i}='\000' then v else acc) nan ba in
  let mask_as_nan i v = if mask'.{i}='\000' then v else a_height in
  let masked_array  = ODN.mapi mask_as_nan ba in
  let min = ODN.(get (min masked_array) [|0|]) in
  let max = ODN.(get (max masked_array) [|0|]) in
  let range = (max -. min) /. 0.99 in
  Printf.printf "A_height %g Max %g min %g range %g\n" a_height max min range;
  let scale i v = if (is_nan v) then 0. else (0.01+.((v -. min) /. range)) in
  mapi_ scale masked_array

let select_subplot plot c r =
    Printf.printf "Select subplot %d,%d\n%!" c r;
    Owl_plot.subplot plot c r;
    plot

let output plot =
    Printf.printf "Write plot file\n%!";
    Owl_plot.output plot;
    Printf.printf "Written plot file\n%!";
    ()

let set_xrange plot = Owl_plot.set_xrange plot

let set_yrange plot = Owl_plot.set_yrange plot

let all = []

let plot_plplot plot plplot =
  let h:Owl_plot.handle = plot in
  let p = h.pages.(h.current_page) in
  p.plots <- Array.append p.plots [|plplot|];
  plot



let pi = atan 1.0 *. 4.0
let transform xmax x y =
  x, y /. 4.0 *. (3.0 -. cos (pi *. x /. xmax))

let arrow_x = [|-0.5; 0.5; 0.3; 0.5; 0.3; 0.5|]
let arrow_y = [|0.0; 0.0; 0.2; 0.0; -0.2; 0.0|]
let arrow2_x = [|-0.5; 0.3; 0.3; 0.5; 0.3; 0.3|]
let arrow2_y = [|0.0; 0.0; 0.2; 0.0; -0.2; 0.0|]

let circulation () =
  let open Plplot in
  let nx = 20 in
  let ny = 20 in
  let dx = 1.0 in
  let dy = 1.0 in

  let xmin = -. float_of_int nx /. 2.0 *. dx in
  let xmax = float_of_int nx /. 2.0 *. dx in
  let ymin = -. float_of_int ny /. 2.0 *. dy in
  let ymax = float_of_int ny /. 2.0 *. dy in

  let xg = Array.make_matrix nx ny 0.0 in
  let yg = Array.make_matrix nx ny 0.0 in
  let u = Array.make_matrix nx ny 0.0 in
  let v = Array.make_matrix nx ny 0.0 in

  (* Create data - circulation around the origin. *)
  for i = 0 to nx - 1 do
    let x = (float_of_int i -. float_of_int nx /. 2.0 +. 0.5) *. dx in
    for j = 0 to ny - 1 do
      let y = (float_of_int j -. float_of_int ny /. 2.0 +. 0.5) *. dy in
      xg.(i).(j) <- x;
      yg.(i).(j) <- y;
      u.(i).(j) <- y;
      v.(i).(j) <- -. x;
    done
  done;

  (* Plot vectors with default arrows *)
  plenv xmin xmax ymin ymax 0 0;
  plsvect_reset ();
  pllab "(x)" "(y)" "#frPLplot Example 22 - circulation";
  plcol0 2;
  plset_pltr (pltr2 xg yg);
  plvect u v 0.0;
  plcol0 1;
  ()


let x =[]
let plotvect ?region:(region=x) u v plot =
  let u = ODM.(cast_s2d (get_slice region u) |> to_arrays) in
  let v = ODM.(cast_s2d (get_slice region v) |> to_arrays) in
  let nx = Array.length u in
  let ny = Array.length u.(0) in
  let dx, dy = 1.0, 1.0 in
  let cgrid2_xg = Array.make_matrix nx ny 0.0 in
  let cgrid2_yg = Array.make_matrix nx ny 0.0 in
  for i = 0 to nx - 1 do
    let x = (float i -. float nx /. 2.0 +. 0.5) *. dx in
    for j = 0 to ny - 1 do
      let y = (float j -. float ny /. 2.0 +. 0.5) *. dy in
      cgrid2_xg.(i).(j) <- x;
      cgrid2_yg.(i).(j) <- y
    done
  done;
  let xmin = -. float_of_int nx /. 2.0 *. dx in
  let xmax = float_of_int nx /. 2.0 *. dx in
  let ymin = -. float_of_int ny /. 2.0 *. dy in
  let ymax = float_of_int ny /. 2.0 *. dy in
  Printf.printf "Plotting vectors of size %d,%d\n%!" nx ny;
  let plplot _ =
    let open Plplot in
    (*plenv xmin xmax ymin ymax 0 0;*)
    plsvect_reset ();
    pllab "(x)" "(y)" "#frUV surface vector for ROI";
    plcol0 2;
    plset_pltr (pltr2 cgrid2_xg cgrid2_yg);
    plvect u v 0. (* last is scale *);
    plunset_pltr ();
    plcol0 1;
    ()
  in
  set_xrange plot xmin xmax;
  set_yrange plot ymin ymax;
  plot_plplot plot plplot
  

let plot_roi_contours ?region:(region=all) data plot =
    Printf.printf "Plot ROI contours\n%!";
    let lx = data.roi_region.(0) in
    let by = data.roi_region.(1) in
    let rx = data.roi_region.(2) in
    let ty = data.roi_region.(3) in
    let x, y = ODM.meshgrid (Bigarray.float64) lx rx by ty data.roi_nx data.roi_ny in
    let x = ODM.get_slice region x in
    let y = ODM.get_slice region y in
    let z = matrix_for_plot_float32 ~region:region data.roi_array in
    Owl_plot.set_title plot "Contours of ROI";
    Owl_plot.mesh ~h:plot ~spec:Owl_plot.([ Azimuth 245.; Altitude 70.; Contour ]) x y z; (* azimuth 0 is (0,+1,0); 90 is (+1,0,0) *)
    Printf.printf "Done\n%!";
    plot


let plot_roi_image ?region:(region=all) data plot =
    Printf.printf "Plot ROI shaded relief image\n%!";
    Owl_plot.set_title plot "ROI";
    let masked_array  = generate_masked_roi data.roi_array data.basin_mask_array in
    let shaded = generate_hillshade data.roi_array 135. 25. in
    let p = plot in
    let z = matrix_for_plot_float32 ~region:region masked_array in
    Owl_plot.(gjs_image ~h:p ~preplot_options:[image_boundary; image_set_page_size; image_remove_axes;] ~plot_options:[set_cmap ~num_col:256 ~masked:0.01 ~alpha:1.0 cmap_terrain] z);
    let z = matrix_for_plot_float32 ~region:region shaded in
    Owl_plot.(gjs_image ~h:p ~plot_options:[set_cmap ~num_col:256 ~alpha:0.3 cmap_grey] z); (* shaded_relief_hillshade_alpha *)
    Printf.printf "Done\n%!";
    plot

let oldplot_roi_image ?region:(region=all) data plot =
    Printf.printf "Plot ROI image\n%!";
    let z = matrix_for_plot_float32 ~region:region data.roi_array in
    Owl_plot.set_title plot "ROI";
    Owl_plot.image ~h:plot  z;
    Printf.printf "Done\n%!";
    plot

let plot_basin_mask ?region:(region=all) data plot =
    Printf.printf "Plot basin_mask image\n%!";
    let z = matrix_for_plot_byte ~region:region data.basin_mask_array in
    Owl_plot.image ~h:plot  z;
    Owl_plot.set_title plot "Basin mask";
    Owl_plot.image ~h:plot  z;
    Printf.printf "Done\n%!";
    plot



(*f plot_roi_shaded_relief

  Hillshade view of ROI of DTM

 *)
let new_figure ?title ?window_title ?region ?x_pixel_scale ?y_pixel_scale t =

  let plot = t.plot in
  if t.current_sub>0 then (
    let (m,n) = t.subplots in
    if t.current_sub = m*n then (
      Owl_plot.new_page plot;
      t.current_sub <- 0;
    )
  );
  t.current_sub <- t.current_sub + 1;

  Owl_plot.set_background_color plot 255 255 255; (* Set before final output, global for all output *)
  Owl_plot.set_pen_size  plot 2.0; (* linewidth=2.0, markersize=10) *)
  Owl_plot.set_font_size plot 10.; (* mm - or possibly pixels? - , not pt (1mm approx = 3pt) *)
  let x_pixel_scale = Option.default 1. x_pixel_scale in
  let y_pixel_scale = Option.default 1. y_pixel_scale in
  let title = Option.default "" title in (* geodata.title *)
  let window_title = Option.default "window_title" window_title in (* state.parameters_file *)

  if Option.is_some region then (
    let max_page_width = 217.  *. 4. in (* for PNG this is pixels *)
    let max_page_height = 292. *. 4. in
    let region = Option.get region in
    let width  = region.(2)-.region.(0) in
    let height = region.(3)-.region.(1) in
    Owl_plot.set_xrange plot (region.(0)*.x_pixel_scale) (region.(2)*.x_pixel_scale);
    Owl_plot.set_yrange plot (region.(1)*.y_pixel_scale) (region.(3)*.y_pixel_scale);
    let width_scale  = max_page_width /. width in
    let height_scale = max_page_height /. height in
    let scale = min width_scale height_scale in
    let page_width  = scale *. width in
    let page_height = scale *. height in
    Owl_plot.set_page_size plot (int_of_float page_width) (int_of_float page_height); (* seems to be landscape *)
  );
  Owl_plot.remove_axes plot; (* the box is drawn, but no axes *)
  Owl_plot.set_title plot "";
  Owl_plot.set_xlabel plot "";
  Owl_plot.set_ylabel plot "";
  let f _ =
    Plplot.plscolbg 255 255 255;
    Plplot.plscol0 15 0 0 0; (* black foreground in default palette *)
    Plplot.plcol0 15; (* black foreground in default palette *)
    Plplot.plschr 0. 0.6;
    Plplot.plbox "bcnsti" 0. 0 "bcnsti" 0. 0 ;
    Plplot.plschr 0. 1.0;
    Plplot.pllab  "" "" title; (* x and y labels are blank *)
  in
  Owl_plot.add_plplot ~h:plot f

(*f plot_updownstreamlines_overlay
        Up or downstreamlines 
 *)
let plot_updownstreamlines_overlay ?do_down:(do_down=true) t data results =
    let marker = t.props.streamline_point_marker in
    let size   = t.props.streamline_point_size in
    let alpha  = t.props.streamline_point_alpha in
    let marker = Owl_plot.plsym_of_marker marker in
    let (color, streamline_array, up_or_down_str) = (
        if do_down then
          (t.props.downstreamline_color, results.streamline_arrays.(0), "down")
        else
          (t.props.upstreamline_color, results.streamline_arrays.(1), "up")
      )
    in
    Owl_stats_prng.self_init ();
    Owl_stats_prng.sfmt_seed t.props.shuffle_random_seed;
    let max_streamlines = Array.length streamline_array in
    let streamline_limit = t.props.plot_streamline_limit in
    let (idx_array, verbose_string) = (
        if (streamline_limit>0) && (streamline_limit<max_streamlines) then (
          let shuffle = Array.init max_streamlines (fun i->i) in
          let sample = Owl_stats.sample shuffle streamline_limit in
          (sample,
            sfmt "%d %s streamlines randomly sampled from a set of %d" streamline_limit up_or_down_str max_streamlines)
        ) else (
          (Array.init max_streamlines (fun i->i),
            sfmt "all %d %s streamlines" max_streamlines up_or_down_str)
        )
      )
    in
    pv_verbose t (fun _ -> Printf.printf "Plotting %s\n%!" verbose_string);
    let todo = Array.length idx_array in
    pv_verbose t (fun _ -> Printf.printf "Progress ...%!");

    let print_interval = (
        let blah = todo*(int_of_float data.properties.trace.max_length) in
        if blah>=300000 then (max 1 (todo/100))
        else if blah>=100000 then (max 1 (todo/30))
        else (max 1 (todo/10))
      )
    in
    let show_progress progress =    
      if ((progress+1) mod print_interval)=0 then (
        let prog = 100. *. (float progress) /. (float todo) in
        let prog = truncate (prog +. 0.5) in
        if prog<100 then (
          Printf.printf "%d%%...%!" prog;
        ) else (
          Printf.printf "100%%%!";
        )
      )
    in
    let (r,g,b) = Owl_plot.color_of_name color in Plplot.plscol0 3 r g b;
    let plot_trajectory i sidx =
      let trajectory = streamline_array.(sidx) in (* bytes *)
      let l = (Bytes.length trajectory)/2+1 in
      let x_origin = (ba_owl1d data.x_roi_n_pixel_centers).{0} +. (ba_owl2d data.seeds).{sidx,0} in
      let y_origin = (ba_owl1d data.y_roi_n_pixel_centers).{0} +. (ba_owl2d data.seeds).{sidx,1} in
      let x_origin = (ba_owl2d data.seeds).{sidx,0} in
      let y_origin = (ba_owl2d data.seeds).{sidx,1} in
      let x_vec = Array.create_float l in
      let y_vec = Array.create_float l in
      x_vec.(0) <- x_origin;
      y_vec.(0) <- y_origin;
      let resolution = float data.properties.trace.trajectory_resolution in
      let float_of_byte b = float ((255 land (128+(int_of_char b)))-128) in
      let blah i _ =
       if i>0 then (
        x_vec.(i) <- x_vec.(i-1) +. (float_of_byte (Bytes.get trajectory (i*2-2))) /. resolution;
        y_vec.(i) <- y_vec.(i-1) +. (float_of_byte (Bytes.get trajectory (i*2-1))) /. resolution;
       )
      in
      Array.iteri blah x_vec;
      Plplot.plcol0 3;
      Plplot.plline x_vec y_vec;
      Plplot.plsym x_vec y_vec marker;
      (* using  marker color (from string) marker_size=size line_width=size alpha=alpha *)
     pv_verbose t (fun _ -> show_progress i)
    in
    Array.iteri plot_trajectory idx_array;
    pv_verbose t (fun _ -> Printf.printf "done\n%!");
    ()

(*f [plot_dtm_shaded_relief t data geodata] - plot shaded relief of DTM
 *)
let plot_dtm_shaded_relief t ?subsample:(subsample=5) data (geodata:Geodata.t_data) =
    let open Geodata in
    let dtm_array = geodata.g.dtm_array in
    let (height, width) = ODM.shape dtm_array in
    let region_array = [| 0.; 0.; float width; float height; |] in
    let region = [ [0;-1;subsample];[0;-1;subsample];] in (* subsample in each direction by 5 *)
    new_figure ~title:data.properties.geodata.title ~x_pixel_scale:2. ~y_pixel_scale:2. ~region:region_array t;
    let shaded = generate_hillshade dtm_array 135. 25. in
    let z = matrix_for_plot_float32 ~region:region shaded in
    Owl_plot.(gjs_image ~h:t.plot ~plot_options:[set_cmap ~num_col:256 cmap_grey] z)

(*f [plot_roi_shaded_relief_overlay t data] - plot shaded relief of ROI with alphas that let other plots also happen
 *)
let plot_roi_shaded_relief_overlay ?do_plot_color_relief ?color_alpha ?hillshade_alpha t data =
    let color_alpha     = Option.default 1.  color_alpha in
    let hillshade_alpha = Option.default 0.5 hillshade_alpha in
    let do_plot_color_relief = Option.default t.props.do_plot_color_shaded_relief do_plot_color_relief in
    if do_plot_color_relief then (
      let masked_array  = generate_masked_roi data.roi_array data.basin_mask_array in
      let z = matrix_for_plot_float32 ~region:t.region masked_array in
      Owl_plot.(gjs_image ~h:t.plot ~plot_options:[set_cmap ~num_col:256 ~masked:0.01 ~alpha:color_alpha cmap_terrain] z)
    );
    let shaded = generate_hillshade data.roi_array 135. 25. in
    let z = matrix_for_plot_float32 ~region:t.region shaded in
    Owl_plot.(gjs_image ~h:t.plot ~plot_options:[set_cmap ~num_col:256 ~alpha:hillshade_alpha cmap_grey] z)

(*f [plot_roi_shaded_relief t data] - plot shaded relief of ROI as figure
 *)
let plot_roi_shaded_relief t data =
    let region_array = data.roi_region in
    new_figure ~title:data.properties.geodata.title ~x_pixel_scale:data.roi_pixel_size ~y_pixel_scale:data.roi_pixel_size ~region:region_array t;
    plot_roi_shaded_relief_overlay
      ~hillshade_alpha:t.props.shaded_relief_hillshade_alpha
      ~color_alpha:t.props.shaded_relief_color_alpha
      t data

(*f [plot_streamlines t data results]
        Streamlines, points on semi-transparent shaded relief
 *)
let plot_streamlines t data results = 
    let region_array = data.roi_region in
    new_figure ~title:data.properties.geodata.title ~x_pixel_scale:data.roi_pixel_size ~y_pixel_scale:data.roi_pixel_size ~region:region_array t;
    plot_roi_shaded_relief_overlay
      ~hillshade_alpha:t.props.streamline_shaded_relief_hillshade_alpha
      ~color_alpha:t.props.streamline_shaded_relief_color_alpha
      t data;
    let f _ =
      (*    if t.props.do_plot_flow_vectors then plot_gradient_vector_field_overlay(axes);*)
      if t.props.do_plot_downstreamlines then plot_updownstreamlines_overlay ~do_down:true  t data results ;
      if t.props.do_plot_upstreamlines   then plot_updownstreamlines_overlay ~do_down:false t data results ;
      (*    if t.props.do_plot_seed_points then plot_seed_points_overlay(axes);
    if t.props.do_plot_blockages then plot_blockages_overlay(axes);
    if t.props.do_plot_loops then plot_loops_overlay(axes);
       *)
    ()
    in
    Owl_plot.add_plplot ~h:t.plot f;
    ()

(*f [plot_maps t data]

 Plot maps of DTM ROI streamlines and processed grids

 *)
let plot_maps t data geodata results =
  let w=workflow_start "plotting maps..." t.props.verbosity in
  if t.props.do_plot_dtm then plot_dtm_shaded_relief t data geodata;
  if t.props.do_plot_roi then plot_roi_shaded_relief t data;
  if t.props.do_plot_streamlines then plot_streamlines t data results;
(*  if t.props.do_plot_flow_maps then plot_flow_maps t data;
  if t.props.do_plot_segments then plot_segments t data;
  if t.props.do_plot_channels then plot_channels t data;
  if t.props.do_plot_hillslope_lengths then plot_hillslope_lengths t data;
  if t.props.do_plot_hillslope_lengths_contoured then plot_hillslope_lengths_contoured t data;
  if t.props.do_plot_hillslope_distributions then plot_hillslope_distributions t data;
 *)
  workflow_end w;
  ()

(*f [plot_distributions t data] blah *)
let plot_distributions t data =
    ()

(*f [create props] - initialize the library *)
let create (props:t_props) =
  let region = [[0;-1]; [0;-1]] in
  let filename = "plot_%n.png" in
  let sw = 1 in
  let sh = 1 in
  Plplot.plsfam 1 0 (10*1000*1000);
  {
    subplots = (sw,sh);
    current_sub = 0;
    filename;
    region;
    plot = Owl_plot.create ~m:sw ~n:sh filename;
    props = props.plot;
  }

(*f [process t data] - do all the plots *)
let process t data geodata results = 
    let w = workflow_start "plot" t.props.verbosity in
    if t.props.do_plot_maps then plot_maps t data geodata results;
    if t.props.do_plot_distributions then plot_distributions t data;
    Owl_plot.output t.plot;
    workflow_end w;
    ()
        
(* Old
  let region = [[0;-1;5]; [0;-1;5]] in
  let plots = [ ("plot_0.png", [| [| Plot.plotvect ~region data.u_array data.v_array;
                                     Plot.plot_roi_contours ~region data |];
                                  [|Plot.plot_roi_image ~region data;
                                    Plot.plot_basin_mask ~region data|];
                               |]);
              ] in
  let plots = [ ("plot_0.png", [| [|Plot.plot_roi_image ~region data|];
                               |]);
              ] in
  Plot.do_plots plots;
 *)
