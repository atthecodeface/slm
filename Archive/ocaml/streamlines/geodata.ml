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
 * @file   geodata.ml
 * @brief  Module to provide loading of Geotiff data and filling out the core data
 *
 * Up to date with python of git CS 189bfccdabc3371eafe8bcafa3bdfa8c241e56e4
 *
 * v}
 *)

(*a Module abbreviations *)
open Globals
open Core
open Properties
module Option = Batteries.Option
module ODM = Owl.Dense.Matrix.Generic
module ODN = Owl.Dense.Ndarray.Generic

(** {1 Types}

  These types are used by the main code and the Geotiff module. They
  should not really be used by code outside of this file; the global
  t_core_data from the Core module contains the data that should be
  used for later analysis. However, the DTM array is, for example,
  only held by the Geodata module, so to plot that the Plot module
  currently picks inside t_geodata.

 *)

(**  t_geotiff

  Structure used by the Geotiff module, abstracting much of the Gdal
  handling from the main module.

 *)
type t_geotiff = {
    gtf_filename : string;
    ds : Gdal.Data_set.t;
    trans: Gdal.Geo_transform.t;
    num_bands : int;
    width : int;
    height : int;
    projection : string;
    orig_x : float;
    orig_y : float;
    pixsz_x : float;
    pixsz_y : float;
    rot_x : float;
    rot_y : float;
  }

(**  t_geodata

  The data structure that is filled out by a Geodata file

 *)
type t_geodata = {
    dtm_array              : t_ba_floats; (* w * h of float32 with NAN for no_data_value *)
    x_easting_bottomleft   : float;
    y_northing_bottomleft  : float;
    roi_x_origin           : float; (* =x_roi_n_pixel_centers[0] *)
    roi_y_origin           : float; (* =y_roi_n_pixel_centers[0] *)
    roi_width              : float; (* width-1.0 *)
    roi_height             : float; (* height-1.0 *)
    roi_dx                 : float; (* 1.0; nothing else supported as yet *)
    roi_dy                 : float; (* 1.0; nothing else supported as yet *)
  }

(**  t_data

  Main data structure for the module, including the data from a
  Geodata file (in t_geodata) and the properties for the Geodata
  workflow.

 *)
type t_data = {
    props : t_props_geodata;
    mutable g : t_geodata;
  }

(**  geodata_dummy

  A t_geodata structure that is effectively empty, so that it may be
  used prior to the reading of an actual Geodata file.

 *)
let geodata_dummy = {
    dtm_array              =  ba_float2d 1 1;
    x_easting_bottomleft   = 0. ;
    y_northing_bottomleft  = 0. ;
    roi_x_origin           = 0. ;
    roi_y_origin           = 0. ;
    roi_width              = 0. ;
    roi_height             = 0. ;
    roi_dx                 = 0. ;
    roi_dy                 = 0. ;
  }

(** {1 pv_verbosity functions} *)

(**  [pv_noisy t]

  Shortcut to use {!type:Geodata.t_data} verbosity for {!val:Properties.pv_noisy}

 *)
let pv_noisy   t = Workflow.pv_noisy t.props.workflow

(**  [pv_debug t]

  Shortcut to use {!type:Geodata.t_data} verbosity for {!val:Properties.pv_debug}

 *)
let pv_debug   t = Workflow.pv_noisy t.props.workflow

(**  [pv_info t]

  Shortcut to use {!type:Geodata.t_data} verbosity for {!val:Properties.pv_info}

 *)
let pv_info    t = Workflow.pv_info t.props.workflow

(**  [pv_verbose t]

  Shortcut to use {!type:Geodata.t_data} verbosity for {!val:Properties.pv_verbose}

 *)
let pv_verbose t = Workflow.pv_verbose t.props.workflow

(** {1 Geotiff submodule} *)

module Geotiff =
struct

  (** [Geotiff of string] exception

    Raised if the Geotiff file is not supported, with a reason string

  *)
  exception Geotiff of string

  (**  [read_header filename]

    Reads a Geotiff file header and prepares for reading data from bands

  *)
  let read_header filename =
    let ds        = Gdal.Data_set.of_source_exn filename in
    let num_bands = Gdal.Data_set.get_count ds in
    let trans     = Gdal.Geo_transform.get ds in
    let projection    = Gdal.Data_set.get_projection ds in
    let width      = Gdal.Data_set.get_x_size ds in
    let height     = Gdal.Data_set.get_y_size ds in
    let (orig_x, orig_y)   = Gdal.Geo_transform.get_origin trans in
    let (pixsz_x, pixsz_y) = Gdal.Geo_transform.get_pixel_size trans in
    let (rot_x, rot_y)     = Gdal.Geo_transform.get_rotation trans in
    {
      gtf_filename=filename;
      ds;
      trans;
      num_bands;
      width;
      height;
      projection;
      orig_x;
      orig_y;
      pixsz_x;
      pixsz_y;
      rot_x;
      rot_y;
    }

  (**  [show_data_band_type geo n]

    Print the data band type for debug

  *)
  let show_data_band_type geo n =
    match Gdal.Data_set.get_band_data_type geo.ds n with
    | `byte -> Printf.printf "byte\n";
    | `int16 -> Printf.printf "int16\n";
    | `uint16 -> Printf.printf "uint16\n";
    | `int32 -> Printf.printf "int32\n";
    | `uint32 -> Printf.printf "uint32\n";
    | `float32 -> Printf.printf "float32\n";
    | `float64 -> Printf.printf "float64\n";
    | _ -> Printf.printf "**Unhandled******************************************************************************\n"

  (**  [read_data_band geo data n]

   Read the data band n for DTM data

   *)
  let read_data_band geo data n =
    let data_band = Gdal.Data_set.get_band geo.ds n Gdal.Band.Data.Float32 in
    (*
    let (data_width, data_height) = Gdal.Band.get_size data_band in
    Printf.printf "%d,%d : %d,%d\n" data_width data_height geo.width geo.height;
     *)
    let no_data_value = Option.default nan (Gdal.Band.get_no_data_value data_band) in
    Gdal.Band.iter_read data_band (fun x y v -> ODM.set data.dtm_array x y  (if v=no_data_value then nan else v););
    ODN.(copy_to (flip ~axis:1 data.dtm_array) data.dtm_array)

  (**  [map_data_band geo n band_type f]

     Read the data band n for masked data (Uint16)

  *)
  let map_data_band geo n band_type f =
    let data_band = Gdal.Data_set.get_band geo.ds n band_type in
    (*
    let (data_width, data_height) = Gdal.Band.get_size data_band in
    Printf.printf "%d,%d : %d,%d\n" data_width data_height geo.width geo.height;
     *)
    Gdal.Band.iter_read data_band f

  (**  [str ?indent t]

    Return a string of the Geotiff structure using the specified indent

    @return a human-readable string

   *)
  let str ?indent:(indent="  ") t =
    let r = sfmt "Geotiff '%s'\n" t.gtf_filename in
    let r = r ^ (sfmt "%s%d x %d samples from %f,%f with size %g,%g\n" indent t.width t.height t.orig_x t.orig_y t.pixsz_x t.pixsz_y) in
    let r = r ^ (sfmt "%susing %d bands\n" indent t.num_bands) in
    r

  (**  [check_supported t]

     Checks that the Geotiff file is supported by the code

   *)
  let check_supported t =
    if (t.num_bands!=1) then
      raise (Geotiff (sfmt "Only a single band in GeoTiff is supoprted in '%s'" t.gtf_filename));
    if (abs_float (t.pixsz_x +. t.pixsz_y))>1E-3 then
      raise (Geotiff (sfmt "Pixel x=%g and y=%g dimensions not equal in '%s': cannot handle non-square pixels" t.pixsz_x t.pixsz_y t.gtf_filename));
    ()

  (*f  All done *)

end

(** {1 Top level Geodata module functions}
  *)

(**  [Geodata of string] exception

  Raised if the ROI is out of bounds

 *)
exception Geodata of string

(**  [update_properties t geo]

  Update the properties based on the Geotiff file

  Raise an exception if the ROI is not sensible

 *)
let update_properties t geo =
  if t.props.roi_x_bounds.(0)=min_int then t.props.roi_x_bounds.(0)<-0;
  if t.props.roi_x_bounds.(1)=min_int then t.props.roi_x_bounds.(1)<-geo.width;
  if t.props.roi_y_bounds.(0)=min_int then t.props.roi_y_bounds.(0)<-0;
  if t.props.roi_y_bounds.(1)=min_int then t.props.roi_y_bounds.(1)<-geo.height;
  if ((t.props.roi_x_bounds.(1)<=t.props.roi_x_bounds.(0)) ||
      (t.props.roi_y_bounds.(1)<=t.props.roi_y_bounds.(0)) ||
      (t.props.roi_x_bounds.(0)<0) ||
      (t.props.roi_y_bounds.(0)<0) ||
      (t.props.roi_x_bounds.(1)>geo.width) ||
      (t.props.roi_y_bounds.(1)>geo.height)) then
     raise (Geodata (sfmt "ROI out of bounds (%d,%d) to (%d,%d) out of (0,0) (%d,%d)" t.props.roi_x_bounds.(0) t.props.roi_y_bounds.(0) t.props.roi_x_bounds.(1) t.props.roi_y_bounds.(1) geo.width geo.height));
  pv_verbose t (fun _ -> Printf.printf "ROI size %d,%d\n" t.props.roi_x_bounds.(0)  t.props.roi_x_bounds.(1));
  ()

(**  [fill_data t data geo ]

  Fill out the Core data and t_data structures based on the Geotiff file header.

  This does not load the DTM data nor fill our the ROI data

 *)
let fill_data t data geo =
  Core.set_roi data [|t.props.roi_x_bounds.(0); t.props.roi_y_bounds.(0);
                      t.props.roi_x_bounds.(1); t.props.roi_y_bounds.(1)|];

  data.roi_pixel_size <- geo.pixsz_x;
  data.pad_width <- t.props.pad_width;

  let roi_dx  = 1.0 in
  let roi_dy  = 1.0 in
  ODN.iteri (fun i _ -> ODN.set data.x_roi_n_pixel_centers [|i|] (((float (i+t.props.roi_x_bounds.(0))) +. 0.5) *. roi_dx)) data.x_roi_n_pixel_centers;
  ODN.iteri (fun i _ -> ODN.set data.y_roi_n_pixel_centers [|i|] (((float (i+t.props.roi_y_bounds.(0))) +. 0.5) *. roi_dy)) data.y_roi_n_pixel_centers;

  let dtm_array = ba_float2d geo.width geo.height in
  ODN.fill dtm_array nan;
  let roi_x_origin = ODN.get data.x_roi_n_pixel_centers [|0|] in
  let roi_y_origin = ODN.get data.y_roi_n_pixel_centers [|0|] in
  let roi_width  = ((float data.roi_nx) -. 1.) *. roi_dx in
  let roi_height = ((float data.roi_ny) -. 1.) *. roi_dy in
  let x_easting_bottomleft = geo.orig_x in
  let y_northing_bottomleft = geo.orig_y +. (float geo.height) *. geo.pixsz_y in

  let geodata = {
    dtm_array;
    roi_x_origin;
    roi_y_origin;
    roi_width;
    roi_height;
    roi_dx;
    roi_dy;
    x_easting_bottomleft;
    y_northing_bottomleft;
    }
  in
  t.g <- geodata

(**  [read_dtm_file t]

  Read a DTM file - fills out the data and geodata based on the header
  of the file, then reads the DTM data and copies the ROI array.

  @return geotiff file, in case that is useful for higher levels

 *)
let read_dtm_file t data = 
  let filename = t.props.filename in
  let geotiff = Geotiff.read_header filename in
  pv_verbose t (fun _ -> Printf.printf "%s" (Geotiff.str geotiff));
  Geotiff.check_supported geotiff;
  update_properties t geotiff;
  fill_data t data geotiff;
  Geotiff.read_data_band geotiff t.g 1;
  let src_area = ODN.area t.props.roi_x_bounds.(0) t.props.roi_y_bounds.(0) (t.props.roi_x_bounds.(1)-1) (t.props.roi_y_bounds.(1)-1) in
  let dst_area = ODN.area 0 0 (data.roi_nx-1) (data.roi_ny-1) in
  ODN.copy_area_to t.g.dtm_array src_area data.roi_array dst_area;
  geotiff

(**  [read_basin t data]

  Read the basin masking file, fleshing out the data basin masks

 *)
let read_basin t data =
  let filename = filename_from_path t.props.dtm_path t.props.basins_file in
  let geo = Geotiff.read_header filename in
  Geotiff.check_supported geo; (* seems to be what the original does *)
  let basin_array = ba_char2d data.roi_nx data.roi_ny in
  let basin_data x y v =
    let y = geo.height-1-y in
    if (x<t.props.roi_x_bounds.(1) && y<t.props.roi_y_bounds.(1) && x>=t.props.roi_x_bounds.(0) && y>=t.props.roi_y_bounds.(0)) then (
      ODM.set basin_array (x-t.props.roi_x_bounds.(0)) (y-t.props.roi_y_bounds.(0)) (if List.mem v t.props.basins then '\000' else '\255' )
    )
  in
  Geotiff.map_data_band geo 1 Gdal.Band.Data.UInt16 basin_data;
  ODM.copy_to basin_array data.basin_mask_array;
  ODM.copy_to basin_array data.basin_fatmask_array;
  ()

(**  [pad_basins ~clear t data]

  Pads and clears the basin masks as required

  Also masks out any DTM data that is NaN in the ROI array

 *)
let pad_basins ~clear t data =
  if clear then (
    ODN.fill data.basin_mask_array    '\000'; (* Use it all *)
    ODN.fill data.basin_fatmask_array '\000'; (* Use it all *)
  );
  data.basin_mask_array    <- get_padded_array data.basin_mask_array    t.props.pad_width '\255';
  data.basin_fatmask_array <- get_padded_array data.basin_fatmask_array t.props.pad_width '\255';
  let bma' = ODN.flatten data.basin_mask_array    |> Bigarray.array1_of_genarray in
  let bfa' = ODN.flatten data.basin_fatmask_array |> Bigarray.array1_of_genarray in
  let mask_if_nan i v = if is_nan v then (bma'.{i}<-'\255'; bfa'.{i}<-'\255';) in
  ODN.iteri mask_if_nan data.roi_array;
  ()

(**  [display t]

  Display the t_data structure for verbose operation

 *)
let display t data geotiff =
    Printf.printf "**Geodata begin**\n";
    Printf.printf "Reading DTM from GeoTIFF file \"%s\"\n" geotiff.gtf_filename;
    Printf.printf "DTM GeotTiff coordinate \"geotransform\": (%f, %f, %f, %f, %f, %f)\n" geotiff.orig_x geotiff.pixsz_x geotiff.rot_x geotiff.orig_y geotiff.rot_y geotiff.pixsz_y;
    Printf.printf "DTM size: %d x %d = %d pixels\n" geotiff.width geotiff.height (geotiff.width*geotiff.height);
    Printf.printf "DTM pixel size: %fm\n" data.roi_pixel_size;
    Printf.printf "DTM origin:\n";
    Printf.printf "  - bottom-left pixel center: [%0.2fmE, %0.2fmN]\n"
                    t.g.x_easting_bottomleft
                    t.g.y_northing_bottomleft;
    Printf.printf "  - bottom-left pixel corner: [%0.2fmE, %0.2fmN]\n"
                    (t.g.x_easting_bottomleft -. 0.5 *. data.roi_pixel_size)
                    (t.g.y_northing_bottomleft -. 0.5 *. data.roi_pixel_size);
    Printf.printf "ROI pixel bounds:  [[%d, %d], [%d, %d]]\n"
                    t.props.roi_x_bounds.(0)
                    (t.props.roi_x_bounds.(1)-1)
                    t.props.roi_y_bounds.(0)
                    (t.props.roi_y_bounds.(1)-1);
    Printf.printf "ROI pixel grid:  %d x %d = %d pixels\n"
                    data.roi_nx
                    data.roi_ny
                    (data.roi_ny*data.roi_nx);
    Printf.printf "ROI pixel-edge boundaries (assuming pixel-as-area)\n";
    Printf.printf "  - in pixel units: [x: %f<=>%f] , [y: %f<=>%f]\n"
                    (t.g.roi_x_origin -. t.g.roi_dx /. 2.)
                    (t.g.roi_x_origin +. t.g.roi_width +. t.g.roi_dx /. 2.)
                    (t.g.roi_y_origin -. t.g.roi_dy /. 2.)
                    (t.g.roi_y_origin +. t.g.roi_height +. t.g.roi_dy /. 2.);
    Printf.printf "  - in meters:      [x: %f<=>%f] , [y: %f<=>%f]\n"
                    (data.roi_pixel_size *. (t.g.roi_x_origin -. t.g.roi_dx /. 2.))
                    (data.roi_pixel_size *. (t.g.roi_x_origin +. t.g.roi_width +. t.g.roi_dx /. 2.))
                    (data.roi_pixel_size *. (t.g.roi_y_origin -. t.g.roi_dy /. 2.))
                    (data.roi_pixel_size *. (t.g.roi_y_origin +. t.g.roi_height +. t.g.roi_dy /. 2.));
    Printf.printf "**Geodata end**\n";
    ()

(**  [process t]

  Run the Geodata workflow - loading the Geodata file and generating
  masks, filling out the core data, and so on.

  Load the required DTM file as a Geodata.t given basic properties

  Read basin mask file if required

  Pad masks as necessary

 *)
let process t data =
    Workflow.workflow_start t.props.workflow;
    let geotiff = read_dtm_file t data in
    if t.props.do_basin_masking then (
      read_basin t data;
      pad_basins ~clear:false t data
    ) else (
      pad_basins ~clear:true t data
    );
    Workflow.workflow_end t.props.workflow;
    pv_verbose t (fun _ -> display t data geotiff);
    (data, geotiff)

(**  [create props]

  Create the t_data structure, and initialize the Gdal library

 *)
let create props =
    Gdal.Lib.init_dynamic ~lib:"libgdal.dylib" ();
    Gdal.Lib.register_all ();
    { props=props.geodata;
      g=geodata_dummy;
    }
