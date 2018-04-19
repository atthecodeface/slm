(*
DYLD_LIBRARY_PATH=/Users/gavinprivate/.opam/system/lib/stubslibs/ utop
#require "owl_top";;

#require "Owl";;
#require "Gdal";;
#require "Imagelib";;
#require "Yojson";;
#require "Batteries";;
#directory "/Users/gavinprivate/Git/streamlines_ocaml/_build/default/src/streamlines";;
#load_rec "streamlines.cmo";;
Streamlines.go ();;

LD_LIBRARY_PATH=/Users/gavinprivate/Git/brew/lib ocaml


To do


analysis.py
connect.py
countlink.py
export.py
label.py
lengths.py
mapping.py
plot.py
segment.py
state.py ???

Code structure

 Globals
    |
Properties
    |
  Core
    |
    +------------+-----------+
    |            |           |
  Pocl      Preprocess    Geodata
    |            |           |
    |            +-----------+
    |                        |               
    |                        |               
    +------------+           |               
    |            |           |               
Integration   CountLink      |
    |                        |               
 Trace                       |
    +------------------------+
    |
   Plot  
    |
Streamlines


The main data structure for the global vectors and ROI DTM are held in the 'Core' data structure.
This also keeps the properties tree, which is read and filled in by 'Properties'.

The data is filled out by 'Geodata', which uses its properties to read in a Geotiff file, filling
out the ROI DTM, and padding as appropriate, producing also the basin masks.

The 'Preprocess' workflow is a pure Ocaml stage that takes the ROI DTM and masks and produces
the U/V array vector field upon which the rest of the work is done.

'Pocl' is a library that provides the access to the OpenCL subsystem, enabling programs and hence
kernels to be compiled and invoked.


 *)
open Globals
open Core
module ODM = Owl.Dense.Matrix.Generic
module ODN = Owl.Dense.Ndarray.Generic

let go root_dir json_dir =
  (* Create workflow objects and fill out properties *)
  set_root root_dir;
  let props = Properties.read_properties [ (["python";"streamlines"], "defaults.json");
                                           ([json_dir], "GuadalupeDemo1.json");
                ] in
  let data        = Core.create props in
  let pocl        = Pocl.create props in
  let geodata     = Geodata.create props in
  let preprocess  = Preprocess.create props in
  let trace       = Trace.create props in
  let plot        = Plot.create props in

  (* Load the data from file *)
  let _ = Geodata.load geodata data in

  (* Preprocess the data - create vector fields u/v and tidy so that it drains properly *)
  Preprocess.process preprocess data;

  (* Create basic info structure based on loaded data and properties *)
  Pocl.rebuild_info_struct pocl data data.info;

  (* Trace streamlines *)
  let results = Trace.process trace pocl data in

  (* Plot a region *)
  Plot.process plot data geodata results;

  (* All done *)
  ()
