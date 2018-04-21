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
 * @file   globals.ml
 * @brief  Globally useful functions and statics for the streamlines analysis
 * v}
 *)
(** 

The code is structured as follows:

{[
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
]}

The main data structure for the global vectors and ROI DTM are held in the 'Core' data structure.
This also keeps the properties tree, which is read and filled in by 'Properties'.

The data is filled out by 'Geodata', which uses its properties to read in a Geotiff file, filling
out the ROI DTM, and padding as appropriate, producing also the basin masks.

The 'Preprocess' workflow is a pure Ocaml stage that takes the ROI DTM and masks and produces
the U/V array vector field upon which the rest of the work is done.

'Pocl' is a library that provides the access to the OpenCL subsystem, enabling programs and hence
kernels to be compiled and invoked.

 *)

(* Notes

h1 { font-size:1.8rem ; text-align:center; margin-top:5px; margin-bottom:2px; }
h2 { font-size:1.6rem ; display:block; background-color:#90bdff; text-align:center; margin-top:5px; margin-bottom:2px; border:1px solid black; }
h3 { font-size:1.2rem ; text-align:left; margin-left:2ex; }
pre { background: #F5F5E7; }
.def code { font-weight: bold; color : #00f; }
.def .keyword { color : #f00; }
body { max-width:none;}
div.def + div.doc { margin-left: 3ex; margin-top: 0.15625rem }
div.def table { margin-left: 6ex; }

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

Use h_min in geodata
Create masked ROI array with correct padding on mask


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

 *)

(*a Module aliases
 with these aliases we get documentation in the _doc directory *)
(** {1 Module aliases} *)

(** Globals - types and support functions used throughout *)
module Globals     = Globals

(** Properties - modules for verbosity, workflows and their properties.
 This contains structures common to all workflows
 *)
module Properties  = Properties

(** Core - module to manage the core data structures (and the Info for some reason) *)
module Core        = Core

(** Pocl - OpenCL abstraction layer, as a workflow (but using global properties) *)
module Pocl        = Pocl

(** Geodata - Workflow module that reads in DTM, sets up ROI and mask, filling out the core data *)
module Geodata     = Geodata

(** Preprocess - Workflow module that takes an ROI and generates the vector field, filling blockages and fixing loops if required *)
module Preprocess  = Preprocess

(** Integration - Used only by trace workflow, invokes GPU to trace streamlines and count flows through ROI pixels *)
module Integration = Integration

(** Trace - Workflow module that traces streamlines *)
module Trace       = Trace

(** Plot - Workflow module that produces plots of DTM, ROI, and processed results *)
module Plot        = Plot

(** {1 Function calls} *)

(** set_root [root_dir] - set the root directory for all relative filenames. *)
let set_root = Globals.set_root

(** [str2bool s] - parse a string as a bool: t, true, yes, y, 1 all indicate true; no, n, f, false all indicate false. Everything else forces an exception
*)
let str2bool s =
  let s=String.lowercase_ascii s in 
  if (s="yes") || (s="y") || (s="t") || (s="true") || s="1" then true
  else if (s="no") || (s="n") || (s="f") || (s="false") || s="0" then false
  else raise (Arg.Bad (Globals.sfmt "Boolean cmd line argument expected, but got '%s'" s))

(** [verbosity_of_string s] - get an integer verbosity level from a string; if the string is an integer, then use that, otherwise try it as a 'boolean' using str2bool
*)
let verbosity_of_string s = 
  match int_of_string_opt s with
  | Some value -> value
  | None -> if str2bool s then 1 else 0

(** [parse_arguments _]

    Parse the command line arguments using the system {!module:Arg} module.

    @return parameters filename and command line arguments override
    structure, to be fed to the {!module:Properties} module when
    reading in the properties files.

*)
let parse_arguments _ =
  let executable = Filename.basename Sys.executable_name in
  let my_usage = Printf.sprintf "Usage: %s [OPTION]\nPlots something\nOptions:" executable in
  let verbosity = ref Properties.PV_Quiet in
  let filename = ref "" in
  let set_verbose s = verbosity := Properties.pv_of_int (verbosity_of_string s) in
  let bool_param x = Arg.String (fun s -> x := Some (str2bool s)) in
  let do_reload_state = ref None in
  let do_geodata = ref None in
  let do_preprocess = ref None in
  let do_condition = ref None in
  let do_trace = ref None in
  let do_analysis = ref None in
  let do_mapping = ref None in
  let do_plot = ref None in
  let do_save_state = ref None in
  let do_export = ref None in
  let open Arg in
  let options =
    [ ("--verbose", String set_verbose, "verbose mode");
      ("-v",        String set_verbose, "verbose mode");
      ("--file",    Set_string filename, "import parameters file");
      ("-f",        Set_string filename, "import parameters file");
      ("-r",        (bool_param do_reload_state), "reload previous runtime state from files");
      ("-g",        (bool_param do_geodata),      "read geodata files (DTM, basins)");
      ("-e",        (bool_param do_preprocess),   "perform preprocessing (optionally do conditioning; compute gradients)");
      ("-c",        (bool_param do_condition),    "condition DTRM for best tracing (fix loops and blockages)");
      ("-t",        (bool_param do_trace),        "perform streamline tracing");
      ("-a",        (bool_param do_analysis),     "analyze streamline patterns, distributions");
      ("-m",        (bool_param do_mapping),      "map channels, midlines");
      ("-p",        (bool_param do_plot),         "carry out all plotting set in parameters files");
      ("-s",        (bool_param do_save_state),   "save runtime state to files at completion");
      ("-x",        (bool_param do_export),       "export figures to files");
    ]
  in
  let anon _ = raise (Arg.Bad "no arguments are supported") in
  parse (align options) anon my_usage;
  if !filename = "" then raise (Arg.Bad "parameters filename MUST be supplied (with -f for example)");
  let cmdline_overrides : Properties.t_cmdline_overrides = {
    verbosity = !verbosity;
    do_reload_state = !do_reload_state;
    do_geodata      = !do_geodata;
    do_preprocess   = !do_preprocess;
    do_condition    = !do_condition;
    do_trace        = !do_trace;
    do_analysis     = !do_analysis;
    do_mapping      = !do_mapping;
    do_plot         = !do_plot;
    do_save_state   = !do_save_state;
    do_export       = !do_export;
    }
  in
  (!filename, cmdline_overrides)

(** [process json_dir parameters_filename cmdline_overrides] 

    Read in the properties and set up the workflows, then perform the
    workflow stages required.

*)
let process json_dir parameters_filename cmdline_overrides =
  (* Create workflow objects and fill out properties *)
  let props = Properties.read_properties [ ([json_dir], "defaults.json");
                                           ([json_dir], parameters_filename);
                ] cmdline_overrides in
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
