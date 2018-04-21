(** Documentation goes here?
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

let set_root = Globals.set_root
(** [parse_arguments _]

    Parse the command line arguments using :mod:`argparse`.
    The arguments are assumed to be passed via `_sys.argv[1:]`.

    Return:
        :obj:`argparse.Namespace`:  parsed command line arguments
**)
let str2bool s =
  let s=String.lowercase_ascii s in 
  if (s="yes") || (s="y") || (s="t") || (s="true") || s="1" then true
  else if (s="no") || (s="n") || (s="f") || (s="false") || s="0" then false
  else raise (Arg.Bad "Boolean cmd line argument expected")

let verbosity_of_string s = 
  match int_of_string_opt s with
  | Some value -> value
  | None -> if str2bool s then 1 else 0

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

(** process

    do stuff
**)
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
