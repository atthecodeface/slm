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
 * @file   properties.ml
 * @brief  Workflow and properties libraries
 * v}
 *)

(*a Module abbreviations *)
module Json = Yojson.Basic
open Globals
module Option = Batteries.Option

(** {1 Verbosity}

  Verbosity is a basic part of the properties of a workflow. The
  verbosity can be set to one of many different levels, and the
  properties for a workflow can each have its own verbosity. Functions
  are provided to permit the execution of code dependent on the
  verbosity level.

 *)

(**  [t_verbosity] enumeration for the different levels of verbosity
 *)
type t_verbosity = 
  | PV_Quiet
  | PV_Info
  | PV_Verbose
  | PV_Noisy
  | PV_Debug

(**  [pv_level verbosity] converts the verbosity to an integer level.

  This function is for internal use - it is best to use the
  pv_verbose/pv_debug etc functions to control execution of code
  dependent on the verbosity level.

  This function is used internally to grade verbosity, and hence to
  select the 'more verbose level' verbosity from a pair of verbositys.

  @return an integer corresponding to 0 being quiet to (currently) 4 being noisiest.

 *)
let pv_level verbosity =
  match verbosity with
  | PV_Quiet   -> 0
  | PV_Info    -> 1
  | PV_Verbose -> 2
  | PV_Noisy   -> 3
  | PV_Debug   -> 4

(**  [pv_of_int level] converts an integer level to a verbosity

  This function is used by argument parsing to convert a command line
  verbosity to a level for execution of the code.

  This is the inverse function to {!val:pv_level}

  @return the verbosity level corresponding to the integer

 *)
let pv_of_int n =
  match n with
  | 0 -> PV_Quiet
  | 1 -> PV_Info
  | 2 -> PV_Verbose
  | 3 -> PV_Noisy
  | _ -> PV_Debug

(**  [pv_str verbosity] produces a string corresponding to the verbosity level

  This function is used to show how verbose a workflow is when it starts

  @return a textual representation of the verbosity level

 *)
let pv_str verbosity =
  match verbosity with
  | PV_Quiet   -> "quiet"
  | PV_Info    -> "info"
  | PV_Verbose -> "verbose"
  | PV_Noisy   -> "noisy"
  | PV_Debug   -> "debug"

(**  [pv_verbosity] ?verbose ?debug ?noisy verbosity - upgrade verbosity level

  The verbosity level of a Workflow is the larger of that specified by
  its properties and that for the whole execution; this function
  permits 'upgrading' the verbosity based on the three properties
  'verbose', 'noisy' and 'debug'.

  @return a verbosity level at least as high as {!val:verbosity}, upgraded based on {!val:verbose}, {!val:noisy} and {!val:debug}

 *)
let pv_verbosity ?verbose:(v=false) ?debug:(d=false) ?noisy:(n=false) verbosity =
  let vdn_verbosity = if d then PV_Debug else if n then PV_Noisy else if v then PV_Verbose else PV_Info in
  let final_verbosity_int = max (pv_level vdn_verbosity) (pv_level verbosity) in
  pv_of_int final_verbosity_int

(**  [pv_if verbosity level f] - if {!val:verbosity} exceeds {!val:level} then invoke {[f ()]}

  This is an internal function, for use in pv_info, pv_verbose, etc.

  It invokes the function {!val:f} with a unit argument if the verbosity at least matches level

  *)
let pv_if verbosity level f = if ((pv_level verbosity) >= (pv_level level)) then (f ())

(**  [pv_info pv]

 Predicate function used with {[fun _ -> ...]} to invoke the function
 {i only} if the verbosity level is at least PV_Info. Use, for
 example, as

 {[pv_info verbosity (fun _ -> Printf.printf "Verbosity is at least info")]}.

 *)
let pv_info pv = pv_if pv PV_Info

(**  [pv_verbose pv]

 Predicate function used with {[fun _ -> ...]} to invoke the function
 {i only} if the verbosity level is at least PV_Verbose. Use, for
 example, as

 {[pv_verbose verbosity (fun _ -> Printf.printf "Verbosity is at least verbose")]}.

 *)
let pv_verbose pv = pv_if pv PV_Verbose

(**  [pv_noisy pv]

 Predicate function used with {[fun _ -> ...]} to invoke the function
 {i only} if the verbosity level is at least PV_Noisy. Use, for
 example, as

 {[pv_noisy verbosity (fun _ -> Printf.printf "Verbosity is at least noisy")]}.

 *)
let pv_noisy pv = pv_if pv PV_Noisy

(**  [pv_debug pv]

 Predicate function used with {[fun _ -> ...]} to invoke the function
 {i only} if the verbosity level is at least PV_Debug. Use, for
 example, as

 {[pv_debug verbosity (fun _ -> Printf.printf "Verbosity is at least debug")]}.

 *)
let pv_debug pv = pv_if pv PV_Debug

(** {1 Properties module}
 *)

module Properties =
struct

  (** [t] type structure of {i Properties}, just a list of json currently

    The JSON files are kept in the list 'least relevant' first - hence
    the 'defaults.json' is first on the list.

  *)
  type t = {
      mutable jsons : Json.json list;
    }

  (** [create _] 

    Create an instance of the Properites 't' structure    

  *)
  let create _ =
    { jsons = [];
    }

  (** [read_json t path leaf] 
    
    Add a new JSON file given by path and leaf, whose contents
    override those of earlier JSON files supplied. Any property not
    specified by the new JSON file, though, is fetched from the
    previous files.

  *)
  let read_json t path leaf =
    let json = Json.from_file (Globals.filename_from_path path leaf) in
    t.jsons <- t.jsons @ [json]

  (** [iter_jsons t f]

    Iterate over all the JSONs, applying f to each, with later
    invocations required to override the earlier invocations.

  *)
  let iter_jsons t f = List.iter f t.jsons

  (*f All done *)
end

(** {1 Workflow module}
 *)

module Workflow =
struct

  (** [Bad_property string] exception
   *)
  exception Bad_property of string

  (** [t] structure

    Structure for a workflow, with its name (for selecting its
    properties from the JSON files, and for displaying when it starts
    and stops), JSON files, and evaluated properties. The {i rebuild}
    property is set to force the properties to be reevaluated before
    they are next used - this is therefore set when JSON files are
    added to the workflow, as this requires such a reevaluation.

    The properties themselves are kept in the {i json} element.

   *)
  type t = {
      properties : Properties.t;
      workflow_name : string;
      mutable json: Json.json;
      mutable rebuild_json : bool;
      mutable verbosity : t_verbosity;
    }

  (** [create properties workflow_name]

    Create a workflow with a given name and Properties.t

   *)
  let create properties workflow_name =
    let json = `Null in
    { properties; workflow_name; json; rebuild_json=true; verbosity=PV_Info}

  (** [add_json t json]

    Internal function, used to extract the properties from the
    {!val:workflow_name} member of the JSON, replacing only those
    properties specified by that JSON.

   *)
  let add_json t json =
    let old_json = t.json in
    let old_json = match old_json with | `Null -> `Assoc [] | x -> x in
    let old_keys = Json.Util.keys old_json in
    let new_json = Json.Util.member t.workflow_name json in
    let new_json = match new_json with | `Null -> `Assoc [] | x -> x in
    let new_keys = Json.Util.keys new_json in
    let keys = List.fold_left (fun acc x -> if List.mem x new_keys then acc else x::acc) new_keys old_keys in
    let new_rather_than_old acc k =
      if (List.mem k new_keys) then
        (k, Json.Util.member k new_json) :: acc
      else
        (k, Json.Util.member k old_json) :: acc
    in
    let final_json = `Assoc (List.fold_left new_rather_than_old [] keys) in
    t.json <- final_json

  (** [rebuild t]

    Rebuild the {i json} element by extracting the contents of the
    workflow from the JSON files in {i properties} - but only do the
    rebuild if marked to do so.

   *)
  let rebuild t =
    if t.rebuild_json then (
      t.json <- `Null;
      t.rebuild_json <- false;
      Properties.iter_jsons t.properties (add_json t)
    )

  (** [str_of t ?default member_name]

    Extract {i member_name} from the json for the workflow if present;
    if not, use default. If this fails, then raise Bad_property.

    A string representation of the extracted property is returned. If
    that does not make sense, then, again, Bad_property is raised.

   *)
  let str_of t ?default member_name =
    rebuild t;
    match (
      match Json.Util.member member_name t.json with
      | `String s -> Some s
      | `Null -> default
      | _ -> None
    ) with
    | Some s -> s
    | _ -> raise (Bad_property (sfmt "Expected string of property %s.%s" t.workflow_name member_name))

  (** [bool_of t ?default member_name]

    Extract {i member_name} from the json for the workflow if present;
    if not, use default. If this fails, then raise Bad_property.

    A boolean representation of the extracted property is returned. If
    that does not make sense, then, again, Bad_property is raised.

   *)
  let bool_of t ?override ?default member_name =
    rebuild t;
    if Option.is_some override then (Option.get override) else (
      match (
        match (
          try Json.Util.(to_bool_option (member member_name t.json)) with
          | _ -> None
        ) with
        | None -> default
        | x -> x
      ) with
      | Some s -> s
      | _ -> raise (Bad_property (sfmt "Expected bool of property %s.%s" t.workflow_name member_name))
    )

  (** [float_of t ?default member_name]

    Extract {i member_name} from the json for the workflow if present;
    if not, use default. If this fails, then raise Bad_property.

    A float representation of the extracted property is returned. If
    that does not make sense, then, again, Bad_property is raised.

   *)
  let float_of t ?default member_name =
    rebuild t;
    match (
      match Json.Util.member member_name t.json with
      | `Int i -> Some (float i)
      | `Float f -> Some f
      | `String s -> Some (float_of_string s)
      | `Null -> default
      | _ -> None
    ) with
    | Some s -> s
    | _ -> raise (Bad_property (sfmt "Expected float of property %s.%s" t.workflow_name member_name))

  (** [int_of t ?default member_name]

    Extract {i member_name} from the json for the workflow if present;
    if not, use default. If this fails, then raise Bad_property.

    An integer representation of the extracted property is returned. If
    that does not make sense, then, again, Bad_property is raised.

   *)
  let int_of t ?default member_name =
    rebuild t;
    match (
      match Json.Util.member member_name t.json with
      | `Int i -> Some i
      | `Float f -> Some (int_of_float f)
      | `String s -> Some (int_of_string s)
      | `Null -> default
      | _ -> None
    ) with
    | Some s -> s
    | _ -> raise (Bad_property (sfmt "Expected int of property %s.%s" t.workflow_name member_name))

  (** [int32_of t ?default member_name]

    Extract {i member_name} from the json for the workflow if present;
    if not, use default. If this fails, then raise Bad_property.

    An Int32 representation of the extracted property is returned. If
    that does not make sense, then, again, Bad_property is raised.

   *)
  let int32_of t ?default member_name =
    rebuild t;
    match (
      match Json.Util.member member_name t.json with
      | `Int i -> Some (Int32.of_int i)
      | `Float f -> Some (Int32.of_float f)
      | `String s -> Some (Int32.of_string s)
      | `Null -> default
      | _ -> None
    ) with
    | Some s -> s
    | _ -> raise (Bad_property (sfmt "Expected int32 of property %s.%s" t.workflow_name member_name))

  (** [str_list_of t ?default member_name]

    Extract {i member_name} from the json for the workflow if present;
    if not, use default. If this fails, then raise Bad_property.

    A string list representation of the extracted property is returned. If
    that does not make sense, then, again, Bad_property is raised.

   *)
  let str_list_of ?default t member_name =
    rebuild t;
    match (
      match Json.Util.member member_name t.json with
      | `String s -> Some [s]
      | `List l -> Some (List.map Json.Util.to_string l)
      | `Null -> default
      | _ -> None
    ) with 
    | Some r -> r
    | _ -> raise (Bad_property (sfmt "Expected list of strings for property %s.%s" t.workflow_name member_name))

  (** [int_list_of t ?default member_name]

    Extract {i member_name} from the json for the workflow if present;
    if not, use default. If this fails, then raise Bad_property.

    An int list representation of the extracted property is returned. If
    that does not make sense, then, again, Bad_property is raised.

   *)
  let int_list_of ?default t member_name =
    rebuild t;
    match (
      match Json.Util.member member_name t.json with
      | `Float s -> Some [int_of_float s]
      | `Int s -> Some [s]
      | `List l -> Some (List.map Json.Util.to_int l)
      | `Null -> default
      | _ -> None
    ) with
    | Some r -> r
    | _ -> raise (Bad_property (sfmt "Expected list of strings for property %s.%s" t.workflow_name member_name))

  (** [float_list_of t ?default member_name]

    Extract {i member_name} from the json for the workflow if present;
    if not, use default. If this fails, then raise Bad_property.

    A float list representation of the extracted property is returned. If
    that does not make sense, then, again, Bad_property is raised.

   *)
  let float_list_of ?default t member_name =
    rebuild t;
    match (
      match Json.Util.member member_name t.json with
      | `Float s -> Some [s]
      | `Int   s -> Some [float s]
      | `List l -> Some (List.map Json.Util.to_number l)
      | `Null -> default
      | _ -> None
    ) with
    | Some r -> r
    | _ -> raise (Bad_property (sfmt "Expected list of floats for property %s.%s" t.workflow_name member_name))

  (** [get_verbosity t]

   Get the verbosity

   *)
  let get_verbosity t = t. verbosity

  (** [set_verbosity t verbosity]

   Upgrade verbosity based on 'verbose', 'noisy' and 'debug' properties and set it

   *)
  let set_verbosity t verbosity =
    let verbose = bool_of t ~default:false "verbose" in
    let debug =   bool_of t ~default:false "debug" in
    let noisy =   bool_of t ~default:false "noisy" in
    let verbosity = pv_verbosity ~verbose:verbose ~noisy:noisy ~debug:debug verbosity in
    t.verbosity <- verbosity

  (** [str t]

    Generate a pretty string of the contents of the workflow json, for
    debug usually.

   *)
  let str t =
    rebuild t;
    Json.pretty_to_string t.json

  (** [pretty_print t]

    Pretty print the workflow json to stdout.

   *)
  let pretty_print t =
    Printf.printf "%s\n" (str t)

  (**  [pv_noisy t f] 

    Execute f () if verbosity is at least noisy

  *)
  let pv_noisy   t = pv_noisy t.verbosity

  (**  [pv_debug t]

    Execute f () if verbosity is at least debug

  *)
  let pv_debug   t = pv_debug t.verbosity

  (**  [pv_info t]

    Execute f () if verbosity is at least info

  *)
  let pv_info    t = pv_info t.verbosity

  (**  [pv_verbose t]

    Execute f () if verbosity is at least verbose

  *)
  let pv_verbose t = pv_verbose t.verbosity

  (** [workflow_start t]
    
    Mark start of a workflow, or possibly a subflow of a workflow.

   *)
  let workflow_start ?subflow t =
    let workflow_name = Option.map_default (fun v->sfmt "%s %s" t.workflow_name v) t.workflow_name subflow in
    if (pv_level t.verbosity) > (pv_level PV_Info) then (
      Printf.printf "Start %s (%s)...\n%!" workflow_name (pv_str t.verbosity);
    ) else (
      Printf.printf "Start %s...\n%!" workflow_name;
    )

  (** [workflow_end t]
    
    Mark end of a workflow (or most recent subflow)

   *)
  let workflow_end t =
    Printf.printf "...completed %s\n\n%!" t.workflow_name

  (*f All done *)
end

(** {1 Properties types}
 *)

(**  Geodata exception *)
exception Geodata of string

(**  t_cmdline_overrides

  Structure containing command line settings, which can override the
  properties from the JSON files.

  If the options are 'None' then there is no override; else it is
  'Some <value>', and the value should override that from the JSON
  properties.

  This structure is filled out in streamlines.ml, and used in read_state

 *)
type t_cmdline_overrides = {
    verbosity : t_verbosity;
    do_reload_state :    bool option;
    do_geodata :         bool option;
    do_preprocess :      bool option;
    do_condition :       bool option;
    do_trace :           bool option;
(*    do_postprocess :     bool option;*)
    do_analysis :        bool option;
    do_mapping :         bool option;
    do_plot :            bool option;
    do_save_state :      bool option;
    do_export :          bool option;
  }

(**  t_props_state

  Properties for the 'toplevel', which is specified in the 'state'
  section of the JSON files.

  Some of these can be overridden on the command line

 *)
type t_props_state = {
    verbosity :          t_verbosity;
    verbose :            bool;
    debug :              bool;
    noisy :              bool;
    cl_platform :        int;
    cl_device :          int;
    gpu_memory_limit_pc: float;
    array_order :        string;
    do_geodata :         bool;
    do_preprocess :      bool;
    do_condition :       bool;
    do_trace :           bool;
    do_postprocess :     bool;
    do_analysis :        bool;
    do_mapping :         bool;
    do_plot :            bool;
    do_save_state :      bool;
    do_export :          bool;
    do_rw_savez :        bool;
    do_rw_hdf5 :         bool;
    do_reload_state :    bool;
	}

(**  t_props_pocl

  Properties for the {i pocl}, these are specified in the 'state'
  section of the JSON files.

 *)
type t_props_pocl = {
    cl_platform :        int;
    cl_device :          int;
    gpu_memory_limit_pc: float;
    workflow :           Workflow.t;
  }

(**  t_props_geodata

  Properties for the {i geodata} workflow.

 *)
type t_props_geodata = {
    filename          : string;
    dtm_file          : string;
    dtm_path          : string list;
    no_data_values    : float list;
    pad_width         : int;
    title             : string;
    roi_x_bounds      : int array; (* start, start+width exception if out of bounds *)
    roi_y_bounds      : int array; (* start, start+height exception if out of bounds *)
    basins            : int list;
    basins_file       : string;
    do_basin_masking  : bool;
    workflow          : Workflow.t;
  }

(**  t_props_preprocess

  Properties for the {i preprocess} workflow.

 *)
type t_props_preprocess = {
    do_simple_gradient_vector_field : bool;
    do_normalize_speed              : bool;
    vecsum_threshold                : float;
    divergence_threshold            : float;
    curl_threshold                  : float;
    workflow                        : Workflow.t;
  }

(**  t_props_trace

  Properties for the {i trace} workflow.

 *)
type t_props_trace = {
    integrator_step_factor       : float;

    do_trace_upstream            : bool;
    do_trace_downstream          : bool;

    max_integration_step_error   : float; (* info_struct *)
    max_length                   : float; (* info_struct *)
    integration_halt_threshold   : float; (* info_struct *)
    trajectory_resolution        : int;   (* info_struct *)
    subpixel_seed_point_density  : int;   (* info_struct *)
    jitter_magnitude             : float; (* info_struct *)
    interchannel_max_n_steps     : int;   (* info_struct *)
    segmentation_threshold       : int;   (* info_struct *)
    left_flank_addition          : int32; (* info_struct *)
    workflow                     : Workflow.t;

  }

(**  t_props_analysis

  Properties for the {i analysis} workflow.

 *)
type t_props_analysis = {
    do_marginal_distbn_dsla :                     bool;
    do_marginal_distbn_usla :                     bool;
    do_marginal_distbn_dslt :                     bool;
    do_marginal_distbn_uslt :                     bool;
    do_marginal_distbn_dslc :                     bool;
    do_marginal_distbn_uslc :                     bool;
    (*marginal_distbn_kde_methods :                 string;*)
    marginal_distbn_kde_kernel :                  string;
    marginal_distbn_kde_bw_method :               string;
    marginal_distbn_kde_bandwidth :               float;
    marginal_distbn_kde_nx_samples :              int;
	
	pdf_slt_min: float;
    do_joint_distribn_dsla_dslt :                     bool;
    do_joint_distribn_usla_uslt :                     bool;
    do_joint_distribn_uslt_dslt :                     bool;
    do_joint_distribn_dsla_usla :                     bool;
    do_joint_distribn_dsla_dslc :                     bool;
    do_joint_distribn_usla_uslc :                     bool;
    do_joint_distribn_uslc_dslc :                     bool;

    (* joint_distbn_kde_methods :                     string;*)
    joint_distbn_kde_kernel :                     string;
    joint_distbn_kde_bw_method :                  string;
    joint_distbn_kde_bandwidth :                  float;
    joint_distbn_kde_nxy_samples :                int;
    joint_distbn_y_shear_factor :                 float;
    (* joint_distbn_mode_threshold_list :                     bool;*)
    joint_distbn_mode2_tilt :                     float; (* unused *)
    joint_distbn_mode2_nearness_factor :          float;
    workflow                     : Workflow.t;
  }

(**  t_props_plot

  Properties for the {i plot} workflow.

 *)
type t_props_plot = {
    do_plot_dtm :                              bool;
    do_plot_roi :                              bool;
    do_plot_streamlines :                      bool;
    do_plot_flow_maps :                    bool;
    do_plot_segments :                    bool;
    do_plot_channels :                    bool;
    do_plot_hillslope_lengths :                    bool;
    do_plot_hillslope_lengths_contoured :                    bool;
    do_plot_hillslope_distributions :                    bool;

    do_plot_downstreamlines :                              bool;
    do_plot_upstreamlines :                              bool;
    do_plot_seed_points :                              bool;
    do_plot_flow_vectors :                              bool;
    do_plot_blockages :                              bool;
    do_plot_loops :                              bool;
	
    plot_interpolation_method :                    string;
    plot_streamline_limit :                    int;
	
    do_plot_color_shaded_relief :                    bool;
    do_plot_curvature_roi :                    bool;
    do_plot_colorized_streamlines :                    bool;
    do_plot_merged_streamline_density_bands :                    bool;
	
	
    plot_window_size_factor :                    float;
    plot_window_pdf_size_factor :                float;
    plot_window_width :                          float;
    plot_window_height :                         float;
	
    hillshade_azimuth :                          float;
    hillshade_angle :                            float;
	
    downstreamline_color :                            string;
    upstreamline_color :                              string;
	
    streamline_point_marker :                         string;
    streamline_point_size :                           float;
    streamline_point_alpha :                          float;

    shaded_relief_hillshade_alpha :             float;
    shaded_relief_color_alpha :                 float;
    streamline_shaded_relief_hillshade_alpha :  float;
    streamline_shaded_relief_color_alpha :      float;
    streamline_density_alpha :                  float;
    streamline_density_cmap :                   string;
	
    grid_shaded_relief_hillshade_alpha :        float;
    grid_shaded_relief_color_alpha :            float;
	
    seed_point_marker :                              string;
    seed_point_marker_size :                         float;
    seed_point_marker_color :                        string;
    seed_point_marker_alpha :                        float;
	
    channel_head_marker :                            string;
    (*channel_head_marker_sizes :                    bool; : [10,13],*)
    (*channel_head_marker_colors :                    bool; : ["cyan","darkblue"],*)
    channel_head_marker_alpha :                       float;
    channel_shaded_relief_hillshade_alpha :           float;
	
    gradient_vector_color :                    string;
    gradient_vector_alpha :                    float;
    gradient_vector_scale :                    float;
	
    blockage_marker_size :                      float;
    loops_marker_size :                         float;
	
    classical_streamplot_density :              float;

    terrain_cmap :                              string;
    shuffle_random_seed :                       int;
    random_cmap_seed :                          int;
	
    do_plot_maps :                               bool;
    do_plot_distributions :                      bool;
    do_plot_marginal_pdf_dsla :                    bool;
    do_plot_marginal_pdf_usla :                    bool;
    do_plot_marginal_pdf_dslt :                    bool;
    do_plot_marginal_pdf_uslt :                    bool;
    do_plot_marginal_pdf_dslc :                    bool;
    do_plot_marginal_pdf_uslc :                    bool;
    do_plot_joint_pdf_dsla_usla :                    bool;
    do_plot_joint_pdf_dsla_dslt :                    bool;
    do_plot_joint_pdf_usla_uslt :                    bool;
    do_plot_joint_pdf_uslt_dslt :                    bool;
    do_plot_joint_pdf_dsla_dslc :                    bool;
    do_plot_joint_pdf_usla_uslc :                    bool;
    do_plot_joint_pdf_uslc_dslc :                    bool;
    joint_distbn_n_contours :                    float;
    (*        "joint_distbn_markers" : [["+","crimson",19,4,"w+",17,2,"r.",2,1.0],
        						  ["x","blue",   16,4,"wx",14,2,"k.",2,0.4]],*)
    marginal_distbn_viz_tilt :                    float;
    marginal_distbn_viz_scale :                    float;
    joint_distbn_viz_tilt :                    float;
    joint_distbn_viz_scale :                    float;
    workflow                     : Workflow.t;
  }

(**  t_props

  Structure pulling together the properties for all the workflows

 *)
type t_props = {
    state      : t_props_state;
    pocl       : t_props_pocl;
    geodata    : t_props_geodata;
    preprocess : t_props_preprocess;
    trace      : t_props_trace;
    analysis   : t_props_analysis;
    plot       : t_props_plot;
  }

(** {1 Properties functions}
 *)

(**  [verbosity_from_properties properties verbosity]

 Upgrade verbosity based on 'verbose', 'noisy' and 'debug' properties

 *)
let verbosity_from_properties properties verbosity =
  let verbose =             Workflow.bool_of  properties ~default:false "verbose" in
  let debug =               Workflow.bool_of  properties ~default:false "debug" in
  let noisy =               Workflow.bool_of  properties ~default:false "noisy" in
  pv_verbosity ~verbose:verbose ~noisy:noisy ~debug:debug verbosity

(**  [read_state properties] 

    Read the properties of 'state'

 *)
let read_state properties (cmdline_overrides:t_cmdline_overrides) =
  let properties =          Workflow.create properties "state" in
  let verbose =             Workflow.bool_of  properties ~default:true "verbose" in
  let debug =               Workflow.bool_of  properties ~default:true "debug" in
  let noisy =               Workflow.bool_of  properties ~default:true "noisy" in
  let cl_platform =         Workflow.int_of  properties ~default:0 "cl_platform" in
  let cl_device =           Workflow.int_of  properties ~default:0 "cl_device" in
  let gpu_memory_limit_pc = Workflow.float_of  properties ~default:50. "gpu_memory_limit_pc" in
  let array_order                  = Workflow.str_of      properties ~default:"C"  "array_order" in
  let do_rw_savez =         Workflow.bool_of  properties ~default:true "do_rw_savez" in
  let do_rw_hdf5 =          Workflow.bool_of  properties ~default:true "do_rw_hdf5" in

  let do_reload_state =     Workflow.bool_of  properties ~default:true ?override:cmdline_overrides.do_reload_state  "do_reload_state" in
  let do_geodata =          Workflow.bool_of  properties ~default:true ?override:cmdline_overrides.do_geodata       "do_geodata" in
  let do_preprocess =       Workflow.bool_of  properties ~default:true ?override:cmdline_overrides.do_preprocess    "do_preprocess" in
  let do_condition =        Workflow.bool_of  properties ~default:true ?override:cmdline_overrides.do_condition     "do_condition" in
  let do_trace =            Workflow.bool_of  properties ~default:true ?override:cmdline_overrides.do_trace         "do_trace" in
  let do_postprocess =      Workflow.bool_of  properties ~default:true (*?override:cmdline_overrides.do_postprocess*)   "do_postprocess" in
  let do_analysis =         Workflow.bool_of  properties ~default:true ?override:cmdline_overrides.do_analysis      "do_analysis" in
  let do_mapping =          Workflow.bool_of  properties ~default:true ?override:cmdline_overrides.do_mapping       "do_mapping" in
  let do_plot =             Workflow.bool_of  properties ~default:true ?override:cmdline_overrides.do_plot          "do_plot" in
  let do_save_state =       Workflow.bool_of  properties ~default:true ?override:cmdline_overrides.do_save_state    "do_save_state" in
  let do_export =           Workflow.bool_of  properties ~default:true ?override:cmdline_overrides.do_export        "do_export" in
  let verbosity =           verbosity_from_properties properties PV_Info in

  let props = {
      verbosity;
      verbose;
      debug;
      noisy;
      cl_platform;
      cl_device;
      gpu_memory_limit_pc;
      array_order;
      do_geodata;
      do_preprocess;
      do_condition;
      do_trace;
      do_postprocess;
      do_analysis;
      do_mapping;
      do_plot;
      do_save_state;
      do_export;
      do_rw_savez;
      do_rw_hdf5;
      do_reload_state;
    }
    in (props, verbosity)

(**  [read_pocl verbosity properties] 

    Read the properties of 'pocl' - actually from state - to set it up for workflow

 *)
let read_pocl verbosity properties =
  let workflow =            Workflow.create properties "state" in
  let cl_platform =         Workflow.int_of  workflow ~default:0 "cl_platform" in
  let cl_device =           Workflow.int_of  workflow ~default:0 "cl_device" in
  let gpu_memory_limit_pc = Workflow.float_of  workflow ~default:50. "gpu_memory_limit_pc" in
  Workflow.set_verbosity workflow verbosity;
  {
    cl_platform;
    cl_device;
    gpu_memory_limit_pc;
    workflow;
  }

(**  [geodata_validate_properties props]

 Validate that the properties for the geodata workflow are supported

*)
let geodata_validate_properties props =
  if (Array.length props.roi_x_bounds)<>2 then
    raise (Geodata (sfmt "Expected ROI X bounds to be a pair of integers in '%s'" props.filename));
  if (Array.length props.roi_y_bounds)<>2 then
    raise (Geodata (sfmt "Expected ROI Y bounds to be a pair of integers in '%s'" props.filename));
  ()

(**  [read_geodata verbosity properties]

 Set up properties for the Geodata workflow, given a higher level verbosity and a properties

*)
let read_geodata verbosity properties =
  let workflow         = Workflow.create properties "geodata" in
  let title            = Workflow.str_of workflow ~default:"Title" "title" in
  let pad_width        = Workflow.int_of workflow ~default:0 "pad_width" in
  let dtm_path         = Workflow.str_list_of workflow "data_path" in
  let dtm_file         = Workflow.str_of workflow "dtm_file" in
  let no_data_values   = Workflow.float_list_of workflow "no_data_values" in
  let roi_x_bounds     = Array.of_list (Workflow.int_list_of ~default:[min_int;min_int] workflow "roi_x_bounds") in
  let roi_y_bounds     = Array.of_list (Workflow.int_list_of ~default:[min_int;min_int] workflow "roi_y_bounds") in
  let filename         = filename_from_path dtm_path dtm_file in
  let basins_file      = Workflow.str_of workflow ~default:"" "basins_file" in
  let basins           = Workflow.int_list_of workflow ~default:[] "basins" in
  let do_basin_masking = Workflow.bool_of workflow ~default:false "do_basin_masking" in
  Workflow.set_verbosity workflow verbosity;
  let props = {
    filename;
    dtm_file;
    dtm_path;
    no_data_values;
    roi_x_bounds;
    roi_y_bounds;
    pad_width;
    title;
    basins;
    basins_file;
    do_basin_masking;
    workflow;
  } in
  geodata_validate_properties props;
  props

(**  [read_preprocess verbosity properties]

 Set up properties for the Preprocess workflow, given a higher level verbosity and a properties

*)
let read_preprocess verbosity properties =
  let workflow                        = Workflow.create properties "preprocess" in
  let do_simple_gradient_vector_field = Workflow.bool_of  workflow ~default:true "do_simple_gradient_vector_field" in
  let do_normalize_speed              = Workflow.bool_of  workflow ~default:true "do_normalize_speed" in
  let vecsum_threshold                = Workflow.float_of workflow ~default:0.95 "vecsum_threshold" in
  let divergence_threshold            = Workflow.float_of workflow ~default:(-0.5) "divergence_threshold" in
  let curl_threshold                  = Workflow.float_of workflow ~default:0.0 "curl_threshold" in
  Workflow.set_verbosity workflow verbosity;
  let props =
  {
    do_simple_gradient_vector_field;
    do_normalize_speed;
    vecsum_threshold;
    divergence_threshold;
    curl_threshold;
    workflow;
  } in
  props

(**  [read_trace verbosity properties]

 Set up properties for the Trace workflow, given a higher level verbosity and a properties

*)
let read_trace verbosity properties =
  let workflow = Workflow.create properties "trace" in
  let do_trace_downstream          = Workflow.bool_of     workflow ~default:true "do_trace_downstream" in
  let do_trace_upstream            = Workflow.bool_of     workflow ~default:true "do_trace_upstream" in
  let integrator_step_factor       = Workflow.float_of    workflow ~default:0.5  "integrator_step_factor" in
  let max_integration_step_error   = Workflow.float_of    workflow ~default:0.03 "max_integration_step_error" in
  let max_length                   = Workflow.float_of    workflow ~default:500. "max_length" in
  let integration_halt_threshold   = Workflow.float_of    workflow ~default:0.01 "integration_halt_threshold" in
  let trajectory_resolution        = Workflow.int_of      workflow ~default:128  "trajectory_resolution" in
  let subpixel_seed_point_density  = Workflow.int_of      workflow ~default:3    "subpixel_seed_point_density" in
  let jitter_magnitude             = Workflow.float_of    workflow ~default:2.9  "jitter_magnitude" in
  let interchannel_max_n_steps     = Workflow.int_of      workflow ~default:200  "interchannel_max_n_steps" in
  let segmentation_threshold       = Workflow.int_of      workflow ~default:50   "segmentation_threshold" in
  let left_flank_addition          = Workflow.int32_of    workflow ~default:0l   "left_flank_addition" in
  let max_length = if max_length=0. then infinity else max_length in
  let max_n_steps = int_of_float (max_length /. integrator_step_factor) in
  let interchannel_max_n_steps = if interchannel_max_n_steps=0 then max_n_steps else interchannel_max_n_steps in
  Workflow.set_verbosity workflow verbosity;
  let props =
    {
      do_trace_upstream;
      do_trace_downstream;
      integrator_step_factor;
      max_integration_step_error;
      max_length;
      integration_halt_threshold;
      trajectory_resolution;
      subpixel_seed_point_density;
      jitter_magnitude;
      interchannel_max_n_steps;
      segmentation_threshold;
      left_flank_addition;
      workflow;
    } in
  props

(**  [read_analysis verbosity properties]

 Set up properties for the Analysis workflow, given a higher level verbosity and a properties

*)
let read_analysis verbosity properties =
  let workflow = Workflow.create properties "analysis" in
  let do_marginal_distbn_dsla        = Workflow.bool_of   workflow ~default:false "do_marginal_distbn_dsla" in
  let do_marginal_distbn_usla        = Workflow.bool_of   workflow ~default:false "do_marginal_distbn_usla" in
  let do_marginal_distbn_dslt        = Workflow.bool_of   workflow ~default:false "do_marginal_distbn_dslt" in
  let do_marginal_distbn_uslt        = Workflow.bool_of   workflow ~default:false "do_marginal_distbn_uslt" in
  let do_marginal_distbn_dslc        = Workflow.bool_of   workflow ~default:false "do_marginal_distbn_dslc" in
  let do_marginal_distbn_uslc        = Workflow.bool_of   workflow ~default:false "do_marginal_distbn_uslc" in
  (*marginal_distbn_kde_methods :                 string;*)
  let marginal_distbn_kde_kernel     = Workflow.str_of   workflow ~default:"gaussian" "marginal_distbn_kde_kernel" in
  let marginal_distbn_kde_bw_method  = Workflow.str_of   workflow ~default:"scott" "marginal_distbn_kde_bw_method"in
  let marginal_distbn_kde_bandwidth  = Workflow.float_of   workflow ~default:0.2 "marginal_distbn_kde_bandwidth" in
  let marginal_distbn_kde_nx_samples = Workflow.int_of   workflow ~default:200 "marginal_distbn_kde_nx_samples" in
  
  let pdf_slt_min                    = Workflow.float_of   workflow ~default:0.5 "pdf_slt_min" in
  let do_joint_distribn_dsla_dslt    = Workflow.bool_of   workflow ~default:false "do_joint_distribn_dsla_dslt" in
  let do_joint_distribn_usla_uslt    = Workflow.bool_of   workflow ~default:false "do_joint_distribn_usla_uslt" in
  let do_joint_distribn_uslt_dslt    = Workflow.bool_of   workflow ~default:false "do_joint_distribn_uslt_dslt" in
  let do_joint_distribn_dsla_usla    = Workflow.bool_of   workflow ~default:false "do_joint_distribn_dsla_usla" in
  let do_joint_distribn_dsla_dslc    = Workflow.bool_of   workflow ~default:false "do_joint_distribn_dsla_dslc" in
  let do_joint_distribn_usla_uslc    = Workflow.bool_of   workflow ~default:false "do_joint_distribn_usla_uslc" in
  let do_joint_distribn_uslc_dslc    = Workflow.bool_of   workflow ~default:false "do_joint_distribn_uslc_dslc" in

  (* joint_distbn_kde_methods :                     string;*)
  let joint_distbn_kde_kernel        = Workflow.str_of   workflow ~default:"epanechnikov" "joint_distbn_kde_kernel" in
  let joint_distbn_kde_bw_method     = Workflow.str_of   workflow ~default:"scott" "joint_distbn_kde_bw_method" in
  let joint_distbn_kde_bandwidth     = Workflow.float_of    workflow ~default:0.4 "joint_distbn_kde_bandwidth" in
  let joint_distbn_kde_nxy_samples   = Workflow.int_of      workflow ~default:200 "joint_distbn_kde_nxy_samples" in
  let joint_distbn_y_shear_factor    = Workflow.float_of    workflow ~default:0.0 "joint_distbn_y_shear_factor" in
  (* joint_distbn_mode_threshold_list :                     bool;*)
  let joint_distbn_mode2_tilt        = Workflow.float_of   workflow ~default:3. "joint_distbn_mode2_tilt" in (* unused *)
  let joint_distbn_mode2_nearness_factor = Workflow.float_of   workflow ~default:30. "joint_distbn_mode2_nearness_factor" in

  Workflow.set_verbosity workflow verbosity;
  let props =
    {
      do_marginal_distbn_dsla;
      do_marginal_distbn_usla;
      do_marginal_distbn_dslt;
      do_marginal_distbn_uslt;
      do_marginal_distbn_dslc;
      do_marginal_distbn_uslc;
      marginal_distbn_kde_kernel;
      marginal_distbn_kde_bw_method;
      marginal_distbn_kde_bandwidth;
      marginal_distbn_kde_nx_samples;
      
      pdf_slt_min;
        do_joint_distribn_dsla_dslt;
      do_joint_distribn_usla_uslt;
      do_joint_distribn_uslt_dslt;
      do_joint_distribn_dsla_usla;
      do_joint_distribn_dsla_dslc;
      do_joint_distribn_usla_uslc;
      do_joint_distribn_uslc_dslc;

      joint_distbn_kde_kernel;
      joint_distbn_kde_bw_method;
      joint_distbn_kde_bandwidth;
      joint_distbn_kde_nxy_samples;
      joint_distbn_y_shear_factor;
      joint_distbn_mode2_tilt;
      joint_distbn_mode2_nearness_factor;

      workflow;
    } in
  props

(**  [read_plot verbosity properties]

 Set up properties for the Plot workflow, given a higher level verbosity and a properties

*)
let read_plot verbosity properties =
  let workflow = Workflow.create properties "plot" in
  let do_plot_dtm                         = Workflow.bool_of   workflow ~default:false "do_plot_dtm" in
  let do_plot_roi                         = Workflow.bool_of   workflow ~default:false "do_plot_roi" in
  let do_plot_streamlines                 = Workflow.bool_of   workflow ~default:false "do_plot_streamlines" in
  let do_plot_flow_maps                   = Workflow.bool_of   workflow ~default:false "do_plot_flow_maps" in
  let do_plot_segments                    = Workflow.bool_of   workflow ~default:false "do_plot_segments" in
  let do_plot_channels                    = Workflow.bool_of   workflow ~default:false "do_plot_channels" in
  let do_plot_hillslope_lengths           = Workflow.bool_of   workflow ~default:false "do_plot_hillslope_lengths" in
  let do_plot_hillslope_lengths_contoured = Workflow.bool_of   workflow ~default:false "do_plot_hillslope_lengths_contoured" in
  let do_plot_hillslope_distributions     = Workflow.bool_of   workflow ~default:false "do_plot_hillslope_distributions" in

  let do_plot_downstreamlines          = Workflow.bool_of   workflow ~default:false "do_plot_downstreamlines" in
  let do_plot_upstreamlines            = Workflow.bool_of   workflow ~default:false "do_plot_upstreamlines" in
  let do_plot_seed_points              = Workflow.bool_of   workflow ~default:false "do_plot_seed_points" in
  let do_plot_flow_vectors             = Workflow.bool_of   workflow ~default:false "do_plot_flow_vectors" in
  let do_plot_blockages                = Workflow.bool_of   workflow ~default:false "do_plot_blockages" in
  let do_plot_loops                    = Workflow.bool_of   workflow ~default:false "do_plot_loops" in
  
  let plot_interpolation_method             = Workflow.str_of   workflow ~default:"nearest" "plot_interpolation_method" in
  let plot_streamline_limit                 = Workflow.int_of   workflow ~default:(-1) "plot_streamline_limit" in
  
  let do_plot_color_shaded_relief             = Workflow.bool_of   workflow ~default:false "do_plot_color_shaded_relief" in
  let do_plot_curvature_roi                   = Workflow.bool_of   workflow ~default:false "do_plot_curvature_roi" in
  let do_plot_colorized_streamlines           = Workflow.bool_of   workflow ~default:false "do_plot_colorized_streamlines" in
  let do_plot_merged_streamline_density_bands = Workflow.bool_of   workflow ~default:false "do_plot_merged_streamline_density_bands" in
  
  let plot_window_size_factor                 = Workflow.float_of   workflow ~default:2.7 "plot_window_size_factor" in
  let plot_window_pdf_size_factor             = Workflow.float_of   workflow ~default:2.7 "plot_window_pdf_size_factor" in
  let plot_window_width                       = Workflow.float_of   workflow ~default:3. "plot_window_width" in
  let plot_window_height                      = Workflow.float_of   workflow ~default:3. "plot_window_height" in
  
  let hillshade_azimuth                       = Workflow.float_of   workflow ~default:135. "hillshade_azimuth" in
  let hillshade_angle                         = Workflow.float_of   workflow ~default:25. "hillshade_angle" in
  
  let downstreamline_color                    = Workflow.str_of   workflow ~default:"blue" "downstreamline_color" in
  let upstreamline_color                      = Workflow.str_of   workflow ~default:"red" "upstreamline_color" in
  
  let streamline_point_marker                 = Workflow.str_of   workflow ~default:"-" "streamline_point_marker" in
  let streamline_point_size                   = Workflow.float_of   workflow ~default:0.5 "streamline_point_size" in
  let streamline_point_alpha                  = Workflow.float_of   workflow ~default:0.7 "streamline_point_alpha" in

  let shaded_relief_hillshade_alpha            = Workflow.float_of   workflow ~default:0.3 "shaded_relief_hillshade_alpha" in
  let shaded_relief_color_alpha                = Workflow.float_of   workflow ~default:0.55 "shaded_relief_color_alpha" in
  let streamline_shaded_relief_hillshade_alpha = Workflow.float_of   workflow ~default:1. "streamline_shaded_relief_hillshade_alpha" in
  let streamline_shaded_relief_color_alpha     = Workflow.float_of   workflow ~default:0.3 "streamline_shaded_relief_color_alpha" in
  let streamline_density_alpha                 = Workflow.float_of   workflow ~default:0.5 "streamline_density_alpha" in
  let streamline_density_cmap                  = Workflow.str_of   workflow ~default:"YlGnBu" "streamline_density_cmap" in
  
  let grid_shaded_relief_hillshade_alpha       = Workflow.float_of   workflow ~default:1. "grid_shaded_relief_hillshade_alpha" in
  let grid_shaded_relief_color_alpha           = Workflow.float_of   workflow ~default:0.3 "grid_shaded_relief_color_alpha" in
  
  let seed_point_marker                        = Workflow.str_of   workflow ~default:"H" "seed_point_marker" in
  let seed_point_marker_size                   = Workflow.float_of   workflow ~default:5. "seed_point_marker_size" in
  let seed_point_marker_color                  = Workflow.str_of   workflow ~default:"indigo" "seed_point_marker_color" in
  let seed_point_marker_alpha                  = Workflow.float_of   workflow ~default:0.7 "seed_point_marker_alpha" in
  
  let channel_head_marker                      = Workflow.str_of   workflow ~default:"." "channel_head_marker" in
  (*channel_head_marker_sizes :                    bool; : [10,13],*)
  (*channel_head_marker_colors :                    bool; : ["cyan","darkblue"],*)
  let channel_head_marker_alpha                = Workflow.float_of   workflow ~default:0.8 "channel_head_marker_alpha" in
  let channel_shaded_relief_hillshade_alpha    = Workflow.float_of   workflow ~default:0.5 "channel_shaded_relief_hillshade_alpha" in
  
  let gradient_vector_color                    = Workflow.str_of   workflow ~default:"black" "gradient_vector_color" in
  let gradient_vector_alpha                    = Workflow.float_of   workflow ~default:0.8 "gradient_vector_alpha" in
  let gradient_vector_scale                    = Workflow.float_of   workflow ~default:(-1.) "gradient_vector_scale" in
  
  let blockage_marker_size                     = Workflow.float_of   workflow ~default:20. "blockage_marker_size" in
  let loops_marker_size                        = Workflow.float_of   workflow ~default:20. "loops_marker_size" in
  
  let classical_streamplot_density             = Workflow.float_of   workflow ~default:1. "classical_streamplot_density" in

  let terrain_cmap                             = Workflow.str_of   workflow ~default:"terrain" "terrain_cmap" in
  let shuffle_random_seed                      = Workflow.int_of   workflow ~default:1 "shuffle_random_seed" in
  let random_cmap_seed                         = Workflow.int_of   workflow ~default:1 "random_cmap_seed" in
  
  let do_plot_maps                             = Workflow.bool_of   workflow ~default:true "do_plot_maps" in
  let do_plot_distributions                    = Workflow.bool_of   workflow ~default:false "do_plot_distributions" in
  let do_plot_marginal_pdf_dsla                = Workflow.bool_of   workflow ~default:false "do_plot_marginal_pdf_dsla" in
  let do_plot_marginal_pdf_usla                = Workflow.bool_of   workflow ~default:false "do_plot_marginal_pdf_usla" in
  let do_plot_marginal_pdf_dslt                = Workflow.bool_of   workflow ~default:false "do_plot_marginal_pdf_dslt" in
  let do_plot_marginal_pdf_uslt                = Workflow.bool_of   workflow ~default:false "do_plot_marginal_pdf_uslt" in
  let do_plot_marginal_pdf_dslc                = Workflow.bool_of   workflow ~default:false "do_plot_marginal_pdf_dslc" in
  let do_plot_marginal_pdf_uslc                = Workflow.bool_of   workflow ~default:false "do_plot_marginal_pdf_uslc" in
  let do_plot_joint_pdf_dsla_usla              = Workflow.bool_of   workflow ~default:false "do_plot_joint_pdf_dsla_usla" in
  let do_plot_joint_pdf_dsla_dslt              = Workflow.bool_of   workflow ~default:false "do_plot_joint_pdf_dsla_dslt" in
  let do_plot_joint_pdf_usla_uslt              = Workflow.bool_of   workflow ~default:false "do_plot_joint_pdf_usla_uslt" in
  let do_plot_joint_pdf_uslt_dslt              = Workflow.bool_of   workflow ~default:false "do_plot_joint_pdf_uslt_dslt" in
  let do_plot_joint_pdf_dsla_dslc              = Workflow.bool_of   workflow ~default:false "do_plot_joint_pdf_dsla_dslc" in
  let do_plot_joint_pdf_usla_uslc              = Workflow.bool_of   workflow ~default:false "do_plot_joint_pdf_usla_uslc" in
  let do_plot_joint_pdf_uslc_dslc              = Workflow.bool_of   workflow ~default:false "do_plot_joint_pdf_uslc_dslc" in
  let joint_distbn_n_contours                  = Workflow.float_of   workflow ~default:25. "joint_distbn_n_contours" in
  (*        "joint_distbn_markers" : [["+","crimson",19,4,"w+",17,2,"r.",2,1.0],
        						  ["x","blue",   16,4,"wx",14,2,"k.",2,0.4]],*)
  let marginal_distbn_viz_tilt                 = Workflow.float_of   workflow ~default:0.5  "marginal_distbn_viz_tilt" in
  let marginal_distbn_viz_scale                = Workflow.float_of   workflow ~default:0.5  "marginal_distbn_viz_scale" in
  let joint_distbn_viz_tilt                    = Workflow.float_of   workflow ~default:0.   "joint_distbn_viz_tilt" in
  let joint_distbn_viz_scale                   = Workflow.float_of   workflow ~default:0.25 "joint_distbn_viz_scale" in
  Workflow.set_verbosity workflow verbosity;
  let props =
    {
      do_plot_dtm;
      do_plot_roi;
      do_plot_streamlines;
      do_plot_flow_maps;
      do_plot_segments;
      do_plot_channels;
      do_plot_hillslope_lengths;
      do_plot_hillslope_lengths_contoured;
      do_plot_hillslope_distributions;

      do_plot_downstreamlines;
      do_plot_upstreamlines;
      do_plot_seed_points;
      do_plot_flow_vectors;
      do_plot_blockages;
      do_plot_loops;
      
      plot_interpolation_method;
      plot_streamline_limit;
      
      do_plot_color_shaded_relief;
      do_plot_curvature_roi;
      do_plot_colorized_streamlines;
      do_plot_merged_streamline_density_bands;
      
      plot_window_size_factor;
      plot_window_pdf_size_factor;
      plot_window_width;
      plot_window_height;
      
      hillshade_azimuth;
      hillshade_angle;
      
      downstreamline_color;
      upstreamline_color;
      
      streamline_point_marker;
      streamline_point_size;
      streamline_point_alpha;

      shaded_relief_hillshade_alpha;
      shaded_relief_color_alpha;
      streamline_shaded_relief_hillshade_alpha;
      streamline_shaded_relief_color_alpha;
      streamline_density_alpha;
      streamline_density_cmap;
      
      grid_shaded_relief_hillshade_alpha;
      grid_shaded_relief_color_alpha;
      
      seed_point_marker;
      seed_point_marker_size;
      seed_point_marker_color;
      seed_point_marker_alpha;
      
      channel_head_marker;
      channel_head_marker_alpha;
      channel_shaded_relief_hillshade_alpha;
      
      gradient_vector_color;
      gradient_vector_alpha;
      gradient_vector_scale;
      
      blockage_marker_size;
      loops_marker_size;
      
      classical_streamplot_density;

      terrain_cmap;
      shuffle_random_seed;
      random_cmap_seed;
      
      do_plot_maps;
      do_plot_distributions;
      do_plot_marginal_pdf_dsla;
      do_plot_marginal_pdf_usla;
      do_plot_marginal_pdf_dslt;
      do_plot_marginal_pdf_uslt;
      do_plot_marginal_pdf_dslc;
      do_plot_marginal_pdf_uslc;
      do_plot_joint_pdf_dsla_usla;
      do_plot_joint_pdf_dsla_dslt;
      do_plot_joint_pdf_usla_uslt;
      do_plot_joint_pdf_uslt_dslt;
      do_plot_joint_pdf_dsla_dslc;
      do_plot_joint_pdf_usla_uslc;
      do_plot_joint_pdf_uslc_dslc;
      joint_distbn_n_contours;
      marginal_distbn_viz_tilt;
      marginal_distbn_viz_scale;
      joint_distbn_viz_tilt;
      joint_distbn_viz_scale;
      workflow;
    } in
  props

(**  [read_properties file_path_list cmdline_overrides]

  Read properties from a list of JSON files, and incorporate overrides
  from the command line

 *)
let read_properties file_path_list cmdline_overrides =
  let props = Properties.create () in
  List.iter (fun (path,filename) -> Properties.read_json props path filename) file_path_list;
  let (state, verbosity) = read_state props cmdline_overrides in
  let pocl       = read_pocl       verbosity props in
  let geodata    = read_geodata    verbosity props in
  let preprocess = read_preprocess verbosity props in
  let trace      = read_trace      verbosity props in
  let analysis   = read_analysis   verbosity props in
  let plot       = read_plot       verbosity props in
  { state; pocl; geodata; preprocess; trace; analysis; plot; }
