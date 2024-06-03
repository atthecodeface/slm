let root_dir = "."
let json_dir = root_dir ^ "/json"
let _ =
  let (parameters_filename, json) = Streamlines.parse_arguments () in
  Streamlines.set_root ".";
  Streamlines.process json_dir parameters_filename json
