let sfmt = Printf.sprintf

open Streamlines

let test_0 () =
  Streamlines.go ();
  Alcotest.(check bool) "test 0" true (1=0)

let test_set = [
  "test 0", `Slow, test_0;
  ]
