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

(*a Module abbreviations *)
module ODM = Owl.Dense.Matrix.Generic
module ODN = Owl.Dense.Ndarray.Generic

(** {1 Filename handling} *)

(**  [root_dir] - global that holds the root directory to be prepended to all relative paths *)
let root_dir = ref "."

(**  [set_root root_dir]
 *)
let set_root new_root =
    root_dir := new_root

(** BadEnvironmentVariable of string *)
exception BadEnvironmentVariable of string

(**  [env_value_or_path path]

  Attempt to interpret a string as an environment variable (such as
$SLMDATA) else return it as a path string.

 *)
let env_value_or_path path =
  if path.[0] <> '$' then path else (
    let var_name = String.(sub path 1 ((length path)-1)) in
    match Sys.getenv_opt var_name with
    | Some v -> v
    | None -> raise (BadEnvironmentVariable var_name)
  )

(**  [filename_from_path path leaf]

    Generate a full filename from a path and a leaf filename.    

 @param path a list of string elements that may be environment
 variables, absolute paths, or relative directories. This path is used
 with the {!val root_dir} as a root directory to generate the final
 path.

@param leaf the leaf filename that is to be appended to the path after
that is resolved

 *)
let filename_from_path path leaf =
    let path = List.map env_value_or_path path in
    let path_stripped = List.fold_left (fun acc n->if n.[0]='/' then [n] else acc@[n]) [!root_dir] path in
    (String.concat "/" path_stripped) ^ "/" ^ leaf

(** {1 Bigarray types and handling} *)

(** {2 Types} *)

(**  t_ba_chars *)
type t_ba_chars   = (char, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Genarray.t
type t_ba_int16s  = (int, Bigarray.int16_unsigned_elt, Bigarray.c_layout) Bigarray.Genarray.t
type t_ba_ints    = (int, Bigarray.int_elt, Bigarray.c_layout) Bigarray.Genarray.t
type t_ba_floats  = (float, Bigarray.float32_elt, Bigarray.c_layout) Bigarray.Genarray.t

(** {2 Mapping functions} *)

(**  *)
let owl_ba3d   ba = Bigarray.genarray_of_array3 ba
let ba_owl3d   ba = Bigarray.array3_of_genarray ba
let owl_ba2d   ba = Bigarray.genarray_of_array2 ba
let ba_owl2d   ba = Bigarray.array2_of_genarray ba
let owl_ba1d   ba = Bigarray.genarray_of_array1 ba
let ba_owl1d   ba = Bigarray.array1_of_genarray ba

(** {2 Array creation functions} *)

let ba_chars     size = owl_ba1d Bigarray.(Array1.create char c_layout size)
let ba_char2d    h w  = owl_ba2d Bigarray.(Array2.create char c_layout h w)
let ba_int16s    size = owl_ba1d Bigarray.(Array1.create int16_unsigned c_layout size)
let ba_int162d   h w  = owl_ba2d Bigarray.(Array2.create int16_unsigned c_layout h w)
let ba_ints    size   = owl_ba1d Bigarray.(Array1.create int c_layout size)
let ba_int2d   h w    = owl_ba2d Bigarray.(Array2.create int c_layout h w)
let ba_floats  size   = owl_ba1d Bigarray.(Array1.create float32 c_layout size)
let ba_float2d h w    = owl_ba2d Bigarray.(Array2.create float32 c_layout h w)
let ba_int3d   d h w  = owl_ba3d Bigarray.(Array3.create int c_layout d h w)
let ba_float3d d h w  = owl_ba3d Bigarray.(Array3.create float32 c_layout d h w)

(** {2 Operation functions} Probably suitable to be migrated to Owl *)

(**  [ba_fold f acc ba]

  Fold over the big array (over any dimension?) 

    @param f the function to apply to {!val acc} and the big array element to produce a new accumulator

    @param acc the base accumulator to use for the first invocation of f

    @param ba the big array to fold over

    @return f applied to acc and every element of ba

  The order of the elements of {!val:ba} that f is applied to is not
  specified; in some implementations f may be applied in parallel to
  many elements (with some {!val:acc}); the final result, though, is
  the application of f to every element of ba exactly once. Hence an
  application of [ba_fold min infinity ba] will always return the
  minimum value in ba, but [ba_fold (fun x->Some x) None ba] will
  return an Some x where x is {i any} element of ba.

 *)
let ba_fold f acc ba =
  let ba' = ba_owl1d (ODM.flatten ba) in
  let b = ref acc in
  for i = 0 to (ODN.numel ba) - 1 do
    let c = Bigarray.Array1.unsafe_get ba' i in
    b := f !b c
  done;
  !b

let ba_foldi f a ba =
  let ba' = ba_owl1d (ODM.flatten ba) in
  let b = ref a in
  for i = 0 to (ODN.numel ba) - 1 do
    let c = Bigarray.Array1.unsafe_get ba' i in
    b := f i !b c
  done;
  !b

(**  [filtered_array f ba] *)
let filtered_array f ba =
  let ba = ODM.flatten ba in
  let ba' = ba_owl1d ba in
  let indices = ODN.filteri f ba in
  let n = Array.length indices in
  ODN.init (Bigarray.Genarray.kind ba) [|n|] (fun i -> ba'.{indices.(i)})

(**  map_ *)
let map_ f x =
  let x' = ODN.flatten x |> Bigarray.array1_of_genarray in
  for i = 0 to (Bigarray.Array1.dim x') - 1 do
    let a = Bigarray.Array1.unsafe_get x' i in
    Bigarray.Array1.unsafe_set x' i (f a)
  done;
  x

(**  mapi_ *)
let mapi_ f x =
  let x' = ODN.flatten x |> Bigarray.array1_of_genarray in
  for i = 0 to (Bigarray.Array1.dim x') - 1 do
    let a = Bigarray.Array1.unsafe_get x' i in
    Bigarray.Array1.unsafe_set x' i (f i a)
  done;
  x

(**  ba_filter *)
let ba_filter xsize ysize f rv src dst =
  let (w,h) = ODM.shape src in
  let min_x = (xsize-1)/2 in
  let min_y = (ysize-1)/2 in
  let max_x = w-1-min_x in
  let max_y = h-1-min_y in
  let ba_src = ba_owl2d src in
  let ba_dst = ba_owl2d dst in
  let f_get = f (fun x y -> Bigarray.Array2.unsafe_get ba_src x y) in
  let f_pruned x y v =
    if ((x<min_x) || (y<min_y) || (x>max_x) || (y>max_y)) then (
      Bigarray.Array2.unsafe_set ba_dst x y rv
    ) else (
      Bigarray.Array2.unsafe_set ba_dst x y (f_get x y)
    )
  in
  ODM.iteri_2d f_pruned src;
  dst

(*a Useful functions *)
(*f trace - use as trace __POS__ to trace execution *)
let trace pos =
    let (a,b,c,d) = pos in
    Printf.printf "trace:%s:%d:%d:%d\n%!" a b c d

(*f sfmt - shorthand for Printf.sprintf *)
let sfmt = Printf.sprintf

(*f get_padded_array ba width value *)
let get_padded_array ba width value = 
  if (width>0) then (
    let padding = [[width;width];[width;width];] in
    ODM.pad ~v:value padding ba;
  ) else (
     ba
  )

(*f [list_assoc_replace name value] - add/replace (name*value) in assoc List *)
let list_assoc_replace l name value =
  let rec find_name l rev_hd =
    match l with 
    | [] -> (rev_hd,[(name,value)])
    | hd::tl -> (
      let (n,v) = hd in
      if (compare n name)=0 then (
        ((name,value)::rev_hd, tl)
      ) else (
        find_name tl (hd::rev_hd)
      )
    )
  in
  let (tl, rev_hd) = find_name l [] in
  List.rev_append rev_hd tl

(*f is_nan - shorthand for the correct way to determine if a float is NAN *)
let is_nan x = (compare x nan)=0

