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
 *
 * Up to date with python of git CS 9b039412ca3e76b47c78bba1593f93e7523fe45d
 *
 * However, might want to suck in some more of useful.py such as seed picking - but that may belong in core.
 *
 * v}
 *)

(*a Module abbreviations *)
module ODM = Owl.Dense.Matrix.Generic
module ODN = Owl.Dense.Ndarray.Generic

(** {1 Filename handling}

  These functions standardize the handling of filename generation from
  paths, directory and leafnames. A global {i rot directory} is
  supported, to provide a base for all relative filenames. Path
  elements may also be environment variables, which are automatically
  substituted.

 *)

(**  BadEnvironmentVariable exception

Raised if an environment variable is referenced that is not actually
within the environment

 *)
exception BadEnvironmentVariable of string

(**  [root_dir] - global that holds the root directory to be prepended to all relative paths *)
let root_dir = ref "."

(**  [set_root root_dir]

Sets the global root directory that is used for all relative paths

 *)
let set_root new_root =
    root_dir := new_root

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

(** {1 Useful functions} *)

(**  [trace pos] 

 Trace function used to debug code, particularly when it is
 crashing. Use as trace __POS__ to trace execution

 *)
let trace pos =
    let (a,b,c,d) = pos in
    Printf.printf "trace:%s:%d:%d:%d\n%!" a b c d

(**  [sfmt] 

 Short-cut for Printf.sprintf, used throughout the code for reduced
 code clutter.

 *)
let sfmt = Printf.sprintf

(**  [get_padded_array ba width value]

 Pad a 2D generic big array by a certain padding width on all four
 sides, using the specified value for that padding.

 *)
let get_padded_array ba width value = 
  if (width>0) then (
    let padding = [[width;width];[width;width];] in
    ODM.pad ~v:value padding ba;
  ) else (
     ba
  )

(**  [list_assoc_replace name value]

 Add or replace (name*value) in assoc List

 *)
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

(**  [is_nan x] 

 Shorthand for the correct way to determine if a float is NAN

 *)
let is_nan x = (compare x nan)=0

(** {1 Bigarray types and handling} *)

(** {2 Types} *)

(**  t_ba_chars

  Type short-cut for a Bigarray of chars that can be used with Owl and OpenCL

 *)
type t_ba_chars   = (char, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Genarray.t

(**  t_ba_int16s

  Type short-cut for a Bigarray of 16-bit unsigned ints that can be used with Owl and OpenCL

 *)
type t_ba_int16s  = (int, Bigarray.int16_unsigned_elt, Bigarray.c_layout) Bigarray.Genarray.t

(**  t_ba_ints

  Type short-cut for a Bigarray of Ocaml ints, for use internally and with Owl. Since
  Ocaml int's are system-size dependent (31 or 63 bits) the t_ba_ints
  is less useful for OpenCL.

 *)
type t_ba_ints    = (int, Bigarray.int_elt, Bigarray.c_layout) Bigarray.Genarray.t

(**  t_ba_floats

  Type short-cut for a Bigarray of single-precision floats for use with Owl and OpenCL

 *)
type t_ba_floats  = (float, Bigarray.float32_elt, Bigarray.c_layout) Bigarray.Genarray.t

(** {2 Array creation functions} *)

(**  [ba_chars size]

  Create a big array of {!val size!} unsigned chars for use with Owl and OpenCL

 *)
let ba_chars     size = Bigarray.(genarray_of_array1 (Array1.create char c_layout size))

(**  [ba_int16s size]

  Create a big array of {!val size!} unsigned 16-bit integers for use with Owl and OpenCL

 *)
let ba_int16s    size = Bigarray.(genarray_of_array1 (Array1.create int16_unsigned c_layout size))

(**  [ba_ints size]

  Create a big array of {!val size!} Ocaml ints chars for use internally and with Owl

 *)
let ba_ints      size = Bigarray.(genarray_of_array1 (Array1.create int c_layout size))

(**  [ba_floats size]

  Create a big array of {!val size!} single-preicision floats for use with Owl and OpenCL

 *)
let ba_floats    size = Bigarray.(genarray_of_array1 (Array1.create float32 c_layout size))

(**  [ba_char2d width height]

  Create a big array of {!val width!} by {!val height!} of unsigned chars for use with Owl and OpenCL

 *)
let ba_char2d    w h  = Bigarray.(genarray_of_array2 (Array2.create char c_layout w h))

(**  [ba_int162d width height]

  Create a big array of {!val width!} by {!val height!} of unsigned 16-bit integers for use with Owl and OpenCL

 *)
let ba_int162d   w h  = Bigarray.(genarray_of_array2 (Array2.create int16_unsigned c_layout w h))

(**  [ba_int2d width height]

  Create a big array of {!val width!} by {!val height!} of Ocaml integers for use internally and with Owl

 *)
let ba_int2d   w h    = Bigarray.(genarray_of_array2 (Array2.create int c_layout w h))

(**  [ba_float2d width height]

  Create a big array of {!val width!} by {!val height!} of single precision floats for use with Owl and OpenCL

 *)
let ba_float2d w h    = Bigarray.(genarray_of_array2 (Array2.create float32 c_layout w h))

(**  [ba_int3d depth width height]

  Create a big array of {!val depth!} by {!val width!} by {!val height!} of Ocaml integers for interal use

 *)
let ba_int3d   d w h  = Bigarray.(genarray_of_array3 (Array3.create int c_layout d w h))

(**  [ba_float3d depth width height]

  Create a big array of {!val depth!} by {!val width!} by {!val height!} of single precision floats for use with Owl and OpenCL

 *)
let ba_float3d d w h  = Bigarray.(genarray_of_array3 (Array3.create float32 c_layout d w h))

(** {2 Bigarray functions - for Owl?} Probably suitable to be migrated to Owl *)

(**  [ba_cast kind f x]
 *)
let ba_cast kind f x =
  let y = ODN.empty kind (ODN.shape x) in
  let y' = ODN.flatten y |> Bigarray.array1_of_genarray in
  let x' = ODN.flatten x |> Bigarray.array1_of_genarray in
  for i = 0 to (Bigarray.Array1.dim x') - 1 do
    let a = Bigarray.Array1.unsafe_get x' i in
    Bigarray.Array1.unsafe_set y' i (f a)
  done;
  y

(**  [ba_fold f acc ba]

  Fold a function over the big array

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
  let ba' = Bigarray.array1_of_genarray (ODM.flatten ba) in
  let b = ref acc in
  for i = 0 to (ODN.numel ba) - 1 do
    let c = Bigarray.Array1.unsafe_get ba' i in
    b := f !b c
  done;
  !b

(**  [ba_foldi f acc ba]

  Fold a function over the big array with flattened element index

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
let ba_foldi f a ba =
  let ba' = Bigarray.array1_of_genarray (ODM.flatten ba) in
  let b = ref a in
  for i = 0 to (ODN.numel ba) - 1 do
    let c = Bigarray.Array1.unsafe_get ba' i in
    b := f i !b c
  done;
  !b

(**  [ba_fold2d f acc ba]

  Fold a function over the 2D big array

    @param f the function to apply to x, y, {!val acc} and the big array element to produce a new accumulator

    @param acc the base accumulator to use for the first invocation of f

    @param ba the big array to fold over

    @return f applied to acc and every element of ba

  The order of the elements of {!val:ba} that f is applied to is not
  specified; in some implementations f may be applied in parallel to
  many elements (with some {!val:acc}); the final result, though, is
  the application of f to every element of ba exactly once.

 *)
let ba_fold2d f acc src =
  let acc_r = ref acc in
  let f_acc x y v =
    acc_r := f x y (!acc_r) v
  in
  ODM.iteri_2d f_acc src;
  !acc_r

(**  [filtered_array f ba] - may well be used by analysis in the future *)
let filtered_array f ba =
  let ba = ODM.flatten ba in
  let ba' = Bigarray.array1_of_genarray ba in
  let indices = ODN.filteri f ba in
  let n = Array.length indices in
  ODN.init (Bigarray.Genarray.kind ba) [|n|] (fun i -> ba'.{indices.(i)})

(**  [ba_map_ f ba]

  Map a function over the big array in-place

    @param f the function to apply to the big array element to produce a new replacement value

    @param ba the big array to fold over

    @return ba after f has been applied to each element

  The order of the elements of {!val:ba} that f is applied to is not
  specified; in some implementations f may be applied in parallel to
  many elements; the final result, though, is
  the application of f to every element of ba exactly once.

 *)
let ba_map_ f x =
  let x' = ODN.flatten x |> Bigarray.array1_of_genarray in
  for i = 0 to (Bigarray.Array1.dim x') - 1 do
    let a = Bigarray.Array1.unsafe_get x' i in
    Bigarray.Array1.unsafe_set x' i (f a)
  done;
  x

(**  [ba_mapi_ f ba]

  Map a function over the big array in-place using element index

    @param f the function to apply to the big array element to produce a new replacement value

    @param ba the big array to fold over

    @return ba after f has been applied to each element

  The order of the elements of {!val:ba} that f is applied to is not
  specified; in some implementations f may be applied in parallel to
  many elements; the final result, though, is
  the application of f to every element of ba exactly once.

 *)
let ba_mapi_ f x =
  let x' = ODN.flatten x |> Bigarray.array1_of_genarray in
  for i = 0 to (Bigarray.Array1.dim x') - 1 do
    let a = Bigarray.Array1.unsafe_get x' i in
    Bigarray.Array1.unsafe_set x' i (f i a)
  done;
  x

(**  [ba_filter xsize ysize f rv src dst]

  Apply an (xsize*ysize) filter to every element of 2D src, storing the
  result in dst; for regions near the edge (i.e. where the filter
  lies beyond the edges of src) then rv is used instead of the
  filter application.

    @param xsize number of X elements required by the filter

    @param ysize number of Y elements required by the filter

    @param f the function to apply to X and Y of src the big array element to produce a new value. 

    @param rv value to store in destination for the edge (X-1)/2 and (Y-1)/2 values

    @param src the big array to apply the filter to

    @param dst the big array to store the result in

    @return dst

  The order of the elements of {!val:src} that f is applied to is not
  specified; in some implementations f may be applied in parallel to
  many elements. f is invoked with a 'get' function and x and y as arguments; an example function would be
  {[let gradient_x get x y = ((get (x+1) y) -. (get (x-1) y)) *. 0.5]}

 *)
let ba_filter xsize ysize f rv src dst =
  let ba_owl2d ba = Bigarray.array2_of_genarray ba in
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

