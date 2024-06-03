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
 * @file   analysis.ml
 * @brief  Tools to compute statistical distributions (pdfs) and model their properties
 *
 *)

(*f Libraries *)
open Globals
open Properties
open Core
module ODM = Owl.Dense.Matrix.Generic
module ODN = Owl.Dense.Ndarray.Generic
module OS  = Owl.Stats
module Option = Batteries.Option

(** {1 Useful functions} *)
let log_or_min x = if x<=0. then min_float else log x

(** {1 Univariate distribution}
    Class for making and recording kernel-density estimate of univariate 
    probability distribution f(x) data and metadata. 
    Provides a method to find the modal average: x | max{f(x}.

A univariate distribution of logx values is created from a pair of
logx and logy arrays.

The logy array is {i only} used to filter out points (logx,logy) which
lie outside of the range logy_min to logy_max.

Points whose logx value lie outside a logx_min to logx_max range are
also filtered out.

 *)

module Univariate_distribution =
struct

  (**  [t_stats]
   *)
  type t_stats = {
      mean     : float;
      stddev   : float;
      variance : float;
    }                             

  (**  [t]
   *)
  type t = {
      pixel_size     : float;
      logx_data      : t_ba_floats; (* log(x) from x_array that is in bounds *)
      n_data         : int; (* Size of logx_data *)
      bounds         : float * float * float * float; (* min x, min y, max x, max y, used to get data (logs) *)
      info           : Info.t; (* Info with data to use for KDE etc *)
      logx_vec       : t_ba_floats; (* n_pdf_points linearly spaced from logx_min to logx_max *)
      x_vec          : t_ba_floats; (* n_pdf_points logarithmically spaced from x_min to x_max *)
      mutable pdf    : t_ba_floats;
      mutable cdf    : t_ba_floats;
      stats_raw      : t_stats;
      mutable stats_kde      : t_stats;
    }

  (**  [create]
    logx_array must have only non-NaN elements as ODN.min requires it
    logy_array must have only non-NaN elements as ODN.min requires it
   *)
  let create ?pixel_size:(pixel_size=1.0) ~n_hist_bins ~n_pdf_points ~kernel ~bandwidth ~logx_array ~logy_array bounds =
    let info = Info.create () in
    Info.set_int     info "n_hist_bins"  n_hist_bins;
    Info.set_int     info "n_pdf_points" n_pdf_points;
    Info.set_str     info "kernel"       kernel;
    Info.set_float32 info "bandwidth"    bandwidth;

    let (logx_min, logy_min, logx_max, logy_max) = bounds in
    let logx_min = Option.default_delayed (fun _ -> ODN.min' logx_array) logx_min in
    let logy_min = Option.default_delayed (fun _ -> ODN.min' logy_array) logy_min in
    let logx_max = Option.default_delayed (fun _ -> ODN.max' logx_array) logx_max in
    let logy_max = Option.default_delayed (fun _ -> ODN.max' logy_array) logy_max in
    let bounds = (logx_min, logy_min, logx_max, logy_max) in

    let f i logx =
     let logy = ODN.get logy_array [|i|] in
     ((logx>=logx_min) && (logx<=logx_max) && (logy>=logy_min) && (logy<=logy_max)) in
    let logx_data = filtered_array f logx_array in
    let n_data = (ODN.shape logx_data).(0) in
    let logx_vec  = ODN.linspace Bigarray.float32 logx_min logx_max n_pdf_points in

    let mean     = exp (OS.mean (ODN.to_array logx_data)) in
    let stddev   = exp (OS.std  (ODN.to_array logx_data)) in
    let variance = exp (OS.var  (ODN.to_array logx_data)) in
    let stats_raw = {mean; stddev; variance;} in

    let x_vec     = ODN.exp logx_vec in
    {
      pixel_size;
      logx_data;
      n_data;
      bounds;
      logx_vec;
      x_vec;
      info;
      pdf = ba_float2d 1 1;
      cdf = ba_float2d 1 1;
      stats_raw;
      stats_kde = {mean=0.; stddev=0.; variance=0.;}
    }

  (**  [compute_kde
   *)
  let compute_kde t props pocl =
    let kernel = Info.str_of t.info "kernel" in
    if kernel="gaussian" then (
      Info.set_float32 t.info "bandwidth" ((Info.float_of t.info "bandwidth") /. 2.);
    );
    let available_kernels = ["tophat"; "triangle"; "epanechnikov"; "cosine"; "gaussian"] in
    let kernel_available = List.fold_left (fun acc s -> if kernel=s then true else acc) false available_kernels in
    if not kernel_available then raise Not_found;

    let (logx_min, logy_min, logx_max, logy_max) = t.bounds in
    let x_range = logx_max -. logx_min in
    let bin_dx  = x_range /. (float (Info.int_of t.info "n_hist_bins")) in
    let pdf_dx  = x_range /. (float (Info.int_of t.info "n_pdf_points")) in
    Info.set_float32 t.info "kdf_bandwidth" (Info.float_of t.info "bandwidth");

    Info.set_float32 t.info "x_min"         logx_min;
    Info.set_float32 t.info "x_max"         logx_max;
    Info.set_float32 t.info "x_range"       x_range;
    Info.set_float32 t.info "bin_dx"        bin_dx;
    Info.set_float32 t.info "pdf_dx"        pdf_dx;

    Info.set_float32 t.info "y_min"         0.;
    Info.set_float32 t.info "y_may"         1.;
    Info.set_float32 t.info "y_range"       1.;
    Info.set_float32 t.info "bin_dy"        (1.0 /. 2000.);
    Info.set_float32 t.info "pdf_dy"        (1.0 /. 200.);

    Info.set_float32 t.info "kdf_width_x"   0.;
    Info.set_float32 t.info "kdf_width_y"   0.;
    Info.set_uint32  t.info "n_kdf_part_points_x"   0l;
    Info.set_uint32  t.info "n_kdf_part_points_x"   0l;

    Info.set_uint32 t.info "n_data" (Int32.of_int t.n_data);

    let n_pdf_points = Info.int_of t.info "n_pdf_points" in
    t.pdf <- Kde.estimate_univariate_pdf props pocl t.info t.logx_data;
    t.cdf <- ODM.(mul_scalar (cumsum t.pdf) bin_dx);
    let should_be_one = ODN.get t.cdf [|n_pdf_points-1;0|] in
    if (abs_float (should_be_one -. 1.)) > 5e-3 then (
      Printf.printf "Error/imprecision when computing cumulative probability distribution:\npdf integrates to %3f not to 1.0\n%!" should_be_one);
    ()
  (**  [show_statistics t]
    *)
  let show_statistics t = 
    Printf.printf "raw mean:  %.2f\n" t.stats_raw.mean;
    Printf.printf " sigma:  %.2f\n"   t.stats_raw.stddev;
    Printf.printf " var:  %.2f\n"     t.stats_raw.variance;
    Printf.printf "kde mean:  %.2f\n" t.stats_kde.mean;
    Printf.printf " sigma:  %.2f\n"   t.stats_kde.stddev;
    Printf.printf " var:  %.2f\n"     t.stats_kde.variance;
    ()

  (**  [calc_statistics t] *)
  let calc_statistics t =
    let x   = t.logx_vec in
    let pdf = t.pdf in
    let log_mean     = ODN.(get (cumsum (mul x pdf)) [|1|]) in (* / sum pdf *)
    let log_variance = ODN.(get (cumsum (mul pdf (sqr (sub_scalar x log_mean)))) [|1|]) in (* / sum pdf *)
    let mean      = exp log_mean in
    let stddev    = exp (sqrt log_variance) in
    let variance  = exp log_variance in
    t.stats_kde <- {mean; stddev; variance;};

    ()
    
    let find_modes t =
(*        x = self.x_vec
        pdf = self.kde['pdf']
        approx_mode = np.round(x[pdf==pdf.max()][0],2)
#             pdebug('kde approx mode: {}'.format(approx_mode))
        for trial in [0,1]:
            peaks = argrelextrema(np.reshape(np.power(x,trial)*pdf,pdf.shape[0],), 
                                  np.greater, order=3)[0]
            peaks = [peak for peak in list(peaks) if x[peak]>=approx_mode*0.5 ]
            if trial==0:
                self.kde['mode_i'] = peaks[0]
                self.kde['mode_x'] = x[peaks[0]][0]
            try:
                self.kde['bimode_i'] = peaks[1]
                self.kde['bimode_x'] = x[peaks[1]][0]
                break
            except:
                self.kde['bimode_x'] = 0
                if trial==0:
                    self.print('can\'t find second mode, trying with pdf*x')
                else:
                    self.print('failed to find second mode')
        self.print('kde modes: {0}, {1}'.format(np.round(self.kde['mode_x'],2),
                                         np.round(self.kde['bimode_x'],2)))
 *)
    ()
  (**  [choose_threshold t] *)
  let choose_threshold t =
(*
        x_vec = self.x_vec
        pdf = self.kde['pdf']
        cdf = self.kde['cdf']
        mode_x = self.kde['mode_x']
        loc =   np.log(self.kde['mean'])
        scale = np.log(self.kde['stddev'])
        norm_pdf = norm.pdf(np.log(x_vec),loc,scale)
        detrended_pdf = pdf/norm_pdf
        extrema_i = argrelextrema(np.reshape(detrended_pdf,pdf.shape[0],), 
                                  np.less, order=3)[0]
        extrema_i = [extremum_i for extremum_i in extrema_i 
                     if x_vec[extremum_i]>mode_x \
                        and cdf[extremum_i]>self.search_cdf_min]
        try:
            self.kde['channel_threshold_i'] = extrema_i[0]
            self.kde['channel_threshold_x'] = x_vec[extrema_i[0]][0]
        except:
            self.print('Choosing threshold: failed to find minima in range')
        self.print(' cdf @ {}'.format(
            [np.round(cdf[extremum_i], 2) for extremum_i in extrema_i] ))
        self.print(' kinks @ {}'.format(
            [np.round(x_vec[extremum_i][0], 2) for extremum_i in extrema_i] ))
 *)
    ()

  (*f  All done *)   
end

(** {1 Bivariate_distribution}

    Container class for kernel-density-estimated bivariate probability distribution
    f(x,y) data and metadata. Also has methods to find the modal average (xm,ym)
    and to find the cluster of points surrounding the mode given a pdf threshold
    and bounding criteria.

 *)
module Bivariate_distribution =
struct
(*    def __init__(self, logx_array=None,logy_array=None, mask_array=None,
                 method='sklearn', n_samples=100, shear_factor=0.0, 
                 logx_min=None,logy_min=None,logx_max=None,logy_max=None,
                 pixel_size=None, verbose=False):
        self.logx_array = logx_array
        self.logy_array = logy_array
        if logx_min is None:
            logx_min = logx_array[logx_array>np.finfo(np.float32).min].min()
        if logx_max is None:
            logx_max = logx_array[logx_array>np.finfo(np.float32).min].max()
        if logy_min is None:
            logy_min = logy_array[logy_array>np.finfo(np.float32).min].min()
        if logy_max is None:
            logy_max = logy_array[logy_array>np.finfo(np.float32).min].max()
        # Transform the y values to shear the pdf and detrend
        if shear_factor!=0.0:
            logy_array -= logx_array*shear_factor
            logy_min -= logx_min*shear_factor
            logy_max -= logx_max*shear_factor
        if mask_array is not None:
            self.logxy_data = np.vstack([
                logx_array[  (logx_array>=logx_min) & (logx_array<=logx_max) 
                           & (logy_array>=logy_min) & (logy_array<=logy_max)
                           & (~mask_array)],
                logy_array[  (logx_array>=logx_min) & (logx_array<=logx_max) 
                           & (logy_array>=logy_min) & (logy_array<=logy_max)
                           & (~mask_array)]
                ]).T
        else:
            self.logxy_data  = np.vstack([
                logx_array[  (logx_array>=logx_min) & (logx_array<=logx_max) 
                           & (logy_array>=logy_min) & (logy_array<=logy_max)],
                logy_array[  (logx_array>=logx_min) & (logx_array<=logx_max) 
                           & (logy_array>=logy_min) & (logy_array<=logy_max)]
                ]).T
            
        # x,y meshgrid for sampling the bivariate pdf f(x,y)
        self.logx_mesh,self.logy_mesh = np.mgrid[logx_min:logx_max:n_samples,
                                                 logy_min:logy_max:n_samples]
        self.logxy_data_indexes = np.vstack([self.logx_mesh.ravel(), 
                                             self.logy_mesh.ravel()]).T
        self.x_mesh = np.exp(self.logx_mesh)
        self.y_mesh = np.exp(self.logy_mesh)
        self.x_vec = self.x_mesh[:,0]
        self.y_vec = self.y_mesh[0,:]

        kde_keys = [
            'model',
            'method',
            'bw_method',
            'pdf',
            'mode_ij_list',
            'mode_xy_list',
            'mode_max_list',
            'near_mode_vec_list'
            ]
        self.kde = dict.fromkeys(kde_keys)        
        self.kde['method'] = method
        self.kde['mode_ij_list'] = [None,None]
        self.kde['mode_xy_list'] = [None,None]
        self.kde['mode_max_list'] = [None,None]
        self.kde['near_mode_vec_list'] = [None,None]
        self.mode_cluster_ij_list = [None,None]
        
        self.pixel_size = pixel_size
        self.verbose = verbose

    def compute_kde_scipy(self, bw_method='scott'):
        # Compute bivariate pdf
        self.kde['bw_method'] = bw_method
        self.kde['model'] = gaussian_kde(self.logxy_data.T, bw_method=bw_method)
        self.kde['pdf'] = np.reshape( self.kde['model'](self.logxy_data_indexes.T
                                                        ),self.logx_mesh.shape)
                                       
    def compute_kde_sklearn(self, kernel='epanechnikov', bandwidth=0.10):
        self.kernel = kernel
        self.bandwidth = bandwidth
        self.kde['model'] = KernelDensity(kernel=self.kernel, 
                                 bandwidth=self.bandwidth).fit(self.logxy_data)
        self.kde['pdf'] = np.reshape( 
            np.exp(self.kde['model'].score_samples(self.logxy_data_indexes)
                                                        ),self.logx_mesh.shape)  

    def find_mode(self, mode_idx, tilt=0):
        # Prep
        kde_pdf = self.kde['pdf']*np.power(self.y_mesh,tilt)
        
        # Find mode = (x,y) @ max{f(x,y)}
        max_pdf_idx = np.argmax(kde_pdf,axis=None)
        mode_ij = np.unravel_index(max_pdf_idx, kde_pdf.shape)
        mode_xy = np.array([ self.x_mesh[mode_ij[0],0],self.y_mesh[0,mode_ij[1]] ])
        
        # If tilt used to locate 2ndary mode, precisely relocate without tilt
        if tilt!=0:
            offzone= np.where(kde_pdf<kde_pdf[mode_ij[0],mode_ij[1]]*0.9)
            kde_pdf = self.kde['pdf'].copy()
            kde_pdf[offzone] = 0.0
            max_pdf_idx = np.argmax(kde_pdf,axis=None)
            mode_ij = np.unravel_index(max_pdf_idx, kde_pdf.shape)
            mode_xy = np.array([ self.x_mesh[mode_ij[0],0],self.y_mesh[0,mode_ij[1]] ])

        # Record mode info
        self.kde['mode_max_list'][mode_idx] = kde_pdf[ mode_ij[0],mode_ij[1] ]
        self.kde['mode_ij_list'][mode_idx] = mode_ij
        self.kde['mode_xy_list'][mode_idx] = mode_xy
            
    def find_near_mode(self, mode_idx=0, tilt=0, marginal_distbn=None,
                       mode_threshold=0.95, nearness=30, upstream_modal_length=None):
        mode_xy = self.kde['mode_xy_list'][mode_idx]
        kde_pdf = self.kde['pdf'] #* np.power(self.y_mesh,tilt)
        mode_max = self.kde['mode_max_list'][mode_idx]
        if mode_idx==1:
            try:
                self.channel_threshold = marginal_distbn.kde['channel_threshold_x']
            except:
                pdebug('Guessing channel threshold')
                self.channel_threshold = 2*self.kde['mode_xy_list'][1][0]
            near_mode_ij \
                = np.where( (self.y_mesh>=self.channel_threshold)
                            & (kde_pdf>=mode_max*mode_threshold))
        else:
            near_mode_ij = np.where(  (kde_pdf>=mode_max*mode_threshold)
                                     & (self.y_mesh/mode_xy[1]>=1/nearness) 
                                     & (self.y_mesh/mode_xy[1]<=nearness) )

        near_mode_xy = (  self.x_mesh[near_mode_ij[0],0],
                              self.y_mesh[0,near_mode_ij[1]]  )
        self.kde['near_mode_vec_list'][mode_idx] \
            = np.vstack([near_mode_xy[0], near_mode_xy[1]]).T
        
        # Calculate the dx,dy span of the discrete pdf aka the 'bin' width 
        logx_vec = np.log(self.x_vec)
        logy_vec = np.log(self.y_vec)
        dlogx = (logx_vec[1]-logx_vec[0])/2
        dlogy = (logy_vec[1]-logy_vec[0])/2
        near_mode_ij = np.vstack([near_mode_ij[0],near_mode_ij[1]]).T
        near_mode_pdf_bands = [
            ( i, min(near_mode_ij[np.where(near_mode_ij[:,0]==i)[0],1]),
                   max(near_mode_ij[np.where(near_mode_ij[:,0]==i)[0],1]) )
                       for i in np.unique(near_mode_ij[:,0])]
        near_mode_pdf_zones = np.exp(np.array([
                            ( (logx_vec[band[0]]-dlogx, logx_vec[band[0]]+dlogx),
                              (logy_vec[band[1]]-dlogy, logy_vec[band[2]]+dlogy) )
                          for band in near_mode_pdf_bands]))
        self.mode_cluster_ij_list[mode_idx] \
            = np.concatenate([
                        np.array(np.where(  (self.logx_array>=np.log(nmpz[0][0]))
                                          & (self.logx_array<=np.log(nmpz[0][1]))
                                          & (self.logy_array>=np.log(nmpz[1][0])) 
                                          & (self.logy_array<=np.log(nmpz[1][1]))  )).T
                        for nmpz in near_mode_pdf_zones
               ])     
    
 *)
end

(** {1 Types} **)
(*t  [t] *)
type t = {
    props : t_props_analysis;
    mutable area_correction_factor : float;
    mutable length_correction_factor : float;
    mutable mpdf_dsla : Univariate_distribution.t option;
    mutable mpdf_usla : Univariate_distribution.t option;
    mutable mpdf_dslc : Univariate_distribution.t option;
    mutable mpdf_uslc : Univariate_distribution.t option;
    mutable mpdf_dslt : Univariate_distribution.t option;
    mutable mpdf_uslt : Univariate_distribution.t option;
  }

(** {1 pv_verbosity functions} *)

(**  [pv_noisy t]

  Shortcut to use {!type:Properties.t_props_trace} verbosity for {!val:Properties.pv_noisy}

 *)
let pv_noisy   data = Workflow.pv_noisy data.props.workflow

(**  [pv_debug t]

  Shortcut to use {!type:Properties.t_props_trace} verbosity for {!val:Properties.pv_debug}

 *)
let pv_debug   data = Workflow.pv_noisy data.props.workflow

(**  [pv_info t]

  Shortcut to use {!type:Properties.t_props_trace} verbosity for {!val:Properties.pv_info}

 *)
let pv_info    data = Workflow.pv_info data.props.workflow

(**  [pv_verbose t]

  Shortcut to use {!type:Properties.t_props_trace} verbosity for {!val:Properties.pv_verbose}

 *)
let pv_verbose data = Workflow.pv_verbose data.props.workflow

(** {1 Dist_data module}
 *)
module Dist_data =
struct
  (**  [t]
   *)
  type t = {
      data    : t_ba_floats;
      opt_min : float option;
      opt_max : float option;
    }

  (**  [create data opt_min opt_max]
   *)
  let create data opt_min opt_max = { data; opt_min; opt_max }

  (**  [create_int data opt_min opt_max]
   *)
  let create_int data opt_min opt_max =
    let data = ba_cast Bigarray.float32 float data in
    create data opt_min opt_max

  (*f All done
   *)
end

(** {1 Toplevel of analysis module}

    Class providing statistics & probability tools to analyze streamline data and its
    probability distributions.

 *)
(**  [compute_marginal_distribn t ?n_hist_bins ?n_pdf_points ?kernel ?bandwidth x_data y_data bounds]
 *)
let compute_marginal_distribn t pocl ?n_hist_bins ?n_pdf_points ?kernel ?bandwidth (x_data:Dist_data.t) (y_data:Dist_data.t) =
  let logx_min = Option.map log x_data.opt_min in
  let logx_max = Option.map log x_data.opt_max in
  let logy_min = Option.map log y_data.opt_min in
  let logy_max = Option.map log y_data.opt_max in
  let bounds = (logx_min, logy_min, logx_max, logy_max) in
  let logx_array = ODN.(map log_or_min x_data.data |> flatten) in
  let logy_array = ODN.(map log_or_min y_data.data |> flatten) in
  (* map NAN to min float *)
  let n_hist_bins   = Option.default t.props.n_hist_bins                   n_hist_bins in
  let n_pdf_points  = Option.default t.props.n_pdf_points                  n_pdf_points in
  let kernel        = Option.default t.props.marginal_distbn_kde_kernel    kernel in
  let bandwidth     = Option.default t.props.marginal_distbn_kde_bandwidth bandwidth in
  let uv_distbn     = Univariate_distribution.create ~n_hist_bins ~n_pdf_points ~kernel ~bandwidth ~logx_array ~logy_array bounds in
  Univariate_distribution.compute_kde uv_distbn t.props pocl;
  (*uv_distbn.find_modes()
  uv_distbn.statistics()
  uv_distbn.choose_threshold()
   *)
  Some uv_distbn

(**  [compute_marginal_distribn_dsla t pocl sla_arrays slc_arrays]
 *)
let compute_marginal_distribn_dsla t pocl sla_arrays slc_arrays =
  pv_verbose t (fun _ -> Printf.printf "Computing marginal distribution 'dsla'...\n%!");
  t.mpdf_dsla <- compute_marginal_distribn t pocl sla_arrays.(0) slc_arrays.(0);
  pv_verbose t (fun _ -> Printf.printf "Done\n%!");
  ()
        
(**  [compute_marginal_distribn_usla t pocl sla_arrays slc_arrays]
 *)
let compute_marginal_distribn_usla t pocl sla_arrays slc_arrays =
  pv_verbose t (fun _ -> Printf.printf "Computing marginal distribution 'usla'...\n%!");
  t.mpdf_usla <- compute_marginal_distribn t pocl sla_arrays.(1) slc_arrays.(1);
  pv_verbose t (fun _ -> Printf.printf "Done\n%!");
  ()
        
(**  [compute_marginal_distribn_dslt t pocl slt_arrays sla_arrays]
 *)
let compute_marginal_distribn_dslt t pocl slt_arrays sla_arrays =
  pv_verbose t (fun _ -> Printf.printf "Computing marginal distribution 'dslt'...\n%!");
  t.mpdf_usla <- compute_marginal_distribn t pocl slt_arrays.(0) sla_arrays.(0);
  pv_verbose t (fun _ -> Printf.printf "Done\n%!");
  ()
        
(**  [compute_marginal_distribn_uslt t pocl slt_arrays sla_arrays]
 *)
let compute_marginal_distribn_uslt t pocl slt_arrays sla_arrays =
  pv_verbose t (fun _ -> Printf.printf "Computing marginal distribution 'uslt'...\n%!");
  t.mpdf_uslt <- compute_marginal_distribn t pocl slt_arrays.(1) sla_arrays.(1);
  pv_verbose t (fun _ -> Printf.printf "Done\n%!");
  ()
        
(**  [compute_marginal_distribn_dslc t pocl slc_arrays sla_arrays]
 *)
let compute_marginal_distribn_dslc t pocl slc_arrays sla_arrays =
  pv_verbose t (fun _ -> Printf.printf "Computing marginal distribution 'dslc'...\n%!");
  t.mpdf_dslc <- compute_marginal_distribn t pocl slc_arrays.(0) sla_arrays.(0);
  pv_verbose t (fun _ -> Printf.printf "Done\n%!");
  ()
        
(**  [compute_marginal_distribn_uslc t pocl slc_arrays sla_arrays]
 *)
let compute_marginal_distribn_uslc t pocl slc_arrays sla_arrays =
  pv_verbose t (fun _ -> Printf.printf "Computing marginal distribution 'uslc'...\n%!");
  t.mpdf_uslc <- compute_marginal_distribn t pocl slc_arrays.(1) sla_arrays.(1);
  pv_verbose t (fun _ -> Printf.printf "Done\n%!");
  ()
        
(**  [compute_joint_distribn_dsla_usla t sla_arrays]
 *)
let compute_joint_distribn_dsla_usla t pocl sla_arrays =
  pv_verbose t (fun _ -> Printf.printf "Computing joint distribution 'dsla_usla'...\n%!");
(*  compute_joint_distribn t sla_arrays.(0) sla_arrays.(1);*)
  pv_verbose t (fun _ -> Printf.printf "Done\n%!");
  ()

(**  [create props] *)
let create props =
  let area_correction_factor = 1.0 in
  let length_correction_factor = 1.0 in
  {
    props=props.analysis;
    area_correction_factor;
    length_correction_factor;
    mpdf_dsla = None;
    mpdf_usla = None;
    mpdf_dslc = None;
    mpdf_uslc = None;
    mpdf_dslt = None;
    mpdf_uslt = None;
  }

(**  [process  t data results]

  Analyze streamline count, length distbns etc, generate stats and pdfs
 *)
let process t pocl data results = 
  Workflow.workflow_start t.props.workflow;

  let dist_data     = Dist_data.create in
  let dist_data_int = Dist_data.create_int in
  let sla_arrays = [|dist_data (ODN.slice_left results.sla_array [|0|]) t.props.pdf_sla_min t.props.pdf_sla_max;
                     dist_data (ODN.slice_left results.sla_array [|1|]) t.props.pdf_sla_min t.props.pdf_sla_max;
                   |] in (* slice_left so it does not copy *)
  let slt_arrays = [|dist_data (ODN.slice_left results.slt_array [|0|]) t.props.pdf_slt_min t.props.pdf_slt_max;
                     dist_data (ODN.slice_left results.slt_array [|1|]) t.props.pdf_slt_min t.props.pdf_slt_max;
                   |] in (* slice_left so it does not copy *)
  let slc_arrays = [|dist_data_int (ODN.slice_left results.slc_array [|0|]) t.props.pdf_slc_min t.props.pdf_slc_max;
                     dist_data_int (ODN.slice_left results.slc_array [|1|]) t.props.pdf_slc_min t.props.pdf_slc_max;
                   |] in (* slice_left so it does not copy *)

  if t.props.do_marginal_distbn_dsla then compute_marginal_distribn_dsla t pocl sla_arrays slc_arrays;
  if t.props.do_marginal_distbn_usla then compute_marginal_distribn_usla t pocl sla_arrays slc_arrays;
  if t.props.do_marginal_distbn_dslt then compute_marginal_distribn_dslt t pocl slt_arrays sla_arrays;
  if t.props.do_marginal_distbn_uslt then compute_marginal_distribn_uslt t pocl slt_arrays sla_arrays;
  if t.props.do_marginal_distbn_dslc then compute_marginal_distribn_dslc t pocl slc_arrays sla_arrays;
  if t.props.do_marginal_distbn_uslc then compute_marginal_distribn_uslc t pocl slc_arrays sla_arrays;
(*        if self.do_joint_distribn_dsla_usla: self.compute_joint_distribn_dsla_usla()
        if self.do_joint_distribn_usla_uslt: self.compute_joint_distribn_usla_uslt()
        if self.do_joint_distribn_dsla_dslt: self.compute_joint_distribn_dsla_dslt()
        if self.do_joint_distribn_uslt_dslt: self.compute_joint_distribn_uslt_dslt()
        if self.do_joint_distribn_usla_uslc: self.compute_joint_distribn_usla_uslc()
        if self.do_joint_distribn_dsla_dslc: self.compute_joint_distribn_dsla_dslc()
        if self.do_joint_distribn_uslc_dslc: self.compute_joint_distribn_uslc_dslc()
 *)
  Workflow.workflow_end t.props.workflow;
  ()
