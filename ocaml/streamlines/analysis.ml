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

(*a Libraries *)
open Globals
open Properties
open Core
module ODM = Owl.Dense.Matrix.Generic
module ODN = Owl.Dense.Ndarray.Generic
module OS  = Owl.Stats
module Option = Batteries.Option

(*t t *)
type t = {
    props : t_props_analysis;
    mutable area_correction_factor : float;
    mutable length_correction_factor : float;
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

(*a Useful functions *)
let log_or_min x = if x<=0. then min_float else log x

(*a Univariate distribution
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

type t = {
    verbose    : bool;
    pixel_size : float;
    logx_data  : t_ba_floats;
    n_samples : int;
    logx_vec  : t_ba_floats;
    x_vec : t_ba_floats;
    search_cdf_min : float;
    pdf : t_ba_floats;
    cdf : t_ba_floats;
  }
  (*f create
    logx_array must have only non-NaN elements
    logy_array must have only non-NaN elements

   *)
  let create ?pixel_size:(pixel_size=1.0)
        ?n_samples:(n_samples=100)
        ?search_cdf_min:(search_cdf_min=0.95)
        ?logx_min ?logy_min ?logx_max ?logy_max
        ~logx_array ~logy_array =
    let logx_min = Option.default_delayed (fun _ -> ODN.min' logx_array) logx_min in
    let logy_min = Option.default_delayed (fun _ -> ODN.min' logy_array) logy_min in
    let logx_max = Option.default_delayed (fun _ -> ODN.max' logx_array) logx_max in
    let logy_max = Option.default_delayed (fun _ -> ODN.max' logy_array) logy_max in

    let f i logx =
     let logy = ODN.get logy_array [|i|] in
     ((logx>=logx_min) && (logx<=logx_max) && (logy>=logy_min) && (logy<=logy_max)) in
    let logx_data = filtered_array f logx_array in
    let logx_vec  = ODN.linspace Bigarray.float32 logx_min logx_max n_samples in (* .reshape((self.n_samples,1)) *)
    let x_vec     = ODN.exp logx_vec in
    let verbose = true in
    {
      verbose;
      pixel_size;
      logx_data;
      n_samples;
      logx_vec;
      x_vec;
      search_cdf_min;
      pdf = ba_float2d 1 1;
      cdf = ba_float2d 1 1;
    }

  (*f compute_kde
  let compute_kde ?bw_method:(m="scott") t =
        self.kernel = kernel
        self.bandwidth = bandwidth
        # hack
        if kernel=='gaussian':
            self.bandwidth /= 2.0
        available_kernels = ['tophat','triangle','epanechnikov','cosine','gaussian']
        if kernel not in available_kernels:
            raise ValueError('PDF kernel "{}" is not among those available: {}'
                             .format(kernel, available_kernels))
        x_range = self.logx_max-self.logx_min;
        bin_dx = x_range/self.n_hist_bins
        pdf_dx = x_range/self.n_pdf_points
        self.info_dict = {
            'kdf_bandwidth' : np.float32(self.bandwidth),
            'kdf_kernel' :    kernel,
            'n_data' :        np.uint32(self.n_data),
            'n_hist_bins' :   np.uint32(self.n_hist_bins),
            'n_pdf_points' :  np.uint32(self.n_pdf_points),
            'x_min' :         np.float32(self.logx_min),
            'x_max' :         np.float32(self.logx_max),
            'x_range' :       np.float32(x_range),
            'bin_dx' :        np.float32(bin_dx),
            'pdf_dx' :        np.float32(pdf_dx),
            'kdf_width_x' :   np.float32(0.0),
            'n_kdf_part_points_x' : np.uint32(0),
            'y_min' :         np.float32(0.0),
            'y_max' :         np.float32(1.0),
            'y_range' :       np.float32(1.0),
            'bin_dy' :        np.float32(1.0/2000),
            'pdf_dy' :        np.float32(1.0/200),
            'kdf_width_y' :         np.float32(0.0),
            'n_kdf_part_points_y' : np.uint32(0)
        }
        t.pdf <- Kde.estimate_univariate_pdf pocl data logx_data;
        t.cdf <- ODM.(mul_scalar_ (cumsum t.pdf) bin_dx);
        let should_be_one = ODN.get t.cdf (n_data-1) in
        if (abs_float (should_be_one -. 1.)) > 5e-3 then (
           Printf.printf "Error/imprecision when computing cumulative probability distribution:\npdf integrates to %3f not to 1.0\n%!" should_be_one);
    ()
 *)  
type t_stats = {
  raw_mean   : float;
  raw_stddev : float;
  raw_var    : float;
  mean     : float;
  stddev   : float;
  variance : float;
   }                             
  (*f statistics *)
    let statistics t =
      let x   = t.logx_vec in
      let pdf = t.pdf in
      let log_mean     = ODN.(get (cumsum (mul x pdf)) [|1|]) in (* / sum pdf *)
      let log_variance = ODN.(get (cumsum (mul pdf (sqr (sub_scalar x log_mean)))) [|1|]) in (* / sum pdf *)
      let mean      = exp log_mean in
      let stddev    = exp (sqrt log_variance) in
      let variance  = exp log_variance in

      let raw_mean   = exp (OS.mean (ODN.to_array t.logx_data)) in
      let raw_stddev = exp (OS.std  (ODN.to_array t.logx_data)) in
      let raw_var    = exp (OS.var  (ODN.to_array t.logx_data)) in

      let show_statistics t = 
        Printf.printf "raw mean:  %.2f\n" t.raw_mean;
        Printf.printf " sigma:  %.2f\n"   t.raw_stddev;
        Printf.printf " var:  %.2f\n"     t.raw_var;
        Printf.printf "kde mean:  %.2f\n" t.mean;
        Printf.printf " sigma:  %.2f\n"   t.stddev;
        Printf.printf " var:  %.2f\n"     t.variance;
        ()
        in
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
  (*f choose_threshold *)
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

  (*f compute_marginal_distribution *)
  let compute_marginal_distribn pixel_size =
(*(self, x_array,y_array,mask_array=None,shear_factor=0.0,
                                  up_down_idx_x=0, up_down_idx_y=0, n_samples=None, 
                                  kernel=None, bandwidth=None, method=None,
                                  logx_min=None, logy_min=None, 
                                  logx_max=None, logy_max=None):*)
(*
    let t = create ~pixel_size ~n_samples ~shear_factor in
        if method is None: method = 'sklearn'
    let n_samples = Option.default self.marginal_distbn_kde_nx_samples n_samples in
    if kernel is None: kernel = self.marginal_distbn_kde_kernel
    if bandwidth is None: bandwidth = self.marginal_distbn_kde_bandwidth  
            
        uv_distbn = Univariate_distribution(logx_array=logx_array, logy_array=logy_array,
                                            method=method, n_samples=n_samples,
                                            shear_factor=shear_factor, 
                                            logx_min=logx_min, logy_min=logy_min, 
                                            logx_max=logx_max, logy_max=logy_max,
                                            pixel_size = self.geodata.roi_pixel_size,
                                            search_cdf_min = self.search_cdf_min,
                                            verbose=self.state.verbose)
        if method=='sklearn':
            uv_distbn.compute_kde_sklearn(kernel=kernel, bandwidth=bandwidth)
        elif method=='scipy':
            uv_distbn.compute_kde_scipy(bw_method=self.marginal_distbn_kde_bw_method)
        else:
            raise NameError('KDE method "{}" not recognized'.format(method))
        uv_distbn.find_modes()
        uv_distbn.statistics()
        uv_distbn.choose_threshold()
        return uv_distbn
 *)
()

  (*f All done *)   
end

(*m Bivariate_distribution

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

(*a Toplevel of analysis module

    Class providing statistics & probability tools to analyze streamline data and its
    probability distributions.

 *)
let create props =
  let area_correction_factor = 1.0 in
  let length_correction_factor = 1.0 in
  {
    props=props.analysis;
    area_correction_factor;
    length_correction_factor;
  }

(**  [compute_marginal_distribn]
 *)
let compute_marginal_distribn info ?n_hist_bins ?n_pdf_points ?kernel ?bandwidth x_array y_array up_down_idx_x up_down_idx_y =
  let logx_array = ODN.(map log_or_min (slice_left x_array [|up_down_idx_x|])) in
  let logy_array = ODN.(map log_or_min (slice_left y_array [|up_down_idx_y|])) in
  (* map NAN to min float *)
  let n_hist_bins  = Option.default (Info.int_of info "n_hist_bins")                    n_hist_bins in
  let n_pdf_points = Option.default (Info.int_of info "n_pdf_points")                   n_pdf_points in
  let kernel       = Option.default (Info.int_of info "marginal_distbn_kde_kernel")     kernel in
  let bandwidth    = Option.default (Info.int_of info "marginal_distbn_kde_bandwidth")  bandwidth in
(*  let uv_distbn    = Univariate_distribution.create ~logx_array ~logy_array
                                            n_hist_bins n_pdf_points bounds
                                            search_cdf_min
  in
  uv_distbn.compute_kde_opencl kernel bandwidth;
 *)
  (*uv_distbn.find_modes()
  uv_distbn.statistics()
  uv_distbn.choose_threshold()
   *)
(*
  uv_distbn
 *)
  ()
   
let compute_marginal_distribn_dsla t data results =
  pv_verbose t (fun _ -> Printf.printf "Computing marginal distribution 'dsla'...\n%!");
  let up_down_idx_x = 0 in
  let up_down_idx_y = 0 in
  let x_array = ODN.slice_left results.sla_array [|up_down_idx_x|] in (* slice_left so it does not copy *)
  let y_array = ODN.slice_left results.slc_array [|up_down_idx_y|] in (* mapped to floats in *)
  let mask_array = data.basin_mask_array in
  let logx_min = Option.map log t.props.pdf_sla_min in
  let logx_max = Option.map log t.props.pdf_sla_max in
  let logy_min = Option.map log t.props.pdf_slc_min in
  let logy_max = Option.map log t.props.pdf_slc_max in
  let bounds = (logx_min, logx_min, logy_max, logy_max) in
  (*t.mpdf_dsla <- compute_marginal_distribn x_array y_array mask_array bounds;*)
  pv_verbose t (fun _ -> Printf.printf "Done\n%!");
  ()
        
(*f process 

  Analyze streamline count, length distbns etc, generate stats and pdfs
 *)
let process t data results = 
  Workflow.workflow_start t.props.workflow;

  if t.props.do_marginal_distbn_dsla then compute_marginal_distribn_dsla t data results;
(*
        if self.do_marginal_distbn_dslt: self.compute_marginal_distribn_dslt()

    t.area_correction_factor   <-  1. /. self.mpdf_dslt.kde['var']
    t.length_correction_factor <-  1. /. self.mpdf_dsla.kde['var']
            
    let scale_factor = t.area_correction_factor /. t.length_correction_factor in
    ODN.mul_scalar_ t.trace.slt_array scale_factor;
            
        if self.do_marginal_distbn_dslt: self.compute_marginal_distribn_dslt()
        if self.do_marginal_distbn_usla: self.compute_marginal_distribn_usla()
        if self.do_marginal_distbn_uslt: self.compute_marginal_distribn_uslt()
        if self.do_marginal_distbn_dslc: self.compute_marginal_distribn_dslc()
        if self.do_marginal_distbn_uslc: self.compute_marginal_distribn_uslc()
   
        if self.do_joint_distribn_dsla_usla: self.compute_joint_distribn_dsla_usla()
        if self.do_joint_distribn_usla_uslt: self.compute_joint_distribn_usla_uslt()
        if self.do_joint_distribn_dsla_dslt: self.compute_joint_distribn_dsla_dslt()
        if self.do_joint_distribn_uslt_dslt: self.compute_joint_distribn_uslt_dslt()
        if self.do_joint_distribn_usla_uslc: self.compute_joint_distribn_usla_uslc()
        if self.do_joint_distribn_dsla_dslc: self.compute_joint_distribn_dsla_dslc()
        if self.do_joint_distribn_uslc_dslc: self.compute_joint_distribn_uslc_dslc()
 *)
  Workflow.workflow_end t.props.workflow;
  ()
(*
      

    def compute_marginal_distribn(self, x_array,y_array,mask_array=None,shear_factor=0.0,
                                  up_down_idx_x=0, up_down_idx_y=0, n_samples=None, 
                                  kernel=None, bandwidth=None, method=None,
                                  logx_min=None, logy_min=None, 
                                  logx_max=None, logy_max=None):
        """
        TBD
        """
        logx_array = x_array[:,:,up_down_idx_x].copy().astype(dtype=np.float32)
        logy_array = y_array[:,:,up_down_idx_y].copy().astype(dtype=np.float32)
        logx_array[logx_array>0.0] = np.log(logx_array[logx_array>0.0])
        logy_array[logy_array>0.0] = np.log(logy_array[logy_array>0.0])
        logx_array[x_array[:,:,up_down_idx_x]<=0.0] = np.finfo(np.float32).min
        logy_array[y_array[:,:,up_down_idx_y]<=0.0] = np.finfo(np.float32).min   
        if method is None:
            method = 'sklearn'
        if n_samples is None:
            n_samples = self.marginal_distbn_kde_nx_samples
        if kernel is None:
            kernel = self.marginal_distbn_kde_kernel
        if bandwidth is None:
            bandwidth = self.marginal_distbn_kde_bandwidth  
            
        uv_distbn = Univariate_distribution(logx_array=logx_array, logy_array=logy_array,
                                            method=method, n_samples=n_samples,
                                            shear_factor=shear_factor, 
                                            logx_min=logx_min, logy_min=logy_min, 
                                            logx_max=logx_max, logy_max=logy_max,
                                            pixel_size = self.geodata.roi_pixel_size,
                                            search_cdf_min = self.search_cdf_min,
                                            verbose=self.state.verbose)
        if method=='sklearn':
            uv_distbn.compute_kde_sklearn(kernel=kernel, bandwidth=bandwidth)
        elif method=='scipy':
            uv_distbn.compute_kde_scipy(bw_method=self.marginal_distbn_kde_bw_method)
        else:
            raise NameError('KDE method "{}" not recognized'.format(method))
        uv_distbn.find_modes()
        uv_distbn.statistics()
        uv_distbn.choose_threshold()
        return uv_distbn
   
    def compute_marginal_distribn_dsla(self):
        """
        TBD
        """
        self.print('Computing marginal distribution "dsla"...')
        x_array, y_array = self.trace.sla_array, self.trace.slc_array
        mask_array = self.geodata.basin_mask_array
        up_down_idx_x, up_down_idx_y = 0, 0
        (logx_min, logx_max, logy_min, logy_max) \
          = self._get_logminmaxes(['pdf_sla_min','pdf_sla_max',
                                   'pdf_slc_min','pdf_slc_max'])
        self.mpdf_dsla \
            = self.compute_marginal_distribn(x_array,y_array, mask_array,
                                             up_down_idx_x=up_down_idx_x,
                                             up_down_idx_y=up_down_idx_y,
                                             logx_min=logx_min,logy_min=logy_min, 
                                             logx_max=logx_max,logy_max=logy_max,
                                             shear_factor=0.0)
        self.print('...done')            
        
    def compute_marginal_distribn_usla(self):
        """
        TBD
        """
        self.print('Computing marginal distribution "usla"...')
        x_array, y_array = self.trace.sla_array, self.trace.slc_array
        mask_array = self.geodata.basin_mask_array
        up_down_idx_x, up_down_idx_y = 1, 1
        (logx_min, logx_max, logy_min, logy_max) \
          = self._get_logminmaxes(['pdf_sla_min','pdf_sla_max',
                                   'pdf_slc_min','pdf_slc_max'])
        self.mpdf_usla \
            = self.compute_marginal_distribn(x_array,y_array, mask_array,
                                             up_down_idx_x=up_down_idx_x,
                                             up_down_idx_y=up_down_idx_y,
                                             logx_min=logx_min,logy_min=logy_min, 
                                             logx_max=logx_max,logy_max=logy_max,
                                             shear_factor=0.0)
        self.print('...done')            
        
    def compute_marginal_distribn_dslt(self):
        """
        TBD
        """
        self.print('Computing marginal distribution "dslt"...')
        x_array, y_array = self.trace.slt_array, self.trace.sla_array
        mask_array = self.geodata.basin_mask_array
        up_down_idx_x, up_down_idx_y = 0, 0
        (logx_min, logx_max, logy_min, logy_max) \
          = self._get_logminmaxes(['pdf_slt_min','pdf_slt_max',
                                   'pdf_sla_min','pdf_sla_max'])
        self.mpdf_dslt \
            = self.compute_marginal_distribn(x_array,y_array, mask_array,
                                             up_down_idx_x=up_down_idx_x,
                                             up_down_idx_y=up_down_idx_y,
                                             logx_min=logx_min,logy_min=logy_min, 
                                             logx_max=logx_max,logy_max=logy_max,
                                             shear_factor=0.0)
        self.print('...done')            
        
    def compute_marginal_distribn_uslt(self):
        """
        TBD
        """
        self.print('Computing marginal distribution "uslt"...')
        x_array, y_array = self.trace.slt_array, self.trace.sla_array
        mask_array = self.geodata.basin_mask_array
        up_down_idx_x, up_down_idx_y = 1, 1
        (logx_min, logx_max, logy_min, logy_max) \
          = self._get_logminmaxes(['pdf_slt_min','pdf_slt_max',
                                   'pdf_sla_min','pdf_sla_max'])
        self.mpdf_uslt \
            = self.compute_marginal_distribn(x_array,y_array, mask_array,
                                             up_down_idx_x=up_down_idx_x,
                                             up_down_idx_y=up_down_idx_y,
                                             logx_min=logx_min,logy_min=logy_min, 
                                             logx_max=logx_max,logy_max=logy_max,
                                             shear_factor=0.0)
        self.print('...done')            

    def compute_marginal_distribn_dslc(self):
        """
        TBD
        """
        self.print('Computing marginal distribution "dslc"...')
        x_array, y_array = self.trace.slc_array, self.trace.sla_array
        mask_array = self.geodata.basin_mask_array
        up_down_idx_x, up_down_idx_y = 0, 0
        (logx_min, logx_max, logy_min, logy_max) \
          = self._get_logminmaxes(['pdf_slc_min','pdf_slc_max',
                                   'pdf_sla_min','pdf_sla_max'])
        self.mpdf_dslc \
            = self.compute_marginal_distribn(x_array,y_array, mask_array,
                                             up_down_idx_x=up_down_idx_x,
                                             up_down_idx_y=up_down_idx_y,
                                             logx_min=logx_min,logy_min=logy_min, 
                                             logx_max=logx_max,logy_max=logy_max,
                                             shear_factor=0.0)
        self.print('...done')            
        
    def compute_marginal_distribn_uslc(self):
        """
        TBD
        """
        self.print('Computing marginal distribution "uslc"...')
        x_array, y_array = self.trace.slc_array, self.trace.sla_array
        mask_array = self.geodata.basin_mask_array
        up_down_idx_x, up_down_idx_y = 1, 1
        (logx_min, logx_max, logy_min, logy_max) \
          = self._get_logminmaxes(['pdf_slc_min','pdf_slc_max',
                                   'pdf_sla_min','pdf_sla_max'])
        self.mpdf_uslc \
            = self.compute_marginal_distribn(x_array,y_array, mask_array,
                                             up_down_idx_x=up_down_idx_x,
                                             up_down_idx_y=up_down_idx_y,
                                             logx_min=logx_min,logy_min=logy_min, 
                                             logx_max=logx_max,logy_max=logy_max,
                                             shear_factor=0.0)
        self.print('...done')            


    def compute_joint_distribn(self, x_array,y_array, mask_array=None, shear_factor=0.0,
                               up_down_idx_x=0, up_down_idx_y=0, n_samples=None, 
                               thresholding_marginal_distbn=None,
                               kernel=None, bandwidth=None, method=None,
                               logx_min=None, logy_min=None, 
                               logx_max=None, logy_max=None, 
                               upstream_modal_length=None,
                               verbose=False):
        """
        TBD
        """
        logx_array = x_array[:,:,up_down_idx_x].copy().astype(dtype=np.float32)
        logy_array = y_array[:,:,up_down_idx_y].copy().astype(dtype=np.float32)
        logx_array[logx_array>0.0] = np.log(logx_array[logx_array>0.0])
        logy_array[logy_array>0.0] = np.log(logy_array[logy_array>0.0])
        logx_array[x_array[:,:,up_down_idx_x]<=0.0] = np.finfo(np.float32).min
        logy_array[y_array[:,:,up_down_idx_y]<=0.0] = np.finfo(np.float32).min   
        if method is None:
            method = 'sklearn'
        if n_samples is None:
            n_samples = np.complex(self.joint_distbn_kde_nxy_samples)
        else:
            n_samples = np.complex(n_samples)
        if kernel is None:
            kernel = self.joint_distbn_kde_kernel
        if bandwidth is None:
            bandwidth = self.joint_distbn_kde_bandwidth  
        bv_distbn = Bivariate_distribution(logx_array=logx_array, logy_array=logy_array,
                                            method=method, n_samples=n_samples,
                                            shear_factor=shear_factor, 
                                            logx_min=logx_min, logy_min=logy_min, 
                                            logx_max=logx_max, logy_max=logy_max,
                                            pixel_size = self.geodata.roi_pixel_size,
                                            verbose=self.state.verbose)
        
        if method=='sklearn':
            bv_distbn.compute_kde_sklearn(kernel=kernel, bandwidth=bandwidth)
        elif method=='scipy':
            bv_distbn.compute_kde_scipy(bw_method=self.joint_distbn_kde_bw_method)
        else:
            raise NameError('KDE method "{}" not recognized'.format(method))

        bv_distbn.find_mode(0)
        bv_distbn.find_mode(1,tilt=self.joint_distbn_mode2_tilt)
        bv_distbn.find_near_mode(0,
                                 mode_threshold=self.joint_distbn_mode_threshold_list[0],
                                 nearness = self.joint_distbn_mode2_nearness_factor)
        bv_distbn.find_near_mode(1, marginal_distbn=thresholding_marginal_distbn,
                                 tilt=self.joint_distbn_mode2_tilt,
                                 mode_threshold=self.joint_distbn_mode_threshold_list[1],
                                 nearness = self.joint_distbn_mode2_nearness_factor,
                                 upstream_modal_length=upstream_modal_length)
        self.print('modes @ {0} , {1}'
              .format(list(np.round(bv_distbn.kde['mode_xy_list'][0],2)),
                      list(np.round(bv_distbn.kde['mode_xy_list'][1],2))) )
        return bv_distbn

    def compute_joint_distribn_dsla_usla(self):
        """
        TBD
        """
        self.print('Computing joint distribution "dsla_usla"...')
        x_array,y_array = self.trace.sla_array,self.trace.sla_array
        mask_array = self.geodata.basin_mask_array
        up_down_idx_x,up_down_idx_y = 0,1
        (logx_min, logx_max, logy_min, logy_max) \
          = self._get_logminmaxes(['pdf_sla_min','pdf_sla_max',
                                     'pdf_sla_min','pdf_sla_max'])
        self.jpdf_dsla_usla \
            = self.compute_joint_distribn(x_array,y_array, mask_array,
                                            up_down_idx_x=up_down_idx_x,
                                            up_down_idx_y=up_down_idx_y,
                                            logx_min=logx_min,logy_min=logy_min, 
                                            logx_max=logx_max,logy_max=logy_max,
                                            verbose=self.state.verbose)
        self.print('...done')

    def compute_joint_distribn_dsla_dslt(self):
        """
        TBD
        """
        self.print('Computing joint distribution "dsla_dslt"...')
        x_array,y_array = self.trace.sla_array,self.trace.slt_array
        mask_array = self.geodata.basin_mask_array
        up_down_idx_x,up_down_idx_y = 0,0
        shear_factor = self.joint_distbn_y_shear_factor
        (logx_min, logx_max, logy_min, logy_max) \
          = self._get_logminmaxes(['pdf_sla_min','pdf_sla_max',
                                     'pdf_slt_min','pdf_slt_max'])
        try:
            upstream_modal_length = self.jpdf_usla_uslt.kde['mode_xy_list'][1][0]
        except:
            upstream_modal_length = None
        try:
            mpdf_dslt = self.mpdf_dslt
        except:
            mpdf_dslt = None
        self.jpdf_dsla_dslt \
            = self.compute_joint_distribn(x_array,y_array, mask_array,
                                          thresholding_marginal_distbn=mpdf_dslt,
                                          up_down_idx_x=up_down_idx_x,
                                          up_down_idx_y=up_down_idx_y,
                                          logx_min=logx_min,logy_min=logy_min, 
                                          logx_max=logx_max,logy_max=logy_max,
                                          shear_factor=shear_factor,
                                          upstream_modal_length=upstream_modal_length,
                                          verbose=self.state.verbose)
        self.print('...done')

    def compute_joint_distribn_usla_uslt(self):
        """
        TBD
        """
        self.print('Computing joint distribution "usla_uslt"...')
        x_array,y_array = self.trace.sla_array,self.trace.slt_array
        mask_array = self.geodata.basin_mask_array
        up_down_idx_x,up_down_idx_y = 1,1
        shear_factor = self.joint_distbn_y_shear_factor
        (logx_min, logx_max, logy_min, logy_max) \
          = self._get_logminmaxes(['pdf_sla_min','pdf_sla_max',
                                     'pdf_slt_min','pdf_slt_max'])
        self.jpdf_usla_uslt \
            = self.compute_joint_distribn(x_array,y_array, mask_array,
                                            up_down_idx_x=up_down_idx_x,
                                            up_down_idx_y=up_down_idx_y,
                                            logx_min=logx_min,logy_min=logy_min, 
                                            logx_max=logx_max,logy_max=logy_max,
                                            shear_factor=shear_factor,
                                            verbose=self.state.verbose)
        self.print('...done')

    def compute_joint_distribn_uslt_dslt(self):
        """
        TBD
        """
        self.print('Computing joint distribution "uslt_dslt"...',flush=True)
        x_array,y_array = self.trace.slt_array,self.trace.slt_array
        up_down_idx_x, up_down_idx_y = 1,0
        (logx_min, logx_max, logy_min, logy_max) \
          = self._get_logminmaxes(['pdf_slt_min','pdf_slt_max',
                                     'pdf_slt_min','pdf_slt_max'])
        self.jpdf_uslt_dslt \
            = self.compute_joint_distribn(x_array,y_array,
                                            up_down_idx_x=up_down_idx_x,
                                            up_down_idx_y=up_down_idx_y,
                                            logx_min=logx_min,logy_min=logy_min, 
                                            logx_max=logx_max,logy_max=logy_max,
                                            verbose=self.state.verbose)
        self.print('...done')

    def compute_joint_distribn_dsla_dslc(self):
        """
        TBD
        """
        self.print('Computing joint distribution "dsla_dslc"...')
        x_array,y_array = self.trace.sla_array,self.trace.slc_array
        mask_array = self.geodata.basin_mask_array
        up_down_idx_x,up_down_idx_y = 0,0
        shear_factor = self.joint_distbn_y_shear_factor
        (logx_min, logx_max, logy_min, logy_max) \
          = self._get_logminmaxes(['pdf_sla_min','pdf_sla_max',
                                     'pdf_slc_min','pdf_slc_max'])
        try:
            upstream_modal_length = self.jpdf_usla_uslc.kde['mode_xy_list'][1][0]
        except:
            upstream_modal_length = None
        try:
            mpdf_dslc = self.mpdf_dslc
        except:
            mpdf_dslc = None
        self.jpdf_dsla_dslc \
            = self.compute_joint_distribn(x_array,y_array, mask_array,
                                          thresholding_marginal_distbn=mpdf_dslc,
                                          up_down_idx_x=up_down_idx_x,
                                          up_down_idx_y=up_down_idx_y,
                                          logx_min=logx_min,logy_min=logy_min, 
                                          logx_max=logx_max,logy_max=logy_max,
                                          shear_factor=shear_factor,
                                          upstream_modal_length=upstream_modal_length,
                                          verbose=self.state.verbose)
        self.print('...done')

    def compute_joint_distribn_usla_uslc(self):
        """
        TBD
        """
        self.print('Computing joint distribution "usla_uslc"...')
        x_array,y_array = self.trace.sla_array,self.trace.slc_array
        mask_array = self.geodata.basin_mask_array
        up_down_idx_x,up_down_idx_y = 1,1
        shear_factor = self.joint_distbn_y_shear_factor
        (logx_min, logx_max, logy_min, logy_max) \
          = self._get_logminmaxes(['pdf_sla_min','pdf_sla_max',
                                     'pdf_slc_min','pdf_slc_max'])
        self.jpdf_usla_uslc \
            = self.compute_joint_distribn(x_array,y_array, mask_array,
                                            up_down_idx_x=up_down_idx_x,
                                            up_down_idx_y=up_down_idx_y,
                                            logx_min=logx_min,logy_min=logy_min, 
                                            logx_max=logx_max,logy_max=logy_max,
                                            shear_factor=shear_factor,
                                            verbose=self.state.verbose)
        self.print('...done')

    def compute_joint_distribn_uslc_dslc(self):
        """
        TBD
        """
        self.print('Computing joint distribution "uslc_dslc"...',flush=True)
        x_array,y_array = self.trace.slc_array,self.trace.slc_array
        up_down_idx_x, up_down_idx_y = 1,0
        (logx_min, logx_max, logy_min, logy_max) \
          = self._get_logminmaxes(['pdf_slc_min','pdf_slc_max',
                                     'pdf_slc_min','pdf_slc_max'])
        self.jpdf_uslc_dslc \
            = self.compute_joint_distribn(x_array,y_array,
                                            up_down_idx_x=up_down_idx_x,
                                            up_down_idx_y=up_down_idx_y,
                                            logx_min=logx_min,logy_min=logy_min, 
                                            logx_max=logx_max,logy_max=logy_max,
                                            verbose=self.state.verbose)
        self.print('...done')


    def _get_logminmaxes(self, attr_list):
        return [np.log(getattr(self,attr)) if hasattr(self,attr) else None 
                for attr in attr_list]
        
 *)
