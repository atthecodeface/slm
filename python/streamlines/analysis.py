"""
Tools to compute statistical distributions (pdfs) and model their properties
"""

import numpy as np
from scipy.stats       import gaussian_kde, norm
from scipy.signal      import argrelextrema
from scipy.ndimage     import median_filter, gaussian_filter, maximum_filter
from sklearn.neighbors import KernelDensity
from os import environ
environ['PYTHONUNBUFFERED']='True'

from streamlines.core import Core
from streamlines import kde

__all__ = ['Univariate_distribution','Bivariate_distribution','Analysis']

pdebug = print
            
class Univariate_distribution():
    """
    Class for making and recording kernel-density estimate of univariate 
    probability distribution f(x) data and metadata. 
    Provides a method to find the modal average: x | max{f(x}.
    """
    def __init__(self, logx_array=None, logy_array=None, pixel_size=None,
                 method='opencl', 
                 n_hist_bins=2048, n_pdf_points=256, 
                 search_cdf_min=0.95, search_cdf_max=0.99,
                 logx_min=None, logy_min=None, logx_max=None, logy_max=None,
                 cl_src_path=None, cl_platform=None, cl_device=None,
                 debug=False, verbose=False):
        if logx_min is None:
            logx_min = logx_array[logx_array>np.finfo(np.float32).min].min()
        if logx_max is None:
            logx_max = logx_array[logx_array>np.finfo(np.float32).min].max()
        if logy_min is None:
            logy_min = logy_array[logy_array>np.finfo(np.float32).min].min()
        if logy_max is None:
            logy_max = logy_array[logy_array>np.finfo(np.float32).min].max()      
        self.logx_data = logx_array[  (logx_array>=logx_min) & (logx_array<=logx_max)
                                    & (logy_array>=logy_min) & (logy_array<=logy_max)  ]
        self.logx_data = self.logx_data.reshape((self.logx_data.shape[0],1))
        # Re-estimate min,max since some array values may have just been eliminated
        self.logx_min = np.min(self.logx_data)
        self.logx_max = np.max(self.logx_data)
        self.logx_range = self.logx_max-self.logx_min
        self.logy_min   = 0.0
        self.logy_max   = 0.0
        self.logy_range = 0.0
        self.n_data = self.logx_data.shape[0]
        self.n_hist_bins = n_hist_bins
        self.n_pdf_points = n_pdf_points
        self.bin_dx = self.logx_range/self.n_hist_bins
        self.pdf_dx = self.logx_range/self.n_pdf_points        
        self.bin_dy = 0.0
        self.pdf_dy = 0.0       
        
        self.logx_vec \
            = np.linspace(self.logx_min,self.logx_max,self.n_pdf_points) \
                        .reshape((self.n_pdf_points,1))
        self.logx_vec_histogram \
            = np.linspace(self.logx_min,self.logx_max,self.n_hist_bins) \
                        .reshape((self.n_hist_bins,1))
        self.x_vec = np.exp(self.logx_vec)
        self.dlogx = self.logx_vec[1]-self.logx_vec[0]
 
        self.method = method
        self.search_cdf_min = search_cdf_min
        self.search_cdf_max = search_cdf_max
        
        self.pixel_size = pixel_size
        self.cl_src_path = cl_src_path
        self.cl_platform = cl_platform
        self.cl_device = cl_device
        
        self.debug = debug
        self.verbose = verbose

    def print(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def compute_kde_scipy(self, bw_method='scott'):
        self.kde.bw_method = bw_method
        self.kde.model \
            = gaussian_kde(self.logx_data.reshape((self.n_data_x,)), 
                           bw_method=self.kde.bw_method)
        self.kde.pdf \
            = self.kde.model.pdf(self.x_vec.reshape((self.x_vec.shape[0],)))
        self.kde.pdf = self.kde.pdf.reshape((self.x_vec.shape[0],1))
        self.kde.cdf = np.cumsum(self.kde.pdf)*self.dlogx
        if not np.isclose(self.kde.cdf[-1], 1.0, rtol=5e-3):
            self.print(
                'Error/imprecision when computing cumulative probability distribution:',
                       'pdf integrates to {:3f} not to 1.0'.format(self.kde.cdf[-1]))
                               
    def compute_kde_sklearn(self, kernel='gaussian', bandwidth=0.15):
        self.kernel = kernel
        self.bandwidth = bandwidth
        # defaults.json specifies a Gaussian, but in practice, when deducing a 
        # channel threshold from the multi-modal pdf of dslt, an Epanechnikov kernel
        # gives a noisy pdf for the same bandwidth. So this is a hack
        # to ensure consistency of channel threshold estimation with either kernel,
        # by forcing the Epanechnikov bandwidth to be double the Gaussian.
#         if kernel=='epanechnikov':
#             self.bandwidth *= 2.0
#         pdebug(self.logx_data.shape,self.x_vec.shape)
        self.kde.model = KernelDensity(kernel=self.kernel, 
                                 bandwidth=self.bandwidth).fit(self.logx_data)
        # Exponentiation needed here because of the (odd) way sklearn generates
        # log pdf values in its score_samples() method
        self.kde.pdf \
            = np.exp(self.kde.model.score_samples(self.logx_vec)) \
                                         .reshape((self.n_pdf_points,1))
        self.kde.cdf = np.cumsum(self.kde.pdf)*self.dlogx
        if not np.isclose(self.kde.cdf[-1], 1.0, rtol=5e-3):
            self.print(
                'Error/imprecision when computing cumulative probability distribution:',
                       'pdf integrates to {:3f} not to 1.0'.format(self.kde.cdf[-1]))

    def compute_kde_opencl(self, kernel='epanechnikov', bandwidth=0.15):
        self.kernel = kernel
        self.bandwidth = bandwidth
        # hack
        if self.kernel=='gaussian':
            self.bandwidth /= 2.0
        available_kernels = ['tophat','triangle','epanechnikov','cosine','gaussian']
        if self.kernel not in available_kernels:
            raise ValueError('PDF kernel "{}" is not among those available: {}'
                             .format(self.kernel, available_kernels))
        self.pdf, self.histogram \
            = kde.estimate_univariate_pdf( self )
        self.cdf = np.cumsum(self.pdf)*self.bin_dx
        if not np.isclose(self.cdf[-1], 1.0, rtol=5e-3):
            self.print(
                'Error/imprecision when computing cumulative probability distribution:',
                       'pdf integrates to {:3f} not to 1.0'.format(self.cdf[-1]))
        self.detrended_pdf, self.detrended_histogram \
            = kde.estimate_univariate_pdf( self, do_detrend=True,
                                           logx_vec=self.logx_vec_histogram )

    def statistics(self):
        logx = self.logx_vec
        pdf = self.pdf
        mean = (np.sum(logx*pdf)/np.sum(pdf))
        variance = (np.sum( (logx-mean)**2 * pdf)/np.sum(pdf))
        self.mean = np.exp(mean)
        self.stddev = np.exp(np.sqrt(variance))
        self.var = np.exp(variance)
        self.raw_mean = np.exp(self.logx_data.mean())
        self.raw_stddev = np.exp(self.logx_data.std())
        self.raw_var = np.exp(self.logx_data.var())

    def find_modes(self):
        x = self.x_vec
        pdf = self.pdf
        approx_mode = np.round(x[pdf==pdf.max()][0],2)
        peaks = argrelextrema(np.reshape(pdf,pdf.shape[0],), 
                              np.greater, order=3)[0]
        peaks = [peak for peak in list(peaks) if x[peak]>=approx_mode*0.5 ]
        try:
            self.mode_i = peaks[0]
            self.mode_x = x[peaks[0]][0]
            self.print('Mode @ {0:2.2f}m'.format(self.mode_x))
        except:
            self.print('Failed to find mode')

    def locate_threshold(self):
        x_vec = self.x_vec
        pdf = self.pdf
        cdf = self.cdf
        mode_x = self.mode_x
        detrended_pdf = pdf/norm.pdf(np.log(x_vec),np.log(self.mean),
                                     np.log(self.stddev))
#         detrended_pdf = self.detrended_pdf
        all_extrema_i = argrelextrema(np.reshape(detrended_pdf,pdf.shape[0],), 
                                  np.less, order=3)[0]
        # Choose lowest-cdf minimum given cdf threshold 
        #   - if necessary, progressively lowered until a minimum is found
        search_cdf_min = self.search_cdf_min
        while (search_cdf_min>=0.80):          
            extrema_i = [extremum_i for extremum_i in all_extrema_i 
                         if x_vec[extremum_i]>mode_x \
                            and (cdf[extremum_i]>search_cdf_min
                                 and cdf[extremum_i]<self.search_cdf_max)]
            if len(extrema_i)>=1:
                break
            search_cdf_min -= 0.01
        try:
            self.channel_threshold_i = extrema_i[0]
            self.channel_threshold_x = x_vec[extrema_i[0]][0]
            self.print('Threshold @ cdf={0:0.3}  x={1:2.0f}m'.format(
                cdf[self.channel_threshold_i],self.channel_threshold_x))
        except:
            self.print('Failed to locate threshold: cannot find any minima in range')


class Bivariate_distribution():
    """
    Container class for kernel-density-estimated bivariate probability distribution
    f(x,y) data and metadata. Also has methods to find the modal average (xm,ym)
    and to find the cluster of points surrounding the mode given a pdf threshold
    and bounding criteria.
    """
    def __init__(self, logx_array=None,logy_array=None, mask_array=None,
                 pixel_size=None, 
                 method='opencl', n_hist_bins=2048, n_pdf_points=256, 
                 logx_min=None, logy_min=None, logx_max=None, logy_max=None,
                 cl_src_path=None, cl_platform=None, cl_device=None,
                 debug=False, verbose=False):
        self.logx_data = logx_array
        self.logy_data = logy_array
        if logx_min is None:
            logx_min = logx_array[logx_array>np.finfo(np.float32).min].min()
        if logx_max is None:
            logx_max = logx_array[logx_array>np.finfo(np.float32).min].max()
        if logy_min is None:
            logy_min = logy_array[logy_array>np.finfo(np.float32).min].min()
        if logy_max is None:
            logy_max = logy_array[logy_array>np.finfo(np.float32).min].max()
        if mask_array is not None:
            logxy_data = np.vstack([
                logx_array[  (logx_array>=logx_min) & (logx_array<=logx_max) 
                           & (logy_array>=logy_min) & (logy_array<=logy_max)
                           & (~mask_array)],
                logy_array[  (logx_array>=logx_min) & (logx_array<=logx_max) 
                           & (logy_array>=logy_min) & (logy_array<=logy_max)
                           & (~mask_array)]
                ]).T
        else:
            logxy_data  = np.vstack([
                logx_array[  (logx_array>=logx_min) & (logx_array<=logx_max) 
                           & (logy_array>=logy_min) & (logy_array<=logy_max)],
                logy_array[  (logx_array>=logx_min) & (logx_array<=logx_max) 
                           & (logy_array>=logy_min) & (logy_array<=logy_max)]
                ]).T
        self.logxy_data = logxy_data.copy().astype(dtype=np.float32)
            
        # x,y meshgrid for sampling the bivariate pdf f(x,y)
        # For some weird reason, the numbers of points in x,y need to be complex-valued 
        self.logx_mesh,self.logy_mesh \
            = np.mgrid[logx_min:logx_max:np.complex(n_pdf_points),
                       logy_min:logy_max:np.complex(n_pdf_points)]
        self.logxy_data_indexes = np.vstack([self.logx_mesh.ravel(), 
                                             self.logy_mesh.ravel()]).T
                                             
        self.logx_min = logx_min
        self.logx_max = logx_max
        self.logy_min = logy_min
        self.logy_max = logy_max
        self.logx_range = self.logx_max-self.logx_min
        self.logy_range = self.logy_max-self.logy_min
        self.n_data = self.logxy_data.shape[0]
        self.n_hist_bins = n_hist_bins
        self.n_pdf_points = n_pdf_points
        self.bin_dx = self.logx_range/self.n_hist_bins
        self.pdf_dx = self.logx_range/self.n_pdf_points
        self.bin_dy = self.logy_range/self.n_hist_bins
        self.pdf_dy = self.logy_range/self.n_pdf_points

        self.x_mesh = np.exp(self.logx_mesh)
        self.y_mesh = np.exp(self.logy_mesh)
#         self.x_vec = self.x_mesh[:,0]
#         self.y_vec = self.y_mesh[0,:]

        self.method = method
        self.mode_ij_list = [None,None]
        self.mode_xy_list = [None,None]
        self.mode_max_list = [None,None]
        self.near_mode_vec_list = [None,None]
        self.mode_cluster_ij_list = [None,None]
        
        self.pixel_size = pixel_size
        self.cl_src_path = cl_src_path
        self.cl_platform = cl_platform
        self.cl_device = cl_device  
             
        self.debug = debug
        self.verbose = verbose

    def print(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def compute_kde_scipy(self, bw_method='scott'):
        # Compute bivariate pdf
        self.bw_method = bw_method
        self.model = gaussian_kde(self.logxy_data.T, bw_method=bw_method)
        self.pdf = np.reshape( self.model(self.logxy_data_indexes.T
                                                        ),self.logx_mesh.shape)
                                       
    def compute_kde_sklearn(self, kernel='epanechnikov', bandwidth=0.10):
        self.kernel = kernel
        self.bandwidth = bandwidth
        self.model = KernelDensity(kernel=self.kernel, 
                                          bandwidth=self.bandwidth).fit(self.logxy_data)
        self.pdf = np.reshape( 
            np.exp(self.model.score_samples(self.logxy_data_indexes)
                                                        ),self.logx_mesh.shape)  

    def compute_kde_opencl(self, kernel='epanechnikov', bandwidth=0.10):
        self.kernel = kernel
        self.bandwidth = bandwidth
        available_kernels = ['tophat','triangle','epanechnikov','cosine','gaussian']
        if self.kernel not in available_kernels:
            raise ValueError('PDF kernel "{}" is not among those available: {}'
                             .format(self.kernel, available_kernels))

        self.pdf,self.histogram = kde.estimate_bivariate_pdf(self)
        self.cdf = np.cumsum(self.pdf)*self.bin_dx
#         if not np.isclose(self.cdf[-1], 1.0, rtol=5e-3):
#             self.print(
#                 'Error/imprecision when computing cumulative probability distribution:',
#                        'pdf integrates to {:3f} not to 1.0'.format(self.cdf[-1]))

    def find_mode(self):
        # Prep
        kde_pdf = self.pdf.copy()
        # Hack
        kde_pdf[(self.x_mesh<=3.0) | (self.y_mesh<=3.0)]=0.0
        # Find mode = (x,y) @ max{f(x,y)}
        max_pdf_idx = np.argmax(kde_pdf,axis=None)
        mode_ij = np.unravel_index(max_pdf_idx, kde_pdf.shape)
        mode_xy = np.array([ self.x_mesh[mode_ij[0],0],self.y_mesh[0,mode_ij[1]] ])
        
        # Record mode info
        self.mode_max = kde_pdf[ mode_ij[0],mode_ij[1] ]
        self.mode_ij  = mode_ij
        self.mode_xy  = mode_xy
            
        
class Analysis(Core):
    """
    Class providing statistics & probability tools to analyze streamline data and its
    probability distributions.
    """
    def __init__(self,state,imported_parameters,geodata,trace):
        """
        TBD
        """
        super().__init__(state,imported_parameters) 
        self.state = state
        self.geodata = geodata
        self.trace = trace
        self.area_correction_factor = 1.0
        self.length_correction_factor = 1.0

    def do(self):
        """
        Analyze streamline count, length distbns etc, generate stats and pdfs
        """
        self.print('\n**Analysis begin**')  
        self.print('Kernel-density estimating marginal PDFs using "{}" kernels'
                   .format(self.marginal_distbn_kde_kernel))
        self.print('Processing using "{}" method'
                   .format(self.marginal_distbn_kde_method))
        if self.do_marginal_distbn_dsla:
            self.compute_marginal_distribn_dsla()
        if self.do_marginal_distbn_usla:
            self.compute_marginal_distribn_usla()
        if self.do_marginal_distbn_dslt:
            self.compute_marginal_distribn_dslt()
        if self.do_marginal_distbn_uslt:
            self.compute_marginal_distribn_uslt()
        if self.do_marginal_distbn_dslc:
            self.compute_marginal_distribn_dslc()
        if self.do_marginal_distbn_uslc:
            self.compute_marginal_distribn_uslc()
#    
#         self.channel_threshold = self.marginal_distribn_dsla.kde.channel_threshold_x
        self.print('Kernel-density estimating joint PDFs using "{}" kernels'
                   .format(self.joint_distbn_kde_kernel))
        self.print('Processing using "{}" method'
                   .format(self.joint_distbn_kde_method))
        if self.do_joint_distribn_dsla_usla:
            self.compute_joint_distribn_dsla_usla()
        if self.do_joint_distribn_usla_uslt:
            self.compute_joint_distribn_usla_uslt()
        if self.do_joint_distribn_dsla_dslt:
            self.compute_joint_distribn_dsla_dslt()
        if self.do_joint_distribn_dslt_dslc:
            self.compute_joint_distribn_dslt_dslc()
        if self.do_joint_distribn_uslt_dslt:
            self.compute_joint_distribn_uslt_dslt()
        if self.do_joint_distribn_usla_uslc:
            self.compute_joint_distribn_usla_uslc()
        if self.do_joint_distribn_dsla_dslc:
            self.compute_joint_distribn_dsla_dslc()
        if self.do_joint_distribn_uslc_dslc:
            self.compute_joint_distribn_uslc_dslc()

        self.print('**Analysis end**\n')  
      
    def compute_marginal_distribn(self, x_array,y_array,mask_array=None,
                                  up_down_idx_x=0, up_down_idx_y=0, 
                                  n_hist_bins=None, n_pdf_points=None, 
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
            method = self.marginal_distbn_kde_method
        if n_hist_bins is None:
            n_hist_bins = self.n_hist_bins
        if n_pdf_points is None:
            n_pdf_points = self.n_pdf_points
        if kernel is None:
            kernel = self.marginal_distbn_kde_kernel
        if bandwidth is None:
            bandwidth = self.marginal_distbn_kde_bandwidth  
        uv_distbn = Univariate_distribution(logx_array=logx_array, logy_array=logy_array,
                                            pixel_size = self.geodata.roi_pixel_size,
                                            method=method, 
                                            n_hist_bins=n_hist_bins,
                                            n_pdf_points=n_pdf_points,
                                            logx_min=logx_min, logy_min=logy_min, 
                                            logx_max=logx_max, logy_max=logy_max,
                                            search_cdf_min = self.search_cdf_min,
                                            search_cdf_max = self.search_cdf_max,
                                            cl_src_path=self.state.cl_src_path, 
                                            cl_platform=self.state.cl_platform, 
                                            cl_device=self.state.cl_device,
                                            debug=self.state.debug,
                                            verbose=self.state.verbose)
        if method=='opencl':
            uv_distbn.compute_kde_opencl(kernel=kernel, bandwidth=bandwidth)
        elif method=='sklearn':
            uv_distbn.compute_kde_sklearn(kernel=kernel, bandwidth=bandwidth)
        elif method=='scipy':
            uv_distbn.compute_kde_scipy(bw_method=self.marginal_distbn_kde_bw_method)
        else:
            raise NameError('KDE method "{}" not recognized'.format(method))
        uv_distbn.find_modes()
        uv_distbn.statistics()
        uv_distbn.locate_threshold()
        return uv_distbn
   
    def compute_marginal_distribn_dsla(self):
        """
        TBD
        """
        self.print('Computing marginal distribution "dsla"...')
        x_array, y_array = self.trace.sla_array.copy(), self.trace.slc_array.copy(), 
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
                                             logx_max=logx_max,logy_max=logy_max)
        self.print('...done')            
        
    def compute_marginal_distribn_usla(self):
        """
        TBD
        """
        self.print('Computing marginal distribution "usla"...')
        x_array, y_array = self.trace.sla_array.copy(), self.trace.slc_array.copy()
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
                                             logx_max=logx_max,logy_max=logy_max)
        self.print('...done')            
        
    def compute_marginal_distribn_dslt(self):
        """
        TBD
        """
        self.print('Computing marginal distribution "dslt"...')
        x_array, y_array = self.trace.slt_array.copy(), self.trace.sla_array.copy()
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
                                             logx_max=logx_max,logy_max=logy_max)
        self.print('...done')            
        
    def compute_marginal_distribn_uslt(self):
        """
        TBD
        """
        self.print('Computing marginal distribution "uslt"...')
        x_array, y_array = self.trace.slt_array.copy(), self.trace.sla_array.copy()
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
                                             logx_max=logx_max,logy_max=logy_max)
        self.print('...done')            

    def compute_marginal_distribn_dslc(self):
        """
        TBD
        """
        self.print('Computing marginal distribution "dslc"...')
        x_array, y_array = self.trace.slc_array.copy(), self.trace.sla_array.copy()
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
                                             logx_max=logx_max,logy_max=logy_max)
        self.print('...done')            
        
    def compute_marginal_distribn_uslc(self):
        """
        TBD
        """
        self.print('Computing marginal distribution "uslc"...')
        x_array, y_array = self.trace.slc_array.copy(), self.trace.sla_array.copy()
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
                                             logx_max=logx_max,logy_max=logy_max)
        self.print('...done')            

    def compute_joint_distribn(self, x_array,y_array, mask_array=None,
                               up_down_idx_x=0, up_down_idx_y=0, 
                               n_hist_bins=None, n_pdf_points=None, 
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
            method = self.joint_distbn_kde_method
        if n_hist_bins is None:
            n_hist_bins = self.n_hist_bins
        else:
            n_hist_bins = n_hist_bins
        if n_pdf_points is None:
            n_pdf_points = self.n_pdf_points
        if kernel is None:
            kernel = self.joint_distbn_kde_kernel
        if bandwidth is None:
            bandwidth = self.joint_distbn_kde_bandwidth  
        bv_distbn = Bivariate_distribution(logx_array=logx_array, logy_array=logy_array,
                                            pixel_size = self.geodata.roi_pixel_size,
                                            method=method, 
                                            n_hist_bins=n_hist_bins,
                                            n_pdf_points=n_pdf_points,
                                            logx_min=logx_min, logy_min=logy_min, 
                                            logx_max=logx_max, logy_max=logy_max,
                                            cl_src_path=self.state.cl_src_path, 
                                            cl_platform=self.state.cl_platform, 
                                            cl_device=self.state.cl_device,
                                            debug=self.state.debug,
                                            verbose=self.state.verbose)
        
        if method=='opencl':
            bv_distbn.compute_kde_opencl(kernel=kernel, bandwidth=bandwidth)
        elif method=='sklearn':
            bv_distbn.compute_kde_sklearn(kernel=kernel, bandwidth=bandwidth)
        elif method=='scipy':
            bv_distbn.compute_kde_scipy(bw_method=self.joint_distbn_kde_bw_method)
        else:
            raise NameError('KDE method "{}" not recognized'.format(method))

        bv_distbn.find_mode()
        return bv_distbn

    def compute_joint_distribn_dsla_usla(self):
        """
        TBD
        """
        self.print('Computing joint distribution "dsla_usla"...')
        x_array,y_array = self.trace.sla_array.copy(),self.trace.sla_array.copy()
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
        x_array,y_array = self.trace.sla_array.copy(),self.trace.slt_array.copy()
        mask_array = self.geodata.basin_mask_array
        up_down_idx_x,up_down_idx_y = 0,0
        (logx_min, logx_max, logy_min, logy_max) \
          = self._get_logminmaxes(['pdf_sla_min','pdf_sla_max',
                                     'pdf_slt_min','pdf_slt_max'])
        try:
            mpdf_dsla = self.mpdf_dsla
        except:
            mpdf_dsla = None
        self.jpdf_dsla_dslt \
            = self.compute_joint_distribn(x_array,y_array, mask_array,
                                          thresholding_marginal_distbn=mpdf_dsla,
                                          up_down_idx_x=up_down_idx_x,
                                          up_down_idx_y=up_down_idx_y,
                                          logx_min=logx_min,logy_min=logy_min, 
                                          logx_max=logx_max,logy_max=logy_max,
                                          verbose=self.state.verbose)
        self.print('...done')

    def compute_joint_distribn_dslt_dslc(self):
        """
        TBD
        """
        self.print('Computing joint distribution "dslt_dslc"...')
        x_array,y_array = self.trace.slt_array.copy(),self.trace.slc_array.copy()
        mask_array = self.geodata.basin_mask_array
        up_down_idx_x,up_down_idx_y = 0,0
        (logx_min, logx_max, logy_min, logy_max) \
          = self._get_logminmaxes(['pdf_slt_min','pdf_slt_max',
                                     'pdf_slc_min','pdf_slc_max'])
        try:
            mpdf_dsla = self.mpdf_dsla
        except:
            mpdf_dsla = None
        self.jpdf_dslt_dslc \
            = self.compute_joint_distribn(x_array,y_array, mask_array,
                                          thresholding_marginal_distbn=mpdf_dsla,
                                          up_down_idx_x=up_down_idx_x,
                                          up_down_idx_y=up_down_idx_y,
                                          logx_min=logx_min,logy_min=logy_min, 
                                          logx_max=logx_max,logy_max=logy_max,
                                          verbose=self.state.verbose)
        self.print('...done')

    def compute_joint_distribn_usla_uslt(self):
        """
        TBD
        """
        self.print('Computing joint distribution "usla_uslt"...')
        x_array,y_array = self.trace.sla_array.copy(),self.trace.slt_array.copy()
        mask_array = self.geodata.basin_mask_array
        up_down_idx_x,up_down_idx_y = 1,1
        (logx_min, logx_max, logy_min, logy_max) \
          = self._get_logminmaxes(['pdf_sla_min','pdf_sla_max',
                                     'pdf_slt_min','pdf_slt_max'])
        self.jpdf_usla_uslt \
            = self.compute_joint_distribn(x_array,y_array, mask_array,
                                            up_down_idx_x=up_down_idx_x,
                                            up_down_idx_y=up_down_idx_y,
                                            logx_min=logx_min,logy_min=logy_min, 
                                            logx_max=logx_max,logy_max=logy_max,
                                            verbose=self.state.verbose)
        self.print('...done')

    def compute_joint_distribn_uslt_dslt(self):
        """
        TBD
        """
        self.print('Computing joint distribution "uslt_dslt"...',flush=True)
        x_array,y_array = self.trace.slt_array.copy(),self.trace.slt_array.copy()
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
        x_array,y_array = self.trace.sla_array.copy(),self.trace.slc_array.copy()
        mask_array = self.geodata.basin_mask_array
        up_down_idx_x,up_down_idx_y = 0,0
        (logx_min, logx_max, logy_min, logy_max) \
          = self._get_logminmaxes(['pdf_sla_min','pdf_sla_max',
                                     'pdf_slc_min','pdf_slc_max'])
        try:
            mpdf_dsla = self.mpdf_dsla
        except:
            mpdf_dsla = None
        self.jpdf_dsla_dslc \
            = self.compute_joint_distribn(x_array,y_array, mask_array,
                                          thresholding_marginal_distbn=mpdf_dsla,
                                          up_down_idx_x=up_down_idx_x,
                                          up_down_idx_y=up_down_idx_y,
                                          logx_min=logx_min,logy_min=logy_min, 
                                          logx_max=logx_max,logy_max=logy_max,
                                          verbose=self.state.verbose)
        self.print('...done')

    def compute_joint_distribn_usla_uslc(self):
        """
        TBD
        """
        self.print('Computing joint distribution "usla_uslc"...')
        x_array,y_array = self.trace.sla_array.copy(),self.trace.slc_array.copy()
        mask_array = self.geodata.basin_mask_array
        up_down_idx_x,up_down_idx_y = 1,1
        (logx_min, logx_max, logy_min, logy_max) \
          = self._get_logminmaxes(['pdf_sla_min','pdf_sla_max',
                                     'pdf_slc_min','pdf_slc_max'])
        self.jpdf_usla_uslc \
            = self.compute_joint_distribn(x_array,y_array, mask_array,
                                            up_down_idx_x=up_down_idx_x,
                                            up_down_idx_y=up_down_idx_y,
                                            logx_min=logx_min,logy_min=logy_min, 
                                            logx_max=logx_max,logy_max=logy_max,
                                            verbose=self.state.verbose)
        self.print('...done')

    def compute_joint_distribn_uslc_dslc(self):
        """
        TBD
        """
        self.print('Computing joint distribution "uslc_dslc"...',flush=True)
        x_array,y_array = self.trace.slc_array.copy(),self.trace.slc_array.copy()
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
        
