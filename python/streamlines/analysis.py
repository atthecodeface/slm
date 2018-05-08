"""
Tools to compute statistical distributions (pdfs) and model their properties
"""

import numpy as np
from sklearn.neighbors import KernelDensity
from scipy.stats import gaussian_kde, norm
from scipy.signal import argrelextrema
from scipy.ndimage import median_filter, gaussian_filter, maximum_filter
from os import environ
environ['PYTHONUNBUFFERED']='True'

from streamlines.core import Core
from streamlines import kde

__all__ = ['Info','Univariate_distribution','Bivariate_distribution','Analysis']

pdebug = print

class Info():    
    def __init__(self, kde, is_one_d=True):
        
        self.debug =         np.bool8(kde.debug)
        self.kdf_bandwidth = np.float32(kde.bandwidth)
        self.kdf_kernel =    kde.kernel
        self.n_data =        np.uint32(kde.n_data)
        self.n_hist_bins =   np.uint32(kde.n_hist_bins)
        self.n_pdf_points =  np.uint32(kde.n_pdf_points)
        self.x_min =         np.float32(kde.logx_min)
        self.x_max =         np.float32(kde.logx_max)
        self.x_range =       np.float32(kde.x_range)
        self.bin_dx =        np.float32(kde.bin_dx)
        self.pdf_dx =        np.float32(kde.pdf_dx)
        self.kdf_width_x =   np.float32(0.0)
        self.n_kdf_part_points_x = np.uint32(0)
        
        if is_one_d:
            self.y_min =         np.float32(0.0)
            self.y_max =         np.float32(1.0)
            self.y_range =       np.float32(1.0)
            self.bin_dy =        np.float32(1.0/2000)
            self.pdf_dy =        np.float32(1.0/200)
            self.kdf_width_y =         np.float32(0.0)
            self.n_kdf_part_points_y = np.uint32(0)
        else:
            self.y_min =         np.float32(kde.logy_min)
            self.y_max =         np.float32(kde.logy_max)
            self.y_range =       np.float32(kde.y_range)
            self.bin_dy =        np.float32(kde.bin_dy)
            self.pdf_dy =        np.float32(kde.pdf_dy)
            self.kdf_width_y =         np.float32(0.0)
            self.n_kdf_part_points_y = np.uint32(0)
            
class Univariate_distribution():
    """
    Class for making and recording kernel-density estimate of univariate 
    probability distribution f(x) data and metadata. 
    Provides a method to find the modal average: x | max{f(x}.
    """
    def __init__(self, logx_array=None, logy_array=None, pixel_size=None,
                 method='sklearn', 
                 n_hist_bins=2000, n_pdf_points=200, 
                 search_cdf_min=0.95,
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
        self.n_data = self.logx_data.shape[0]
        self.n_hist_bins = n_hist_bins
        self.n_pdf_points = n_pdf_points
        self.logx_vec \
            = np.linspace(self.logx_min,self.logx_max,self.n_pdf_points) \
                        .reshape((self.n_pdf_points,1))
        self.x_vec = np.exp(self.logx_vec)
        self.dlogx = self.logx_vec[1]-self.logx_vec[0]
        
        raw_keys = [
            'raw_mean',
            'raw_stddev',
            'raw_var'
            ]
        kde_keys = [
            'model',
            'method',
            'bw_method',
            'pdf',
            'cdf',
            'mode_i',
            'mode_x',
            'biased_mode_i',
            'biased_mode_x',
            'bimode_i',
            'bimode_x',
            'channel_threshold_i',
            'channel_threshold_x',
            'mean',
            'stddev',
            'var'
            ]
        self.raw = dict.fromkeys(raw_keys)
        self.kde = dict.fromkeys(kde_keys)        
        self.kde['method'] = method
        self.search_cdf_min = search_cdf_min
        
        self.pixel_size = pixel_size
        self.cl_src_path = cl_src_path
        self.cl_platform = cl_platform
        self.cl_device = cl_device
        
        self.info_dict = {} 
        self.info = None      
             
        self.debug = debug
        self.verbose = verbose

    def print(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def compute_kde_scipy(self, bw_method='scott'):
        self.kde['bw_method'] = bw_method
        self.kde['model'] \
            = gaussian_kde(self.logx_data.reshape((self.n_data_x,)), 
                           bw_method=self.kde['bw_method'])
        self.kde['pdf'] \
            = self.kde['model'].pdf(self.x_vec.reshape((self.x_vec.shape[0],)))
        self.kde['pdf'] = self.kde['pdf'].reshape((self.x_vec.shape[0],1))
        self.kde['cdf'] = np.cumsum(self.kde['pdf'])*self.dlogx
        if not np.isclose(self.kde['cdf'][-1], 1.0, rtol=5e-3):
            self.print(
                'Error/imprecision when computing cumulative probability distribution:',
                       'pdf integrates to {:3f} not to 1.0'.format(self.kde['cdf'][-1]))
                               
    def compute_kde_sklearn(self, kernel='gaussian', bandwidth=0.15):
        self.kernel = kernel
        self.bandwidth = bandwidth
        # defaults.json specifies a Gaussian, but in practice, when deducing a 
        # channel threshold from the multi-modal pdf of dslt, an Epanechnikov kernel
        # gives a noisy pdf for the same bandwidth. So this is a hack
        # to ensure consistency of channel threshold estimation with either kernel,
        # by forcing the Epanechnikov bandwidth to be double the Gaussian.
        if kernel=='epanechnikov':
            self.bandwidth *= 2.0
#         pdebug(self.logx_data.shape,self.x_vec.shape)
        self.kde['model'] = KernelDensity(kernel=self.kernel, 
                                 bandwidth=self.bandwidth).fit(self.logx_data)
        # Exponentiation needed here because of the (odd) way sklearn generates
        # log pdf values in its score_samples() method
        self.kde['pdf'] \
            = np.exp(self.kde['model'].score_samples(self.logx_vec)) \
                                         .reshape((self.n_pdf_points,1))
        self.kde['cdf'] = np.cumsum(self.kde['pdf'])*self.dlogx
        if not np.isclose(self.kde['cdf'][-1], 1.0, rtol=5e-3):
            self.print(
                'Error/imprecision when computing cumulative probability distribution:',
                       'pdf integrates to {:3f} not to 1.0'.format(self.kde['cdf'][-1]))
#         pdebug('kde array:', self.logx_vec.shape, 
#                'logx min max:', np.min(self.logx_data),np.max(self.logx_data),
#                'logxvec:', self.logx_vec[0],self.logx_vec[-1],
#                self.logx_data.shape, self.kde['pdf'].shape)

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
        self.x_range = self.logx_max-self.logx_min;
        self.bin_dx = self.x_range/self.n_hist_bins
        self.pdf_dx = self.x_range/self.n_pdf_points
        self.info = Info(self, is_one_d=True)
        self.info_dict = {
            'debug' :         np.bool8(self.debug),
            'kdf_bandwidth' : np.float32(self.bandwidth),
            'kdf_kernel' :    self.kernel,
            'n_data' :        np.uint32(self.n_data),
            'n_hist_bins' :   np.uint32(self.n_hist_bins),
            'n_pdf_points' :  np.uint32(self.n_pdf_points),
            'x_min' :         np.float32(self.logx_min),
            'x_max' :         np.float32(self.logx_max),
            'x_range' :       np.float32(self.x_range),
            'bin_dx' :        np.float32(self.bin_dx),
            'pdf_dx' :        np.float32(self.pdf_dx),
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
        self.kde['pdf'] = kde.estimate_univariate_pdf(self.cl_src_path, 
                                                        self.cl_platform, 
                                                        self.cl_device, 
                                                        self.info,
                                                        self.logx_data, 
                                                        self.verbose)
        self.kde['cdf'] = np.cumsum(self.kde['pdf'])*self.bin_dx
        if not np.isclose(self.kde['cdf'][-1], 1.0, rtol=5e-3):
            self.print(
                'Error/imprecision when computing cumulative probability distribution:',
                       'pdf integrates to {:3f} not to 1.0'.format(self.kde['cdf'][-1]))

    def statistics(self):
        x = self.logx_vec
        pdf = self.kde['pdf']
        mean = (np.sum(x*pdf)/np.sum(pdf))
        variance = (np.sum( (x-mean)**2 * pdf)/np.sum(pdf))
        self.kde['mean'] = np.exp(mean)
        self.kde['stddev'] = np.exp(np.sqrt(variance))
        self.kde['var'] = np.exp(variance)
        self.raw_mean = np.exp(self.logx_data.mean())
        self.raw_stddev = np.exp(self.logx_data.std())
        self.raw_var = np.exp(self.logx_data.var())
        self.print('raw mean:  {:.2f}'.format(self.raw_mean), end='')
        self.print(' sigma:  {:.2f}'.format(self.raw_stddev), end='')
        self.print(' var:  {:.2f}'.format(self.raw_var))
        self.print('kde mean:  {:.2f}'.format(self.kde['mean']), end='')
        self.print(' sigma:  {:.2f}'.format(self.kde['stddev']), end='')
        self.print(' var:  {:.2f}'.format(self.kde['var']))

    def find_modes(self):
        x = self.x_vec
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

    def choose_threshold(self):
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


class Bivariate_distribution():
    """
    Container class for kernel-density-estimated bivariate probability distribution
    f(x,y) data and metadata. Also has methods to find the modal average (xm,ym)
    and to find the cluster of points surrounding the mode given a pdf threshold
    and bounding criteria.
    """
    def __init__(self, logx_array=None,logy_array=None, mask_array=None,
                 pixel_size=None, 
                 method='sklearn', n_hist_bins=2000, n_pdf_points=200, 
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
        self.n_data = self.logxy_data.shape[0]
        self.n_hist_bins = n_hist_bins
        self.n_pdf_points = n_pdf_points
                                             
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
        self.cl_src_path = cl_src_path
        self.cl_platform = cl_platform
        self.cl_device = cl_device  
             
        self.debug = debug
        self.verbose = verbose
        
        self.info_dict = {}      
        self.info = None  

    def print(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

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

    def compute_kde_opencl(self, kernel='epanechnikov', bandwidth=0.10):
        self.kernel = kernel
        self.bandwidth = bandwidth
        available_kernels = ['tophat','triangle','epanechnikov','cosine','gaussian']
        if self.kernel not in available_kernels:
            raise ValueError('PDF kernel "{}" is not among those available: {}'
                             .format(self.kernel, available_kernels))
        self.x_range = self.logx_max-self.logx_min;
        self.bin_dx = self.x_range/self.n_hist_bins
        self.pdf_dx = self.x_range/self.n_pdf_points
        self.y_range = self.logy_max-self.logy_min;
        self.bin_dy = self.y_range/self.n_hist_bins
        self.pdf_dy = self.y_range/self.n_pdf_points

        self.info = Info(self, is_one_d=False)        
        self.info_dict = {
            'debug' :         np.bool8(self.debug),
            'kdf_bandwidth' : np.float32(self.bandwidth),
            'kdf_kernel' :    self.kernel,
            'n_data' :        np.uint32(self.n_data),
            'n_hist_bins' :   np.uint32(self.n_hist_bins),
            'n_pdf_points' :  np.uint32(self.n_pdf_points),
            'x_min' :         np.float32(self.logx_min),
            'x_max' :         np.float32(self.logx_max),
            'x_range' :       np.float32(self.x_range),
            'bin_dx' :        np.float32(self.bin_dx),
            'pdf_dx' :        np.float32(self.pdf_dx),
            'kdf_width_x' :   np.float32(0.0),
            'n_kdf_part_points_x' : np.uint32(0),
            'y_min' :         np.float32(self.logy_min),
            'y_max' :         np.float32(self.logy_max),
            'y_range' :       np.float32(self.y_range),
            'bin_dy' :        np.float32(self.bin_dy),
            'pdf_dy' :        np.float32(self.pdf_dy),
            'kdf_width_y' :         np.float32(0.0),
            'n_kdf_part_points_y' : np.uint32(0)
        }
        self.kde['pdf'] = kde.estimate_bivariate_pdf(   self.cl_src_path, 
                                                        self.cl_platform, 
                                                        self.cl_device, 
                                                        self.info,
                                                        self.logxy_data, 
                                                        self.verbose  )
        self.kde['cdf'] = np.cumsum(self.kde['pdf'])*self.bin_dx
#         if not np.isclose(self.kde['cdf'][-1], 1.0, rtol=5e-3):
#             self.print(
#                 'Error/imprecision when computing cumulative probability distribution:',
#                        'pdf integrates to {:3f} not to 1.0'.format(self.kde['cdf'][-1]))


    def find_mode(self, mode_idx, tilt=0):
        # Prep
        kde_pdf = self.kde['pdf']*np.power(self.y_mesh,tilt)
        
        # Find mode = (x,y) @ max{f(x,y)}
        max_pdf_idx = np.argmax(kde_pdf,axis=None)
        mode_ij = np.unravel_index(max_pdf_idx, kde_pdf.shape)
#         pdebug('mode_ij',mode_ij,self.x_mesh.shape,self.x_mesh.shape)
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
                pdebug('Error: Guessing channel threshold')
                self.channel_threshold = 2*self.kde['mode_xy_list'][1][0]
            try:
                near_mode_ij \
                    = np.where( (self.y_mesh>=self.channel_threshold)
                                & (kde_pdf>=mode_max*mode_threshold))
            except:
                pdebug('Error: Guessing mode')
                near_mode_ij \
                    = np.where( (self.y_mesh>=0.1)
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
                        np.array(np.where(  (self.logx_data>=np.log(nmpz[0][0]))
                                          & (self.logx_data<=np.log(nmpz[0][1]))
                                          & (self.logy_data>=np.log(nmpz[1][0])) 
                                          & (self.logy_data<=np.log(nmpz[1][1]))  )).T
                        for nmpz in near_mode_pdf_zones
               ])     
    
        
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

        if self.do_marginal_distbn_dsla:
            self.compute_marginal_distribn_dsla()
            
        # No clear what the purpose of this was...
#         if self.do_marginal_distbn_dslt:
#             self.compute_marginal_distribn_dslt()
#         self.area_correction_factor   =  1/self.mpdf_dslt.kde['var']
#         self.length_correction_factor =  1/self.mpdf_dsla.kde['var']
#             
#         self.trace.slt_array \
#             = self.trace.slt_array*( self.area_correction_factor
#                                            /self.length_correction_factor )
            
        if self.do_marginal_distbn_dslt:
            self.compute_marginal_distribn_dslt()
        if self.do_marginal_distbn_usla:
            self.compute_marginal_distribn_usla()
        if self.do_marginal_distbn_uslt:
            self.compute_marginal_distribn_uslt()
        if self.do_marginal_distbn_dslc:
            self.compute_marginal_distribn_dslc()
        if self.do_marginal_distbn_uslc:
            self.compute_marginal_distribn_uslc()
#    
        if self.do_joint_distribn_dsla_usla:
            self.compute_joint_distribn_dsla_usla()
        if self.do_joint_distribn_usla_uslt:
            self.compute_joint_distribn_usla_uslt()
        if self.do_joint_distribn_dsla_dslt:
            self.compute_joint_distribn_dsla_dslt()
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
        logx_array = x_array[:,:,up_down_idx_x].astype(dtype=np.float32)
        logy_array = y_array[:,:,up_down_idx_y].astype(dtype=np.float32)
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
        uv_distbn.choose_threshold()
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
        logx_array = x_array[:,:,up_down_idx_x].astype(dtype=np.float32)
        logy_array = y_array[:,:,up_down_idx_y].astype(dtype=np.float32)
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
                                          upstream_modal_length=upstream_modal_length,
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
                                          upstream_modal_length=upstream_modal_length,
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
        
