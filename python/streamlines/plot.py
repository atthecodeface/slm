"""
Map and graph plotting
"""
# Notes on potentially useful mpl:
#  plt.style.use('classic')
#  axes[0].axhline,axes[0].axvline
#  data=data obj dict
#  cax=fig.add_axes to add custom color bar; cax=cax in fig.colorbar
#  color map range: vmin, vmax; also norm
#  Use fig, (axes1,axes2) =  to access multiple axes 
#  CartoPy
#  mpl_toolkits.axes_grid1

import matplotlib as mpl
from matplotlib import streamplot
from random import shuffle, seed
from scipy.ndimage import median_filter, gaussian_filter, maximum_filter
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
from matplotlib.pyplot import streamplot
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import colorsys
import numpy as np
from numpy import pi, arctan, arctan2, sin, cos, sqrt
from scipy.stats import norm, gaussian_kde
from scipy.signal import decimate
from os import environ
environ['PYTHONUNBUFFERED']='True'
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    try:
        matplotlib.use('Qt5Agg');  # Worth trying?
    except:
        pass

from streamlines.core import Core

pdebug = print

__all__ = ['Plot']

def force_display(fig):
    """
    TBD
    """
    try:
        fig.canvas.manager.show() 
        # this makes sure that the gui window gets shown
        # if this is needed depends on rcparams, this is just to be safe
        fig.canvas.flush_events() 
        # this make sure that if the event loop integration is not 
        # set up by the gui framework the plot will update
    except:
        pass
    
def gaussian_pdf(x, mu, sigma):
    """
    Gaussian pdf:  y(x) = N(x; mu, sigma)
    where:
        x = ordinate
        N(x) = normal distribution probability density function
        mu = mean
        sigma = standard deviation
    """
    return norm.pdf(x,mu,sigma)

# Generate random colormap
def random_colormap(cmap_name='randomized', n_colors=1000,random_seed=1):
    """
    Create a cmap with a randomized sequence of bright colors
    """
    ru = np.random.uniform
    np.random.seed(random_seed)
    color_palette \
        = [colorsys.hsv_to_rgb(ru(),ru(low=0.9),ru(low=0.9)) for i in range(n_colors)]
    return LinearSegmentedColormap.from_list(cmap_name, color_palette, N=n_colors)

class Plot(Core):       
    """
    Plot maps and distributions.
    """
    def __init__(self,state, 
                 imported_parameters,
                 geodata,preprocess,trace,analysis,mapping):
        """
        TBD
        """
        super().__init__(state,imported_parameters)  
        self.geodata = geodata
        self.preprocess = preprocess
        self.trace = trace
        self.analysis = analysis
        self.mapping = mapping
        self.figs = {}
        
    def _new_figure(self, title=None, window_title=None, pdf=False, 
                    x_pixel_scale=1,y_pixel_scale=1):
        """
        TBD
        """
        mpl.rc( 'savefig', dpi=75)
        mpl.rc( 'figure', autolayout=False,  titlesize='Large',dpi=75)
        mpl.rc( 'lines', linewidth=2.0, markersize=10)
        # mpl.rc( 'font', size=14,family='Times New Roman', serif='cm')
        # mpl.rc( 'font', size=14,family='DejaVu Sans', serif='cm')
        mpl.rc( 'font', size=14)
        mpl.rc( 'axes', labelsize=14) 
        # mpl.rc( 'legend', fontsize=10)
        mpl.rc( 'text', usetex=False)
        
        if title is None:
            title = self.geodata.title
        if window_title is None:
            window_title = self.state.parameters_file
        try:
            self.figure_count += 1
        except:
            self.figure_count = 1
        if not pdf:
            plot_window_size_factor = self.plot_window_size_factor
        else:
            plot_window_size_factor = self.plot_window_pdf_size_factor
        fig, axes = plt.subplots( 
#                         figsize = plt.figaspect(1)*plot_window_size_factor
                        figsize=(self.plot_window_width *plot_window_size_factor,
                                 self.plot_window_height*plot_window_size_factor)
                         )
        fig.canvas.set_window_title(window_title)
#         if self.suptitle_y!=0:
#             fig.suptitle(title, fontweight='bold', y=self.suptitle_y)

        axes.set_title(title, fontsize=14, fontweight='bold')
        
        ticks_x = ticker.FuncFormatter(
            lambda x, pos: '{0:g}'.format(x*x_pixel_scale) )
        axes.xaxis.set_major_formatter(ticks_x)
        ticks_y = ticker.FuncFormatter(
            lambda y, pos: '{0:g}'.format(y*y_pixel_scale) )
        axes.yaxis.set_major_formatter(ticks_y)

        return fig, axes
          
    def _record_fig(self,fig_name,fig):
        self.print('Recording figure "{}"'.format(fig_name))
        self.figs.update({fig_name : fig})

    @staticmethod
    def show():
        plt.show()

    def do(self):
        """
        Display all output
        """
        self.print('\n**Plot all begin**') 
        if self.do_plot_maps:
            self.plot_maps()
        if self.do_plot_distributions:
            self.plot_distributions()
            self.plot_hillslope_distributions()

        self.print('**Plot all end**\n')  
        
    def plot_maps(self):
        """
        Plot maps of DTM ROI streamlines and processed grids
        """
        self.print('Plotting maps...')
        # Hillshade view of source DTM with correct axis labeling
        if self.do_plot_dtm:
            self.plot_dtm_shaded_relief()
        # Hillshade view of roi with correct axis labeling of pixels 
        #   but not meters (for non 1m pixels)
        if self.do_plot_roi:
            self.plot_roi_shaded_relief()
        # Streamlines, points  on semi-transparent shaded relief
        if self.do_plot_streamlines:
            self.plot_streamlines()
        # Try to map channels etc
        if self.do_plot_flow_maps:
            self.plot_flow_maps()
        if self.do_plot_segments:
            self.plot_segments()
        if self.do_plot_channels:
            self.plot_channels()
        if self.do_plot_hillslope_lengths:
            self.plot_hillslope_lengths()
        if self.do_plot_hillslope_lengths_contoured:
            self.plot_hillslope_lengths_contoured()
        self.print('...done')
            
    def plot_dtm_shaded_relief(self):
        """
        Hillshade view of source DTM
        """
        fig,axes = self._new_figure(x_pixel_scale=self.geodata.roi_pixel_size,
                                    y_pixel_scale=self.geodata.roi_pixel_size)
        dtm_hillshade_array = self.generate_hillshade(self.geodata.dtm_array,
                                                      self.hillshade_azimuth,
                                                      self.hillshade_angle)
        axes.imshow(dtm_hillshade_array, cmap='Greys', 
                  extent=(0,self.geodata.dtm_array.shape[0],
                          0,self.geodata.dtm_array.shape[1]))
        force_display(fig)
        self._record_fig('dtm_shaded_relief',fig)
    
    def plot_roi_shaded_relief(self, interp_method=None):
        """
        Hillshade view of ROI of DTM
        """
        fig,axes = self._new_figure(x_pixel_scale=self.geodata.roi_pixel_size,
                                    y_pixel_scale=self.geodata.roi_pixel_size)
        self.plot_roi_shaded_relief_overlay(axes, 
                                    color_alpha=self.shaded_relief_color_alpha,
                                    hillshade_alpha=self.shaded_relief_hillshade_alpha,
                                    interp_method=interp_method)
        force_display(fig)
        self._record_fig('roi_shaded_relief',fig)

    def plot_roi_shaded_relief_overlay(self, axes, 
                                       do_plot_color_relief=None,
                                       color_alpha=None, hillshade_alpha=None,
                                       interp_method=None):
        """
        Hillshade view of ROI of DTM (overlay method)
        """
        if self.geodata.do_basin_masking:                
            mask_array = self.geodata.basin_mask_array[
                self.geodata.pad_width:-self.geodata.pad_width,
                self.geodata.pad_width:-self.geodata.pad_width]
        else:
            mask_array = np.zeros_like(self.geodata.roi_array)

        try:
            self.roi_hillshade_array
        except:
            self.roi_hillshade_array = self.generate_hillshade(self.geodata.roi_array,
                                                               self.hillshade_azimuth,
                                                               self.hillshade_angle)
        if interp_method is None:
            interp_method = self.plot_interpolation_method
            
#         rounded_pixel_sf = (self.geodata.roi_pixel_size
#                             /np.round(self.geodata.roi_pixel_size))
#         rounded_bounds = [*self.geodata.roi_x_bounds/rounded_pixel_sf,
#                           *self.geodata.roi_y_bounds/rounded_pixel_sf]
#         pdebug('rounded_bounds',rounded_bounds)
        
        if do_plot_color_relief is None:
            do_plot_color_relief = self.do_plot_color_shaded_relief
        if do_plot_color_relief:
            if self.geodata.do_basin_masking:                
                masked_array = np.ma.masked_array(self.geodata.roi_array,
                                                  mask=mask_array)
            else:
                masked_array = self.geodata.roi_array
            axes.imshow(np.fliplr(masked_array).T,
                      cmap=self.terrain_cmap, 
                      extent=[*self.geodata.roi_x_bounds,*self.geodata.roi_y_bounds], 
                      alpha=color_alpha,
                      interpolation=interp_method)
        axes.imshow(np.fliplr(self.roi_hillshade_array).T,
                  cmap='Greys', 
                  extent=[*self.geodata.roi_x_bounds,*self.geodata.roi_y_bounds], 
                  alpha=hillshade_alpha,
                  interpolation=interp_method)

    @staticmethod
    def generate_hillshade(array, azimuth_degrees, angle_altitude_degrees):
        """
        Hillshade render DTM topo with light from direction azimuth_degrees and tilt
        angle_attitude_degrees
        """
        x_grad, y_grad = np.gradient(array)
        slope = pi/2.0 - arctan(np.hypot(x_grad,y_grad))
        aspect = arctan2(-x_grad, y_grad)
         # rotate from np array to true orientation
        azimuth_radians = np.deg2rad(azimuth_degrees+180)
        altitude_radians = np.deg2rad(angle_altitude_degrees)
        shadedImage = (sin(altitude_radians)*sin(slope) + cos(altitude_radians)
                       *cos(slope)*cos(azimuth_radians - aspect) )
        return (255*(shadedImage+1))/2

    def plot_streamlines(self):
        """
        Streamlines, points on semi-transparent shaded relief
        """
        fig,axes = self._new_figure(x_pixel_scale=self.geodata.roi_pixel_size,
                                    y_pixel_scale=self.geodata.roi_pixel_size)
        self.plot_roi_shaded_relief_overlay(axes,
                        color_alpha=self.streamline_shaded_relief_color_alpha,
                        hillshade_alpha=self.streamline_shaded_relief_hillshade_alpha)
        if self.do_plot_flow_vectors:
            self.plot_gradient_vector_field_overlay(axes)
        if self.do_plot_downstreamlines:
            self.plot_updownstreamlines_overlay(axes,do_down=True)
        if self.do_plot_upstreamlines:
            self.plot_updownstreamlines_overlay(axes,do_down=False)
        if self.do_plot_seed_points:
            self.plot_seed_points_overlay(axes)
        if self.do_plot_blockages: 
            self.plot_blockages_overlay(axes)
        if self.do_plot_loops:
            self.plot_loops_overlay(axes)
        # Force map limits to match those of the ROI
        #   - turn this off to check for boundary overflow errors in e.g. streamlines
        axes.set_xlim(xmin=self.geodata.roi_x_bounds[0],xmax=self.geodata.roi_x_bounds[1])
        axes.set_ylim(ymin=self.geodata.roi_y_bounds[0],ymax=self.geodata.roi_y_bounds[1])
        force_display(fig)
        self._record_fig('streamlines',fig)
    
    def plot_channels(self):
        try:
            self.mapping.mapping_array
        except: 
            self.print('Channels array not computed')

        cmap = 'bwr'
        fig_name='channels'
        window_title='channels & midslopes'
        do_flip_cmap=False
        do_balance_cmap=True
                    
        fig,axes = self._new_figure(window_title=window_title,
                                    x_pixel_scale=self.geodata.roi_pixel_size,
                                    y_pixel_scale=self.geodata.roi_pixel_size)

        self.plot_roi_shaded_relief_overlay(axes,
                do_plot_color_relief=False, color_alpha=0,
                hillshade_alpha=self.channel_shaded_relief_hillshade_alpha)

        is_channel         = self.trace.is_channel
        is_thinchannel     = self.trace.is_thinchannel
        is_interchannel    = self.trace.is_interchannel
        is_channelhead     = self.trace.is_channelhead
        is_channeltail     = self.trace.is_channeltail
        is_majorconfluence = self.trace.is_majorconfluence
        is_minorconfluence = self.trace.is_minorconfluence
        is_majorinflow     = self.trace.is_majorinflow
        is_minorinflow     = self.trace.is_minorinflow
        is_leftflank       = self.trace.is_leftflank
        is_midslope        = self.trace.is_midslope
        is_ridge           = self.trace.is_ridge
        was_channelhead    = self.trace.was_channelhead
        is_subsegmenthead  = self.trace.is_subsegmenthead
        is_loop            = self.trace.is_loop
        
        basin_mask_array = self.geodata.basin_mask_array
        
        grid_array = (self.mapping.mapping_array & is_thinchannel).copy().astype(np.bool)
        mask_array = basin_mask_array | ~(grid_array)
        self.plot_simple_grid(grid_array, mask_array, axes, cmap='Blues', alpha=0.8)

        grid_array = (self.mapping.mapping_array & is_midslope).copy().astype(np.bool)
        mask_array = basin_mask_array | ~(grid_array)
        self.plot_simple_grid(grid_array, mask_array, axes, cmap='Greens', alpha=0.8)
        
        grid_array = (self.mapping.mapping_array & is_ridge).copy().astype(np.bool)
        mask_array = basin_mask_array | ~(grid_array)
        self.plot_simple_grid(grid_array, mask_array, axes, cmap='Oranges', alpha=0.8)
        
#         self.plot_compound_markers(axes, is_majorconfluence, ['blue','black'])
#         self.plot_compound_markers(axes, is_loop,     ['pink','black'], msf=2)
        self.plot_compound_markers(axes, is_channeltail,     ['cyan','black'], msf=1.5)
#         self.plot_compound_markers(axes, was_channelhead,     ['purple','black'], msf=0.5)
#         self.plot_compound_markers(axes, is_subsegmenthead,  ['orange','black'], msf=0.5)
        self.plot_compound_markers(axes, is_channelhead,     ['red','black'], msf=0.5)
#         self.plot_compound_markers(axes, is_midslope,        ['purple','black'])
#         self.plot_compound_markers(axes, is_ridge,        ['purple','black'])
        
        self._record_fig(fig_name,fig)

    def plot_compound_markers(self, axes, flag, colors, msf=1.0):
        pad = self.geodata.pad_width
        markers_array = (np.flipud(np.argwhere(
                                        self.mapping.mapping_array & flag
                                        ).astype(np.float32)-pad))
        marker = self.channel_head_marker
        marker_sizes = [ms*msf for ms in self.channel_head_marker_sizes]
        colors = colors
        alpha = self.channel_head_marker_alpha
        axes.plot(markers_array[:,0]+self.geodata.roi_x_origin,
                  markers_array[:,1]+self.geodata.roi_y_origin,
                  marker, ms=marker_sizes[1], 
                  color=colors[1], alpha=alpha, fillstyle='full')
        axes.plot(markers_array[:,0]+self.geodata.roi_x_origin,
                  markers_array[:,1]+self.geodata.roi_y_origin,
                  marker, ms=marker_sizes[0], 
                  color=colors[0], alpha=alpha, fillstyle='full')

    def plot_simple_grid(self,grid_array,mask_array,axes,cmap='Blues',alpha=0.8,
                         do_vlimit=True):
        grid_array = grid_array[self.geodata.pad_width:-self.geodata.pad_width,
                                self.geodata.pad_width:-self.geodata.pad_width]    
        mask_array = mask_array[self.geodata.pad_width:-self.geodata.pad_width,
                                self.geodata.pad_width:-self.geodata.pad_width]    
        masked_grid_array = np.ma.masked_array(grid_array, mask=mask_array)
        if do_vlimit:
            im = axes.imshow(np.flipud(masked_grid_array.T), 
                      cmap=cmap, 
                      extent=[*self.geodata.roi_x_bounds,*self.geodata.roi_y_bounds],
                      alpha=alpha,
                      interpolation=self.plot_interpolation_method
                    ,vmin=0, vmax=1
                      )
        else:
            im = axes.imshow(np.flipud(masked_grid_array.T), 
                      cmap=cmap, 
                      extent=[*self.geodata.roi_x_bounds,*self.geodata.roi_y_bounds],
                      alpha=alpha,
                      interpolation=self.plot_interpolation_method
                      )
        clim=im.properties()['clim']
        return im
    
    def plot_flow_maps(self):        
        tmp_array = self.trace.sla_array[:,:,0].copy()
        tmp_array[(~np.isnan(tmp_array)) & (tmp_array>0.0)] \
            = np.sqrt(tmp_array[(~np.isnan(tmp_array)) & (tmp_array>0.0)])
        
        self.plot_gridded_data(tmp_array,
                               'gist_earth', 
                               fig_name='dsla',
                               window_title='dsla',
                               do_flip_cmap=True, do_balance_cmap=False)
        tmp_array = self.trace.slt_array[:,:,0].copy()
        tmp_array[(~np.isnan(tmp_array)) & (tmp_array>0.0)] \
            = np.log(tmp_array[(~np.isnan(tmp_array)) & (tmp_array>0.0)])
        self.plot_gridded_data(tmp_array,
                               'gist_earth', #'seismic', 
                               fig_name='dslt',
                               window_title='dslt',
                               do_flip_cmap=True, do_balance_cmap=False)
        
    def plot_segments(self,do_shaded_relief=True):
        tmp_array = (self.mapping.label_array.copy().astype(np.int32))
        self.plot_gridded_data(tmp_array,
                               'randomized', 
                               fig_name='label',
                               window_title='label',
                               do_shaded_relief=do_shaded_relief, 
                               do_flip_cmap=False, do_balance_cmap=False)

    def plot_hillslope_lengths(self, cmap=None,
                               z_min=None,z_max=None, 
                               do_shaded_relief=None,
                               colorbar_aspect=None):
        if cmap is None:
            cmap = self.plot_hsl_cmap
        if colorbar_aspect is None:
            colorbar_aspect = self.plot_hsl_colorbar_aspect
        if do_shaded_relief is None:
            do_shaded_relief = self.plot_hsl_do_shaded_relief
        if z_min is None:
            if self.plot_hsl_z_min=='full':
                z_min = np.percentile(self.mapping.hillslope_length_smoothed_array, 0.0)
            elif self.plot_hsl_z_min=='auto':
                z_min = np.percentile(self.mapping.hillslope_length_smoothed_array, 1.0)
            else:
                z_min = self.plot_hsl_z_min
        if z_max is None:
            if self.plot_hsl_z_max=='full':
                z_max = np.percentile(self.mapping.hillslope_length_smoothed_array,100.0)  
            elif self.plot_hsl_z_max=='auto':
                z_max = np.percentile(self.mapping.hillslope_length_smoothed_array,99.9)  
            else:
                z_max = self.plot_hsl_z_max     
        grid_array = np.clip(self.mapping.hillslope_length_array,z_min,z_max)
        mask_array = np.zeros_like(grid_array).astype(np.bool)            
        mask_array[self.mapping.label_array==0] = True
        self.plot_gridded_data(grid_array,
                               cmap,  # rainbow
                               mask_array=mask_array,
                               fig_name='hillslope_length',
                               window_title='hillslope length',
                               do_flip_cmap=False, do_balance_cmap=False,
                               do_shaded_relief=do_shaded_relief, 
                               do_colorbar=True, 
                               colorbar_title='mean hillslope length [m]',
                               colorbar_aspect=colorbar_aspect)
    
    def plot_hillslope_distributions(self, x_stretch=None):
        df = self.mapping.hillslope_stats_df
        kde_min_labels = 20
        if x_stretch is None:
            x_stretch = self.mhsl_pdf_x_stretch
            
        name = 'hillslope_mean'
        fig,_ = self._new_figure(window_title=name)
        if df.shape[0]>kde_min_labels:
            axes = df['mean [m]'].plot.density(figsize=(8,8),
                                          title='Mean hillslope length')
        else:
            axes = df['mean [m]'].plot.hist(figsize=(8,8),
                                          title='Mean hillslope length')
        axes.set_xlabel(r'distance $L$  [m]')
        axes.set_ylabel(r'probability density  $f(L)$  [m$^{-1}$]')
        axes.set_xlim(0,df['mean [m]'].quantile(q=1)*x_stretch)
        axes.set_ylim(0,None)
        self._record_fig(name,fig)

        name = 'hillslope_stddev'
        fig,_ = self._new_figure(window_title=name)
        if df.shape[0]>kde_min_labels:
            axes = df['stddev [m]'].plot.density(figsize=(8,8),
                        title='Hillslope length std deviation');
        else:
            axes = df['stddev [m]'].plot.hist(figsize=(8,8),
                        title='Hillslope length std deviation');
        axes.set_xlabel(r'distance std deviation $\sigma_L$  [m]')
        axes.set_ylabel(r'probability density  $f(\sigma_L)$  [m$^{-1}$]')
        axes.set_xlim(0,df['stddev [m]'].quantile(q=0.99))
        axes.set_ylim(0,None)
        # axes.legend(['hillslope length std dev'],frameon=False)
        self._record_fig(name,fig)
        
        name = 'hillslope_count'
        fig,_ = self._new_figure(window_title=name)
        axes = df['count'].plot.hist(bins=20,figsize=(8,8),
                                title='Hillslope streamline count')
        axes.set_xlabel(r'number $N_{sl}$   [-]');
        axes.set_ylabel(r'probability density  $f(N_{sl})$  [-]')
        axes.set_xlim(0,df['count'].quantile(q=0.99))
        axes.set_ylim(0,None)
        self._record_fig(name,fig)

    def plot_gridded_data(self, 
                            grid_array,
                            gridded_cmap, 
                            fig_name=None,
                            mask_array=None,
                            window_title='',
                            do_flip_cmap=False,
                            do_balance_cmap=True,
                            do_shaded_relief=True,
                            do_colorbar=False,
                            colorbar_title=None,
                            colorbar_aspect=0.07):
        """
        TBD
        """
        if mask_array is None:
            mask_array = np.zeros_like(grid_array).astype(np.bool)            
        mask_array = mask_array[self.geodata.pad_width:-self.geodata.pad_width,
                                self.geodata.pad_width:-self.geodata.pad_width]
        if self.geodata.do_basin_masking:
            mask_array |= self.geodata.basin_mask_array[
                self.geodata.pad_width:-self.geodata.pad_width,
                self.geodata.pad_width:-self.geodata.pad_width]
            
        if self.seed_point_marker_size==0:
            seed_point_marker = 'g,'
            seed_point_marker_size = 1
        else:
            seed_point_marker = 'gD'     
            seed_point_marker_size = self.seed_point_marker_size
            
        fig,axes = self._new_figure(window_title=window_title,
                                    x_pixel_scale=self.geodata.roi_pixel_size,
                                    y_pixel_scale=self.geodata.roi_pixel_size)

        if do_shaded_relief:
            self.plot_roi_shaded_relief_overlay(axes, do_plot_color_relief=False,
                    color_alpha=self.grid_shaded_relief_color_alpha,
                    hillshade_alpha=self.grid_shaded_relief_hillshade_alpha)
            grid_alpha = self.streamline_density_alpha
        else:
            grid_alpha = 1.0

        grid_array = grid_array[self.geodata.pad_width:-self.geodata.pad_width,
                                self.geodata.pad_width:-self.geodata.pad_width]
        extra_mask_array = grid_array.astype(np.bool)
        mask_array = mask_array | (~extra_mask_array)

        masked_grid_array = np.ma.masked_array(grid_array, mask=mask_array)
        if do_flip_cmap:
            masked_grid_array = -masked_grid_array
            
        vmin = masked_grid_array[ (~np.isnan(masked_grid_array))
                                 & (~np.isinf(masked_grid_array)) ].min()
        vmax = masked_grid_array[ (~np.isnan(masked_grid_array))
                                 & (~np.isinf(masked_grid_array)) ].max()
        if do_balance_cmap:
            if abs(vmin)>abs(vmax):
                vmax = -vmin
            else:
                vmin = -vmax
        if gridded_cmap=='randomized':
            gridded_cmap = random_colormap(cmap_name='randomized', 
                                               random_seed=self.random_cmap_seed)
        im = axes.imshow(np.flipud(masked_grid_array.T), 
                  cmap=gridded_cmap, 
                  extent=[*self.geodata.roi_x_bounds,*self.geodata.roi_y_bounds],
                  alpha=grid_alpha,
                  interpolation=self.plot_interpolation_method,
                  vmin=vmin, vmax=vmax
                  )
        clim=im.properties()['clim']
        if do_colorbar:
            divider = make_axes_locatable(axes)
            cax = divider.append_axes("bottom", size="4%", pad=0.5,
                                      aspect=colorbar_aspect)
            cbar = plt.colorbar(im, cax=cax, orientation="horizontal")
            cbar.set_label(colorbar_title)
        self._record_fig(fig_name,fig)

    def plot_hillslope_lengths_contoured(self,cmap=None,
                                         do_colorbar=False, 
                                         colorbar_title='mean hillslope length [m]',
                                         n_contours=None,
                                         z_min=None,z_max=None,
                                         do_shaded_relief=None,
                                         colorbar_aspect=None,
                                         contour_label_suffix='m',
                                         contour_label_fontsize=12):
        """
        Streamlines, points on semi-transparent shaded relief
        """
        fig,axes = self._new_figure(x_pixel_scale=self.geodata.roi_pixel_size,
                                    y_pixel_scale=self.geodata.roi_pixel_size)
        if cmap is None:
            cmap = self.contour_hsl_cmap
        if colorbar_aspect is None:
            colorbar_aspect = self.contour_hsl_colorbar_aspect
        if do_shaded_relief is None:
            do_shaded_relief = self.contour_hsl_do_shaded_relief
        if z_min is None:
            if self.contour_hsl_z_min=='full':
                z_min = np.percentile(self.mapping.hillslope_length_smoothed_array, 0.0)
            elif self.contour_hsl_z_min=='auto':
                z_min = np.percentile(self.mapping.hillslope_length_smoothed_array, 1.0)
            else:
                z_min = self.contour_hsl_z_min
        if z_max is None:
            if self.contour_hsl_z_max=='full':
                z_max = np.percentile(self.mapping.hillslope_length_smoothed_array,100.0)  
            elif self.contour_hsl_z_max=='auto':
                z_max = np.percentile(self.mapping.hillslope_length_smoothed_array,99.0)  
            else:
                z_max = self.contour_hsl_z_max     
        grid_array = np.clip(self.mapping.hillslope_length_smoothed_array,z_min,z_max)
        if n_contours is None and self.contour_hsl_n_contours!='auto':
                n_contours = self.contour_hsl_n_contours
                
        mask_array = self.geodata.basin_mask_array[
                        self.geodata.pad_width:-self.geodata.pad_width,
                        self.geodata.pad_width:-self.geodata.pad_width]  
        mask_array[(self.mapping.label_array[
                        self.geodata.pad_width:-self.geodata.pad_width,
                        self.geodata.pad_width:-self.geodata.pad_width]==0)] = True
        if do_shaded_relief:
            hillshade_alpha = self.grid_shaded_relief_hillshade_alpha
            hsl_alpha = 0.3
        else:
            hillshade_alpha = 0.0
            hsl_alpha = 1.0
        self.plot_roi_shaded_relief_overlay(axes, do_plot_color_relief=False,
                                            hillshade_alpha=hillshade_alpha)
        im = self.plot_simple_grid(np.fliplr(grid_array.T),
                              mask_array,axes,cmap=cmap,alpha=hsl_alpha,do_vlimit=False)
        if do_colorbar:
            divider = make_axes_locatable(axes)
            cax = divider.append_axes("bottom", size="4%", 
                                      pad=0.5, aspect=colorbar_aspect)
            cbar = plt.colorbar(im, cax=cax, orientation="horizontal")
            cbar.set_label(colorbar_title)
        self.plot_contours_overlay(axes,np.flipud(grid_array),mask=mask_array,
                                   n_contours=n_contours,
                                   contour_label_suffix=contour_label_suffix, 
                                   contour_label_fontsize=contour_label_fontsize)
        # Force map limits to match those of the ROI
        #   - turn this off to check for boundary overflow errors in e.g. streamlines
        axes.set_xlim(xmin=self.geodata.roi_x_bounds[0],xmax=self.geodata.roi_x_bounds[1])
        axes.set_ylim(ymin=self.geodata.roi_y_bounds[0],ymax=self.geodata.roi_y_bounds[1])
        force_display(fig)
        self._record_fig('hillslope_length_contours',fig)
    
    def plot_contours_overlay(self,axes,Z,mask=None, n_contours=None,
                              contour_label_suffix='m', contour_label_fontsize=12):
        """
        TBD
        """
        x_pixel_centers_array,y_pixel_centers_array \
            = np.meshgrid(self.geodata.x_roi_n_pixel_centers,
                          self.geodata.y_roi_n_pixel_centers)
        Z = np.ma.array(Z,mask=mask.T)
#         z_min = np.percentile(Z, 1.0)
        z_max = np.percentile(Z,100.0)
        if n_contours is None:
            c_interval = 5
            while c_interval<50:
                c_min = np.floor(np.min(Z)/c_interval)*c_interval
                c_max = np.ceil(z_max/c_interval)*c_interval
                c_count = (c_max-c_min)/c_interval
                if c_count<=15:
                    break
                c_interval *= 2
            n_contours = np.arange(c_min,c_max+c_interval,c_interval)
        contours = axes.contour(x_pixel_centers_array,y_pixel_centers_array,Z,
                                n_contours, colors='k')
        axes.clabel(contours, fmt='%.0f'+contour_label_suffix, 
                    fontsize=contour_label_fontsize);

    def plot_updownstreamlines_overlay(self,axes,do_down=True):
        """
        Up or downstreamlines 
        """
        if do_down:
            up_or_down_str = 'down'
            streamline_arrays_list = self.trace.streamline_arrays_list[0]
            color = self.downstreamline_color
            marker = self.streamline_point_marker   
            size = self.streamline_point_size
            alpha = self.streamline_point_alpha
        else:
            up_or_down_str = 'up'
            streamline_arrays_list = self.trace.streamline_arrays_list[1]
            color = self.upstreamline_color
            marker = self.streamline_point_marker   
            size = self.streamline_point_size
            alpha = self.streamline_point_alpha
        
        try:
            n_streamlines = len(streamline_arrays_list)
        except:
            return
        idx_list = list(range(n_streamlines))
        if self.shuffle_random_seed is not None:
            seed(self.shuffle_random_seed)
        if self.plot_streamline_limit!='none':
            shuffle(idx_list)
            idx_list = idx_list[0:min(n_streamlines,self.plot_streamline_limit)]

#         trajectories = [streamline_arrays_list[idx] for idx in idx_list]
        todo = len(idx_list)
        if self.plot_streamline_limit!='none' \
            and self.plot_streamline_limit<n_streamlines:
            self.print('Plotting {0:,}'.format(todo)+' '
                  +up_or_down_str+'streamlines'
                  +' randomly sampled from a set of {0:,}'.format(n_streamlines))
        else:
            self.print('Plotting all {0:,} {1} streamlines'.format(todo,up_or_down_str))
        self.print('Progress: ', end='')
        progress = 0    
        if todo*self.trace.max_length>=300000:
            self.print_interval = max(1,todo//100)
        elif todo*self.trace.max_length>=100000:
            self.print_interval = max(1,todo//30)
        else:
            self.print_interval = max(1,todo//10)
        for sidx in range(todo): 
            trajectory = np.concatenate((np.array([[0,0]],dtype=np.float32),
                        streamline_arrays_list[idx_list[sidx]].astype(np.float32)
                                    /np.float32(self.trace.trajectory_resolution)))
            seed_point = self.trace.seed_point_array[idx_list[sidx],:]
            x_vec = ( self.geodata.roi_x_origin
                      + np.cumsum(trajectory.T[0])
                      + seed_point[0] )
            y_vec = ( self.geodata.roi_y_origin
                      + np.cumsum(trajectory.T[1])
                      + seed_point[1] )

            axes.plot(x_vec,y_vec,
                      marker, 
                      color=color,
                      ms=size, 
                      lw=size,
                      alpha=alpha,
                      fillstyle='full')
    
            progress += 1
            if progress%self.print_interval==0:
                prog = int(100*int(0.5+100*progress/todo)//100)
                self.print(str(prog)+'%',end='') 
                if prog!=100:
                    self.print('...', end='')       
 
        self.print('')    
    
    def plot_classical_streamlines(self):
        """
        Classic streamlines on color shaded relief
        """
        fig,axes = self._new_figure(x_pixel_scale=self.geodata.roi_pixel_size,
                                    y_pixel_scale=self.geodata.roi_pixel_size)
        self.plot_roi_shaded_relief_overlay(axes, 
                        color_alpha=self.streamline_shaded_relief_color_alpha,
                        hillshade_alpha=self.streamline_shaded_relief_hillshade_alpha)
        self.plot_classical_streamlines_overlay(axes)
        axes.set_xlim(xmin=self.geodata.roi_x_bounds[0],xmax=self.geodata.roi_x_bounds[1])
        axes.set_ylim(ymin=self.geodata.roi_y_bounds[0],ymax=self.geodata.roi_y_bounds[1])
        force_display(fig)
        self._record_fig('classical_streamlines',fig)

    def plot_classical_streamlines_and_vectors(self):
        """
        Classic streamlines and gradient vector field on color shaded relief
        """
        fig,axes = self._new_figure(x_pixel_scale=self.geodata.roi_pixel_size,
                                    y_pixel_scale=self.geodata.roi_pixel_size)
        self.plot_roi_shaded_relief_overlay(axes, 
                        color_alpha=self.streamline_shaded_relief_color_alpha,
                        hillshade_alpha=self.streamline_shaded_relief_hillshade_alpha)
        self.plot_gradient_vector_field_overlay(axes,color='darkred')
        self.plot_classical_streamlines_overlay(axes)
        axes.set_xlim(xmin=self.geodata.roi_x_bounds[0],xmax=self.geodata.roi_x_bounds[1])
        axes.set_ylim(ymin=self.geodata.roi_y_bounds[0],ymax=self.geodata.roi_y_bounds[1])
        force_display(fig)
        self._record_fig('classical_streamlines_and_vectors',fig)

    def plot_classical_streamlines_overlay(self,axes):
        """
        Classic streamlines (overlay method)
        """
        x_pixel_centers_array,y_pixel_centers_array \
            = np.meshgrid(self.geodata.x_roi_n_pixel_centers,
                          self.geodata.y_roi_n_pixel_centers)
        axes.streamplot(x_pixel_centers_array,y_pixel_centers_array,
                      np.ma.array(self.preprocess.uv_array[:,:,0],
                                  mask=self.geodata.basin_mask_array).T[
                        self.geodata.pad_width:-self.geodata.pad_width,
                        self.geodata.pad_width:-self.geodata.pad_width],
                      np.ma.array(self.preprocess.uv_array[:,:,1],
                                  mask=self.geodata.basin_mask_array).T[
                        self.geodata.pad_width:-self.geodata.pad_width,
                        self.geodata.pad_width:-self.geodata.pad_width], 
                      density=min(1.0,self.classical_streamplot_density), color='k' )
#                           color=self.geodata.roi_array, cmap='terrain')

    def plot_gradient_vector_field(self):
        """
        Topographic gradient vector field on color shaded relief
        """
        fig,axes = self._new_figure(x_pixel_scale=self.geodata.roi_pixel_size,
                                    y_pixel_scale=self.geodata.roi_pixel_size)
        self.plot_roi_shaded_relief_overlay(axes, 
                        color_alpha=self.streamline_shaded_relief_color_alpha,
                        hillshade_alpha=self.streamline_shaded_relief_hillshade_alpha)
        self.plot_gradient_vector_field_overlay(axes)
        force_display(fig)
        self._record_fig('gradient_vector_field',fig)

    def plot_gradient_vector_field_overlay(self,axes,color='k'):
        """
       Topographic gradient vector field (overlay method)
        """
        max_dim = max(self.geodata.roi_array.shape)
        if self.gradient_vector_scale!='none':
            scale = self.gradient_vector_scale
        else:
            scale=None
        if max_dim>40:
            decimation_factor = 10*int(max_dim//40/10.0+0.5)
            u = decimate( 
                    decimate(
                            self.preprocess.uv_array[:,:,0],
                                decimation_factor,axis=0,n=2), 
                                    decimation_factor,axis=1,n=2)
            v = decimate( 
                    decimate(
                            self.preprocess.uv_array[:,:,1],
                            decimation_factor,axis=0,n=2), 
                                decimation_factor,axis=1,n=2)
            m = decimate( 
                    decimate(
                        ~self.geodata.basin_mask_array,
                            decimation_factor,axis=0,n=2), 
                                decimation_factor,axis=1,n=2)
        else:
            u = self.preprocess.uv_array[:,:,0]
            v = self.preprocess.uv_array[:,:,1]
            m = ~self.geodata.basin_mask_array
        # Normalize for clarity
        speed = np.hypot(u,v)
        u = u/speed
        v = v/speed
        x_roi_n_pixel_centers = np.linspace(self.geodata.roi_x_bounds[0]+0.5,
                                          self.geodata.roi_x_bounds[1]-0.5, 
                                          u.shape[0])
        y_roi_n_pixel_centers = np.linspace(self.geodata.roi_y_bounds[0]+0.5,
                                          self.geodata.roi_y_bounds[1]-0.5, 
                                          u.shape[1])
        axes.quiver(*np.meshgrid(x_roi_n_pixel_centers,y_roi_n_pixel_centers), 
                    (u*m).T, 
                    (v*m).T,
                    pivot='mid',
                    color=self.gradient_vector_color,
                    alpha=self.gradient_vector_alpha
                    ,scale=scale
                    )
            
    def plot_blockages_overlay(self,axes,color='k',shape='s'):
        """
        Blocked zone pixels
        """
        # Blocked outflows (narrow diagonal outflows)
        if self.state.noisy:
            self.print('Blockage @', (self.preprocess.where_blockages_array))
        try:
            self.preprocess.where_blockages_array
        except:
            self.print('Unable to plot blockages: no such array')
            return
        axes.plot(self.preprocess.where_blockages_array.T[0]
                            +self.geodata.roi_x_bounds[0]+0.5,
                  self.preprocess.where_blockages_array.T[1]
                            +self.geodata.roi_y_bounds[0]+0.5,
                  # Blockages plotted as hexagons 'H'
                 'kH', ms=self.blockage_marker_size, fillstyle='none')
        # Blocked outflow neighbors
        if self.state.noisy:
            self.print('Blockage neighbor @', (self.preprocess.where_blocked_neighbors_array))
        axes.plot(self.preprocess.where_blocked_neighbors_array.T[0]
                    +self.geodata.roi_x_bounds[0]+0.5,
                  self.preprocess.where_blocked_neighbors_array.T[1]
                    +self.geodata.roi_y_bounds[0]+0.5,
                # Blockage neighbors plotted as squares 's'
                color+shape, ms=self.blockage_marker_size, 
                fillstyle='none')
 
    def plot_loops_overlay(self,axes,color='k',shape='o'):
        """
        Loop zone pixels
        """
        try:
            self.preprocess.where_looped_array
        except:
            self.print('Unable to plot loops: no such array')
            return
        axes.plot(self.preprocess.where_looped_array.T[0]
                    +self.geodata.roi_x_bounds[0]+0.5,
                  self.preprocess.where_looped_array.T[1]
                    +self.geodata.roi_y_bounds[0]+0.5,
                    # Loops plotted as circles 'o'
                    color+shape, ms=self.loops_marker_size, 
                    fillstyle='none')

    def plot_seed_points_overlay(self,axes,color='k',marker=',',size=2):
        """
        Seed points
        """
        if self.seed_point_marker_size==0:
            marker = marker
            size = size
            color = color
        else:
            marker = self.seed_point_marker   
            size = self.seed_point_marker_size
            color = self.seed_point_marker_color
        
        x = self.trace.seed_point_array[:,0]+self.geodata.roi_x_origin
        y = self.trace.seed_point_array[:,1]+self.geodata.roi_y_origin
        axes.plot(x, y,
                marker, 
                ms=size, 
                color=color,
                alpha=self.seed_point_marker_alpha,
                fillstyle='full')


    def plot_distributions(self):
        """
        Plot probability distributions of processed streamline data.
        """
        self.print('Plotting distributions...')

        # Marginal univariate pdfs
        if self.do_plot_marginal_pdf_dsla:
            self.plot_marginal_pdf_dsla()
        if self.do_plot_marginal_pdf_usla:
            self.plot_marginal_pdf_usla()
        if self.do_plot_marginal_pdf_dslt:
            self.plot_marginal_pdf_dslt()
        if self.do_plot_marginal_pdf_uslt:
            self.plot_marginal_pdf_uslt()
        if self.do_plot_marginal_pdf_dslc:
            self.plot_marginal_pdf_dslc()
        if self.do_plot_marginal_pdf_uslc:
            self.plot_marginal_pdf_uslc()

        # Joint bivariate pdfs
        if self.do_plot_joint_pdf_dsla_usla:
            self.plot_joint_pdf_dsla_usla()
        if self.do_plot_joint_pdf_usla_uslt:
            self.plot_joint_pdf_usla_uslt()
        if self.do_plot_joint_pdf_dsla_dslt:
            self.plot_joint_pdf_dsla_dslt()
        if self.do_plot_joint_pdf_dslt_dslc:
            self.plot_joint_pdf_dslt_dslc()
        if self.do_plot_joint_pdf_dslt_dsla:
            self.plot_joint_pdf_dslt_dsla()
        if self.do_plot_joint_pdf_uslt_dslt:
            self.plot_joint_pdf_uslt_dslt()
        if self.do_plot_joint_pdf_usla_uslc:
            self.plot_joint_pdf_usla_uslc()
        if self.do_plot_joint_pdf_dsla_dslc:
            self.plot_joint_pdf_dsla_dslc()
        if self.do_plot_joint_pdf_uslc_dslc:
            self.plot_joint_pdf_uslc_dslc()

        # Their marginal 1d pdfs

        self.print('...done')

    def plot_marginal_pdf(self, marginal_distbn, fig_name=None,
                          title='',x_label='',y_label=''):
        """
        TBD
        """

        # Get ready
        x_vec = marginal_distbn.x_vec
        x_min,x_max = x_vec[0], x_vec[-1]
        legend = []
        
        # Generate curves
#         legend += ['']
        legend += ['kde pdf']
        pdf = marginal_distbn.pdf
        pdf_list = [ pdf ]
        y_max = pdf_list[0].max()
        pdf_ysf_list = [ y_max ]
        line_colors = ['crimson']
        line_styles = ['-']
        line_alphas = [1]
        
    
                
    
        legend += ['Gaussian']
        loc   = np.log(marginal_distbn.mean)
        scale = np.log(marginal_distbn.stddev)
        pdf   = norm.pdf(np.log(x_vec),loc,scale)
        pdf_max = pdf.max() 
        pdf_ysf_list += [ pdf_max ]
        pdf_list += [ y_max*pdf/pdf_max ]
        line_colors += ['darkmagenta']
        line_styles += ['-.']
        line_alphas += [0.9]
                                 
        legend += ['detrended']
        pdf = marginal_distbn.pdf
        norm_pdf = norm.pdf(np.log(x_vec),loc,scale)
        detrended_pdf = pdf/norm_pdf
#         detrended_pdf = marginal_distbn.detrended_pdf
        dt_min_idx = marginal_distbn.mode_i//2
        try:
            dt_max_idx = (min(1+1*marginal_distbn.mode_i,
                              marginal_distbn.channel_threshold_i))
        except:
            dt_max_idx = (min(2*marginal_distbn.mode_i,detrended_pdf.shape[0]))
        detrended_pdf /= max(detrended_pdf[dt_min_idx:dt_max_idx])
        pdf_max = norm_pdf[marginal_distbn.mode_i]
        pdf_ysf_list +=  [ pdf_max ]
        pdf_list += [ y_max*detrended_pdf ]
        line_colors += ['darkblue']
        line_styles += ['-']
        line_alphas += [1]

#         fig,axes = self._new_figure(title=title)
#         axes.grid(color='gray', linestyle='-')
# #         plt.plot(np.log(x_vec), np.log(pdf),'.')
# #         plt.plot(np.log(x_vec), np.log(x_vec**1.3)-1.8)
# #         plt.plot(np.log(x_vec), np.log(x_vec**1.3)-2.1)
# #         plt.plot(np.log(x_vec), np.log(x_vec**0.75)-3.4)
# #         plt.plot(np.log(x_vec), np.log(x_vec**1.5)-2)
#         axes.set_xscale('log')
# #         axes.set_yscale('log')
#         plt.plot(x_vec, detrended_pdf,'.')
        
#         return
            
        # Create graph
        fig,axes = self._new_figure(title=title)
        axes.set_xscale('log')
        
        # Do the plotting
        plt.plot(x_vec, pdf_list[0], color=line_colors[0], alpha=line_alphas[0], 
                  linestyle=line_styles[0] )
        [plt.plot(x_vec, pdf, color=line_colors[idx+1], alpha=line_alphas[idx+1], 
                  linestyle=line_styles[idx+1])  for idx,pdf in enumerate(pdf_list[1:])]
        
        try:
            x= marginal_distbn.channel_threshold_x
            y= y_max*detrended_pdf[marginal_distbn.channel_threshold_i,0] \
                        /pdf_max
            legend += ['thresh={0:2.0f}m'.format(x)]
            plt.plot([x,x],[-0.1,1.0*axes.get_ylim()[1]],'--', 
                     color='blue', linewidth=3, alpha=0.7)
        except:
            self.print('Cannot plot threshold: none found')
        
        # Presentation
        axes.grid(color='gray', linestyle='dotted', linewidth=0.5, which='both')
        x_ticks = self._choose_ticks(x_min,x_max)
        plt.xticks(x_ticks,x_ticks)
        axes.set_xlabel(x_label)
        axes.set_ylabel(y_label)
        loc = 'upper right'
#         loc = 'upper left'
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            axes.legend(legend, loc=loc,fontsize=14,framealpha=0.97)

        axes.set_xlim(xmin=0.99, xmax=x_max*1.001)
        axes.set_ylim(ymin=0,ymax=y_max*1.1)


        # Display & record
        force_display(fig)
        self._record_fig(fig_name,fig)    
        
    def plot_marginal_pdf_dsla(self):
        """
        TBD
        """
        fig_name = 'marginal_pdf_dsla'
        title = r'Downstreamline length distribution $f(\log(L_{md}))$'
        x_label = r'Downstreamline mean length  $L_{md}$ [meters]'
        y_label = r'Probability density  $f(\log(L_{md}))$'
        try:
            marginal_distbn = self.analysis.mpdf_dsla
        except:
            self.print('"'+title+'" not computed: cannot plot')
            return
        self.plot_marginal_pdf(marginal_distbn, fig_name=fig_name, 
                               title=title, x_label=x_label,y_label=y_label)
        
    def plot_marginal_pdf_usla(self):
        """
        TBD
        """
        fig_name = 'marginal_pdf_usla'
        title = r'Upstreamline length distribution $f(\log(L_{mu}))$'
        x_label = r'Upstreamline mean length  $L_{mu}$ [m]'
        y_label = r'Probability density  $f(\log(L_{mu}))$'
        try:
            marginal_distbn = self.analysis.mpdf_usla
        except:
            self.print('"'+title+'" not computed: cannot plot')
            return
        self.plot_marginal_pdf(marginal_distbn, fig_name=fig_name, 
                               title=title, x_label=x_label,y_label=y_label)
        
    def plot_marginal_pdf_dslt(self):
        """
        TBD
        """
        fig_name = 'marginal_pdf_dslt'
        title = r'Downstreamline root equiv area distribution $f(\log\sqrt{A_{ed}})$'
        x_label = r'Downstreamline root equiv area  $\sqrt{A_{ed}}$ [m]'
        y_label = r'Probability density  $f(\log\sqrt{A_{ed}})$'
        try:
            marginal_distbn = self.analysis.mpdf_dslt
        except:
            self.print('"'+title+'" not computed: cannot plot')
            return
        self.plot_marginal_pdf(marginal_distbn, fig_name=fig_name, 
                               title=title, x_label=x_label,y_label=y_label)
        
    def plot_marginal_pdf_uslt(self):
        """
        TBD
        """
        fig_name = 'marginal_pdf_uslt'
        title = r'Upstreamline root equiv area distribution $f(\log\sqrt{A_{eu}})$'
        x_label = r'Upstreamline root equiv area  $\sqrt{A_{eu}}$ [m]'
        y_label = r'Probability density  $f(\log\sqrt{A_{eu}})$'
        try:
            marginal_distbn = self.analysis.mpdf_uslt
        except:
            self.print('"'+title+'" not computed: cannot plot')
            return
        self.plot_marginal_pdf(marginal_distbn, fig_name=fig_name, 
                               title=title, x_label=x_label,y_label=y_label)

    def plot_marginal_pdf_dslc(self):
        """
        TBD
        """
        fig_name = 'marginal_pdf_dslc'
        title = r'Downstreamline concentration distribution $f(\log(C_{d}))$'
        x_label = r'Downstreamline concentration  $C_{d}$ [lines/m]'
        y_label = r'Probability density  $f(\log(C_{d}))$'
        try:
            marginal_distbn = self.analysis.mpdf_dslc
        except:
            self.print('"'+title+'" not computed: cannot plot')
            return
        self.plot_marginal_pdf(marginal_distbn, fig_name=fig_name, 
                               title=title, x_label=x_label,y_label=y_label)
        
    def plot_marginal_pdf_uslc(self):
        """
        TBD
        """
        fig_name = 'marginal_pdf_uslc'
        title = r'Upstreamline concentration distribution $f(\log(C_{u}))'
        x_label = r'Upstreamline concentration  $C_{u}$ [lines/m]'
        y_label = r'Probability density  $f(\log(C_{u}))$'
        try:
            marginal_distbn = self.analysis.mpdf_uslc
        except:
            self.print('"'+title+'" not computed: cannot plot')
            return
        self.plot_marginal_pdf(marginal_distbn, fig_name=fig_name, 
                               title=title, x_label=x_label,y_label=y_label)

    def plot_joint_pdf(self, bivariate_distribution, mx_distbn=None, my_distbn=None,
                       fig_name=None, title='', swap_xy=False,
                       x_label='', y_label='', 
                       xsym_label = r'$L_m^{*}$', ysym_label = r'$\sqrt{A_e^*}$', 
                       do_plot_mode=True):
        """
        TBD
        """
        # Preparation
        if not swap_xy:
            x_mesh = bivariate_distribution.x_mesh
            y_mesh = bivariate_distribution.y_mesh
            x_vec  = bivariate_distribution.x_mesh[:,0]
            y_vec  = bivariate_distribution.y_mesh[0,:]
        else:
            x_mesh = bivariate_distribution.y_mesh
            y_mesh = bivariate_distribution.x_mesh
            x_vec  = bivariate_distribution.y_mesh[0,:]
            y_vec  = bivariate_distribution.x_mesh[:,0]
        mode_xy = bivariate_distribution.mode_xy
        x_min = x_mesh.min()
        x_max = x_mesh.max()
        y_min = y_mesh.min()
        y_max = y_mesh.max()
        
        # Create mpl figure
        fig,axes = self._new_figure(title=title)
        axes.set_xscale('log')
        axes.set_yscale('log')
        legend = []

        # Pdfs
        if not swap_xy:
            kde_pdf = bivariate_distribution.pdf.copy()
        else:
            kde_pdf = bivariate_distribution.pdf.copy().T
        kde_pdf = np.power(kde_pdf,self.joint_distbn_viz_scale)
        
        # Plot bivariate pdf - distorted for emphasis as appropriate
        axes.pcolormesh(x_vec,y_vec,kde_pdf.T, cmap='GnBu', 
                        antialiased=True, shading='gouraud')
        axes.contour(x_vec,y_vec, kde_pdf.T, self.joint_distbn_n_contours,
                     colors='k',linewidths=1,alpha=0.5, antialiased=True)
        
        # Plot thresholds
        try:
            legend += [r'threshold '+xsym_label
                       +' = {:.0f}m'.format(np.round(
                        mx_distbn.channel_threshold_x,0))]
            legend += [r'threshold '+ysym_label
                       +' = {:.0f}m'.format(np.round(
                        my_distbn.channel_threshold_x,0))]
            if not swap_xy:
                axes.plot(x_vec*0+mx_distbn.channel_threshold_x,y_vec,
                          '--', color='blue',linewidth=3,alpha=1.0)
                axes.plot(x_vec,y_vec*0+my_distbn.channel_threshold_x,
                          ':', color='navy',linewidth=3,alpha=1.0)
            else:
#                 pdebug('swapping xy')
                axes.plot(x_vec*0+mx_distbn.channel_threshold_x,y_vec,
                          ':', color='navy',linewidth=3,alpha=1.0)
                axes.plot(x_vec,y_vec*0+my_distbn.channel_threshold_x,
                          '--', color='blue',linewidth=3,alpha=1.0)
            do_extras = True  
        except:
            print('Problem with channel threshold')
            do_extras = False

        if do_extras:
            # Plot cross @ hillslope mode
            cross_alpha=0.6
            mode_idx = 0
            if not swap_xy:
                mode_xy = mode_xy
            else:
                mode_xy = np.flipud(mode_xy)
            if do_plot_mode:
                (mx,mxc,msx,mewx,mi,msi,mewi,mc,msc,ma) = self.joint_distbn_markers
                if mode_xy is not None:
                    legend += ['_no_legend_']
                    axes.plot(mode_xy[0],mode_xy[1],mi,ms=msx,mew=mewx)
                    legend += ['hillslope mode']
                    axes.plot(mode_xy[0],mode_xy[1],mx,color=mxc,ms=msi,mew=mewi)
                     
            # Linear trend x=y
    #         h_grad = mode_xy[1]/mode_xy[0]
            h_grad = 1.0
            if not swap_xy:
                h_grad_str = '{0:1.1}'.format(h_grad)
                legend += [r'hillslope  $x =$'+'$y $']
                x = x_vec
            else:
                h_grad_str = '{0:0.2}'.format(h_grad)
                legend += [r'hillslope  $x =$'+'$y^{1/3}$']
                x = x_vec
            axes.plot(x,x*h_grad,'-.', color='crimson',linewidth=2,alpha=0.7)
    
            # Nonlinear trend y=x^n
            h_grad = mode_xy[1]/mode_xy[0]
            h_grad = 1.0
            if not swap_xy:
                h_grad_str = '{0:1.1}'.format(h_grad)
                legend += [r'channel  $x =$'+'$y^{3/2} $']
                x = x_vec
                axes.plot(x,np.power(x,1.5),'--', 
                          color='magenta',linewidth=2,alpha=0.7)
            else:
                h_grad_str = '{0:0.2}'.format(h_grad)
                legend += [r'channel  $x =$'+'${y^{1/3}$']
                x = x_vec
                axes.plot(x,np.power(x,2.0/3),'--', #np.power(1.5,0.5)*
                          color='magenta',linewidth=2,alpha=0.7)

        # Presentation
#         loc = 'lower right'
        loc = 'upper left'
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            axes.legend(legend, loc=loc,fontsize=12,framealpha=0.97)
        axes.xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=15))
        axes.yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=15))
        axes.set_xlabel(x_label)
        axes.set_ylabel(y_label)
        x_ticks = self._choose_ticks(x_min,x_max)
        y_ticks = self._choose_ticks(y_min,y_max)
        plt.xticks(x_ticks,x_ticks)
        plt.yticks(y_ticks,y_ticks)
        try:
            axes.set_xlim(xmin=0.999,xmax=x_max*1.001)
            axes.set_ylim(ymin=y_min*0.999,ymax=y_max*1.001)
        except:
            self.print('x,y min,max not provided')
        axes.grid(color='gray', linestyle='dotted', linewidth=0.5, which='both')
        
        # Push to screen
        force_display(fig)
        self._record_fig(fig_name,fig)

    def plot_joint_pdf_dsla_usla(self):
        """
        TBD
        """
        fig_name = 'joint_pdf_dsla_usla'
        title = r'Streamline length distribution$ f(L_{md},L_{mu}$'
        x_label = r'Downstreamline mean length  $L_{md}$ [m]'
        y_label = r'Upstreamline mean length  $L_{mu}$ [m]'
        xsym_label = r'$L_{md}^{*}$'
        ysym_label = r'$L_{mu}^{*}$'
        try:
            joint_distbn = self.analysis.jpdf_dsla_usla
            mx_distbn    = self.analysis.mpdf_dsla
            my_distbn    = self.analysis.mpdf_usla
        except:
            self.print('"'+title+'" not computed: cannot plot')
            return
        self.plot_joint_pdf(joint_distbn, mx_distbn=mx_distbn, my_distbn=my_distbn,
                            fig_name=fig_name, 
                            title=title, x_label=x_label, y_label=y_label,
                            xsym_label=xsym_label, ysym_label=ysym_label)
            
    def plot_joint_pdf_usla_uslt(self):
        """
        TBD
        """
        fig_name = 'joint_pdf_usla_uslt'
        title = r'Upstreamline length distribution $f(\log(L_{mu}),\log\sqrt{A_{eu}})$ '
        x_label = r'Upstreamline mean length  $L_{mu}$ [m]'
        y_label = r'Upstreamline root equiv area  $\sqrt{A_{eu}}$ [m]'
        xsym_label = r'$L_{mu}^{*}$'
        ysym_label = r'$\sqrt{A_{eu}^*}$'
        try:
            joint_distbn = self.analysis.jpdf_usla_uslt
            mx_distbn    = self.analysis.mpdf_usla
            my_distbn    = self.analysis.mpdf_uslt
        except:
            self.print('"'+title+'" not computed: cannot plot')
            return
        self.plot_joint_pdf(joint_distbn, mx_distbn=mx_distbn, my_distbn=my_distbn,
                            fig_name=fig_name,
                            title=title, x_label=x_label, y_label=y_label,
                            xsym_label=xsym_label, ysym_label=ysym_label,
                            do_plot_mode=True)

    def plot_joint_pdf_dsla_dslt(self):
        """
        TBD
        """
        fig_name = 'joint_pdf_dsla_dslt'
        title = r'Downstreamline length distribution $f(\log(L_{md}),\log\sqrt{A_{ed}})$'
        x_label = r'Downstreamline mean length  $L_{md}$ [m]'
        y_label = r'Downstreamline root equiv area  $\sqrt{A_{ed}}$ [m]'
        xsym_label = r'$L_{md}^{*}$'
        ysym_label = r'$\sqrt{A_{ed}^*}$'
        try:
            joint_distbn = self.analysis.jpdf_dsla_dslt
            mx_distbn    = self.analysis.mpdf_dsla
            my_distbn    = self.analysis.mpdf_dslt
        except:
            self.print('"'+title+'" not computed: cannot plot')
            return
        self.plot_joint_pdf(joint_distbn, mx_distbn=mx_distbn, my_distbn=my_distbn,
                            fig_name=fig_name, swap_xy=False,
                            title=title, x_label=x_label, y_label=y_label,
                            xsym_label=xsym_label, ysym_label=ysym_label,
                            do_plot_mode=True)

    def plot_joint_pdf_dslt_dsla(self):
        """
        TBD
        """
        fig_name = 'joint_pdf_dslt_dsla'
        title = r'Downstreamline length distribution $f(\log\sqrt{A_{ed}},\log(L_{md}))$'
        x_label = r'Downstreamline root equiv area  $\sqrt{A_{ed}}$ [m]'
        y_label = r'Downstreamline mean length  $L_{md}$ [m]'
        xsym_label = r'$\sqrt{A_{ed}^*}$'
        ysym_label = r'$L_{md}^{*}$'
        try:
            joint_distbn = self.analysis.jpdf_dsla_dslt
            mx_distbn    = self.analysis.mpdf_dsla
            my_distbn    = self.analysis.mpdf_dslt
        except:
            self.print('"'+title+'" not computed: cannot plot')
            return
        self.plot_joint_pdf(joint_distbn, mx_distbn=mx_distbn, my_distbn=my_distbn,
                            fig_name=fig_name, swap_xy=True,
                            title=title, x_label=x_label, y_label=y_label,
                            xsym_label=xsym_label, ysym_label=ysym_label,
                            do_plot_mode=True)

    def plot_joint_pdf_uslt_dslt(self):
        """
        TBD
        """
        fig_name = 'joint_pdf_uslt_dslt'
        title = r'Streamline root equiv area distribution $f\sqrt{A_{eu}},\sqrt{A_{ed}}$'
        x_label = r'Upstreamline root equiv area  $\sqrt{A_{eu}}$ [m]'
        y_label = r'Downstreamline root equiv area  $\sqrt{A_{ed}}$ [m]'
        xsym_label = r'$\sqrt{A_{eu}^*}$'
        ysym_label = r'$\sqrt{A_{ed}^*}$'
        try:
            joint_distbn = self.analysis.jpdf_uslt_dslt
            mx_distbn    = self.analysis.mpdf_uslt
            my_distbn    = self.analysis.mpdf_dslt
        except:
            self.print('"'+title+'" not computed: cannot plot')
            return
        self.plot_joint_pdf(joint_distbn, mx_distbn=mx_distbn, my_distbn=my_distbn,
                            fig_name=fig_name,
                            title=title, x_label=x_label, y_label=y_label,
                            xsym_label=xsym_label, ysym_label=ysym_label)

    def plot_joint_pdf_usla_uslc(self):
        """
        TBD
        """
        fig_name = 'joint_pdf_usla_uslc'
        title = r'Upstreamline distribution $f(\log(L_{mu}),\log(C_{u}))$'
        x_label = r'Upstreamline mean length  $L_{mu}$ [m]'
        y_label = r'Upstreamline concentration  $C_{u}$ [lines/m]'
        xsym_label = r'$L_{mu}^{*}$'
        ysym_label = r'$C_{u}^*$'
        try:
            joint_distbn = self.analysis.jpdf_usla_uslc
            mx_distbn    = self.analysis.mpdf_usla
            my_distbn    = self.analysis.mpdf_uslc
        except:
            self.print('"'+title+'" not computed: cannot plot')
            return
        self.plot_joint_pdf(joint_distbn, mx_distbn=mx_distbn, my_distbn=my_distbn,
                            fig_name=fig_name,
                            title=title, x_label=x_label, y_label=y_label,
                            xsym_label=xsym_label, ysym_label=ysym_label,
                            do_plot_mode=True)

    def plot_joint_pdf_dsla_dslc(self):
        """
        TBD
        """
        fig_name = 'joint_pdf_dsla_dslc'
        title = r'Downstreamline distribution $f(\log(L_{md}),\log(C_{d}))$'
        x_label = r'Downstreamline mean length  $L_{md}$ [m]'
        y_label = r'Downstreamline concentration  $C_{d}$ [lines/m]'
        xsym_label = r'$L_{md}^{*}$'
        ysym_label = r'$C_{d}^*$'
        try:
            joint_distbn = self.analysis.jpdf_dsla_dslc
            mx_distbn    = self.analysis.mpdf_dsla
            my_distbn    = self.analysis.mpdf_dslc
        except:
            self.print('"'+title+'" not computed: cannot plot')
            return
        self.plot_joint_pdf(joint_distbn, mx_distbn=mx_distbn, my_distbn=my_distbn,
                            fig_name=fig_name,
                            title=title, x_label=x_label, y_label=y_label,
                            xsym_label=xsym_label, ysym_label=ysym_label,
                            do_plot_mode=True)

    def plot_joint_pdf_dslt_dslc(self):
        """
        TBD
        """
        fig_name = 'joint_pdf_dslt_dslc'
        title = r'Downstreamline distribution $f(\sqrt{A_{ed}},\log(C_{d}))$'
        x_label = r'Downstreamline root equiv area  $\sqrt{A_{ed}}$ [m]'
        y_label = r'Downstreamline concentration  $C_{d}$ [lines/m]'
        xsym_label = r'$\sqrt{A_{ed}^*}$'
        ysym_label = r'$C_{d}^*$'
        try:
            joint_distbn = self.analysis.jpdf_dslt_dslc
            mx_distbn    = self.analysis.mpdf_dslt
            my_distbn    = self.analysis.mpdf_dslc
        except:
            self.print('"'+title+'" not computed: cannot plot')
            return
        self.plot_joint_pdf(joint_distbn, mx_distbn=mx_distbn, my_distbn=my_distbn,
                            fig_name=fig_name,
                            title=title, x_label=x_label, y_label=y_label,
                            xsym_label=xsym_label, ysym_label=ysym_label,
                            do_plot_mode=True)

    def plot_joint_pdf_uslc_dslc(self):
        """
        TBD
        """
        fig_name = 'joint_pdf_uslc_dslc'
        title = r'Streamline concentration distribution $f(\log(C_{u}),\log(C_{d}))$'
        x_label = r'Upstreamline concentration  $C_{u}$ [lines/m]'
        y_label = r'Downstreamline concentration  $C_{d}$ [lines/m]'
        xsym_label = r'$C_{u}^*$'
        ysym_label = r'$C_{d}^*$'
        try:
            joint_distbn = self.analysis.jpdf_uslc_dslc
            mx_distbn    = self.analysis.mpdf_uslc
            my_distbn    = self.analysis.mpdf_dslc
        except:
            self.print('"'+title+'" not computed: cannot plot')
            return
        self.plot_joint_pdf(joint_distbn, mx_distbn=mx_distbn, my_distbn=my_distbn,
                            fig_name=fig_name,
                            title=title, x_label=x_label, y_label=y_label,
                            xsym_label=xsym_label, ysym_label=ysym_label)

    @staticmethod
    def _choose_ticks(x_min,x_max):
        xtick_range= [int(np.round(np.log10(x_min))),int(np.round(np.log10(x_max)))+1] 
        xtick_range3= [int(np.round(np.log10(x_min))),
                       min(3,int(np.round(np.log10(x_max)))+1)] 
        x_ticks = [10**x for x in list(range(*xtick_range,1))] \
                + [( float(str(10**x).replace('1','3'))
                     if x<0 else int(str(10**x).replace('1','3')) )
                                            for x in list(range(*xtick_range3,1))] 
#                 + [( float(str(10**x).replace('1','6'))
#                      if x<0 else int(str(10**x).replace('1','6')) )
#                                             for x in list(range(*xtick_range,1))]
        x_ticks.sort()
        return x_ticks
        self._record_fig(fig_name,fig)

