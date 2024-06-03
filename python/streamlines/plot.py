"""
---------------------------------------------------------------------

Module for map and graph plotting.

---------------------------------------------------------------------

Requires Python packages/modules:
  -  :mod:`random`
  -  :mod:`matplotlib.pyplot`
  -  :mod:`matplotlib.gridspec`
  -  :mod:`matplotlib.ticker`
  -  :mod:`matplotlib.colors`
  -  :mod:`matplotlib.patches`
  -  :mod:`mpl_toolkits.axes_grid1 <mpl_toolkits.axes_grid1.axes_divider>`
  -  :mod:`colorsys`
  -  :mod:`scipy.interpolate`
  -  :mod:`scipy.stats`
  -  :mod:`scipy.signal`

Imports :class:`.Core` class.

---------------------------------------------------------------------

.. _matplotlib: https://matplotlib.org/
.. _scipy: https://www.scipy.org/
.. _random: https://docs.python.org/3/library/random.html
.. _mpl_toolkits: https://matplotlib.org/mpl_toolkits/index.html
.. _colorsys: https://docs.python.org/3/library/colorsys.html

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

import numpy as np
from   numpy  import pi, arctan, arctan2, sin, cos, sqrt
from   random import shuffle, seed
import matplotlib as mpl
import matplotlib.pyplot   as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker   as ticker
from   matplotlib.pyplot   import streamplot
from   matplotlib.colors   import LinearSegmentedColormap
from   matplotlib.patches  import ArrowStyle, FancyArrowPatch
from   mpl_toolkits.axes_grid1 import make_axes_locatable
# from   mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import colorsys
from   scipy.interpolate import interp1d 
from   scipy.signal      import decimate
from   scipy.stats import norm
from scipy.ndimage.morphology import binary_dilation, generate_binary_structure
                                     
from   os import environ
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

__all__ = ['Plot', 'random_colormap']
    
# Generate random colormap
def random_colormap(cmap_name='randomized', n_colors=1000,random_seed=1):
    """
    Create a cmap with a randomized sequence of bright colors
    
    Args:
        TBD (TBD): 
    
    TBD

    Returns:
        TBD: 
        TBD
    """
    ru = np.random.uniform
    np.random.seed(random_seed)
    color_palette \
        = [colorsys.hsv_to_rgb(ru(),ru(low=0.9),ru(low=0.9)) for i in range(n_colors)]
    return LinearSegmentedColormap.from_list(cmap_name, color_palette, N=n_colors)

class Plot(Core):       
    """
    Plot maps and distributions.
    
    Args:
        TBD (TBD): 
    
    TBD

    Returns:
        TBD: 
        TBD
    """
    def __init__(self, state, imported_parameters,
                 geodata, preprocess, trace, analysis, mapping):
        """    
        Args:
            TBD (TBD): 
        
        TBD
    
        Returns:
            TBD: 
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
                    x_pixel_scale=1,y_pixel_scale=1,
                    window_size_factor=None, projection=None):
        """  
        Args:
            TBD (TBD): 
        
        TBD
    
        Returns:
            TBD: 
            TBD
        """
#         mpl.rc( 'savefig', dpi=300)
        mpl.rc( 'figure', autolayout=False,  titlesize='Large',dpi=self.fig_dpi)
        mpl.rc( 'lines', linewidth=2.0, markersize=10)
        # mpl.rc( 'font', size=14,family='Times New Roman', serif='cm')
        # mpl.rc( 'font', size=14,family='DejaVu Sans', serif='cm')
        mpl.rc( 'font', size=self.general_font_size, family='Arial')
        mpl.rc( 'axes', labelsize=self.axes_font_size) 
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
            if window_size_factor is None:
                window_size_factor = self.window_size_factor
        else:
            window_size_factor = self.window_pdf_size_factor
        
        if projection is None:
            fig, axes = plt.subplots( 
                            figsize=(self.window_width *window_size_factor,
                                     self.window_height*window_size_factor))
            ticks_x = ticker.FuncFormatter(
                lambda x, pos: '{0:g}'.format(x*x_pixel_scale) )
            axes.xaxis.set_major_formatter(ticks_x)
            ticks_y = ticker.FuncFormatter(
                lambda y, pos: '{0:g}'.format(y*y_pixel_scale) )
            axes.yaxis.set_major_formatter(ticks_y)
        elif projection=='polar':
            fig, axes = plt.subplots( 
                            figsize=(self.window_width *window_size_factor,
                                     self.window_height*window_size_factor),
                            subplot_kw=dict(projection=projection))
        else:
             raise Exception('Projection not understood') 

        axes.set_rasterization_zorder(1)
        # HACK: turning off title because method now missing?
        # fig.canvas.set_window_title(window_title)
        axes.set_title(title, fontsize=self.title_font_size, fontweight='bold')
        
        return fig, axes
          
    def _record_fig(self,fig_name,fig):
        """  
        Args:
            TBD (TBD): 
        
        TBD
    
        Returns:
            TBD: 
            TBD
        """
        self.print('Recording figure "{}"'.format(fig_name))
        self.figs.update({fig_name : fig})  
        return fig_name     

    def _force_display(self, fig):
        """
        TBD
        """
        if self.state.do_display:
            try:
                fig.canvas.manager.show() 
                # this makes sure that the gui window gets shown
                # if this is needed depends on rcparams, this is just to be safe
                fig.canvas.flush_events() 
                # this make sure that if the event loop integration is not 
                # set up by the gui framework the plot will update
            except:
                pass
        else:
            plt.close(fig)

    @staticmethod
    def show():
        """  
        Args:
            TBD (TBD): 
        
        TBD
    
        Returns:
            TBD: 
            TBD
        """
        plt.show()

    def do(self):
        """
        Display all output  
        
        Args:
            TBD (TBD): 
        
        TBD
    
        Returns:
            TBD: 
            TBD
        """
        self.print('\n**Plot all begin**') 
        if self.do_plot_maps:
            self.plot_maps()
        if self.do_plot_distributions:
            self.plot_distributions()
        if self.do_plot_hsl_distributions:
            self.plot_hsl_distributions()

        self.print('**Plot all end**\n')  
        
    def plot_maps(self):
        """
        Plot maps
          
        Args:
            TBD (TBD): 
        
        TBD
    
        Returns:
            TBD: 
            TBD
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
        if self.do_plot_hsl:
            self.plot_hsl()
        if self.do_plot_hsl_contoured:
            self.plot_hsl_contoured()
        if self.do_plot_aspect:
            self.plot_aspect()
        if self.do_plot_hsl_aspect_distribution:
            self.plot_hsl_aspect_distribution()
        self.print('...done')
            
    def plot_dtm_shaded_relief(self, window_size_factor=None):
        """
        Hillshade view of source DTM
          
        Args:
            TBD (TBD): 
        
        TBD
    
        Returns:
            TBD: 
            TBD
        """
        fig,axes = self._new_figure(x_pixel_scale=self.geodata.roi_pixel_size/self.units,
                                    y_pixel_scale=self.geodata.roi_pixel_size/self.units,
                                    window_size_factor=window_size_factor)
        dtm_hillshade_array = self.generate_hillshade(self.geodata.dtm_array,
                                                      self.hillshade_azimuth,
                                                      self.hillshade_angle)
        axes.imshow(dtm_hillshade_array, cmap='Greys', 
                  extent=(0,self.geodata.dtm_array.shape[0],
                          0,self.geodata.dtm_array.shape[1]))
        self._force_display(fig)
        self._record_fig('dtm_shaded_relief',fig)
    
    def plot_roi_shaded_relief(self, interp_method=None, window_size_factor=None):
        """
        Hillshade view of ROI of DTM
          
        Args:
            TBD (TBD): 
        
        TBD
    
        Returns:
            TBD: 
            TBD
        """
        fig,axes = self._new_figure(x_pixel_scale=self.geodata.roi_pixel_size/self.units,
                                    y_pixel_scale=self.geodata.roi_pixel_size/self.units,
                                    window_size_factor=window_size_factor)
        self.plot_roi_shaded_relief_overlay(axes, 
                                    color_alpha=self.shaded_relief_color_alpha,
                                    hillshade_alpha=self.shaded_relief_hillshade_alpha,
                                    interp_method=interp_method)
        self._force_display(fig)
        self._record_fig('roi_shaded_relief',fig)

    def plot_roi_shaded_relief_overlay(self, axes, do_plot_color_relief=None,
                                       color_alpha=None, hillshade_alpha=None,
                                       interp_method=None):
        """
        Hillshade view of ROI of DTM (overlay method)
          
        Args:
            TBD (TBD): 
        
        TBD
    
        Returns:
            TBD: 
            TBD
        """
        if self.geodata.do_basin_masking: 
            if self.geodata.pad_width>0:               
                mask_array = self.state.merge_active_masks()[
                    self.geodata.pad_width:-self.geodata.pad_width,
                    self.geodata.pad_width:-self.geodata.pad_width]
            else:
                mask_array = self.state.merge_active_masks()
        else:
            mask_array = np.zeros_like(self.geodata.roi_array)

        try:
            self.roi_hillshade_array
        except:
            self.roi_hillshade_array = self.generate_hillshade(self.geodata.roi_array,
                                                               self.hillshade_azimuth,
                                                               self.hillshade_angle)
        if interp_method is None:
            interp_method = self.interpolation_method
        
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
          
        Args:
            TBD (TBD): 
        
        TBD
    
        Returns:
            TBD: 
            TBD
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

    def plot_streamlines(self, window_size_factor=None):
        """
        Streamlines, points on semi-transparent shaded relief
          
        Args:
            TBD (TBD): 
        
        TBD
    
        Returns:
            TBD: 
            TBD
        """
        fig_name='streamlines'
        window_title='streamlines'
        fig,axes = self._new_figure(window_title=window_title,
                                    x_pixel_scale=self.geodata.roi_pixel_size,
                                    y_pixel_scale=self.geodata.roi_pixel_size,
                                    window_size_factor=window_size_factor)
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
        axes.set_xlim(left=self.geodata.roi_x_bounds[0],
                      right=self.geodata.roi_x_bounds[1])
        axes.set_ylim(bottom=self.geodata.roi_y_bounds[0],
                      top=self.geodata.roi_y_bounds[1])
        self._force_display(fig)
        self._record_fig(fig_name,fig)
    
    def plot_channels(self, window_size_factor=None, do_reset_mask=True):
        """
        Streamlines, points on semi-transparent shaded relief
          
        Args:
            TBD (TBD): 
        
        TBD
    
        Returns:
            TBD: 
            TBD
        """
        try:
            self.mapping.mapping_array
        except: 
            self.print('Channels array not computed')
        if do_reset_mask:
            self.state.reset_active_masks()

        cmap = 'bwr'
        fig_name='channels'
        window_title='channels, midslopes & ridges'
        do_flip_cmap=False
        do_balance_cmap=True
                    
        fig,axes = self._new_figure(window_title=window_title,
                                    x_pixel_scale=self.geodata.roi_pixel_size/self.units,
                                    y_pixel_scale=self.geodata.roi_pixel_size/self.units,
                                    window_size_factor=window_size_factor)

        self.plot_roi_shaded_relief_overlay(axes,
                do_plot_color_relief=False, color_alpha=0,
                hillshade_alpha=self.channel_shaded_relief_hillshade_alpha)

        is_channel         = self.mapping.info.is_channel
        is_thinchannel     = self.mapping.info.is_thinchannel
        is_interchannel    = self.mapping.info.is_interchannel
        is_channelhead     = self.mapping.info.is_channelhead
        is_channeltail     = self.mapping.info.is_channeltail
        is_majorconfluence = self.mapping.info.is_majorconfluence
        is_minorconfluence = self.mapping.info.is_minorconfluence
        is_majorinflow     = self.mapping.info.is_majorinflow
        is_minorinflow     = self.mapping.info.is_minorinflow
        is_leftflank       = self.mapping.info.is_leftflank
        is_midslope        = self.mapping.info.is_midslope
        is_ridge           = self.mapping.info.is_ridge
        was_channelhead    = self.mapping.info.was_channelhead
        is_subsegmenthead  = self.mapping.info.is_subsegmenthead
        is_loop            = self.mapping.info.is_loop
        
        active_mask_array = self.state.merge_active_masks()
        
        grid_array = (self.mapping.mapping_array & is_thinchannel).copy().astype(np.bool)
        mask_array = active_mask_array | ~(grid_array)
        self.plot_simple_grid(grid_array, mask_array, axes, cmap='Blues', alpha=0.8)

        grid_array = (self.mapping.mapping_array & is_midslope).copy().astype(np.bool)
        fat_grid_array = np.zeros_like(grid_array, dtype=np.bool)
        dilation_structure = generate_binary_structure(2, 2)
        binary_dilation(grid_array, structure=dilation_structure, 
                        iterations=self.n_midslope_iterations, 
                        output=fat_grid_array)
        mask_array = active_mask_array | ~(fat_grid_array)
        self.plot_simple_grid(fat_grid_array, mask_array, axes, cmap='Greens', alpha=0.8)
        
        grid_array = (self.mapping.mapping_array & is_ridge).copy().astype(np.bool)
        fat_grid_array = np.zeros_like(grid_array, dtype=np.bool)
        dilation_structure = generate_binary_structure(2, 2)
        binary_dilation(grid_array, structure=dilation_structure,
                        iterations=self.n_ridge_iterations, 
                        output=fat_grid_array)
        mask_array = active_mask_array | ~(fat_grid_array)
        self.plot_simple_grid(fat_grid_array, mask_array, axes, cmap='Oranges', alpha=0.8)
        
#         self.plot_compound_markers(axes, is_majorconfluence, ['blue','black'])
#         self.plot_compound_markers(axes, is_loop,     ['pink','black'], msf=2)
        if self.do_plot_tails:
            self.plot_compound_markers(axes, is_channeltail,    ['cyan','black'], msf=1.5)
#         self.plot_compound_markers(axes, is_leftflank,     ['purple','black'], msf=0.2)
        if self.do_plot_heads:
            self.plot_compound_markers(axes, is_channelhead,    ['blue','black'], msf=0.5)
#         self.plot_compound_markers(axes, was_channelhead,   ['orange','black'], msf=0.3)
#         self.plot_compound_markers(axes, is_subsegmenthead,['orange','black'], msf=0.25)
#         self.plot_compound_markers(axes, is_midslope,        ['purple','black'])
#         self.plot_compound_markers(axes, is_ridge,        ['purple','black'])

        self._force_display(fig)
        self._record_fig(fig_name,fig)

    def plot_compound_markers(self, axes, flag, colors, msf=1.0):
        """
        Streamlines, points on semi-transparent shaded relief
          
        Args:
            TBD (TBD): 
        
        TBD
    
        Returns:
            TBD: 
            TBD
        """
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
                         do_vlimit=True, v_min=None, v_max=None):
        """
        Streamlines, points on semi-transparent shaded relief
          
        Args:
            TBD (TBD): 
        
        TBD
    
        Returns:
            TBD: 
            TBD
        """
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
                      interpolation=self.interpolation_method, vmin=0, vmax=1
                      )
        else:
            im = axes.imshow(np.flipud(masked_grid_array.T), 
                      cmap=cmap, 
                      extent=[*self.geodata.roi_x_bounds,*self.geodata.roi_y_bounds],
                      alpha=alpha,
                      interpolation=self.interpolation_method, vmin=v_min, vmax=v_max
                      )
        clim=im.properties()['clim']
        return im
    
    def plot_flow_maps(self, window_size_factor=None): 
        """
        Streamlines, points on semi-transparent shaded relief
          
        Args:
            TBD (TBD): 
        
        TBD
    
        Returns:
            TBD: 
            TBD
        """
        if self.do_plot_flow_dsla:
            fig_name='dsla'
            window_title='dsla'     
            tmp_array = self.trace.sla_array[:,:,0].copy()
            tmp_array[(~np.isnan(tmp_array)) & (tmp_array>0.0)] \
                = np.sqrt(tmp_array[(~np.isnan(tmp_array)) & (tmp_array>0.0)])
            
            self.plot_gridded_data(tmp_array,
                                   'gist_earth', 
                                   fig_name=fig_name, 
                                   window_size_factor=window_size_factor,
                                   window_title=window_title,
                                   do_flip_cmap=True, do_balance_cmap=False)
            tmp_array = self.trace.slt_array[:,:,0].copy()
            tmp_array[(~np.isnan(tmp_array)) & (tmp_array>0.0)] \
                = np.log(tmp_array[(~np.isnan(tmp_array)) & (tmp_array>0.0)])
            
        if self.do_plot_flow_dslt:
            fig_name='dslt'
            window_title='dslt'     
            self.plot_gridded_data(tmp_array,
                                   'gist_earth', #'seismic', 
                                   fig_name=fig_name, 
                                   window_size_factor=window_size_factor,
                                   window_title=window_title,
                                   do_flip_cmap=True, do_balance_cmap=False)
        
    def plot_segments(self,do_shaded_relief=True, window_size_factor=None):
        """
        Streamlines, points on semi-transparent shaded relief
          
        Args:
            TBD (TBD): 
        
        TBD
    
        Returns:
            TBD: 
            TBD
        """
        fig_name='label'
        window_title='label'     
        tmp_array = (self.mapping.coarse_subsegment_array.copy().astype(np.int32))
#         pdebug('label', tmp_array.shape,np.unique(tmp_array))
#         tmp_array[tmp_array!=1]=0
        self.state.add_active_mask(
            {'merged_coarse_mask_array':self.mapping.merged_coarse_mask_array})
        mask_array = self.state.merge_active_masks()
#         pdebug('list_active_masks',self.state.list_active_masks())
        self.plot_gridded_data(tmp_array,
                               'randomized', 
                               fig_name=fig_name, window_size_factor=window_size_factor,
                               mask_array=mask_array,
                               window_title=window_title,
                               do_shaded_relief=False, 
                               do_flip_cmap=False, do_balance_cmap=False)

    def plot_hsl(self, cmap=None, window_size_factor=None,
                 z_min=None,z_max=None,z_pctile=None, do_shaded_relief=None, 
                 colorbar_size=None, colorbar_aspect=None, grid_alpha=None):
        """
        Streamlines, points on semi-transparent shaded relief
          
        Args:
            TBD (TBD): 
        
        TBD
    
        Returns:
            TBD: 
            TBD
        """
        fig_name='hsl'
        window_title='hsl'     
        if cmap is None:
            cmap = self.hsl_cmap
        if colorbar_size is None:
            colorbar_size = self.hsl_colorbar_size
        if colorbar_aspect is None:
            colorbar_aspect = self.hsl_colorbar_aspect
        if do_shaded_relief is None:
            do_shaded_relief = self.hsl_do_shaded_relief
        if z_pctile is None:
            z_pctile = self.hsl_z_pctile
        if z_min is None:
            if self.hsl_z_min=='full':
                z_min = np.percentile(self.mapping.hsl_array, 0.0)
            elif self.hsl_z_min=='auto':
                z_min = np.percentile(self.mapping.hsl_array, 1.0)
            else:
                z_min = self.hsl_z_min
        if z_max is None:
            if self.hsl_z_max=='full':
                z_max = np.percentile(self.mapping.hsl_array,100.0)  
            elif self.hsl_z_max=='auto':
                z_max = np.percentile(self.mapping.hsl_array,z_pctile)  
            else:
                z_max = self.hsl_z_max
        grid_array = np.clip(self.mapping.hsl_array.copy(),z_min,z_max)
#         mask_array = np.zeros_like(grid_array).astype(np.bool)
        mask_array = self.state.merge_active_masks()
        mask_array[grid_array<=5] = True
#         pdebug('list_active_masks',self.state.list_active_masks())
        self.plot_gridded_data(grid_array,
                               cmap,  # rainbow
                               mask_array=mask_array,
                               fig_name=fig_name, window_size_factor=window_size_factor,
                               window_title=window_title,
                               do_flip_cmap=False, do_balance_cmap=False,
                               do_shaded_relief=do_shaded_relief, 
                               do_colorbar=True, 
                               colorbar_title='hillslope length [m]',
                               colorbar_size=colorbar_size,
                               colorbar_aspect=colorbar_aspect,
                               grid_alpha=self.hsl_alpha)
    
    def plot_hsl_contoured(self, window_size_factor=None, cmap=None,
                           do_colorbar=True, colorbar_title='hillslope length [m]',
                           n_contours=None, contour_interval=None, linewidth=None,
                           z_min=None,z_max=None,z_pctile=None, do_shaded_relief=None,
                           colorbar_size=None, colorbar_aspect=None, 
                           contour_label_suffix=None,
                           contour_label_fontsize=None, do_plot_contours=None):
        """
        Streamlines, points on semi-transparent shaded relief
          
        Args:
            TBD (TBD): 
        
        TBD
    
        Returns:
            TBD: 
            TBD
        """
        fig_name='hsl_contours'
        window_title='hsl contours'    
        fig,axes = self._new_figure(window_title=window_title,
                                    x_pixel_scale=self.geodata.roi_pixel_size,
                                    y_pixel_scale=self.geodata.roi_pixel_size,
                                    window_size_factor=window_size_factor)
        pad = self.geodata.pad_width
        pslice = np.index_exp[pad:-pad,pad:-pad]
        if cmap is None:
            cmap = self.contour_hsl_cmap
        if do_plot_contours is None:
            do_plot_contours = self.do_plot_hsl_contours
        if colorbar_size is None:
            colorbar_size = self.contour_hsl_colorbar_size
        if colorbar_aspect is None:
            colorbar_aspect = self.contour_hsl_colorbar_aspect
        if do_shaded_relief is None:
            do_shaded_relief = self.contour_hsl_do_shaded_relief
        if contour_label_fontsize is None:
            contour_label_fontsize = self.contour_label_fontsize
        if contour_label_suffix is None:
            contour_label_suffix = self.contour_hsl_label_suffix
        if z_pctile is None:
            z_pctile = self.contour_hsl_z_pctile
        if z_min is None:
            if self.contour_hsl_z_min=='full':
                z_min = np.percentile(self.mapping.hsl_smoothed_array[pslice], 0.0)
            elif self.contour_hsl_z_min=='auto':
                z_min = np.percentile(self.mapping.hsl_smoothed_array, 1.0)
            else:
                z_min = self.contour_hsl_z_min
        if z_max is None:
            if self.contour_hsl_z_max=='full':
                z_max = np.percentile(self.mapping.hsl_smoothed_array,100.0)  
            elif self.contour_hsl_z_max=='auto':
                z_max = np.percentile(self.mapping.hsl_array,z_pctile)  
            else:
                z_max = self.contour_hsl_z_max     
        grid_array = np.clip(self.mapping.hsl_smoothed_array[pslice].copy(),z_min,z_max)
#         if n_contours is None and self.contour_hsl_n_contours!='auto':
#                 n_contours = self.contour_hsl_n_contours
        if linewidth is None:
            linewidth = self.contour_hsl_linewidth
                
        mask_array = self.state.merge_active_masks()[pslice]
        mask_array[self.mapping.hsl_array[pslice]<=5] = True
#         pdebug('list_active_masks',self.state.list_active_masks())
#         mask_array &= False
        if do_shaded_relief:
            hillshade_alpha = self.grid_shaded_relief_hillshade_alpha
            hsl_alpha = self.contour_hsl_alpha
        else:
            hillshade_alpha = 0.0
            hsl_alpha = 1.0
        self.plot_roi_shaded_relief_overlay(axes, do_plot_color_relief=False,
                                            hillshade_alpha=hillshade_alpha)
        im = self.plot_simple_grid((grid_array),
                              mask_array,axes,cmap=cmap,alpha=hsl_alpha,do_vlimit=False)
#         if do_colorbar:
#             divider = make_axes_locatable(axes)
#             cax = divider.append_axes("bottom", size="4%", 
#                                       pad=0.5, aspect=colorbar_aspect)
#             cbar = plt.colorbar(im, cax=cax, orientation="horizontal")
#             cbar.set_label(colorbar_title)
            
        if do_colorbar:
            divider = make_axes_locatable(axes)
            cax = divider.append_axes("bottom", size="{}%".format(colorbar_size), 
                                      pad=0.5, aspect=colorbar_aspect)
            cbar = plt.colorbar(im, cax=cax, orientation="horizontal")
            cbar.set_label(colorbar_title)
            
        if do_plot_contours:
            self.plot_contours_overlay(axes,grid_array.T,mask=mask_array,
                                       n_contours=n_contours,
                                       contour_interval=contour_interval,
                                       linewidth=linewidth,
                                       contour_label_suffix=contour_label_suffix, 
                                       contour_label_fontsize=contour_label_fontsize)
        # Force map limits to match those of the ROI
        #   - turn this off to check for boundary overflow errors in e.g. streamlines
        axes.set_xlim(left=self.geodata.roi_x_bounds[0],
                      right=self.geodata.roi_x_bounds[1])
        axes.set_ylim(bottom=self.geodata.roi_y_bounds[0],
                      top=self.geodata.roi_y_bounds[1])
        self._force_display(fig)
        self._record_fig(fig_name,fig)
    
    def plot_aspect(self, window_size_factor=None,cmap=None,do_plot_contours=None,
                    contour_label_suffix=None, contour_label_fontsize=None):
        """
        Streamlines, points on semi-transparent shaded relief
          
        Args:
            TBD (TBD): 
        
        TBD
    
        Returns:
            TBD: 
            TBD
        """
        fig_name='aspect'
        window_title='aspect'
        try:
            self.mapping.aspect_array
        except: 
            self.print('Aspect array not computed')
        
        if cmap==None:
            cmap = 'RdYlBu' #'seismic'  #bwr
        do_flip_cmap=False
        do_balance_cmap=True
        if do_plot_contours is None:
            do_plot_contours = self.do_plot_aspect_contours
        if contour_label_fontsize is None:
            contour_label_fontsize = self.contour_label_fontsize
        if contour_label_suffix is None:
            contour_label_suffix = self.contour_aspect_label_suffix
                            
        fig,axes = self._new_figure(window_title=window_title,
                                    x_pixel_scale=self.geodata.roi_pixel_size,
                                    y_pixel_scale=self.geodata.roi_pixel_size,
                                    window_size_factor=window_size_factor)

        self.state.add_active_mask({'merged_coarse': 
                                    self.mapping.merged_coarse_mask_array})
        mask_array = self.state.merge_active_masks()
        grid_array = np.rad2deg(self.mapping.aspect_array.copy())
        
        self.plot_roi_shaded_relief_overlay(axes,
                do_plot_color_relief=False, color_alpha=0,
                hillshade_alpha=self.channel_shaded_relief_hillshade_alpha)
        im = self.plot_simple_grid(grid_array, mask_array, axes, cmap=cmap, 
                                   alpha=0.5, do_vlimit=False, v_min=-180, v_max=+180)
        if do_plot_contours:
            pad = self.geodata.pad_width
            pslice = np.index_exp[pad:-pad,pad:-pad]
            self.plot_contours_overlay(axes,
                                       grid_array[pslice].T,
                                       mask=mask_array[pslice],
                                       contour_interval=10,
                                       linewidth=1,
                                       contour_label_suffix=contour_label_suffix, 
                                       contour_label_fontsize=contour_label_fontsize)
        
        divider = make_axes_locatable(axes)
        cax = divider.append_axes("right",  pad=0.3,
                                  size="{}%".format(self.contour_aspect_colorbar_size), 
                                  aspect=self.contour_aspect_colorbar_aspect)
        ticks       = [-180,-90,0,90,180]
        tick_labels = ['W','S','E','N','W']
        # Omit top W tick/label
        cbar = plt.colorbar(im, cax=cax, ticks=ticks[0:-1], 
                            orientation='vertical')
#         cbar.set_label('aspect', rotation=0,y=-0.1,labelpad=-20)
#         cbar.ax.set_yticklabels([r'-180$\degree$',r'-90$\degree$',
#                                  r'0$\degree$ E',
#                                  r'+90$\degree$',r'+180$\degree$'])
        cbar.ax.set_yticklabels(tick_labels[0:-1]) 

        self.state.reset_active_masks()
        self._force_display(fig)
        self._record_fig(fig_name,fig)

    def plot_hsl_aspect_distribution(self, window_size_factor=None, cmap=None):
        """
        Streamlines, points on semi-transparent shaded relief
          
        Args:
            TBD (TBD): 
        
        TBD
    
        Returns:
            TBD: 
            TBD
        """
        try:
            hsl_aspect_array = self.mapping.hsl_aspect_averages_array
        except: 
            self.print('HSL-aspect array not computed')
            
        fig_name='hsl_aspect_distribution'
        window_title='HSL-aspect distribution'
                    
        fig,axes = self._new_figure(window_title=window_title,
                                    x_pixel_scale=self.geodata.roi_pixel_size,
                                    y_pixel_scale=self.geodata.roi_pixel_size,
                                    window_size_factor=window_size_factor,
                                    projection='polar')
        hsl = hsl_aspect_array[:,0][~np.isnan(hsl_aspect_array[:,0])]
        asp = hsl_aspect_array[:,1][~np.isnan(hsl_aspect_array[:,0])]
        hsl_max = np.max(hsl)
        hsl_min = np.min(hsl)
        n_bins  = hsl.shape[0]
        hsl_north = hsl[asp>=0.0]
        asp_north = asp[asp>=0.0]
        hsl_south = hsl[asp<=0.0]
        asp_south = asp[asp<=0.0]
        if hsl_north!=[] and hsl_south!=[]:
            hsl_north = np.concatenate( 
                (np.array(hsl_north[0:1]), hsl_north, 
                 (np.array(hsl_south[0:1])+np.array(hsl_north[-2:-1]))/2.0 ) )
            asp_north = np.concatenate(
                (np.array([0.0]), asp_north, np.array([np.pi])) )
            hsl_south = np.concatenate(
                 ( (np.array(hsl_south[0:1])+np.array(hsl_north[-2:-1]))/2.0, 
                  hsl_south, np.array(hsl_south[-2:-1])))
            asp_south = np.concatenate(
                 (np.array([-np.pi]), asp_south, np.array([0.0])) )
        if cmap is None:
            cmap = 'Greys'
        cmap = mpl.cm.get_cmap(cmap)
        rgba_north = cmap((self.mapping.hsl_mean_north-hsl_min)/(hsl_max-hsl_min))
        rgba_south = cmap((self.mapping.hsl_mean_south-hsl_min)/(hsl_max-hsl_min))
        axes.fill_between(asp_north, hsl_north, facecolor=rgba_north, alpha=0.4)
        axes.fill_between(asp_south, hsl_south, facecolor=rgba_south, alpha=0.4)
        axes.plot(asp, hsl, 'black', lw=1)
        
#         axes.set_theta_zero_location('N')
#         axes.set_theta_direction(-1)
        c_interval = 5
        while c_interval<=100:
            c_max = np.ceil(hsl_max//c_interval)*c_interval
            c_count = c_max/c_interval
            if c_count<=5:
                break
            c_interval *= 2        
        bands = np.arange(0,c_max+2*c_interval,c_interval).astype(np.uint32)
        band_labels = ['{}m'.format(band) for band in bands]
        axes.set_rgrids(bands, labels=band_labels, color='blue',style=None)
        angles = np.arange(0,360,45).astype(np.uint32)
        spc = u'\N{space}'
        angle_labels = [spc*6+r'0$\degree$ = E',r'45$\degree$',
                        r'90$\degree$ = N',r'135$\degree$',
                        r'$\pm$180$\degree$'+spc*6, r'-135$\degree$'+spc*5,
                        r'-90$\degree$ = S',r'-45$\degree$']
        axes.set_thetagrids(angles, labels=angle_labels)
#         axes.tick_params(pad=8)
        axes.grid(color='blue',alpha=0.5,linestyle='dashed')
        
        mha = np.deg2rad(self.mapping.hsl_mean_azimuth)
        mhm = self.mapping.hsl_mean_magnitude
        mhl = hsl_max/2.0
        head_length = 40   # hack - how to correctly scale??
        arrow_style = ArrowStyle.Fancy(head_length=head_length,head_width=head_length/2,
                                      tail_width=1)
        arrow_patch = FancyArrowPatch((mha-np.pi,mhl/3.0), (mha,mhl),
                                      shrinkA=1, shrinkB=1,
                                      arrowstyle=arrow_style,
                                      facecolor='blue', edgecolor='blue', 
                                      linewidth=4)
        axes.add_patch(arrow_patch)
        axes.plot(mha,mhm,'.',ms=40,color='blue')
        axes.plot(mha,mhm,'.',ms=30,color='lightblue')

        color = 'purple'
        h_position = hsl_max/1.8
        axes.text(+np.deg2rad(145), h_position, 
                  r'$\overline{\mathbf{L}}_\mathbf{N}=$'+'{:2.0f}m'
                    .format(self.mapping.hsl_mean_north),
                  size=18, color=color, fontweight='bold',
                  horizontalalignment='center', verticalalignment='center')
        axes.text(-np.deg2rad(145), h_position, 
                  r'$\overline{\mathbf{L}}_\mathbf{S}=$'+'{:2.0f}m'
                    .format(self.mapping.hsl_mean_south),
                  size=18, color=color, fontweight='bold',
                  horizontalalignment='center', verticalalignment='center')

        self._force_display(fig)
        self._record_fig(fig_name,fig)

    def plot_contours_overlay(self,axes,Z,mask=None, n_contours=None,
                              contour_interval=None, linewidth=None,
                              contour_label_suffix='m', contour_label_fontsize=12):
        """
        Streamlines, points on semi-transparent shaded relief
          
        Args:
            TBD (TBD): 
        
        TBD
    
        Returns:
            TBD: 
            TBD
        """
        x_pixel_centers_array,y_pixel_centers_array \
            = np.meshgrid(self.geodata.x_roi_n_pixel_centers,
                          self.geodata.y_roi_n_pixel_centers)
        Z = np.ma.array(Z,mask=mask.T)
#         z_min = np.percentile(Z, 1.0)
        z_max = np.percentile(Z,100.0)
        if n_contours is None:
            if contour_interval is None:
                c_interval = 5
            else:
                c_interval = contour_interval
            while c_interval<50:
                c_min = np.floor(np.min(Z)//c_interval)*c_interval
                c_max = np.ceil(z_max//c_interval)*c_interval
                c_count = (c_max-c_min)/c_interval
                if c_count<=17:
                    break
                c_interval *= 2
            n_contours = np.arange(c_min,c_max+c_interval,c_interval)
        contours = axes.contour(x_pixel_centers_array,y_pixel_centers_array,Z,
                                n_contours, colors='k', linewidths=linewidth)
        axes.clabel(contours, fmt='%.0f'+contour_label_suffix, 
                    fontsize=contour_label_fontsize);

    def plot_hsl_distributions(self, x_stretch=None):
        """
        Streamlines, points on semi-transparent shaded relief
          
        Args:
            TBD (TBD): 
        
        TBD
    
        Returns:
            TBD: 
            TBD
        """
        self.print('Plotting hillslope length distributions...')
        df = self.mapping.hsl_aspect_df    
        
        if x_stretch is None:
            x_stretch = self.mhsl_pdf_x_stretch
        hsl_n = df.hsl[(df.aspect>45) & (df.aspect<135)]
        hsl_s = df.hsl[(df.aspect<-45) & (df.aspect>-135)]
        
        name = 'hsl_nsall_distbn'
        title = 'N-facing vs S-facing hillslope length distributions'
        fig,_ = self._new_figure(window_title=name)
        axes  = df['hsl'].plot.density(figsize=(8,8), title=title, color='k',#style='-.',
                                      label='360°', alpha=0.7, lw=1, secondary_y='360°')
        x_max = df['hsl'].quantile(q=1)*x_stretch
        try:
            axes2 = hsl_n.plot.density(figsize=(8,8),color='steelblue', label='N-facing')
        except:
            axes2 = None
        try:
            axes3 = hsl_s.plot.density(figsize=(8,8),color='darkorange',label='S-facing')
        except:
            axes3 = None
        if axes2 is None:
            axes_ns = axes3
        else:
            axes_ns = axes2
        lines = axes.get_lines()
        axes.legend(lines, [l.get_label() for l in lines], loc='upper right')
        lines = axes_ns.get_lines()
        axes_ns.legend(lines, [l.get_label() for l in lines], loc='upper left')
        x_label = r'distance $L$  [m]'
        y_label = r'probability density  $f(L)$  [m$^{-1}$]'
        axes.set_xlabel(x_label)
        axes.set_ylabel(y_label)
        axes_ns.set_ylabel(y_label)
        axes.set_ylim(0,None)
        axes_ns.set_ylim(0,None)
        axes.set_xlim(0,x_max)
        axes_ns.set_xlim(0,x_max)
        self._force_display(fig)
        self._record_fig(name,fig)
        
        if self.mapping.hsl_ns_min is not None and self.mapping.hsl_ns_max is not None:
            name = 'hsl_ns_qq'
            title = 'Hillslope length-aspect Q-Q plot'
            fig,_ = self._new_figure(window_title=name)
            x_label = r'South-facing HSL $L_S$ percentiles  [m]'
            y_label = r'North-facing HSL $L_N$ percentiles  [m]'
            hsl_ns_min = self.mapping.hsl_ns_min
            hsl_ns_max = self.mapping.hsl_ns_max
            hsls = np.linspace(hsl_ns_min, hsl_ns_max, num=50)
            [plt.plot(self.mapping.hsl_ns_qq_array[0], self.mapping.hsl_ns_qq_array[1], 
                     marker, ms='8', color='darkblue' ) for marker in ('-','.')]
            axes = plt.gca()
            axes.set_xlabel(x_label)
            axes.set_ylabel(y_label)
            # Find plot limits (not tight)
            xy_min = min(axes.get_xlim()[0],axes.get_ylim()[0])
            xy_max = max(axes.get_xlim()[1],axes.get_ylim()[1])
            # Force equal x,y limits
            axes.set_xlim(xy_min,xy_max)
            axes.set_ylim(xy_min,xy_max)
            # Plot a diagonal x=y red dashed line for guidance
            hsls = np.linspace(xy_min, xy_max, num=50)
            plt.plot(hsls,hsls, 'r--', lw=1)
            # Replot to ensure q-q line is on top
            [plt.plot(self.mapping.hsl_ns_qq_array[0], self.mapping.hsl_ns_qq_array[1], 
                     marker, ms='8', color='darkblue' ) for marker in ('-','.')]
            axes.set_title(title)
            self._force_display(fig)
            self._record_fig(name,fig)

        if self.mapping.hsl_ns_min is not None and self.mapping.hsl_ns_max is not None:
            name = 'hsl_ns_pp'
            title = 'Hillslope length-aspect P-P plot'
            fig,_ = self._new_figure(window_title=name)
            x_label = r'North-facing HSL cumulative prob $F(L_N)$  [%]'
            y_label = r'South-facing HSL cumulative prob $F(L_S)$  [%]'
            percents = self.mapping.hsl_ns_pp_array[0]
            plt.plot(percents,percents, 'r--', lw=1)
            axes = plt.gca()
            axes.fill_between(self.mapping.hsl_ns_pp_array[1],
                              self.mapping.hsl_ns_pp_array[2], 
                              self.mapping.hsl_ns_pp_array[1],
                              facecolor='darkblue', alpha=0.1)
            plt.plot(self.mapping.hsl_ns_pp_array[1],self.mapping.hsl_ns_pp_array[2], 
                     color='darkblue')
            axes.set_xlabel(x_label)
            axes.set_ylabel(y_label)
            axes.set_title(title)
    #         plt.autoscale(tight=True)
            axes.set_xlim(0,100.25)
            axes.set_ylim(0,100.25)
            dxy = 0.025
            [plt.text(0.5-ns[1], 0.5+ns[1], 'longer {}-facing'.format(ns[0]), 
                      horizontalalignment='center', verticalalignment='center', 
                      rotation=45, transform=axes.transAxes, color='r') 
                      for ns in (('N',dxy),('S',-dxy))]
            self._force_display(fig)
            self._record_fig(name,fig)    
    
        return
    
        name = 'hsl_means_distbn'
        title = 'Distribution of means of hillslope length'
        fig,_ = self._new_figure(window_title=name)
        df = self.mapping.hsl_stats_df
        # Cut
        df = df[df['count']>1]
        kde_min_labels = 20
        x_label  = r'distance $L$  [m]'
        y_label  = r'probability density  $f(L)$  [m$^{-1}$]'
        y_label2 = r'number  $N(L)$  [$-$]'
        print(df)
        if df.shape[0]>kde_min_labels:
            axes = df['mean [m]'].plot.density(figsize=(8,8), title=title,
                                               label='pdf',secondary_y='pdf')
            axes2 = df['mean [m]'].plot.hist(figsize=(8,8), title=title,alpha=0.2,
                                             color='b')
            axes2.set_ylabel(y_label2)
        else:
            axes = df['mean [m]'].plot.hist(figsize=(8,8), title=title)
        axes.set_xlabel(x_label)
        axes.set_ylabel(y_label)
        axes.set_xlim(0,df['mean [m]'].quantile(q=1)*x_stretch)
        axes.set_ylim(0,None)
        self._force_display(fig)
        self._record_fig(name,fig)

        name = 'hsl_stddevs_distbn'
        fig,_ = self._new_figure(window_title=name)
        title = 'Hillslope length standard deviations'
        if df.shape[0]>kde_min_labels:
            axes = df['stddev [m]'].plot.density(figsize=(8,8), title=title)
        else:
            axes = df['stddev [m]'].plot.hist(figsize=(8,8), title=title)
        axes.set_xlabel(r'distance std deviation $\sigma_L$  [m]')
        axes.set_ylabel(r'probability density  $f(\sigma_L)$  [m$^{-1}$]')
        axes.set_xlim(0,df['stddev [m]'].quantile(q=0.99))
        axes.set_ylim(0,None)
        # axes.legend(['hillslope length std dev'],frameon=False)
        self._force_display(fig)
        self._record_fig(name,fig)
  
        name = 'hsl_counts_distbn'
        try:
            csum = np.int(df['count'].sum())
            cmax = np.int(df['count'].quantile(q=0.99))
            n_bins = min(50,max(10,csum//cmax))
        except:
            n_bins = 20
        fig,_ = self._new_figure(window_title=name)
        title = 'Hillslope length streamline counts'
        axes = df['count'].plot.hist(bins=n_bins,figsize=(8,8), title=title)
        axes.set_xlabel(r'number of streamlines per length average $N_{sl}$   [-]');
        axes.set_ylabel(r'frequency  $n(N_{sl})$  [-]')
        axes.set_xlim(0,df['count'].quantile(q=0.99))
        axes.set_ylim(0,None)
        self._force_display(fig)
        self._record_fig(name,fig)
        
        self.print('...done')

    def plot_gridded_data(self, grid_array, gridded_cmap, 
                          window_size_factor=None, fig_name=None, mask_array=None,
                          window_title='', do_flip_cmap=False, do_balance_cmap=True,
                          do_shaded_relief=True, do_colorbar=False, 
                          colorbar_title=None, colorbar_aspect=0.07,
                          colorbar_size=4, grid_alpha=None, extent=None):
        """
        Streamlines, points on semi-transparent shaded relief
          
        Args:
            TBD (TBD): 
        
        TBD
    
        Returns:
            TBD: 
            TBD
        """
        if mask_array is None:
            mask_array = np.zeros_like(grid_array).astype(np.bool)
        pad = self.geodata.pad_width
        pslice = np.index_exp[pad:-pad,pad:-pad]
        mask_array = mask_array[pslice]
        if extent is None:
            extent = [*self.geodata.roi_x_bounds,*self.geodata.roi_y_bounds]
        if self.geodata.do_basin_masking:
            mask_array |= self.state.merge_active_masks()[pslice]
            
        if self.seed_point_marker_size==0:
            seed_point_marker = 'g,'
            seed_point_marker_size = 1
        else:
            seed_point_marker = 'gD'     
            seed_point_marker_size = self.seed_point_marker_size
            
        fig,axes = self._new_figure(window_title=window_title,
                                    x_pixel_scale=self.geodata.roi_pixel_size,
                                    y_pixel_scale=self.geodata.roi_pixel_size,
                                    window_size_factor=window_size_factor)

        if do_shaded_relief:
            self.plot_roi_shaded_relief_overlay(axes, do_plot_color_relief=False,
                    color_alpha=self.grid_shaded_relief_color_alpha,
                    hillshade_alpha=self.grid_shaded_relief_hillshade_alpha)
            if grid_alpha is None:
                grid_alpha = self.streamline_density_alpha
        else:
            grid_alpha = 1.0

        grid_array = grid_array[pslice]
#         extra_mask_array = grid_array.astype(np.bool)
#         mask_array = mask_array | (~extra_mask_array)
#         mask_array &= False
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
                  extent=extent,
                  alpha=grid_alpha,
                  interpolation=self.interpolation_method,
                  vmin=vmin, vmax=vmax
                  )
        clim=im.properties()['clim']
        if do_colorbar:
            divider = make_axes_locatable(axes)
            cax = divider.append_axes("bottom", size="{}%".format(colorbar_size), 
                                      pad=0.5, aspect=colorbar_aspect)
            cbar = plt.colorbar(im, cax=cax, orientation="horizontal")
            cbar.set_label(colorbar_title)

#         try:
#             is_thinchannel = self.mapping.info.is_thinchannel
#             grid_array = ( self.mapping.mapping_array[pslice] 
#                            & is_thinchannel ).astype(np.bool)
#             im = axes.imshow(np.flipud(grid_array.T), 
#                       cmap='Blues', 
#                       extent=[*self.geodata.roi_x_bounds,*self.geodata.roi_y_bounds],
#                       alpha=0.5,
#                       interpolation=self.interpolation_method
#                       )
#         except:
#             pass

        self._force_display(fig)
        self._record_fig(fig_name,fig)

    def plot_updownstreamlines_overlay(self,axes,do_down=True):
        """
        Up or downstreamlines 
          
        Args:
            TBD (TBD): 
        
        TBD
    
        Returns:
            TBD: 
            TBD
        """
        if do_down:
            up_or_down_str = 'down'
            streamline_arrays_list = self.trace.streamline_arrays_list[0]
            color = self.downstreamline_color
        else:
            up_or_down_str = 'up'
            streamline_arrays_list = self.trace.streamline_arrays_list[1]
            color = self.upstreamline_color
        marker = self.streamline_point_marker   
        size = self.streamline_point_size
        linewidth = self.streamline_linewidth
        alpha = self.streamline_point_alpha
        
        try:
            n_streamlines = len(streamline_arrays_list)
        except:
            return
        idx_list = list(range(n_streamlines))
        if self.shuffle_rng_seed is not None:
            seed(self.shuffle_rng_seed)
        if self.n_streamlines_limit!='none':
            shuffle(idx_list)
            idx_list = idx_list[0:min(n_streamlines,self.n_streamlines_limit)]

        todo = len(idx_list)
        if self.n_streamlines_limit!='none' and n_streamlines>self.n_streamlines_limit:
            self.print('Plotting {0:,}'.format(todo)+' '
                  +up_or_down_str+'streamlines'
                  +' randomly sampled from a set of {0:,}'.format(n_streamlines))
        else:
            self.print('Plotting all {0:,} {1}-streamlines'.format(todo,up_or_down_str))
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
                      lw=linewidth,
                      alpha=alpha,
                      fillstyle='full')
    
            progress += 1
            if progress%self.print_interval==0:
                prog = int(100*int(0.5+100*progress/todo)//100)
                self.print(str(prog)+'%',end='') 
                if prog!=100:
                    self.print('...', end='')       
 
        self.print('')    
    
    def plot_classical_streamlines(self, window_size_factor=None):
        """
        Classic streamlines on color shaded relief
          
        Args:
            TBD (TBD): 
        
        TBD
    
        Returns:
            TBD: 
            TBD
        """
        fig,axes = self._new_figure(x_pixel_scale=self.geodata.roi_pixel_size,
                                    y_pixel_scale=self.geodata.roi_pixel_size,
                                    window_size_factor=window_size_factor)
        self.plot_roi_shaded_relief_overlay(axes, 
                        color_alpha=self.streamline_shaded_relief_color_alpha,
                        hillshade_alpha=self.streamline_shaded_relief_hillshade_alpha)
        self.plot_classical_streamlines_overlay(axes)
        axes.set_xlim(left=self.geodata.roi_x_bounds[0],
                      right=self.geodata.roi_x_bounds[1])
        axes.set_ylim(bottom=self.geodata.roi_y_bounds[0],
                      top=self.geodata.roi_y_bounds[1])
        self._force_display(fig)
        self._record_fig('classical_streamlines',fig)

    def plot_classical_streamlines_and_vectors(self, window_size_factor=None, 
                                               sl_color=None,  vec_color=None, 
                                               vec_alpha=None, vec_scale=None):
        """
        Classic streamlines and gradient vector field on color shaded relief
          
        Args:
            TBD (TBD): 
        
        TBD
    
        Returns:
            TBD: 
            TBD
        """
        fig,axes = self._new_figure(x_pixel_scale=self.geodata.roi_pixel_size,
                                    y_pixel_scale=self.geodata.roi_pixel_size,
                                    window_size_factor=window_size_factor)
        self.plot_roi_shaded_relief_overlay(axes, 
                        color_alpha=self.streamline_shaded_relief_color_alpha,
                        hillshade_alpha=self.streamline_shaded_relief_hillshade_alpha)
        if sl_color is None:
            sl_color = self.downstreamline_color
        if vec_color is None:
            vec_color = self.gradient_vector_color        
        if vec_alpha is None:
            vec_alpha = self.gradient_vector_alpha       
        if vec_scale is None:
            vec_scale = self.gradient_vector_scale    
        self.plot_gradient_vector_field_overlay(axes, vec_color=vec_color,
                                                vec_alpha=vec_alpha, vec_scale=vec_scale)
        self.plot_classical_streamlines_overlay(axes, sl_color=sl_color)
        axes.set_xlim(left=self.geodata.roi_x_bounds[0],
                      right=self.geodata.roi_x_bounds[1])
        axes.set_ylim(bottom=self.geodata.roi_y_bounds[0],
                      top=self.geodata.roi_y_bounds[1])
        self._force_display(fig)
        self._record_fig('classical_streamlines_and_vectors',fig)

    def plot_classical_streamlines_overlay(self,axes, sl_color='blue'):
        """
        Classic streamlines (overlay method)
          
        Args:
            TBD (TBD): 
        
        TBD
    
        Returns:
            TBD: 
            TBD
        """
        x_pixel_centers_array,y_pixel_centers_array \
            = np.meshgrid(self.geodata.x_roi_n_pixel_centers,
                          self.geodata.y_roi_n_pixel_centers)
        axes.streamplot(x_pixel_centers_array,y_pixel_centers_array,
                      np.ma.array(self.preprocess.uv_array[:,:,0],
                                  mask=self.state.merge_active_masks()).T[
                        self.geodata.pad_width:-self.geodata.pad_width,
                        self.geodata.pad_width:-self.geodata.pad_width],
                      np.ma.array(self.preprocess.uv_array[:,:,1],
                                  mask=self.state.merge_active_masks()).T[
                        self.geodata.pad_width:-self.geodata.pad_width,
                        self.geodata.pad_width:-self.geodata.pad_width], 
                      density=min(1.0,self.classical_streamplot_density), 
                      color=sl_color, linewidth=self.classical_streamline_linewidth)
#                           color=self.geodata.roi_array, cmap='terrain')

    def plot_gradient_vector_field(self, window_size_factor=None, 
                                   vec_color='purple',vec_alpha=0.5, vec_scale=30):
        """
        Topographic gradient vector field on color shaded relief
          
        Args:
            TBD (TBD): 
        
        TBD
    
        Returns:
            TBD: 
            TBD
        """
        fig,axes = self._new_figure(x_pixel_scale=self.geodata.roi_pixel_size,
                                    y_pixel_scale=self.geodata.roi_pixel_size,
                                    window_size_factor=window_size_factor)
        self.plot_roi_shaded_relief_overlay(axes, 
                        color_alpha=self.streamline_shaded_relief_color_alpha,
                        hillshade_alpha=self.streamline_shaded_relief_hillshade_alpha)
        self.plot_gradient_vector_field_overlay(axes, vec_color=vec_color,
                                                vec_alpha=vec_alpha, vec_scale=vec_scale)
        self._force_display(fig)
        self._record_fig('gradient_vector_field',fig)

    def plot_gradient_vector_field_overlay(self,axes,
                                           vec_color='purple',vec_alpha=0.5,
                                           vec_scale=30):
        """
       Topographic gradient vector field (overlay method)
          
        Args:
            TBD (TBD): 
        
        TBD
    
        Returns:
            TBD: 
            TBD
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
                        ~self.state.merge_active_masks(),
                            decimation_factor,axis=0,n=2), 
                                decimation_factor,axis=1,n=2)
        else:
            u = self.preprocess.uv_array[:,:,0]
            v = self.preprocess.uv_array[:,:,1]
            m = ~self.state.merge_active_masks()
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
                    color=vec_color,
                    alpha=vec_alpha,
                    scale=vec_scale
                    )
            
    def plot_blockages_overlay(self,axes,color='k',shape='s'):
        """
        Blocked zone pixels
          
        Args:
            TBD (TBD): 
        
        TBD
    
        Returns:
            TBD: 
            TBD
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
          
        Args:
            TBD (TBD): 
        
        TBD
    
        Returns:
            TBD: 
            TBD
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
          
        Args:
            TBD (TBD): 
        
        TBD
    
        Returns:
            TBD: 
            TBD
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

    def plot_midslope_elevations_pdf(self):
        """
        Plot distribution of midslope as kernel-smoothed pdf
          
        Args:
            TBD (TBD): 
        
        TBD
    
        Returns:
            TBD: 
            TBD
        """
        fig_name = 'midslope_elevations_pdf'
        window_title = 'midslope_elevations_pdf'
        title = self.geodata.title+':  Elevation distribution'  
        fig,axes = self._new_figure(window_title=window_title, title=title)

        h_midline_pdf_array  = self.analysis.h_midline_pdf_array
        h_midline_pdf_max1   = self.analysis.h_midline_pdf_max1
        h_midline_pdf_max1_h = self.analysis.h_midline_pdf_max1_h
        h_all_pdf_array      = self.analysis.h_all_pdf_array
        h_array              = self.analysis.h_array
        sf                   = self.analysis.h_pdf_sf        
        plt.plot(h_midline_pdf_array/sf, h_array,label='midline', c='k')
        plt.plot(
            h_midline_pdf_max1/sf, 
            h_midline_pdf_max1_h, 
            'o', c='k', ms=8,
            label=rf"{np.int(np.round(h_midline_pdf_max1_h)):4}$\,$m"  
        )
        # plt.plot(h_midline_pdf_max2/sf, h_midline_pdf_max_h2, 's', c='k', ms=8,
        #          label='{:4}$\,$m'.format(np.int(h_midline_pdf_max_h2))  )
        plt.plot(h_all_pdf_array/sf, h_array,label='all', c='b')
        plt.autoscale(enable=True, tight=True, axis='y')
        plt.xlim(-0.002,)
        plt.grid('on',ls=':')
        plt.legend()
#         plt.title(title)
        plt.ylabel('Elevation $h$  [m]')
        plt.xlabel('Density $p(h)$  [rescaled m$^{-1}$]')
        self._record_fig(fig_name,fig)

    def plot_slope_angles_pdf(self):
        """
        Plot distribution of slope angles as kernel-smoothed pdf
          
        Args:
            TBD (TBD): 
        
        TBD
    
        Returns:
            TBD: 
            TBD
        """
        fig_name='slope_angles_pdf'
        window_title='slope_angles_pdf'
        title = self.geodata.title+':  Midslope angle distribution'   
        fig,axes = self._new_figure(window_title=window_title, title=title)
        
        slope_midline_pdf_array     = self.analysis.slope_midline_pdf_array
        slope_midline_pdf_max       = self.analysis.slope_midline_pdf_max
        slope_midline_pdf_max_slope = self.analysis.slope_midline_pdf_max_slope
        slope_array                 = self.analysis.slope_array        
        plt.plot(slope_midline_pdf_array/slope_midline_pdf_max, slope_array, c='k')
        plt.plot(
            slope_midline_pdf_max/slope_midline_pdf_max, 
            slope_midline_pdf_max_slope, 
            'o', 
            c='k',
            ms=8,
            label=rf"modal slope ={np.int(np.round(slope_midline_pdf_max_slope)):4}$^\circ$",
        )
        plt.autoscale(enable=True, tight=True, axis='y')
        plt.xlim(-0.002,)
        plt.grid('on',ls=':')
        plt.legend()
#         plt.title(title)
        plt.ylabel(r"Slope angle   $\\mathrm{atan}|\\nabla{h}|$  [$^{\circ}$]")
        plt.xlabel(r"Density  $p(\\mathrm{atan}|\\nabla{h}|)$  [rescaled 1/$^{\circ}$]")
        # plt.xlabel(
        #     'Relative frequency $p(|\\nabla{h}|)$  [rescaled 1/$^{\circ}$]')
        self._record_fig(fig_name,fig)
        
        
    def plot_distributions(self):
        """
        Plot probability distributions of processed streamline data.
          
        Args:
            TBD (TBD): 
        
        TBD
    
        Returns:
            TBD: 
            TBD
        """
        self.print('Plotting distributions...')

        # Marginal univariate pdfs
        marginals_list = ['dsla','usla','dslt','uslt','dslc','uslc']
        for marginal in marginals_list:
            marginal_fn = 'plot_marginal_pdf_'+marginal
            do_plot = getattr(self,'do_'+marginal_fn)
            if do_plot:
                try:
                    plot_fn = getattr(self,marginal_fn)
                    plot_fn()
                except Exception as error:
                    self.print('Failed in "plot_distributions":\n', error)

        # Joint bivariate pdfs
        joint_list = ['dsla_usla','usla_uslt','dsla_dslt','dslt_dslc','dslt_dsla',
                      'uslt_dslt','usla_uslc','dsla_dslc','uslc_dslc']
        for joint in joint_list:
            joint_fn = 'plot_joint_pdf_'+joint
            do_plot = getattr(self,'do_'+joint_fn)
            if do_plot:
                try:
                    plot_fn = getattr(self,joint_fn)
                    plot_fn()
                except Exception as error:
                    self.print('Failed in "plot_distributions":\n', error)

        self.print('...done')

    def plot_marginal_pdf(self, marginal_distbn, fig_name=None,
                          title='',x_label='',y_label=''):
        """
        TBD
          
        Args:
            TBD (TBD): 
        
        TBD
    
        Returns:
            TBD: 
            TBD
        """
        window_title=fig_name

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
            
        # Create graph
        fig,axes = self._new_figure(window_title=window_title, title=title)
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

        axes.set_xlim(left=0.99, right=x_max*1.001)
        axes.set_ylim(bottom=0,top=y_max*1.1)

        # Display & record
        self._force_display(fig)
        return self._record_fig(fig_name,fig)    
        
    def plot_marginal_pdf_dsla(self):
        """
        TBD
          
        Args:
            TBD (TBD): 
        
        TBD
    
        Returns:
            TBD: 
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
          
        Args:
            TBD (TBD): 
        
        TBD
    
        Returns:
            TBD: 
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
        return self.plot_marginal_pdf(marginal_distbn, fig_name=fig_name, 
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
                       fig_name=None, title='', swap_xy=False, do_nl_trend=False,
                       x_label='', y_label='', 
                       xsym_label = r'$L_m^{*}$', ysym_label = r'$\sqrt{A_e^*}$', 
                       do_plot_mode=True, do_overlay_histogram=False,
                       do_legend=True):
        """
        TBD
          
        Args:
            TBD (TBD): 
        
        TBD
    
        Returns:
            TBD: 
            TBD
        """
        window_title=fig_name

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
        fig,axes = self._new_figure(window_title=window_title, title=title)
        axes.set_xscale('log')
        axes.set_yscale('log')
        legend = []

        # Pdfs
        kde_pdf      = bivariate_distribution.pdf.copy()
        if swap_xy:
            kde_pdf  = kde_pdf.T
        kde_pdf = np.power(kde_pdf,self.joint_distbn_viz_scale)

        if do_overlay_histogram:
            from skimage.transform import downscale_local_mean
            downscale_factor = 2
            kde_hist    = downscale_local_mean(bivariate_distribution.histogram.copy(),
                                               (downscale_factor,downscale_factor))
            kde_hist[kde_hist>0] = 1
            kde_hist = np.ma.masked_where(kde_hist==0,kde_hist)
            n_hist_bins = bivariate_distribution.n_hist_bins
            x_hist_vec  = np.exp(np.linspace(np.log(x_min),np.log(x_max),
                                             n_hist_bins//downscale_factor))
            y_hist_vec  = np.exp(np.linspace(np.log(y_min),np.log(y_max),
                                             n_hist_bins//downscale_factor))        
            if swap_xy:
                kde_hist = kde_hist.T
        
        # Plot bivariate pdf - distorted for emphasis as appropriate
        axes.pcolormesh(x_vec,y_vec,kde_pdf.T, cmap='GnBu', 
                        antialiased=True, shading='gouraud')
        if do_overlay_histogram:
            axes.pcolormesh(x_hist_vec,y_hist_vec,kde_hist.T, cmap='gray', 
                            antialiased=False,alpha=0.5)
        axes.contour(x_vec,y_vec, kde_pdf.T, self.joint_distbn_n_contours,
                     colors='k',linewidths=1,alpha=0.5, antialiased=True)
        
        # Plot thresholds
        if do_legend:
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
        else:
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
                legend += [r'hillslope  $y =$'+'$x$']
                x = x_vec
            else:
                h_grad_str = '{0:0.2}'.format(h_grad)
                legend += [r'hillslope  $y =$'+'$x$']
                x = x_vec
            axes.plot(x,x*h_grad,'-.', color='crimson',linewidth=2,alpha=0.7)
    
            # Nonlinear trend y=x^n
            if do_nl_trend:
                h_grad = mode_xy[1]/mode_xy[0]
                h_grad = 1.0
                if not swap_xy:
                    h_grad_str = '{0:1.1}'.format(h_grad)
                    legend += [r'channel  $y =$'+'$x^{3/2}$']
                    x = x_vec
                    axes.plot(x,np.power(x,1.5),'--', 
                              color='magenta',linewidth=2,alpha=0.7)
                else:
                    h_grad_str = '{0:0.2}'.format(h_grad)
                    legend += [r'channel  $y =$'+'${x^{1/3}$']
                    x = x_vec
                    axes.plot(x,np.power(x,2.0/3),'--', #np.power(1.5,0.5)*
                              color='magenta',linewidth=2,alpha=0.7)

        # Presentation
#         loc = 'lower right'
        if do_legend:
            loc = 'upper left'
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                axes.legend(legend, loc=loc,fontsize=12,framealpha=0.97)
        axes.xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=15))
        axes.yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=15))
        if swap_xy:
            axes.set_xlabel(y_label)
            axes.set_ylabel(x_label)
        else:
            axes.set_xlabel(x_label)
            axes.set_ylabel(y_label)
        x_ticks = self._choose_ticks(x_min,x_max)
        y_ticks = self._choose_ticks(y_min,y_max)
        plt.xticks(x_ticks,x_ticks)
        plt.yticks(y_ticks,y_ticks)
        try:
            axes.set_xlim(left=0.999,right=x_max*1.001)
            axes.set_ylim(bottom=y_min*0.999,top=y_max*1.001)
        except:
            self.print('x,y min,max not provided')
        axes.grid(color='gray', linestyle='dotted', linewidth=0.5, which='both')
        
        # Push to screen
        self._force_display(fig)
        self._record_fig(fig_name,fig)

    def plot_joint_pdf_dsla_usla(self, do_legend=False):
        """
        TBD
        """
        fig_name = 'joint_pdf_dsla_usla'
        title = r'Streamline length distribution$ f(L_{md},L_{mu})$'
        x_label = r'Downstreamline mean length  $L_{md}$ [m]'
        y_label = r'Upstreamline mean length  $L_{mu}$ [m]'
        xsym_label = r'$L_{md}^{*}$'
        ysym_label = r'$L_{mu}^{*}$'
        try:
            joint_distbn = self.analysis.jpdf_dsla_usla
        except:
            self.print('"'+title+'" not computed: cannot plot')
            return
        try:
            mx_distbn    = self.analysis.mpdf_dsla
            my_distbn    = self.analysis.mpdf_usla
        except:
            mx_distbn    = None
            my_distbn    = None

        self.plot_joint_pdf(joint_distbn, mx_distbn=mx_distbn, my_distbn=my_distbn,
                            fig_name=fig_name, 
                            title=title, x_label=x_label, y_label=y_label,
                            xsym_label=xsym_label, ysym_label=ysym_label,
                            do_legend=do_legend)
            
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
                            fig_name=fig_name, do_nl_trend=True,
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
                            fig_name=fig_name, swap_xy=False, do_nl_trend=True,
                            title=title, x_label=x_label, y_label=y_label,
                            xsym_label=xsym_label, ysym_label=ysym_label,
                            do_plot_mode=True)

    def plot_joint_pdf_dslt_dsla(self,do_legend=True):
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
                            do_plot_mode=True, do_legend=do_legend)

    def plot_joint_pdf_uslt_dslt(self,swap_xy=False,do_legend=False):
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
                            xsym_label=xsym_label, ysym_label=ysym_label,
                            swap_xy=swap_xy, do_legend=do_legend)

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
                            fig_name=fig_name, do_nl_trend=True,
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
                            fig_name=fig_name, do_nl_trend=True,
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
        self._force_display(fig)
        self._record_fig(fig_name,fig)

