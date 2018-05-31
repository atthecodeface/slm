"""
TBD
"""

import numpy  as np
import pandas as pd
from skimage.morphology   import skeletonize, thin, medial_axis, disk
from skimage.filters      import gaussian
from skimage.filters.rank import mean,median
from scipy.ndimage            import gaussian_filter
from scipy.ndimage.morphology import binary_fill_holes
from scipy.ndimage.morphology import binary_dilation,generate_binary_structure
import warnings
import sys
from os import environ
environ['PYTHONUNBUFFERED']='True'

from streamlines.core  import Core
from streamlines       import connect, channelheads, countlink, label, \
                              segment, linkhillslopes, lengths
from streamlines.trace import Info, Data
from streamlines.pocl  import Initialize_cl

__all__ = ['Mapping']

pdebug = print

class Mapping(Core):
    """
    TBD.
    """
    def __init__(self,state,imported_parameters,geodata,preprocess,trace,analysis):
        """
        TBD
        """
        super().__init__(state,imported_parameters)  
        self.geodata = geodata
        self.preprocess = preprocess
        self.trace = trace
        self.analysis = analysis
        self.cl_state = Initialize_cl(self.state.cl_src_path, 
                                      self.state.cl_platform, 
                                      self.state.cl_device )
        
    def do(self):
        """
        TBD.
        """
        self.print('\n**Mapping begin**') 
        
        # Shorthand
        self.prepare()
         
        # Use downstream slt,sla pdfs to designate pixels as channels
        #    - no GPU invocation
        self.map_channels()
        
        # Join up disconnected channel pixels if they are not too widely spaced
        #    - atomic_or
        self.connect_channel_pixels()
        
        # Skeletonize channel pixels into thin network
        #    - no GPU invocation
        self.thin_channels()
        
        # Locate upstream ends of thinned channel network & designate as heads
        #    - atomic_and, atomic_or
        self.map_channel_heads()
        
        # Link downstream from channel heads
        #    - atomic_or mapping_array[]
        #    - atomic_xchg count_array[], link_array[]
        self.count_downchannels()
                
        # Count downstream from channel heads
        #    - atomic_or mapping_array[]
        #    - atomic_max count_array[]
        self.flag_downchannels()
        
        # Map locations of channel confluences & designate types
        #    - atomic_or
        self.label_confluences()
        
        # Label channel segments with channel head idxs
        #    - atomic_xchg label_array[]
        self.segment_downchannels()
        
        # Designate downstream linkages for all hillslope pixels
        #    - atomic_xchg link_array[]
        self.link_hillslopes()
        
        # Label correspondingly upstream hillslope pixels
        #    - atomic_xchg label_array[]
        self.segment_hillslopes()
        
        # Designate as L or R of channel to subsegment hillslope flanks
        #    - no atomics
        self.subsegment_flanks()

        # Use up and downstream sla to designate midslope pixels
        #    - no GPU invocation
        self.map_midslope()
        
        # Use up and downstream sla to designate ridge pixels
        #    - no GPU invocation
        self.map_ridges()
        
        # Measure mean streamline distances from midslope to channel pixels
        self.measure_hsl()
        #    - no atomics
        
        # Measure mean streamline distances from midslope to channel pixels
        self.map_hsl()
        #    - no GPU
        
        # Gradient-thresholded hillslope horizontal orientation = aspect
        self.map_aspect()
        self.compute_hsl_aspect()
        
        self.print('**Mapping end**\n')  
        
    def prepare(self):
        self.print('Preparing...',end='')  
        # Shorthand
        try:
            del(self.mapping_array)
            del(self.data)
            del(self.label_array)
            del(self.hsl_array)
            del(self.hsl_smoothed_array)
        except:
            pass
        self.mapping_array = self.trace.mapping_array.copy()
        self.data = Data( mask_array    = self.geodata.basin_mask_array,
                          uv_array      = self.preprocess.uv_array,
                          mapping_array = self.mapping_array )  
        self.verbose = self.state.verbose
        self.info = Info(self.trace, mapping=self)

        self.print('done')    
            
    def map_channels(self):
        self.print('Channels...',end='')  
        slt_threshold = self.analysis.mpdf_dslt.channel_threshold_x
        # Designate channel pixels according to dslt pdf analysis
        self.data.mapping_array[  (self.trace.slt_array[:,:,0]>=slt_threshold)
                        & (self.trace.slt_array[:,:,0]*2>=self.trace.slc_array[:,:,0])
                                ] = self.info.is_channel                                
        self.print('done')  

    def connect_channel_pixels(self):
        connect.connect_channel_pixels(self.cl_state,self.info,self.data,self.verbose)
        
    def thin_channels(self):
        self.print('Thinning channels...',end='')  
        is_channel      = self.info.is_channel
        is_interchannel = self.info.is_interchannel
        mapping_array   = self.data.mapping_array
        channel_array   = np.zeros_like(mapping_array, dtype=np.bool)
        channel_array[  ((mapping_array & is_channel)==is_channel)
                      | ((mapping_array & is_interchannel)==is_interchannel)
                     ] = True
        self.print('skeletonizing...',end='')  
        skeleton_array = skeletonize(medial_axis(channel_array))
        mapping_array[skeleton_array] |= self.info.is_thinchannel
        self.print('done')  

    def map_channel_heads(self):
        channelheads.map_channel_heads(self.cl_state, self.info, self.data, 
                                       self.verbose)
        mapping_array = self.data.mapping_array
        channelheads.prune_channel_heads(self.cl_state, self.info, self.data, 
                                         self.verbose)
        
    def count_downchannels(self):
        self.data.count_array = np.zeros_like(self.mapping_array, dtype=np.uint32)
        self.data.link_array  = np.zeros_like(self.mapping_array, dtype=np.uint32)
        countlink.count_downchannels(self.cl_state, self.info, self.data, 
                                     self.verbose)
        
    def flag_downchannels(self):
        countlink.flag_downchannels(self.cl_state, self.info, self.data,
                                    self.verbose)

    def label_confluences(self):
        self.data.dn_slt_array = self.trace.slt_array[:,:,0].copy()
        label.label_confluences(self.cl_state, self.info, self.data, self.verbose)
        # Three passes to try to eliminate all 'parasite' streamlets
        countlink.flag_downchannels(self.cl_state, self.info, self.data,
                                    self.verbose)
        countlink.flag_downchannels(self.cl_state, self.info, self.data,
                                    self.verbose, do_reset_count=False)
        countlink.flag_downchannels(self.cl_state, self.info, self.data,
                                    self.verbose, do_reset_count=False)
        
    def segment_downchannels(self):
        self.data.label_array = np.zeros_like(self.mapping_array, dtype=np.uint32)
        self.n_segments = segment.segment_channels(self.cl_state, self.info, 
                                                   self.data, self.verbose)
        # Save the channel-only segment labeling for now
        self.data.channel_label_array = self.data.label_array.copy().astype(np.uint32)
        is_majorconfluence = self.info.is_majorconfluence
        
#         thinchannel_array = np.zeros_like(self.mapping_array, dtype=np.bool)
#         thinchannel_array[(self.mapping_array & self.info.is_thinchannel)!=0] |= True
#         skeleton_thinchannel_array = skeletonize(thinchannel_array)
#         self.mapping_array[(self.mapping_array & self.info.is_thinchannel)!=0] \
#             ^= self.info.is_thinchannel
#         self.mapping_array[skeleton_thinchannel_array] |= self.info.is_thinchannel        

    def link_hillslopes(self):
        linkhillslopes.link_hillslopes(self.cl_state, self.info, self.data,
                                       self.verbose)

    def segment_hillslopes(self):
        segment.segment_hillslopes(self.cl_state, self.info, self.data, 
                                   self.verbose )

    def subsegment_flanks(self):
        segment.subsegment_flanks(self.cl_state, self.info, self.data, self.verbose)
        self.data.label_array = self.data.label_array.astype(dtype=np.int32)
        self.data.label_array[self.data.label_array<0] \
            = - (  self.data.label_array[self.data.label_array<0] 
                 + self.info.left_flank_addition )
        self.label_array = self.data.label_array
        
    def map_midslope(self):
        self.print('Midslopes...',end='')  
        dsla = self.trace.sla_array[:,:,0]
        usla = self.trace.sla_array[:,:,1]
        mask = self.data.mask_array

        midslope_array = np.zeros_like(dsla, dtype=np.bool)
        midslope_array[ (~mask) & (np.fabs(
            gaussian_filter((np.arctan2(dsla,usla)-np.pi/4), self.midslope_filter_sigma))
                             <=self.midslope_threshold)] = True
#         dilation_structure = generate_binary_structure(2, 2)
#         fat_midslope_array = binary_dilation(midslope_array, 
#                                              structure=dilation_structure, iterations=1)
#         filled_midslope_array = binary_fill_holes(midslope_array)
        skeleton_midslope_array = skeletonize(midslope_array)
#         fat_midslope_array = binary_dilation(skeleton_midslope_array, 
#                                              structure=dilation_structure, iterations=1)
        self.data.mapping_array[skeleton_midslope_array] |= self.info.is_midslope
        self.print('done')  

    def map_ridges(self):
        self.print('Ridges...',end='')  
        dsla = self.trace.sla_array[:,:,0]
        usla = self.trace.sla_array[:,:,1]
        mask = self.data.mask_array
        ridge_array = np.zeros_like(dsla, dtype=np.bool)
        ridge_array[ (~mask) & ((
            gaussian_filter((np.arctan2(dsla,usla)-np.pi/4), self.ridge_filter_sigma))
                             <= -(np.pi/4)*self.ridge_threshold)] = True
        dilation_structure = generate_binary_structure(2, 2)
        fat_ridge_array = binary_dilation(ridge_array, 
                                          structure=dilation_structure, iterations=1)
        filled_ridge_array = binary_fill_holes(fat_ridge_array)
        skeleton_ridge_array = skeletonize(filled_ridge_array)
#         fat_ridge_array = binary_dilation(skeleton_ridge_array, 
#                                           structure=dilation_structure, iterations=1)
        self.data.mapping_array[skeleton_ridge_array] |= self.info.is_ridge
        self.print('done')  

    def measure_hsl(self):
        if self.do_measure_hsl_from_ridges:
            flag = self.info.is_ridge
        else:
            flag = self.info.is_midslope
        self.data.traj_label_array = (self.data.label_array[
                                       ((self.data.mapping_array & flag)>0)
                                       &   (~self.data.mask_array)
                                                ].astype(np.int32)).ravel().copy()
        # BUG sort of - cleaner ways to create a zero array than this
        self.data.traj_length_array \
            = np.zeros_like(self.data.traj_label_array,dtype=np.float32)
        
        lengths.hsl(self.cl_state, self.info, self.data, self.verbose)

        unique_labels = np.unique(self.data.traj_label_array)
        self.hillslope_labels \
            = unique_labels[unique_labels!=0].astype(np.int32)
        df  = pd.DataFrame(np.zeros((self.data.traj_length_array.shape[0],), 
                                    dtype=[('label', np.int32), ('length', np.float32)]))
        df['label']  = self.data.traj_label_array
        df['length'] = self.data.traj_length_array
        df = df[df.label!=0]
        self.hsl_df = df
        
        stats_df = pd.DataFrame(self.hillslope_labels,columns=['label'])
        stats_list = ( ('count','count'),('mean','mean [m]'), ('std','stddev [m]') )
        for stat in stats_list:
            stats_df = stats_df.join( getattr(df.groupby('label'),stat[0])() ,on='label')
            stats_df.rename(index=str, columns={'length':stat[1]}, inplace=True)
        stats_df.set_index('label',inplace=True)
        self.hsl_stats_df = stats_df
        
        self.hsl_array=np.zeros_like(self.data.label_array,dtype=np.float32)
        for idx,row in stats_df.iterrows():
            if row['count']>=self.n_hsl_averaging_threshold:
                self.hsl_array[self.data.label_array==idx] = row['mean [m]']
            else:
                self.hsl_array[self.data.label_array==idx] = 0

    def map_hsl(self):
        self.print('Mapping hillslope lengths...',end='',flush=True)
        sys.stdout.flush()
        
        hsl = np.flipud(self.hsl_array.T)
        hsl_bool = hsl.astype(np.bool)
        hsl_clipped = np.ma.array(hsl, mask=~hsl_bool)
        hsl_min = np.min(hsl_clipped)
        hsl_max = np.max(hsl_clipped)
        hsl_clipped = 65535*(hsl_clipped-hsl_min)/(hsl_max-hsl_min)
        hsl_masked = np.ma.array(hsl_clipped.astype(np.uint16),mask=~hsl_bool)

        median_radius = int(self.hsl_median_radius
                            /self.geodata.roi_pixel_size)
        mean_radius   = int(self.hsl_mean_radius
                            /self.geodata.roi_pixel_size)
        median_disk = disk(median_radius)
        mean_disk   = disk(mean_radius)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.print('median filtering with {0}m ({1}-pixel) diameter disk...'
                       .format(self.hsl_median_radius,median_radius), 
                       end='',flush=True)
            if self.state.verbose:
                sys.stdout.flush()
            # Strangely, mask logic is backwards for median(): 
            #    - true pixels are used (median-filtered), while false are untouched
            if median_radius==0:
                hsl_median = hsl_masked
            else:
                hsl_median = np.ma.array(median(hsl_masked,median_disk,mask=hsl_bool),
                                         mask=~hsl_bool)
            self.print('mean filtering with {0}m ({1}-pixel) diameter disk...'
                       .format(self.hsl_mean_radius,mean_radius),
                       end='',flush=True) 
            if self.state.verbose:
                sys.stdout.flush()
            hsl_median_nm = mean(hsl_median,mean_disk)
            
        self.hsl_smoothed_array \
            = ((hsl_median_nm[self.geodata.pad_width:-self.geodata.pad_width,
                              self.geodata.pad_width:-self.geodata.pad_width]
                                .astype(np.float32))/65535)*(hsl_max-hsl_min)+hsl_min
        self.print('done')  
        
    def map_aspect(self):
        self.print('Computing hillslope aspect...',end='',flush=True)
        sys.stdout.flush()

        slope_threshold = self.aspect_slope_threshold
        median_radius   = self.aspect_median_filter_radius
        slope_array     = self.preprocess.slope_array.copy()
        uv_array        = self.preprocess.uv_array.copy()
        is_channel      = self.info.is_channel
        mapping_array   = self.data.mapping_array
        mask_array      = np.zeros_like(mapping_array, dtype=np.bool)
        slope_array[((mapping_array & is_channel)==is_channel)] = 0.0
        if self.do_aspect_median_filtering:
            sf = np.max(slope_array)/255.0
            median_slope_array = sf*median(np.uint8(slope_array/sf),disk(median_radius))
            slope_array = median_slope_array
        uv_filter_width = 3      
        uv_array[:,:,0] = gaussian_filter(uv_array[:,:,0],uv_filter_width)
        uv_array[:,:,1] = gaussian_filter(uv_array[:,:,1],uv_filter_width)
        mask_array[ (slope_array<slope_threshold)
                   | ((mapping_array & is_channel)==is_channel) ] = True
        self.aspect_array = np.ma.masked_array(
                 np.arctan2(uv_array[:,:,1],uv_array[:,:,0]),
                                       mask=mask_array )
        self.print('done')  
                                                         
    def compute_hsl_aspect(self, n_bins=None):
        self.print('Computing hillslope length-aspect function...',end='',flush=True)
        sys.stdout.flush()
        
#         aspect_range = np.pi
        aspect_range = 180
        if n_bins is None:
            n_bins = 60
        pad = self.geodata.pad_width
        aspect_array = np.rad2deg(self.aspect_array[pad:-pad,pad:-pad].copy())
        hsl_array = self.hsl_smoothed_array.T.copy()
        aspect_array = aspect_array[hsl_array>0.0]
        hsl_array    = hsl_array[hsl_array>0.0]
        hsl_array    = hsl_array[np.abs(aspect_array)>0.0]
        aspect_array = aspect_array[np.abs(aspect_array)>0.0]
        mask_array   = np.ma.getmaskarray(aspect_array)| np.ma.getmaskarray(hsl_array)
        hsl_aspect_array = np.stack(
                       (np.ma.masked_array(hsl_array,    mask=mask_array).ravel(),
                        np.ma.masked_array(aspect_array, mask=mask_array).ravel()),
                        axis=1)
        # Sort in-place using column 1 (aspect) as key
        self.hsl_aspect_array = hsl_aspect_array[hsl_aspect_array[:,1].argsort()]
        self.hsl_aspect_df = pd.DataFrame(data=self.hsl_aspect_array,
                                          columns=['hsl','aspect'])
        half_bin = aspect_range/n_bins
        bins = np.linspace(-aspect_range,+aspect_range+2*half_bin,n_bins+2)-half_bin
#         pdebug('\n',(bins))
        self.hsl_aspect_df['groups'] = pd.cut(self.hsl_aspect_df['aspect'], bins)
        self.hsl_aspect_averages = self.hsl_aspect_df.groupby('groups')['hsl'].mean()
        bins = np.deg2rad(bins[:-1]+half_bin)
        hsl = self.hsl_aspect_averages.values
        south_hsl = (hsl[0]+hsl[-1])/2.0
        hsl[0] = south_hsl
        hsl[-1] = south_hsl
#         pdebug(hsl[0],hsl[-1],(hsl[0]+hsl[-1])/2.0)
#         pdebug(np.rad2deg(bins))
#         pdebug(self.hsl_aspect_averages)
        self.hsl_aspect_averages_array = np.stack((hsl,bins),axis=1)
#         pdebug(self.hsl_aspect_averages_array)
        self.print('done')  
                                                         
