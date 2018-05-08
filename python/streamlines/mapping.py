"""
TBD
"""

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from skimage.morphology import skeletonize, thin, medial_axis, disk
from skimage.filters.rank import mean,modal,median
from skimage.filters import gaussian
import warnings
import sys

from os import environ
environ['PYTHONUNBUFFERED']='True'

from streamlines.core import Core
from streamlines import connect, countlink, label, segment, lengths
from streamlines.trace import Info
from streamlines.pocl import Initialize_cl

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
        self.mapping_array = self.trace.mapping_array
         
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
        
        # Measure mean streamline distances from midslope to channel pixels
        self.measure_hillslope_lengths()
        #    - no atomics
        
        # Measure mean streamline distances from midslope to channel pixels
        self.map_hillslope_lengths()
        #    - no GPU
        
        self.print('**Mapping end**\n')  
      
      
    def map_channels(self):
        self.print('Channels...',end='')  
        # Shorthand
        try:
            self.mapping_array
        except:
            self.mapping_array = self.trace.mapping_array
        jpdf = self.analysis.jpdf_dsla_dslt
        # Designate channel pixels according to dsla pdf analysis
        self.mapping_array[jpdf.mode_cluster_ij_list[1][:,0],
                           jpdf.mode_cluster_ij_list[1][:,1]] = self.trace.is_channel
        self.print('done')  

    def connect_channel_pixels(self):
        connect.connect_channel_pixels(
                self.cl_state, 
                Info(self.trace),
                self.geodata.basin_mask_array,
                self.preprocess.uv_array,
                self.mapping_array, 
                self.state.verbose 
                )
        
    def thin_channels(self):
        self.print('Thinning channels...',end='')  
        is_channel = self.trace.is_channel
        is_interchannel = self.trace.is_interchannel
        channel_array = np.zeros(self.mapping_array.shape, dtype=np.bool)
        channel_array[  ((self.mapping_array & is_channel)==is_channel)
                      | ((self.mapping_array & is_interchannel)==is_interchannel)
                     ] = True
        self.print('skeletonizing...',end='')  
        channel_array = skeletonize(channel_array)
        self.mapping_array[channel_array] |= self.trace.is_thinchannel
        self.print('done')  

    def map_channel_heads(self):
        connect.map_channel_heads(
                self.cl_state, 
                Info(self.trace),
                self.geodata.basin_mask_array,
                self.preprocess.uv_array,
                self.mapping_array, 
                self.state.verbose 
                )
        
    def count_downchannels(self):
        self.count_array = np.zeros(self.mapping_array.shape,dtype=np.uint32)
        self.link_array  = np.zeros(self.mapping_array.shape, dtype=np.uint32)
        countlink.count_downchannels(
                            self.cl_state, 
                            Info(self.trace),
                            self.geodata.basin_mask_array,
                            self.preprocess.uv_array,
                            self.mapping_array, self.count_array, self.link_array, 
                            self.state.verbose 
                            )

    def flag_downchannels(self):
        countlink.flag_downchannels(
                            self.cl_state, 
                            Info(self.trace),
                            self.geodata.basin_mask_array,
                            self.preprocess.uv_array,
                            self.mapping_array, self.count_array, self.link_array, 
                            self.state.verbose 
                            )

    def label_confluences(self):
        label.label_confluences(
                            self.cl_state,
                            Info(self.trace), 
                            self.geodata.basin_mask_array,
                            self.preprocess.uv_array,
                            self.trace.slt_array[:,:,0], 
                            self.mapping_array, self.count_array, self.link_array, 
                            self.state.verbose 
                            )

    def segment_downchannels(self):
        self.label_array = np.zeros(self.mapping_array.shape, dtype=np.uint32)
        self.n_segments \
            = segment.segment_channels(
                    self.cl_state,
                    Info(self.trace),
                    self.geodata.basin_mask_array,
                    self.preprocess.uv_array,
                    self.mapping_array,self.count_array,self.link_array,self.label_array,
                    self.state.verbose 
                    )
        # Save the channel-only segment labeling for now
        self.channel_label_array = self.label_array.copy().astype(np.uint32)
        is_majorconfluence = self.trace.is_majorconfluence
#         pdebug('Major confluences',self.label_array[(~self.geodata.basin_mask_array) 
#                  & ((self.mapping_array & is_majorconfluence)==is_majorconfluence)])
#         pdebug('Unique labels:',np.unique(self.label_array))
        
    def link_hillslopes(self):
        countlink.link_hillslopes(
                            self.cl_state,
                            Info(self.trace),
                            self.geodata.basin_mask_array,
                            self.preprocess.uv_array,
                            self.mapping_array,self.count_array,self.link_array,
                            self.state.verbose 
                            )

    def segment_hillslopes(self):
        segment.segment_hillslopes(
                self.cl_state,
                Info(self.trace),
                self.geodata.basin_mask_array,
                self.preprocess.uv_array,
                self.mapping_array,self.count_array,self.link_array,self.label_array,
                self.state.verbose 
                )

    def subsegment_flanks(self):
        segment.subsegment_flanks(
                self.cl_state,
                Info(self.trace),
                self.geodata.basin_mask_array,
                self.preprocess.uv_array,
                self.mapping_array,
                self.channel_label_array,self.link_array,self.label_array,
                self.state.verbose 
                )
        self.label_array = self.label_array.astype(dtype=np.int32)
        self.label_array[self.label_array<0] \
            = -(self.label_array[self.label_array<0] + self.trace.left_flank_addition)
        is_leftflank = self.trace.is_leftflank
        
    def map_midslope(self):
        self.print('Midslopes...',end='')  
        dsla = self.trace.sla_array[:,:,0]
        usla = self.trace.sla_array[:,:,1]
        mask = self.geodata.basin_mask_array

        midslope_array = np.zeros_like(dsla, dtype=np.bool)
        midslope_array[ (~mask) & (np.fabs(
            gaussian_filter((np.arctan2(dsla,usla)-np.pi/4), self.midslope_filter_sigma))
                             <=self.midslope_threshold)] = True
        self.mapping_array[midslope_array] \
            = self.mapping_array[midslope_array] | self.trace.is_midslope
        self.print('done')  

    def measure_hillslope_lengths(self):
        traj_label_array = (self.label_array[
                                (self.mapping_array&self.trace.is_midslope)>0
                    ].astype(np.int32)).ravel().copy()
#         pdebug('traj_label_array',traj_label_array.shape,traj_label_array)
        traj_length_array= 0.0*traj_label_array.copy().astype(dtype=np.float32)
        
        lengths.hillslope_lengths(
                self.cl_state,
                Info(self.trace),
                self.geodata.basin_mask_array,
                self.preprocess.uv_array,
                self.mapping_array,
                self.label_array, traj_length_array,
                self.state.verbose 
                )
        unique_labels = np.unique(traj_label_array)
        self.hillslope_labels \
            = unique_labels[unique_labels!=0].astype(np.int32)
        df  = pd.DataFrame(np.zeros((traj_length_array.shape[0],), 
                                    dtype=[('label', np.int32), ('length', np.float32)]))
        df['label']  = traj_label_array
        df['length'] = traj_length_array
        df = df[df.label!=0]
        self.hillslope_length_df = df
        
        stats_df = pd.DataFrame(self.hillslope_labels,columns=['label'])
        stats_list = ( ('count','count'),('mean','mean [m]'), ('std','stddev [m]') )
        for stat in stats_list:
            stats_df = stats_df.join( getattr(df.groupby('label'),stat[0])() ,on='label')
            stats_df.rename(index=str, columns={'length':stat[1]}, inplace=True)
        stats_df.set_index('label',inplace=True)
        self.hillslope_stats_df = stats_df
        
        self.hillslope_length_array = np.zeros_like(self.label_array, dtype=np.float32)
        for idx,row in stats_df.iterrows():     
            self.hillslope_length_array[self.label_array==idx] = row['mean [m]']

    def map_hillslope_lengths(self):
        self.print('Mapping hillslope lengths...',end='',flush=True)
        sys.stdout.flush()
        hsl = np.flipud(self.hillslope_length_array.T)
        hsl_bool = hsl.astype(np.bool)
        hsl_clipped = np.ma.array(hsl, mask=~hsl_bool)
        hsl_min = np.min(hsl_clipped)
        hsl_max = np.max(hsl_clipped)
        hsl_clipped = 65535*(hsl_clipped-hsl_min)/(hsl_max-hsl_min)
        hsl_masked = np.ma.array(hsl_clipped.astype(np.uint16),mask=~hsl_bool)

        median_radius = int(self.hillslope_length_median_radius
                            /self.geodata.roi_pixel_size)
        mean_radius   = int(self.hillslope_length_mean_radius
                            /self.geodata.roi_pixel_size)
        median_disk = disk(median_radius)
        mean_disk   = disk(mean_radius)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.print('median filtering with {0}m ({1}-pixel) diameter disk...'
                       .format(self.hillslope_length_median_radius,median_radius), 
                       end='',flush=True)
            sys.stdout.flush()
            hsl_median \
               = np.ma.array(median(hsl_masked,median_disk,mask=hsl_bool),mask=~hsl_bool)
            self.print('mean filtering with {0}m ({1}-pixel) diameter disk...'
                       .format(self.hillslope_length_mean_radius,mean_radius),
                       end='',flush=True) 
            sys.stdout.flush()
            hsl_median_nm = mean(hsl_median,mean_disk)
        self.hillslope_length_smoothed_array \
            = ((hsl_median_nm[self.geodata.pad_width:-self.geodata.pad_width,
                              self.geodata.pad_width:-self.geodata.pad_width]
                                .astype(np.float32))/65535)*(hsl_max-hsl_min)+hsl_min
        self.print('done')  
        