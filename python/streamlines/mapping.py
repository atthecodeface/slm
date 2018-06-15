"""
TBD
"""

import numpy  as np
import pandas as pd
from cmath import rect, polar
from sklearn.preprocessing import normalize
from skimage.morphology    import skeletonize, thin, medial_axis, disk
from skimage.filters       import gaussian
from skimage.filters.rank  import mean, median
from scipy.ndimage            import gaussian_filter
from scipy.ndimage.morphology import binary_fill_holes, grey_dilation, \
                                     binary_dilation, generate_binary_structure
import warnings
from os import environ
environ['PYTHONUNBUFFERED']='True'

from streamlines.core   import Core
from streamlines.useful import Data, Info, vprint
from streamlines        import connect, channelheads, countlink, label, \
                               segment, linkhillslopes, lengths, useful
from streamlines.pocl   import Initialize_cl

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
    
    def _augment(self, plot):
        self.plot = plot
     
    def do(self):
        """
        TBD.
        """
        self.print('\n**Mapping begin**') 
        self.pass1()
        self.pass2()
        self.pass3()
        self.print('**Mapping end**\n')  

    def prepare_arrays_data_mask(self):
        self.print('Preparing...',end='')  
        self.verbose = self.state.verbose
        try:
            del self.mapping_array
        except:
            pass
        try:
            del self.data
        except:
            pass
        try:
            del self.label_array
        except:
            pass
        self.mapping_array = np.zeros_like(self.trace.mapping_array).astype(np.uint32)
        self.data = Data( mask_array    = self.state.merge_active_masks(),
                          uv_array      = self.preprocess.uv_array,
                          mapping_array = self.mapping_array )  
        self.print('done')    
            
    def _switch_to_quiet_mode(self):
        self.verbose_backup = self.state.verbose
        self.verbose = True
        self.state.verbose = self.state.very_verbose

    def _switch_back_to_verbose_mode(self):
        self.state.verbose = self.verbose_backup

    def get_bbox(self, array):
        x = np.any(array, axis=0)
        y = np.any(array, axis=1)
        x_min, x_max = np.where(x)[0][[0, -1]]
        y_min, y_max = np.where(y)[0][[0, -1]]
        return x_min,x_max, y_min,y_max

    def pass1(self):
        self.print('\n**Pass#1 begin**')
        self._switch_to_quiet_mode()
        # Only deploy border padding, uv-error, and basin+?height-threshold masks
        self.state.reset_active_masks()
        # Ensure a fresh start with data, mapping sub-objects
        self.prepare_arrays_data_mask()
        # Create an info object for passing parameters to CL wrappers etc
        self.info = Info(self.state, self.geodata, self.trace, mapping=self)
        # Force bogus channel delineation at coarse scale
        self.info.channel_threshold=self.coarse_channel_threshold
        # Force subsegmentation at coarse scale
        self.info.segmentation_threshold=self.coarse_segmentation_threshold
        # Do the forced coarse channel mapping & subsegmentation
        self.do_map_channels_segments()
        # Save the coarse subsegmentation labels
        self.coarse_label_array = self.label_array
        # Make a list of all the subsegments with enough ridge/midslope pixels for HSL
        self.select_subsegments(do_without_ridges_midslopes=True)
        self.coarse_labels = np.sort(self.hillslope_labels)
        # Generate a mask of all pixels outside the listed subsegments
        self.merged_coarse_mask = np.ones_like(self.coarse_label_array,dtype=np.bool)
        for label in self.coarse_labels:
            self.merged_coarse_mask[self.coarse_label_array==label] = False
        self._switch_back_to_verbose_mode()
        self.print('**Pass#1 end**') 

    def pass2(self):
        self.print('\n**Pass#2 begin**') 
        self._switch_to_quiet_mode()
        self.print('Subsegment labels: {}'.format(self.coarse_labels))
        # Mask off all but these coarse subsegments
        self.state.add_active_mask({'merged_coarse': self.merged_coarse_mask})
        # Initialize the HSL grid
        try:
            del self.hsl_array
        except:
            pass
        self.hsl_array = None
        # Count how many coarse subsegments need to be iterated over
        n_segments = self.coarse_labels.shape[0]
#         for idx,coarse_label in enumerate([71]):
#         vprint(True, '{:2.1f}% '.format(0/n_segments),end='')
        # Iterate over the coarse subsegments
        for idx,coarse_label in enumerate(self.coarse_labels):
            # Flag if subsegment is left or right flank
            #   - important because a left flank subseg omits the channel pixels
            is_left_or_right = ('left' if coarse_label<0 else 'right')
            self.print('\n--- Mapping HSL on subsegment §{0} = {1}/{2} ({3})'
                       .format(coarse_label,idx+1,n_segments,is_left_or_right))

            # Basic masking first
            self.state.reset_active_masks()
            # Create a mask array for this coarse subsegment and add to active list
            segment_mask_array = np.zeros_like(self.coarse_label_array, dtype=np.bool)
            # Start with inverted raw mask
            segment_mask_array[self.coarse_label_array==coarse_label] = True
            # Spread mask by 2 if left or 1 if right flank subsegment
            #   (left spread needs to 1st encompass bordering right-flank channel pixels)
            n_iterations = (2 if is_left_or_right=='left' else 1)
            dilated_segment_mask_array=np.invert(useful.dilate(segment_mask_array.copy(),
                                                              n_iterations=n_iterations))
            # Now invert the raw mask as well
            segment_mask_array = np.invert(segment_mask_array)
            # Deploy this iteration's dilated coarse-subsegment mask
            self.state.add_active_mask({'dilated_segment':dilated_segment_mask_array})
            
            # Define bbox
            bbox_dilated_segment = self.get_bbox(~dilated_segment_mask_array)
            self.print('Dilated subsegment mask bbox = {}'.format(bbox_dilated_segment))

            # Report % progress
            progress = ((idx+1)/n_segments)*100.0
            vprint(self.verbose_backup, '{:2.1f}% '.format(progress),end='')

            # Get ready to map HSL
            self.prepare_arrays_data_mask()
            self.info = Info(self.state, self.geodata, self.trace, mapping=self)
            self.info.segmentation_threshold=self.fine_segmentation_threshold
            # Compute slt pdf and estimate channel threshold from it
            try:
                self.info.channel_threshold = self.analysis.estimate_channel_threshold()
            except: 
                self.print('Failed channel threshold estimation')
                self.state.remove_active_mask('dilated_segment')
                continue

            # Map channel heads, thin channel pixels, subsegments
            try:
                self.do_map_channels_segments()
            except:
                self.print('Failed during segment/channel mapping')
                self.state.remove_active_mask('dilated_segment')
                continue
                
            # Map ridges and midslopes
            self.map_midslopes()
            self.map_ridges()
            # Find the HSL-mappable subsegments
            self.select_subsegments(do_without_ridges_midslopes=False)
            self.print('Selected {} subsegments'.format(self.n_subsegments))

            # Measure HSL from ridges or midslopes to thin channels per subsegment
            if not self.measure_hsl():
                self.state.remove_active_mask('dilated_segment')
                continue
            vprint(self.verbose_backup, 'Mean HSL = {0:0.1f}m'.format(self.hsl_mean))

            # Remove this iteration's dilated coarse-subsegment mask
            self.state.remove_active_mask('dilated_segment')
            # Replace with this iteration's raw coarse-subsegment mask
            self.state.add_active_mask({'raw_segment': segment_mask_array})
            # Deploy this coarse subsegment's HSL data to the 'global' HSL map
            self.merge_hsl()
            # Delete this iteration's raw coarse mask from the active list
            self.state.remove_active_mask('raw_segment')
            del segment_mask_array, dilated_segment_mask_array
                    
        self._switch_back_to_verbose_mode()
        self.print('\n**Pass#2 end**') 
                
    def pass3(self):        
        self.print('\n**Pass#3 begin**') 
        self.state.add_active_mask({'merged_coarse': self.merged_coarse_mask})
        self.map_hsl()
        self.map_aspect()
        self.compute_hsl_aspect()
        self.state.remove_active_mask('merged_coarse')
        self.print('**Pass#3 end**') 
                
    def do_map_channels_segments(self):
        """
        TBD.
        """
        # Use downstream slt,sla pdfs to designate pixels as channels
        self.map_channels()
        # Join up disconnected channel pixels if they are not too widely spaced
        self.connect_channel_pixels()
        # Skeletonize channel pixels into thin network
        self.thin_channels()
        # Locate upstream ends of thinned channel network & designate as heads
        self.map_channel_heads()
        # Link downstream from channel heads
        self.count_downchannels()
        # Count downstream from channel heads
        self.flag_downchannels()
        # Map locations of channel confluences & designate types
        self.label_confluences()
        # Label channel segments with channel head idxs
        self.segment_downchannels()
        # Designate downstream linkages for all hillslope pixels
        self.link_hillslopes()
        # Label correspondingly upstream hillslope pixels
        self.segment_hillslopes()
        # Designate as L or R of channel to subsegment hillslope flanks
        self.subsegment_flanks()
      
        
    def map_channels(self):
        self.print('Channels...',end='')
        # Designate channel pixels according to dslt pdf analysis
        self.data.mapping_array[  
                          (self.trace.slt_array[:,:,0]>=self.info.channel_threshold)
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
        # Hack
#         channelheads.prune_channel_heads(self.cl_state, self.info, self.data, 
#                                          self.verbose)
        
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
        # Channel edges first, and then flanks
        segment.subsegment_flanks(self.cl_state, self.info, self.data, self.verbose)
        self.data.label_array = self.data.label_array.astype(dtype=np.int32)
        self.data.label_array[self.data.label_array<0] \
            = - (  self.data.label_array[self.data.label_array<0] 
                 + self.info.left_flank_addition )
        self.label_array = self.data.label_array
        
    def map_midslopes(self):
        self.print('Midslopes...',end='')  
        dsla = self.trace.sla_array[:,:,0]
        usla = self.trace.sla_array[:,:,1]
        mask = self.data.mask_array

        midslope_array = np.zeros_like(dsla, dtype=np.bool)
        midslope_array[ (~mask) & (np.fabs(
            gaussian_filter((np.arctan2(dsla,usla)-np.pi/4), self.midslope_filter_sigma))
                             <=self.midslope_threshold)] = True
        skeleton_midslope_array = skeletonize(midslope_array)
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
        fat_ridge_array    = binary_dilation(ridge_array, 
                                             structure=dilation_structure, iterations=1)
        filled_ridge_array   = binary_fill_holes(fat_ridge_array)
        skeleton_ridge_array = skeletonize(filled_ridge_array)
        self.data.mapping_array[skeleton_ridge_array] |= self.info.is_ridge
        self.print('done')  

    def select_subsegments(self, do_without_ridges_midslopes=False):
        self.print('Selecting subsegments for HSL mapping...',end='')  
        if do_without_ridges_midslopes:
            self.data.traj_label_array \
                = self.data.label_array[~self.data.mask_array].astype(np.int32).ravel()
        else:
            if self.do_measure_hsl_from_ridges:
                flag = self.info.is_ridge
                self.print('measuring from ridges...')
            else:
                flag = self.info.is_midslope
                self.print('measuring from midslopes...')
            self.data.traj_label_array = self.data.label_array[
                     ((self.data.mapping_array & flag)>0) & (~self.data.mask_array)
                                        ].astype(np.int32).ravel()
        unique_labels         = np.unique(self.data.traj_label_array)
        self.hillslope_labels = unique_labels[unique_labels!=0].astype(np.int32)
        self.n_subsegments    = unique_labels.shape[0]
        self.print('...done')  
                
    def measure_hsl(self):
        self.print('Measuring hillslope lengths...')
        self.data.traj_length_array \
            = np.zeros_like(self.data.traj_label_array,dtype=np.float32)
        lengths.hsl(self.cl_state, self.info, self.data, self.verbose)
        
        df  = pd.DataFrame(np.zeros((self.data.traj_length_array.shape[0],), 
                                    dtype=[('label', np.int32), ('length', np.float32)]))
        df['label']  = self.data.traj_label_array
        df['length'] = self.data.traj_length_array
        df = df[df.label!=0]
        self.hsl_df = df
        
        try:
            stats_df = pd.DataFrame(self.hillslope_labels,columns=['label'])
            stats_list = ( ('count','count'), ('mean','mean [m]'), ('std','stddev [m]') )
            for stat in stats_list:
                stats_df = stats_df.join( getattr(df.groupby('label'),stat[0])() ,on='label')
                stats_df.rename(index=str, columns={'length':stat[1]}, inplace=True)
            stats_df.set_index('label',inplace=True)
            self.hsl_stats_df = stats_df
        except:
            self.print('Problem constructing HSL stats dataframe')
            return False
            
        try:   
            self.data.hsl_array=np.zeros_like(self.data.label_array,dtype=np.float32)
            for idx,row in stats_df.iterrows():
                if row['count']>=self.n_hsl_averaging_threshold:
                    self.data.hsl_array[self.data.label_array==idx] = row['mean [m]']
                else:
                    self.data.hsl_array[self.data.label_array==idx] = 0
        except:
            self.print('Problem parsing HSL stats dataframe')
            return False

        if self.hsl_stats_df.empty:
            self.print('Unable to map HSL here - skipping')
            return False
        
        hsl_nonan = self.data.hsl_array[~np.isnan(self.data.hsl_array)]
        hsl_nonan = hsl_nonan[hsl_nonan>=self.hsl_averaging_threshold]
        if hsl_nonan.shape[0]==0:
            self.print('No HSL values in dataframe - skipping')
            return False
        
        self.hsl_mean  = np.mean(hsl_nonan)
        self.print('...done')  
        return True

    def merge_hsl(self):
        self.print('Merging hillslope lengths...',end='')
        mask_array = self.state.merge_active_masks()
        if self.hsl_array is None:
            self.hsl_array = np.zeros_like(self.data.hsl_array)
            self.print('created HSL array...',end='')
        else:
            self.print('merging with prior HSL...',end='')
        self.hsl_array[(~mask_array) & (~np.isnan(self.hsl_array))] \
            += self.data.hsl_array[(~mask_array) & (~np.isnan(self.hsl_array))]
        self.print('done')  

    def map_hsl(self):
        self.print('Mapping hillslope lengths...',end='')
        
        hsl         = self.hsl_array.copy()
        mask        = self.state.merge_active_masks()
        pad         = self.geodata.pad_width
        hsl_min     = np.min(hsl)
        hsl_max     = np.max(hsl)
        hsl_clipped = (65535*(hsl-hsl_min)/(hsl_max-hsl_min)).astype(np.uint16)

        mdr       = int(self.hsl_mean_radius/self.geodata.roi_pixel_size)
        mean_disk = disk(mdr)
        dfw       = int(self.hsl_dilation_width/self.geodata.roi_pixel_size)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if dfw==0:
                hsl_dilated = hsl_clipped
            else:
                self.print('dilation with {0}m ({1}-pixel) width filter...'
                           .format(self.hsl_dilation_width,dfw), end='')
                hsl_dilated = grey_dilation(hsl_clipped,size=(dfw,dfw))
                hsl_dilated[~mask] = hsl_clipped[~mask]

            self.print('mean filtering with {0}m ({1}-pixel) diameter disk...'
                       .format(self.hsl_mean_radius*2,mdr*2), end='') 
            hsl_mean = mean(hsl_dilated,mean_disk)
            
        self.hsl_smoothed_array \
            = (((hsl_mean[pad:-pad,pad:-pad].astype(np.float32))/65535)
                                *(hsl_max-hsl_min)+hsl_min)
        self.print('done')  
        
    def map_aspect(self):
        self.print('Computing hillslope aspect...',end='')

        slope_threshold = self.aspect_slope_threshold
        median_radius   = self.aspect_median_filter_radius/self.geodata.pixel_size
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

        median_radius   = self.uv_median_radius/self.geodata.pixel_size
        sf = 2.0/255.0
        uvx_array = sf*median(np.uint8((uv_array[:,:,0]+1.0)/sf),disk(median_radius))-1.0
        uvy_array = sf*median(np.uint8((uv_array[:,:,1]+1.0)/sf),disk(median_radius))-1.0
        mask_array[  (slope_array<slope_threshold)
                   | ((mapping_array & is_channel)==is_channel) ] = True
        self.aspect_array = np.ma.masked_array( np.arctan2(uvy_array,uvx_array), 
                                                mask=mask_array )
        self.print('done')  
                                                         
    def compute_hsl_aspect(self, n_bins=None, hsl_averaging_threshold=None):
        self.print('Computing hillslope length-aspect function...',end='')
        
        # Default to e.g. aspect bins 6° wide
        if n_bins is None:
            n_bins = self.n_aspect_bins
        # HSL value below hsl_averaging_threshold will be ignored
        if hsl_averaging_threshold is None:
            hsl_averaging_threshold = self.hsl_averaging_threshold
        # Shorthand
        pad = self.geodata.pad_width
        # Use basic masks plus the merged coarse-subsegmentation mask
        self.state.reset_active_masks()
        self.state.add_active_mask({'merged_coarse': self.merged_coarse_mask})
        # Also exclude any np.ma masked pixels in self.aspect_array
        mask_array   =   self.state.merge_active_masks()[pad:-pad,pad:-pad] \
                       | self.aspect_array[pad:-pad,pad:-pad].mask
        # Fetch non-masked aspect and HSL values
        aspect_array = self.aspect_array[pad:-pad,pad:-pad][~mask_array]
        hsl_array    = self.hsl_smoothed_array[~mask_array]
        # Convert aspect to degrees
        aspect_array = np.rad2deg(aspect_array[hsl_array>=hsl_averaging_threshold])
        # Exclude "negligibly" small HSL values aka near zero mismeasurements
        hsl_array    = hsl_array[hsl_array>=hsl_averaging_threshold]
        # Combine into HSL(aspect) array
        hsl_aspect_array = np.stack( (hsl_array,aspect_array), axis=1)
        # Sort in-place using column 1 (aspect) as key
        self.hsl_aspect_array = hsl_aspect_array[hsl_aspect_array[:,1].argsort()]
        # Convert into a pandas dataframe for easier processing
        self.hsl_aspect_df    = pd.DataFrame(data=self.hsl_aspect_array,
                                             columns=['hsl','aspect'])
        # Likely 3° bin half-width and 6° bin width
        half_bin_width = 180/n_bins
        bin_width = half_bin_width*2
        # Vector of all bin edges, e.g., -183°,-177°,...,+177°,+183°
        bins = np.linspace(-180,+180+bin_width,n_bins+2)-half_bin_width
        # Group HSL values using degree-valued bin edges
        self.hsl_aspect_df['groups'] = pd.cut(self.hsl_aspect_df['aspect'], bins)
        # Average HSL values in each group
        self.hsl_aspect_averages = self.hsl_aspect_df.groupby('groups')['hsl'].mean()
        hsl = self.hsl_aspect_averages.values
        # Force HSL(±180°) values to match
        west_hsl = (hsl[0]+hsl[-1])/2.0
        hsl[0]  = west_hsl
        hsl[-1] = west_hsl
        # Convert bins to radians
        bins = np.deg2rad(bins[:-1]+half_bin_width)
        # Combine into average-HSL(aspect) array
        self.hsl_aspect_averages_array = np.stack((hsl,bins),axis=1)
        # Shorthand
        haa = self.hsl_aspect_averages_array
        hsl = haa[~np.isnan(haa[:,0]),0]
        asp = haa[~np.isnan(haa[:,0]),1]
        # Split into N and S HSL arrays
        hsl_south_array = hsl[asp<=0.0]
        hsl_north_array = hsl[asp>=0.0]
        # If we have HSL() at +180° and -180°, don't repeat in mean
        if -asp[0]==asp[-1]:
            self.hsl_mean = np.mean(hsl[1:])
        else:
            self.hsl_mean = np.mean(hsl)
        # Mean HSL north and south
        self.hsl_mean_south = (np.mean(hsl_south_array) if hsl_south_array!=[] else 0.0)
        self.hsl_mean_north = (np.mean(hsl_north_array) if hsl_north_array!=[] else 0.0)
        # Disparity between north and south means
        self.hsl_ns_disparity = np.abs(self.hsl_mean_north-self.hsl_mean_south)
        self.hsl_ns_disparity_normed = self.hsl_ns_disparity/self.hsl_mean
        # Overall HSL standard deviation
        self.hsl_stddev = np.std(hsl)
        # Split HSL(aspect) into n_hsl_split groups e.g. 4 to compute std devn
        hsl_split = np.array([np.std(hsl_split[~np.isnan(hsl_split)])
                              if hsl_split[~np.isnan(hsl_split)]!=[] else 0.0
                              for hsl_split in np.split(haa[1:,0],self.n_hsl_split)  ])
        # Compute mean split std devn
        self.hsl_split_stddev = (np.mean(hsl_split) if hsl_split!=[] else 0.0)
        # Normalize split std devn by mean = coefficient of variation
        self.hsl_split_stddev_normed = self.hsl_split_stddev/self.hsl_mean
        # Eliminate NaNs from hsl_aspect_averages_array
        haa[np.isnan(haa[:,0])]=0.0
        # Convert HSL(aspect) vectors into complex numbers
        hsl_complex_vec_array = (np.array([rect(ha[0],ha[1]) for ha in haa]))
        hsl_complex_vec_array = hsl_complex_vec_array[~np.isnan(hsl_complex_vec_array)]
        # Compute the mean complex HSL vector
        hsl_mean_complex_vector = np.mean(hsl_complex_vec_array)
        # Convert the mean complex HSL vector back into a polar vector HSL,aspect
        mhsl = polar(hsl_mean_complex_vector)
#         pdebug()
#         pdebug(hsl_complex_vec_array)
#         pdebug('hsl_mean_complex_vector',hsl_mean_complex_vector)
#         pdebug('mhsl',mhsl)
        self.hsl_mean_magnitude = mhsl[0]
        self.hsl_mean_azimuth = np.rad2deg(mhsl[1])
        # Calculate a confidence measure for any N-S disparity
        #   - effectively a signal:noise ratio 
        #     defined as reciprocal coefficient of variation
        #         =    (N HSL_mean - S HSL_mean)/(mean split HSL std devn)
        self.hsl_ns_disparity_confidence = self.hsl_ns_disparity/self.hsl_split_stddev

        self.print('done')  
        
    def check_hsl_ns_disparity(self):
        is_n_or_s_disparity \
            = ('north' if self.hsl_mean_north>self.hsl_mean_south else 'south')
        self.print('HSL mean ± σ(split), σ(all):'
                   +'\t      {:2.1f}m'
                   .format(self.hsl_mean)
                   +' ± {:2.1f}m (split)'
                   .format(self.hsl_split_stddev)
                   +' {:2.1f}m (all)'
                   .format(self.hsl_stddev) )
        self.print('HSL N-S disparity:'
                   +'\t\t      {0:2.1f}mN vs {1:2.1f}mS   ∆≈{2:2.1f}m'
                   .format(self.hsl_mean_north, self.hsl_mean_south,
                           self.hsl_ns_disparity) )
        self.print('HSL N-S rel disparity vs variation: '
                   +'  {0:2.1f}% ({1:2.1f}m) NS vs {2:2.1f}% ({3:2.1f}m)'
                   .format(self.hsl_ns_disparity_normed*100,
                           self.hsl_ns_disparity,
                           self.hsl_split_stddev_normed*100,
                           self.hsl_split_stddev) )
        self.print('HSL N-S disparity signal-noise ratio: ∆/σ = {0:2.1f}'
                   .format(self.hsl_ns_disparity_confidence) )
        indent = '\t'*5
        if self.hsl_ns_disparity_confidence<0.5:
            self.print(indent+'=> very weak {} disparity'.format(is_n_or_s_disparity))
        elif self.hsl_ns_disparity_confidence<1.0:
            self.print(indent+'=> weak {} disparity'.format(is_n_or_s_disparity))
        elif self.hsl_ns_disparity_confidence<2.0:
            self.print(indent+'=> moderate {} disparity'.format(is_n_or_s_disparity))
        else:
            self.print(indent+'=> strong {} disparity'.format(is_n_or_s_disparity))                                                         
