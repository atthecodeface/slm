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
from streamlines.useful import Data, Info, vprint, dilate, get_bbox
from streamlines        import connect, channelheads, countlink, label, \
                               segment, linkhillslopes, lengths
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
        self.verbose = self.state.verbose
        self.vbackup = self.state.verbose
        self.vprogress = self.state.verbose
    
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

    def _switch_to_quiet_mode(self):
        self.state.verbose = False
        self.verbose = False

    def _switch_back_to_verbose_mode(self):
        self.state.verbose = self.vbackup
        self.verbose = self.vbackup

    def report_progress(self ,idx, n_segments):
        progress = ((idx)/n_segments)*100.0
        vprint(self.vprogress, '{:2.1f}% '.format(progress),end='')

    def create_coarse_subsegment_mask(self, coarse_subsegment, is_left_or_right):
        # BBOX
        segment_mask_array = np.zeros_like(self.coarse_subsegment_array, dtype=np.bool)
        # Start with inverted raw mask
        # BBOX
#         pdebug(self.coarse_subsegment_array[self.coarse_subsegment_array!=0])
        segment_mask_array[self.coarse_subsegment_array==coarse_subsegment] = True
        # Dilate mask by 2 if left or 1 if right flank coarse subsegment
        #   - left dilation needs to encompass bordering right-flank channel pixels
        n_iterations = (2 if is_left_or_right=='left' else 1)
        # BBOX
        dilated_segment_mask_array=np.invert(dilate(segment_mask_array.copy(),
                                                    n_iterations=n_iterations))
#         pdebug('segment_mask_array.shape',segment_mask_array.shape, 
#                dilated_segment_mask_array.shape)
        # Now invert the raw mask as well
        segment_mask_array = np.invert(segment_mask_array)
        pad = self.geodata.pad_width
        padded_mask_array = np.pad(dilated_segment_mask_array, (pad,pad),
                                   'constant', constant_values=(True,True))
        # Define bbox
        bbox_dilated_segment, nx,ny = get_bbox(~dilated_segment_mask_array)
        self.print('Dilated subsegment mask bbox = {}'.format(bbox_dilated_segment))
        return segment_mask_array, dilated_segment_mask_array, bbox_dilated_segment

    def pass1(self):
        vprint(self.vprogress,'\n**Pass#1 begin**')
        self._switch_to_quiet_mode()
        # Shorthand
        pad = self.geodata.pad_width
        nxp = self.geodata.roi_nx+pad*2
        nyp = self.geodata.roi_ny+pad*2
        # Create arrays for mapping and coarse subsegmentation
        self.mapping_array           = np.zeros((nxp,nyp), dtype=np.uint32)
        self.coarse_subsegment_array = np.zeros((nxp,nyp), dtype=np.int32)
        self.merged_coarse_mask      = np.ones((nxp,nyp),  dtype=np.bool)
        # Create an info object for passing parameters to CL wrappers etc
        info = Info(self.state, self.trace, self.geodata.roi_pixel_size, mapping=self)
        # Revert to 'dtm', 'basin' (if set), and 'uv' masks only
        self.state.reset_active_masks()
        # Generate a combined mask array from this set
        mask_array = self.state.merge_active_masks()
        # Find the bounding box of this mask and its x,y size
        bbox, bnx, bny = get_bbox(~mask_array)
        # Record this bbox, its padding, its x,y size and the padded x,y size
        # BUG - need boundary masking
        info.set_xy(bnx,bny, pad)
        # Force coarse subsegmentation and related dummy channel thresholds
        info.set_thresholds(channel_threshold=self.coarse_channel_threshold,
                            segmentation_threshold=self.coarse_segmentation_threshold)
        # Create a data array container
        #   - these arrays are sliced from their source arrays bounded by the padded bbox
        #   - slicing bounds for simple arrays and for two-layer (up & downstream)
        #     arrays are recorded in data.bounds_grid and data.bounds_slx respectively. 
        data = Data( info=info, bbox=bbox, pad=pad,
                     mapping_array = self.mapping_array,
                     mask_array    = mask_array,
                     uv_array      = self.preprocess.uv_array,
                     sla_array     = self.trace.sla_array,
                     slc_array     = self.trace.slc_array,
                     slt_array     = self.trace.slt_array )
        # Do the forced coarse channel mapping & subsegmentation
        self.do_map_channels_segments(info, data)
        # Save the coarse subsegmentation labels
        #   - inserted into full size grid arrays using the data.bounds_grid slice
        self.mapping_array[data.bounds_grid]           = data.mapping_array
        self.coarse_subsegment_array[data.bounds_grid] = data.label_array
        # Make a list of all the subsegments with enough ridge/midslope pixels for HSL
        coarse_subsegments        = np.unique(data.label_array[~data.mask_array])
        self.coarse_subsegments   = np.sort(coarse_subsegments[coarse_subsegments!=0])
        self.n_coarse_subsegments = self.coarse_subsegments.shape[0]
        # Make a mask to select all coarse subsegments
        for label in self.coarse_subsegments:
            self.merged_coarse_mask[self.coarse_subsegment_array==label] = False
        # Copy the coarse subsegments so they can be readily visualized
        self.label_array = self.coarse_subsegment_array.copy()
        self._switch_back_to_verbose_mode()
        vprint(self.vprogress,'**Pass#1 end**') 

    def pass2(self):
        vprint(self.vprogress,'\n**Pass#2 begin**') 
        self._switch_to_quiet_mode()
        self.print('Subsegment labels: {}'.format(self.coarse_subsegments))
        # Mask off all but these coarse subsegments
        self.state.add_active_mask({'merged_coarse': self.merged_coarse_mask})
        # Count how many coarse subsegments need to be iterated over
        n_segments = self.n_coarse_subsegments
        # Shorthand
        pad = self.geodata.pad_width
        nxp = self.geodata.roi_nx+pad*2
        nyp = self.geodata.roi_ny+pad*2
        pixel_size = self.geodata.roi_pixel_size
        # Initialize the full ROI-scale HSL grid - delete first in case the np array
        #   already exists and to ensure memory is freed
        try:
            del self.hsl_array
        except:
            pass
        self.hsl_array = np.zeros((nxp,nyp), dtype=np.float32)
        tmp_hsl_array  = np.zeros((nxp,nyp), dtype=np.float32)  
        # Mapping array for channels etc
        self.mapping_array = np.zeros((nxp,nyp), dtype=np.uint32)        
#         for idx,coarse_subsegment in enumerate([71]):
        # Iterate over the coarse subsegments
        for idx,coarse_subsegment in enumerate(self.coarse_subsegments):
            # Report % progress
            self.report_progress(idx, n_segments)
            # Create metadata container for this coarse subsegment
            info = Info(self.state,self.trace, pixel_size, mapping=self)
            # Flag if this coarse subsegment is left or right flank
            #   - important because a left flank subseg omits the channel pixels
            is_left_or_right = ('left' if coarse_subsegment<0 else 'right')
            self.print('--- Mapping HSL on subsegment §{0} = {1}/{2} ({3})'
                       .format(coarse_subsegment,idx+1,n_segments,is_left_or_right))
            # Revert to 'dtm', 'basin' (if set), and 'uv' masks only
            self.state.reset_active_masks()
            # Convert this coarse subsegment labeled pixels into a mask
            #    - also dilate this pixel set and generate a wider mask 
            #      to ensure flank-adjacent channel pixels are incorporated 
            #    - dilate by 1 for R flank and by 2 for L flank to ensure this
            segment_mask_array, dilated_segment_mask_array, bbox \
                = self.create_coarse_subsegment_mask(coarse_subsegment, is_left_or_right)
            # Deploy the dilated coarse-subsegment mask
            self.state.add_active_mask({'dilated_segment': dilated_segment_mask_array})
            self.print('Dilated coarse subsegment mask bounding box: {}'.format(bbox))
            
            mask_array = self.state.merge_active_masks()
            bbox, bnx, bny = get_bbox(~mask_array)
            info.set_xy(bnx,bny, pad)

            # BBOX
            data = Data( info=info, bbox=bbox, pad=pad,
                         mapping_array = self.mapping_array,
                         mask_array    = mask_array,
                         uv_array      = self.preprocess.uv_array,
                         sla_array     = self.trace.sla_array,
                         slc_array     = self.trace.slc_array,
                         slt_array     = self.trace.slt_array )

            # Compute slt pdf and estimate channel threshold from it
            # BBOX
            channel_threshold \
                = self.analysis.estimate_channel_threshold(data, verbose=self.vbackup)
            # HACK - make adjustable
            if channel_threshold is None or channel_threshold<20.0: 
                continue
            info.set_thresholds(channel_threshold=channel_threshold,
                                segmentation_threshold=self.fine_segmentation_threshold)
            
            if not self.do_map_channels_segments(info, data):
                del data
                continue

            # Map ridges and midslopes
            self.map_midslopes(info, data)
            self.map_ridges(info, data)
            
            # Find the HSL-mappable subsegments
            self.select_subsegments(info, data)
            # Measure HSL from ridges or midslopes to thin channels per subsegment
            if not self.measure_hsl(info, data):
                del data
                continue
            vprint(self.vprogress,'Mean HSL = {0:0.1f}m'.format(self.hsl_mean))
            # Remove this iteration's dilated coarse-subsegment mask
            self.state.remove_active_mask('dilated_segment')
            
            # Deploy this coarse subsegment's HSL data to the 'global' HSL map
            # BBOX
            self.print('Merging hillslope lengths...',end='')
            # Deploy this iteration's raw coarse-subsegment mask
            self.state.add_active_mask({'raw_segment': segment_mask_array})
            bounds = data.bounds_grid
            mask_array = self.state.merge_active_masks()
            tmp_hsl_array.fill(0.0)
            tmp_hsl_array[bounds] = data.hsl_array
            tmp_hsl_array[np.isnan(tmp_hsl_array)] = 0.0
            self.hsl_array[~mask_array] += tmp_hsl_array[~mask_array]
            # Delete this iteration's raw coarse mask from the active list
            self.state.remove_active_mask('raw_segment')
            
            del segment_mask_array, dilated_segment_mask_array, data
                    
        self._switch_back_to_verbose_mode()
        self.report_progress(idx+1, n_segments)
        vprint(self.vprogress,'\n**Pass#2 end**') 
                
    def pass3(self):        
        vprint(self.vprogress,'\n**Pass#3 begin**') 
        self.state.add_active_mask({'merged_coarse': self.merged_coarse_mask})
        info = Info(self.state, self.trace, self.geodata.roi_pixel_size, mapping=self)
        info.set_xy(self.geodata.roi_nx,self.geodata.roi_ny,self.geodata.pad_width)
        # No data obj needed - grid arrays provided by this mapping instance
        self.map_hsl(info)
        self.map_aspect(info)
        self.compute_hsl_aspect(info)
#         self.state.remove_active_mask('merged_coarse')
        vprint(self.vprogress,'**Pass#3 end**') 
                
    def do_map_channels_segments(self, info, data):
        """
        TBD.
        """
        try:
            # Use downstream slt,sla pdfs to designate pixels as channels
            self.map_channels(info, data)
            # Join up disconnected channel pixels if they are not too widely spaced
            self.connect_channel_pixels(info, data)
            # Skeletonize channel pixels into thin network
            self.thin_channels(info, data)
            # Locate upstream ends of thinned channel network & designate as heads
            self.map_channel_heads(info, data)
            # Link downstream from channel heads
            self.count_downchannels(info, data)
            # Count downstream from channel heads
            self.flag_downchannels(info, data)
            # Map locations of channel confluences & designate types
            self.label_confluences(info, data)
            # Label channel segments with channel head idxs
            self.segment_downchannels(info, data)
            # Designate downstream linkages for all hillslope pixels
            self.link_hillslopes(info, data)
            # Label correspondingly upstream hillslope pixels
            self.segment_hillslopes(info, data)
            # Designate as L or R of channel to subsegment hillslope flanks
            self.subsegment_flanks(info, data)
            # Success
            return True
        except Exception as error:
            # Failure
            vprint(self.vbackup, 'Failed in "do_map_channels_segments":\n', error)
            raise
            return False
      
        
    def map_channels(self, info, data):
        self.print('Channels...',end='')
        # Designate channel pixels according to dslt pdf analysis
        slt = data.slt_array[:,:,0]
        slc = data.slc_array[:,:,0]
        # HACK slt*2>=slc 
        data.mapping_array[(slt>=info.channel_threshold) & (slt*2>=slc)]= info.is_channel   
        self.print('done')  

    def connect_channel_pixels(self, info, data):
        connect.connect_channel_pixels(self.cl_state,info,data,self.verbose)
        
    def thin_channels(self, info, data):
        self.print('Thinning channels...',end='')  
        mapping_array   = data.mapping_array
        nxp = info.nx_padded
        nyp = info.ny_padded
        channel_array   = np.zeros((nxp,nyp), dtype=np.bool)
        channel_array[  ((mapping_array & info.is_channel)==info.is_channel)
                      | ((mapping_array & info.is_interchannel)==info.is_interchannel)
                     ] = True
        self.print('skeletonizing...',end='')  
        skeleton_array = skeletonize(medial_axis(channel_array))
        mapping_array[skeleton_array] |= info.is_thinchannel
        self.print('done')  

    def map_channel_heads(self, info, data):
        channelheads.map_channel_heads(self.cl_state, info, data, self.verbose)
        # HACK - this step used to be necessary
#         channelheads.prune_channel_heads(self.cl_state, info, data, 
#                                          self.verbose)
        
    def count_downchannels(self, info, data):
        nxp = info.nx_padded
        nyp = info.ny_padded
        data.count_array = np.zeros((nxp,nyp), dtype=np.uint32)
        data.link_array  = np.zeros((nxp,nyp), dtype=np.uint32)
        countlink.count_downchannels(self.cl_state, info, data, self.verbose)
        
    def flag_downchannels(self, info, data):
        countlink.flag_downchannels(self.cl_state, info, data, self.verbose)

    def label_confluences(self, info, data):
        label.label_confluences(self.cl_state, info, data, self.verbose)
        # Three passes to try to eliminate all 'parasite' streamlets
        countlink.flag_downchannels(self.cl_state, info, data, self.verbose)
        countlink.flag_downchannels(self.cl_state, info, data, self.verbose, 
                                    do_reset_count=False)
        countlink.flag_downchannels(self.cl_state, info, data, self.verbose, 
                                    do_reset_count=False)
        
    def segment_downchannels(self, info, data):
        nxp = info.nx_padded
        nyp = info.ny_padded
        data.label_array = np.zeros((nxp,nyp), dtype=np.uint32)
        segment.segment_channels(self.cl_state, info, data, self.verbose)
        # Save the channel-only segment labeling for now
        data.channel_label_array = data.label_array.copy().astype(np.uint32)
        is_majorconfluence = info.is_majorconfluence
        
    def link_hillslopes(self, info, data):
        linkhillslopes.link_hillslopes(self.cl_state, info, data, self.verbose)

    def segment_hillslopes(self, info, data):
        segment.segment_hillslopes(self.cl_state, info, data, self.verbose )

    def subsegment_flanks(self, info, data):
        # Channel edges first, and then flanks
        segment.subsegment_flanks(self.cl_state, info, data, self.verbose)
        #?????
        data.label_array = data.label_array.astype(dtype=np.int32)
        data.label_array[data.label_array<0] \
            = -(  data.label_array[data.label_array<0] + info.left_flank_addition )
        
    def map_midslopes(self, info, data):
        self.print('Midslopes...',end='')  
        dsla = data.sla_array[:,:,0]
        usla = data.sla_array[:,:,1]
        mask = data.mask_array
        nxp = info.nx_padded
        nyp = info.ny_padded
        midslope_array = np.zeros((nxp,nyp), dtype=np.bool)
        midslope_array[ (~mask) & (np.fabs(
            gaussian_filter((np.arctan2(dsla,usla)-np.pi/4), self.midslope_filter_sigma))
                             <=self.midslope_threshold)] = True
        skeleton_midslope_array = skeletonize(midslope_array)
        data.mapping_array[skeleton_midslope_array] |= info.is_midslope
        self.print('done')  

    def map_ridges(self, info, data):
        self.print('Ridges...',end='')  
        dsla = data.sla_array[:,:,0]
        usla = data.sla_array[:,:,1]
        mask = data.mask_array
        nxp = info.nx_padded
        nyp = info.ny_padded
        ridge_array = np.zeros((nxp,nyp), dtype=np.bool)
        ridge_array[ (~mask) & ((
            gaussian_filter((np.arctan2(dsla,usla)-np.pi/4), self.ridge_filter_sigma))
                             <= -(np.pi/4)*self.ridge_threshold)] = True
        dilation_structure = generate_binary_structure(2, 2)
        fat_ridge_array    = binary_dilation(ridge_array, structure=dilation_structure, 
                                             iterations=1)
        filled_ridge_array   = binary_fill_holes(fat_ridge_array)
        skeleton_ridge_array = skeletonize(filled_ridge_array)
        data.mapping_array[skeleton_ridge_array] |= info.is_ridge
        self.print('done')  

    def select_subsegments(self, info, data):
        self.print('Selecting subsegments for HSL mapping...',end='')  
        if self.do_measure_hsl_from_ridges:
            flag = info.is_ridge
            self.print('measuring from ridges...')
        else:
            flag = info.is_midslope
            self.print('measuring from midslopes...')
        data.traj_label_array = data.label_array[
                 ((data.mapping_array & flag)>0) & (~data.mask_array)
                                    ].astype(np.int32)
        unique_labels         = np.unique(data.traj_label_array)
        self.hillslope_labels = unique_labels[unique_labels!=0].astype(np.int32)
        self.n_subsegments    = unique_labels.shape[0]
        self.print('selected {}'.format(self.n_subsegments))
        self.print('...done')  
                
    def measure_hsl(self, info, data):
        self.print('Measuring hillslope lengths...')
        data.traj_length_array = np.zeros_like(data.traj_label_array,dtype=np.float32)
        lengths.hsl(self.cl_state, info, data, self.verbose)
        df = pd.DataFrame(np.zeros((data.traj_length_array.shape[0],), 
                                    dtype=[('label', np.int32), ('length', np.float32)]))
        df['label']  = data.traj_label_array
        df['length'] = data.traj_length_array
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
        nxp = info.nx_padded
        nyp = info.ny_padded
        try:   
            data.hsl_array = np.zeros((nxp,nyp), dtype=np.float32)
            for idx,row in stats_df.iterrows():
                if row['count']>=self.n_hsl_averaging_threshold:
                    data.hsl_array[data.label_array==idx] = row['mean [m]']
                else:
                    data.hsl_array[data.label_array==idx] = 0
        except:
            self.print('Problem parsing HSL stats dataframe')
            return False

        if self.hsl_stats_df.empty:
            self.print('Unable to map HSL here - skipping')
            return False
        
        hsl_nonan = data.hsl_array[~np.isnan(data.hsl_array)]
        hsl_nonan = hsl_nonan[hsl_nonan>=self.hsl_averaging_threshold]
        if hsl_nonan.shape[0]==0:
            self.print('No HSL values in dataframe - skipping')
            return False
        
        self.hsl_mean  = np.mean(hsl_nonan)
        self.print('...done')  
        return True

    def map_hsl(self, info):
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
        
    def map_aspect(self, info):
        self.print('Computing hillslope aspect...',end='')

        nxp = info.nx_padded
        nyp = info.ny_padded
        slope_threshold = self.aspect_slope_threshold
        median_radius   = self.aspect_median_filter_radius/self.geodata.pixel_size
        slope_array     = self.preprocess.slope_array.copy()
        uv_array        = self.preprocess.uv_array
        mapping_array   = self.mapping_array
        mask_array      = np.zeros((nxp,nyp), dtype=np.bool)
        slope_array[((mapping_array & info.is_channel)==info.is_channel)] = 0.0
        if self.do_aspect_median_filtering:
            sf = np.max(slope_array)/255.0
            median_slope_array = sf*median(np.uint8(slope_array/sf),disk(median_radius))
            slope_array = median_slope_array

        median_radius   = self.uv_median_radius/self.geodata.pixel_size
        sf = 2.0/255.0
        uvx_array = sf*median(np.uint8((uv_array[:,:,0]+1.0)/sf),disk(median_radius))-1.0
        uvy_array = sf*median(np.uint8((uv_array[:,:,1]+1.0)/sf),disk(median_radius))-1.0
        mask_array[  (slope_array<slope_threshold)
                   | ((mapping_array & info.is_channel)==info.is_channel) ] = True
        self.aspect_array = np.ma.masked_array( np.arctan2(uvy_array,uvx_array), 
                                                mask=mask_array )
        self.print('done')  
                                                         
    def compute_hsl_aspect(self, info, n_bins=None, hsl_averaging_threshold=None):
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
