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
from streamlines.useful import Data, Info, vprint, dilate, get_bbox, npamem
from streamlines        import connect, channelheads, countlink, label, \
                               segment, hillslopes, lengths
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
        # Estimate the DTM-wide channel threshold and use it 
        #   to coarsely subsegment into "moderate"-size watersheds
        #   - where "sub" means split into L and R flanks along channels
        self.pass1()
        # Iterate over the coarse subsegments and in each:
        #   - estimate the channel threshold
        #   - map channels, ridges & midslopes
        #   - measure HSL from either ridges or midslopes to channels
        #   - merge the HSL and (TBD) channel mapping into "global" results grid(s)
        self.pass2()
        # Filter the HSL results into a smoothed, contourable grid
        # Compute filtered terrain aspect and combine with this HSL grid
        # Generate a mean HSL(aspect) function and related statistics
        #    to determine whether or not there is a N-S bias
        self.pass3()
        self.print('**Mapping end**\n')  

    def pass1(self):
        vprint(self.vprogress,'\n**Pass#1 begin**')
#         self._switch_to_quiet_mode()
        # Shorthand
        pad = self.geodata.pad_width
        nxp = self.geodata.roi_nx+pad*2
        nyp = self.geodata.roi_ny+pad*2
        # Create arrays for mapping and coarse subsegmentation
        self.mapping_array           = np.zeros((nxp,nyp), dtype=np.uint32)
        self.coarse_subsegment_array = np.zeros((nxp,nyp), dtype=np.int32)
        self.merged_coarse_mask_array      = np.ones((nxp,nyp),  dtype=np.bool)
        # Create an info object for passing parameters to CL wrappers etc
        info = Info(self.state, self.trace, self.geodata.roi_pixel_size, mapping=self)
        # Revert to 'dtm', 'basin' (if set), and 'uv' masks only
        self.state.reset_active_masks()
        # Generate a combined mask array from this set
        mask_array = self.state.merge_active_masks()
        # Find the bounding box of this mask and its x,y size
        bbox, nxb, nyb = get_bbox(~mask_array)
        # Record this bbox, its padding, its x,y size and the padded x,y size
        # BUG - need boundary masking
        info.set_xy(nxb,nyb, pad)
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
#         pdebug('merged_coarse_mask_array',self.merged_coarse_mask_array[self.merged_coarse_mask_array==False].shape)
#         pdebug('coarse_subsegment_array',self.coarse_subsegment_array[self.coarse_subsegment_array!=0].shape)
        for label in self.coarse_subsegments:
#             pdebug('merging',label)
            self.merged_coarse_mask_array[self.coarse_subsegment_array==label] = False
#             pdebug(self.merged_coarse_mask_array[self.merged_coarse_mask_array==False].shape)
        # Copy the coarse subsegments so they can be readily visualized
        self.label_array = self.coarse_subsegment_array.copy()
#         self._switch_back_to_verbose_mode()
        del info, data
        self.state.reset_active_masks()
        
        mask_array = self.state.merge_active_masks()
        bbox, nxb,nyb = get_bbox(~mask_array)
        info = Info(self.state, self.trace, self.geodata.roi_pixel_size, mapping=self)
        info.set_xy(nxb,nyb, pad)
        info.set_thresholds(segmentation_threshold=self.fine_segmentation_threshold)
        data = Data( info=info, bbox=bbox, pad=pad,
                     mapping_array = self.mapping_array,
                     mask_array    = mask_array,
                     uv_array      = self.preprocess.uv_array,
                     slc_array     = self.trace.slc_array,
                     slt_array     = self.trace.slt_array,
                     sla_array     = self.trace.sla_array )
        # Map ridges and midslopes
        self.map_midslopes(info, data)
        self.map_ridges(info, data)
        self.info = info
        bounds = data.bounds_grid
        self.mapping_array[bounds][~mask_array[bounds]] \
                |= data.mapping_array[~mask_array[bounds]]

        vprint(self.vprogress,'**Pass#1 end**') 

    def make_coarse_subsegment_masks(self, coarse_subsegment, is_left_or_right,
                                     raw_mask, dilated_mask):
        # Initialize raw mask with masked everywhere
        raw_mask.fill(True)
        # Unmask this coarse segment
        raw_mask[self.coarse_subsegment_array==coarse_subsegment] = False
        # Dilate this coarse segment mask by 2 if left or 1 if right flank
        #   - cos left dilation needs to grow to cover right-flank channel pixels
        n = (2 if is_left_or_right=='left' else 1)
        dilate(~raw_mask,n_iterations=n, out=dilated_mask)
        np.invert(dilated_mask, out=dilated_mask)
        # Shorthand
        pad = self.geodata.pad_width
        # Ensure dilated unmask does not encroach on boundary mask
        dilated_mask[:pad,:]  = True
        dilated_mask[:,:pad]  = True
        dilated_mask[-pad:,:] = True
        dilated_mask[:,-pad:] = True
        # Define bbox and dimensions of the dilated mask
        bbox_dilated_mask, nxd,nyd = get_bbox(~dilated_mask)
        return bbox_dilated_mask, nxd,nyd

    def pass2(self):
        vprint(self.vprogress,'\n**Pass#2 begin**') 
        self._switch_to_quiet_mode()
        # Shorthand
        pad = self.geodata.pad_width
        nxp = self.geodata.roi_nx+pad*2
        nyp = self.geodata.roi_ny+pad*2
        pixel_size = self.geodata.roi_pixel_size
        # Count how many coarse subsegments need to be iterated over
        n_segments = self.n_coarse_subsegments
        self.print('Subsegment labels: {}'.format(n_segments))
        # Mask off all but these coarse subsegments
        self.state.add_active_mask({'merged_coarse': self.merged_coarse_mask_array})
        # Initialize the full ROI-scale HSL grid and a buffer
        # Also a mapping array for channels etc
        raw_mask_array     = np.zeros((nxp,nyp), dtype=np.bool)
#         pdebug('self.geodata.roi_nx,ny = ',(self.geodata.roi_nx,self.geodata.roi_ny))
#         pdebug('nxp,nyp = ',(nxp,nyp))
#         pdebug('raw_mask_array',raw_mask_array.shape)
        dilated_mask_array = np.zeros((nxp,nyp), dtype=np.bool)
        merged_mask_array  = np.zeros((nxp,nyp), dtype=np.bool)
        self.hsl_array     = np.zeros((nxp,nyp), dtype=np.float32)
        # Iterate over the coarse subsegments
        for idx,coarse_subsegment in enumerate(self.coarse_subsegments):
#         for idx,coarse_subsegment in enumerate([71]):
            if idx>0:
                merged_mask_array.fill(False)
            # Report % progress
            self.report_progress(idx, n_segments, subsegment=coarse_subsegment)
            # Revert to 'dtm', 'basin' (if set), and 'uv' masks only
            self.state.reset_active_masks()
            # Flag if this coarse subsegment is left or right flank
            #   - important because a left flank subseg omits the channel pixels
            is_left_or_right = ('left' if coarse_subsegment<0 else 'right')
            # Convert this coarse subsegment-labeled pixels into a mask
            #    - also dilate this pixel set and generate a wider mask 
            #      to ensure flank-adjacent channel pixels are incorporated 
            #    - dilate by 1 for R flank and by 2 for L flank to ensure this
            bbox, nxb,nyb \
                = self.make_coarse_subsegment_masks(coarse_subsegment, is_left_or_right,
                                                    raw_mask_array, dilated_mask_array)
            # Create metadata container for this coarse subsegment
            info = Info(self.state,self.trace, pixel_size, mapping=self)
            info.set_xy(nxb,nyb, pad)
            self.print('--- Mapping HSL on subsegment §{0} = {1}/{2} ({3})'
                       .format(coarse_subsegment,idx+1,n_segments,is_left_or_right))
            # Deploy the dilated coarse-subsegment mask
            self.state.add_active_mask({'dilated_segment': dilated_mask_array})
            self.state.merge_active_masks(out=merged_mask_array)
            self.print('Dilated coarse subsegment mask bounding box: {}'.format(bbox))

            data = Data( info=info, bbox=bbox, pad=pad,
                         mapping_array = self.mapping_array,
                         mask_array    = merged_mask_array,
                         uv_array      = self.preprocess.uv_array,
                         slc_array     = self.trace.slc_array,
                         slt_array     = self.trace.slt_array,
                         sla_array     = self.trace.sla_array )

            # Compute slt pdf and estimate channel threshold from it
            channel_threshold \
                = self.analysis.estimate_channel_threshold(data, verbose=self.vbackup)
            # Don't HSL map if there's a problem with channel threshold estimation here
            if channel_threshold is None or channel_threshold<self.min_channel_threshold: 
                vprint(self.vprogress,'   ---')
                continue
            info.set_thresholds(channel_threshold=channel_threshold,
                                segmentation_threshold=self.fine_segmentation_threshold)
            
            # Big step - map subsegments using above channel threshold
            if not self.do_map_channels_segments(info, data):
                del data
                vprint(self.vprogress,'   ---')
                continue
            
            # Find the HSL-mappable subsegments
            self.select_subsegments(info, data)
            # Measure HSL from ridges or midslopes to thin channels per subsegment
            if not self.measure_hsl(info, data):
                del data
                vprint(self.vprogress,'   ---')
                continue
            vprint(self.vprogress,'Mean HSL = {0:0.1f}m'.format(self.hsl_mean))
            
            # Remove this iteration's dilated coarse-subsegment mask
            self.state.remove_active_mask('dilated_segment')
            # Deploy this coarse subsegment's HSL data to the 'global' HSL map
            self.print('Merging hillslope lengths...',end='')
            # Use raw coarse mask to only keep HSL values actually on coarse subsegment
            self.state.add_active_mask({'raw_segment': raw_mask_array})
            self.state.merge_active_masks(out=merged_mask_array)
            # Merge this coarse subsegment's HSL values into the 'global' HSL map
            bounds = data.bounds_grid
            data.hsl_array[np.isnan(data.hsl_array)] = 0.0
            self.hsl_array[bounds][~merged_mask_array[bounds]] \
                += data.hsl_array[~merged_mask_array[bounds]]
            # Merge the channel mapping results for this coarse subsegment
            #   - which will allow recomputation of the entire channel network
            #     during pass#3
            self.mapping_array[bounds][~merged_mask_array[bounds]] \
                |= data.mapping_array[~merged_mask_array[bounds]]
            # Delete this iteration's raw coarse mask from the active list
            self.state.remove_active_mask('raw_segment')
            # Erase the working instance of data to ensure 
            #   that the next iteration starts fresh
            del data
                    
        self._switch_back_to_verbose_mode()
        self.report_progress(idx+1, n_segments)
        vprint(self.vprogress,'\n**Pass#2 end**') 
                
    def pass3(self):        
        vprint(self.vprogress,'\n**Pass#3 begin**') 
        self.state.reset_active_masks()
        self.state.add_active_mask({'merged_coarse': self.merged_coarse_mask_array})
        mask_array = self.state.merge_active_masks()
        bbox, nxb,nyb = get_bbox(~mask_array)
        pad = self.geodata.pad_width
        info = Info(self.state, self.trace, self.geodata.roi_pixel_size, mapping=self)
        info.set_xy(nxb,nyb, pad)
        info.set_thresholds(segmentation_threshold=self.fine_segmentation_threshold)
        data = Data( info=info, bbox=bbox, pad=pad,
                     mapping_array = self.mapping_array,
                     mask_array    = mask_array,
                     uv_array      = self.preprocess.uv_array,
                     slc_array     = self.trace.slc_array,
                     slt_array     = self.trace.slt_array,
                     sla_array     = self.trace.sla_array )
        if not self.do_map_channels_segments(info, data, do_map_channels=False):
            self.print('Error occurred during channel mapping')
        del data.mapping_array, data.slc_array, data.slt_array

        # Revert to working on the whole DTM (ROI), mainly because 
        #   extrapolation (dilation) into masked pixels needs to spread 
        #   as far as needed, which is user-defined and may be quite large
        #   - thus not worth the trouble to add bboxing complexity here
        nx = self.geodata.roi_nx
        ny = self.geodata.roi_ny
        info.set_xy(nx,ny, pad)
        # Filter HSL into a contourable, somewhat smooth grid
        self.map_hsl(info)
        # Map smoothed terrain aspect aka orientation relative to east from uv array
        self.map_aspect(info)
        # Calculate an empirical HSL(aspect) function and related statistics
        self.compute_hsl_aspect(info)
#         self.state.remove_active_mask('merged_coarse')
        self.info = info
        vprint(self.vprogress,'**Pass#3 end**') 
                
    def do_map_channels_segments(self, info, data, do_map_channels=True):
        """
        TBD.
        """
        try:
            # Flag to report failure at any point
            is_success = True
            # Use downstream slt,sla pdfs to designate pixels as channels
            if do_map_channels:
                self.map_channels(info, data)
            # Join up disconnected channel pixels if they are not too widely spaced
            is_success &= self.connect_channel_pixels(info, data)
            # Skeletonize channel pixels into thin network
            self.thin_channels(info, data)
            # Locate upstream ends of thinned channel network & designate as heads
            is_success &= self.map_channel_heads(info, data)
            # Link downstream from channel heads
            is_success &= self.count_downchannels(info, data)
            # Count downstream from channel heads
            is_success &= self.flag_downchannels(info, data)
            if not is_success:
                return False
            # Map locations of channel confluences & designate types
            is_success &= self.label_confluences(info, data)
            # Label channel segments with channel head idxs
            is_success &= self.segment_downchannels(info, data)
            # Designate downstream linkages for all hillslope pixels
            is_success &= self.link_hillslopes(info, data)
            # Label correspondingly upstream hillslope pixels
            is_success &= self.segment_hillslopes(info, data)
            # Designate as L or R of channel to subsegment hillslope flanks
            is_success &= self.subsegment_flanks(info, data)
            # Success
            return is_success
        except Exception as error:
            # Failure
            vprint(self.vbackup, 'Failed in "do_map_channels_segments":\n', error)
            raise
            return False


    def _switch_to_quiet_mode(self):
        self.state.verbose = False
        self.verbose = False

    def _switch_back_to_verbose_mode(self):
        self.state.verbose = self.vbackup
        self.verbose = self.vbackup

    def report_progress(self, idx, n_segments, subsegment=None):
        progress = ((idx)/n_segments)*100.0
        if subsegment is not None:
            vprint(self.vprogress, '{0:2.1f}% {1:+} '.format(progress,subsegment),end='')
        else:
            vprint(self.vprogress, '{0:2.1f}% '.format(progress),end='')


    def map_channels(self, info, data):
        self.print('Channels...',end='')
        # Designate channel pixels according to dslt pdf analysis
        slt = data.slt_array[:,:,0]
        slc = data.slc_array[:,:,0]
        # HACK slt*2>=slc 
        data.mapping_array[(slt>=info.channel_threshold) & (slt*2>=slc)]= info.is_channel   
        self.print('done')  

    def connect_channel_pixels(self, info, data):
        return connect.connect_channel_pixels(self.cl_state,info,data,self.verbose)
        
    def thin_channels(self, info, data):
        self.print('Thinning channels...',end='')  
        nxp = info.nx_padded
        nyp = info.ny_padded
        mapping_array = data.mapping_array
        channel_array = np.zeros((nxp,nyp), dtype=np.bool)
        channel_array[  ((mapping_array & info.is_channel)==info.is_channel)
                      | ((mapping_array & info.is_interchannel)==info.is_interchannel)
                     ] = True
        self.print('skeletonizing...',end='')  
        skeleton_array = skeletonize(medial_axis(channel_array))
        mapping_array[skeleton_array] |= info.is_thinchannel
        self.print('done')  

    def map_channel_heads(self, info, data):
        return channelheads.map_channel_heads(self.cl_state, info, data, self.verbose)
        # HACK - this step used to be necessary - can it really be dropped?
#         channelheads.prune_channel_heads(self.cl_state, info, data, 
#                                          self.verbose)
        
    def count_downchannels(self, info, data):
        nxp = info.nx_padded
        nyp = info.ny_padded
        data.count_array = np.zeros((nxp,nyp), dtype=np.uint32)
        data.link_array  = np.zeros((nxp,nyp), dtype=np.uint32)
        return countlink.count_downchannels(self.cl_state, info, data, self.verbose)
        
    def flag_downchannels(self, info, data):
        return countlink.flag_downchannels(self.cl_state, info, data, self.verbose)

    def label_confluences(self, info, data):
        if not label.label_confluences(self.cl_state, info, data, self.verbose) \
           or not countlink.flag_downchannels(self.cl_state, info, data, self.verbose) \
           or not countlink.flag_downchannels(self.cl_state, info, data, self.verbose, 
                                    do_reset_count=False) \
           or not countlink.flag_downchannels(self.cl_state, info, data, self.verbose, 
                                    do_reset_count=False):
            # Three passes to try to eliminate all 'parasite' streamlets
            return False
        else:
            return True
        
    def segment_downchannels(self, info, data):
        nxp = info.nx_padded
        nyp = info.ny_padded
        data.label_array = np.zeros((nxp,nyp), dtype=np.int32)
        n = segment.segment_channels(self.cl_state, info, data, self.verbose)
        if n==0:
            return False
        # Save the channel-only segment labeling for now - also convert to unsigned
        data.channel_label_array = data.label_array.copy().astype(np.uint32)
        is_majorconfluence = info.is_majorconfluence
        return True
        
    def link_hillslopes(self, info, data):
        return hillslopes.link_hillslopes(self.cl_state, info, data, self.verbose)

    def segment_hillslopes(self, info, data):
        return segment.segment_hillslopes(self.cl_state, info, data, self.verbose )

    def subsegment_flanks(self, info, data):
        # Channel edges first, and then flanks
        if not segment.subsegment_flanks(self.cl_state, info, data, self.verbose):
            # Failure
            return False
#         data.label_array = data.label_array.astype(dtype=np.int32)
        data.label_array[data.label_array<0] \
            = -(  data.label_array[data.label_array<0] + info.left_flank_addition )
        # Flag all went well
        return True
        
    # Shift to pass#1
    
    def map_midslopes(self, info, data):
        self.print('Midslopes...',end='')  
        dsla = data.sla_array[:,:,0]
        usla = data.sla_array[:,:,1]
        mask = data.mask_array
        nxp  = info.nx_padded
        nyp  = info.ny_padded
        midslope_array = np.zeros((nxp,nyp), dtype=np.bool)
        midslope_array[ (~mask) & (np.fabs(
            gaussian_filter((np.arctan2(dsla,usla)-np.pi/4), self.midslope_filter_sigma))
                             <=self.midslope_threshold)] = True
        data.mapping_array[skeletonize(midslope_array)] |= info.is_midslope
        self.print('done')  

    def map_ridges(self, info, data):
        self.print('Ridges...',end='')  
        dsla = data.sla_array[:,:,0]
        usla = data.sla_array[:,:,1]
        mask = data.mask_array
        nxp  = info.nx_padded
        nyp  = info.ny_padded
        ridge_array = np.zeros((nxp,nyp), dtype=np.bool)
        fat_ridge_array = np.zeros((nxp,nyp), dtype=np.bool)
        ridge_array[ (~mask) & ((
            gaussian_filter((np.arctan2(dsla,usla)-np.pi/4), self.ridge_filter_sigma))
                             <= -(np.pi/4)*self.ridge_threshold)] = True
        dilation_structure = generate_binary_structure(2, 2)
        binary_dilation(ridge_array, structure=dilation_structure, 
                        iterations=1, output=fat_ridge_array)
        binary_fill_holes(fat_ridge_array, output=fat_ridge_array)
#         dilation_structure = generate_binary_structure(2, 2)
#         binary_dilation(skeletonize(fat_ridge_array), 
#                         structure=dilation_structure, 
#                         iterations=5, output=ridge_array)
#         data.mapping_array[ridge_array] |= info.is_ridge
#         data.mapping_array[fat_ridge_array] |= info.is_ridge
        data.mapping_array[skeletonize(fat_ridge_array)] |= info.is_ridge
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
        data.hillslope_labels = unique_labels[unique_labels!=0].astype(np.int32)
        self.n_subsegments    = unique_labels.shape[0]
        self.print('selected {}'.format(self.n_subsegments))
        self.print('...done')  
                
    def measure_hsl(self, info, data):
        self.print('Measuring hillslope lengths...')
        data.traj_length_array = np.zeros_like(data.traj_label_array,dtype=np.float32)
        if not lengths.hsl(self.cl_state, info, data, self.do_measure_hsl_from_ridges,
                           self.verbose):
            return False
        df = pd.DataFrame(np.zeros((data.traj_length_array.shape[0],), 
                                    dtype=[('label', np.int32), ('length', np.float32)]))
        df['label']  = data.traj_label_array
        df['length'] = data.traj_length_array
        df = df[df.label!=0]
        self.hsl_df = df
        
        try:
            stats_df = pd.DataFrame(data.hillslope_labels,columns=['label'])
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
        # Make a working copy of HSL and find limits
        hsl      = self.hsl_array
        hsl_min  = np.min(hsl)
        hsl_max  = np.max(hsl)
        # Convert to unsigned 16bit so HSL can be filtered using skimage tools
        hsl_clipped = (65535*(hsl-hsl_min)/(hsl_max-hsl_min)).astype(np.uint16)
        # Get the current mask
        mask_array = self.state.merge_active_masks()
        # BUG ? perhaps we shouldn't slice off padding in hsl_smoothed?
        pad    = self.geodata.pad_width
        pslice = np.index_exp[pad:-pad,pad:-pad]
        # Sizes of dilation and mean filters in pixels
        dfw = int(self.hsl_dilation_width/self.geodata.roi_pixel_size)
        mdr = int(self.hsl_mean_radius/self.geodata.roi_pixel_size)
        # Make a mean disk filter 
        mean_disk = disk(mdr)
        # Spread HSL values into masked pixels by dilation
        #   - when contouring, this spreading has the effect of avoiding
        #     nasty mask-boundary edge effects, i.e., masked areas are treated
        #     as zeros and force dense erroneous contours to arise at boundaries
        # Also mean filter to make for reasonably smooth contouring. 
        #    - plus helps to skip over skinny isolated HSL strips with
        #      somewhat bogus extreme values.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if dfw==0:
                hsl_dilated = hsl_clipped
            else:
                self.print('dilation with {0}m ({1}-pixel) width filter...'
                           .format(self.hsl_dilation_width,dfw), end='')
                hsl_dilated = grey_dilation(hsl_clipped,size=(dfw,dfw))
                hsl_dilated[~mask_array] = hsl_clipped[~mask_array]

            self.print('mean filtering with {0}m ({1}-pixel) diameter disk...'
                       .format(self.hsl_mean_radius*2,mdr*2), end='') 
            hsl_mean = mean(hsl_dilated,mean_disk)
        # Rescale filtered HSL back into floats and slice off padding
        self.hsl_smoothed_array \
            = ((hsl_mean[pslice].astype(np.float32))/65535)*(hsl_max-hsl_min)+hsl_min
        self.print('done')  
        
    def map_aspect(self, info):
        self.print('Computing hillslope aspect...',end='')
        
        nxp = info.nx_padded
        nyp = info.ny_padded
        slope_threshold = self.aspect_slope_threshold
        median_radius   = self.aspect_median_filter_radius/self.geodata.pixel_size
        # Copy the slope grid because we're going to monkey with it
        slope_array     = self.preprocess.slope_array.copy()
        uv_array        = self.preprocess.uv_array
        mask_array      = np.zeros((nxp,nyp), dtype=np.bool)
        slope_array[((self.mapping_array & info.is_channel)==info.is_channel)] = 0.0
        if self.do_aspect_median_filtering:
            sf = np.max(slope_array)/255.0
            slope_array = sf*median(np.uint8(slope_array/sf),disk(median_radius))

        median_disk = disk(self.uv_median_radius/self.geodata.pixel_size)
        sf = 2.0/255.0
        uvx_array = sf*median(np.uint8((uv_array[:,:,0]+1.0)/sf),median_disk)-1.0
        uvy_array = sf*median(np.uint8((uv_array[:,:,1]+1.0)/sf),median_disk)-1.0
        mask_array[  (slope_array<slope_threshold)
                   | ((self.mapping_array & info.is_channel)==info.is_channel) ] = True
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
        pslice = np.index_exp[pad:-pad,pad:-pad]
        # Use basic masks plus the merged coarse-subsegmentation mask
        self.state.reset_active_masks()
        self.state.add_active_mask({'merged_coarse': self.merged_coarse_mask_array})
        # Also exclude any np.ma masked pixels in self.aspect_array
        mask_array = self.state.merge_active_masks()[pslice] \
                   | self.aspect_array[pslice].mask
        # Fetch non-masked aspect and HSL values
        aspect_array = self.aspect_array[pslice][~mask_array]
        hsl_array    = self.hsl_smoothed_array[~mask_array]
        # Convert aspect to degrees
        np.rad2deg(aspect_array, out=aspect_array)
        # Combine into HSL(aspect) array
        # Exclude "negligibly" small HSL values aka near zero mismeasurements
        hsl_aspect_array = np.stack( (hsl_array[hsl_array>=hsl_averaging_threshold],
                                      aspect_array[hsl_array>=hsl_averaging_threshold]), 
                                      axis=1)
        # Sort in-place using column 1 (aspect) as key
        self.hsl_aspect_array = hsl_aspect_array[hsl_aspect_array[:,1].argsort()]
        # Convert into a pandas dataframe for easier processing
        self.hsl_aspect_df = pd.DataFrame(data=self.hsl_aspect_array,
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
