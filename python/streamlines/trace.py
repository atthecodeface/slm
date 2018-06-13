"""
Trace streamlines and their density grids

Todo:
    Fix likely bug in parameters file path wrangling
"""

import sys
import numpy  as np
import pandas as pd
import timeit
from os import environ
environ['PYTHONUNBUFFERED']='True'

from streamlines.core         import Core
from streamlines.trajectories import Trajectories
from streamlines.fields       import Fields

__all__ = ['Info','Trace']

pdebug = print

class Data():    
    def __init__(self,
                 mask_array      = None,
                 uv_array        = None,
                 mapping_array   = None,
                 traj_stats_df   = None):
        self.mask_array          = mask_array
        self.uv_array            = uv_array
        self.mapping_array       = mapping_array
        self.traj_stats_df       = traj_stats_df
        
class Info():    
    def __init__(self, trace, mapping=None, n_seed_points=None):
        state = trace.state
        geodata = trace.geodata
        if trace.max_length==np.float32(0.0):
            max_length = np.finfo(numpy.float32).max
        else:
            max_length = trace.max_length
        max_n_steps = np.uint32(trace.max_length/trace.integrator_step_factor)
        if trace.interchannel_max_n_steps==0:
            interchannel_max_n_steps = max_n_steps
        else:
            interchannel_max_n_steps = trace.interchannel_max_n_steps
         
        grid_scale = np.sqrt(np.float32(geodata.roi_nx*geodata.roi_ny))
        nxf = np.float32(geodata.roi_nx)
        nyf = np.float32(geodata.roi_ny)
        dt_max = min(min(1.0/nxf,1.0/nyf),0.1)
        subpixel_seed_span = 1.0-1.0/np.float32(trace.subpixel_seed_point_density)
        subpixel_seed_step \
            = subpixel_seed_span/(np.float32(trace.subpixel_seed_point_density)-1.0 
                                  if trace.subpixel_seed_point_density>1 else 1.0)
        self.debug =                   np.bool8(state.debug)
        self.verbose =                 np.bool8(state.gpu_verbose)
        self.n_trajectory_seed_points= np.uint32(trace.n_trajectory_seed_points)
        self.n_seed_points =           np.uint32(0)
        self.n_padded_seed_points =    np.uint32(0)
        self.do_shuffle =              np.bool8(trace.do_shuffle_seed_points)
        self.shuffle_rng_seed =        np.uint32(trace.shuffle_rng_seed)
        self.downup_sign =             np.float32(np.nan)
        self.gpu_memory_limit_pc =        np.uint32(state.gpu_memory_limit_pc)
        self.n_work_items =               np.uint32(state.n_work_items)
        self.chunk_size_factor =          np.uint32(state.chunk_size_factor)
        self.max_time_per_kernel =        np.float32(state.max_time_per_kernel)
        self.integrator_step_factor =     np.float32(trace.integrator_step_factor)
        self.max_integration_step_error = np.float32(trace.max_integration_step_error)
        self.adjusted_max_error =         np.float32(0.85*np.sqrt(
                                                  trace.max_integration_step_error))
        self.max_length =   np.float32(max_length/geodata.roi_pixel_size)
        self.pixel_size =   np.float32(geodata.roi_pixel_size)
        self.integration_halt_threshold = np.float32(trace.integration_halt_threshold)
        self.pad_width =    np.uint32(geodata.pad_width)
        self.pad_width_pp5= np.float32(geodata.pad_width)+0.5
        self.nx =           np.uint32(geodata.roi_nx)
        self.ny =           np.uint32(geodata.roi_ny)
        self.nxf =          np.float32(nxf)
        self.nyf =          np.float32(nyf)
        self.nx_padded =    np.uint32(geodata.roi_nx+2*geodata.pad_width)
        self.ny_padded =    np.uint32(geodata.roi_ny+2*geodata.pad_width)
        self.nxy_padded =   np.uint32( (geodata.roi_nx+2*geodata.pad_width)
                                      *(geodata.roi_ny+2*geodata.pad_width) )
        self.x_max =        np.float32(nxf-0.5)
        self.y_max =        np.float32(nyf-0.5)
        self.grid_scale =   np.float32(grid_scale)
        self.combo_factor = np.float32(grid_scale*trace.integrator_step_factor)
        self.dt_max =       np.float32(dt_max)
        self.max_n_steps =  np.uint32(max_n_steps)
        self.trajectory_resolution =    np.uint32(trace.trajectory_resolution)
        self.seeds_chunk_offset =       np.uint32(0)
        self.subpixel_seed_point_density = np.uint32(trace.subpixel_seed_point_density)
        self.subpixel_seed_halfspan =   np.float32(subpixel_seed_span/2.0)
        self.subpixel_seed_step =       np.float32(subpixel_seed_step)
        self.jitter_magnitude =         np.float32(trace.jitter_magnitude)
        self.interchannel_max_n_steps = np.uint32(interchannel_max_n_steps)
        if mapping is not None:
            self.do_measure_hsl_from_ridges = mapping.do_measure_hsl_from_ridges
#             self.segmentation_threshold     = np.uint32(mapping.segmentation_threshold)
#             try:
#                 self.channel_threshold      = np.uint32(mapping.channel_threshold)
#             except:
#                 pass
        else:
            self.do_measure_hsl_from_ridges = False
            self.segmentation_threshold     = np.uint32(0)
            self.channel_threshold = 0
            
        self.left_flank_addition = 2147483648
        flags = [
            'is_channel',         # 1
            'is_thinchannel',     # 2
            'is_interchannel',    # 4
            'is_channelhead',     # 8
            'is_channeltail',     # 16
            'is_majorconfluence', # 32
            'is_minorconfluence', # 64
            'is_majorinflow',     # 128
            'is_minorinflow',     # 256
            'is_leftflank',       # 512
            'is_rightflank',      # 1024
            'is_midslope',        # 2048
            'is_ridge',           # 4028
            'was_channelhead',    # 8196
            'is_subsegmenthead',  # 16384
            'is_loop',            # 32768
            'is_blockage'         # 65536
            ]
        [setattr(self,flag,np.uint(2**idx)) for idx,flag in enumerate(flags)]

class Trace(Core):
    """
    Class providing set of methods to compute streamline trajectories and densities 
    from raw DTM data.
    
    Provides top-level methods to: 
    (1) set seed points aka start locations (sub-pixel positions) of streamlines; 
    (2) trace streamlines from seed points both upstream amd downstream; 
    (3) compute gridded measures of mean streamline length and mean effective area.
    
    Args:
        Core (class):
    
        """
    def __init__(self,state,imported_parameters,geodata,preprocess):
        """
        Initialize a class instance.
        
        Args:
            state (object):
            imported_parameters (dict):
            geodata (object):
            preprocess (object):

        Attributes:
            self.geodata (obj):
            self.preprocess (object):
            self.seed_point_array (numpy.ndarray):
            self.perform_RungeKutta2_integration (function):
            
        """
        super(Trace,self).__init__(state,imported_parameters)  
        self.geodata = geodata
        self.preprocess = preprocess
        self.mapping_array = None
        self.seed_point_array = None
        
    def do(self):
        """
        Trace all streamlines both upstream and downstream
        and derive mean streamline point spacing.
            
        Attributes:
            seed_point_array (numpy.ndarray):
            streamline_arrays_list (list):
            traj_nsteps_array (numpy.ndarray):
            traj_length_array (numpy.ndarray):
            traj_stats_df (pandas.DataFrame):
            slc_array (numpy.ndarray):
            slt_array (numpy.ndarray):
            sla_array (numpy.ndarray):
            """
        self.print('\n**Trace begin**')  
        # Create mapping flag array 
        self.mapping_array = np.zeros((self.geodata.roi_nx+2*self.geodata.pad_width,
                                       self.geodata.roi_ny+2*self.geodata.pad_width),
                                       dtype=np.uint32)
        # Integrate streamlines downstream and upstream
        self.compute_trajectories()
        # Map mean streamline integrations downstream and upstream
        self.compute_fields()
        # Done
        self.print('**Trace end**\n')  

    def compute_trajectories(self):
        """
        Trace up or downstreamlines across region of interest (ROI) of DTM grid.
    
        """
        data = Data( mask_array    = self.state.merge_active_masks(),
                     uv_array      = self.preprocess.uv_array,
                     mapping_array = self.mapping_array # Currently unused
                     )
        trajectories = Trajectories(self.state.cl_platform, self.state.cl_device,
                                    cl_src_path         = self.state.cl_src_path,
                                    info                = Info(self),
                                    data                = data,
                                    do_trace_downstream = self.do_trace_downstream,
                                    do_trace_upstream   = self.do_trace_upstream,
                                    verbose             = self.state.verbose,
                                    gpu_verbose         = self.state.gpu_verbose )
        trajectories.integrate()
        # Only preserve what we need from the trajectories class instance
        self.seed_point_array       = trajectories.data.seed_point_array
        self.streamline_arrays_list = trajectories.data.streamline_arrays_list
        self.traj_stats_df          = trajectories.data.traj_stats_df

    def compute_fields(self):
        """
        Trace up or downstreamlines across region of interest (ROI) of DTM grid.
    
        """
        data = Data( mask_array    = self.state.merge_active_masks(),
                     uv_array      = self.preprocess.uv_array,
                     mapping_array = self.mapping_array, # Currently unused
                     traj_stats_df = self.traj_stats_df )
        fields = Fields(self.state.cl_platform, self.state.cl_device,
                        cl_src_path         = self.state.cl_src_path,
                        info                = Info(self),
                        data                = data,
                        verbose             = self.state.verbose,
                        gpu_verbose         = self.state.gpu_verbose )
        fields.integrate()
        # Only preserve what we need from the trajectories class instance
        self.slc_array = fields.data.slc_array
        self.slt_array = fields.data.slt_array
        self.sla_array = fields.data.sla_array
        
