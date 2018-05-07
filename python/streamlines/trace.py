"""
Trace streamlines and their density grids

Todo:
    Fix likely bug in parameters file path wrangling
"""

import sys
import numpy as np
import pandas as pd
import timeit
from os import environ
environ['PYTHONUNBUFFERED']='True'
from streamlines.core import Core
from streamlines.trajectories import integrate_trajectories
from streamlines.fields import integrate_fields

__all__ = ['Trace']

pdebug = print

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
            'is_loop',            # 16384
            'is_blockage'         # 32768
            ]
        [setattr(self,flag,2**idx) for idx,flag in enumerate(flags)]
        
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
        self.print('\n**Trace begin**', flush=True)  
        # Create mapping flag array 
        self.mapping_array = np.zeros((self.geodata.roi_nx+2*self.geodata.pad_width,
                                       self.geodata.roi_ny+2*self.geodata.pad_width),
                                       dtype=np.uint32)
        # Integrate streamlines downstream and upstream
        self.trajectories()
        # Map mean streamline integrations downstream and upstream
        self.fields()
        # Done
        self.print('**Trace end**\n', flush=True)  
        
    def build_info_dict(self, n_seed_points=None):
        """
        TBD.
    
        Returns:
            numpy.ndarray: info_dict
        """

#         if n_seed_points is None:
#             n_seed_points = self.n_seed_points
#             n_padded_seed_points = self.n_padded_seed_points
#         else:
#             n_seed_points = 0
#             n_padded_seed_points = 0
        if self.max_length==np.float32(0.0):
            max_length = np.finfo(numpy.float32).max
        else:
            max_length = self.max_length
        max_n_steps = np.uint32(self.max_length/self.integrator_step_factor)
        if self.interchannel_max_n_steps==0:
            interchannel_max_n_steps = max_n_steps
        else:
            interchannel_max_n_steps = self.interchannel_max_n_steps
         
        grid_scale = np.sqrt(np.float32(self.geodata.roi_nx*self.geodata.roi_ny))
        nxf = np.float32(self.geodata.roi_nx)
        nyf = np.float32(self.geodata.roi_ny)
        dt_max = min(min(1.0/nxf,1.0/nyf),0.1)
        subpixel_seed_span = 1.0-1.0/np.float32(self.subpixel_seed_point_density)
        subpixel_seed_step \
            = subpixel_seed_span/(np.float32(self.subpixel_seed_point_density)-1.0 
                                  if self.subpixel_seed_point_density>1 else 1.0)
        info_dict = {
            'debug' :                   np.bool8(self.state.debug),
            'n_trajectory_seed_points': np.uint32(self.n_trajectory_seed_points),
            'n_seed_points' :           np.uint32(0),
            'n_padded_seed_points' :    np.uint32(0),
            'do_shuffle' :              np.bool8(self.do_shuffle_seed_points),
            'shuffle_rng_seed' :        np.uint32(self.shuffle_rng_seed),
            'downup_sign' :             np.float32(np.nan),
            'gpu_memory_limit_pc' :        np.uint32(self.state.gpu_memory_limit_pc),
            'n_work_items' :               np.uint32(self.state.n_work_items),
            'chunk_size_factor' :          np.uint32(self.state.chunk_size_factor),
            'max_time_per_kernel' :        np.float32(self.state.max_time_per_kernel),
            'integrator_step_factor' :     np.float32(self.integrator_step_factor),
            'max_integration_step_error' : np.float32(self.max_integration_step_error),
            'adjusted_max_error' :         np.float32(0.85*np.sqrt(
                                                      self.max_integration_step_error)),
            'max_length' :   np.float32(max_length/self.geodata.roi_pixel_size),
            'pixel_size' :   np.float32(self.geodata.roi_pixel_size),
            'integration_halt_threshold' : np.float32(self.integration_halt_threshold),
            'pad_width' :    np.uint32(self.geodata.pad_width),
            'pad_width_pp5': np.float32(self.geodata.pad_width)+0.5,
            'nx' :           np.uint32(self.geodata.roi_nx),
            'ny' :           np.uint32(self.geodata.roi_ny),
            'nxf' :          np.float32(nxf),
            'nyf' :          np.float32(nyf),
            'nx_padded' :    np.uint32(self.geodata.roi_nx+2*self.geodata.pad_width),
            'ny_padded' :    np.uint32(self.geodata.roi_ny+2*self.geodata.pad_width),
            'nxy_padded' :   np.uint32( (self.geodata.roi_nx+2*self.geodata.pad_width)
                                       *(self.geodata.roi_ny+2*self.geodata.pad_width) ),
            'x_max' :        np.float32(nxf-0.5),
            'y_max' :        np.float32(nyf-0.5),
            'grid_scale' :   np.float32(grid_scale),
            'combo_factor' : np.float32(grid_scale*self.integrator_step_factor),
            'dt_max' :       np.float32(dt_max),
            'max_n_steps' :  np.uint32(max_n_steps),
            'trajectory_resolution' :    np.uint32(self.trajectory_resolution),
            'seeds_chunk_offset' :       np.uint32(0),
            'subpixel_seed_point_density' : np.uint32(self.subpixel_seed_point_density),
            'subpixel_seed_halfspan' :   np.float32(subpixel_seed_span/2.0),
            'subpixel_seed_step' :       np.float32(subpixel_seed_step),
            'jitter_magnitude' :         np.float32(self.jitter_magnitude),
            'interchannel_max_n_steps' : np.uint32(interchannel_max_n_steps),
            'segmentation_threshold' :   np.uint32(self.segmentation_threshold),
            'left_flank_addition': np.uint32(self.left_flank_addition),
            'is_channel' :         np.uint32(self.is_channel),
            'is_thinchannel' :     np.uint32(self.is_thinchannel),
            'is_interchannel' :    np.uint32(self.is_interchannel),
            'is_channelhead' :     np.uint32(self.is_channelhead),
            'is_channeltail' :     np.uint32(self.is_channeltail),
            'is_majorconfluence' : np.uint32(self.is_majorconfluence),
            'is_minorconfluence' : np.uint32(self.is_minorconfluence),
            'is_majorinflow' :     np.uint32(self.is_majorinflow),
            'is_minorinflow' :     np.uint32(self.is_minorinflow),
            'is_leftflank' :       np.uint32(self.is_leftflank),
            'is_rightflank' :      np.uint32(self.is_rightflank),
            'is_midslope' :        np.uint32(self.is_midslope),
            'is_ridge' :           np.uint32(self.is_ridge),
            'was_channelhead' :    np.uint32(self.was_channelhead),
            'is_loop' :            np.uint32(self.is_loop),
            'is_blockage' :        np.uint32(self.is_blockage)
        }
        return info_dict

    def trajectories(self):
        """
        Trace up or downstreamlines across region of interest (ROI) of DTM grid.
    
        Returns:
            list, numpy.ndarray, numpy.ndarray, pandas.DataFrame,
            numpy.ndarray, numpy.ndarray, numpy.ndarray: 
            streamline_arrays_list, traj_nsteps_array, traj_length_array, traj_stats_df
        """
        (self.seed_point_array, self.streamline_arrays_list,
         self.traj_nsteps_array, self.traj_length_array, self.traj_stats_df) \
            = integrate_trajectories(
                self.state.cl_src_path, self.state.cl_platform, self.state.cl_device, 
                self.build_info_dict(),
                self.geodata.basin_mask_array,
                self.preprocess.uv_array,
                self.mapping_array,
                self.do_trace_downstream, self.do_trace_upstream, 
                self.state.verbose
            )
        return

    def fields(self):
        """
        Trace up or downstreamlines across region of interest (ROI) of DTM grid.
    
        Returns:
            list, numpy.ndarray, numpy.ndarray, pandas.DataFrame,
            numpy.ndarray, numpy.ndarray, numpy.ndarray: 
            slc_array, slt_array, sla_array
        """
        (self.slc_array, self.slt_array, self.sla_array) \
            = integrate_fields(
                self.state.cl_src_path, self.state.cl_platform, self.state.cl_device, 
                self.build_info_dict(),
                self.geodata.basin_mask_array,
                self.preprocess.uv_array,
                self.mapping_array,
                self.do_trace_downstream, self.do_trace_upstream, 
                self.traj_stats_df,
                self.state.verbose
            )
        return
