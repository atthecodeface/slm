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
from streamlines.integration import integrate_trajectories

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
            'is_stuck',           # 8196
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
        # Assign a possibly irregular (near bdries) grid of initial streamline points
        self.create_seeds()
        # Do the streamline integrations up and then downstream
        self.trace_streamlines()
        # Done
        self.print('**Trace end**\n', flush=True)  
        
    def create_seeds(self):
        """
        Parent method to generate streamline seed points.   
            
        Attributes:
            self.seed_point_array (numpy.ndarray):
        
        """    
        self.print('Generating seed points')
        mask = self.geodata.basin_mask_array
        pad = self.geodata.pad_width
        if self.state.array_order=='F':
            self.seed_point_array \
                = ((np.argwhere(~mask).astype(np.float32) - pad)
                   ).T.copy(order='F')
        else:
            self.seed_point_array \
                = ((np.argwhere(~mask).astype(np.float32) - pad)
                   ).copy(order='C')
        self.print('...done',flush=True)

    def build_info_struct(self):
        """
        TBD.
    
        Returns:
            numpy.ndarray: info_struct
        """

        if self.max_length==np.float32(0.0):
            max_length = np.finfo(numpy.float32).max
        else:
            max_length = self.max_length
        max_n_steps = np.uint32(self.max_length/self.integrator_step_factor)
        if self.interchannel_max_n_steps==0:
            interchannel_max_n_steps = max_n_steps
        else:
            interchannel_max_n_steps = self.interchannel_max_n_steps

        info_dtype = np.dtype([
                ('array_order', 'U1'),
                ('downup_sign', np.float32),
                ('gpu_memory_limit_pc', np.uint32),
                ('integrator_step_factor', np.float32),
                ('max_integration_step_error', np.float32),
                ('adjusted_max_error', np.float32),
                ('max_length', np.float32),
                ('pixel_size', np.float32),
                ('integration_halt_threshold', np.float32),
                ('pad_width', np.uint32),
                ('pad_width_pp5', np.float32),
                ('nx', np.uint32),
                ('ny', np.uint32),
                ('nxf', np.float32),
                ('nyf', np.float32),
                ('nx_padded', np.uint32),
                ('ny_padded', np.uint32),
                ('x_max', np.float32),
                ('y_max', np.float32),
                ('grid_scale', np.float32),
                ('combo_factor', np.float32),
                ('dt_max', np.float32),
                ('max_n_steps', np.uint32),
                ('trajectory_resolution', np.uint32),
                ('seeds_chunk_offset', np.uint32),
                ('subpixel_seed_point_density', np.uint32),
                ('subpixel_seed_halfspan', np.float32),
                ('subpixel_seed_step', np.float32),
                ('jitter_magnitude', np.float32),
                ('interchannel_max_n_steps', np.uint32),
                ('segmentation_threshold', np.uint32),
                ('left_flank_addition', np.uint32),
                ('is_channel', np.uint32),
                ('is_thinchannel', np.uint32),
                ('is_interchannel', np.uint32),
                ('is_channelhead', np.uint32),
                ('is_channeltail', np.uint32),
                ('is_majorconfluence', np.uint32),
                ('is_minorconfluence', np.uint32),
                ('is_majorinflow', np.uint32),
                ('is_minorinflow', np.uint32),
                ('is_leftflank', np.uint32),
                ('is_rightflank', np.uint32),
                ('is_midslope', np.uint32),
                ('is_ridge', np.uint32),
                ('is_stuck', np.uint32),
                ('is_loop', np.uint32),
                ('is_blockage', np.uint32)
            ])          
        grid_scale = np.sqrt(np.float32(self.geodata.roi_nx*self.geodata.roi_ny))
        nxf = np.float32(self.geodata.roi_nx)
        nyf = np.float32(self.geodata.roi_ny)
        dt_max = min(min(1.0/nxf,1.0/nyf),0.1)
        subpixel_seed_span = 1.0-1.0/np.float32(self.subpixel_seed_point_density)
        subpixel_seed_step = subpixel_seed_span/(np.float32(self.subpixel_seed_point_density)-1.0 
                               if self.subpixel_seed_point_density>1 else 1.0)
        return np.array([(
            np.string_(self.state.array_order),
            np.float32(np.nan),
            np.int32(self.state.gpu_memory_limit_pc),
            np.float32(self.integrator_step_factor),
            np.float32(self.max_integration_step_error),
            np.float32(0.85*np.sqrt((self.max_integration_step_error))),
            np.float32(max_length/self.geodata.roi_pixel_size),
            np.float32(self.geodata.roi_pixel_size),
            np.float32(self.integration_halt_threshold),
            np.uint32(self.geodata.pad_width),
            np.float32(self.geodata.pad_width)+0.5,
            np.uint32(self.geodata.roi_nx),
            np.uint32(self.geodata.roi_ny),
            np.float32(nxf),
            np.float32(nyf),
            np.uint32(self.geodata.roi_nx+2*self.geodata.pad_width),
            np.uint32(self.geodata.roi_ny+2*self.geodata.pad_width),
            np.float32(nxf-0.5),
            np.float32(nyf-0.5),
            np.float32(grid_scale),
            np.float32(grid_scale*self.integrator_step_factor),
            np.float32(dt_max),
            np.uint32(max_n_steps),
            np.uint32(self.trajectory_resolution),
            np.uint32(0),
            np.uint32(self.subpixel_seed_point_density),
            np.float32(subpixel_seed_span/2.0),
            np.float32(subpixel_seed_step),
            np.float32(self.jitter_magnitude),
            np.uint32(interchannel_max_n_steps),
            np.uint32(self.segmentation_threshold),
            np.uint32(self.left_flank_addition),
            np.uint32(self.is_channel),
            np.uint32(self.is_thinchannel),
            np.uint32(self.is_interchannel),
            np.uint32(self.is_channelhead),
            np.uint32(self.is_channeltail),
            np.uint32(self.is_majorconfluence),
            np.uint32(self.is_minorconfluence),
            np.uint32(self.is_majorinflow),
            np.uint32(self.is_minorinflow),
            np.uint32(self.is_leftflank),
            np.uint32(self.is_rightflank),
            np.uint32(self.is_midslope),
            np.uint32(self.is_ridge),
            np.uint32(self.is_stuck),
            np.uint32(self.is_loop),
            np.uint32(self.is_blockage)
        )], dtype = info_dtype)

    def trace_streamlines(self):
        """
        Trace up or downstreamlines across region of interest (ROI) of DTM grid.
    
        Returns:
            list, numpy.ndarray, numpy.ndarray, pandas.DataFrame,
            numpy.ndarray, numpy.ndarray, numpy.ndarray: 
            streamline_arrays_list, traj_nsteps_array, traj_length_array, traj_stats_df,
            slc_array, slt_array, sla_array
        """
        (self.streamline_arrays_list,
         self.traj_nsteps_array, self.traj_length_array, self.traj_stats_df,
         self.slc_array, self.slt_array, self.sla_array) \
            = integrate_trajectories(
                self.state.path, self.state.cl_platform, self.state.cl_device, 
                self.build_info_struct(),
                self.seed_point_array, 
                self.geodata.basin_mask_array,
                self.preprocess.u_array,self.preprocess.v_array,
                self.do_trace_downstream, self.do_trace_upstream, 
                self.state.verbose
            )
        return