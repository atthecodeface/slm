"""
Trace streamlines and their density grids

Todo:
    Fix likely bug in parameters file path wrangling
"""

import sys
import numpy  as np
import pandas as pd
from os import environ
environ['PYTHONUNBUFFERED']='True'

from streamlines.core         import Core
from streamlines.trajectories import Trajectories
from streamlines.fields       import Fields
from streamlines.useful       import Data, Info, get_bbox

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
        mask_array = self.state.merge_active_masks()
        bbox, bnx, bny = get_bbox(~mask_array)
        pdebug('raw bbox',bbox)
        pad = self.geodata.pad_width
        nxp = self.geodata.roi_nx+pad*2
        nyp = self.geodata.roi_ny+pad*2
        mapping_array = np.zeros((nxp,nyp),dtype=np.uint32)
        info = Info(self.state, self, self.geodata.roi_pixel_size)
        info.set_xy(bnx,bny, pad)
        data = Data( info=info, bbox=bbox, pad=pad,
                     mapping_array = mapping_array,
                     mask_array    = mask_array,
                     uv_array      = self.preprocess.uv_array
                     )
        trajectories = Trajectories( self.state.cl_platform, self.state.cl_device,
                                     cl_src_path         = self.state.cl_src_path,
                                     info                = info,
                                     data                = data,
                                     do_trace_downstream = self.do_trace_downstream,
                                     do_trace_upstream   = self.do_trace_upstream,
                                     verbose             = self.state.verbose,
                                     gpu_verbose         = self.state.gpu_verbose 
                                     )
        trajectories.integrate()
        # Only preserve what we need from the trajectories class instance
        offset_xy = np.array((bbox[0]-pad,bbox[2]-pad))
        pdebug('offset bbox',bbox,offset_xy)
        self.seed_point_array       = trajectories.data.seed_point_array+offset_xy
        self.streamline_arrays_list = trajectories.data.streamline_arrays_list
        self.traj_stats_df          = trajectories.data.traj_stats_df

    def compute_fields(self):
        """
        Trace up or downstreamlines across region of interest (ROI) of DTM grid.
    
        """
        mask_array = self.state.merge_active_masks()
        bbox, bnx, bny = get_bbox(~mask_array)
        pad = self.geodata.pad_width
        nxp = self.geodata.roi_nx+pad*2
        nyp = self.geodata.roi_ny+pad*2
        mapping_array = np.zeros((nxp,nyp),dtype=np.uint32)
        info = Info(self.state, self, self.geodata.roi_pixel_size)
        info.set_xy(bnx,bny, pad)
        data = Data( info=info, bbox=bbox, pad=pad,
                     mask_array    = mask_array,
                     uv_array      = self.preprocess.uv_array,
                     mapping_array = mapping_array,
                     traj_stats_df = self.traj_stats_df 
                     )
        fields = Fields( self.state.cl_platform, self.state.cl_device,
                         cl_src_path = self.state.cl_src_path,
                         info        = info,
                         data        = data,
                         verbose     = self.state.verbose,
                         gpu_verbose = self.state.gpu_verbose 
                         )
        fields.integrate()
        # Only preserve what we need from the trajectories class instance
        self.slc_array = np.zeros((nxp,nyp,2), dtype=np.uint32)
        self.slt_array = np.zeros((nxp,nyp,2), dtype=np.float32)
        self.sla_array = np.zeros((nxp,nyp,2), dtype=np.float32)
        # Insert results back into full (padded) DTM ROI grid arrays
        bounds = data.bounds_slx
        self.slc_array[bounds] = fields.data.slc_array
        self.slt_array[bounds] = fields.data.slt_array
        self.sla_array[bounds] = fields.data.sla_array
        
