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
from streamlines.useful       import Data, Info

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
        # Create mapping flag array 
        self.mapping_array = np.zeros((self.geodata.roi_padded_nx,
                                       self.geodata.roi_padded_ny),
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
        info = Info(self.state, self.geodata, self)
        info.set_xy()
        trajectories = Trajectories(self.state.cl_platform, self.state.cl_device,
                                    cl_src_path         = self.state.cl_src_path,
                                    info                = info,
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
        info = Info(self.state, self.geodata, self)
        info.set_xy()
        fields = Fields(self.state.cl_platform, self.state.cl_device,
                        cl_src_path         = self.state.cl_src_path,
                        info                = info,
                        data                = data,
                        verbose             = self.state.verbose,
                        gpu_verbose         = self.state.gpu_verbose )
        fields.integrate()
        # Only preserve what we need from the trajectories class instance
        self.slc_array = fields.data.slc_array
        self.slt_array = fields.data.slt_array
        self.sla_array = fields.data.sla_array
        
