"""
Link each hillslope pixel to its inflow-dominant upstream pixel.
"""

import pyopencl as cl
import pyopencl.array
import numpy as np
import os
os.environ['PYTHONUNBUFFERED']='True'
import warnings

from streamlines import pocl
from streamlines.useful import vprint, pick_seeds, check_sizes

__all__ = ['link_hillslopes']

pdebug = print

def link_hillslopes( cl_state, info, data, verbose):
    """
    Link hillslope pixels downstream.
    
    Args:
        cl_state (obj):
        info (obj):
        data (obj):
        verbose (bool):
        
    """
    vprint(verbose,'Linking hillslopes...')
    
    # Prepare CL essentials
    cl_state.kernel_source \
        = pocl.read_kernel_source(cl_state.src_path,['essentials.cl','updatetraj.cl',
                                                     'computestep.cl','rungekutta.cl',
                                                     'linkhillslopes.cl'])
            
    # Generate a list (array) of seed points from all non-thin-channel pixels
    pad            = info.pad_width
    is_thinchannel = info.is_thinchannel
    seed_point_array = pick_seeds(mask=data.mask_array, map=~data.mapping_array, 
                                  flag=is_thinchannel, pad=pad)    
        
    # Specify arrays & CL buffers 
    array_dict = { 'seed_point': {'array': seed_point_array,      'rwf': 'RO'},
                   'mask':       {'array': data.mask_array,       'rwf': 'RO'}, 
                   'uv':         {'array': data.uv_array,         'rwf': 'RO'}, 
                   'mapping':    {'array': data.mapping_array,    'rwf': 'RW'}, 
                   'count':      {'array': data.count_array,      'rwf': 'RO'}, 
                   'link':       {'array': data.link_array,       'rwf': 'RW'} }
    info.n_seed_points = seed_point_array.shape[0]
    check_sizes(info.nx_padded,info.ny_padded, array_dict)
    
    # Do integrations on the GPU
    cl_state.kernel_fn = 'link_hillslopes'
    pocl.gpu_compute(cl_state, info, array_dict, info.verbose)
    
    # Done
    vprint(verbose,'...done')  
    