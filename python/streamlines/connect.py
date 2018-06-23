"""
Connect missing links between channel pixels.

"""

import pyopencl as cl
import pyopencl.array
import numpy as np
import os
os.environ['PYTHONUNBUFFERED']='True'
import warnings

from streamlines import pocl
from streamlines.useful import vprint, pick_seeds, check_sizes

__all__ = ['connect_channel_pixels']

pdebug = print

def connect_channel_pixels(cl_state, info, data, verbose):
    """
    Connect missing links between channel pixels.
    
    Args:
        cl_state (obj):
        info (obj):
        data (obj):
        verbose (bool):
    """
    vprint(verbose,'Connecting channel pixels...')
    # Prepare CL essentials
    cl_state.kernel_source \
        = pocl.read_kernel_source(cl_state.src_path,['essentials.cl','updatetraj.cl',
                                                     'computestep.cl','rungekutta.cl',
                                                     'connect.cl'])
            
    # Generate a list (array) of seed points from the set of channel pixels
    pad        = info.pad_width
    is_channel = info.is_channel
    # Trace downstream from all channel pixels
    seed_point_array = pick_seeds(map=data.mapping_array, flag=is_channel, pad=pad)
    if ( seed_point_array.shape[0]==0 ):
        vprint(verbose,'no channel pixels found...exiting')
        return
    
    # Specify arrays & CL buffers 
    array_dict = { 'seed_point':  {'array': seed_point_array,      'rwf': 'RO'},
                   'mask':        {'array': data.mask_array,       'rwf': 'RO'}, 
                   'uv':          {'array': data.uv_array,         'rwf': 'RO'}, 
                   'mapping':     {'array': data.mapping_array,    'rwf': 'RW'} }
    info.n_seed_points = seed_point_array.shape[0]
    if ( info.n_seed_points==0 ):
        # Flag an error - empty seeds list
        return False
    check_sizes(info.nx_padded,info.ny_padded, array_dict)
    
    # Do integrations on the GPU
    cl_state.kernel_fn = 'connect_channels'
    pocl.gpu_compute(cl_state, info, array_dict, info.verbose)
    
    # Done
    vprint(verbose,'...done')  
    # Flag all went well
    return True
