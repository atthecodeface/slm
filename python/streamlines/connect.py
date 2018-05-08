"""
1) Connect missing links between channel pixels.
2) Locate channel heads.

"""

import pyopencl as cl
import pyopencl.array
import numpy as np
import os
os.environ['PYTHONUNBUFFERED']='True'
import warnings

from streamlines import pocl
from streamlines.useful import vprint, pick_seeds

__all__ = ['connect_channel_pixels','map_channel_heads']

pdebug = print

def connect_channel_pixels(cl_state, info, mask_array, uv_array, mapping_array, verbose):
    """
    Connect missing links between channel pixels.
    
    Args:
        cl_state (obj):
        info (obj):
        mask_array (numpy.ndarray):
        uv_array (numpy.ndarray):
        mapping_array (numpy.ndarray):
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
    seed_point_array = pick_seeds(map=mapping_array, flag=is_channel, pad=pad)
    if ( seed_point_array.shape[0]==0 ):
        vprint(verbose,'no channel pixels found...exiting')
        return
    
    # Specify arrays & CL buffers 
    array_dict = { 'seed_point':  {'array': seed_point_array, 'rwf': 'RO'},
                   'mask':        {'array': mask_array,       'rwf': 'RO'}, 
                   'uv':          {'array': uv_array,         'rwf': 'RO'}, 
                   'mapping':     {'array': mapping_array,    'rwf': 'RW'} }
    info.n_seed_points = seed_point_array.shape[0]
    
    # Do integrations on the GPU
    cl_state.kernel_fn = 'connect_channels'
    pocl.gpu_compute(cl_state, info, array_dict, verbose)
    
    # Done
    vprint(verbose,'...done')  

def map_channel_heads(cl_state, info, mask_array, uv_array, mapping_array, verbose ):
    """
    Find channel head pixels.
    
    Args:
        cl_state (obj):
        info (obj):
        mask_array (numpy.ndarray):
        uv_array (numpy.ndarray):
        mapping_array (numpy.ndarray):
        verbose (bool):
    """
    vprint(verbose,'Mapping channel heads...')
    
    # Prepare CL essentials
    cl_state.kernel_source \
        = pocl.read_kernel_source(cl_state.src_path,['essentials.cl','updatetraj.cl',
                                                     'computestep.cl','rungekutta.cl',
                                                     'channelheads.cl'])

    # Pre-designate every channel pixel as a channel head
    #   - and expect to eliminate all non-heads during the GPU compute
    is_channel     = info.is_channel
    is_thinchannel = info.is_thinchannel
    is_channelhead = info.is_channelhead
    mapping_array[(mapping_array&is_thinchannel)==is_thinchannel] |= is_channelhead
    pad = info.pad_width
        
    # Trace downstream from all non-masked pixels
    seed_point_array = pick_seeds(mask=mask_array, flag=is_channel, pad=pad)
    # Specify arrays & CL buffers 
    array_dict = { 'seed_point':  {'array': seed_point_array, 'rwf': 'RO'},
                   'mask':        {'array': mask_array,       'rwf': 'RO'}, 
                   'uv':          {'array': uv_array,         'rwf': 'RO'}, 
                   'mapping':     {'array': mapping_array,    'rwf': 'RW'} }
    info.n_seed_points = seed_point_array.shape[0]
    
    # Do integrations on the GPU
    cl_state.kernel_fn = 'map_channel_heads'
    pocl.gpu_compute(cl_state, info, array_dict, verbose)
    
    vprint(verbose,'pruning...')
    
    # Trace downstream from all provisional channel head pixels
    seed_point_array \
        = pick_seeds(mask=mask_array, map=mapping_array, flag=is_channelhead, pad=pad)
    # Specify arrays & CL buffers 
    array_dict['seed_point']['array'] = seed_point_array
    info.n_seed_points = seed_point_array.shape[0]
    
    # Do integrations on the GPU
    cl_state.kernel_fn = 'prune_channel_heads'
    pocl.gpu_compute(cl_state, info, array_dict, verbose)
    
    # Done
    vprint(verbose,'...done')  
