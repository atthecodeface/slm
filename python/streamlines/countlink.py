"""
1) Link each channel pixel to its inflow-dominant upstream pixel;
2) Count pixels downstream from channels heads, ensuring longest dominates;
"""

import pyopencl as cl
import pyopencl.array
import numpy as np
from os import environ
environ['PYTHONUNBUFFERED']='True'
import warnings

from streamlines        import pocl
from streamlines.useful import vprint, pick_seeds, check_sizes

__all__ = ['count_downchannels','flag_downchannels']

pdebug = print

def count_downchannels( cl_state, info, data, verbose ):
    """
    Integrate and count downstream designating downstream links & thin channel status.
    
    Args:
        cl_state (obj):
        info (obj):
        data (obj):
        verbose (bool):
    """
    vprint(verbose,'Counting down channels...')
    
    # Prepare CL essentials
    cl_state.kernel_source \
        = pocl.read_kernel_source(cl_state.src_path,['essentials.cl','updatetraj.cl',
                                                     'computestep.cl','rungekutta.cl',
                                                     'countlink.cl'])
            
    # Generate a list (array) of seed points from the set of channel heads
    # Turn off the thin channel flag aka erase mapping so far of thin channels
    data.mapping_array[(data.mapping_array&info.is_thinchannel)!=0] ^= info.is_thinchannel
    seed_point_array = pick_seeds(mask=data.mask_array, map=data.mapping_array, 
                                  flag=info.is_channelhead, pad=info.pad_width)
#     pdebug('count down channels seed_point_array:',seed_point_array)
        
    # Specify arrays & CL buffers 
    array_dict = { 'seed_point': {'array': seed_point_array,      'rwf': 'RO'},
                   'mask':       {'array': data.mask_array,       'rwf': 'RO'}, 
                   'uv':         {'array': data.uv_array,         'rwf': 'RO'}, 
                   'mapping':    {'array': data.mapping_array,    'rwf': 'RW'}, 
                   'count':      {'array': data.count_array,      'rwf': 'RW'}, 
                   'link':       {'array': data.link_array,       'rwf': 'RW'} }
    info.n_seed_points = seed_point_array.shape[0]
    if ( info.n_seed_points==0 ):
        # Flag an error - empty seeds list
        return False
    check_sizes(info.nx_padded,info.ny_padded, array_dict)
    
    # Do integrations on the GPU
    cl_state.kernel_fn = 'count_downchannels'
    pocl.gpu_compute(cl_state, info, array_dict, info.verbose)
    
    # Done
    vprint(verbose,'...done')  
    # Flag all went well
    return True

def flag_downchannels( cl_state, info, data, verbose, do_reset_count=True ):    
    """
    Integrate downstream along channels & count pixel steps as we go.
    
    Args:
        cl_state (obj):
        info (obj):
        data (obj):
        verbose (bool):
        
    """
    vprint(verbose,'Flagging down channels...')
    
    # Prepare CL essentials
    cl_state.kernel_source \
        = pocl.read_kernel_source(cl_state.src_path,['essentials.cl','updatetraj.cl',
                                                     'rungekutta.cl','countlink.cl'])
            
    # Generate a list (array) of seed points from the set of channel heads
    # Reset thin channel flag and downstream count - both are recomputed here
    data.mapping_array[(data.mapping_array&info.is_thinchannel)!=0]^= info.is_thinchannel
    if do_reset_count:
        data.count_array *= 0
    seed_point_array = pick_seeds(mask=data.mask_array, map=data.mapping_array, 
                                  flag=info.is_channelhead, pad=info.pad_width)
#     pdebug('flag down channels seed_point_array:',seed_point_array)

    # Specify arrays & CL buffers 
    array_dict = { 'seed_point': {'array': seed_point_array,      'rwf': 'RO'},
                   'mask':       {'array': data.mask_array,       'rwf': 'RO'}, 
                   'uv':         {'array': data.uv_array,         'rwf': 'RO'}, 
                   'mapping':    {'array': data.mapping_array,    'rwf': 'RW'}, 
                   'count':      {'array': data.count_array,      'rwf': 'RW'}, 
                   'link':       {'array': data.link_array,       'rwf': 'RO'} }
    info.n_seed_points = seed_point_array.shape[0]
    if ( info.n_seed_points==0 ):
        # Flag an error - empty seeds list
        return False
    check_sizes(info.nx_padded,info.ny_padded, array_dict)
    
    # Do integrations on the GPU
    cl_state.kernel_fn = 'flag_downchannels'
    pocl.gpu_compute(cl_state, info, array_dict, info.verbose)
    
    # Done
    vprint(verbose,'...done')  
    # Flag all went well
    return True

 