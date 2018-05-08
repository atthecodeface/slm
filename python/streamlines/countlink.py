"""
1) Link each channel pixel to its inflow-dominant upstream pixel;
2) Count pixels downstream from channels heads, ensuring longest dominates;
3) Link each hillslope pixel to its inflow-dominant upstream pixel.
"""

import pyopencl as cl
import pyopencl.array
import numpy as np
import os
os.environ['PYTHONUNBUFFERED']='True'
import warnings

from streamlines import pocl
from streamlines.useful import vprint, pick_seeds

__all__ = ['count_downchannels','flag_downchannels','link_hillslopes']

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
    pad            = info.pad_width
    is_channelhead = info.is_channelhead
    is_thinchannel = info.is_thinchannel
    data.mapping_array[(data.mapping_array&is_thinchannel)==is_thinchannel] \
       = data.mapping_array[(data.mapping_array&is_thinchannel)==is_thinchannel] \
                                ^is_thinchannel
    seed_point_array = pick_seeds(mask=data.mask_array, map=data.mapping_array, 
                                  flag=is_channelhead, pad=pad)
        
    # Specify arrays & CL buffers 
    array_dict = {' seed_point': {'array': seed_point_array,      'rwf': 'RO'},
                   'mask':       {'array': data.mask_array,       'rwf': 'RO'}, 
                   'uv':         {'array': data.uv_array,         'rwf': 'RO'}, 
                   'mapping':    {'array': data.mapping_array,    'rwf': 'RW'}, 
                   'count':      {'array': data.count_array,      'rwf': 'RW'}, 
                   'link':       {'array': data.link_array,       'rwf': 'RW'} }
    info.n_seed_points = seed_point_array.shape[0]
    
    # Do integrations on the GPU
    cl_state.kernel_fn = 'count_downchannels'
    pocl.gpu_compute(cl_state, info, array_dict, verbose)
    
    # Done
    vprint(verbose,'...done')  

def flag_downchannels( cl_state, info, data, verbose ):    
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
    pad            = info.pad_width
    is_channelhead = info.is_channelhead
    is_thinchannel = info.is_thinchannel
    data.mapping_array[(data.mapping_array&is_thinchannel)==is_thinchannel] \
        = data.mapping_array[(data.mapping_array&is_thinchannel)==is_thinchannel]\
                                ^is_thinchannel
    data.count_array *= 0
    seed_point_array = pick_seeds(mask=data.mask_array, map=data.mapping_array, 
                                  flag=is_channelhead, pad=pad)

    # Specify arrays & CL buffers 
    array_dict = {' seed_point': {'array': seed_point_array,      'rwf': 'RO'},
                   'mask':       {'array': data.mask_array,       'rwf': 'RO'}, 
                   'uv':         {'array': data.uv_array,         'rwf': 'RO'}, 
                   'mapping':    {'array': data.mapping_array,    'rwf': 'RW'}, 
                   'count':      {'array': data.count_array,      'rwf': 'RW'}, 
                   'link':       {'array': data.link_array,       'rwf': 'RW'} }
    info.n_seed_points = seed_point_array.shape[0]
    
    # Do integrations on the GPU
    cl_state.kernel_fn = 'flag_downchannels'
    pocl.gpu_compute(cl_state, info, array_dict, verbose)
    
    # Done
    vprint(verbose,'...done')  

def link_hillslopes( cl_state, info, data, verbose ):
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
                                                     'countlink.cl'])
            
    # Generate a list (array) of seed points from all non-thin-channel pixels
    pad            = info.pad_width
    is_thinchannel = info.is_thinchannel
    seed_point_array = pick_seeds(mask=data.mask_array, map=~data.mapping_array, 
                                  flag=is_thinchannel, pad=pad)    
        
    # Specify arrays & CL buffers 
    array_dict = {' seed_point': {'array': seed_point_array,      'rwf': 'RO'},
                   'mask':       {'array': data.mask_array,       'rwf': 'RO'}, 
                   'uv':         {'array': data.uv_array,         'rwf': 'RO'}, 
                   'mapping':    {'array': data.mapping_array,    'rwf': 'RW'}, 
                   'count':      {'array': data.count_array,      'rwf': 'RW'}, 
                   'link':       {'array': data.link_array,       'rwf': 'RW'} }
    info.n_seed_points = seed_point_array.shape[0]
    
    # Do integrations on the GPU
    cl_state.kernel_fn = 'link_hillslopes'
    pocl.gpu_compute(cl_state, info, array_dict, verbose)
    
    # Done
    vprint(verbose,'...done')  
    