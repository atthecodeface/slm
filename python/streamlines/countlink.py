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

def count_downchannels( cl_src_path, which_cl_platform, which_cl_device, info_dict, 
                        mask_array, uv_array, 
                        mapping_array, count_array, link_array, verbose ):
        
    """
    Integrate and count downstream designating downstream links & thin channel status.
    
    Args:
        cl_src_path (str):
        which_cl_platform (int):
        which_cl_device (int):
        info_dict (numpy.ndarray):
        mask_array (numpy.ndarray):
        uv_array (numpy.ndarray):
        mapping_array (numpy.ndarray):
        count_array (numpy.ndarray):
        link_array (numpy.ndarray):
        verbose (bool):
        
    """
    vprint(verbose,'Counting down channels...')
    
    # Prepare CL essentials
    platform, device, context= pocl.prepare_cl_context(which_cl_platform,which_cl_device)
    queue = cl.CommandQueue(context,
                            properties=cl.command_queue_properties.PROFILING_ENABLE)
    cl_files = ['essentials.cl','updatetraj.cl','computestep.cl',
                'rungekutta.cl','countlink.cl']
    cl_kernel_source = ''
    for cl_file in cl_files:
        with open(os.path.join(cl_src_path,cl_file), 'r') as fp:
            cl_kernel_source += fp.read()
            
    # Generate a list (array) of seed points from the set of channel heads
    pad            = info_dict['pad_width']
    is_channelhead = info_dict['is_channelhead']
    is_thinchannel = info_dict['is_thinchannel']
    mapping_array[(mapping_array&is_thinchannel)==is_thinchannel] \
        = mapping_array[(mapping_array&is_thinchannel)==is_thinchannel]^is_thinchannel
    seed_point_array \
        = pick_seeds(mask=mask_array, map=mapping_array, flag=is_channelhead, pad=pad)
        
    # Specify arrays & CL buffers 
    array_dict = {' seed_point': {'array': seed_point_array, 'rwf': 'RO'},
                   'mask':       {'array': mask_array,       'rwf': 'RO'}, 
                   'uv':         {'array': uv_array,         'rwf': 'RO'}, 
                   'mapping':    {'array': mapping_array,    'rwf': 'RW'}, 
                   'count':      {'array': count_array,      'rwf': 'RW'}, 
                   'link':       {'array': link_array,       'rwf': 'RW'} }
    info_dict['n_seed_points'] = seed_point_array.shape[0]
    
    # Do integrations on the GPU
    cl_kernel_fn = 'count_downchannels'
    pocl.gpu_compute(device, context, queue, cl_kernel_source,cl_kernel_fn, 
                     info_dict, array_dict, verbose)
    
    # Done
    vprint(verbose,'...done')  

def flag_downchannels( cl_src_path, which_cl_platform, which_cl_device, info_dict, 
                       mask_array, uv_array, 
                       mapping_array, count_array, link_array, verbose ):
        
    """
    Integrate downstream along channels & count pixel steps as we go.
    
    Args:
        cl_src_path (str):
        which_cl_platform (int):
        which_cl_device (int):
        info_dict (numpy.ndarray):
        mask_array (numpy.ndarray):
        uv_array (numpy.ndarray):
        mapping_array (numpy.ndarray):
        count_array (numpy.ndarray):
        link_array (numpy.ndarray):
        verbose (bool):
        
    """
    vprint(verbose,'Flagging down channels...')
    
    # Prepare CL essentials
    platform, device, context= pocl.prepare_cl_context(which_cl_platform,which_cl_device)
    queue = cl.CommandQueue(context,
                            properties=cl.command_queue_properties.PROFILING_ENABLE)
    cl_files = ['essentials.cl','updatetraj.cl',
                'rungekutta.cl','countlink.cl']
    cl_kernel_source = ''
    for cl_file in cl_files:
        with open(os.path.join(cl_src_path,cl_file), 'r') as fp:
            cl_kernel_source += fp.read()
            
    # Generate a list (array) of seed points from the set of channel heads
    pad            = info_dict['pad_width']
    is_channelhead = info_dict['is_channelhead']
    is_thinchannel = info_dict['is_thinchannel']
    mapping_array[(mapping_array&is_thinchannel)==is_thinchannel] \
        = mapping_array[(mapping_array&is_thinchannel)==is_thinchannel]^is_thinchannel
    count_array *= 0
    seed_point_array \
        = pick_seeds(mask=mask_array, map=mapping_array, flag=is_channelhead, pad=pad)

    # Specify arrays & CL buffers 
    array_dict = {' seed_point': {'array': seed_point_array, 'rwf': 'RO'},
                   'mask':       {'array': mask_array,       'rwf': 'RO'}, 
                   'uv':         {'array': uv_array,         'rwf': 'RO'}, 
                   'mapping':    {'array': mapping_array,    'rwf': 'RW'}, 
                   'count':      {'array': count_array,      'rwf': 'RW'}, 
                   'link':       {'array': link_array,       'rwf': 'RW'} }
    info_dict['n_seed_points'] = seed_point_array.shape[0]
    
    # Do integrations on the GPU
    cl_kernel_fn = 'flag_downchannels'
    pocl.gpu_compute(device, context, queue, cl_kernel_source,cl_kernel_fn, 
                     info_dict, array_dict, verbose)
    
    # Done
    vprint(verbose,'...done')  

def link_hillslopes( cl_src_path, which_cl_platform, which_cl_device, info_dict, 
                     mask_array, uv_array, 
                     mapping_array, count_array, link_array, verbose ):
        
    """
    Link hillslope pixels downstream.
    
    Args:
        cl_src_path (str):
        which_cl_platform (int):
        which_cl_device (int):
        info_dict (numpy.ndarray):
        mask_array (numpy.ndarray):
        uv_array (numpy.ndarray):
        mapping_array (numpy.ndarray):
        count_array (numpy.ndarray):
        link_array (numpy.ndarray):
        verbose (bool):
        
    """
    vprint(verbose,'Linking hillslopes...')
    
    # Prepare CL essentials
    platform, device, context= pocl.prepare_cl_context(which_cl_platform,which_cl_device)
    queue = cl.CommandQueue(context,
                            properties=cl.command_queue_properties.PROFILING_ENABLE)
    cl_files = ['essentials.cl','updatetraj.cl','computestep.cl',
                'rungekutta.cl','countlink.cl']
    cl_kernel_source = ''
    for cl_file in cl_files:
        with open(os.path.join(cl_src_path,cl_file), 'r') as fp:
            cl_kernel_source += fp.read()
            
    # Generate a list (array) of seed points from all non-thin-channel pixels
    pad            = info_dict['pad_width']
    is_thinchannel = info_dict['is_thinchannel']
    seed_point_array \
        = pick_seeds(mask=mask_array, map=~mapping_array, flag=is_thinchannel, pad=pad)    
        
    # Specify arrays & CL buffers 
    array_dict = {' seed_point': {'array': seed_point_array, 'rwf': 'RO'},
                   'mask':       {'array': mask_array,       'rwf': 'RO'}, 
                   'uv':         {'array': uv_array,         'rwf': 'RO'}, 
                   'mapping':    {'array': mapping_array,    'rwf': 'RW'}, 
                   'count':      {'array': count_array,      'rwf': 'RW'}, 
                   'link':       {'array': link_array,       'rwf': 'RW'} }
    info_dict['n_seed_points'] = seed_point_array.shape[0]
    
    # Do integrations on the GPU
    cl_kernel_fn = 'link_hillslopes'
    pocl.gpu_compute(device, context, queue, cl_kernel_source,cl_kernel_fn, 
                     info_dict, array_dict, verbose)
    
    # Done
    vprint(verbose,'...done')  
    