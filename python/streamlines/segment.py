"""
Segment downstream.
"""

import pyopencl as cl
import pyopencl.array
import numpy as np
import os
os.environ['PYTHONUNBUFFERED']='True'
import warnings

from streamlines import pocl
from streamlines.useful import vprint, pick_seeds

__all__ = ['segment_channels','segment_hillslopes','subsegment_flanks']

pdebug = print

def segment_channels( cl_state, info, data, verbose ):
    """
    Label channel confluences.
    
    Args:
        cl_state (obj):
        info (obj):
        data (obj):
        verbose (bool):
        
    """
    vprint(verbose,'Segmenting channels...')
    
    # Prepare CL essentials
    cl_state.kernel_source \
        = pocl.read_kernel_source(cl_state.src_path,['essentials.cl','updatetraj.cl',
                                                     'rungekutta.cl','segment.cl'])
            
    # Trace downstream from all channel heads until masked boundary is reached
    #    /or/ a major confluence is reached, only keeping going if dominant
    pad            = info.pad_width
    is_channelhead = info.is_channelhead
    flag           = is_channelhead
    seed_point_array \
        = pick_seeds(mask=data.mask_array, map=data.mapping_array, 
                     flag=flag, pad=pad)
        
    # Prepare memory, buffers 
    array_dict = { 'seed_point': {'array': seed_point_array,      'rwf': 'RO'},
                   'mask':       {'array': data.mask_array,       'rwf': 'RO'}, 
                   'uv':         {'array': data.uv_array,         'rwf': 'RO'}, 
                   'mapping':    {'array': data.mapping_array,    'rwf': 'RW'}, 
                   'count':      {'array': data.count_array,      'rwf': 'RO'}, 
                   'link':       {'array': data.link_array,       'rwf': 'RO'}, 
                   'label':      {'array': data.label_array,      'rwf': 'RW'} }
    info.n_seed_points = seed_point_array.shape[0]
    
    # Do integrations on the GPU
    cl_state.kernel_fn = 'segment_downchannels'
    pocl.gpu_compute(cl_state, info, array_dict, verbose)

    # Relabel channel segments in simple sequence 1,2,3,... 
    #  instead of using array indices as labels
    channel_segments_array \
        = data.label_array[data.label_array>0 & (~data.mask_array)].ravel()
    channel_segment_labels_array = np.unique(channel_segments_array)
    for idx,label in enumerate(channel_segment_labels_array):
        data.label_array[data.label_array==label]=idx+1
    n_segments = idx+1
    vprint(verbose, ' number of segments={}...'.format(n_segments),end='')

    # Done
    vprint(verbose,'...done')  
    return n_segments

def segment_hillslopes( cl_state, info, data, verbose ):
    """
    Label hillslope pixels.
    
    Args:
        cl_state (obj):
        info (obj):
        data (obj):
        verbose (bool):
        
    """
    vprint(verbose,'Segmenting hillslopes...')
    
    # Prepare CL essentials
    cl_state.kernel_source \
        = pocl.read_kernel_source(cl_state.src_path,['essentials.cl','updatetraj.cl',
                                                     'computestep.cl','rungekutta.cl',
                                                     'segment.cl'])
            
    # Trace downstream from all channel heads until masked boundary is reached
    #    /or/ if a major confluence is reached, only keeping going if dominant
    pad            = info.pad_width
    is_channelhead = info.is_channelhead
    flag           = is_channelhead
    seed_point_array = pick_seeds(mask=data.mask_array, map=~data.mapping_array, 
                                  flag=flag,pad=pad)
    
    # Prepare memory, buffers 
    array_dict = { 'seed_point': {'array': seed_point_array,      'rwf': 'RO'},
                   'mask':       {'array': data.mask_array,       'rwf': 'RO'}, 
                   'uv':         {'array': data.uv_array,         'rwf': 'RO'}, 
                   'mapping':    {'array': data.mapping_array,    'rwf': 'RW'}, 
                   'count':      {'array': data.count_array,      'rwf': 'RO'}, 
                   'link':       {'array': data.link_array,       'rwf': 'RO'}, 
                   'label':      {'array': data.label_array,      'rwf': 'RW'} }
    info.n_seed_points = seed_point_array.shape[0]
    
    # Do integrations on the GPU
    cl_state.kernel_fn = 'segment_hillslopes'
    pocl.gpu_compute(cl_state, info, array_dict, verbose)

    # Done
    vprint(verbose,'...done')  

def subsegment_flanks( cl_state, info, data, verbose ):
    """
    Subsegment left (and implicitly right) flanks.
    
    Args:
        cl_state (obj):
        info (obj):
        data (obj):
        verbose (bool):
        
    """
    vprint(verbose,'Subsegmenting flanks...')
    
    # Prepare CL essentials
    cl_state.kernel_source \
        = pocl.read_kernel_source(cl_state.src_path,['essentials.cl','updatetraj.cl',
                                                     'computestep.cl','rungekutta.cl',
                                                     'segment.cl'])
            
    # Trace downstream from all major confluences /or/ channel heads
    pad                = info.pad_width
    is_channelhead     = info.is_channelhead
    was_channelhead    = info.was_channelhead
    is_majorconfluence = info.is_majorconfluence
    is_thinchannel     = info.is_thinchannel
    is_leftflank       = info.is_leftflank
    flag               = is_channelhead | is_majorconfluence
    seed_point_array = pick_seeds(mask=data.mask_array, map=data.mapping_array, 
                                  flag=flag, pad=pad)
    
    # Specify arrays & CL buffers 
    array_dict = { 'seed_point':    {'array': seed_point_array,         'rwf': 'RO'},
                   'mask':          {'array': data.mask_array,          'rwf': 'RO'}, 
                   'uv':            {'array': data.uv_array,            'rwf': 'RO'}, 
                   'mapping':       {'array': data.mapping_array,       'rwf': 'RW'}, 
                   'channel_label': {'array': data.channel_label_array, 'rwf': 'RO'}, 
                   'link':          {'array': data.link_array,          'rwf': 'RO'}, 
                   'label':         {'array': data.label_array,         'rwf': 'RW'} }
    info.n_seed_points = seed_point_array.shape[0]
    
    # Do integrations on the GPU
    if ( info.n_seed_points>0 ):
        cl_state.kernel_fn = 'subsegment_channel_edges'
        pocl.gpu_compute(cl_state, info, array_dict, verbose)
            
    # Trace downstream from all non-left-flank hillslope pixels
    flag = is_leftflank | is_thinchannel
    seed_point_array = pick_seeds(mask=data.mask_array, map=~data.mapping_array, 
                                  flag=flag,pad=pad)
    array_dict['seed_point']['array'] = seed_point_array
    info.n_seed_points = seed_point_array.shape[0]
    
    # Do integrations on the GPU
    cl_state.kernel_fn = 'subsegment_flanks'
    pocl.gpu_compute(cl_state, info, array_dict, verbose)
    
    # Done
    vprint(verbose,'...done')  
  