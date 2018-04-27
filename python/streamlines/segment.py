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

__all__ = ['segment_channels','segment_hillslopes','subsegment_flanks',
           'gpu_compute','prepare_memory']

pdebug = print

def segment_channels( cl_src_path, which_cl_platform, which_cl_device, info_dict, 
                          mask_array, u_array, v_array,
                          mapping_array, count_array, link_array, label_array, verbose ):
        
    """
    Label channel confluences.
    
    Args:
        cl_src_path (str):
        which_cl_platform (int):
        which_cl_device (int):
        info_dict (numpy.ndarray):
        mask_array (numpy.ndarray):
        u_array (numpy.ndarray):
        v_array (numpy.ndarray):
        mapping_array (numpy.ndarray):
        count_array (numpy.ndarray):
        link_array (numpy.ndarray):
        label_array (numpy.ndarray):
        verbose (bool):
        
    """
    vprint(verbose,'Segmenting channels...',end='')
    
    # Prepare CL essentials
    platform, device, context= pocl.prepare_cl_context(which_cl_platform,which_cl_device)
    queue = cl.CommandQueue(context)
    cl_files = ['essentials.cl','trajectoryfns.cl',
                'integrationfns.cl','segment.cl']
    cl_kernel_source = ''
    for cl_file in cl_files:
        with open(os.path.join(cl_src_path,cl_file), 'r') as fp:
            cl_kernel_source += fp.read()
            
    # Trace downstream from all channel heads until masked boundary is reachedd
    #    /or/ if a major confluence is reached, only keeping going if dominant
    pad = info_dict['pad_width']
    is_channelhead = info_dict['is_channelhead']
    order = info_dict['array_order']
    seed_point_array \
        = pick_seeds(mask=mask_array, map=mapping_array, flag=is_channelhead, 
                     order=order, pad=pad)
    # Do integrations on the GPU
    cl_kernel_fn = 'segment_downchannels'
    gpu_compute(device, context, queue, cl_kernel_source,cl_kernel_fn, info_dict, 
                seed_point_array, mask_array, u_array,v_array, 
                mapping_array, count_array, link_array, label_array, verbose)

    # Relabel channel segments in simple sequence 1,2,3,... 
    #  instead of using array indices as labels
    channel_segments_array = label_array[label_array>0 & (~mask_array)].ravel()
    channel_segment_labels_array = np.unique(channel_segments_array)
    for idx,label in enumerate(channel_segment_labels_array):
        label_array[label_array==label]=idx+1
    n_segments = idx+1
    vprint(verbose, ' number of segments={}... '.format(n_segments),end='')

    # Done
    vprint(verbose,'done')  
    return n_segments

def segment_hillslopes( cl_src_path, which_cl_platform, which_cl_device, info_dict, 
                        mask_array, u_array, v_array,
                        mapping_array, count_array, link_array, label_array, verbose ):
        
    """
    Label hillslope pixels.
    
    Args:
        cl_src_path (str):
        which_cl_platform (int):
        which_cl_device (int):
        info_dict (numpy.ndarray):
        mask_array (numpy.ndarray):
        u_array (numpy.ndarray):
        v_array (numpy.ndarray):
        mapping_array (numpy.ndarray):
        count_array (numpy.ndarray):
        link_array (numpy.ndarray):
        label_array (numpy.ndarray):
        verbose (bool):
        
    """
    vprint(verbose,'Segmenting hillslopes...',end='')
    
    # Prepare CL essentials
    platform, device, context= pocl.prepare_cl_context(which_cl_platform,which_cl_device)
    queue = cl.CommandQueue(context)
    cl_files = ['essentials.cl','trajectoryfns.cl','computestep.cl',
                'integrationfns.cl','segment.cl']
    cl_kernel_source = ''
    for cl_file in cl_files:
        with open(os.path.join(cl_src_path,cl_file), 'r') as fp:
            cl_kernel_source += fp.read()
            
    # Trace downstream from all channel heads until masked boundary is reachedd
    #    /or/ if a major confluence is reached, only keeping going if dominant
    pad            = info_dict['pad_width']
    is_channelhead = info_dict['is_channelhead']
    order          = info_dict['array_order']
    flag = is_channelhead
    seed_point_array \
        = pick_seeds(mask=mask_array, map=~mapping_array, flag=flag, order=order, pad=pad)
    # Do integrations on the GPU
    cl_kernel_fn = 'segment_hillslopes'
    gpu_compute(device, context, queue, cl_kernel_source,cl_kernel_fn, info_dict, 
                seed_point_array, mask_array, u_array,v_array, 
                mapping_array, count_array, link_array, label_array, verbose)

    # Done
    vprint(verbose,'done')  

def subsegment_flanks( cl_src_path, which_cl_platform, which_cl_device, info_dict, 
                       mask_array, u_array, v_array,
                       mapping_array, channel_label_array, link_array, label_array, 
                       verbose ):
        
    """
    Subsegment left (and implicitly right) flanks.
    
    Args:
        cl_src_path (str):
        which_cl_platform (int):
        which_cl_device (int):
        info_dict (numpy.ndarray):
        mask_array (numpy.ndarray):
        u_array (numpy.ndarray):
        v_array (numpy.ndarray):
        mapping_array (numpy.ndarray):
        channel_label_array (numpy.ndarray):
        link_array (numpy.ndarray):
        label_array (numpy.ndarray):
        verbose (bool):
        
    """
    vprint(verbose,'Subsegmenting flanks...',end='')
    
    # Prepare CL essentials
    platform, device, context= pocl.prepare_cl_context(which_cl_platform,which_cl_device)
    queue = cl.CommandQueue(context)
    cl_files = ['essentials.cl','trajectoryfns.cl','computestep.cl',
                'integrationfns.cl','segment.cl']
    cl_kernel_source = ''
    for cl_file in cl_files:
        with open(os.path.join(cl_src_path,cl_file), 'r') as fp:
            cl_kernel_source += fp.read()
            
    # Trace downstream from all major confluences /or/ channel heads
    pad                = info_dict['pad_width']
    is_channelhead     = info_dict['is_channelhead']
    is_majorconfluence = info_dict['is_majorconfluence']
    is_thinchannel     = info_dict['is_thinchannel']
    is_leftflank       = info_dict['is_leftflank']
    order              = info_dict['array_order']
    flag = is_channelhead | is_majorconfluence
    seed_point_array \
        = pick_seeds(mask=mask_array, map=mapping_array, flag=flag, order=order, pad=pad)
    # Do integrations on the GPU
    if (    (order=='F' and seed_point_array.shape[1]>0)
         or (order=='C' and seed_point_array.shape[0]>0) ):
        cl_kernel_fn = 'subsegment_channel_edges'
        gpu_compute(device, context, queue, cl_kernel_source,cl_kernel_fn, info_dict, 
                    seed_point_array, mask_array, u_array,v_array, 
                    mapping_array, channel_label_array, link_array, label_array, verbose)
        
    # Trace downstream from all non-left-flank hillslope pixels
    flag = is_leftflank | is_thinchannel
    seed_point_array \
        = pick_seeds(mask=mask_array, map=~mapping_array, flag=flag, order=order,pad=pad)
    # Do integrations on the GPU
    cl_kernel_fn = 'subsegment_flanks'
    gpu_compute(device, context, queue, cl_kernel_source, cl_kernel_fn, info_dict, 
                seed_point_array, mask_array, u_array,v_array, 
                mapping_array, channel_label_array, link_array, label_array, verbose)
        
    
    # Done
    vprint(verbose,'done')  
  
def gpu_compute(device, context, queue, cl_kernel_source,cl_kernel_fn, info_dict, 
                seed_point_array, mask_array, u_array, v_array, 
                mapping_array, count_array, link_array, label_array, verbose):
    """
    Carry out GPU computation.
    
    Args:
        device (pyopencl.Device):
        context (pyopencl.Context):
        queue (pyopencl.CommandQueue):
        cl_kernel_source (str):
        cl_kernel_fn (str):
        info_dict (numpy.ndarray):
        seed_point_array (numpy.ndarray):
        mask_array (numpy.ndarray):
        u_array (numpy.ndarray):
        v_array (numpy.ndarray):
        slt_array (numpy.ndarray):
        mapping_array (numpy.ndarray):
        count_array (numpy.ndarray):
        link_array (numpy.ndarray):
        label_array (numpy.ndarray):
        verbose (bool):  
        
    """
        
    # Prepare memory, buffers 
    order = info_dict['array_order']
    (seed_point_buffer, uv_buffer, mask_buffer, 
     mapping_buffer, count_buffer, link_buffer, label_buffer) \
        = prepare_memory(context, queue, order, 
                         seed_point_array, mask_array, u_array,v_array, 
                         mapping_array, count_array, link_array, label_array, verbose)    
    # Specify this integration job's parameters
    if order=='F':
        global_size = [seed_point_array.shape[1],1]
    else:
        global_size = [seed_point_array.shape[0],1]
    local_size = None
    # Compile the CL code
    compile_options = pocl.set_compile_options(info_dict, cl_kernel_fn, downup_sign=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        program = cl.Program(context, cl_kernel_source).build(options=compile_options)
    pocl.report_build_log(program, device, verbose)
    # Set the GPU kernel
    kernel = getattr(program,cl_kernel_fn)
    # Designate buffered arrays
    buffer_list = [seed_point_buffer, mask_buffer, uv_buffer, 
                   mapping_buffer, count_buffer, link_buffer, label_buffer]
    kernel.set_args(*buffer_list)
    kernel.set_scalar_arg_dtypes( [None]*len(buffer_list) )
    # Do the GPU compute
    event = cl.enqueue_nd_range_kernel(queue, kernel, global_size, local_size)
    # Fetch the data back from the GPU and finish
    cl.enqueue_copy(queue, mapping_array, mapping_buffer)
    cl.enqueue_copy(queue, label_array, label_buffer)
    
    queue.finish()   
    
def prepare_memory(context,queue, order, seed_point_array, mask_array, u_array,v_array, 
                   mapping_array, count_array, link_array, label_array, verbose):
    """
    Create PyOpenCL buffers and np-workalike arrays to allow CPU-GPU data transfer.
    
    Args:
        context (pyopencl.Context):
        queue (pyopencl.CommandQueue):
        order (str):
        seed_point_array (numpy.ndarray):
        mask_array (numpy.ndarray):
        u_array (numpy.ndarray):
        v_array (numpy.ndarray):
        mapping_array (numpy.ndarray):
        count_array (numpy.ndarray):
        link_array (numpy.ndarray):
        label_array (numpy.ndarray):
        verbose (bool):
        
    Returns:
        pyopencl.Buffer, pyopencl.Buffer, pyopencl.Buffer, \
        pyopencl.Buffer, pyopencl.Buffer, pyopencl.Buffer, pyopencl.Buffer:
            seed_point_buffer, uv_buffer, mask_buffer, \
            mapping_buffer, count_buffer, link_buffer, label_buffer
    """
    # Buffer for mask, (u,v) velocity array and more 
    if order=='F':
        uv_array = np.stack((u_array,v_array)).copy().astype(dtype=np.float32,
                                                             order=order)
    else:
        uv_array = np.stack((u_array,v_array),axis=2).copy().astype(dtype=np.float32,
                                                                    order=order)
     # Buffers to GPU memory
    COPY_READ_ONLY  = cl.mem_flags.READ_ONLY  | cl.mem_flags.COPY_HOST_PTR
    COPY_READ_WRITE = cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR
    seed_point_buffer = cl.Buffer(context, COPY_READ_ONLY,  hostbuf=seed_point_array)
    uv_buffer         = cl.Buffer(context, COPY_READ_ONLY,  hostbuf=uv_array)
    mask_buffer       = cl.Buffer(context, COPY_READ_ONLY,  hostbuf=mask_array)
    mapping_buffer    = cl.Buffer(context, COPY_READ_WRITE, hostbuf=mapping_array)
    count_buffer      = cl.Buffer(context, COPY_READ_ONLY,  hostbuf=count_array)
    link_buffer       = cl.Buffer(context, COPY_READ_ONLY,  hostbuf=link_array)
    label_buffer      = cl.Buffer(context, COPY_READ_WRITE, hostbuf=label_array)
    return (seed_point_buffer, uv_buffer, mask_buffer, 
            mapping_buffer, count_buffer, link_buffer, label_buffer)
