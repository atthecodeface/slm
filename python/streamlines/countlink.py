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

__all__ = ['count_downchannels','flag_downchannels','link_hillslopes',
           'gpu_compute','prepare_memory']

pdebug = print

def count_downchannels( module_path, which_cl_platform, which_cl_device, info_struct, 
                        mask_array, u_array, v_array, 
                        mapping_array, count_array, link_array, verbose ):
        
    """
    Integrate and count downstream designating downstream links & thin channel status.
    
    Args:
        module_path (str):
        which_cl_platform (int):
        which_cl_device (int):
        info_struct (numpy.ndarray):
        mask_array (numpy.ndarray):
        u_array (numpy.ndarray):
        v_array (numpy.ndarray):
        mapping_array (numpy.ndarray):
        count_array (numpy.ndarray):
        link_array (numpy.ndarray):
        verbose (bool):
        
    """
    vprint(verbose,'Counting down channels...',end='')
    
    # Prepare CL essentials
    platform, device, context= pocl.prepare_cl_context(which_cl_platform,which_cl_device)
    queue = cl.CommandQueue(context)
    cl_files = ['essentials.cl','trajectoryfns.cl','computestep.cl',
                'integrationfns.cl','countlink.cl']
    cl_kernel_source = ''
    for cl_file in cl_files:
        with open(os.path.join(module_path,cl_file), 'r') as fp:
            cl_kernel_source += fp.read()
            
    # Generate a list (array) of seed points from the set of channel heads
    pad            = info_struct['pad_width'][0]
    is_channelhead = info_struct['is_channelhead'][0]
    is_thinchannel = info_struct['is_thinchannel'][0]
    order          = info_struct['array_order'][0]
    mapping_array[(mapping_array&is_thinchannel)==is_thinchannel] \
        = mapping_array[(mapping_array&is_thinchannel)==is_thinchannel]^is_thinchannel
    seed_point_array \
        = pick_seeds(mask=mask_array, map=mapping_array, flag=is_channelhead, 
                     order=order, pad=pad)
    # Do integrations on the GPU
    cl_kernel_fn = 'count_downchannels'
    gpu_compute(device, context, queue, cl_kernel_source,cl_kernel_fn, info_struct, 
                 seed_point_array, mask_array, u_array,v_array, 
                 mapping_array, count_array, link_array, verbose)
    
    # Done
    vprint(verbose,'done')  

def flag_downchannels( module_path, which_cl_platform, which_cl_device, info_struct, 
                       mask_array, u_array, v_array, 
                       mapping_array, count_array, link_array, verbose ):
        
    """
    Integrate downstream along channels & count pixel steps as we go.
    
    Args:
        module_path (str):
        which_cl_platform (int):
        which_cl_device (int):
        info_struct (numpy.ndarray):
        mask_array (numpy.ndarray):
        u_array (numpy.ndarray):
        v_array (numpy.ndarray):
        mapping_array (numpy.ndarray):
        count_array (numpy.ndarray):
        link_array (numpy.ndarray):
        verbose (bool):
        
    """
    vprint(verbose,'Flagging down channels...',end='')
    
    # Prepare CL essentials
    platform, device, context= pocl.prepare_cl_context(which_cl_platform,which_cl_device)
    queue = cl.CommandQueue(context)
    cl_files = ['essentials.cl','trajectoryfns.cl',
                'integrationfns.cl','countlink.cl']
    cl_kernel_source = ''
    for cl_file in cl_files:
        with open(os.path.join(module_path,cl_file), 'r') as fp:
            cl_kernel_source += fp.read()
            
    # Generate a list (array) of seed points from the set of channel heads
    pad            = info_struct['pad_width'][0]
    is_channelhead = info_struct['is_channelhead'][0]
    is_thinchannel = info_struct['is_thinchannel'][0]
    order          = info_struct['array_order'][0]
    mapping_array[(mapping_array&is_thinchannel)==is_thinchannel] \
        = mapping_array[(mapping_array&is_thinchannel)==is_thinchannel]^is_thinchannel
    count_array *= 0
    seed_point_array \
        = pick_seeds(mask=mask_array, map=mapping_array, flag=is_channelhead, 
                     order=order, pad=pad)                           
    # Do integrations on the GPU
    cl_kernel_fn = 'flag_downchannels'
    gpu_compute(device, context, queue, cl_kernel_source,cl_kernel_fn, info_struct, 
                 seed_point_array, mask_array, u_array,v_array, 
                 mapping_array, count_array, link_array, verbose)
    
    # Done
    vprint(verbose,'done')  

def link_hillslopes( module_path, which_cl_platform, which_cl_device, info_struct, 
                     mask_array, u_array, v_array, 
                     mapping_array, count_array, link_array, verbose ):
        
    """
    Link hillslope pixels downstream.
    
    Args:
        module_path (str):
        which_cl_platform (int):
        which_cl_device (int):
        info_struct (numpy.ndarray):
        mask_array (numpy.ndarray):
        u_array (numpy.ndarray):
        v_array (numpy.ndarray):
        mapping_array (numpy.ndarray):
        count_array (numpy.ndarray):
        link_array (numpy.ndarray):
        verbose (bool):
        
    """
    vprint(verbose,'Linking hillslopes...',end='')
    
    # Prepare CL essentials
    platform, device, context= pocl.prepare_cl_context(which_cl_platform,which_cl_device)
    queue = cl.CommandQueue(context)
    cl_files = ['essentials.cl','trajectoryfns.cl','computestep.cl',
                'integrationfns.cl','countlink.cl']
    cl_kernel_source = ''
    for cl_file in cl_files:
        with open(os.path.join(module_path,cl_file), 'r') as fp:
            cl_kernel_source += fp.read()
            
    # Generate a list (array) of seed points from all non-thin-channel pixels
    pad            = info_struct['pad_width'][0]
    is_thinchannel = info_struct['is_thinchannel'][0]
    order          = info_struct['array_order'][0]
    seed_point_array \
        = pick_seeds(mask=mask_array, map=~mapping_array, flag=is_thinchannel, 
                     order=order, pad=pad)                           
    # Do integrations on the GPU
    cl_kernel_fn = 'link_hillslopes'
    gpu_compute(device, context, queue, cl_kernel_source,cl_kernel_fn, info_struct, 
                 seed_point_array, mask_array, u_array,v_array, 
                 mapping_array, count_array, link_array, verbose)
    
    # Done
    vprint(verbose,'done')  
    
def gpu_compute(device,context,queue, cl_kernel_source,cl_kernel_fn, info_struct, 
                     seed_point_array, mask_array, u_array,v_array, 
                     mapping_array, count_array, link_array, verbose):
    """
    Carry out GPU computation.
    
    Args:
        device (pyopencl.Device):
        context (pyopencl.Context):
        queue (pyopencl.CommandQueue):
        cl_kernel_source (str):
        cl_kernel_fn (str):
        info_struct (numpy.ndarray):
        seed_point_array (numpy.ndarray):
        mask_array (numpy.ndarray):
        u_array (numpy.ndarray):
        v_array (numpy.ndarray):
        mapping_array (numpy.ndarray):
        count_array (numpy.ndarray):
        link_array (numpy.ndarray):
        verbose (bool):  
        
    """
    # Prepare memory, buffers 
    order = info_struct['array_order'][0]
    (seed_point_buffer, uv_buffer, mask_buffer,
     mapping_buffer, count_buffer, link_buffer) \
        = prepare_memory(context, queue, order, seed_point_array, mask_array, 
                         u_array,v_array, mapping_array,count_array,link_array, verbose)    
    # Specify this integration job's parameters
    if order=='F':
        global_size = [seed_point_array.shape[1],1]
    else:
        global_size = [seed_point_array.shape[0],1]
    local_size = None
    # Compile the CL code
    compile_options = pocl.set_compile_options(info_struct, cl_kernel_fn, downup_sign=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        program = cl.Program(context, cl_kernel_source).build(options=compile_options)
    pocl.report_build_log(program, device, verbose)
    # Set the GPU kernel
    kernel = getattr(program,cl_kernel_fn)
    # Designate buffered arrays
    buffer_list = [seed_point_buffer, mask_buffer ,uv_buffer, 
                   mapping_buffer, count_buffer, link_buffer]
    kernel.set_args(*buffer_list)
    kernel.set_scalar_arg_dtypes( [None]*len(buffer_list) )
    # Do the GPU compute
    event = cl.enqueue_nd_range_kernel(queue, kernel, global_size, local_size)
    # Fetch the data back from the GPU and finish
    cl.enqueue_copy(queue, mapping_array, mapping_buffer)
    cl.enqueue_copy(queue, count_array, count_buffer)
    cl.enqueue_copy(queue, link_array, link_buffer)
    
    queue.finish()   
    
def prepare_memory(context, queue, order,
                    seed_point_array, mask_array, u_array,v_array, 
                    mapping_array, count_array, link_array, verbose):
    """
    Create PyOpenCL buffers and np-workalike arrays to allow CPU-GPU data transfer.
    
    Args:
        context (pyopencl.Context):
        queue (pyopencl.CommandQueue):
        seed_point_array (numpy.ndarray):
        mask_array (numpy.ndarray):
        u_array (numpy.ndarray):
        v_array (numpy.ndarray):
        mapping_array (numpy.ndarray):
        count_array (numpy.ndarray):
        link_array (numpy.ndarray):
        verbose (bool):
        
    Returns:
        pyopencl.Buffer, pyopencl.Buffer, pyopencl.Buffer, \
        pyopencl.Buffer, pyopencl.Buffer, pyopencl.Buffer:
            seed_point_buffer, uv_buffer, mask_buffer,  \
            mapping_buffer, count_buffer, link_buffer
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
    count_buffer      = cl.Buffer(context, COPY_READ_WRITE, hostbuf=count_array)
    link_buffer       = cl.Buffer(context, COPY_READ_WRITE, hostbuf=link_array)
    return (seed_point_buffer, uv_buffer, mask_buffer, 
            mapping_buffer, count_buffer, link_buffer)
