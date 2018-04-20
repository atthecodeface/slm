"""
Kernel density estimation.
"""

import pyopencl as cl
import pyopencl.array
import numpy as np
import os
os.environ['PYTHONUNBUFFERED']='True'
import warnings

from streamlines import pocl
from streamlines.useful import vprint, pick_seeds

__all__ = ['histogram_univariate_pdf','gpu_compute','prepare_memory']

pdebug = print

def histogram_univariate_pdf( cl_src_path, which_cl_platform, which_cl_device, 
                              info_struct, sl_array, verbose ):
        
    """
    Compute univariate histogram.
    
    Args:
        cl_src_path (str):
        which_cl_platform (int):
        which_cl_device (int):
        info_struct (numpy.ndarray):
        sl_array (numpy.ndarray):
        verbose (bool):
    
    Returns:
        
        
    """
    vprint(verbose,'Computing univariate histogram...',end='')
    
    # Prepare CL essentials
    platform, device, context= pocl.prepare_cl_context(which_cl_platform,which_cl_device)
    queue = cl.CommandQueue(context)
    cl_files = ['kde.cl']
    cl_kernel_source = ''
    for cl_file in cl_files:
        with open(os.path.join(cl_src_path,cl_file), 'r') as fp:
            cl_kernel_source += fp.read()
            
    n_bins_x = info_struct['n_bins_x'][0]
    x_range = info_struct['x_range'][0]
    # Do integrations on the GPU
    cl_kernel_fn = 'histogram_univariate'
    histogram_array \
        = gpu_compute(device, context, queue, cl_kernel_source,cl_kernel_fn, info_struct, 
                      sl_array, verbose)
    histogram_array /= x_range 
    # Done
    vprint(verbose,'done')
    return histogram_array
    
def gpu_compute(device, context, queue, cl_kernel_source,cl_kernel_fn, info_struct, 
                sl_array, verbose):
    """
    Carry out GPU computation.
    
    Args:
        device (pyopencl.Device):
        context (pyopencl.Context):
        queue (pyopencl.CommandQueue):
        cl_kernel_source (str):
        cl_kernel_fn (str):
        info_struct (numpy.ndarray):
        sl_array (numpy.ndarray):
        verbose (bool):  
        
    """
        
    # Prepare memory, buffers 
    order = info_struct['array_order'][0]
    n_bins_x = info_struct['n_bins_x'][0]
    (histogram_array, sl_buffer, histogram_buffer) \
        = prepare_memory(context, queue, order, n_bins_x, sl_array, verbose)    
    # Specify this integration job's parameters
    global_size = [sl_array.shape[0],1]
    pdebug(global_size)
    local_size = None
    # Compile the CL code
    compile_options = pocl.set_compile_options(info_struct, cl_kernel_fn, 
                                               job_type='kde')
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        program = cl.Program(context, cl_kernel_source).build(options=compile_options)
    pocl.report_build_log(program, device, verbose)
    # Set the GPU kernel
    kernel = getattr(program,cl_kernel_fn)
    # Designate buffered arrays
    buffer_list = [sl_buffer, histogram_buffer]
    kernel.set_args(*buffer_list)
    kernel.set_scalar_arg_dtypes( [None]*len(buffer_list) )
    # Do the GPU compute
    event = cl.enqueue_nd_range_kernel(queue, kernel, global_size, local_size)
    # Fetch the data back from the GPU and finish
    cl.enqueue_copy(queue, histogram_array, histogram_buffer)
    queue.finish()   
    return histogram_array
    
def prepare_memory(context, queue, order, n_bins, sl_array, verbose):
    """
    Create PyOpenCL buffers and np-workalike arrays to allow CPU-GPU data transfer.
    
    Args:
        context (pyopencl.Context):
        queue (pyopencl.CommandQueue):
        order (str):
        n_bins (int):
        sl_array (numpy.ndarray):
        verbose (bool):
        
    Returns:
        numpy.ndarray, pyopencl.Buffer, pyopencl.Buffer: 
        histogram_array, sl_buffer, histogram_buffer
    """
    histogram_array = np.array((n_bins,1), dtype=np.float32,order=order)
     # Buffers to GPU memory
    COPY_READ_ONLY  = cl.mem_flags.READ_ONLY  | cl.mem_flags.COPY_HOST_PTR
    COPY_READ_WRITE = cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR
    sl_buffer         = cl.Buffer(context, COPY_READ_ONLY,  hostbuf=sl_array)
    histogram_buffer  = cl.Buffer(context, COPY_READ_WRITE, hostbuf=histogram_array)
    return (histogram_array, sl_buffer, histogram_buffer)
