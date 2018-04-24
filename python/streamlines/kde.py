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

__all__ = ['estimate_univariate_pdf','estimate_bivariate_pdf',
           'gpu_compute_1d','prepare_memory_1d']

pdebug = print

def estimate_bivariate_pdf( cl_src_path, which_cl_platform, which_cl_device, 
                            info_struct, sl_array, verbose ):
        
    """
    Compute bivariate histogram and subsequent kernel-density smoothed pdf.
    
    Args:
        cl_src_path (str):
        which_cl_platform (int):
        which_cl_device (int):
        info_struct (numpy.ndarray):
        sl_array (numpy.ndarray):
        verbose (bool):
    
    Returns:
        
        
    """
    vprint(verbose,'Computing bivariate pdf...',end='')
    
    # Prepare CL essentials
    platform, device, context= pocl.prepare_cl_context(which_cl_platform,which_cl_device)
    queue = cl.CommandQueue(context)
    
    cl_files = ['kde.cl']
    cl_kernel_source = ''
    for cl_file in cl_files:
        with open(os.path.join(cl_src_path,cl_file), 'r') as fp:
            cl_kernel_source += fp.read()
            
    n_data       = info_struct['n_data'][0]
    
    x_range      = info_struct['x_range'][0]
    bin_dx       = info_struct['bin_dx'][0]
    pdf_dx       = info_struct['pdf_dx'][0]
    stddev_x     = np.std(sl_array[:,0])
    
    y_range      = info_struct['y_range'][0]
    bin_dy       = info_struct['bin_dy'][0]
    pdf_dy       = info_struct['pdf_dy'][0]
    stddev_y      = np.std(sl_array[:,1])
        
    # Set up kernel filter
    # Silverman hack for now
    kdf_width_x = 1.06*stddev_x*np.power(n_data,-0.2)*8
    kdf_width_y = 1.06*stddev_y*np.power(n_data,-0.2)*8
    info_struct['kdf_width_x'][0] = kdf_width_x
    info_struct['kdf_width_y'][0] = kdf_width_y
    info_struct['n_kdf_part_points_x'][0] = np.uint32(np.floor(kdf_width_x/pdf_dx))//2
    info_struct['n_kdf_part_points_y'][0] = np.uint32(np.floor(kdf_width_y/pdf_dy))//2
    
#     pdebug('\nstddev_x',stddev_x,'stddev_y',stddev_y)
            
    vprint(verbose,'histogram...',end='')
    # Histogram
    cl_kernel_fn = 'histogram_bivariate'
    histogram_array \
        = gpu_compute(device, context, queue, cl_kernel_source, cl_kernel_fn, 
                      info_struct, sl_array,histogram_array='create', 
                      is_bivariate=True, verbose=verbose)
        
    vprint(verbose,'kernel filtering...',end='')
    histogram_array = histogram_array.astype(np.float32)
#     return histogram_array/(np.sum(histogram_array)*bin_dx*bin_dy)
    # PDF
    cl_kernel_fn = 'pdf_bivariate'
    pdf_array \
        = gpu_compute(device, context, queue, cl_kernel_source, cl_kernel_fn, 
                      info_struct, histogram_array=histogram_array,pdf_array='create',
                      is_bivariate=True, verbose=verbose)
    # Done
    vprint(verbose,'done')
    pdebug('\n',pdf_array)
    pdebug('\n',np.sum(pdf_array))
    return pdf_array/(np.sum(pdf_array)*bin_dx*bin_dy)
    
def estimate_univariate_pdf( cl_src_path, which_cl_platform, which_cl_device, 
                             info_struct, sl_array, verbose ):
        
    """
    Compute univariate histogram and subsequent kernel-density smoothed pdf.
    
    Args:
        cl_src_path (str):
        which_cl_platform (int):
        which_cl_device (int):
        info_struct (numpy.ndarray):
        sl_array (numpy.ndarray):
        verbose (bool):
    
    Returns:
        
        
    """
    vprint(verbose,'Computing univariate pdf...',end='')
    
    # Prepare CL essentials
    platform, device, context= pocl.prepare_cl_context(which_cl_platform,which_cl_device)
    queue = cl.CommandQueue(context)
    
    cl_files = ['kde.cl']
    cl_kernel_source = ''
    for cl_file in cl_files:
        with open(os.path.join(cl_src_path,cl_file), 'r') as fp:
            cl_kernel_source += fp.read()
            
    n_data       = info_struct['n_data'][0]
    x_range      = info_struct['x_range'][0]
    bin_dx       = info_struct['bin_dx'][0]
    pdf_dx       = info_struct['pdf_dx'][0]
    stddev       = np.std(sl_array)
    
    # Set up kernel filter
    # Silverman hack for now
    kdf_width_x = 1.06*stddev*np.power(n_data,-0.2)*8
    info_struct['kdf_width_x'][0] = kdf_width_x
    info_struct['n_kdf_part_points_x'][0] = np.uint32(np.floor(kdf_width_x/pdf_dx))//2
        
    vprint(verbose,'histogram...',end='')
    # Histogram
    cl_kernel_fn = 'histogram_univariate'
    histogram_array \
        = gpu_compute(device, context, queue, cl_kernel_source, cl_kernel_fn, 
                      info_struct, sl_array=sl_array, histogram_array='create', 
                      verbose=verbose)
        
    vprint(verbose,'kernel filtering...',end='')
    # PDF
    cl_kernel_fn = 'pdf_univariate'
    pdf_array \
        = gpu_compute(device, context, queue, cl_kernel_source, cl_kernel_fn, 
                      info_struct, histogram_array=histogram_array, pdf_array='create',
                      verbose=verbose)
    # Done
    vprint(verbose,'done')
    
    return pdf_array/(sum(pdf_array)*bin_dx)
    
def gpu_compute( device, context, queue, cl_kernel_source, cl_kernel_fn, info_struct,
                    sl_array=None, histogram_array='create', pdf_array=None, 
                    is_bivariate=False, verbose=False ):
    """
    Carry out GPU computation of histogram.
    
    Args:
        device (pyopencl.Device):
        context (pyopencl.Context):
        queue (pyopencl.CommandQueue):
        cl_kernel_source (str):
        cl_kernel_fn (str):
        info_struct (numpy.ndarray):
        sl_array (numpy.ndarray):
        histogram_array (numpy.ndarray):
        verbose (bool):  
        
    """
        
    # Prepare memory, buffers 
    order        = info_struct['array_order'][0]
    n_hist_bins  = info_struct['n_hist_bins'][0]
    n_data       = info_struct['n_data'][0]
    n_pdf_points = info_struct['n_pdf_points'][0]
    
    if type(histogram_array) is str and histogram_array=='create':
        do_pdf = False
        # Compute histogram
        (histogram_array, sl_buffer, histogram_buffer) \
            = prepare_memory(context, queue, order, 
                             n_hist_bins=n_hist_bins, 
                             sl_array=sl_array, 
                             histogram_array='create', is_bivariate=is_bivariate,
                             verbose=verbose)    
        global_size = [n_data,1]
        buffer_list = [sl_buffer, histogram_buffer]
    else:
        # Compute pdf
        do_pdf = True
        (pdf_array, histogram_buffer, pdf_buffer) \
            = prepare_memory(context, queue, order, 
                             n_pdf_points=n_pdf_points, 
                             histogram_array=histogram_array, 
                             pdf_array='create', is_bivariate=is_bivariate,
                             verbose=verbose)    
        global_size = [n_pdf_points,1]
        buffer_list = [histogram_buffer, pdf_buffer]
    local_size = None
    # Compile the CL code
    compile_options = pocl.set_compile_options(info_struct, cl_kernel_fn, job_type='kde')
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        program = cl.Program(context, cl_kernel_source).build(options=compile_options)
    pocl.report_build_log(program, device, verbose)
    # Set the GPU kernel
#     pdebug(compile_options)
#     pdebug(buffer_list)
    kernel = getattr(program,cl_kernel_fn)
    # Designate buffered arrays
    kernel.set_args(*buffer_list)
    kernel.set_scalar_arg_dtypes( [None]*len(buffer_list) )
    # Do the GPU compute
    event = cl.enqueue_nd_range_kernel(queue, kernel, global_size, local_size)
    # Fetch the data back from the GPU and finish
    if do_pdf:
        # Compute pdf
        cl.enqueue_copy(queue, pdf_array, pdf_buffer)
        queue.finish()
        return pdf_array
    else:
        # Compute histogram
        cl.enqueue_copy(queue, histogram_array, histogram_buffer)
        queue.finish()   
        return histogram_array
    
def prepare_memory(context, queue, order, 
                   n_hist_bins=0, n_pdf_points=0,
                   sl_array=None, histogram_array=None, 
                   pdf_array=None, is_bivariate=False,
                   verbose=False ):
    """
    Create PyOpenCL buffers and np-workalike arrays to allow CPU-GPU data transfer.
    
    Args:
        context (pyopencl.Context):
        queue (pyopencl.CommandQueue):
        order (str):
        n_bins (int):
        sl_array (numpy.ndarray):
        histogram_array (numpy.ndarray):
        pdf_array (numpy.ndarray):
        verbose (bool):
        
    Returns:
        numpy.ndarray, pyopencl.Buffer, pyopencl.Buffer: 
        histogram_array or pdf_array, sl_buffer or histogram_buffer, 
        histogram_buffer or pdf_buffer
    """
    COPY_READ_ONLY  = cl.mem_flags.READ_ONLY  | cl.mem_flags.COPY_HOST_PTR
    COPY_READ_WRITE = cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR

    if sl_array is not None:
        sl_buffer         = cl.Buffer(context, COPY_READ_ONLY,  hostbuf=sl_array)
    nx = n_hist_bins
    if is_bivariate:
        ny = n_hist_bins
    else:
        ny = 1
    if type(histogram_array) is str and histogram_array=='create':
        histogram_array   = np.zeros((nx,ny), dtype=np.uint32,order=order)
        histogram_buffer  = cl.Buffer(context, COPY_READ_WRITE, hostbuf=histogram_array)
    else:
        histogram_buffer  = cl.Buffer(context, COPY_READ_ONLY, hostbuf=histogram_array)
    nx = n_pdf_points
    if is_bivariate:
        ny = n_pdf_points
    else:
        ny = 1
    if type(pdf_array) is str and pdf_array=='create':
        pdf_array         = np.zeros((nx,ny), dtype=np.float32,order=order)
        pdf_buffer        = cl.Buffer(context, COPY_READ_WRITE, hostbuf=pdf_array)
        
    # Deduce which array and buffers to return from context
    if pdf_array is None:
        return (histogram_array, sl_buffer, histogram_buffer) 
    else:
        return (pdf_array, histogram_buffer, pdf_buffer) 
