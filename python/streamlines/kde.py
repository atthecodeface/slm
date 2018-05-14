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
                            info, sl_array, verbose ):
        
    """
    Compute bivariate histogram and subsequent kernel-density smoothed pdf.
    
    Args:
        cl_src_path (str):
        which_cl_platform (int):
        which_cl_device (int):
        info (numpy.ndarray):
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
            
    n_data       = info.n_data
    bandwidth    = info.kdf_bandwidth
    
    x_range      = info.x_range
    bin_dx       = info.bin_dx
    pdf_dx       = info.pdf_dx
    stddev_x     = np.std(sl_array[0,:])
    
    y_range      = info.y_range
    bin_dy       = info.bin_dy
    pdf_dy       = info.pdf_dy
    stddev_y     = np.std(sl_array[1,:])
            
    # Set up kernel filter
    # Silverman hack turned off
#     kdf_width_x = 1.06*stddev_x*np.power(n_data,-0.2)*20
#     kdf_width_y = 1.06*stddev_y*np.power(n_data,-0.2)*20
    kdf_width_x = stddev_x*bandwidth*3
    kdf_width_y = stddev_y*bandwidth*3*2
    info.kdf_width_x = kdf_width_x
    info.n_kdf_part_points_x= 2*(np.uint32(np.floor(kdf_width_x/bin_dx))//2)
    info.kdf_width_y = kdf_width_y
    info.n_kdf_part_points_y= 2*(np.uint32(np.floor(kdf_width_y/bin_dy))//2)
            
    vprint(verbose,'histogram...',end='')
    # Histogram
    cl_kernel_fn = 'histogram_bivariate'
    histogram_array \
        = gpu_compute(device, context, queue, cl_kernel_source, cl_kernel_fn, 
                      info, action='histogram',  is_bivariate=True, 
                      sl_array=sl_array, verbose=verbose)
        
    vprint(verbose,'kernel filtering rows...',end='')
    # PDF
    cl_kernel_fn = 'pdf_bivariate_rows'
    partial_pdf_array \
        = gpu_compute(device, context, queue, cl_kernel_source, cl_kernel_fn, 
                      info, action='partial_pdf',  is_bivariate=True, 
                      histogram_array=histogram_array,
                      verbose=verbose)

    vprint(verbose,'kernel filtering columns...',end='')
    cl_kernel_fn = 'pdf_bivariate_cols'
    pdf_array \
        = gpu_compute(device, context, queue, cl_kernel_source, cl_kernel_fn, 
                      info,  action='full_pdf',  is_bivariate=True, 
                      partial_pdf_array=partial_pdf_array,
                      verbose=verbose)
    # Done
    vprint(verbose,'done')
    return pdf_array/(np.sum(pdf_array)*bin_dx*bin_dy)
    
def estimate_univariate_pdf( cl_src_path, which_cl_platform, which_cl_device, 
                             info, sl_array, verbose ):
        
    """
    Compute univariate histogram and subsequent kernel-density smoothed pdf.
    
    Args:
        cl_src_path (str):
        which_cl_platform (int):
        which_cl_device (int):
        info (numpy.ndarray):
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
            
    n_data       = info.n_data
    x_range      = info.x_range
    bandwidth    = info.kdf_bandwidth
    bin_dx       = info.bin_dx
    pdf_dx       = info.pdf_dx
    stddev       = np.std(sl_array)
    
    # Set up kernel filter
    # Hacked Silverman hack - why is 8x good?
    kdf_width_x = 1.06*stddev*np.power(n_data,-0.2)*bandwidth*10
    info.kdf_width_x = kdf_width_x
    info.n_kdf_part_points_x = np.uint32(np.floor(kdf_width_x/pdf_dx))//2
        
    vprint(verbose,'histogram...',end='')
    # Histogram
    cl_kernel_fn = 'histogram_univariate'
    histogram_array \
        = gpu_compute(device, context, queue, cl_kernel_source, cl_kernel_fn, 
                      info, action='histogram', sl_array=sl_array, 
                      verbose=verbose)
        
    vprint(verbose,'kernel filtering...',end='')
    # PDF
    cl_kernel_fn = 'pdf_univariate'
    pdf_array \
        = gpu_compute(device, context, queue, cl_kernel_source, cl_kernel_fn, 
                      info, action='full_pdf', histogram_array=histogram_array,
                      verbose=verbose)
    # Done
    vprint(verbose,'done')
    
    return pdf_array/(sum(pdf_array)*bin_dx)
    
def gpu_compute( device, context, queue, cl_kernel_source, cl_kernel_fn, info,
                 action='histogram', is_bivariate=False, 
                 sl_array=None,  histogram_array=None, 
                 partial_pdf_array=None,  pdf_array=None, 
                 verbose=False ):
    """
    Carry out GPU computation of histogram.
    
    Args:
        device (pyopencl.Device):
        context (pyopencl.Context):
        queue (pyopencl.CommandQueue):
        cl_kernel_source (str):
        cl_kernel_fn (str):
        info (numpy.ndarray):
        sl_array (numpy.ndarray):
        histogram_array (numpy.ndarray):
        verbose (bool):  
        
    """
        
    # Prepare memory, buffers 
    n_hist_bins  = info.n_hist_bins
    n_data       = info.n_data
    n_pdf_points = info.n_pdf_points
    info.verbose = verbose
        
    if action=='histogram':
        # Compute histogram
        (histogram_array, sl_buffer, histogram_buffer) \
            = prepare_memory(context, queue, 
                             n_hist_bins=n_hist_bins, 
                             sl_array=sl_array, 
                             action=action, is_bivariate=is_bivariate,
                             verbose=verbose)    
        global_size = [n_data,1]
        buffer_list = [sl_buffer, histogram_buffer]
    elif action=='partial_pdf':
        # Compute partially (rows-only) kd-smoothed pdf
        (partial_pdf_array, histogram_buffer, partial_pdf_buffer) \
            = prepare_memory(context, queue, 
                             n_hist_bins=n_hist_bins,
                             histogram_array=histogram_array,
                             action=action, is_bivariate=is_bivariate,
                             verbose=verbose)    
        global_size = [partial_pdf_array.shape[0]*partial_pdf_array.shape[1],1]
        buffer_list = [histogram_buffer, partial_pdf_buffer]
    else:
        # Compute kd-smoothed pdf
        if is_bivariate:
            (pdf_array, partial_pdf_buffer, pdf_buffer) \
                = prepare_memory(context, queue, 
                                 n_pdf_points=n_pdf_points, 
                                 partial_pdf_array=partial_pdf_array, 
                                 action=action, is_bivariate=True,
                                 verbose=verbose)    
            global_size = [pdf_array.shape[0]*pdf_array.shape[1],1]
            buffer_list = [partial_pdf_buffer, pdf_buffer]
        else:
            (pdf_array, histogram_buffer, pdf_buffer) \
                = prepare_memory(context, queue, 
                                 n_pdf_points=n_pdf_points, 
                                 histogram_array=histogram_array,
                                 action=action, is_bivariate=False,
                                 verbose=verbose)    
            global_size = [pdf_array.shape[0],1]
            buffer_list = [histogram_buffer, pdf_buffer]
        
    local_size = None
    # Compile the CL code
    compile_options = pocl.set_compile_options(info, cl_kernel_fn, job_type='kde')
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        program = cl.Program(context, cl_kernel_source).build(options=compile_options)
    pocl.report_build_log(program, device, verbose)
    # Set the GPU kernel
    kernel = getattr(program,cl_kernel_fn)
    # Designate buffered arrays
    kernel.set_args(*buffer_list)
    kernel.set_scalar_arg_dtypes( [None]*len(buffer_list) )
    # Do the GPU compute
    event = cl.enqueue_nd_range_kernel(queue, kernel, global_size, local_size)
    
    # Fetch the data back from the GPU and finish
    if action=='histogram':
        # Compute histogram
        cl.enqueue_copy(queue, histogram_array, histogram_buffer)
        queue.finish()   
        return histogram_array
    elif action=='partial_pdf':
        # Compute partial pdf
        cl.enqueue_copy(queue, partial_pdf_array, partial_pdf_buffer)
        queue.finish()
        return partial_pdf_array
    else:
        # Compute pdf
        cl.enqueue_copy(queue, pdf_array, pdf_buffer)
        queue.finish()
        return pdf_array
    
def prepare_memory(context, queue, 
                   action='histogram', is_bivariate=False, 
                   n_hist_bins=0, n_pdf_points=0,
                   sl_array=None, histogram_array=None, 
                   partial_pdf_array=None, pdf_array=None, 
                   verbose=False ):
    """
    Create PyOpenCL buffers and np-workalike arrays to allow CPU-GPU data transfer.
    
    Args:
        context (pyopencl.Context):
        queue (pyopencl.CommandQueue):
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

    if action=='histogram':
        nx = n_hist_bins
        if is_bivariate:
            ny = n_hist_bins
        else:
            ny = 1
        sl_buffer         = cl.Buffer(context, COPY_READ_ONLY,  hostbuf=sl_array)
        histogram_array   = np.zeros((nx,ny), dtype=np.uint32)
        histogram_buffer  = cl.Buffer(context, COPY_READ_WRITE, hostbuf=histogram_array)
        return (histogram_array, sl_buffer, histogram_buffer) 
    elif action=='partial_pdf':
        nx = n_hist_bins
        ny = n_hist_bins
        histogram_buffer  = cl.Buffer(context, COPY_READ_ONLY, hostbuf=histogram_array)
        partial_pdf_array = np.zeros((nx,ny), dtype=np.float32)
        partial_pdf_buffer= cl.Buffer(context, COPY_READ_WRITE,hostbuf=partial_pdf_array)
        return (partial_pdf_array, histogram_buffer, partial_pdf_buffer) 
    else:
        nx = n_pdf_points
        if is_bivariate:
            ny = n_pdf_points
            partial_pdf_buffer=cl.Buffer(context,COPY_READ_ONLY,hostbuf=partial_pdf_array)
            pdf_array         = np.zeros((nx,ny), dtype=np.float32)
            pdf_buffer        = cl.Buffer(context, COPY_READ_WRITE, hostbuf=pdf_array)
            return (pdf_array, partial_pdf_buffer, pdf_buffer) 
        else:
            histogram_buffer = cl.Buffer(context, COPY_READ_ONLY, hostbuf=histogram_array)
            pdf_array         = np.zeros((nx,1), dtype=np.float32)
            pdf_buffer        = cl.Buffer(context, COPY_READ_WRITE, hostbuf=pdf_array)
            return (pdf_array, histogram_buffer, pdf_buffer) 
        