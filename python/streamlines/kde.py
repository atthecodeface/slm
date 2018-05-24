"""
Kernel density estimation.
"""

import pyopencl as cl
import pyopencl.array
import numpy as np
from scipy.stats import norm
import os
os.environ['PYTHONUNBUFFERED']='True'
import warnings

from streamlines import pocl
from streamlines.useful import vprint, pick_seeds

__all__ = ['estimate_univariate_pdf','estimate_bivariate_pdf',
           'gpu_compute_1d','prepare_memory_1d']

pdebug = print

def estimate_bivariate_pdf( distbn ):
        
    """
    Compute bivariate histogram and subsequent kernel-density smoothed pdf.
    
    Args:
        distbn (obj):
    
    Returns:
        
        
    """
    verbose = distbn.verbose
    vprint(verbose,'Estimating bivariate pdf...',end='')
    
    # Prepare CL essentials
    platform,device,context= pocl.prepare_cl_context(distbn.cl_platform,distbn.cl_device)
    queue = cl.CommandQueue(context)
    
    cl_files = ['kde.cl']
    cl_kernel_source = ''
    for cl_file in cl_files:
        with open(os.path.join(distbn.cl_src_path,cl_file), 'r') as fp:
            cl_kernel_source += fp.read()
            
    sl_array     = distbn.logxy_data
    n_data       = distbn.n_data
    bandwidth    = distbn.bandwidth
    
    x_range      = distbn.logx_range
    bin_dx       = distbn.bin_dx
    pdf_dx       = distbn.pdf_dx
    stddev_x     = np.std(sl_array[:,0])
#     pdebug('\nstd x={0:0.2} lxmin={1:0.2}  lxmax={2:0.2}=>{3:0.0f}'
#            .format(stddev_x,  distbn.logx_min,distbn.logx_max,
#                    np.exp(distbn.logx_max)))
    
    y_range      = distbn.logy_range
    bin_dy       = distbn.bin_dy
    pdf_dy       = distbn.pdf_dy
    stddev_y     = np.std(sl_array[:,1])
#     pdebug('std y={0:0.2} lymin={1:0.2}  lymax={2:0.2}' 
#            .format(stddev_y,distbn.logy_min,distbn.logy_max))

    # Set up kernel filter
    # Hacked Silverman hack
    distbn.kdf_width_x = 1.06*stddev_x*np.power(n_data,-0.2)*bandwidth
    distbn.kdf_width_y = 1.06*stddev_y*np.power(n_data,-0.2)*bandwidth
    distbn.n_kdf_part_points_x= (np.uint32(np.floor(distbn.kdf_width_x/bin_dx))//2)
    distbn.n_kdf_part_points_y= (np.uint32(np.floor(distbn.kdf_width_y/bin_dy))//2)
            
    vprint(verbose,'histogram...',end='')
    # Histogram
    cl_kernel_fn = 'histogram_bivariate'
    histogram_array \
        = gpu_compute(device, context, queue, cl_kernel_source, cl_kernel_fn, 
                      distbn, action='histogram',  is_bivariate=True, 
                      sl_array=sl_array, verbose=verbose)
        
    vprint(verbose,'kernel filtering rows..',end='')
    # PDF
    cl_kernel_fn = 'pdf_bivariate_rows'
    partial_pdf_array \
        = gpu_compute(device, context, queue, cl_kernel_source, cl_kernel_fn, 
                      distbn, action='partial_pdf',  is_bivariate=True, 
                      histogram_array=histogram_array,
                      verbose=verbose)

    vprint(verbose,'kernel filtering columns...',end='')
    cl_kernel_fn = 'pdf_bivariate_cols'
    pdf_array \
        = gpu_compute(device, context, queue, cl_kernel_source, cl_kernel_fn, 
                      distbn,  action='full_pdf',  is_bivariate=True, 
                      partial_pdf_array=partial_pdf_array,
                      verbose=verbose)
    # Done
    vprint(verbose,'done')
    return (pdf_array/(np.sum(pdf_array)*bin_dx*bin_dy), 
            histogram_array/(np.sum(histogram_array)*bin_dx*bin_dy))
    
def estimate_univariate_pdf( distbn, do_detrend=False, logx_vec=None):
        
    """
    Compute univariate histogram and subsequent kernel-density smoothed pdf.
    
    Args:
        distbn (obj):
        do_detrend (bool):
        logx_vec (numpy.ndarray):
    
    Returns:
        
        
    """
    verbose = distbn.verbose
    vprint(verbose,'Estimating univariate pdf...',end='')
    
    # Prepare CL essentials
    platform,device,context= pocl.prepare_cl_context(distbn.cl_platform,distbn.cl_device)
    queue = cl.CommandQueue(context)
    
    cl_files = ['kde.cl']
    cl_kernel_source = ''
    for cl_file in cl_files:
        with open(os.path.join(distbn.cl_src_path,cl_file), 'r') as fp:
            cl_kernel_source += fp.read()
            
    sl_array     = distbn.logx_data
    n_data       = distbn.n_data
    x_range      = distbn.logx_range
    bandwidth    = distbn.bandwidth
    bin_dx       = distbn.bin_dx
    pdf_dx       = distbn.pdf_dx
    stddev_x     = np.std(sl_array)
    
    # Set up kernel filter
    # Hacked Silverman hack
    distbn.kdf_width_x = 1.06*stddev_x*np.power(n_data,-0.2)*bandwidth
    distbn.kdf_width_y = 0.0
    distbn.n_kdf_part_points_x = np.uint32(np.floor(distbn.kdf_width_x/pdf_dx))//2
    distbn.n_kdf_part_points_y = 0
    
#     if not do_detrend:
#         pdebug('\nstd x={0:0.2} lxmin={1:0.2}  lxmax={2:0.2}'
#            .format(stddev_x,  distbn.logx_min,distbn.logx_max))
        
    vprint(verbose,'histogram...',end='')
    # Histogram
    cl_kernel_fn = 'histogram_univariate'
    histogram_array \
        = gpu_compute(device, context, queue, cl_kernel_source, cl_kernel_fn, 
                      distbn, action='histogram', sl_array=sl_array, 
                      verbose=verbose)
        
    if do_detrend:
        x = logx_vec
        pdf = histogram_array.copy().astype(np.float32)
        mean = (np.sum(x*pdf)/np.sum(pdf))
        stddev = np.sqrt(np.sum( (x-mean)**2 * pdf)/np.sum(pdf))
#         pdebug(np.exp(mean),np.exp(stddev))
        norm_pdf = norm.pdf(logx_vec,mean,stddev)
        pdf /= norm_pdf
        pdf /= np.sum(pdf)
        pdf *= np.sum(histogram_array)
        histogram_array = pdf.copy().astype(np.uint32)
        
    vprint(verbose,'kernel filtering..',end='')
    # PDF
    cl_kernel_fn = 'pdf_univariate'
    pdf_array \
        = gpu_compute(device, context, queue, cl_kernel_source, cl_kernel_fn, 
                      distbn, action='full_pdf', histogram_array=histogram_array,
                      verbose=verbose)
    # Done
    vprint(verbose,'done')
    
    return (pdf_array/(sum(pdf_array)*bin_dx),
            histogram_array/(sum(histogram_array)*bin_dx))
    
def gpu_compute( device, context, queue, cl_kernel_source, cl_kernel_fn, distbn,
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
        distbn (obj):
        sl_array (numpy.ndarray):
        histogram_array (numpy.ndarray):
        verbose (bool):  
        
    """
        
    # Prepare memory, buffers 
    n_hist_bins  = distbn.n_hist_bins
    n_data       = distbn.n_data
    n_pdf_points = distbn.n_pdf_points
        
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
    compile_options = pocl.set_compile_options(distbn, cl_kernel_fn, job_type='kde')
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
        