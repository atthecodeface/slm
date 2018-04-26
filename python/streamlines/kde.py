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
    bandwidth    = info_struct['kdf_bandwidth'][0]
    
    x_range      = info_struct['x_range'][0]
    bin_dx       = info_struct['bin_dx'][0]
    pdf_dx       = info_struct['pdf_dx'][0]
    stddev_x     = np.std(sl_array[0,:])
    
    y_range      = info_struct['y_range'][0]
    bin_dy       = info_struct['bin_dy'][0]
    pdf_dy       = info_struct['pdf_dy'][0]
    stddev_y     = np.std(sl_array[1,:])
    
#     tmp_sl_array = sl_array.T.copy()
#     sl_array =tmp_sl_array
#     pdebug('bin_dy',bin_dy,'pdf_dx',pdf_dy,'ratio',pdf_dy/bin_dy)
        
    # Set up kernel filter
    # Silverman hack for now
#     kdf_width_x = 1.06*stddev_x*np.power(n_data,-0.2)*20
#     kdf_width_y = 1.06*stddev_y*np.power(n_data,-0.2)*20
    kdf_width_x = stddev_x*bandwidth*3
    kdf_width_y = stddev_y*bandwidth*3*2
    info_struct['kdf_width_x'][0] = kdf_width_x
    info_struct['n_kdf_part_points_x'][0]= 2*(np.uint32(np.floor(kdf_width_x/bin_dx))//2)
    info_struct['kdf_width_y'][0] = kdf_width_y
    info_struct['n_kdf_part_points_y'][0]= 2*(np.uint32(np.floor(kdf_width_y/bin_dy))//2)
    
#     pdebug('\n stddev_x',stddev_x)
#     pdebug('\n stddev_y',stddev_y)
#     pdebug('\n kdf_width_x',info_struct['kdf_width_x'][0])
#     pdebug('\n kdf_width_y',info_struct['kdf_width_y'][0])
#     pdebug('\n x_min',info_struct['x_min'][0])
#     pdebug('\n x_max',info_struct['x_max'][0])
#     pdebug('\n y_min',info_struct['y_min'][0])
#     pdebug('\n y_max',info_struct['y_max'][0])
#     pdebug('\n n_kdf_part_points_x',info_struct['n_kdf_part_points_x'][0])
#     pdebug('\n n_kdf_part_points_y',info_struct['n_kdf_part_points_y'][0])
#     pdebug('\n n_data',info_struct['n_data'][0])
#     pdebug('\n n_hist_bins/n_pdf_points',
#            info_struct['n_hist_bins'][0]//info_struct['n_pdf_points'][0])
            
    vprint(verbose,'histogram...',end='')
    # Histogram
    cl_kernel_fn = 'histogram_bivariate'
    histogram_array \
        = gpu_compute(device, context, queue, cl_kernel_source, cl_kernel_fn, 
                      info_struct, action='histogram',  is_bivariate=True, 
                      sl_array=sl_array, verbose=verbose)
        
    vprint(verbose,'kernel filtering rows...',end='')
    # PDF
    cl_kernel_fn = 'pdf_bivariate_rows'
    partial_pdf_array \
        = gpu_compute(device, context, queue, cl_kernel_source, cl_kernel_fn, 
                      info_struct, action='partial_pdf',  is_bivariate=True, 
                      histogram_array=histogram_array,
                      verbose=verbose)
#     return partial_pdf_array/(np.sum(partial_pdf_array)*bin_dx*bin_dy)

    vprint(verbose,'kernel filtering columns...',end='')
    cl_kernel_fn = 'pdf_bivariate_cols'
    pdf_array \
        = gpu_compute(device, context, queue, cl_kernel_source, cl_kernel_fn, 
                      info_struct,  action='full_pdf',  is_bivariate=True, 
                      partial_pdf_array=partial_pdf_array,
                      verbose=verbose)
    # Done
    vprint(verbose,'done')
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
                      info_struct, action='histogram', sl_array=sl_array, 
                      verbose=verbose)
        
    vprint(verbose,'kernel filtering...',end='')
    # PDF
    cl_kernel_fn = 'pdf_univariate'
    pdf_array \
        = gpu_compute(device, context, queue, cl_kernel_source, cl_kernel_fn, 
                      info_struct, action='full_pdf', histogram_array=histogram_array,
                      verbose=verbose)
    # Done
    vprint(verbose,'done')
    
    return pdf_array/(sum(pdf_array)*bin_dx)
    
def gpu_compute( device, context, queue, cl_kernel_source, cl_kernel_fn, info_struct,
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
        
    if action=='histogram':
        # Compute histogram
        (histogram_array, sl_buffer, histogram_buffer) \
            = prepare_memory(context, queue, order, 
                             n_hist_bins=n_hist_bins, 
                             sl_array=sl_array, 
                             action=action, is_bivariate=is_bivariate,
                             verbose=verbose)    
        global_size = [n_data,1]
        buffer_list = [sl_buffer, histogram_buffer]
    elif action=='partial_pdf':
        # Compute partially (rows-only) kd-smoothed pdf
        (partial_pdf_array, histogram_buffer, partial_pdf_buffer) \
            = prepare_memory(context, queue, order, 
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
                = prepare_memory(context, queue, order, 
                                 n_pdf_points=n_pdf_points, 
                                 partial_pdf_array=partial_pdf_array, 
                                 action=action, is_bivariate=True,
                                 verbose=verbose)    
            global_size = [pdf_array.shape[0]*pdf_array.shape[1],1]
            buffer_list = [partial_pdf_buffer, pdf_buffer]
        else:
            (pdf_array, histogram_buffer, pdf_buffer) \
                = prepare_memory(context, queue, order, 
                                 n_pdf_points=n_pdf_points, 
                                 histogram_array=histogram_array,
                                 action=action, is_bivariate=False,
                                 verbose=verbose)    
            global_size = [pdf_array.shape[0],1]
            buffer_list = [histogram_buffer, pdf_buffer]
        
    local_size = None
#     pdebug('global_size',global_size)
    # Compile the CL code
    compile_options = pocl.set_compile_options(info_struct, cl_kernel_fn, job_type='kde')
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        program = cl.Program(context, cl_kernel_source).build(options=compile_options)
    pocl.report_build_log(program, device, verbose)
    # Set the GPU kernel
#     pdebug(compile_options)
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
    
def prepare_memory(context, queue, order, 
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

    if action=='histogram':
        nx = n_hist_bins
        if is_bivariate:
            ny = n_hist_bins
        else:
            ny = 1
        sl_buffer         = cl.Buffer(context, COPY_READ_ONLY,  hostbuf=sl_array)
        histogram_array   = np.zeros((nx,ny), dtype=np.uint32,order=order)
        histogram_buffer  = cl.Buffer(context, COPY_READ_WRITE, hostbuf=histogram_array)
        return (histogram_array, sl_buffer, histogram_buffer) 
    elif action=='partial_pdf':
        nx = n_hist_bins
        ny = n_hist_bins
        histogram_buffer  = cl.Buffer(context, COPY_READ_ONLY, hostbuf=histogram_array)
        partial_pdf_array = np.zeros((nx,ny), dtype=np.float32,order=order)
        partial_pdf_buffer= cl.Buffer(context, COPY_READ_WRITE,hostbuf=partial_pdf_array)
        return (partial_pdf_array, histogram_buffer, partial_pdf_buffer) 
    else:
        nx = n_pdf_points
        if is_bivariate:
            ny = n_pdf_points
            partial_pdf_buffer=cl.Buffer(context,COPY_READ_ONLY,hostbuf=partial_pdf_array)
            pdf_array         = np.zeros((nx,ny), dtype=np.float32,order=order)
            pdf_buffer        = cl.Buffer(context, COPY_READ_WRITE, hostbuf=pdf_array)
            return (pdf_array, partial_pdf_buffer, pdf_buffer) 
        else:
            histogram_buffer = cl.Buffer(context, COPY_READ_ONLY, hostbuf=histogram_array)
            pdf_array         = np.zeros((nx,1), dtype=np.float32,order=order)
            pdf_buffer        = cl.Buffer(context, COPY_READ_WRITE, hostbuf=pdf_array)
            return (pdf_array, histogram_buffer, pdf_buffer) 
        