"""
Label downstream.
"""

import pyopencl as cl
import pyopencl.array
import numpy as np
import os
os.environ['PYTHONUNBUFFERED']='True'
import warnings

from streamlines import pocl
from streamlines.useful import vprint, pick_seeds

__all__ = ['label_confluences','gpu_compute','prepare_memory']

pdebug = print

def label_confluences( cl_src_path, which_cl_platform, which_cl_device, info_dict, 
                       mask_array, uv_array, slt_array,
                       mapping_array, count_array, link_array, verbose ):
        
    """
    Label channel confluences.
    
    Args:
        cl_src_path (str):
        which_cl_platform (int):
        which_cl_device (int):
        info_dict (numpy.ndarray):
        mask_array (numpy.ndarray):
        uv_array (numpy.ndarray):
        slt_array (numpy.ndarray):
        mapping_array (numpy.ndarray):
        count_array (numpy.ndarray):
        link_array (numpy.ndarray):
        verbose (bool):
        
    """
    vprint(verbose,'Labeling confluences...')
    
    # Prepare CL essentials
    platform, device, context= pocl.prepare_cl_context(which_cl_platform,which_cl_device)
    queue = cl.CommandQueue(context,
                            properties=cl.command_queue_properties.PROFILING_ENABLE)
    cl_files = ['essentials.cl','updatetraj.cl','label.cl']
    cl_kernel_source = ''
    for cl_file in cl_files:
        with open(os.path.join(cl_src_path,cl_file), 'r') as fp:
            cl_kernel_source += fp.read()
            
    # Check all thin channel pixels
    pad = info_dict['pad_width']
    is_thinchannel = info_dict['is_thinchannel']
    seed_point_array \
        = pick_seeds(mask=mask_array, map=mapping_array, flag=is_thinchannel, 
                     pad=pad)
    # Do integrations on the GPU
    cl_kernel_fn = 'label_confluences'
    gpu_compute(device, context, queue, cl_kernel_source,cl_kernel_fn, info_dict, 
                 seed_point_array, mask_array, uv_array, 
                 slt_array, mapping_array, count_array, link_array, verbose)
    
    # Done
    vprint(verbose,'...done')  
    
def gpu_compute(device, context, queue, cl_kernel_source,cl_kernel_fn, info_dict, 
                seed_point_array, mask_array, uv_array, 
                dn_slt_array, mapping_array, count_array, link_array, verbose):
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
        uv_array (numpy.ndarray):
        dn_slt_array (numpy.ndarray):
        mapping_array (numpy.ndarray):
        count_array (numpy.ndarray):
        link_array (numpy.ndarray):
        verbose (bool):  
        
    """
    # Prepare memory, buffers 
    # Buffer for mask, (u,v) velocity array and more 
    array_dict = { 'seed_point': {'array': seed_point_array, 'rwf': 'RO'},
                   'mask':       {'array': mask_array,       'rwf': 'RO'}, 
                   'uv':         {'array': uv_array,         'rwf': 'RO'}, 
                   'dn_slt':     {'array': dn_slt_array,     'rwf': 'RO'}, 
                   'mapping':    {'array': mapping_array,    'rwf': 'RW'}, 
                   'count':      {'array': count_array,      'rwf': 'RW'}, 
                   'link':       {'array': link_array,       'rwf': 'RW'} }
    buffer_dict = pocl.prepare_buffers(context, array_dict, verbose)    
    # Compile the CL code
    global_size = [seed_point_array.shape[0],1]
    info_dict['n_seed_points'] = global_size[0]
    compile_options = pocl.set_compile_options(info_dict, cl_kernel_fn, downup_sign=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        program = cl.Program(context, cl_kernel_source).build(options=compile_options)
    pocl.report_build_log(program, device, verbose)
    # Set the GPU kernel
    kernel = getattr(program,cl_kernel_fn)
    # Designate buffered arrays
    kernel.set_args(*list(buffer_dict.values()))
    kernel.set_scalar_arg_dtypes( [None]*len(buffer_dict) )
    
    # Specify this integration job's parameters
    n_work_items        = info_dict['n_work_items']
    local_size          = [n_work_items,1]
    chunk_size_factor   = info_dict['chunk_size_factor']
    max_time_per_kernel = info_dict['max_time_per_kernel']
    # Do the GPU compute
    vprint(verbose,
           '#### GPU/OpenCL computation: {0} work items... ####'.format(global_size[0]))
    pocl.report_kernel_info(device,kernel,verbose)
    elapsed_time \
        = pocl.adaptive_enqueue_nd_range_kernel(queue, kernel, global_size, 
                                           local_size, n_work_items,
                                           chunk_size_factor=chunk_size_factor,
                                           max_time_per_kernel=max_time_per_kernel,
                                           verbose=verbose )
    vprint(verbose,
           '#### ...elapsed time for {1} work items: {0:.3f}s ####'
           .format(elapsed_time,global_size[0]))
    queue.finish()   

    # Fetch the data back from the GPU and finish
    cl.enqueue_copy(queue, mapping_array, buffer_dict['mapping'])
    cl.enqueue_copy(queue, count_array,   buffer_dict['count'])
    cl.enqueue_copy(queue, link_array,    buffer_dict['link'])
    queue.finish()   
