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

__all__ = ['hillslope_lengths','gpu_compute','prepare_memory']

pdebug = print

def hillslope_lengths( cl_src_path, which_cl_platform, which_cl_device, info_dict, 
                       mask_array, uv_array,
                       mapping_array, label_array, traj_length_array, verbose ):
        
    """
    Measure mean (half) hillslope lengths.
    
    Args:
        cl_src_path (str):
        which_cl_platform (int):
        which_cl_device   (int):
        info_dict (numpy.ndarray):
        mask_array  (numpy.ndarray):
        uv_array (numpy.ndarray):
        mapping_array (numpy.ndarray):
        label_array   (numpy.ndarray):
        traj_length_array (numpy.ndarray):
        verbose (bool):
        
    """
    vprint(verbose,'Measuring hillslope lengths...')
    
    # Prepare CL essentials
    platform, device, context= pocl.prepare_cl_context(which_cl_platform,which_cl_device)
    queue = cl.CommandQueue(context,
                            properties=cl.command_queue_properties.PROFILING_ENABLE)
    cl_files = ['essentials.cl','updatetraj.cl','computestep.cl',
                'rungekutta.cl','lengths.cl']
    cl_kernel_source = ''
    for cl_file in cl_files:
        with open(os.path.join(cl_src_path,cl_file), 'r') as fp:
            cl_kernel_source += fp.read()
            
    # Trace downstream from midslope pixels to thin channel pixels, 
    #   measuring streamline distance; double and scale by pixel width 
    #   to estimate hillslope length for that midslope pixel
    pad = info_dict['pad_width']
    is_midslope = info_dict['is_midslope']
    pixel_size = info_dict['pixel_size']
    flag = is_midslope
    seed_point_array \
        = pick_seeds(mask=mask_array, map=mapping_array, flag=flag, pad=pad)
    if ( seed_point_array.shape[0]!=traj_length_array.shape[0] ):
        print('\nMismatched midslope point arrays: ',
              seed_point_array.shape,traj_length_array.shape)
    # Do integrations on the GPU
    cl_kernel_fn = 'hillslope_lengths'
    gpu_compute(device, context, queue, cl_kernel_source,cl_kernel_fn, info_dict, 
                seed_point_array, mask_array, uv_array, 
                mapping_array, label_array, traj_length_array, verbose)
    
    # Scale by pixel size and by two because we measured only half lengths
    traj_length_array *= pixel_size*2
    # Done
    vprint(verbose,'...done')  
      
def gpu_compute(device, context, queue, cl_kernel_source,cl_kernel_fn, info_dict, 
                seed_point_array, mask_array, uv_array, 
                mapping_array, label_array, traj_length_array, verbose):
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
        mapping_array (numpy.ndarray):
        label_array (numpy.ndarray):
        traj_length_array (numpy.ndarray):
        verbose (bool):  
        
    """
    # Buffer for mask, (u,v) velocity array and more 
    array_dict = { 'seed_point': {'array': seed_point_array, 'rwf': 'RO'},
                   'mask':       {'array': mask_array,       'rwf': 'RO'}, 
                   'uv':         {'array': uv_array,         'rwf': 'RO'}, 
                   'mapping':    {'array': mapping_array,    'rwf': 'RO'}, 
                   'label':      {'array': label_array,      'rwf': 'RO'}, 
                   'traj_length':{'array': traj_length_array,'rwf': 'RW'} }
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
    cl.enqueue_copy(queue, traj_length_array, buffer_dict['traj_length'])
    queue.finish()
    