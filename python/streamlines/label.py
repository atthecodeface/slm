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

__all__ = ['label_confluences']

pdebug = print

def label_confluences( cl_src_path, which_cl_platform, which_cl_device, info_dict, 
                       mask_array, uv_array, dn_slt_array,
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
        dn_slt_array (numpy.ndarray):
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
        = pick_seeds(mask=mask_array, map=mapping_array, flag=is_thinchannel, pad=pad)
        
    # Prepare memory, buffers 
    array_dict = { 'seed_point': {'array': seed_point_array, 'rwf': 'RO'},
                   'mask':       {'array': mask_array,       'rwf': 'RO'}, 
                   'uv':         {'array': uv_array,         'rwf': 'RO'}, 
                   'dn_slt':     {'array': dn_slt_array,     'rwf': 'RO'}, 
                   'mapping':    {'array': mapping_array,    'rwf': 'RW'}, 
                   'count':      {'array': count_array,      'rwf': 'RW'}, 
                   'link':       {'array': link_array,       'rwf': 'RW'} }
    info_dict['n_seed_points'] = seed_point_array.shape[0]
    
    # Do integrations on the GPU
    cl_kernel_fn = 'label_confluences'
    pocl.gpu_compute(device, context, queue, cl_kernel_source,cl_kernel_fn, 
                     info_dict, array_dict, verbose)
    
    # Done
    vprint(verbose,'...done')  
    