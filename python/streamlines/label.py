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

def label_confluences( cl_state, info_dict, 
                       mask_array, uv_array, dn_slt_array,
                       mapping_array, count_array, link_array, verbose ):
        
    """
    Label channel confluences.
    
    Args:
        cl_state (obj):
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
    cl_state.kernel_source \
        = pocl.read_kernel_source(cl_state.src_path,['essentials.cl','updatetraj.cl',
                                                     'label.cl'])
            
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
    cl_state.kernel_fn = 'label_confluences'
    pocl.gpu_compute(cl_state, info_dict, array_dict, verbose)
    
    # Done
    vprint(verbose,'...done')  
    