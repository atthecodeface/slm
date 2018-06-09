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

def label_confluences( cl_state, info, data, verbose ):
    """
    Label channel confluences.
    
    Args:
        cl_state (obj):
        info (obj):
        data (obj):
        verbose (bool):
        
    """
    vprint(verbose,'Labeling confluences...')
    
    # Prepare CL essentials
    cl_state.kernel_source \
        = pocl.read_kernel_source(cl_state.src_path,['essentials.cl','updatetraj.cl',
                                                     'label.cl'])
            
    # Check all thin channel pixels
    pad            = info.pad_width
    is_thinchannel = info.is_thinchannel
    seed_point_array \
        = pick_seeds(mask=data.mask_array, map=data.mapping_array, 
                     flag=is_thinchannel, pad=pad)
        
    # Prepare memory, buffers 
    array_dict = { 'seed_point': {'array': seed_point_array,      'rwf': 'RO'},
                   'mask':       {'array': data.mask_array,       'rwf': 'RO'}, 
                   'uv':         {'array': data.uv_array,         'rwf': 'RO'}, 
                   'dn_slt':     {'array': data.dn_slt_array,     'rwf': 'RO'}, 
                   'mapping':    {'array': data.mapping_array,    'rwf': 'RW'}, 
                   'count':      {'array': data.count_array,      'rwf': 'RW'}, 
                   'link':       {'array': data.link_array,       'rwf': 'RW'} }
    info.n_seed_points = seed_point_array.shape[0]
    
    # Do integrations on the GPU
    cl_state.kernel_fn = 'label_confluences'
    pocl.gpu_compute(cl_state, info, array_dict, info.verbose)
    
    # Done
    vprint(verbose,'...done')  
    