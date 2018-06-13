"""
Segment downstream.
"""

import pyopencl as cl
import pyopencl.array
import numpy as np
from os import environ
environ['PYTHONUNBUFFERED']='True'
import warnings

from streamlines        import pocl
from streamlines.useful import vprint, pick_seeds

__all__ = ['hillslope_lengths']

pdebug = print

def hsl( cl_state, info, data, verbose, do_use_ridges=False ): 
    """
    Measure mean (half) hillslope lengths.
    
    Args:
        cl_state (obj):
        info (obj):
        data (obj):
        verbose (bool):
    """
    vprint(verbose,'Computing lengths...',end='')
    
    # Prepare CL essentials
    cl_state.kernel_source \
        = pocl.read_kernel_source(cl_state.src_path,['essentials.cl','updatetraj.cl',
                                                     'computestep.cl','rungekutta.cl',
                                                     'lengths.cl'])
            
    # Trace downstream from midslope pixels to thin channel pixels, 
    #   measuring streamline distance; double and scale by pixel width 
    #   to estimate hillslope length for that midslope pixel
    pad         = info.pad_width
    is_midslope = info.is_midslope
    is_ridge    = info.is_ridge
    pixel_size  = info.pixel_size
    if do_use_ridges:
        flag    = is_ridge
        vprint(verbose,'from ridges...',end='')
    else:
        flag    = is_midslope
        vprint(verbose,'from midslopes...',end='')
    seed_point_array = pick_seeds(mask=data.mask_array, map=data.mapping_array, 
                                  flag=flag, pad=pad)
#     pdebug(seed_point_array*2+np.array([2800,2800]))
    if ( seed_point_array.shape[0]!=data.traj_length_array.shape[0] ):
        raise ValueError(
            'Mismatched midslope/ridge point arrays: seed pts={0} traj len={1}'
              .format(seed_point_array.shape,data.traj_length_array.shape))
    array_dict = { 'seed_point': {'array': seed_point_array,      'rwf': 'RO'},
                   'mask':       {'array': data.mask_array,       'rwf': 'RO'}, 
                   'uv':         {'array': data.uv_array,         'rwf': 'RO'}, 
                   'mapping':    {'array': data.mapping_array,    'rwf': 'RO'}, 
                   'label':      {'array': data.label_array,      'rwf': 'RO'}, 
                   'traj_length':{'array': data.traj_length_array,'rwf': 'RW'} }
    info.n_seed_points = seed_point_array.shape[0]
    
    # Do integrations on the GPU
    cl_state.kernel_fn = 'hillslope_lengths'
    pocl.gpu_compute(cl_state, info, array_dict, info.verbose)
    
    # Scale by two if we measured only half hillslope lengths from midslope pixels
    if not do_use_ridges:
        data.traj_length_array *= 2.0
    # Done
    vprint(verbose,'...done')  
