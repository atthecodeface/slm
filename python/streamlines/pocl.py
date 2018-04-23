"""
Prep PyOpenCL kernel
"""

import pyopencl as cl
import pyopencl.tools as cltools
from os import environ
from streamlines import state
from streamlines.useful import vprint

environ['PYOPENCL_COMPILER_OUTPUT']='0'

__all__ = ['prepare_cl','choose_platform_and_device',
           'prepare_cl_context','prepare_cl_queue',
           'make_cl_dtype',
           'set_compile_options', 'report_kernel_info', 'report_device_info',
           'report_build_log']

def prepare_cl_context(cl_platform=0, cl_device=2):
    """
    Prepare PyOpenCL platform, device and context.
    
    Args:
        cl_platform (int):
        cl_device (int):
    
    Returns:
        pyopencl.Platform, pyopencl.Device, pyopencl.Context:
            PyOpenCL platform, PyOpenCL device, PyOpenCL context
    """
    cl_platform, cl_device = choose_platform_and_device(cl_platform,cl_device)
    platform = cl.get_platforms()[cl_platform]
    devices = platform.get_devices()
    device = devices[cl_device]
    context = cl.Context([device])
    return platform, device, context

def choose_platform_and_device(cl_platform='env',cl_device='env'):
    """
    Get OpenCL platform & device from environment variables if they are set.
    
    Args:
        cl_platform (int):
        cl_device (int):
    
    Returns:
        int, int:
            CL platform, CL device
    """
    if cl_platform=='env':
        try:
            cl_platform = int(environ['PYOPENCL_CTX'].split(':')[0])
        except:
            cl_platform = 0
    if cl_device=='env':
        try:
            cl_device = int(environ['PYOPENCL_CTX'].split(':')[1])
        except:
            cl_device = 2
    return cl_platform, cl_device

def prepare_cl_queue(context=None, cl_kernel_source=None, compile_options=[]):
    """
    Build PyOpenCL program and prepare command queue.
    
    Args:
        context (pyopencl.Context): GPU/OpenCL device context
        cl_kernel_source (str): OpenCL kernel code string
    
    Returns:
        pyopencl.Program, pyopencl.CommandQueue: 
            PyOpenCL program, PyOpenCL command queue
    """
#     compile_options = ['-cl-fast-relaxed-math',
#                        '-cl-single-precision-constant',
#                        '-cl-unsafe-math-optimizations',
#                        '-cl-no-signed-zeros',
#                        '-cl-finite-math-only']
    program = cl.Program(context, cl_kernel_source).build(cache_dir=False,
                                                          options=compile_options)
    queue = cl.CommandQueue(context)
    return program, queue

def prepare_cl(cl_platform=0, cl_device=2, cl_kernel_source=None, compile_options=[]):
    """
    Prepare PyOpenCL stuff.
    
    Args:
        cl_device (int):
        cl_kernel_source (str): OpenCL kernel code string
    
    Returns:
        pyopencl.Platform, pyopencl.Device, pyopencl.Context, pyopencl.Program, \
        pyopencl.CommandQueue:
            PyOpenCL platform, PyOpenCL device, PyOpenCL context, PyOpenCL program, \
            PyOpenCL command queue
    """
    platform,device,context = prepare_cl_context(cl_platform,cl_device)
    program,queue = prepare_cl_queue(context,cl_kernel_source,compile_options)
    return platform, device, context, program, queue

def make_cl_dtype(device,name,dtype):
    """
    Generate an OpenCL structure typedef codelet from a numpy structured 
    array dtype.
    
    Currently unused.
    
    Args:
        cl_device (int):
        name (str):
        dtype (numpy.dtype):
    
    Returns:
        numpy.dtype, pyopencl.dtype, str: 
            processed dtype, cl dtype, CL typedef codelet
    """
    processed_dtype, c_decl = cltools.match_dtype_to_c_struct(device, name, dtype)
    return processed_dtype, cltools.get_or_register_dtype(name, processed_dtype), c_decl

def set_compile_options(info_struct, kernel_def, downup_sign=1,
                        job_type='integration'):
    """
    Convert the info struct into a list of '-D' compiler macros.
    
    Args:
        info_struct (numpy.ndarray): container for myriad parameters controlling
            trace() and mapping() workflow and corresponding GPU/OpenCL device operation
        kernel_def (str): name of the kernel in the program source string; 
            is used by #ifdef kernel-wrapper commands in the OpenCL codes
        downup_sign (bool): flag used to indicate desired sense of streamline integration,
               with +1 for downstream and -1 for upstream ['integration' mode]
        job_type (str): switch between 'integration' (default) and 'kde'
        
    Returns:
        list:
            compile options
    """
    if job_type=='kde':
        return [
            '-D','KERNEL_{}'.format(kernel_def.upper()),
            '-D','{}_ORDER'.format(info_struct['array_order'][0]),
            '-D','KDF_BANDWIDTH={}f'.format(info_struct['kdf_bandwidth'][0]),
            '-D','KDF_IS_{}'.format(info_struct['kdf_kernel'][0].upper()),
            '-D','N_DATA={}u'.format(info_struct['n_data'][0]),
            '-D','N_HIST_BINS={}u'.format(info_struct['n_hist_bins'][0]),
            '-D','N_PDF_POINTS={}u'.format(info_struct['n_pdf_points'][0]),
            '-D','X_MIN={}f'.format(info_struct['x_min'][0]),
            '-D','X_MAX={}f'.format(info_struct['x_max'][0]),
            '-D','X_RANGE={}f'.format(info_struct['x_range'][0]),
            '-D','BIN_DX={}f'.format(info_struct['bin_dx'][0]),
            '-D','PDF_DX={}f'.format(info_struct['pdf_dx'][0]),
            '-D','KDF_WIDTH_X={}f'.format(info_struct['kdf_width_x'][0]),
            '-D','N_KDF_POINTS_X={}u'.format(info_struct['n_kdf_points_x'][0]),
            '-D','N_KDF_PART_POINTS_X={}u'.format(info_struct['n_kdf_points_x'][0]),
            '-D','Y_MIN={}f'.format(info_struct['y_min'][0]),
            '-D','Y_MAX={}f'.format(info_struct['y_max'][0]),
            '-D','Y_RANGE={}f'.format(info_struct['y_range'][0]),
            '-D','BIN_DY={}f'.format(info_struct['bin_dy'][0]),
            '-D','PDF_DY={}f'.format(info_struct['pdf_dy'][0]),
            '-D','KDF_WIDTH_Y={}f'.format(info_struct['kdf_width_y'][0]),
            '-D','N_KDF_POINTS_Y={}u'.format(info_struct['n_kdf_points_y'][0]),
            '-D','N_KDF_PART_POINTS_Y={}u'.format(info_struct['n_kdf_points_y'][0])
        ]
    else:
        return [
        '-D','KERNEL_{}'.format(kernel_def.upper()),
        '-D','{}_ORDER'.format(info_struct['array_order'][0]),
        '-D','DOWNUP_SIGN={}'.format(downup_sign),
        '-D','INTEGRATOR_STEP_FACTOR={}f'.format( 
                                            info_struct['integrator_step_factor'][0]),
        '-D','MAX_INTEGRATION_STEP_ERROR={}f'.format(
                                info_struct['max_integration_step_error'][0]),
        '-D','ADJUSTED_MAX_ERROR={}f'.format( info_struct['adjusted_max_error'][0]),
        '-D','MAX_LENGTH={}f'.format(info_struct['max_length'][0]),
        '-D','PIXEL_SIZE={}f'.format(info_struct['pixel_size'][0]),
        '-D','INTEGRATION_HALT_THRESHOLD={}f'.format(
                                info_struct['integration_halt_threshold'][0]),
        '-D','PAD_WIDTH={}u'.format(info_struct['pad_width'][0]),
        '-D','PAD_WIDTH_PP5={}f'.format(info_struct['pad_width_pp5'][0]),
        '-D','NX={}u'.format(info_struct['nx'][0]),
        '-D','NY={}u'.format(info_struct['ny'][0]),
        '-D','NXF={}f'.format(info_struct['nxf'][0]),
        '-D','NYF={}f'.format(info_struct['nyf'][0]),
        '-D','NX_PADDED={}u'.format(info_struct['nx_padded'][0]),
        '-D','NY_PADDED={}u'.format(info_struct['ny_padded'][0]),
        '-D','X_MAX={}f'.format(info_struct['x_max'][0]),
        '-D','Y_MAX={}f'.format(info_struct['y_max'][0]),
        '-D','GRID_SCALE={}f'.format(info_struct['grid_scale'][0]),
        '-D','COMBO_FACTOR={}f'.format(info_struct['combo_factor'][0]*downup_sign),
        '-D','DT_MAX={}f'.format(info_struct['dt_max'][0]),
        '-D','MAX_N_STEPS={}u'.format(info_struct['max_n_steps'][0]),
        '-D','TRAJECTORY_RESOLUTION={}u'.format(info_struct['trajectory_resolution'][0]),
        '-D','SEEDS_CHUNK_OFFSET={}u'.format(info_struct['seeds_chunk_offset'][0]),
        '-D','SUBPIXEL_SEED_POINT_DENSITY={}u'.format(
                                        info_struct['subpixel_seed_point_density'][0]),
        '-D','SUBPIXEL_SEED_HALFSPAN={}f'.format(
                                        info_struct['subpixel_seed_halfspan'][0]),
        '-D','SUBPIXEL_SEED_STEP={}f'.format(info_struct['subpixel_seed_step'][0]),
        '-D','JITTER_MAGNITUDE={}f'.format(info_struct['jitter_magnitude'][0]),
        '-D','INTERCHANNEL_MAX_N_STEPS={}u'.format(
                                             info_struct['interchannel_max_n_steps'][0]),
        '-D','SEGMENTATION_THRESHOLD={}u'.format(
                                            info_struct['segmentation_threshold'][0]),
        '-D','LEFT_FLANK_ADDITION={}u'.format(info_struct['left_flank_addition'][0]),
        '-D','IS_CHANNEL={}u'.format(info_struct['is_channel'][0]),
        '-D','IS_THINCHANNEL={}u'.format(info_struct['is_thinchannel'][0]),
        '-D','IS_INTERCHANNEL={}u'.format(info_struct['is_interchannel'][0]),
        '-D','IS_CHANNELHEAD={}u'.format(info_struct['is_channelhead'][0]),
        '-D','IS_CHANNELTAIL={}u'.format(info_struct['is_channeltail'][0]),
        '-D','IS_MAJORCONFLUENCE={}u'.format(info_struct['is_majorconfluence'][0]),
        '-D','IS_MINORCONFLUENCE={}u'.format(info_struct['is_minorconfluence'][0]),
        '-D','IS_MAJORINFLOW={}u'.format(info_struct['is_majorinflow'][0]),
        '-D','IS_MINORINFLOW={}u'.format(info_struct['is_minorinflow'][0]),
        '-D','IS_LEFTFLANK={}u'.format(info_struct['is_leftflank'][0]),
        '-D','IS_RIGHTFLANK={}u'.format(info_struct['is_rightflank'][0]),
        '-D','IS_MIDSLOPE={}u'.format(info_struct['is_midslope'][0]),
        '-D','IS_RIDGE={}u'.format(info_struct['is_ridge'][0]),
        '-D','IS_STUCK={}u'.format(info_struct['is_stuck'][0]),
        '-D','IS_LOOP={}u'.format(info_struct['is_loop'][0]),
        '-D','IS_BLOCKAGE={}u'.format(info_struct['is_blockage'][0])
        ]

def report_kernel_info(device,kernel,verbose):
    """
    Fetch and print GPU/OpenCL kernel info.
    
    Args:
        device (pyopencl.Device):
        kernel (pyopencl.Kernel):
        verbose (bool):
    """
    if verbose:
        # Report some GPU info
        print('Kernel reference count:',
              kernel.get_info(
                  cl.kernel_info.REFERENCE_COUNT))
        print('Kernel number of args:',
              kernel.get_info(
                  cl.kernel_info.NUM_ARGS))
        print('Kernel function name:',
              kernel.get_info(
                  cl.kernel_info.FUNCTION_NAME))
        print('Maximum work group size:',
              kernel.get_work_group_info(
                  cl.kernel_work_group_info.WORK_GROUP_SIZE, device))
        print('Recommended work group size:',
              kernel.get_work_group_info(
                  cl.kernel_work_group_info.PREFERRED_WORK_GROUP_SIZE_MULTIPLE, device))
        print('Local memory size:',
              kernel.get_work_group_info(
                  cl.kernel_work_group_info.LOCAL_MEM_SIZE, device), 'bytes')
        print('Private memory size:',
              kernel.get_work_group_info(
                  cl.kernel_work_group_info.PRIVATE_MEM_SIZE, device), 'bytes')    

def report_device_info(cl_platform, cl_device, platform, device, verbose):
    """
    Fetch and print GPU/OpenCL device info.
    
    Args:
        cl_platform (int):
        cl_device (int):
        platform (pyopencl.Platform):
        device (pyopencl.Device):
        verbose (bool):
    """
    if verbose:
        print('OpenCL platform #{0} = {1}'.format(cl_platform,platform))
        print('OpenCL device #{0} = {1}'.format(cl_device,device))
        n_bytes = device.get_info(cl.device_info.GLOBAL_MEM_SIZE)
        print('Global memory size: {} bytes = {}'.format(n_bytes,state.neatly(n_bytes)))
        n_bytes = device.get_info(cl.device_info.MAX_MEM_ALLOC_SIZE)
        print('Max memory alloc size: {} bytes = {}'
                            .format(n_bytes,state.neatly(n_bytes)))
        
        device_info_list = [s for s in dir(cl.device_info) if not s.startswith('__')]
        for s in device_info_list:
            try:
                print('{0} = {1}'.format(s,device.get_info(getattr(cl.device_info,s))))
            except:
                pass

def report_build_log(program, device, verbose):
    """
    Fetch and print GPU/OpenCL program build log.
    
    Args:
        program (pyopencl.Program):
        device (pyopencl.Device):
        verbose (bool):
    """
    build_log = program.get_build_info(device,cl.program_build_info.LOG)
    if len(build_log.replace(' ',''))>0:
        vprint(verbose,'\nOpenCL build log: {}'.format(build_log))