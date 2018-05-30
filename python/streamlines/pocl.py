"""
Prep PyOpenCL kernel
"""

import pyopencl       as cl
import pyopencl.tools as cltools
import numpy as np
import os
from streamlines        import state
from streamlines.useful import vprint
import warnings

os.environ['PYOPENCL_COMPILER_OUTPUT']='0'

__all__ = ['Initialize_cl',
           'prepare_cl_context','choose_platform_and_device',
           'prepare_cl_queue', 'prepare_cl',
           'make_cl_dtype',
           'set_compile_options', 
           'report_kernel_info', 'report_device_info', 'report_build_log',
           'adaptive_enqueue_nd_range_kernel',
           'prepare_buffers', 'gpu_compute']

pdebug = print

class Initialize_cl():
    def __init__(self, cl_src_path, which_cl_platform, which_cl_device ):
        # Prepare CL essentials
        self.platform, self.device, self.context \
            = prepare_cl_context(which_cl_platform,which_cl_device)
        self.queue = cl.CommandQueue(self.context,
                                properties=cl.command_queue_properties.PROFILING_ENABLE)
        self.src_path      = cl_src_path
        self.kernel_source = None
        self.kernel_fn     = ''

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

def prepare_cl_queue(context=None, kernel_source=None, compile_options=[]):
    """
    Build PyOpenCL program and prepare command queue.
    
    Args:
        context (pyopencl.Context): GPU/OpenCL device context
        kernel_source (str): OpenCL kernel code string
    
    Returns:
        pyopencl.Program, pyopencl.CommandQueue: 
            PyOpenCL program, PyOpenCL command queue
    """
#     compile_options = ['-cl-fast-relaxed-math',
#                        '-cl-single-precision-constant',
#                        '-cl-unsafe-math-optimizations',
#                        '-cl-no-signed-zeros',
#                        '-cl-finite-math-only']
    program = cl.Program(context, kernel_source).build(cache_dir=False,
                                                          options=compile_options)
    queue = cl.CommandQueue(context)
    return program, queue

def prepare_cl(cl_platform=0, cl_device=2, kernel_source=None, compile_options=[]):
    """
    Prepare PyOpenCL stuff.
    
    Args:
        cl_device (int):
        kernel_source (str): OpenCL kernel code string
    
    Returns:
        pyopencl.Platform, pyopencl.Device, pyopencl.Context, pyopencl.Program, \
        pyopencl.CommandQueue:
            PyOpenCL platform, PyOpenCL device, PyOpenCL context, PyOpenCL program, \
            PyOpenCL command queue
    """
    platform,device,context = prepare_cl_context(cl_platform,cl_device)
    program,queue = prepare_cl_queue(context,kernel_source,compile_options)
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

def set_compile_options(info, kernel_def, downup_sign=1,
                        job_type='integration'):
    """
    Convert info obj data into a list of '-D' compiler macros.
    
    Args:
        info (obj): container for myriad parameters controlling
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
        rtn_dict = {
            'bandwidth' :            'f',
            'n_data' :               'u',
            'n_hist_bins' :          'u',
            'n_pdf_points' :         'u',
            'logx_min' :             'f',
            'logx_max' :             'f',
            'logx_range' :           'f',
            'bin_dx' :               'f',
            'pdf_dx' :               'f',
            'kdf_width_x' :          'f',
            'n_kdf_part_points_x' :  'u',
            'logy_min' :             'f',
            'logy_max' :             'f',
            'logy_range' :           'f',
            'bin_dy' :               'f',
            'pdf_dy' :               'f',
            'kdf_width_y' :          'f',
            'n_kdf_part_points_y' :  'u'
        }
        rtn_list =  ['-D','KERNEL_{}'.format(kernel_def.upper())]
        rtn_list += ['-D','KDF_IS_{}'.format(info.kernel.upper())]
        for item in rtn_dict.items():
            value = getattr(info,item[0])
            list_item = ['-D','{0}={1}{2}'.format(item[0].upper(),value,item[1])]
            rtn_list += list_item

    else:
        rtn_dict = {
            'n_seed_points' :               ('u',''),
            'downup_sign' :                 ('u', np.int8(downup_sign)),
            'integrator_step_factor' :      ('f',''),
            'max_integration_step_error' :  ('f',''),
            'adjusted_max_error' :          ('f',''),
            'integration_halt_threshold' :  ('f',''),
            'max_length' :                  ('f',''),
            'pixel_size' :                  ('f',''),
            'pad_width' :                   ('u',''),
            'pad_width_pp5' :               ('f',''),
            'nx' :                          ('u',''),
            'ny' :                          ('u',''),
            'nxf' :                         ('f',''),
            'nyf' :                         ('f',''),
            'nx_padded' :                   ('u',''),
            'ny_padded' :                   ('u',''),
            'nxy_padded' :                  ('u',''),
            'x_max' :                       ('f',''),
            'y_max' :                       ('f',''),
            'grid_scale' :                  ('f',''),
            'combo_factor' :                ('f', info.combo_factor*downup_sign),
            'dt_max' :                      ('f',''),
            'max_n_steps' :                 ('u',''),
            'trajectory_resolution' :       ('u',''),
            'seeds_chunk_offset' :          ('u',''),
            'subpixel_seed_point_density' : ('u',''),
            'subpixel_seed_halfspan' :      ('f',''),
            'subpixel_seed_step' :          ('f',''),
            'jitter_magnitude' :            ('f',''),
            'interchannel_max_n_steps' :    ('u',''),
            'left_flank_addition' :         ('u',''),
            'is_channel' :                  ('u',''),
            'is_thinchannel' :              ('u',''),
            'is_interchannel' :             ('u',''),
            'is_channelhead' :              ('u',''),
            'is_channeltail' :              ('u',''),
            'is_majorconfluence' :          ('u',''),
            'is_minorconfluence' :          ('u',''),
            'is_majorinflow' :              ('u',''),
            'is_minorinflow' :              ('u',''),
            'is_leftflank' :                ('u',''),
            'is_rightflank' :               ('u',''),
            'is_midslope' :                 ('u',''),
            'is_ridge' :                    ('u',''),
            'was_channelhead' :             ('u',''),
            'is_subsegmenthead' :           ('u',''),
            'is_loop' :                     ('u',''),
            'is_blockage' :                 ('u',''),
            'do_measure_hsl_from_ridges' :  ('flag',''),
            'debug' :                       ('flag',''),
            'verbose' :                     ('flag','')
        }
        if info.segmentation_threshold>0:
            rtn_dict.update({'segmentation_threshold' : ('u','')})            

        rtn_list = ['-D','KERNEL_{}'.format(kernel_def.upper())]
        for item in rtn_dict.items():
            if item[1][0]=='flag':
                if getattr(info,item[0]):
                    list_item = ['-D', '{}'.format(item[0].upper())]
            elif item[1][1]!='':
                list_item = ['-D', 
                             '{0}={1}{2}'.format(item[0].upper(),item[1][1],item[1][0])]
            else:
                value = getattr(info,item[0])
                list_item = ['-D', 
                             '{0}={1}{2}'.format(item[0].upper(),value,item[1][0])]
            rtn_list += list_item

    return rtn_list

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
        
def adaptive_enqueue_nd_range_kernel(queue, kernel, global_size, local_size, 
                                     n_work_items, chunk_size_factor=10, 
                                     max_time_per_kernel=4.0, verbose=True):
    work_size  = n_work_items*int(np.ceil(global_size[0]/n_work_items))
    work_left  = work_size
    chunk_size = min(n_work_items*chunk_size_factor,work_left)
    offset = 0
    cumulative_time = 0.0
    time_per_item   = 0.0
    while work_left>0:
#         event = cl.enqueue_nd_range_kernel(queue, kernel, [chunk_size+offset,1], 
        event = cl.enqueue_nd_range_kernel(queue, kernel, [chunk_size,1], 
                                           local_size,global_work_offset=[offset,0])
        progress = 100.0*(min(work_size,(offset+chunk_size))/work_size)
        vprint(verbose,
               '{0:.2f}%: enqueued {1}/{2} workitems'
                    .format(progress,chunk_size,work_size),
                'in [{0}-{1}]'.format(offset,offset+chunk_size-1),
                'estimated t={0:.3f}s...'.format(time_per_item*chunk_size),
                end='')
        offset    += chunk_size
        work_left -= chunk_size
        event.wait()
        try:
            elapsed_time = 1e-9*(event.profile.end-event.profile.start)
            vprint(verbose, '...actual t={0:.3f}s'.format(elapsed_time))
        except:
            vprint(verbose, '...profiling failed')
            continue
        cumulative_time += elapsed_time
        time_per_item    = elapsed_time/chunk_size
        chunk_size = n_work_items*(min(
                            int(max_time_per_kernel/(time_per_item*n_work_items)),
                            int(np.ceil(work_left/n_work_items)) ))
        queue.finish()   
    return cumulative_time

def read_kernel_source(cl_src_path, cl_files):
    kernel_source = ''
    for cl_file in cl_files:
        with open(os.path.join(cl_src_path,cl_file), 'r') as fp:
            kernel_source += fp.read()
    return kernel_source

def prepare_buffers(context, array_dict, verbose):
    """
    Create PyOpenCL buffers and np-workalike arrays to allow CPU-GPU data transfer.
    
    Args:
        context (pyopencl.Context):
        array_dict (dict):
        verbose (bool):
        
    Returns:
        dict: buffer_dict
    """
    # Buffers to GPU memory
    COPY_READ_ONLY  = cl.mem_flags.READ_ONLY  | cl.mem_flags.COPY_HOST_PTR
    COPY_READ_WRITE = cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR
    WRITE_ONLY      = cl.mem_flags.WRITE_ONLY
    # The following could be packed into a list comprehension but would be
    #   rather harder to read in that form
    buffer_dict = {}
    for array_info in array_dict.items():
        if 'R' in array_info[1]['rwf']:
            if array_info[1]['rwf']=='RO':
                flags = COPY_READ_ONLY
            elif array_info[1]['rwf']=='RW':
                flags = COPY_READ_WRITE 
            buffer_dict.update({
                array_info[0]: cl.Buffer(context, flags, hostbuf=array_info[1]['array'])
            })
        elif array_info[1]['rwf']=='WO':
            flags = WRITE_ONLY
            buffer_dict.update({
                array_info[0]: cl.Buffer(context, flags, array_info[1]['array'].nbytes)
            })
        else:
            pass
    return buffer_dict

def gpu_compute(cl_state, info, array_dict, verbose):
    """
    Carry out GPU computation.
    
    Args:
        cl_state (obj):
        info (dict):
        array_dict (dict):
        verbose (bool):  
        
    """
    # Shorthand
    device        = cl_state.device
    context       = cl_state.context
    queue         = cl_state.queue
    kernel_source = cl_state.kernel_source
    kernel_fn     = cl_state.kernel_fn
    
    # Prepare memory, buffers 
    buffer_dict = prepare_buffers(context, array_dict, verbose)    

    # Specify this integration job's parameters
    global_size         = [info.n_seed_points,1]
    local_size          = [info.n_work_items,1]

    # Compile the CL code
    compile_options = set_compile_options(info, kernel_fn, downup_sign=1)
#     vprint(verbose,'Compile options:\n', compile_options)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        program = cl.Program(context, kernel_source).build(options=compile_options)
    report_build_log(program, device, verbose)
    # Set the GPU kernel
    kernel = getattr(program, kernel_fn)
    # Designate buffered arrays
    kernel.set_args(*list(buffer_dict.values()))
    kernel.set_scalar_arg_dtypes( [None]*len(buffer_dict) )
    
    # Do the GPU compute
    vprint(verbose,
           '#### GPU/OpenCL computation: {0} work items... ####'.format(global_size[0]))
    report_kernel_info(device, kernel, verbose)
    elapsed_time \
        = adaptive_enqueue_nd_range_kernel( queue, kernel, global_size, 
                                            local_size, info.n_work_items,
                                            chunk_size_factor=info.chunk_size_factor,
                                            max_time_per_kernel=info.max_time_per_kernel,
                                            verbose=verbose )
    vprint(verbose,
           '#### ...elapsed time for {1} work items: {0:.3f}s ####'
           .format(elapsed_time,global_size[0]))
    queue.finish()   
    
    # Fetch the data back from the GPU and finish
    for array_info in array_dict.items():
        if 'W' in array_info[1]['rwf']:
            cl.enqueue_copy(queue, array_info[1]['array'], buffer_dict[array_info[0]])
            queue.finish()   
