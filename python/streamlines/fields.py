"""
GPU-based streamline integration.

Provides PyOpenCL-accelerated functions to integrate streamlines using
2nd-order Runge-Kutta and (streamline tail-step only) Euler methods.
Basins of interest can be delimited by masking. 

"""

import pyopencl as cl
import pyopencl.array
import numpy as np
import pandas as pd
import os
os.environ['PYTHONUNBUFFERED']='True'
os.environ['PYOPENCL_NO_CACHE']='True'
os.environ['PYOPENCL_COMPILER_OUTPUT']='0'

from streamlines import pocl
from streamlines import state
from streamlines.useful import vprint

__all__ = ['integrate_fields','gpu_integrate','prepare_memory','compute_stats']

import warnings
pdebug = print

def integrate_fields( 
        cl_src_path, which_cl_platform, which_cl_device, info_dict, 
        seed_point_array, mask_array, u_array, v_array,  
        do_trace_downstream, do_trace_upstream, traj_stats_df, verbose
     ):
    """
    Trace each streamline from its corresponding seed point using 2nd-order Runge-Kutta 
    integration of the topographic gradient vector field.
    
    This function is a master wrapper connecting the streamlines object and its trace()
    method to the GPU/OpenCL wrapper function gpu_integrate(). As such it
    acts on a set of parameters passed as arguments here rather than by accessing
    object variables. 
    
    Workflow parameters are transferred here bundled in the Numpy structure array 
    info_dict, which is parsed as well as passed on to gpu_integrate.
    
    The tasks undertaken by this function are to:
       1) prepare the OpenCL context, device and kernel source string
       2) calculate how to split the streamline tracing into chunks
       3) invoke the GPU/OpenCL device computation
       4) post-process the streamline total length (slt) array (scale, sqrt)
          and compute streamline trajectories statistics
    
    Args:
        cl_src_path (str):
        which_cl_platform (int):
        which_cl_device (int):
        info_dict (numpy.ndarray):
        seed_point_array (numpy.ndarray):
        mask_array (numpy.ndarray):
        u_array (numpy.ndarray):
        v_array (numpy.ndarray):
        do_trace_downstream (bool):
        do_trace_upstream (bool):
        traj_stats_df (pandas.DataFrame):
        verbose (bool):
        
    Returns:
        list, numpy.ndarray, numpy.ndarray,  \
        numpy.ndarray, numpy.ndarray, numpy.ndarray:
        slc_array, slt_array, sla_array
    """
    vprint(verbose,'Integrating streamlines'+'...')

    # Prepare CL essentials
    platform, device, context= pocl.prepare_cl_context(which_cl_platform,which_cl_device)
    queue = cl.CommandQueue(context,
                            properties=cl.command_queue_properties.PROFILING_ENABLE)
    cl_files = ['rng.cl','essentials.cl',
                'writearray.cl','trajectoryfns.cl','computestep.cl',
                'integrationfns.cl','jittertrajectory.cl','integratefields.cl']
    cl_kernel_source = ''
    for cl_file in cl_files:
        with open(os.path.join(cl_src_path,cl_file), 'r') as fp:
            cl_kernel_source += fp.read()
    n_padded_seed_points = info_dict['n_padded_seed_points']

    # Memory check - not really needed 
    gpu_traj_memory_limit = (device.get_info(cl.device_info.GLOBAL_MEM_SIZE) 
                             *info_dict['gpu_memory_limit_pc'])//100
    full_traj_memory_request = (mask_array.shape[0]*mask_array.shape[1]
                                *np.dtype(np.float32).itemsize*2*3)    
    vprint(verbose,
           'GPU/OpenCL device global memory limit for streamline trajectories: {}' 
              .format(state.neatly(gpu_traj_memory_limit)))
    vprint(verbose,'GPU/OpenCL device memory required for streamline trajectories {}'
              .format(state.neatly(full_traj_memory_request)), end='')
    
    # 
    n_global = info_dict['n_padded_seed_points']
    n_work_items = info_dict['n_work_items']
    pad_length = np.uint32(np.round(n_global/n_work_items+0.5))*n_work_items-n_global
    if pad_length>0:
        padding_array = -np.ones([pad_length,2], dtype=np.float32)
        vprint(verbose,'Chunk size adjustment for {0} CL work items/group: {1}->{2}...'
             .format(n_work_items, n_global, n_global+pad_length))
    n_global += pad_length*1
    
    vprint(verbose,'Compile options:\n',pocl.set_compile_options(info_dict,
                                                                 'INTEGRATE_FIELDS'))
    # Do integrations on the GPU
    (rtn_slc_array, rtn_slt_array, rtn_sla_array) \
        = gpu_integrate(device, context, queue, cl_kernel_source,
                         info_dict, n_global,
                         seed_point_array, mask_array, u_array, v_array,  
                         verbose)
    # Streamline stats
    pixel_size = info_dict['pixel_size']
    dds =  traj_stats_df['ds']['downstream','mean']
    uds =  traj_stats_df['ds']['upstream','mean']
    # slt: sum of line lengths crossing a pixel * number of line-points per pixel
    # slt: <=> sum of line-points per pixel
    rtn_slt_array[:,:,0] = rtn_slt_array[:,:,0]*(dds/pixel_size)
    rtn_slt_array[:,:,1] = rtn_slt_array[:,:,1]*(uds/pixel_size)
    # slt:  => sum of line-points per meter
    rtn_slt_array = np.sqrt(rtn_slt_array) #*np.exp(-1/4) # *(3/4)
    # slt:  =>  sqrt(area)

    # Done
    vprint(verbose,'...done')
    return (rtn_slc_array, rtn_slt_array, rtn_sla_array)



def gpu_integrate(device, context, queue, cl_kernel_source, 
                   info_dict, n_global, 
                   seed_point_array, mask_array, u_array, v_array, verbose):
    """
    Carry out GPU/OpenCL device computations in chunks.
    This function is the basic wrapper interfacing Python with the GPU/OpenCL device.
    
    The tasks undertaken by this function are to:
        1) prepare Numpy data arrays and their corresponding PyOpenCL buffers
        2) iterate over each OpenCL computation chunk
        3) postprocess the streamline count and length arrays
       
    Chunk iteration involves:
        1) building the CL program
        2) preparing the CL kernel
        3) enqueueing kernel events (inc. passing data to CL device via buffers)
        4) reporting kernel status
        5) enqueueing data returns from CL device to CPU
        6) parsing returned streamline trajectories into master np array
        7) parsing returned streamline array (slt, slc) chunks into master np arrays

    Args:
        device (pyopencl.Device):
        context (pyopencl.Context):
        queue (pyopencl.CommandQueue):
        cl_kernel_source (str):
        info_dict (numpy.ndarray):
        chunk_size (int):
        seed_point_array (numpy.ndarray):
        mask_array (numpy.ndarray):
        u_array (numpy.ndarray):
        v_array (numpy.ndarray):
        verbose (bool):  
        
    Returns: 
        numpy.ndarray, numpy.ndarray, numpy.ndarray:
        rtn_slc_array, rtn_slt_array, rtn_sla_array
    """
        
    # Prepare memory, buffers 
    (slc_array, slt_array, 
     seed_point_buffer, uv_buffer, mask_buffer, slc_buffer, slt_buffer) \
        = prepare_memory(context, seed_point_array, mask_array, u_array,v_array, verbose)
    roi_nxy = slc_array.shape
    rtn_slc_array = np.zeros((roi_nxy[0],roi_nxy[1],2), dtype=np.uint32)
    rtn_slt_array = np.zeros((roi_nxy[0],roi_nxy[1],2), dtype=np.float32)
    rtn_sla_array = np.zeros((roi_nxy[0],roi_nxy[1],2), dtype=np.float32)
    
    # Downstream and upstream passes aka streamline integrations from
    #   chunks of seed points aka subsets of the total set
    for downup_idx, downup_sign in [[0,+1.0],[1,-1.0]]:

        # Specify this integration job's parameters
        global_size = [n_global,1]
        vprint(verbose,
               'Seed point buffer size = {}*8 bytes'.format(seed_point_buffer.size/8))
        local_size = [info_dict['n_work_items'],1]
        info_dict['downup_sign'] = downup_sign
        
        ##################################
        
        # Compile the CL code
        compile_options = pocl.set_compile_options(info_dict, 'INTEGRATE_FIELDS', 
                                                   downup_sign=downup_sign)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            program = cl.Program(context,cl_kernel_source).build(options=compile_options)
        pocl.report_build_log(program, device, verbose)
        # Set the GPU kernel
        kernel = program.integrate_fields
        
        # Designate buffered arrays
        buffer_list = [seed_point_buffer, mask_buffer, uv_buffer, slc_buffer, slt_buffer]
        kernel.set_args(*buffer_list)
        kernel.set_scalar_arg_dtypes( [None]*len(buffer_list) )
        
        # Trace the streamlines on the GPU
        event = cl.enqueue_nd_range_kernel(queue, kernel, global_size, local_size)
#         event.wait()
        queue.finish()
        
        # Calculate the time it took to execute the kernel
        elapsed = 1e-9*(event.profile.end - event.profile.start)  
        pocl.report_kernel_info(device,kernel,verbose)
        vprint(verbose,"##### Kernel lapsed time: {0:.3f} secs #####\n".format(elapsed))  
        queue.finish()   
        
        # Copy back the streamline length, distance density grid
        cl.enqueue_copy(queue, slc_array, slc_buffer)
        queue.finish()   
        cl.enqueue_copy(queue, slt_array, slt_buffer)
        queue.finish()   
                
        ##################################
                
        # Copy out the slc, slt results for this pass only
        # Count of streamlines entering per pixel width: n/meter
        rtn_slc_array[:,:,downup_idx] += slc_array.copy()
        # Average streamline length of streamlines entering each pixel: meters
        rtn_slt_array[:,:,downup_idx] += slt_array.copy().astype(np.float32)
        # Zero the GPU slt, slc arrays before using in the next pass
        slc_array.fill(0)
        slt_array.fill(0.0)
        cl.enqueue_copy(queue, slc_buffer,slc_array)
        queue.finish()   
        cl.enqueue_copy(queue, slt_buffer,slt_array)
        queue.finish()   

    # Compute average streamline lengths (sla) from total lengths (slt) and counts (slc)
    # Shorthand
    (slc,sla,slt) = (rtn_slc_array,rtn_sla_array,rtn_slt_array)
    # slc: count of lines crossing a pixel * number of line-points per pixel
    # slt: sum of line lengths crossing a pixel * number of line-points per pixel
    # sla: sum of line lengths / count of lines
    sla[slc==0] = 0.0
    sla[slc>0]  = slt[slc>0]/slc[slc>0]
    slt[slc>0]  = slt[slc>0]/info_dict['subpixel_seed_point_density']**2
    
    return (slc, slt, sla)

def prepare_memory(context, seed_point_array, mask_array, u_array, v_array, verbose):
    """
    Create Numpy array and PyOpenCL buffers to allow CPU-GPU data transfer.
    
    Args:
        context (pyopencl.Context):
        seed_point_array (numpy.ndarray):
        mask_array (numpy.ndarray):
        u_array (numpy.ndarray):
        v_array (numpy.ndarray):
        verbose (bool):
        
    Returns:
        numpy.ndarray, numpy.ndarray, \
        pyopencl.Buffer, pyopencl.Buffer, pyopencl.Buffer, \
        pyopencl.Buffer, pyopencl.Buffer:
        slc_array, slt_array, 
        seed_point_buffer, uv_buffer, mask_buffer, 
        slc_buffer, slt_buffer
    """
    # PyOpenCL array for seed points
    # Buffer for mask, (u,v) velocity array and more
    roi_nxy = mask_array.shape
#     n_padded_seed_points = seed_point_array.shape[0]
    uv_array = np.stack((u_array,v_array),axis=2).copy().astype(dtype=np.float32)
    slc_array = np.zeros((roi_nxy[0], roi_nxy[1]), dtype=np.uint32)
    slt_array = np.zeros((roi_nxy[0], roi_nxy[1]), dtype=np.uint32)

    
    # Buffers to GPU memory
    COPY_READ_ONLY  = cl.mem_flags.READ_ONLY  | cl.mem_flags.COPY_HOST_PTR
    COPY_READ_WRITE = cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR
    WRITE_ONLY      = cl.mem_flags.WRITE_ONLY
    seed_point_buffer   = cl.Buffer(context, COPY_READ_ONLY,  hostbuf=seed_point_array)
    mask_buffer         = cl.Buffer(context, COPY_READ_ONLY,  hostbuf=mask_array)
    uv_buffer           = cl.Buffer(context, COPY_READ_ONLY,  hostbuf=uv_array)
    slc_buffer          = cl.Buffer(context, COPY_READ_WRITE, hostbuf=slc_array)
    slt_buffer          = cl.Buffer(context, COPY_READ_WRITE, hostbuf=slt_array)
    vprint(verbose,'Array sizes:\n',
           'ROI-type =', mask_array.shape, '\n',
           'uv =', uv_array.shape, '\n',
           'slx-type = ',slc_array.shape)

    return (slc_array, slt_array, 
            seed_point_buffer, uv_buffer, mask_buffer, slc_buffer, slt_buffer)
