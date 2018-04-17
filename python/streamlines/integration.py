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
os.environ['PYOPENCL_COMPILER_OUTPUT']='1'

from streamlines import pocl
from streamlines import state
from streamlines.useful import vprint

__all__ = ['integrate_trajectories','gpu_integrate_trajectories','prepare_memory',
           'compute_stats']

import warnings
pdebug = print

def integrate_trajectories( 
        module_path, which_cl_platform, which_cl_device, info_struct, 
        seed_point_array, mask_array, u_array, v_array,  
        do_trace_downstream, do_trace_upstream, verbose
     ):
    """
    Trace each streamline from its corresponding seed point using 2nd-order Runge-Kutta 
    integration of the topographic gradient vector field.
    
    This function is a master wrapper connecting the streamlines object and its trace()
    method to the GPU/OpenCL wrapper function gpu_integrate_trajectories(). As such it
    acts on a set of parameters passed as arguments here rather than by accessing
    object variables. 
    
    Workflow parameters are transferred here bundled in the Numpy structure array 
    info_struct, which is parsed as well as passed on to gpu_integrate_trajectories.
    
    The tasks undertaken by this function are to:
       1) prepare the OpenCL context, device and kernel source string
       2) calculate how to split the streamline tracing into chunks
       3) invoke the GPU/OpenCL device computation
       4) post-process the streamline total length (slt) array (scale, sqrt)
          and compute streamline trajectories statistics
    
    Args:
        module_path (str):
        which_cl_platform (int):
        which_cl_device (int):
        info_struct (numpy.ndarray):
        seed_point_array (numpy.ndarray):
        mask_array (numpy.ndarray):
        u_array (numpy.ndarray):
        v_array (numpy.ndarray):
        do_trace_downstream (bool):
        do_trace_upstream (bool):
        verbose (bool):
        
    Returns:
        list, numpy.ndarray, numpy.ndarray, pandas.DataFrame, \
        numpy.ndarray, numpy.ndarray, numpy.ndarray:
        streamline_arrays_list, traj_nsteps_array, traj_length_array, traj_stats_df,
        slc_array, slt_array, sla_array
    """
    vprint(verbose,'Integrating streamlines'+'...')

    # Prepare CL essentials
    platform, device, context= pocl.prepare_cl_context(which_cl_platform,which_cl_device)
    queue = cl.CommandQueue(context,
                            properties=cl.command_queue_properties.PROFILING_ENABLE)
    cl_files = ['rng.cl','essentials.cl',
                'writearray.cl','trajectoryfns.cl','computestep.cl',
                'integrationfns.cl','trajectory.cl','integration.cl']
    cl_kernel_source = ''
    for cl_file in cl_files:
        with open(os.path.join(module_path,cl_file), 'r') as fp:
            cl_kernel_source += fp.read()
    order = info_struct['array_order'][0]
    if order=='F':
        n_seed_points = seed_point_array.shape[1]
    else:
        n_seed_points = seed_point_array.shape[0]

    # Chunkification
    gpu_traj_memory_limit = (device.get_info(cl.device_info.GLOBAL_MEM_SIZE) 
                             *info_struct['gpu_memory_limit_pc'][0])//100
    full_traj_memory_request = (n_seed_points*np.dtype(np.uint8).itemsize
                                *info_struct['max_n_steps'][0]*2)
    n_chunks_required = max(1,int(np.ceil(
                        full_traj_memory_request/gpu_traj_memory_limit)) )
    
#     pocl.report_device_info(which_cl_platform,which_cl_device, platform, device, verbose)
    
    vprint(verbose,
           'GPU/OpenCL device global memory limit for streamline trajectories: {}' 
              .format(state.neatly(gpu_traj_memory_limit)))
    vprint(verbose,'GPU/OpenCL device memory required for streamline trajectories {}'
              .format(state.neatly(full_traj_memory_request)), end='')
    vprint(verbose,' => {}'.format('no need to chunkify' if n_chunks_required==1
                    else 'need to split into {} chunks'.format(n_chunks_required) ))
    trace_do_chunks,chunk_size \
        = choose_chunks(seed_point_array,info_struct,n_chunks_required,
                        do_trace_downstream,do_trace_upstream,verbose)
    vprint(verbose,'Compile options:\n',pocl.set_compile_options(info_struct,
                                                                 'INTEGRATE_TRAJECTORY'))
    # Do integrations on the GPU
    (streamline_arrays_list, traj_nsteps_array, traj_length_array,
        rtn_slc_array, rtn_slt_array, rtn_sla_array) \
        = gpu_integrate_trajectories(device, context, queue, cl_kernel_source,
                                     info_struct, trace_do_chunks, chunk_size, 
                                     seed_point_array, mask_array, u_array, v_array,  
                                     verbose)
    # Streamline stats
    pixel_size = info_struct['pixel_size'][0]
    traj_stats_df = compute_stats(traj_length_array,traj_nsteps_array,pixel_size,verbose)
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
    return (streamline_arrays_list, traj_nsteps_array, traj_length_array, traj_stats_df,
            rtn_slc_array, rtn_slt_array, rtn_sla_array)

def choose_chunks(seed_point_array, info_struct, n_chunks_required,
                  do_trace_downstream, do_trace_upstream, verbose):
    """
    Compute lists of parameters needed to carry out GPU/OpenCL device computations
    in chunks.
    
    Args:
        seed_point_array (numpy.ndarray):
        info_struct (numpy.ndarray):
        n_chunks_required (int):
        do_trace_downstream (bool):  
        do_trace_upstream (bool):  
        verbose (bool):  
        
    Returns: 
        list, int,
            trace_do_chunks, chunk_size
    """
    order = info_struct['array_order'][0]
    if order=='F':
        n_global = seed_point_array.shape[1]
    else:
        n_global = seed_point_array.shape[0]
    n_chunks = n_chunks_required    
    chunk_size = int(np.round(n_global/n_chunks+0.5))
    chunk_list = [[chunk_idx,chunk,min(n_global,chunk+chunk_size)-chunk] 
                   for chunk_idx,chunk in enumerate(range(0,n_global,chunk_size))]
    trace_do_list = [[do_trace_downstream, 'Downstream:', 0, np.float32(+1.0)]] \
                  + [[do_trace_upstream,   'Upstream:  ', 1, np.float32(-1.0)]]
    trace_do_chunks = [td+chunk for td in trace_do_list for chunk in chunk_list]
        
    vprint(verbose,'Number of seed points = total number of kernel instances: {0:,}'
                .format(n_global))
    vprint(verbose,'Number of chunks = seed point array divisor:', n_chunks)   
    vprint(verbose,'Chunk size = number of kernel instances per chunk: {0:,}'
                .format(chunk_size))   
    return trace_do_chunks, chunk_size

def gpu_integrate_trajectories(device, context, queue, cl_kernel_source, 
                               info_struct, trace_do_chunks, chunk_size, 
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
        info_struct (numpy.ndarray):
        trace_do_chunks (list):
        chunk_size (int):
        seed_point_array (numpy.ndarray):
        mask_array (numpy.ndarray):
        u_array (numpy.ndarray):
        v_array (numpy.ndarray):
        verbose (bool):  
        
    Returns: 
        list, numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray:
        streamline_arrays_list, traj_nsteps_array, traj_length_array,
        rtn_slc_array, rtn_slt_array, rtn_sla_array
    """
        
    # Prepare memory, buffers 
    streamline_arrays_list = [[],[]]
    order = info_struct['array_order'][0]
    (traj_nsteps_array, traj_length_array, 
     chunk_trajcs_array, chunk_nsteps_array, chunk_length_array, 
     slc_array, slt_array, 
     seed_point_buffer, uv_buffer, mask_buffer, 
     chunk_trajcs_buffer, chunk_nsteps_buffer, chunk_length_buffer, 
     slc_buffer, slt_buffer) \
        = prepare_memory(context, queue, order, chunk_size, info_struct['max_n_steps'][0],
                        seed_point_array, mask_array, u_array, v_array,  verbose)
    roi_nxy = slc_array.shape
    rtn_slc_array = np.zeros((roi_nxy[0],roi_nxy[1],2), dtype=np.uint32, order=order)
    rtn_slt_array = np.zeros((roi_nxy[0],roi_nxy[1],2), dtype=np.float32, order=order)
    rtn_sla_array = np.zeros((roi_nxy[0],roi_nxy[1],2), dtype=np.float32, order=order)
    
    # Downstream and upstream passes aka streamline integrations from
    #   chunks of seed points aka subsets of the total set
    for downup_str, downup_idx, downup_sign, chunk_idx, \
        seeds_chunk_offset, n_chunk_seeds in [td[1:] for td in trace_do_chunks if td[0]]:
        vprint(verbose,'\n{0} downup={1} sgn(uv)={2:+} chunk={3} seeds: {4}+{5} => {6:}'
                   .format(downup_str, downup_idx, downup_sign, chunk_idx,
                           seeds_chunk_offset, n_chunk_seeds, 
                           seeds_chunk_offset+n_chunk_seeds))

        # Specify this integration job's parameters
        global_size = [n_chunk_seeds,1]
        local_size = None
        info_struct['downup_sign'] = downup_sign
        info_struct['seeds_chunk_offset'] = seeds_chunk_offset
        
        ##################################
        
        # Compile the CL code
        compile_options = pocl.set_compile_options(info_struct, 'INTEGRATE_TRAJECTORY', 
                                                   downup_sign=downup_sign)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            program = cl.Program(context,cl_kernel_source).build(options=compile_options)
        pocl.report_build_log(program, device, verbose)
        # Set the GPU kernel
        kernel = program.integrate_trajectory
        
        # Designate buffered arrays
        buffer_list = [seed_point_buffer, mask_buffer, uv_buffer,
                       chunk_trajcs_buffer, chunk_nsteps_buffer, chunk_length_buffer,
                       slc_buffer, slt_buffer ]
        kernel.set_args(*buffer_list)
        kernel.set_scalar_arg_dtypes( [None]*len(buffer_list) )
        
        # Trace the streamlines on the GPU
        event = cl.enqueue_nd_range_kernel(queue, kernel, global_size, local_size)
        event.wait()
        
        # Calculate the time it took to execute the kernel
        elapsed = 1e-9*(event.profile.end - event.profile.start)  
        print("\n##### Kernel lapsed time: {0:.3f} secs #####\n".format(elapsed))  
        pocl.report_kernel_info(device,kernel,verbose)
        queue.finish()   
        
        # Copy GPU-computed results back to CPU
        cl.enqueue_copy(queue, chunk_trajcs_array, chunk_trajcs_buffer)
        cl.enqueue_copy(queue, chunk_nsteps_array, chunk_nsteps_buffer)
        cl.enqueue_copy(queue, chunk_length_array, chunk_length_buffer)
        # Copy back the streamline length, distance density grid
        cl.enqueue_copy(queue, slc_array, slc_buffer)
        cl.enqueue_copy(queue, slt_array, slt_buffer)
        queue.finish()   
                
        ##################################
                
        # This is part going to be slow...
        for traj_nsteps,traj_vec in \
            zip(chunk_nsteps_array[:n_chunk_seeds], chunk_trajcs_array[:n_chunk_seeds]): 
            # Pair traj point counts with traj vectors
            #   over the span of seed points in this chunk
            streamline_arrays_list[downup_idx] += [  traj_vec[:traj_nsteps].copy()  ]
            # Add this traj vector nparray to the list of streamline nparrays
        
        # Fetch number of steps (integration points) per trajectory, and the lengths,
        #   and transfer the streamline trajectories to a compact np array of arrays
        traj_nsteps_array[seeds_chunk_offset:(seeds_chunk_offset+n_chunk_seeds),
                          downup_idx] = chunk_nsteps_array[:n_chunk_seeds].copy()
        traj_length_array[seeds_chunk_offset:(seeds_chunk_offset+n_chunk_seeds),
                          downup_idx] = chunk_length_array[:n_chunk_seeds].copy()
        # Copy out the slc, slt results for this pass only
        # Count of streamlines entering per pixel width: n/meter
        rtn_slc_array[:,:,downup_idx] += slc_array.copy()
        # Average streamline length of streamlines entering each pixel: meters
        rtn_slt_array[:,:,downup_idx] += slt_array.copy().astype(np.float32)
        # Zero the GPU slt, slc arrays before using in the next pass
        slc_array.fill(0)
        slt_array.fill(0.0)
        cl.enqueue_copy(queue, slc_buffer,slc_array)
        cl.enqueue_copy(queue, slt_buffer,slt_array)

    # Make absolutely sure we have all the data back from the GPU
    queue.finish()   
    # Compute average streamline lengths (sla) from total lengths (slt) and counts (slc)
    # Shorthand
    (slc,sla,slt) = (rtn_slc_array,rtn_sla_array,rtn_slt_array)
    # slc: count of lines crossing a pixel * number of line-points per pixel
    # slt: sum of line lengths crossing a pixel * number of line-points per pixel
    # sla: sum of line lengths / count of lines
    sla[slc==0] = 0.0
    sla[slc>0]  = slt[slc>0]/slc[slc>0]
    slt[slc>0]  = slt[slc>0]/info_struct['subpixel_seed_point_density']**2
    
    vprint(verbose,'Building streamlines compressed array')
    for downup_idx in [0,1]:
        streamline_arrays_list[downup_idx] = np.array(streamline_arrays_list[downup_idx])
    vprint(verbose,'Streamlines actual array allocation:  size={}'.format(state.neatly(
        np.sum(traj_nsteps_array[:,:])*np.dtype(chunk_trajcs_array.dtype).itemsize)))
   
    return (streamline_arrays_list, traj_nsteps_array, traj_length_array,
            slc, slt, sla)

def prepare_memory(context, queue, order, chunk_size, max_traj_length, 
                   seed_point_array, mask_array, u_array, v_array,  verbose):
    """
    Create Numpy array and PyOpenCL buffers to allow CPU-GPU data transfer.
    
    Args:
        context (pyopencl.Context):
        queue (pyopencl.CommandQueue):
        order (str):
        chunk_size (int):
        max_traj_length (int):
        seed_point_array (numpy.ndarray):
        mask_array (numpy.ndarray):
        u_array (numpy.ndarray):
        v_array (numpy.ndarray):
        verbose (bool):
        
    Returns:
        numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray,\
        numpy.ndarray, numpy.ndarray, pyopencl.Buffer, pyopencl.Buffer, pyopencl.Buffer,\
        pyopencl.Buffer, pyopencl.Buffer, pyopencl.Buffer, \
        pyopencl.Buffer, pyopencl.Buffer:
        traj_nsteps_array, traj_length_array, 
        chunk_trajcs_array, chunk_nsteps_array, chunk_length_array,
        slc_array, slt_array, seed_point_buffer, uv_buffer, mask_buffer,
        chunk_trajcs_buffer, chunk_nsteps_buffer, chunk_length_buffer,
        slc_buffer, slt_buffer
    """
    # PyOpenCL array for seed points
    # Buffer for mask, (u,v) velocity array and more
    roi_nxy = mask_array.shape
    if order=='F':
        n_seed_points = seed_point_array.shape[1]
        uv_array = np.stack((u_array,v_array)).copy().astype(dtype=np.float32,
                                                             order=order)
    else:
        n_seed_points = seed_point_array.shape[0]
        uv_array = np.stack((u_array,v_array),axis=2).copy().astype(dtype=np.float32,
                                                                    order=order)
    slc_array = np.zeros((roi_nxy[0], roi_nxy[1]), dtype=np.uint32, order=order)
    slt_array = np.zeros((roi_nxy[0], roi_nxy[1]), dtype=np.uint32, order=order)
    traj_nsteps_array  = np.zeros([n_seed_points,2], dtype=np.uint16)
    traj_length_array  = np.zeros([n_seed_points,2], dtype=np.float32)

    # Chunk-sized temporary arrays
    # Use "bag o' bytes" buffer for huge trajectories array. Write (by GPU) only.
    chunk_trajcs_array = np.zeros([chunk_size,max_traj_length,2], dtype=np.int8)
    chunk_nsteps_array = np.zeros([chunk_size], dtype=traj_nsteps_array.dtype)
    chunk_length_array = np.zeros([chunk_size], dtype=traj_length_array.dtype)
    
    # Buffers to GPU memory
    COPY_READ_ONLY  = cl.mem_flags.READ_ONLY  | cl.mem_flags.COPY_HOST_PTR
    COPY_READ_WRITE = cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR
    WRITE_ONLY      = cl.mem_flags.WRITE_ONLY
    seed_point_buffer   = cl.Buffer(context, COPY_READ_ONLY,  hostbuf=seed_point_array)
    mask_buffer         = cl.Buffer(context, COPY_READ_ONLY,  hostbuf=mask_array)
    uv_buffer           = cl.Buffer(context, COPY_READ_ONLY,  hostbuf=uv_array)
    slc_buffer          = cl.Buffer(context, COPY_READ_WRITE, hostbuf=slc_array)
    slt_buffer          = cl.Buffer(context, COPY_READ_WRITE, hostbuf=slt_array)
    chunk_nsteps_buffer = cl.Buffer(context, WRITE_ONLY, chunk_nsteps_array.nbytes )
    chunk_length_buffer = cl.Buffer(context, WRITE_ONLY, chunk_length_array.nbytes )
    chunk_trajcs_buffer = cl.Buffer(context, WRITE_ONLY, chunk_trajcs_array.nbytes )     
    vprint(verbose,'Array sizes:\n',
           'ROI-type =', mask_array.shape, '\n',
           'uv =', uv_array.shape, '\n',
           'slx-type = ',slc_array.shape)
    vprint(verbose,'Streamlines virtual array allocation:  ',
                 '   dims={0}'.format((n_seed_points,
                                       chunk_trajcs_array.shape[1],2)), 
                 '  size={}'.format(state.neatly(
                     n_seed_points*chunk_trajcs_array.shape[1]*2
                        *np.dtype(chunk_trajcs_array.dtype).itemsize)))
    vprint(verbose,'Streamlines array allocation per chunk:',
                     '   dims={0}'.format(chunk_trajcs_array.shape), 
                     '  size={}'.format(state.neatly(
      chunk_trajcs_array.size*np.dtype(chunk_trajcs_array.dtype).itemsize)))

    return (traj_nsteps_array, traj_length_array, 
            chunk_trajcs_array, chunk_nsteps_array, chunk_length_array,
            slc_array, slt_array, 
            seed_point_buffer, uv_buffer, mask_buffer,
            chunk_trajcs_buffer, chunk_nsteps_buffer, chunk_length_buffer,
            slc_buffer, slt_buffer)

def compute_stats(traj_length_array, traj_nsteps_array, pixel_size, verbose):
    """
    Compute streamline integration point spacing and trajectory length statistics 
    (min, mean, max) for the sets of both downstream and upstream trajectories.
    Return them as a small Pandas dataframe table.
    
    Args:
        traj_length_array (numpy.ndarray):
        traj_nsteps_array (numpy.ndarray):
        pixel_size (float):
        verbose (bool):
        
    Returns:
        pandas.DataFrame:  lnds_stats_df
    """
    vprint(verbose,'Computing streamlines statistics')
    lnds_stats = []
    for downup_idx in [0,1]:
        lnds = np.array( [ [ln[0],ln[1],ln[0]/ln[1]] 
                            for ln in (np.stack(
                                 (traj_length_array[:,downup_idx]*pixel_size, 
                                            traj_nsteps_array[:,downup_idx])   ).T) ] )
        lnds_stats += [np.min(lnds,axis=0), np.mean(lnds,axis=0), np.max(lnds,axis=0)]
    lnds_stats_array = np.array(lnds_stats,dtype=np.float32)
    lnds_indexes = [np.array(['downstream', 'downstream', 'downstream', 
                              'upstream', 'upstream', 'upstream']),
                         np.array(['min','mean','max','min','mean','max'])]
    lnds_stats_df = pd.DataFrame(data=lnds_stats_array, 
                                 columns=['l','n','ds'],
                                 index=lnds_indexes)
    vprint(verbose,lnds_stats_df.T)
    return lnds_stats_df
