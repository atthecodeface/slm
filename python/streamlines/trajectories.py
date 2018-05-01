"""
GPU-based streamline integration.

Provides PyOpenCL-accelerated functions to integrate streamlines using
2nd-order Runge-Kutta and (streamline tail-step only) Euler methods.
Basins of interest can be delimited by masking. 

"""

import pyopencl as cl
import pyopencl.array
import numpy as np
import os
os.environ['PYTHONUNBUFFERED']='True'
os.environ['PYOPENCL_NO_CACHE']='True'
os.environ['PYOPENCL_COMPILER_OUTPUT']='0'

from streamlines import pocl
from streamlines import state
from streamlines.useful import vprint, create_seeds, compute_stats

__all__ = ['integrate_trajectories','gpu_integrate','prepare_memory']

import warnings
pdebug = print

def integrate_trajectories(cl_src_path, which_cl_platform, which_cl_device, info_dict, 
                           mask_array, u_array, v_array, mapping_array, 
                           do_trace_downstream, do_trace_upstream, verbose):
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
        mask_array (numpy.ndarray):
        u_array (numpy.ndarray):
        v_array (numpy.ndarray):
        mapping_array (numpy.ndarray):
        do_trace_downstream (bool):
        do_trace_upstream (bool):
        verbose (bool):
        
    Returns:
        list, numpy.ndarray, numpy.ndarray, pandas.DataFrame:
        streamline_arrays_list, traj_nsteps_array, traj_length_array, traj_stats_df
    """
    vprint(verbose,'Integrating trajectories'+'...')

    # Prepare CL essentials
    platform, device, context= pocl.prepare_cl_context(which_cl_platform,which_cl_device)
    queue = cl.CommandQueue(context,
                            properties=cl.command_queue_properties.PROFILING_ENABLE)
    cl_files = ['rng.cl','essentials.cl',
                'writearray.cl','trajectoryfns.cl','computestep.cl',
                'integrationfns.cl','trajectory.cl','integratetraj.cl']
    cl_kernel_source = ''
    for cl_file in cl_files:
        with open(os.path.join(cl_src_path,cl_file), 'r') as fp:
            cl_kernel_source += fp.read()
            
    # Seed point selection, padding and shuffling
    n_trajectory_seed_points = info_dict['n_trajectory_seed_points']
    pad_width                = info_dict['pad_width']
    n_work_items             = info_dict['n_work_items']
    do_shuffle               = info_dict['do_shuffle']
    shuffle_rng_seed         = info_dict['shuffle_rng_seed']
    seed_point_array, n_seed_points, n_padded_seed_points \
        = create_seeds(mask_array, pad_width, n_work_items, 
                       n_seed_points=n_trajectory_seed_points, 
                       do_shuffle=do_shuffle, rng_seed=shuffle_rng_seed,
                       verbose=verbose)
    info_dict['n_seed_points']        = n_seed_points
    info_dict['n_padded_seed_points'] = n_padded_seed_points
    
    # Mapping flag array - not likely already defined, but just in case...
    if mapping_array is None:
        mapping_array = np.zeros_like(mask_array, dtype=np.uint32)

    # Chunkification
    gpu_traj_memory_limit = (device.get_info(cl.device_info.GLOBAL_MEM_SIZE) 
                             *info_dict['gpu_memory_limit_pc'])//100
    full_traj_memory_request = (n_seed_points*np.dtype(np.uint8).itemsize
                                *info_dict['max_n_steps']*2)
    n_chunks_required = max(1,int(np.ceil(
                        full_traj_memory_request/gpu_traj_memory_limit)) )
    
#     pocl.report_device_info(which_cl_platform,which_cl_device, platform, device, verbose)
    
    vprint(verbose,
           'GPU/OpenCL device global memory limit for streamline trajectories: {}' 
              .format(state.neatly(gpu_traj_memory_limit)))
    vprint(verbose,'GPU/OpenCL device memory required for streamline trajectories: {}'
              .format(state.neatly(full_traj_memory_request)), end='')
    vprint(verbose,' => {}'.format('no need to chunkify' if n_chunks_required==1
                    else 'need to split into {} chunks'.format(n_chunks_required) ))
    trace_do_chunks,chunk_size \
        = choose_chunks(seed_point_array,info_dict,n_chunks_required,
                        do_trace_downstream,do_trace_upstream,verbose)
    vprint(verbose,'Compile options:\n',pocl.set_compile_options(info_dict,
                                                                 'INTEGRATE_TRAJECTORY'))
    # Do integrations on the GPU
    (streamline_arrays_list, traj_nsteps_array, traj_length_array) \
        = gpu_integrate(device, context, queue, cl_kernel_source,
                        info_dict, trace_do_chunks, chunk_size, 
                        seed_point_array, mask_array, 
                        u_array, v_array, mapping_array, verbose)
    # Streamline stats
    pixel_size = info_dict['pixel_size']
    traj_stats_df = compute_stats(traj_length_array,traj_nsteps_array,pixel_size,verbose)
    dds =  traj_stats_df['ds']['downstream','mean']
    uds =  traj_stats_df['ds']['upstream','mean']

    # Done
    vprint(verbose,'...done')
    return (seed_point_array, streamline_arrays_list, 
            traj_nsteps_array, traj_length_array, traj_stats_df)

def choose_chunks(seed_point_array, info_dict, n_chunks,
                  do_trace_downstream, do_trace_upstream, verbose):
    """
    Compute lists of parameters needed to carry out GPU/OpenCL device computations
    in chunks.
    
    Args:
        seed_point_array (numpy.ndarray):
        info_dict (numpy.ndarray):
        n_chunks_required (int):
        do_trace_downstream (bool):  
        do_trace_upstream (bool):  
        verbose (bool):  
        
    Returns: 
        list, int:
            trace_do_chunks, chunk_size
    """
    n_seed_points        = info_dict['n_seed_points']
    n_padded_seed_points = info_dict['n_padded_seed_points']
    n_work_items         = info_dict['n_work_items']
    
    chunk_size = int(np.round(n_padded_seed_points/n_chunks))
    n_global = n_padded_seed_points
    chunk_list = [[chunk_idx,chunk,min(n_seed_points,chunk+chunk_size)-chunk,chunk_size] 
                   for chunk_idx,chunk in enumerate(range(0,n_global,chunk_size))]            

    trace_do_list = [[do_trace_downstream, 'Downstream:', 0, np.float32(+1.0)]] \
                  + [[do_trace_upstream,   'Upstream:  ', 1, np.float32(-1.0)]]
    trace_do_chunks = [td+chunk for td in trace_do_list for chunk in chunk_list]
        
    vprint(verbose,'Total number of kernel instances: {0:,}'
                .format(n_global))
    vprint(verbose,'Number of chunks = seed point array divisor:', n_chunks)   
    vprint(verbose,'Chunk size = number of kernel instances per chunk: {0:,}'
                .format(chunk_size))   
    return trace_do_chunks, chunk_size

def gpu_integrate(device, context, queue, cl_kernel_source, 
                  info_dict, trace_do_chunks, chunk_size, 
                  seed_point_array, mask_array,
                  u_array, v_array, mapping_array, verbose):
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
        trace_do_chunks (list):
        chunk_size (int):
        seed_point_array (numpy.ndarray):
        mask_array (numpy.ndarray):
        u_array (numpy.ndarray):
        v_array (numpy.ndarray):
        mapping_array (numpy.ndarray):
        verbose (bool):  
        
    Returns: 
        list, numpy.ndarray, numpy.ndarray:
        streamline_arrays_list, traj_nsteps_array, traj_length_array
    """
        
    # Prepare memory, buffers 
    streamline_arrays_list = [[],[]]
    (traj_nsteps_array, traj_length_array, 
     chunk_trajcs_array, chunk_nsteps_array, chunk_length_array, 
     seed_point_buffer, uv_buffer, mask_buffer, mapping_buffer, 
     chunk_trajcs_buffer, chunk_nsteps_buffer, chunk_length_buffer) \
        = prepare_memory(context, info_dict['n_seed_points'],
                         chunk_size, info_dict['max_n_steps'],
                         seed_point_array, mask_array, u_array, v_array, 
                         mapping_array, verbose)
    
    # Downstream and upstream passes aka streamline integrations from
    #   chunks of seed points aka subsets of the total set
    for downup_str, downup_idx, downup_sign, chunk_idx, \
        seeds_chunk_offset, n_chunk_seeds, n_chunk_ki in [td[1:] \
            for td in trace_do_chunks if td[0]]:
        vprint(verbose,'\n{0} downup={1} sgn(uv)={2:+} chunk={3} seeds: {4}+{5} => {6:}'
                   .format(downup_str, downup_idx, downup_sign, chunk_idx,
                           seeds_chunk_offset, n_chunk_seeds, 
                           seeds_chunk_offset+n_chunk_seeds))

        # Specify this integration job's parameters
        global_size = [n_chunk_ki,1]
        vprint(verbose,
               'Seed point buffer size = {}*8 bytes'.format(seed_point_buffer.size/8))
        local_size = [info_dict['n_work_items'],1]
        info_dict['downup_sign'] = downup_sign
        info_dict['seeds_chunk_offset'] = seeds_chunk_offset
        
        ##################################
        
        # Compile the CL code
        compile_options = pocl.set_compile_options(info_dict, 'INTEGRATE_TRAJECTORY', 
                                                   downup_sign=downup_sign)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            program = cl.Program(context,cl_kernel_source).build(options=compile_options)
        pocl.report_build_log(program, device, verbose)
        # Set the GPU kernel
        kernel = program.integrate_trajectory
        
        # Designate buffered arrays
        buffer_list = [seed_point_buffer, mask_buffer, uv_buffer, mapping_buffer, 
                       chunk_trajcs_buffer, chunk_nsteps_buffer, chunk_length_buffer]
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
        
        # Copy GPU-computed results back to CPU
        cl.enqueue_copy(queue, chunk_trajcs_array, chunk_trajcs_buffer)
        queue.finish()   
        cl.enqueue_copy(queue, chunk_nsteps_array, chunk_nsteps_buffer)
        queue.finish()   
        cl.enqueue_copy(queue, chunk_length_array, chunk_length_buffer)
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
    
    vprint(verbose,'Building streamlines compressed array')
    for downup_idx in [0,1]:
        streamline_arrays_list[downup_idx] = np.array(streamline_arrays_list[downup_idx])
    vprint(verbose,'Streamlines actual array allocation:  size={}'.format(state.neatly(
        np.sum(traj_nsteps_array[:,:])*np.dtype(chunk_trajcs_array.dtype).itemsize)))
   
    n = info_dict['n_seed_points']
    return (streamline_arrays_list[0:n], traj_nsteps_array[0:n], traj_length_array[0:n])

def prepare_memory(context, n_seed_points, chunk_size, max_traj_length, 
                   seed_point_array, mask_array, 
                   u_array, v_array, mapping_array, verbose):
    """
    Create Numpy array and PyOpenCL buffers to allow CPU-GPU data transfer.
    
    Args:
        context (pyopencl.Context):
        chunk_size (int):
        max_traj_length (int):
        seed_point_array (numpy.ndarray):
        mask_array (numpy.ndarray):
        u_array (numpy.ndarray):
        v_array (numpy.ndarray):
        mapping_array (numpy.ndarray):
        verbose (bool):
        
    Returns:
        numpy.ndarray, numpy.ndarray,\
        numpy.ndarray, numpy.ndarray, numpy.ndarray, \
        pyopencl.Buffer, pyopencl.Buffer, pyopencl.Buffer, pyopencl.Buffer, \
        pyopencl.Buffer, pyopencl.Buffer, pyopencl.Buffer:
        traj_nsteps_array, traj_length_array, 
        chunk_trajcs_array, chunk_nsteps_array, chunk_length_array,
        seed_point_buffer, uv_buffer, mask_buffer, mapping_buffer,
        chunk_trajcs_buffer, chunk_nsteps_buffer, chunk_length_buffer
    """
    # PyOpenCL array for seed points
    # Buffer for mask, (u,v) velocity array and more
    uv_array = np.stack((u_array,v_array),axis=2).copy().astype(dtype=np.float32)
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
    mapping_buffer      = cl.Buffer(context, COPY_READ_WRITE, hostbuf=mapping_array)
    chunk_nsteps_buffer = cl.Buffer(context, WRITE_ONLY, chunk_nsteps_array.nbytes )
    chunk_length_buffer = cl.Buffer(context, WRITE_ONLY, chunk_length_array.nbytes )
    chunk_trajcs_buffer = cl.Buffer(context, WRITE_ONLY, chunk_trajcs_array.nbytes )     
    vprint(verbose,'Array sizes:\n',
           'ROI-type =', mask_array.shape, '\n',
           'uv =',       uv_array.shape,   '\n')
    vprint(verbose,'Streamlines virtual array allocation:  ',
                 '   dims={0}'.format(
                     (n_seed_points, chunk_trajcs_array.shape[1],2)), 
                 '  size={}'.format(state.neatly(
                     n_seed_points*chunk_trajcs_array.shape[1]*2
                        *np.dtype(chunk_trajcs_array.dtype).itemsize)))
    vprint(verbose,'Streamlines array allocation per chunk:',
                     '   dims={0}'.format(chunk_trajcs_array.shape), 
                     '  size={}'.format(state.neatly(
      chunk_trajcs_array.size*np.dtype(chunk_trajcs_array.dtype).itemsize)))

    return (traj_nsteps_array, traj_length_array, 
            chunk_trajcs_array, chunk_nsteps_array, chunk_length_array,
            seed_point_buffer, uv_buffer, mask_buffer, mapping_buffer,
            chunk_trajcs_buffer, chunk_nsteps_buffer, chunk_length_buffer)

