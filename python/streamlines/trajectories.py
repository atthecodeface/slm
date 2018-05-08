"""
GPU-based streamline integration.

Provides PyOpenCL-accelerated functions to integrate streamlines using
2nd-order Runge-Kutta and (streamline tail-step only) Euler methods.
Basins of interest can be delimited by masking. 

"""

import pyopencl as cl
import numpy as np
import os
os.environ['PYTHONUNBUFFERED']='True'
os.environ['PYOPENCL_NO_CACHE']='True'
os.environ['PYOPENCL_COMPILER_OUTPUT']='0'

from streamlines import pocl
from streamlines.state import neatly
from streamlines.useful import vprint, create_seeds, compute_stats

__all__ = ['Trajectories']

import warnings
pdebug = print

class Trajectories():
    def __init__(   self,
                    which_cl_platform,
                    which_cl_device,
                    cl_src_path         = None,
                    info                = None,
                    mask_array          = None,
                    uv_array            = None,
                    mapping_array       = None,
                    do_trace_downstream = True,
                    do_trace_upstream   = True,
                    verbose             = False ):
        """
        Initialize.
        
        Args:
            which_cl_platform (int):
            which_cl_device (int):
            cl_src_path (str):
            info (obj):
            mask_array (numpy.ndarray):
            uv_array (numpy.ndarray):
            mapping_array (numpy.ndarray):
            do_trace_downstream (bool):
            do_trace_upstream (bool):
            verbose (bool):
        """
        self.platform, self.device, self.context \
            = pocl.prepare_cl_context(which_cl_platform, which_cl_device)
        self.queue = cl.CommandQueue(self.context,
                                properties=cl.command_queue_properties.PROFILING_ENABLE)
        self.cl_src_path         = cl_src_path
        self.info                = info
        self.mask_array          = mask_array
        self.uv_array            = uv_array
        self.mapping_array       = mapping_array
        self.do_trace_downstream = do_trace_downstream
        self.do_trace_upstream   = do_trace_upstream
        self.verbose             = verbose
        
    def integrate(self):
        """
        Trace each streamline from its corresponding seed point using 2nd-order 
        Runge-Kutta integration of the topographic gradient vector field.
        
        This function is a master wrapper connecting the streamlines object and its 
        trace() method to the GPU/OpenCL wrapper function gpu_compute_trajectories(). 
        As such it acts on a set of parameters passed as arguments here rather than by 
        accessing object variables. 
        
        Workflow parameters are transferred here bundled in the info object,
        which is parsed as well as passed on to gpu_compute_trajectories.
        
        The tasks undertaken by this function are to:
           1) prepare the OpenCL context, device and kernel source string
           2) calculate how to split the streamline tracing into chunks
           3) invoke the GPU/OpenCL device computation
           4) post-process the streamline total length (slt) array (scale, sqrt)
              and compute streamline trajectories statistics
        """
        vprint(self.verbose,'Integrating trajectories...')
        
        # Shorthand
        cl_src_path         = self.cl_src_path
        info                = self.info
        mask_array          = self.mask_array
        uv_array            = self.uv_array
        mapping_array       = self.mapping_array

        # Prepare CL essentials
        device        = self.device
        context       = self.context
        queue         = self.queue
        cl_kernel_source \
            = pocl.read_kernel_source(cl_src_path,['rng.cl','essentials.cl',
                    'writearray.cl','updatetraj.cl','computestep.cl',
                    'rungekutta.cl','trajectory.cl','integratetraj.cl'])
                
        # Seed point selection, padding and shuffling
        n_trajectory_seed_points = info.n_trajectory_seed_points
        pad_width                = info.pad_width
        n_work_items             = info.n_work_items
        do_shuffle               = info.do_shuffle
        shuffle_rng_seed         = info.shuffle_rng_seed
        self.seed_point_array, n_seed_points, n_padded_seed_points \
            = create_seeds(mask_array, pad_width, n_work_items, 
                           n_seed_points=n_trajectory_seed_points, 
                           do_shuffle=do_shuffle, rng_seed=shuffle_rng_seed,
                           verbose=self.verbose)
        info.n_seed_points        = n_seed_points
        info.n_padded_seed_points = n_padded_seed_points
        
        # Mapping flag array - not likely already defined, but just in case...
        if mapping_array is None:
            mapping_array = np.zeros_like(mask_array, dtype=np.uint32)
    
        # Chunkification
        gpu_traj_memory_limit = (device.get_info(cl.device_info.GLOBAL_MEM_SIZE) 
                                 *info.gpu_memory_limit_pc)//100
        full_traj_memory_request = (n_seed_points*np.dtype(np.uint8).itemsize
                                    *info.max_n_steps*2)
        n_chunks_required = max(1,int(np.ceil(
                            full_traj_memory_request/gpu_traj_memory_limit)) )
                
        vprint(self.verbose,
               'GPU/OpenCL device global memory limit for streamline trajectories: {}' 
                  .format(neatly(gpu_traj_memory_limit)))
        vprint(self.verbose,
               'GPU/OpenCL device memory required for streamline trajectories: {}'
                  .format(neatly(full_traj_memory_request)), end='')
        vprint(self.verbose,' => {}'.format('no need to chunkify' if n_chunks_required==1
                        else 'need to split into {} chunks'.format(n_chunks_required) ))
        self.choose_chunks(n_chunks_required)
#         compile_options = pocl.set_compile_options_alt(info,'INTEGRATE_TRAJECTORY')
#         vprint(self.verbose,'Compile options:\n',compile_options)
        
        # Do integrations on the GPU
        self.gpu_compute_trajectories( device, context, queue, cl_kernel_source )
            
        # Streamline stats
        pixel_size = info.pixel_size
        self.traj_stats_df = compute_stats(self.traj_length_array, self.traj_nsteps_array,
                                      pixel_size, self.verbose)
        dds = self.traj_stats_df['ds']['downstream','mean']
        uds = self.traj_stats_df['ds']['upstream','mean']
            
        # Done
        vprint(self.verbose,'...done')
    
    def choose_chunks(self, n_chunks):
        """
        Compute lists of parameters needed to carry out GPU/OpenCL device computations
        in chunks.
        
        Args:
            n_chunks (int):
            
        """
        n_seed_points        = self.info.n_seed_points
        n_padded_seed_points = self.info.n_padded_seed_points
        n_work_items         = self.info.n_work_items
        
        self.chunk_size = int(np.round(n_padded_seed_points/n_chunks))
        n_global = n_padded_seed_points
        chunk_list = [[chunk_idx, chunk,
                       min(n_seed_points,chunk+self.chunk_size)-chunk, self.chunk_size] 
                       for chunk_idx,chunk 
                        in enumerate(range(0,n_global,self.chunk_size))]            
    
        trace_do_list = [[self.do_trace_downstream, 'Downstream:', 0, np.float32(+1.0)]]\
                      + [[self.do_trace_upstream,   'Upstream:  ', 1, np.float32(-1.0)]]
        self.trace_do_chunks = [td+chunk for td in trace_do_list for chunk in chunk_list]
            
        vprint(self.verbose,'Total number of kernel instances: {0:,}'
                    .format(n_global))
        vprint(self.verbose,'Number of chunks = seed point array divisor:', n_chunks)   
        vprint(self.verbose,'Chunk size = number of kernel instances per chunk: {0:,}'
                    .format(self.chunk_size))   
    
    def gpu_compute_trajectories( self, device, context, queue, cl_kernel_source ):
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
            

        """
        # Shorthand
        info                = self.info
        seed_point_array    = self.seed_point_array
        mask_array          = self.mask_array
        uv_array            = self.uv_array
        mapping_array       = self.mapping_array
        
        # Prepare memory, buffers 
        streamline_arrays_list = [[],[]]
        ns = info.n_seed_points
        nl = info.max_n_steps
        traj_nsteps_array  = np.zeros([ns,2], dtype=np.uint16)
        traj_length_array  = np.zeros([ns,2], dtype=np.float32)
        # Chunk-sized temporary arrays
        # Use "bag o' bytes" buffer for huge trajectories array. Write (by GPU) only.
        chunk_trajcs_array = np.zeros([self.chunk_size,nl,2], dtype=np.int8)
        chunk_nsteps_array = np.zeros([self.chunk_size], dtype=traj_nsteps_array.dtype)
        chunk_length_array = np.zeros([self.chunk_size], dtype=traj_length_array.dtype)
        array_dict = { 'seed_point':   {'array': seed_point_array,  'rwf': 'RO'},
                       'mask':         {'array': mask_array,        'rwf': 'RO'}, 
                       'uv':           {'array': uv_array,          'rwf': 'RO'}, 
                       'mapping':      {'array': mapping_array,     'rwf': 'RW'}, 
                       'chunk_trajcs': {'array': chunk_trajcs_array,'rwf': 'WO'},
                       'chunk_nsteps': {'array': chunk_nsteps_array,'rwf': 'WO'}, 
                       'chunk_length': {'array': chunk_length_array,'rwf': 'WO'}, 
                       'traj_nsteps':  {'array': traj_nsteps_array, 'rwf': 'NB'}, 
                       'traj_length':  {'array': traj_length_array, 'rwf': 'NB'} }
        buffer_dict = pocl.prepare_buffers(context, array_dict, self.verbose)
             
        # Downstream and upstream passes aka streamline integrations from
        #   chunks of seed points aka subsets of the total set
        n_work_items = info.n_work_items
        chunk_size_factor = info.chunk_size_factor
        max_time_per_kernel = info.max_time_per_kernel
        for downup_str, downup_idx, downup_sign, chunk_idx, \
            seeds_chunk_offset, n_chunk_seeds, n_chunk_ki in [td[1:] \
                for td in self.trace_do_chunks if td[0]]:
            vprint(self.verbose,
                   '{0} downup={1} sgn(uv)={2:+} chunk={3} seeds: {4}+{5} => {6:}'
                       .format(downup_str, downup_idx, downup_sign, chunk_idx,
                               seeds_chunk_offset, n_chunk_seeds, 
                               seeds_chunk_offset+n_chunk_seeds))
    
            # Specify this integration job's parameters
            global_size = [n_chunk_ki,1]
            vprint(self.verbose,
                   'Seed point buffer size = {}*8 bytes'
                   .format(buffer_dict['seed_point'].size/8))
            local_size = [info.n_work_items,1]
            info.downup_sign = downup_sign
            info.seeds_chunk_offset = seeds_chunk_offset
            
            ##################################
            
            # Compile the CL code
            compile_options = pocl.set_compile_options(info, 'INTEGRATE_TRAJECTORY', 
                                                       downup_sign=downup_sign)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                program = cl.Program(context,
                                     cl_kernel_source).build(options=compile_options)
            pocl.report_build_log(program, device, self.verbose)
            # Set the GPU kernel
            kernel = program.integrate_trajectory
            
            # Designate buffered arrays
            kernel.set_args(*list(buffer_dict.values()))
            kernel.set_scalar_arg_dtypes( [None]*len(buffer_dict) )
            
            # Trace the streamlines on the GPU    
            vprint(self.verbose,
                '#### GPU/OpenCL computation: {0} work items... ####'
                .format(global_size[0]))    
            pocl.report_kernel_info(device,kernel,self.verbose)
            elapsed_time \
                = pocl.adaptive_enqueue_nd_range_kernel(queue, kernel, global_size, 
                                               local_size, n_work_items,
                                               chunk_size_factor=chunk_size_factor,
                                               max_time_per_kernel=max_time_per_kernel,
                                               verbose=self.verbose )
            vprint(self.verbose,
                   '#### ...elapsed time for {1} work items: {0:.3f}s ####'
                   .format(elapsed_time,global_size[0]))
            queue.finish()   
            
            # Copy GPU-computed results back to CPU
            for array_info in array_dict.items():
                if 'W' in array_info[1]['rwf']:
                    cl.enqueue_copy(queue, array_info[1]['array'], 
                                    buffer_dict[array_info[0]])
                    queue.finish()
    
            ##################################
                            
            # This is part going to be slow...
            for traj_nsteps,traj_vec in \
                zip(chunk_nsteps_array[:n_chunk_seeds], 
                    chunk_trajcs_array[:n_chunk_seeds]): 
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
        
        vprint(self.verbose,'Building streamlines compressed array')
        for downup_idx in [0,1]:
            streamline_arrays_list[downup_idx] \
                = np.array(streamline_arrays_list[downup_idx])
        vprint(self.verbose,
               'Streamlines actual array allocation:  size={}'.format(neatly(
           np.sum(traj_nsteps_array[:,:])*np.dtype(chunk_trajcs_array.dtype).itemsize)))
       
       # Do copy() to force array truncation rather than return of a truncated view
        self.streamline_arrays_list = streamline_arrays_list[0:ns].copy()
        self.traj_nsteps_array      = traj_nsteps_array[0:ns].copy()
        self.traj_length_array      = traj_length_array[0:ns].copy()
        
    
    #     vprint(self.verbose,'Array sizes:\n',
    #            'ROI-type =', mask_array.shape, '\n',
    #            'uv =',       uv_array.shape)
    #     vprint(self.verbose,'Streamlines virtual array allocation:  ',
    #                  '   dims={0}'.format(
    #                      (n_seed_points, chunk_trajcs_array.shape[1],2)), 
    #                  '  size={}'.format(neatly(
    #                      n_seed_points*chunk_trajcs_array.shape[1]*2
    #                         *np.dtype(chunk_trajcs_array.dtype).itemsize)))
    #     vprint(self.verbose,'Streamlines array allocation per chunk:',
    #                      '   dims={0}'.format(chunk_trajcs_array.shape), 
    #                      '  size={}'.format(neatly(
    #       chunk_trajcs_array.size*np.dtype(chunk_trajcs_array.dtype).itemsize)))
