"""
GPU-based streamline integration.

Provides PyOpenCL-accelerated functions to integrate streamlines using
2nd-order Runge-Kutta and (streamline tail-step only) Euler methods.
Basins of interest can be delimited by masking. 

"""

import pyopencl as cl
import numpy    as np
from os import environ
environ['PYTHONUNBUFFERED']='True'
environ['PYOPENCL_NO_CACHE']='True'
environ['PYOPENCL_COMPILER_OUTPUT']='0'
from streamlines        import pocl
from streamlines.useful import neatly, vprint, create_seeds

__all__ = ['Fields']

import warnings
pdebug = print

class Fields():
    def __init__(   self,
                    which_cl_platform,
                    which_cl_device,
                    cl_src_path         = None,
                    info                = None,
                    data                = None,
                    verbose             = False,
                    gpu_verbose         = False ):
        """
        Initialize.
        
        Args:
            cl_src_path (str):
            which_cl_platform (int):
            which_cl_device (int):
            info (obj):
            data (obj):
            verbose (bool):
            gpu_verbose (bool):
        """
        self.platform, self.device, self.context \
            = pocl.prepare_cl_context(which_cl_platform, which_cl_device)
        self.queue = cl.CommandQueue(self.context,
                                properties=cl.command_queue_properties.PROFILING_ENABLE)
        self.cl_src_path         = cl_src_path
        self.info                = info
        self.data                = data
        self.verbose             = verbose
        self.gpu_verbose         = gpu_verbose
        
    def integrate(self):
        """
        Trace each streamline from its corresponding seed point using 2nd-order 
        Runge-Kutta integration of the topographic gradient vector field.
        
        This function is a master wrapper connecting the streamlines object and its 
        trace() method to the GPU/OpenCL wrapper function gpu_integrate(). As such it
        acts on a set of parameters passed as arguments here rather than by accessing
        object variables. 
        
        Workflow parameters are transferred here bundled in the Numpy structure array 
        info, which is parsed as well as passed on to gpu_integrate.
        
        The tasks undertaken by this function are to:
           1) prepare the OpenCL context, device and kernel source string
           2) calculate how to split the streamline tracing into chunks
           3) invoke the GPU/OpenCL device computation
           4) post-process the streamline total length (slt) array (scale, sqrt)
              and compute streamline trajectories statistics
        """
        vprint(self.verbose,'Integrating streamline fields...')
        
        # Shorthand
        cl_src_path         = self.cl_src_path
        info                = self.info
        mask_array          = self.data.mask_array
        uv_array            = self.data.uv_array
        mapping_array       = self.data.mapping_array
        traj_stats_df       = self.data.traj_stats_df

        # Prepare CL essentials
        device        = self.device
        context       = self.context
        queue         = self.queue
        cl_kernel_source \
            = pocl.read_kernel_source(cl_src_path,['rng.cl','essentials.cl',
                    'writearray.cl','updatetraj.cl','computestep.cl',
                    'rungekutta.cl','jittertrajectory.cl','integratefields.cl'])
        n_padded_seed_points = info.n_padded_seed_points
    
        # Memory check - not really needed 
        gpu_traj_memory_limit = (device.get_info(cl.device_info.GLOBAL_MEM_SIZE) 
                                 *info.gpu_memory_limit_pc)//100
        full_traj_memory_request = (mask_array.shape[0]*mask_array.shape[1]
                                    *np.dtype(np.float32).itemsize*2*3)    
        vprint(self.verbose,
               'GPU/OpenCL device global memory limit for streamline trajectories: {}' 
                  .format(neatly(gpu_traj_memory_limit)))
        vprint(self.verbose,
               'GPU/OpenCL device memory required for streamline trajectories: {}'
                  .format(neatly(full_traj_memory_request)))
         
        # Seed point selection, padding and shuffling
        pad_width                = info.pad_width
        n_work_items             = info.n_work_items
        do_shuffle               = info.do_shuffle
        shuffle_rng_seed         = info.shuffle_rng_seed
        (self.data.seed_point_array, info.n_seed_points, info.n_padded_seed_points) \
            = create_seeds(mask_array, pad_width, n_work_items, 
                           do_shuffle=do_shuffle, rng_seed=shuffle_rng_seed,
                           verbose=self.verbose)
        
        # Prep for GPU compute
        n_global = info.n_padded_seed_points
        pad_length = np.uint32(np.round(n_global/n_work_items))*n_work_items-n_global
        if pad_length>0:
            padding_array = -np.ones([pad_length,2], dtype=np.float32)
            vprint(self.verbose,
                   'Chunk size adjustment for {0} CL work items/group: {1}->{2}...'
                 .format(n_work_items, n_global, n_global+pad_length))
        n_global += pad_length
        
        # Do integrations on the GPU
        self.gpu_integrate(device, context, queue, cl_kernel_source, n_global)
        
        # Streamline stats
        pixel_size = info.pixel_size
        dds =  traj_stats_df['ds']['downstream','mean']
        uds =  traj_stats_df['ds']['upstream','mean']
        # slc: simple counter of streamline points in a pixel
        # slt: total streamline lengths running across a pixel
        # slt: sum of line lengths crossing a pixel * number of line-points per pixel
        # slt: <=> sum of line-points per pixel integrator_step_factor
#         dds = info.integrator_step_factor
#         uds = info.integrator_step_factor
        self.data.slt_array[:,:,0] = self.data.slt_array[:,:,0]*(dds/pixel_size)
        self.data.slt_array[:,:,1] = self.data.slt_array[:,:,1]*(uds/pixel_size)
#         self.data.slt_array[:,:,0] = self.data.slt_array[:,:,0]*(dds/pixel_size)
#         self.data.slt_array[:,:,1] = self.data.slt_array[:,:,1]*(uds/pixel_size)
        # slt:  => sum of line-points per meter
        self.data.slt_array = np.sqrt(self.data.slt_array) 
        # slt:  =>  sqrt(area)
        
        self.data.slc_array \
            = np.sqrt(2.0)*np.power(self.data.slc_array
                                    /info.subpixel_seed_point_density**2,2/3)
    
        # Done
        vprint(self.verbose,'...done')

    def gpu_integrate(self, device, context, queue, cl_kernel_source, n_global):
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
            n_global (int):
        """

        # Shorthand
        info                = self.info
        seed_point_array    = self.data.seed_point_array
        mask_array          = self.data.mask_array
        uv_array            = self.data.uv_array
        mapping_array       = self.data.mapping_array
        
        # Prepare memory, buffers 
        roi_nxy = mapping_array.shape
#         pdebug('mask_array',mask_array.shape)
#         pdebug('uv_array',uv_array.shape)
#         pdebug('mapping_array',mapping_array.shape)
        slc_array     = np.zeros((roi_nxy[0],roi_nxy[1]), dtype=np.uint32)
        slt_array     = np.zeros((roi_nxy[0],roi_nxy[1]), dtype=np.uint32)
        self.data.slc_array = np.zeros((roi_nxy[0],roi_nxy[1],2), dtype=np.uint32)
        # Note the returned slt array is FLOAT32 but the GPU computes are done on UINT32
        self.data.slt_array = np.zeros((roi_nxy[0],roi_nxy[1],2), dtype=np.float32)
        self.data.sla_array = np.zeros((roi_nxy[0],roi_nxy[1],2), dtype=np.float32)
        array_dict = { 'seed_point': {'array': seed_point_array, 'rwf': 'RO'},
                       'mask':       {'array': mask_array,       'rwf': 'RO'}, 
                       'uv':         {'array': uv_array,         'rwf': 'RO'}, 
                       'mapping':    {'array': mapping_array,    'rwf': 'RW'}, 
                       'slc':        {'array': slc_array,        'rwf': 'RW'}, 
                       'slt':        {'array': slt_array,        'rwf': 'RW'} }
        buffer_dict =  pocl.prepare_buffers(context, array_dict, self.verbose)
    
        # Downstream and upstream passes aka streamline integrations from
        #   chunks of seed points aka subsets of the total set
        global_size         = [n_global,1]
        local_size          = [info.n_work_items,1]
        n_work_items        = info.n_work_items
        chunk_size_factor   = info.chunk_size_factor
        max_time_per_kernel = info.max_time_per_kernel
        vprint(self.verbose,
               'Seed point buffer size = {}*8 bytes'.
               format(buffer_dict['seed_point'].size/8))
        # Downstream then upstream loop
        for downup_idx, downup_sign in [[0,+1.0],[1,-1.0]]:
            info.downup_sign = downup_sign
            
            ##################################
            
            # Compile the CL code
            compile_options = pocl.set_compile_options(info, 'INTEGRATE_FIELDS', 
                                                       downup_sign=downup_sign)
            vprint(self.gpu_verbose,'Compile options:\n',compile_options)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                program = cl.Program(context,
                                     cl_kernel_source).build(options=compile_options)
            pocl.report_build_log(program, device, self.gpu_verbose)
            
            # Set the GPU kernel
            kernel = program.integrate_fields
            
            # Designate buffered arrays
            kernel.set_args(*list(buffer_dict.values()))
            kernel.set_scalar_arg_dtypes( [None]*len(buffer_dict) )
            
            # Trace the streamlines on the GPU
            vprint(self.gpu_verbose,
                   '#### GPU/OpenCL computation: {0} work items... ####'
                   .format(global_size[0]))
            pocl.report_kernel_info(device,kernel,self.gpu_verbose)
            elapsed_time \
                = pocl.adaptive_enqueue_nd_range_kernel(
                                                queue, kernel, global_size, 
                                                local_size, n_work_items,
                                                chunk_size_factor=chunk_size_factor,
                                                max_time_per_kernel=max_time_per_kernel,
                                                verbose=self.gpu_verbose )
            vprint(self.gpu_verbose,
                   '#### ...elapsed time for {1} work items: {0:.3f}s ####'
                   .format(elapsed_time,global_size[0]))
            queue.finish()   
    
            # Copy back the streamline length, distance density grid
            cl.enqueue_copy(queue, slc_array, buffer_dict['slc'])
            queue.finish()   
            cl.enqueue_copy(queue, slt_array, buffer_dict['slt'])
            queue.finish()   
                    
            ##################################
                    
            # Copy out the slc, slt results for this pass only
            # Count of streamlines entering per pixel width: n/meter
            self.data.slc_array[:,:,downup_idx] += slc_array
            # Average streamline length of streamlines entering each pixel: meters
            self.data.slt_array[:,:,downup_idx] += slt_array.astype(np.float32)
            if downup_idx==0:
                # Zero the GPU slt, slc arrays before using in the next pass
                slc_array.fill(0)
                slt_array.fill(0.0)
                cl.enqueue_copy(queue, buffer_dict['slc'],slc_array)
                queue.finish()   
                cl.enqueue_copy(queue, buffer_dict['slt'],slt_array)
                queue.finish()   
    
        cl.enqueue_copy(queue, mapping_array,  buffer_dict['mapping'])
        queue.finish()   
        
        # Compute average streamline lengths (sla) from total lengths (slt) & counts (slc)
        # Shorthand
        (slc,sla,slt) = (self.data.slc_array, self.data.sla_array, self.data.slt_array)
        # slc: count of lines crossing a pixel * number of line-points per pixel
        # slt: sum of line lengths crossing a pixel * number of line-points per pixel
        # sla: sum of line lengths / count of lines
        sla[slc==0] = 0.0
        sla[slc>0]  = np.sqrt(2.0)*slt[slc>0]/slc[slc>0]
        slt[slc>0]  = slt[slc>0]/info.subpixel_seed_point_density**2
#         slc = slc/info.subpixel_seed_point_density**2
        
