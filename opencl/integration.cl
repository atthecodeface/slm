///
/// @file integration.cl
///
/// Streamline trajectory integration kernel and related tracing functions.
///
/// @author CPS
///
/// @todo Fix big-DTM crash issue
/// @todo Perhaps use compiler directive volatile where variables not const?
/// @todo Update doc about trajectory integration to describe subpixel seeding & jittering
///
/// @bug Crashes (reported as 'abort 6' by PyOpenCL) occur for very large DTMs.
///      The reason remains obscure: it may be because of GPU timeout, but more likely
///      is because of a memory leakage.
///
///

///
/// Byte reversal
///
/// Macro to perform byte reversal per GJS's suggestion.
///
/// @param[in,out] initial_rng_state: RNG state (thus initally the seed) and RNG variate
///
#define BYTE_REVERSAL(initial_rng_state) \
    initial_rng_state =   (initial_rng_state>>24)&0xff \
                        | (initial_rng_state>> 8)&0xff00 \
                        | (initial_rng_state<< 8)&0xff0000 \
                        | (initial_rng_state<<24)&0xff000000;



///
/// Initialize the Lehmer random number generator
///
/// Macro to scramble the initial RNG state to reduce correlation of
///    neighboring streamline jitters
///
/// @param[in,out] initial_rng_state: RNG state (thus initally the seed) and RNG variate
///
#define INITIALIZE_RNG(initial_rng_state,seed_idx) \
    initial_rng_state = seed_idx; \
    /*initial_rng_state = i+(j+seed_idx);*/ \
    /*initial_rng_state = i+(j+seed_idx*SUBPIXEL_SEED_POINT_DENSITY)\
     * *SUBPIXEL_SEED_POINT_DENSITY;*/ \
    BYTE_REVERSAL(initial_rng_state); \
    lehmer_rand_uint(&initial_rng_state); \
    BYTE_REVERSAL(initial_rng_state);

#ifdef KERNEL_INTEGRATE_TRAJECTORY
///
/// GPU kernel that drives streamline integration from seed positions
/// given in @p seed_point_array, controlled by the 'flow' vector field
/// given in @p uv_array, and either terminated at pixels masked in
/// mask_array or because a streamline exceeds a threshold
/// distance (length or number of integration points) given by parameters
/// stored in info. Further integration parameters are provided in this struct.
///
/// The kernel acts on one seed point only. It chooses this seed point
/// by computing a global id and using it to index the @p seed_point_array.
/// UPDATE: now doing sub-pixel streamlines as a set per seed point... need to doc here
///
/// Each streamline trajectory is returned in the appropriate location
/// in @p trajectories_array as a list of compressed-into-byte dx,dy values.
///
/// Compiled if KERNEL_INTEGRATE_TRAJECTORY is defined.
///
/// @param[in]  seed_point_array: list of initial streamline point vectors,
///                               one allotted to each kernel instance
/// @param[in]  mask_array: grid pixel mask (padded),
///                         with @p true = masked, @p false = good
/// @param[in]  uv_array: flow unit velocity vector grid (padded)
/// @param[out] trajectories_array: lists of streamline trajectories, stored as
///                                 compressed-into-byte dx,dy vector sequences;
///                                 one list per @p seed_point_array vector
/// @param[out] traj_nsteps_array: list of number of steps along each trajectory;
///                                 one per @p seed_point_array vector
/// @param[out] traj_length_array: list of lengths of each trajectory;
///                                 one per @p seed_point_array vector
/// @param[out] slc_array: grid recording accumulated count of streamline integration
///                        steps across each pixel (padded)
/// @param[out] slt_array: grid recording accumulated count of streamline segment lengths
///                        crossing each pixel (padded)
///
/// @returns void
///
/// @ingroup integrate
///
__kernel void integrate_trajectory( __global const float2 *seed_point_array,
                                    __global const bool   *mask_array,
                                    __global const float2 *uv_array,
                                    __global       char2  *trajectories_array,
                                    __global       ushort *traj_nsteps_array,
                                    __global       float  *traj_length_array,
                                    __global       uint   *slc_array,
                                    __global       uint   *slt_array )
{
    // global_id plus the chunk SEEDS_CHUNK_OFFSET is a seed point index
    const uint global_id = get_global_id(0u)+get_global_id(1u)*get_global_size(0u),
               seed_idx = (SEEDS_CHUNK_OFFSET)+global_id,
               trajectory_index = global_id*(MAX_N_STEPS);
    const float2 current_seed_point_vec = seed_point_array[seed_idx];
    __global char2 *trajectory_vec = &trajectories_array[trajectory_index];
    __private uint i=0,j=0, initial_rng_state;

    // Report how kernel instances are distributed
    if (seed_idx==0) {
        printf("On GPU/OpenCL device: #workitems=%d  #workgroups=%d\n",
                get_local_size(0u), get_num_groups(0u));
    }

    // Trace a "smooth" streamline from the seed point coordinate
    trajectory_record( uv_array, mask_array, traj_nsteps_array, traj_length_array,
                       trajectory_vec, global_id, seed_idx,
                       seed_point_array[seed_idx]);

    // Trace a set of streamlines from a grid of sub-pixel positions centered
    //    on the seed point
    // Generate an initial RNG state (aka 'seed the RNG')
    //   [was: using the sum of the current pixel index and the sub-pixel index]
    //   using the current pixel index ("seed_idx")
    //   byte-reversed per GJS suggestion
    INITIALIZE_RNG(initial_rng_state, seed_idx);
    for (j=0u;j<SUBPIXEL_SEED_POINT_DENSITY;j++) {
        for (i=0u;i<SUBPIXEL_SEED_POINT_DENSITY;i++){
            // Trace a jittered streamline from a sub-pixel-offset first point
            trajectory_jittered(uv_array, mask_array, slc_array, slt_array,
                               global_id, seed_idx,
                               current_seed_point_vec + (float2)(
                                    (float)i*SUBPIXEL_SEED_STEP-SUBPIXEL_SEED_HALFSPAN,
                                    (float)j*SUBPIXEL_SEED_STEP-SUBPIXEL_SEED_HALFSPAN ),
                               initial_rng_state);
        }
    }
}
#endif
