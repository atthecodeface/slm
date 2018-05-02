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
/// @param[out]  mapping_array: multi-flag array
/// @param[out] trajectories_array: lists of streamline trajectories, stored as
///                                 compressed-into-byte dx,dy vector sequences;
///                                 one list per @p seed_point_array vector
/// @param[out] traj_nsteps_array: list of number of steps along each trajectory;
///                                 one per @p seed_point_array vector
/// @param[out] traj_length_array: list of lengths of each trajectory;
///                                 one per @p seed_point_array vector
///
/// @returns void
///
/// @ingroup integrate
///
__kernel void integrate_trajectory( __global const float2 *seed_point_array,
                                    __global const bool   *mask_array,
                                    __global const float2 *uv_array,
                                    __global       uint   *mapping_array,
                                    __global       char2  *trajectories_array,
                                    __global       ushort *traj_nsteps_array,
                                    __global       float  *traj_length_array )
{
    // global_id plus the chunk SEEDS_CHUNK_OFFSET is a seed point index
    const uint global_id = get_global_id(0u)+get_global_id(1u)*get_global_size(0u),
               seed_idx = (SEEDS_CHUNK_OFFSET)+global_id,
               trajectory_index = global_id*(MAX_N_STEPS);
    __global char2 *trajectory_vec;

    if (seed_idx>=N_SEED_POINTS) {
        // This is a "padding seed", so let's bail
        return;
    }
    // Report how kernel instances are distributed
    if (seed_idx==0) {
        printf("\nOn GPU/OpenCL device: #workitems=%d  #workgroups=%d => work size=%d\n",
                get_local_size(0u), get_num_groups(0u),
                get_local_size(0u)*get_num_groups(0u));
    }

    // Bug fix: bail BEFORE reading this element, because trajectories_array
    //   isn't padded and shouldn't be accessed for seed_idx>=N_SEED_POINTS
    trajectory_vec = &trajectories_array[trajectory_index];

    // Trace a "smooth" streamline from the seed point coordinate
    trajectory_record( uv_array, mask_array,
                       mapping_array, traj_nsteps_array, traj_length_array,
                       trajectory_vec, global_id, seed_idx,
                       seed_point_array[seed_idx] );
}
#endif
