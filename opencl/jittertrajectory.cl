///
/// @file jittertrajectory.cl
///
/// Streamline integration functions.
///
/// @author CPS
///

///
/// @defgroup integrate Streamline integration
/// Kernels and functions used to integrate streamlines.
///

#ifdef KERNEL_INTEGRATE_FIELDS
///
/// Integrate a jittered flow path downstream or upstream.
/// Write the streamline count and lengths to slc, slt arrays.
/// Don't record the trajectory itself.
///
/// Compiled if KERNEL_INTEGRATE_TRAJECTORY is defined.
///
/// @param[in]  uv_array: flow unit velocity vector grid (padded)
/// @param[in]  mask_array: grid pixel mask (padded),
///                         with @p true = masked, @p false = good
/// @param[out] slc_array: grid recording accumulated count of streamline integration
///                        steps across each pixel (padded)
/// @param[out] slt_array: grid recording accumulated count of streamline segment lengths
///                        crossing each pixel (padded)
/// @param[in]  global_id: ID of the kernel instance
/// @param[in]  seed_idx: index of the seed vector in the list seed_point_array;
///                       if chunkified, the sequence of indexes is offset from
///                       @p global_id by @p SEEDS_CHUNK_OFFSET
/// @param[in]  current_seed_point_vec: vector (real, float2) for the current point
///                                     along the streamline trajectory
/// @param[in]  initial_rng_state: RNG state and integer variate
///
/// @returns void
///
/// @ingroup integrate
///
static inline void trajectory_jittered( __global const float2 *uv_array,
                                        __global const bool   *mask_array,
                                        __global       uint   *slc_array,
                                        __global       uint   *slt_array,
                                                 const uint    global_id,
                                                 const uint    seed_idx,
                                                 const float2  current_seed_point_vec,
                                                 const uint    initial_rng_state )
{
    // Private variables - non-constant within this kernel instance
    __private uint idx, prev_idx, n_steps=0u, rng_state=initial_rng_state;
    __private float l_trajectory=0.0f, dl=0.0f, dt=DT_MAX;
    __private float2 uv1_vec, uv2_vec, dxy1_vec, dxy2_vec,
                     vec=current_seed_point_vec, prev_vec, next_vec;
    prev_vec = vec;

    // Start by recording the seed point
    idx = get_array_idx(vec);
    if (!mask_array[idx])
        atomic_write_sl_data(&slt_array[idx], &slc_array[idx], l_trajectory);

    // Loop downstream until the pixel is masked, i.e., we've exited the basin or grid,
    //   or if the streamline is too long (in l_trajectory or n_steps)
    while (idx<(NX_PADDED*NY_PADDED) && !mask_array[idx]
                        && (l_trajectory<MAX_LENGTH && n_steps<(MAX_N_STEPS-1))) {
        compute_step_vec_jittered(dt, uv_array, &rng_state, &dxy1_vec, &dxy2_vec,
                                  &uv1_vec, &uv2_vec, vec, &next_vec, &idx);
        if (idx<(NX_PADDED*NY_PADDED)) {
            if (!mask_array[idx])
                if (runge_kutta_step_write_sl_data(&dt, &dl, &l_trajectory,
                                                   &dxy1_vec, &dxy2_vec,
                                                   &vec, &prev_vec, next_vec,
                                                   &n_steps, &idx, &prev_idx,
                                                   mask_array, slt_array, slc_array))
                                          break;
            } else {
                euler_step_write_sl_data(&dt, &dl, &l_trajectory, uv1_vec,
                                         &vec, prev_vec, &n_steps, &idx, &prev_idx,
                                         mask_array, slt_array, slc_array);
                break;
            }
    }
    return;
}
#endif

