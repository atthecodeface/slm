///
/// @file trajectory.cl
///
/// Streamline integration functions.
///
/// @author CPS
///

///
/// @defgroup integrate Streamline integration
/// Kernels and functions used to integrate streamlines.
///

#ifdef KERNEL_INTEGRATE_TRAJECTORY
///
/// Integrate a streamline downstream or upstream; record the trajectory.
///
/// Compiled if KERNEL_INTEGRATE_TRAJECTORY is defined.
///
/// @param[in]  uv_array: flow unit velocity vector grid (padded)
/// @param[in]  mask_array: grid pixel mask (padded),
///                         with @p true = masked, @p false = good
/// @param[out] traj_nsteps_array: list of number of steps along each trajectory;
///                                 one per @p seed_point_array vector
/// @param[out] traj_length_array: list of lengths of each trajectory;
///                                 one per @p seed_point_array vector
/// @param[out] trajectory_vec: sequence of compressed-as-byte dx,dy integration steps
///                             along this streamline trajectory
/// @param[in]  global_id: ID of the kernel instance
/// @param[in]  seed_idx: index of the seed vector in the list seed_point_array;
///                       if chunkified, the sequence of indexes is offset from
///                       @p global_id by @p SEEDS_CHUNK_OFFSET
/// @param[in]  current_seed_point_vec: vector (real, float2) for the current point
///                                     along the streamline trajectory
///
/// @returns void
///
/// @ingroup integrate
///
static inline void trajectory_record( __global const float2 *uv_array,
                                      __global const bool   *mask_array,
                                      __global       ushort *traj_nsteps_array,
                                      __global       float  *traj_length_array,
                                      __global       char2  *trajectory_vec,
                                               const uint    global_id,
                                               const uint    seed_idx,
                                               const float2  current_seed_point_vec )
{
    // Private variables - non-constant within this kernel instance
    __private uint idx, prev_idx, n_steps=0u;
    __private float l_trajectory=0.0f, dl=0.0f, dt=DT_MAX;
    __private float2 uv1_vec, uv2_vec, dxy1_vec, dxy2_vec,
                     vec=current_seed_point_vec, prev_vec=vec, next_vec;
    // Start
    prev_vec = vec;
    idx = get_array_idx(vec);
    // Loop downstream until the pixel is masked, i.e., we've exited the basin or grid,
    //   or if the streamline is too long (l or n)
    while (!mask_array[idx] && (l_trajectory<MAX_LENGTH && n_steps<(MAX_N_STEPS-1))) {
        compute_step_vec(dt, uv_array, &dxy1_vec, &dxy2_vec, &uv1_vec, &uv2_vec,
                         vec, &next_vec, &idx);
        if (!mask_array[idx]) {
            if (runge_kutta_step_record(&dt, &dl, &l_trajectory, &dxy1_vec, &dxy2_vec,
                  &vec, &prev_vec, &next_vec, &n_steps, &idx, &prev_idx, trajectory_vec))
                break;
        } else {
            euler_step_record(&dt, &dl, &l_trajectory, uv1_vec,
                              &vec, prev_vec, &n_steps, trajectory_vec);
            break;
        }
    }
    // Record this final trajectory point and return
    finalize_trajectory(global_id, n_steps, l_trajectory,
                        traj_nsteps_array, traj_length_array);
    return;
}
#endif
