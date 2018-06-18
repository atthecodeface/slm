/// @file connect.cl
///
/// Kernels to connect discontinous and dangling channels.
///
/// @author CPS
/// @bug No known bugs
///

#ifdef KERNEL_CONNECT_CHANNELS
///
/// Connect up channel strands by designating intervening pixels as channel pixels.
///
/// Compiled if KERNEL_CONNECT_CHANNELS is defined.
///
/// @param[in]     seed_point_array: list of initial streamline point vectors,
///                                  one allotted to each kernel instance
/// @param[in]     mask_array: grid pixel mask (padded),
///                            with @p true = masked, @p false = good
/// @param[in]     uv_array: flow unit velocity vector grid (padded)
/// @param[in,out] mapping_array: flag grid recording status of each pixel (padded)
///
/// @returns void
///
/// @ingroup structure
///
__kernel void connect_channels(
        __global const float2 *seed_point_array,
        __global const bool   *mask_array,
        __global const float2 *uv_array,
        __global       uint   *mapping_array
   )
{
    // For every non-masked pixel...

    const uint global_id = get_global_id(0u)+get_global_id(1u)*get_global_size(0u);
    if (global_id>=N_SEED_POINTS) {
        // This is a "padding" seed, so let's bail
        return;
    }
#ifdef VERBOSE
    // Report how kernel instances are distributed
    if (global_id==0 || global_id==get_global_offset(0u)) {
        printf("\n  >>> on GPU/OpenCL device: id=%d offset=%d ",
                get_global_id(0u),
                get_global_offset(0u));
        printf("#workitems=%d x #workgroups=%d = %d=%d\n",
                get_local_size(0u), get_num_groups(0u),
                get_local_size(0u)*get_num_groups(0u),
                get_global_size(0u));
    }
#endif
    const float2 current_seed_point_vec = seed_point_array[global_id];
    __private uint idx, prev_idx, n_steps = 0u, step=0u;
    __private float l_trajectory = 0.0f, dl = 0.0f, dt = DT_MAX;
    __private float2 next_vec, uv1_vec, uv2_vec, dxy1_vec, dxy2_vec,
                     vec = current_seed_point_vec, prev_vec = vec;
    __private char2 trajectory_vec[INTERCHANNEL_MAX_N_STEPS];

    // Remember here
    prev_vec = vec;
    idx = get_array_idx(vec);
    prev_idx = idx;
    // Integrate downstream one pixel
    while (prev_idx==idx && !mask_array[idx] && n_steps!=MAX_N_STEPS) {
        compute_step_vec(dt, uv_array, &dxy1_vec, &dxy2_vec, &uv1_vec, &uv2_vec,
                         vec, &next_vec, &idx);
        if (!mask_array[idx]) {
            if (connect_runge_kutta_step_record(&dt, &dl, &l_trajectory,
                                                &dxy1_vec, &dxy2_vec, &vec, &prev_vec,
                                                &next_vec, &n_steps, &idx, &prev_idx,
                                                trajectory_vec))
                continue;
        }
    }
    // Integrate until we're back onto a channel pixel OR we reach a masked pixel
    while ((mapping_array[idx] & IS_CHANNEL)==0 && !mask_array[idx]
                                          && n_steps!=INTERCHANNEL_MAX_N_STEPS) {
        compute_step_vec(dt, uv_array, &dxy1_vec, &dxy2_vec, &uv1_vec, &uv2_vec,
                         vec, &next_vec, &idx);
        if (!mask_array[idx]) {
            if (connect_runge_kutta_step_record(&dt, &dl, &l_trajectory,
                                                &dxy1_vec, &dxy2_vec, &vec, &prev_vec,
                                                &next_vec, &n_steps, &idx, &prev_idx,
                                                trajectory_vec))
                continue;
        }
    }
    if (n_steps>2 && n_steps<INTERCHANNEL_MAX_N_STEPS) {
        // At this point, we have either connected some channel pixels (type=1)
        //   or simply reached the trajectory tracking limit, or reached a masked pixel.
        // Now we need to designate all intervening pixels since the last channel pixel
        //   as 'intervening channel' type=3.
        vec = current_seed_point_vec;
        idx = get_array_idx(vec);
        step = 0u;
        while (!mask_array[idx] && step<n_steps-1) {
            // If this pixel was between channels, flag as both (1=channel; 2=between)
            if (mapping_array[idx]==0u) {
                atomic_or(&mapping_array[idx],IS_INTERCHANNEL);
            }
            // Increment along recorded trajectory, skipping first point
            vec = vec + uncompress(trajectory_vec[step]);
            idx = get_array_idx(vec);
            step++;
        }
    }
    return;
}
#endif
