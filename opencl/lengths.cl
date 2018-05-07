///
/// @file lengths.cl
///
/// Kernel to measure distances from midslope to thin channel pixels aka hillslope length.
///
/// @author CPS
/// @bug No known bugs
///

///
/// @defgroup lengths Hillslope lengths
/// Measure hillslope lengths
///


#ifdef KERNEL_HILLSLOPE_LENGTHS
///
/// TBD
///
/// Compiled if KERNEL_HILLSLOPE_LENGTHS is defined.
///
/// @param[in]     seed_point_array: list of initial streamline point vectors,
///                                  one allotted to each kernel instance
/// @param[in]     mask_array: grid pixel mask (padded),
///                            with @p true = masked, @p false = good
/// @param[in]     uv_array: flow unit velocity vector grid (padded)
/// @param[in,out] mapping_array: flag grid recording status of each pixel (padded)
/// @param[in,out] label_array: label grid giving the ID of the subsegment to which
///                             this pixel belongs (padded); the MSB is set if left flank
/// @param[out] traj_length_array: list of lengths of each trajectory;
///                                one per @p seed_point_array vector
///
/// @returns void
///
/// @ingroup lengths
///
__kernel void hillslope_lengths(
        __global const float2 *seed_point_array,
        __global const bool   *mask_array,
        __global const float2 *uv_array,
        __global const uint   *mapping_array,
        __global const uint   *label_array,
        __global       float  *traj_length_array
   )
{
    // For every non-thin-channel pixel

    const uint global_id = get_global_id(0u)+get_global_id(1u)*get_global_size(0u);
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
    if (global_id>=N_SEED_POINTS) {
        // This is a "padding" seed, so let's bail
        return;
    }
    __private uint idx, hillslope_idx, n_steps=0u;
    __private float dl=0.0f, dt=DT_MAX, l_trajectory=0.0f;
    __private float2 uv1_vec, uv2_vec, dxy1_vec, dxy2_vec,
                     vec = seed_point_array[global_id], prev_vec, next_vec;

    // Remember here
    idx = get_array_idx(vec);
    hillslope_idx = idx;
    // Integrate downstream until a channel pixel (or masked pixel) is reached
    while (((~mapping_array[idx])&IS_THINCHANNEL) && !mask_array[idx]
           && n_steps<(MAX_N_STEPS-1)) {
        compute_step_vec(dt, uv_array, &dxy1_vec, &dxy2_vec, &uv1_vec, &uv2_vec,
                         vec, &next_vec, &idx);
        if (lengths_runge_kutta_step(&dt, &dl, &l_trajectory, &dxy1_vec, &dxy2_vec,
                                     &vec, &prev_vec,  &next_vec, &n_steps, &idx)) {
            break;
        }
    }
    if (mapping_array[idx]&IS_THINCHANNEL) {
        // We've reached a (thin) channel, so save this trajectory length
        // No need for atomic here since we're writing to the source pixel
        traj_length_array[global_id] = l_trajectory;
    }
    return;
}
#endif
