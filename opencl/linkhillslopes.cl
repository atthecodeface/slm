///
/// @file linkhillslopes.cl
///
/// Kernels to (re)map thin channels, branching structure, and single outflow directions.
///
/// @author CPS
/// @bug No known bugs
///

#ifdef KERNEL_LINK_HILLSLOPES
///
/// TBD
///
/// Compiled if KERNEL_LINK_HILLSLOPES is defined.
///
/// @param[in]     seed_point_array: list of initial streamline point vectors,
///                                  one allotted to each kernel instance
/// @param[in]     mask_array: grid pixel mask (padded),
///                            with @p true = masked, @p false = good
/// @param[in]     uv_array: flow unit velocity vector grid (padded)
/// @param[in,out] mapping_array: flag grid recording status of each pixel (padded)
/// @param[in,out] count_array: counter grid recording number of pixel steps
///                             downstream from dominant channel head (padded)
/// @param[in,out] link_array: link grid providing the grid array index of the next
///                             downstream pixel (padded)
///
/// @returns void
///
/// @ingroup structure
///
__kernel void link_hillslopes(
        __global const float2 *seed_point_array,
        __global const bool   *mask_array,
        __global const float2 *uv_array,
        __global       uint   *mapping_array,
        __global const uint   *count_array,
        __global       uint   *link_array
   )
{
    const uint global_id = get_global_id(0u)+get_global_id(1u)*get_global_size(0u);
    // For every hillslope aka non-thin-channel pixel...
    if (global_id>=N_SEED_POINTS) {
        // This is a "padding" seed, so let's bail
#ifdef DEBUG
        printf("Bailing @ %d !in [%d-%d]\n",
                global_id,get_global_offset(0u),N_SEED_POINTS-1);
#endif
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
    __private uint idx, prev_idx, n_steps=0u;
    __private float dl=0.0f, dt=DT_MAX;
    __private float2 uv1_vec, uv2_vec, dxy1_vec, dxy2_vec,
                     vec = seed_point_array[global_id], next_vec;

    // Remember here
    idx = get_array_idx(vec);
    prev_idx = idx;
    // Integrate downstream one pixel
    // HACK: factor 100x
    while (prev_idx==idx && !mask_array[idx] && n_steps<100*(MAX_N_STEPS)) {
        prev_idx = idx;
        compute_step_vec(dt, uv_array, &dxy1_vec, &dxy2_vec, &uv1_vec, &uv2_vec,
                         vec, &next_vec, &idx);
        if (countlink_runge_kutta_step(&dt, &dl, &dxy1_vec, &dxy2_vec,
                                       &vec, &next_vec, &idx, mapping_array)) {
#ifdef DEBUG
            printf("Link hillslopes (%d): stuck @ %d ...bailing\n",global_id,idx);
#endif
            break;
        }
        n_steps++;
    }
    // If we're on a new pixel, link previous pixel to here
    if (prev_idx!=idx && !mask_array[idx]) {
        atomic_xchg(&link_array[prev_idx],idx);
    }
    return;
}
#endif
