///
/// @file countlink.cl
///
/// Kernels to (re)map thin channels, branching structure, and single outflow directions.
///
/// @author CPS
/// @bug No known bugs
///

#ifdef KERNEL_COUNT_DOWNCHANNELS
///
/// REVISE? Integrate downstream from all channel heads until either a masked boundary
/// pixel is reached or until a channel pixel with a non-zero count is reached.
/// At each new pixel step, link the previous pixel to the current pixel.
/// (Re)designate traversed pixels as 'thin channel' along the way.
///
/// Compiled if KERNEL_COUNT_DOWNCHANNELS is defined.
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
__kernel void count_downchannels(
        __global const float2 *seed_point_array,
        __global const bool   *mask_array,
        __global const float2 *uv_array,
        __global uint  *mapping_array,
        __global uint  *count_array,
        __global uint  *link_array
   )
{
    // For every channel head pixel...

    const uint global_id = get_global_id(0u)+get_global_id(1u)*get_global_size(0u);
    __private uint idx, prev_idx, n_steps=0u, counter=1u;
    __private float dl=0.0f, dt=DT_MAX;
    __private float2 uv1_vec, uv2_vec, dxy1_vec, dxy2_vec,
                     vec = seed_point_array[global_id], next_vec;

    // Remember here
    idx = get_array_idx(vec);
    prev_idx = idx;
    // Counter is already set=1 above
    atomic_xchg(&count_array[idx],counter++);
    atomic_or(&mapping_array[idx],IS_THINCHANNEL);
    // Integrate downstream until the masked boundary is reached or n_steps too big
    // HACK: factor 100x
    while (!mask_array[idx] && n_steps<100*(MAX_N_STEPS-1)) {
        compute_step_vec(dt, uv_array, &dxy1_vec, &dxy2_vec, &uv1_vec, &uv2_vec,
                         vec, &next_vec, &idx);
        if (countlink_runge_kutta_step(&dt, &dl, &dxy1_vec, &dxy2_vec,
                                       &vec, &next_vec, &idx, mapping_array)) {
            break;
        }
        n_steps++;
        // If at a new pixel
        if (prev_idx!=idx) {
            // Redesignate as a thin channel pixel
            atomic_or(&mapping_array[idx],IS_THINCHANNEL);
            // Link to here from the last pixel,
            // i.e., point the previous pixel to this its downstream neighbor
            atomic_xchg(&link_array[prev_idx],idx);
            // If we've landed on a pixel whose channel length count
            //    exceeds our counter, we must have stepped off a minor onto a major
            //    channel, and thus need to stop
            if (counter++<count_array[idx]) {
                break;
            }
            prev_idx = idx;
        }
    }
    if (!mask_array[prev_idx]) atomic_xchg(&link_array[prev_idx],idx);
    return;
}
#endif

#ifdef KERNEL_FLAG_DOWNCHANNELS
///
/// TBD.
///
/// Compiled if KERNEL_FLAG_DOWNCHANNELS is defined.
///
/// @param[in]      seed_point_array   (float2 *, RO):
/// @param[in]      mask_array         (bool *,   RO):
/// @param[in]      uv_array           (float2 *, RO):
/// @param[in/out]  mapping_array      (uint *,   RW):
/// @param[in/out]  count_array        (uint *,   RW):
/// @param[in/out]  link_array         (uint *,   RW):
///
/// @returns void
///
/// @ingroup structure
///
__kernel void flag_downchannels(
        __global const float2 *seed_point_array,
        __global const bool   *mask_array,
        __global const float2 *uv_array,
        __global uint  *mapping_array,
        __global uint  *count_array,
        __global uint  *link_array
   )
{
    // For every channel head pixel...

    const uint global_id = get_global_id(0u)+get_global_id(1u)*get_global_size(0u);
    __private uint idx, prev_idx, counter=1u;
    __private float2 vec = seed_point_array[global_id];

    // Remember here
    idx = get_array_idx(vec);
    prev_idx = idx+1;
    // Counter=1 at channel head (set by count_downchannels)
    //    atomic_xchg(&count_array[idx],counter++);
    atomic_or(&mapping_array[idx],IS_THINCHANNEL);
    // Integrate downstream until the masked boundary is reached
    while (!mask_array[idx] && prev_idx!=idx) {
        prev_idx = idx;
        idx = link_array[idx];
        // Assume this idx is on the grid?
        if (!mask_array[idx]) {
            atomic_or(&mapping_array[idx],IS_THINCHANNEL);
            atomic_max(&count_array[idx],counter++);
        } else {
            break;
        }
    }
    // We have just stepped onto a masked pixel, so let's tag the previous pixel
    //    as a channel tail
    if (!mask_array[prev_idx]) atomic_or(&mapping_array[prev_idx],IS_CHANNELTAIL);
    return;
}
#endif

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
        __global uint  *mapping_array,
        __global uint  *count_array,
        __global uint  *link_array
   )
{
    // For every hillslope aka non-thin-channel pixel...

    const uint global_id = get_global_id(0u)+get_global_id(1u)*get_global_size(0u);
    __private uint idx, prev_idx, n_steps=0u;
    __private float dl=0.0f, dt=DT_MAX;
    __private float2 uv1_vec, uv2_vec, dxy1_vec, dxy2_vec,
                     vec = seed_point_array[global_id], next_vec;

    // Remember here
    idx = get_array_idx(vec);
    prev_idx = idx;
    // Integrate downstream one pixel
    // HACK: factor 100x
    while (prev_idx==idx && !mask_array[idx] && n_steps<100*(MAX_N_STEPS-1)) {
        prev_idx = idx;
        compute_step_vec(dt, uv_array, &dxy1_vec, &dxy2_vec, &uv1_vec, &uv2_vec,
                         vec, &next_vec, &idx);
        if (countlink_runge_kutta_step(&dt, &dl, &dxy1_vec, &dxy2_vec,
                                       &vec, &next_vec, &idx, mapping_array)) {
            printf("stuck\n");
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
