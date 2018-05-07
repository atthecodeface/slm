///
/// @file segment.cl
///
/// Kernels to (sub)segment landscape into smallish patches from channels to ridges
///
/// @author CPS
/// @bug No known bugs
///

///
/// @defgroup segmentation Channel and hillslope segmentation
/// Segment and subsegment hillslopes and adjacent channels into smallish zones
///

#ifdef KERNEL_SEGMENT_DOWNCHANNELS
///
/// TBD
///
/// Compiled if KERNEL_SEGMENT_DOWNCHANNELS is defined.
///
/// @param[in]     seed_point_array: list of initial streamline point vectors,
///                                  one allotted to each kernel instance
/// @param[in]     mask_array: grid pixel mask (padded),
///                            with @p true = masked, @p false = good
/// @param[in]     uv_array: flow unit velocity vector grid (padded)
/// @param[in]     mapping_array: flag grid recording status of each pixel (padded)
/// @param[in]     count_array: counter grid recording number of pixel steps
///                             downstream from dominant channel head (padded)
/// @param[in]     link_array: link grid providing the grid array index of the next
///                             downstream pixel (padded)
/// @param[in,out] label_array: label grid giving the ID of the subsegment to which
///                             this pixel belongs (padded); the MSB is set if left flank
///
/// @returns void
///
/// @ingroup segmentation
///
__kernel void segment_downchannels(
        __global const float2 *seed_point_array,
        __global const bool   *mask_array,
        __global const float2 *uv_array,
        __global const uint   *mapping_array,
        __global const uint   *count_array,
        __global const uint   *link_array,
        __global       uint   *label_array
   )
{
    // For every channel head pixel...

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
    __private uint idx, prev_idx, segment_label=0u,
                   segmentation_counter=SEGMENTATION_THRESHOLD;
    __private float2 vec = seed_point_array[global_id];

    // Remember here
    prev_idx = get_array_idx(vec);
    // Label this stream segment, starting with the head pixel
    segment_label = prev_idx;
    atomic_xchg(&label_array[prev_idx],segment_label);
    // Step downstream
    idx = link_array[prev_idx];
    // Continue stepping downstream until a dominant confluence
    //    or a masked pixel is reached
    while (!mask_array[idx] && prev_idx!=idx) {
        if ((mapping_array[idx]) & IS_MAJORCONFLUENCE) {
            if ((mapping_array[prev_idx]) & IS_MAJORINFLOW) {
                if (count_array[idx]>=segmentation_counter) {
                    segment_label = idx;
                    segmentation_counter += SEGMENTATION_THRESHOLD;
                }
            } else {
                break;
            }
        }
        // Label here with the current segment's label
        atomic_xchg(&label_array[idx],segment_label);
        // Continue downstream
        prev_idx = idx;
        idx = link_array[idx];
    }
    return;
}
#endif

#ifdef KERNEL_SEGMENT_HILLSLOPES
///
/// TBD
///
/// Compiled if KERNEL_SEGMENT_HILLSLOPES is defined.
///
/// @param[in]     seed_point_array: list of initial streamline point vectors,
///                                  one allotted to each kernel instance
/// @param[in]     mask_array: grid pixel mask (padded),
///                            with @p true = masked, @p false = good
/// @param[in]     uv_array: flow unit velocity vector grid (padded)
/// @param[in]     mapping_array: flag grid recording status of each pixel (padded)
/// @param[in]     count_array: counter grid recording number of pixel steps
///                             downstream from dominant channel head (padded)
/// @param[in]     link_array: link grid providing the grid array index of the next
///                             downstream pixel (padded)
/// @param[in,out] label_array: label grid giving the ID of the subsegment to which
///                             this pixel belongs (padded); the MSB is set if left flank
///
/// @returns void
///
/// @ingroup segmentation
///
__kernel void segment_hillslopes(
        __global const float2 *seed_point_array,
        __global const bool   *mask_array,
        __global const float2 *uv_array,
        __global const uint   *mapping_array,
        __global const uint   *count_array,
        __global const uint   *link_array,
        __global       uint   *label_array
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
    __private float dl=0.0f, dt=DT_MAX;
    __private float2 uv1_vec, uv2_vec, dxy1_vec, dxy2_vec,
                     vec = seed_point_array[global_id], next_vec;

    // Remember here
    idx = get_array_idx(vec);
    hillslope_idx = idx;
    // Integrate downstream until a channel pixel (or masked pixel) is reached
    while (!mask_array[idx] && ((~mapping_array[idx])&IS_THINCHANNEL)
           && n_steps<(MAX_N_STEPS-1)) {
        compute_step_vec(dt, uv_array, &dxy1_vec, &dxy2_vec, &uv1_vec, &uv2_vec,
                         vec, &next_vec, &idx);
        if (segment_runge_kutta_step(&dt, &dl, &dxy1_vec, &dxy2_vec,
                                     &vec, &next_vec, &n_steps, &idx))
            break;
    }
    if (mapping_array[idx]&IS_THINCHANNEL) {
        // We've reached a (thin) channel, so grab its label and apply it to
        //   the source hillslope pixel
        atomic_xchg(&label_array[hillslope_idx],label_array[idx]);
    }
    return;
}
#endif

#ifdef KERNEL_SUBSEGMENT_CHANNEL_EDGES
///
/// TBD
///
/// Compiled if KERNEL_SUBSEGMENT_CHANNEL_EDGES is defined.
///
/// @param[in]     seed_point_array: list of initial streamline point vectors,
///                                  one allotted to each kernel instance
/// @param[in]     mask_array: grid pixel mask (padded),
///                            with @p true = masked, @p false = good
/// @param[in]     uv_array: flow unit velocity vector grid (padded)
/// @param[in,out] mapping_array: flag grid recording status of each pixel (padded)
/// @param[in]     channel_label_array: copy of the label grid array
/// @param[in]     link_array: link grid providing the grid array index of the next
///                            downstream pixel (padded)
/// @param[in,out] label_array: label grid giving the ID of the subsegment to which
///                             this pixel belongs (padded); the MSB is set if left flank
///
/// @returns void
///
/// @ingroup segmentation
///
__kernel void subsegment_channel_edges(
        __global const float2 *seed_point_array,
        __global const bool   *mask_array,
        __global const float2 *uv_array,
        __global       uint   *mapping_array,
        __global const uint   *channel_label_array,
        __global const uint   *link_array,
        __global       uint   *label_array
   )
{
    // For every channel head and major confluence pixel...

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
    __private uint idx, prev_idx, left_idx, segment_label=0u, prev_x,prev_y, x,y, n_turns;
    __private char dx,dy, rotated_dx,rotated_dy;
    __private float2 vec = seed_point_array[global_id];

    // Remember here
    prev_idx = get_array_idx(vec);
    segment_label = channel_label_array[prev_idx];
    // Step downstream off channel head / major confluence pixel
    idx = link_array[prev_idx];
    // Even if this pixel is masked, we still need to try to subsegment
    while (prev_idx!=idx) {

        prev_x = prev_idx/NY_PADDED;
        prev_y = prev_idx%NY_PADDED;
        x =  idx/NY_PADDED;
        y =  idx%NY_PADDED;
        dx = (char)(x-prev_x);
        dy = (char)(y-prev_y);

        // Rotate 45deg anticlockwise repeatedly until a non-fillable pixel is reached
        n_turns = 0;
        while (left_idx!=prev_idx && ++n_turns<=4) {
            rotated_dx = dx-dy;
            rotated_dy = dx+dy;
            rotated_dx = rotated_dx/clamp((char)abs(rotated_dx),(char)1,(char)2);
            rotated_dy = rotated_dy/clamp((char)abs(rotated_dy),(char)1,(char)2);
            dx = rotated_dx;
            dy = rotated_dy;
            left_idx = (prev_x+rotated_dx)*NY_PADDED + (prev_y+rotated_dy);
            if (!mask_array[left_idx] && label_array[left_idx]==segment_label) {
                if ((mapping_array[left_idx]) & IS_THINCHANNEL) {
                    if (n_turns>1) {
                        break;
                    }
                } else {
                    atomic_or(&mapping_array[left_idx],IS_LEFTFLANK);
                    atomic_or(&label_array[left_idx],LEFT_FLANK_ADDITION);
                }
            }
        }

        // Step further downstream if necessary
        prev_idx = idx;
        idx = link_array[idx];
        // Stop if we've reached the next major confluence pixel, or the mask
        if (!mask_array[idx] && (   ((mapping_array[prev_idx]) & IS_MAJORCONFLUENCE)
                                 || ((mapping_array[prev_idx]) & IS_MAJORINFLOW)
                                 || ((mapping_array[prev_idx]) & IS_MINORINFLOW)
                                 ) ) {
            break;
        }
    }
    return;
}
#endif

#ifdef KERNEL_SUBSEGMENT_FLANKS
///
/// TBD
///
/// Compiled if KERNEL_SUBSEGMENT_FLANKS is defined.
///
/// @param[in]  seed_point_array: list of initial streamline point vectors,
///                               one allotted to each kernel instance
/// @param[in]  mask_array: grid pixel mask (padded),
///                         with @p true = masked, @p false = good
/// @param[in]  uv_array: flow unit velocity vector grid (padded)
/// @param[in,out] mapping_array: flag grid recording status of each pixel (padded)
/// @param[in,out] count_array: counter grid recording number of pixel steps
///                             downstream from dominant channel head (padded)
/// @param[in,out] link_array: link grid providing the grid array index of the next
///                             downstream pixel (padded)
/// @param[in,out] label_array: label grid giving the ID of the subsegment to which
///                             this pixel belongs (padded); the MSB is set if left flank
///
/// @returns void
///
/// @ingroup segmentation
///
__kernel void subsegment_flanks(
        __global const float2 *seed_point_array,
        __global const bool   *mask_array,
        __global const float2 *uv_array,
        __global const uint   *mapping_array,
        __global const uint   *count_array,
        __global const uint   *link_array,
        __global       uint   *label_array
   )
{
    // For every non-left-flank hillslope pixel...

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
    __private float dl=0.0f, dt=DT_MAX;
    __private float2 uv1_vec, uv2_vec, dxy1_vec, dxy2_vec,
                     vec = seed_point_array[global_id], next_vec;

    // Remember here
    idx = get_array_idx(vec);
    hillslope_idx = idx;
    // Integrate downstream until thin channel or left-flank pixel is reached
    while (!mask_array[idx] && ((~mapping_array[idx])&IS_LEFTFLANK)
            && ((~mapping_array[idx])&IS_THINCHANNEL) && n_steps<(MAX_N_STEPS-1)) {
        compute_step_vec(dt, uv_array, &dxy1_vec, &dxy2_vec, &uv1_vec, &uv2_vec,
                         vec, &next_vec, &idx);
        if (segment_runge_kutta_step(&dt, &dl, &dxy1_vec, &dxy2_vec,
                                     &vec, &next_vec, &n_steps, &idx))
            break;
    }
    if (mapping_array[idx]&IS_LEFTFLANK) {
        // We've reached a (thin) channel, so grab its label and apply it to
        //   the source hillslope pixel
        // No need for atomic here since we're writing to the source pixel
        label_array[hillslope_idx] |= LEFT_FLANK_ADDITION;
//        atomic_or(&label_array[hillslope_idx],LEFT_FLANK_ADDITION);
    }
    return;
}
#endif
