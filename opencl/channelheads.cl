///
/// @file channelheads.cl
///
/// Kernels to map provisional channel heads and the prune those not on thin channels.
///
/// @author CPS
///

///
/// @defgroup structure Connectivity structure
/// Map channel connectivity such as channel heads, confluences, downstream pixels
///

#ifdef KERNEL_MAP_CHANNEL_HEADS
///
/// Map provisional channel heads, even including those not on an IS_THINCHANNEL pixel
///     and thus extraneous. The latter are removed by prune_channel_heads().
///
/// Compiled if KERNEL_MAP_CHANNEL_HEADS is defined.
///
/// @param[in]  seed_point_array: list of initial streamline point vectors,
///                               one allotted to each kernel instance
/// @param[in]  mask_array: grid pixel mask (padded),
///                         with @p true = masked, @p false = good
/// @param[in]  uv_array: flow unit velocity vector grid (padded)
/// @param[in,out] mapping_array: flag grid recording status of each pixel (padded)
///
/// @returns void
///
/// @ingroup structure
///
__kernel void map_channel_heads(
        __global const float2 *seed_point_array,
        __global const bool   *mask_array,
        __global const float2 *uv_array,
        __global       uint   *mapping_array
   )
{
    // For every non-masked pixel, all of which are temporarily flagged as
    //   channel heads, and most of which will be unflagged here

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
    __private uint idx, prev_idx, n_steps = 0u;
    __private float dl = 0.0f, dt = DT_MAX;
    __private float2 uv1_vec, uv2_vec, dxy1_vec, dxy2_vec,
                     vec = seed_point_array[global_id], next_vec;

    // Remember here
    idx = get_array_idx(vec);
//#ifdef DEBUG
//    printf("Map channel heads (%d) @ %d %g,%g\n",
//            global_id, idx,vec[0],vec[1]);
//#endif
    prev_idx = idx;
    // Integrate downstream one pixel
    while (prev_idx==idx && !mask_array[idx] && n_steps<MAX_N_STEPS) {
        compute_step_vec(dt, uv_array, &dxy1_vec, &dxy2_vec, &uv1_vec, &uv2_vec,
                         vec, &next_vec, &idx);
        channelheads_runge_kutta_step(&dt, &dl, &dxy1_vec, &dxy2_vec,
                                      &vec, &next_vec, &n_steps, &idx);
    }
    if (n_steps>=MAX_N_STEPS) {
        return;
    }
    // Unset the channel head flag unless we're at a provisional channel head
    if (!mask_array[idx]) {
        idx = get_array_idx(vec);
        if (   ((~mapping_array[idx]) & IS_THINCHANNEL)
            // If here is not channel...
            || (mapping_array[prev_idx] & IS_THINCHANNEL) ) {
            // However, if here is a channel, and if previous pixel was a channel...
            atomic_and(&mapping_array[idx],~IS_CHANNELHEAD);
            // ... flag here as not channel head
        }
    }
    return;
}
#endif

#ifdef KERNEL_PRUNE_CHANNEL_HEADS

// Check if this nbr is a thin channel pixel and not masked
// If so, add one to the 'flag'.
// Add 16 if it's masked, thus recording if *any* nbr is masked.
#define CHECK_IS_THINCHANNEL(idx) ((mapping_array[idx] & IS_THINCHANNEL)!=0)
//#define CHECK_ISNOT_CHANNELHEAD(idx) ((mapping_array[idx] & WAS_CHANNELHEAD)==0)
#define CHECK_IS_MASKED(idx) (mask_array[idx])
#define CHECK_THINCHANNEL(nbr_vec_x,nbr_vec_y) { \
           idx = get_array_idx((float2)(nbr_vec_x,nbr_vec_y)); \
           flag += ( (CHECK_IS_THINCHANNEL(idx) ) \
                     ); \
        }

// check for masked nbr - turned off
//| CHECK_IS_MASKED(idx)*16);

//*CHECK_ISNOT_CHANNELHEAD(idx)

// Check all eight pixel-nbr directions
#define CHECK_E(vec)  CHECK_THINCHANNEL( vec[0]+1.0f, vec[1]      )
#define CHECK_NE(vec) CHECK_THINCHANNEL( vec[0]+1.0f, vec[1]+1.0f )
#define CHECK_N(vec)  CHECK_THINCHANNEL( vec[0]     , vec[1]+1.0f )
#define CHECK_NW(vec) CHECK_THINCHANNEL( vec[0]-1.0f, vec[1]+1.0f )
#define CHECK_W(vec)  CHECK_THINCHANNEL( vec[0]-1.0f, vec[1]      )
#define CHECK_SW(vec) CHECK_THINCHANNEL( vec[0]-1.0f, vec[1]-1.0f )
#define CHECK_S(vec)  CHECK_THINCHANNEL( vec[0]     , vec[1]-1.0f )
#define CHECK_SE(vec) CHECK_THINCHANNEL( vec[0]+1.0f, vec[1]-1.0f )

///
/// Keep only those provisional channel heads that lie on the 'thin channel'
///    skeletonized network and have only one such thin channel pixel neighbor.
/// Also exclude any provisional channel head with any masked-pixel neighbors.
///
/// Compiled if KERNEL_PRUNE_CHANNEL_HEADS is defined.
///
/// @param[in]  seed_point_array: list of initial streamline point vectors,
///                               one allotted to each kernel instance
/// @param[in]  mask_array: grid pixel mask (padded),
///                         with @p true = masked, @p false = good
/// @param[in]  uv_array: flow unit velocity vector grid (padded)
/// @param[in,out] mapping_array: flag grid recording status of each pixel (padded)
///
/// @returns void
///
/// @ingroup structure
///
__kernel void prune_channel_heads(
        __global const float2 *seed_point_array,
        __global const bool   *mask_array,
        __global const float2 *uv_array,
        __global       uint   *mapping_array
   )
{
    // For every provisional channel head pixel...

    const uint global_id = get_global_id(0u)+get_global_id(1u)*get_global_size(0u);
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
    __private uint idx;
    __private uint flag=0u;
    const float2 vec = seed_point_array[global_id];
    // Scan all 8 next/nearest neighbors:
    //   - add 1 to flag if the nbr is a thin channel pixel
    //   - add 16 if the nbr is masked (pathological case: 8*16=128)
    CHECK_N(vec);
    CHECK_S(vec);
    CHECK_E(vec);
    CHECK_W(vec);
    CHECK_NE(vec);
    CHECK_SE(vec);
    CHECK_NW(vec);
    CHECK_SW(vec);
    // If flag==1, one and only one nbr is a thin channel pixel
    // Otherwise, remove this provisional channel head.
    if (flag!=1u) {
        idx = get_array_idx(vec);
        atomic_and(&mapping_array[idx],~IS_CHANNELHEAD);
        // If there are no thin channel neighbors AT ALL we must be at an isolated pixel.
        // Thus redesignate this pixel as 'not channelized at all'.
        if (flag==0) { // || flag>=16) {
            atomic_and(&mapping_array[idx], ~(IS_THINCHANNEL | IS_CHANNEL));
        }
    }
    return;
}
#endif
