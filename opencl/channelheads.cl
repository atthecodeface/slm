///
/// @file channelheads.cl
///
/// Kernels to map provisional channel heads and the prune those not on thin channels.
///
/// @author CPS
/// @bug No known bugs
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
    // For every non-masked pixel...

    const uint global_id = get_global_id(0u)+get_global_id(1u)*get_global_size(0u);
    // Report how kernel instances are distributed
    if (global_id==get_global_offset(0u)) {
        printf("\n   >>> on GPU/OpenCL device: #workitems=%d  #workgroups=%d \
=> work size=%d   global offset=%d\n",
                get_local_size(0u), get_num_groups(0u),
                get_local_size(0u)*get_num_groups(0u),
                get_global_offset(0u));
    }
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
    prev_idx = idx;
    // Integrate downstream one pixel
    while (prev_idx==idx && !mask_array[idx] && n_steps<(MAX_N_STEPS-1)) {
        compute_step_vec(dt, uv_array, &dxy1_vec, &dxy2_vec, &uv1_vec, &uv2_vec,
                         vec, &next_vec, &idx);
        channelheads_runge_kutta_step(&dt, &dl, &dxy1_vec, &dxy2_vec,
                                      &vec, &next_vec, &n_steps, &idx);
    }
    // If need be, integrate further downstream until a IS_THINCHANNEL pixel is reached
    n_steps = 0u;
    while (!mask_array[idx] && !(mapping_array[idx] & IS_THINCHANNEL)
           && n_steps<(MAX_N_STEPS-1)) {
        compute_step_vec(dt, uv_array, &dxy1_vec, &dxy2_vec, &uv1_vec, &uv2_vec,
                         vec, &next_vec, &idx);
        channelheads_runge_kutta_step(&dt, &dl, &dxy1_vec, &dxy2_vec,
                                      &vec, &next_vec, &n_steps, &idx);
    }
    if (n_steps>=(MAX_N_STEPS-1)) {
//        printf("stuck...");
        return;
    }
    // Unset the channel head flag unless we're really at a channel head maybe
    if (!mask_array[idx]) {
        idx = get_array_idx(vec);
        // If here is not channel...
        if ((~mapping_array[idx]) & IS_THINCHANNEL){
            // ...flag here as not channel head
            atomic_and(&mapping_array[idx],~IS_CHANNELHEAD);
        } else {
            // Here is a channel
            // If previous pixel was channel...
            if (mapping_array[prev_idx] & IS_THINCHANNEL) {
                // ...flag here as not channel head
                atomic_and(&mapping_array[idx],~IS_CHANNELHEAD);
            }
        }
    }
    return;
}
#endif

#ifdef KERNEL_PRUNE_CHANNEL_HEADS

// Check if this nbr is a thin channel pixel and not masked
// If so, add one to the 'flag'.
// Add 16 if it's masked, thus recording if *any* nbr is masked.
#define CHECK_IS_THINCHANNEL(idx) ((mapping_array[idx] & IS_THINCHANNEL)>0)
#define CHECK_IS_MASKED(idx) (mask_array[idx])
#define CHECK_THINCHANNEL(nbr_vec) { \
           idx = get_array_idx(nbr_vec); \
           flag += (CHECK_IS_THINCHANNEL(idx) | CHECK_IS_MASKED(idx)*16); \
        }
// Check all eight pixel-nbr directions
#define CHECK_E(vec)  CHECK_THINCHANNEL((float2)( vec[0]+1.0f, vec[1]      ))
#define CHECK_NE(vec) CHECK_THINCHANNEL((float2)( vec[0]+1.0f, vec[1]+1.0f ))
#define CHECK_N(vec)  CHECK_THINCHANNEL((float2)( vec[0]     , vec[1]+1.0f ))
#define CHECK_NW(vec) CHECK_THINCHANNEL((float2)( vec[0]-1.0f, vec[1]+1.0f ))
#define CHECK_W(vec)  CHECK_THINCHANNEL((float2)( vec[0]-1.0f, vec[1]      ))
#define CHECK_SW(vec) CHECK_THINCHANNEL((float2)( vec[0]-1.0f, vec[1]-1.0f ))
#define CHECK_S(vec)  CHECK_THINCHANNEL((float2)( vec[0]     , vec[1]-1.0f ))
#define CHECK_SE(vec) CHECK_THINCHANNEL((float2)( vec[0]+1.0f, vec[1]-1.0f ))

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
//    if (global_id>=N_SEED_POINTS) {
//        // This is a "padding" seed, so let's bail
//#ifdef DEBUG
//        printf("Bailing @ %d !in [%d-%d]\n",
//                global_id,get_global_offset(0u),N_SEED_POINTS-1);
//#endif
//        return;
//    }
    // Report how kernel instances are distributed
    if (global_id==get_global_offset(0u)) {
        printf("\n   >>> on GPU/OpenCL device: #workitems=%d  #workgroups=%d \
=> work size=%d   global offset=%d\n",
                get_local_size(0u), get_num_groups(0u),
                get_local_size(0u)*get_num_groups(0u),
                get_global_offset(0u));
    }
    __private uint idx;
    __private uint flag = 0;
    const float2 vec = seed_point_array[global_id];
#ifdef DEBUG
    if (1) {
        printf("*#*#*#*#*#*#*#    %d %d,%d@ %g,%g\n",
                global_id, get_global_size(0u), get_global_size(1u),
                vec[0]*2.0f,vec[1]*2.0f); //427,1227
    }
#endif
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
    if (flag!=1) {
        idx = get_array_idx(vec);
//        atomic_and(&mapping_array[idx],~IS_CHANNELHEAD);
        atomic_or(&mapping_array[idx],WAS_CHANNELHEAD);
        // If there are no thin channel neighbors AT ALL,
        //   we must be at an isolated pixel.
        // Thus redesignate this pixel as 'not channelized at all'.
        if (flag==0 || flag>=16) {
            atomic_and(&mapping_array[idx], ~(IS_THINCHANNEL | IS_CHANNEL));
        }
    }
    return;
}
#endif
