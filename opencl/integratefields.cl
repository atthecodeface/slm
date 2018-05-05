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

///
/// Byte reversal
///
/// Macro to perform byte reversal per GJS's suggestion.
///
/// @param[in,out] initial_rng_state: RNG state (thus initally the seed) and RNG variate
///
#define BYTE_REVERSAL(initial_rng_state) \
    initial_rng_state =   (initial_rng_state>>24)&0xff \
                        | (initial_rng_state>> 8)&0xff00 \
                        | (initial_rng_state<< 8)&0xff0000 \
                        | (initial_rng_state<<24)&0xff000000;

#ifdef KERNEL_INTEGRATE_FIELDS
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
/// Compiled if KERNEL_INTEGRATE_FIELDS is defined.
///
/// @param[in]  seed_point_array: list of initial streamline point vectors,
///                               one allotted to each kernel instance
/// @param[in]  mask_array: grid pixel mask (padded),
///                         with @p true = masked, @p false = good
/// @param[in]  uv_array: flow unit velocity vector grid (padded)
/// @param[out] mapping_array: multi-flag array
/// @param[out] slc_array: grid recording accumulated count of streamline integration
///                        steps across each pixel (padded)
/// @param[out] slt_array: grid recording accumulated count of streamline segment lengths
///                        crossing each pixel (padded)
///
/// @returns void
///
/// @ingroup integrate
///
__kernel void integrate_fields( __global const float2 *seed_point_array,
                                __global const bool   *mask_array,
                                __global const float2 *uv_array,
                                __global       uint   *mapping_array,
                                __global       uint   *slc_array,
                                __global       uint   *slt_array )
{
    // global_id plus the chunk SEEDS_CHUNK_OFFSET is a seed point index
    const uint global_id = get_global_id(0u)+get_global_id(1u)*get_global_size(0u),
               seed_idx = (SEEDS_CHUNK_OFFSET)+global_id;

//    // Report how kernel instances are distributed
//    if (seed_idx==0) {
//        printf("\nOn GPU/OpenCL device: #workitems=%d  #workgroups=%d => work size=%d\n",
//                get_local_size(0u), get_num_groups(0u),
//                get_local_size(0u)*get_num_groups(0u));
//    }
    // Report how kernel instances are distributed
    if (global_id==get_global_offset(0u)) {
        printf("\n   >>> on GPU/OpenCL device: #workitems=%d * #workgroups=%d \
= worksize=%d   global offset=%d\n",
                get_local_size(0u), get_num_groups(0u),
                get_local_size(0u)*get_num_groups(0u),
                get_global_offset(0u));
    }
    if (seed_idx>=N_SEED_POINTS) {
        // This is a "padding" seed, so let's bail
        return;
    }

    const float2 current_seed_point_vec = seed_point_array[seed_idx];
    __private uint i=0,j=0, initial_rng_state;
    const uint idx = get_array_idx(current_seed_point_vec);
    atomic_or(&mapping_array[idx],WAS_CHANNELHEAD);

    // Trace a set of streamlines from a grid of sub-pixel positions centered
    //    on the seed point
    // Generate an initial RNG state (aka 'seed the RNG')
    //   [was: using the sum of the current pixel index and the sub-pixel index]
    //   using the current pixel index ("seed_idx")
    //   byte-reversed per GJS suggestion
    initial_rng_state = seed_idx;
    BYTE_REVERSAL(initial_rng_state);
    lehmer_rand_uint(&initial_rng_state);
    BYTE_REVERSAL(initial_rng_state);


    for (j=0u;j<SUBPIXEL_SEED_POINT_DENSITY;j++) {
        for (i=0u;i<SUBPIXEL_SEED_POINT_DENSITY;i++){
            // Trace a jittered streamline from a sub-pixel-offset first point
            jittered_trajectory(uv_array, mask_array,
                                mapping_array, slc_array, slt_array,
                                global_id, seed_idx,
                                current_seed_point_vec + (float2)(
                                    (float)i*SUBPIXEL_SEED_STEP-SUBPIXEL_SEED_HALFSPAN,
                                    (float)j*SUBPIXEL_SEED_STEP-SUBPIXEL_SEED_HALFSPAN ),
                                initial_rng_state);
        }
    }
}
#endif
