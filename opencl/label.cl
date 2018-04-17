///
/// @file label.cl
///
/// Kernel to map channel confluences and identify their major & minor upstream pixels.
///
/// @author CPS
/// @bug No known bugs
///

#ifdef KERNEL_LABEL_CONFLUENCES

#define CHECK_INFLOWS(here_idx,nbr_vec) { \
   nbr_idx = get_array_idx(nbr_vec); \
   if ( !mask_array[nbr_idx] && (mapping_array[nbr_idx]&IS_THINCHANNEL) ) { \
       if ( link_array[nbr_idx]==here_idx ) { \
           /* The nbr pixel flows into here */ \
           inflows_list[n_inflows++] = nbr_idx; \
       } \
   } \
}
// Check in all 8 pixel-nbr directions
#define CHECK_E_INFLOWS(idx,vec)  CHECK_INFLOWS(idx,(float2)(vec[0]+1.0f, vec[1]      ))
#define CHECK_NE_INFLOWS(idx,vec) CHECK_INFLOWS(idx,(float2)(vec[0]+1.0f, vec[1]+1.0f ))
#define CHECK_N_INFLOWS(idx,vec)  CHECK_INFLOWS(idx,(float2)(vec[0],      vec[1]+1.0f ))
#define CHECK_NW_INFLOWS(idx,vec) CHECK_INFLOWS(idx,(float2)(vec[0]-1.0f, vec[1]+1.0f ))
#define CHECK_W_INFLOWS(idx,vec)  CHECK_INFLOWS(idx,(float2)(vec[0]-1.0f, vec[1]      ))
#define CHECK_SW_INFLOWS(idx,vec) CHECK_INFLOWS(idx,(float2)(vec[0]-1.0f, vec[1]-1.0f ))
#define CHECK_S_INFLOWS(idx,vec)  CHECK_INFLOWS(idx,(float2)(vec[0],      vec[1]-1.0f ))
#define CHECK_SE_INFLOWS(idx,vec) CHECK_INFLOWS(idx,(float2)(vec[0]+1.0f, vec[1]-1.0f ))

///
/// Flag if a pixel IS_MAJORCONFLUENCE and if so flag which upstream pixel IS_MAJORINFLOW
///   or IS_MINORINFLOW.
///
/// Compiled if KERNEL_LABEL_CONFLUENCES is defined.
///
/// @param[in]  seed_point_array       (float2 *, RO):
/// @param[in]  mask_array             (bool *,   RO):
/// @param[in]  uv_array               (float2 *, RO):
/// @param[in]  slt_array              (uint *,   RO):
/// @param[in,out]  mapping_array      (uint *,   RW):
/// @param[in]  count_array            (uint *,   RO):
/// @param[in]  link_array             (uint *,   RO):
///
/// @returns void
///
/// @ingroup kernels structure
///
__kernel void label_confluences(
        __global const float2 *seed_point_array,
        __global const bool   *mask_array,
        __global const float2 *uv_array,
        __global const float  *slt_array,
        __global       uint   *mapping_array,
        __global const uint   *count_array,
        __global const uint   *link_array
   )
{
    // For every (redesignated) thin channel pixel...

    const uint global_id = get_global_id(0u)+get_global_id(1u)*get_global_size(0u);
    __private uchar n_inflows=0, n_equal_dominant_inflows=0;
    __private uint i, idx, nbr_idx, inflows_list[7], equal_dominant_inflows_list[7],
                   dominant_slt_index=0;
    __private float dominant_slt=-MAXFLOAT;
    __private float2 vec=seed_point_array[global_id];

    // Remember here
    idx = get_array_idx(vec);
    // Check upstream neighbors
    CHECK_N_INFLOWS(idx,vec);
    CHECK_S_INFLOWS(idx,vec);
    CHECK_E_INFLOWS(idx,vec);
    CHECK_W_INFLOWS(idx,vec);
    CHECK_NE_INFLOWS(idx,vec);
    CHECK_SE_INFLOWS(idx,vec);
    CHECK_NW_INFLOWS(idx,vec);
    CHECK_SW_INFLOWS(idx,vec);
    if (n_inflows>1) {
        atomic_or(&mapping_array[idx],IS_MAJORCONFLUENCE);
        for (i=0;i<n_inflows;i++) {
            if ( (count_array[inflows_list[i]]+1)!=count_array[idx] ) {
                atomic_or(&mapping_array[inflows_list[i]],IS_MINORINFLOW);
            } else {
                equal_dominant_inflows_list[n_equal_dominant_inflows++]= inflows_list[i];
                if (slt_array[inflows_list[i]]>dominant_slt) {
                    dominant_slt_index = inflows_list[i];
                    dominant_slt = slt_array[dominant_slt_index];
//                    printf("di=%d\n",dominant_slt_index);
                }
            }
        }
        if (n_equal_dominant_inflows==0) {
            printf("n_equal_dominant_inflows=0 @ %g,%g\n",vec[0]*2,vec[1]*2);
            for (i=0;i<n_inflows;i++) {
                printf("%d=>  %d : %d\n",inflows_list[i],
                        count_array[inflows_list[i]]+1, count_array[idx]);
            }
            for (i=0;i<n_inflows;i++) {
                printf("%g : %g\n",slt_array[inflows_list[i]]+1, dominant_slt);
            }
        }
        for (i=0;i<n_equal_dominant_inflows;i++) {
            if (equal_dominant_inflows_list[i]==dominant_slt_index) {
                atomic_or(&mapping_array[equal_dominant_inflows_list[i]],IS_MAJORINFLOW);
            } else {
                atomic_or(&mapping_array[equal_dominant_inflows_list[i]],IS_MINORINFLOW);
            }
        }
    }
    return;
}
#endif
