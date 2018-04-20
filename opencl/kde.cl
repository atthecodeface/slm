///
/// @file kde.cl
///
/// Kernels to do (pdf) kernel density estimation
///
/// @author CPS
/// @bug No known bugs
///

///
/// @defgroup kde Kernel density estimation
/// Kernel density estimation
///

#ifdef KERNEL_HISTOGRAM_UNIVARIATE

///
/// TBD.
///
/// Compiled if KERNEL_HISTOGRAM_UNIVARIATE is defined.
///
/// @param[in]  sl_array             (uint *,   RO):
/// @param[in,out] histogram_array   (float *,  RW):
///
/// @returns void
///
/// @ingroup kde
///
__kernel void histogram_univariate(
        __global const float  *sl_array,
        __global       float  *histogram_array
   )
{
    // For every ROI pixel...

    const uint global_id = get_global_id(0u)+get_global_id(1u)*get_global_size(0u);
    const float bin = (sl_array[global_id]-X_MIN)/X_RANGE;
    const uint idx = min(max(0u,(uint)(bin*(float)N_BINS_X)),N_BINS_X-1u);
    atomic_inc(&histogram_array[idx]);
    return;
}
#endif
