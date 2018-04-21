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
/// Compute the raw (unnormalized) histogram of an slx-type array.
///
/// Compiled if KERNEL_HISTOGRAM_UNIVARIATE is defined.
///
/// @param[in]     sl_array          (float *, RO):
/// @param[in,out] histogram_array   (uint *,  RW):
///
/// @returns void
///
/// @ingroup kde
///
__kernel void histogram_univariate(
        __global const    float  *sl_array,
        __global          uint   *histogram_array
   )
{
    // For every ROI pixel...

    const uint global_id = get_global_id(0u)+get_global_id(1u)*get_global_size(0u);
    const float bin_x = (sl_array[global_id]-X_MIN)/X_RANGE;
    const uint idx = min(max(0u,(uint)(bin_x*(float)N_HIST_BINS)),N_HIST_BINS-1u);
    atomic_inc(&histogram_array[idx]);
    return;
}
#endif


#ifdef KERNEL_PDF_UNIVARIATE
///
/// Use kernel-density estimation to map a histogram into a smooth pdf (scaled uint).
///
/// Compiled if KERNEL_PDF_UNIVARIATE is defined.
///
/// @param[in]     histogram_array (uint *,  RO):
/// @param[in,out] pdf_array       (uint *,  RW):
///
/// @returns void
///
/// @ingroup kde
///
__kernel void pdf_univariate(
        __global const  uint    *histogram_array,
        __global        uint    *pdf_array
   )
{
    // For every histogram bin...

    const uint global_id = get_global_id(0u)+get_global_id(1u)*get_global_size(0u);
    return;
}
#endif
