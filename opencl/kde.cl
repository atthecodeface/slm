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


// None of the filter kernels below trap for bounds exceedance |x|>w.
// Shouldn't be necessary, but I need to check it's really safe.

#ifdef KDF_IS_TOPHAT
static inline double kdf_sample(const double x, const double w) {
    return 0.5f;
}
#endif

#ifdef KDF_IS_TRIANGLE
static inline double kdf_sample(const double x, const double w) {
    return (1.0f-fabs(x/w));
}
#endif

#ifdef KDF_IS_EPANECHNIKOV
static inline double kdf_sample(const double x, const double w) {
    return 0.75f*(1.0f-(x/w)*(x/w));
}
#endif

#ifdef KDF_IS_COSINE
static inline double kdf_sample(const double x, const double w) {
    return M_PI_4*cos((M_PI_2*x)/w);
}
#endif

#ifdef KDF_IS_GAUSSIAN
// This is a hack: needs to be rescaled in w and renormalized given the clipping
static inline double kdf_sample(const double x, const double w) {
    return M_SQRT1_2*exp(-0.5f*(x/w)*(x/w));
}
#endif



#ifdef KERNEL_HISTOGRAM_UNIVARIATE
///
/// Compute the raw (unnormalized) histogram of an sl-type array data vector.
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

#ifdef KERNEL_HISTOGRAM_BIVARIATE
///
/// Compute the raw (unnormalized) histogram of a joint slx,sly data-pair vector.
///
/// Compiled if KERNEL_HISTOGRAM_BIVARIATE is defined.
///
/// @param[in]     slx_array         (float *, RO):
/// @param[in]     sly_array         (float *, RO):
/// @param[in,out] histogram_array   (uint *,  RW):
///
/// @returns void
///
/// @ingroup kde
///
__kernel void histogram_bivariate(
        __global const    float  *slx_array,
        __global const    float  *sly_array,
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


#ifdef KERNEL_PDF_BIVARIATE
///
/// Use kernel-density estimation to map a histogram into a smooth pdf.
///
/// Compiled if KERNEL_PDF_BIVARIATE is defined.
///
/// @param[in]     histogram_array (uint *,  RO):
/// @param[in]     kdfilter_array  (float *, RO):
/// @param[in,out] pdf_array       (float *, RW):
///
/// @returns void
///
/// @ingroup kde
///
__kernel void pdf_bivariate(
        __global const  uint    *histogram_array,
        __global const  float   *kdf_array,
        __global        float   *pdf_array
   )
{
    // For every histogram bin...

    const uint global_id = get_global_id(0u)+get_global_id(1u)*get_global_size(0u);

    // Get this pdf point's index
    const uint pdf_idx = global_id;

    // Create a buffer variable for the pdf point "bin" accumulator
    __private double pdf_bin_accumulator=0.0f;
    // Create a buffer variable for the kdf-integral accumulator
    __private double kdf_weight_accumulator=0.0f;

    // Set the number of histogram bins per pdf point "bin"
    //   e.g., 2000 hist bins, 200 pdf point bins, 10 hist bins per pdf point bin
    const uint n_bins_per_point = N_HIST_BINS/N_PDF_POINTS;

    __private double kdf_bin_point, kdf_bin_edge;
    __private int lpki;
    __private uint ki, hi, rpki;
    __private double kdf_value, x;

    // Step through half kdf taking advantage of symmetry
    //   e.g., 0...2  for 5-point kdf
    for (ki=0;ki<=N_KDF_PART_POINTS_X;ki++) {
        // Center of kdf bin
        kdf_bin_point = ((double)ki)*PDF_DX;
        // Left edge of kdf bin
        kdf_bin_edge = kdf_bin_point-PDF_DX/2.0f;
        // First left kdf-sampled pdf bin index
        rpki = pdf_idx+ki;
        lpki = pdf_idx-ki+1;
        // Step through span of histogram bins
        for (hi=0u;hi<n_bins_per_point;hi++) {
            // Center of histogram bin
            x = kdf_bin_edge+BIN_DX*((double)hi+0.5f);
            // Calculate the kdf (kernel density filter) weight for this pdf point
            kdf_value = kdf_sample(x,KDF_WIDTH_X);
            // Get histogram count for this hist bin
            //    and add to the pdf point "bin" accumulator
            pdf_bin_accumulator
                += select((double)0.0f,
                          (double)histogram_array[rpki*n_bins_per_point+hi]*kdf_value,
                          (unsigned long)isless(rpki,N_PDF_POINTS));
            pdf_bin_accumulator
                += select((double)0.0f,
                          (double)histogram_array[lpki*n_bins_per_point-1-hi]*kdf_value,
                          (unsigned long)(isnotequal(ki,0) & isgreater(lpki,0)) );
            // Add the kdf weight to the kdf-integral accumulator
            kdf_weight_accumulator
                += select((double)0.0f, (double)kdf_value,
                          (unsigned long)isless(rpki,N_PDF_POINTS));
            kdf_weight_accumulator
                += select((double)0.0f, (double)kdf_value,
                          (unsigned long)(isnotequal(ki,0) & isgreater(lpki,0)) );
        }
//        if (pdf_idx==100) {
//            printf("ki=%d, rpki=%d, lpki=%d kdfbp=%g kdfbe=%g  %d,%d  %d,%d\n",
//                ki,rpki,lpki,kdf_bin_point, kdf_bin_edge,
//                rpki*n_bins_per_point+0,rpki*n_bins_per_point+n_bins_per_point-1,
//                lpki*n_bins_per_point-1-0,lpki*n_bins_per_point-1-(n_bins_per_point-1));
//            printf("x=%g  kdv=%g   %g, %g\n", x, kdf_value,
//                    pdf_bin_accumulator, kdf_weight_accumulator);
//        }
    }
    // Normalize the pdf point accumlation by dividing by the kdf weight accumulation
    //   and write to the return pdf_array element for this CL-kernel instance
    pdf_array[pdf_idx] = pdf_bin_accumulator/kdf_weight_accumulator;
//    printf("%g / %g = %g\n",
//            pdf_bin_accumulator,kdf_weight_accumulator,pdf_array[pdf_idx]);
    return;
}
#endif

#ifdef KERNEL_PDF_UNIVARIATE
///
/// Use kernel-density estimation to map a histogram into a smooth pdf.
///
/// Compiled if KERNEL_PDF_UNIVARIATE is defined.
///
/// @param[in]     histogram_array (uint *,  RO):
/// @param[in]     kdfilter_array  (float *, RO):
/// @param[in,out] pdf_array       (float *, RW):
///
/// @returns void
///
/// @ingroup kde
///
__kernel void pdf_univariate(
        __global const  uint    *histogram_array,
        __global const  float   *kdf_array,
        __global        float   *pdf_array
   )
{
    // For every histogram bin...

    const uint global_id = get_global_id(0u)+get_global_id(1u)*get_global_size(0u);

    // Get this pdf point's index
    const uint pdf_idx = global_id;

    // Create a buffer variable for the pdf point "bin" accumulator
    __private double pdf_bin_accumulator=0.0f;
    // Create a buffer variable for the kdf-integral accumulator
    __private double kdf_weight_accumulator=0.0f;

    // Set the number of histogram bins per pdf point "bin"
    //   e.g., 2000 hist bins, 200 pdf point bins, 10 hist bins per pdf point bin
    const uint n_bins_per_point = N_HIST_BINS/N_PDF_POINTS;

    __private double kdf_bin_point, kdf_bin_edge;
    __private int lpki;
    __private uint ki, hi, rpki;
    __private double kdf_value, x;

    // Step through half kdf taking advantage of symmetry
    //   e.g., 0...2  for 5-point kdf
    for (ki=0;ki<=N_KDF_PART_POINTS_X;ki++) {
        // Center of kdf bin
        kdf_bin_point = ((double)ki)*PDF_DX;
        // Left edge of kdf bin
        kdf_bin_edge = kdf_bin_point-PDF_DX/2.0f;
        // First left kdf-sampled pdf bin index
        rpki = pdf_idx+ki;
        lpki = pdf_idx-ki+1;
        // Step through span of histogram bins
        for (hi=0u;hi<n_bins_per_point;hi++) {
            // Center of histogram bin
            x = kdf_bin_edge+BIN_DX*((double)hi+0.5f);
            // Calculate the kdf (kernel density filter) weight for this pdf point
            kdf_value = kdf_sample(x,KDF_WIDTH_X);
            // Get histogram count for this hist bin
            //    and add to the pdf point "bin" accumulator
            pdf_bin_accumulator
                += select((double)0.0f,
                          (double)histogram_array[rpki*n_bins_per_point+hi]*kdf_value,
                          (unsigned long)isless(rpki,N_PDF_POINTS));
            pdf_bin_accumulator
                += select((double)0.0f,
                          (double)histogram_array[lpki*n_bins_per_point-1-hi]*kdf_value,
                          (unsigned long)(isnotequal(ki,0) & isgreater(lpki,0)) );
            // Add the kdf weight to the kdf-integral accumulator
            kdf_weight_accumulator
                += select((double)0.0f, (double)kdf_value,
                          (unsigned long)isless(rpki,N_PDF_POINTS));
            kdf_weight_accumulator
                += select((double)0.0f, (double)kdf_value,
                          (unsigned long)(isnotequal(ki,0) & isgreater(lpki,0)) );
        }
//        if (pdf_idx==100) {
//            printf("ki=%d, rpki=%d, lpki=%d kdfbp=%g kdfbe=%g  %d,%d  %d,%d\n",
//                ki,rpki,lpki,kdf_bin_point, kdf_bin_edge,
//                rpki*n_bins_per_point+0,rpki*n_bins_per_point+n_bins_per_point-1,
//                lpki*n_bins_per_point-1-0,lpki*n_bins_per_point-1-(n_bins_per_point-1));
//            printf("x=%g  kdv=%g   %g, %g\n", x, kdf_value,
//                    pdf_bin_accumulator, kdf_weight_accumulator);
//        }
    }
    // Normalize the pdf point accumlation by dividing by the kdf weight accumulation
    //   and write to the return pdf_array element for this CL-kernel instance
    pdf_array[pdf_idx] = pdf_bin_accumulator/kdf_weight_accumulator;
//    printf("%g / %g = %g\n",
//            pdf_bin_accumulator,kdf_weight_accumulator,pdf_array[pdf_idx]);
    return;
}
#endif
