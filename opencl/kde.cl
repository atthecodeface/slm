///
/// @file kde.cl
///
/// Kernels to do pdf-kernel density estimation
///
/// @author CPS
/// @bug No known bugs
///

///
/// @defgroup kde Kernel density estimation
/// Kernel density estimation
///

// None of the kernel-density filters below trap for bounds exceedance |@p x|>@p w.
// Shouldn't be necessary, but I need to check to see if it's really safe as I think.

#ifdef KDF_IS_TOPHAT
///
/// @brief  One of the following kernel-density filter functions: top hat,
/// triangle, Epanechnikov, cosine, Gaussian.
///
/// @param[in] x   (double, RO):
/// @param[in] w   (double, RO):
///
/// @returns double
///
/// @details If @p KDF_IS_TOPHAT is defined:
/// Top-hat (also known as 'boxcar') filter function.
/// Return the value of the filter at the sample point @p x given a bandwidth of @p w.
///
/// @ingroup kde
///
static inline double kdf_sample(const double x, const double w) {
    return 0.5f;
}
#endif

#ifdef KDF_IS_TRIANGLE
///
/// @details If @p KDF_IS_TRIANGLE is defined: Triangle kernel filter function.
/// Return the value of the filter at the sample point @p x given a bandwidth of @p w.
///
/// @ingroup kde
///
static inline double kdf_sample(const double x, const double w) {
    return (1.0f-fabs(x/w));
}
#endif

#ifdef KDF_IS_EPANECHNIKOV
///
/// @details If @p KDF_IS_EPANECHNIKOV is defined: Epanechnikov kernel filter function.
/// Return the value of the filter at the sample point @p x given a bandwidth of @p w.
/// This best-choice kd-filter is a simple quadratic.
///
/// @ingroup kde
///
static inline double kdf_sample(const double x, const double w) {
    return 0.75f*(1.0f-(x/w)*(x/w));
}
#endif

#ifdef KDF_IS_COSINE
///
/// @details If @p KDF_IS_COSINE is defined: Cosine kernel filter function.
/// Return the value of the filter at the sample point @p x given a bandwidth of @p w.
/// Spans the @p w -scaled equivalent of [-pi/2,+pi/2].
///
/// @ingroup kde
///
static inline double kdf_sample(const double x, const double w) {
    return M_PI_4*cos((M_PI_2*x)/w);
}
#endif

#ifdef KDF_IS_GAUSSIAN
///
/// @details If @p KDF_IS_GAUSSIAN is defined: Clipped Gaussian kernel filter function.
/// Return the value of the filter at the sample point @p x given a bandwidth of @p w.
/// Currently a bit of a hack - not normalized because pdf is normalized later anyway.
///
/// @ingroup kde
///
// This is a hack: ought to be rescaled in w and renormalized given the clipping
static inline double kdf_sample(const double x, const double w) {
    return exp(-0.5f*(x/(w/2.0f))*(x/(w/2.0f))); //M_SQRT1_2*
}
#endif


#ifdef KERNEL_HISTOGRAM_BIVARIATE
///
/// Compute the 2d raw (unnormalized) histogram of an @p sl_array data-pair vector.
///
/// The two columns of the data vector @p sl_array span [@p X_MIN,@p X_MAX]
///   and  [@p Y_MIN,@p Y_MAX] with ranges @p X_RANGE and  @p Y_RANGE respectively.
///   Each kernel instance maps one element pair of this data
///   vector onto its matching @p histogram_array bin element through an @p atomic_inc
///   operation on that element.
///
/// Compiled if @p KERNEL_HISTOGRAM_BIVARIATE is defined.
///
/// @ingroup kde
///
__kernel void histogram_bivariate(
        __global const    float2 *sl_array,
        __global          uint   *histogram_array
   )
{
    // For every ROI pixel...

    const uint global_id = get_global_id(0u)+get_global_id(1u)*get_global_size(0u);
    const float bin_x = (sl_array[global_id][0]-X_MIN)/X_RANGE;
    const float bin_y = (sl_array[global_id][1]-Y_MIN)/Y_RANGE;
    const uint idx_x = min(max(0u,(uint)(bin_x*(float)N_HIST_BINS)),N_HIST_BINS-1u);
    const uint idx_y = min(max(0u,(uint)(bin_y*(float)N_HIST_BINS)),N_HIST_BINS-1u);
    atomic_inc(&histogram_array[idx_y+N_HIST_BINS*idx_x]);
    return;
}
#endif

#ifdef KERNEL_PDF_BIVARIATE
#define X_AXIS 0
#define Y_AXIS 1
///
/// Kernel-density filter and reduce a 2d @p histogram_array
///
///
///
/// Compiled if @p KERNEL_PDF_BIVARIATE is defined.
///
/// @param[in]     histogram_array        (uint *,   RO):
/// @param[in]     pdf_idx                (uint,     RO):
/// @param[in]     ki                     (uint,     RO):
/// @param[in,out] pdf_bin_accumulator    (double *, RW):
/// @param[in,out] kdf_weight_accumulator (double *, RW):
/// @param[in]     kdf_width_x            (double,   RO):
/// @param[in]     bin_dx                 (double,   RO):
/// @param[in]     pdf_dx                 (double,   RO):
/// @param[in]     axis                   (uchar,    RO):
///
/// @returns void
///
/// @ingroup kde
///
static inline void filter2d(
        __global const uint *histogram_array, const uint pdf_idx,
        const uint ki, double *pdf_bin_accumulator, double *kdf_weight_accumulator,
        const double kdf_width_x,const double bin_dx, const double pdf_dx,
        const uchar axis )
{
    __private double kdf_bin_point, kdf_bin_edge, kdf_value, x;
    __private uint hi, lpki, rpki, h_left_idx, h_right_idx;

    // Set the number of histogram bins per pdf bin
    //   e.g., 2000 hist bins, 200 pdf point bins, 10 hist bins per pdf point bin
    const uint n_bins_per_point = N_HIST_BINS/N_PDF_POINTS;

    // Figure out where we are on the 2d pdf array
    const uint pdf_idx_x = pdf_idx / N_PDF_POINTS;
    const uint pdf_idx_y = pdf_idx % N_PDF_POINTS;
//    if (ki==0 && pdf_idx==210) printf("%d,%d\n",pdf_idx_x,pdf_idx_y);

    // Center of kdf bin
    kdf_bin_point = ((double)ki)*pdf_dx;
    // Left edge of kdf bin
    kdf_bin_edge = kdf_bin_point-pdf_dx/2.0f;
    // First right kdf-sampled pdf bin index...
    rpki = pdf_idx+ki;
    // ...then left kdf-sampled pdf bin index
    lpki = pdf_idx-ki+1;
    // Step through span of histogram bins for this pdf bin
    for (hi=0u;hi<n_bins_per_point;hi++) {
        // Center of histogram bin
        x = kdf_bin_edge+bin_dx*((double)hi+0.5f);
        // Calculate the kdf (kernel density filter) weight for this pdf point
        kdf_value = kdf_sample(x,kdf_width_x);
        // Get histogram count for this hist bin
        //    and add to the pdf bin accumulator
        if (axis==X_AXIS) {
            h_left_idx  = rpki*n_bins_per_point+hi;
            h_right_idx = lpki*n_bins_per_point-1-hi;
        } else {
            h_left_idx  = rpki*n_bins_per_point+hi;
            h_right_idx = lpki*n_bins_per_point-1-hi;
        }
        *pdf_bin_accumulator
            += select((double)0.0f,
                      (double)histogram_array[h_right_idx]*kdf_value,
                      (unsigned long)isless(rpki,N_PDF_POINTS));
        *pdf_bin_accumulator
            += select((double)0.0f,
                      (double)histogram_array[h_left_idx]*kdf_value,
                      (unsigned long)(isnotequal(ki,0) & isgreater(lpki,0)) );
        // Add the kdf weight to the kdf-integral accumulator
        *kdf_weight_accumulator
            += select((double)0.0f, (double)kdf_value,
                      (unsigned long)isless(rpki,N_PDF_POINTS));
        *kdf_weight_accumulator
            += select((double)0.0f, (double)kdf_value,
                      (unsigned long)(isnotequal(ki,0) & isgreater(lpki,0)) );
//        if (pdf_idx==100) {
//            printf("ki=%d, rpki=%d, lpki=%d kdfbp=%g kdfbe=%g  %d,%d  %d,%d\n",
//                ki,rpki,lpki,kdf_bin_point, kdf_bin_edge,
//                rpki*n_bins_per_point+0,rpki*n_bins_per_point+n_bins_per_point-1,
//                lpki*n_bins_per_point-1-0,lpki*n_bins_per_point-1-(n_bins_per_point-1));
//            printf("x=%g  kdv=%g   %g, %g\n", x, kdf_value,
//                    pdf_bin_accumulator, kdf_weight_accumulator);
//        }
    }
    return;
}
#endif

#ifdef KERNEL_PDF_BIVARIATE
///
/// Use kernel-density estimation to map a 2d histogram into a smooth bivariate pdf.
///
/// Compiled if @p KERNEL_PDF_BIVARIATE is defined.
///
/// @param[in]     histogram_array (uint *,  RO):
/// @param[in,out] pdf_array       (float *, RW):
///
/// @returns void
///
/// @ingroup kde
///
__kernel void pdf_bivariate(
        __global const  uint  *histogram_array,
        __global        float *pdf_array
   )
{
    // For every histogram bin...

    const uint global_id = get_global_id(0u)+get_global_id(1u)*get_global_size(0u);

    // Get this pdf point's index
    const uint pdf_idx = global_id;
    // Create a buffer variable for the pdf bin accumulator
    __private double pdf_bin_accumulator=0.0f;
    // Create a buffer variable for the kdf-integral accumulator
    __private double kdf_weight_accumulator=0.0f;
    __private uint ki;

    // Step through half kdf taking advantage of symmetry
    //   e.g., 0...2  for 5-point kdf
    for (ki=0;ki<=N_KDF_PART_POINTS_X;ki++) {
        filter2d(histogram_array, pdf_idx, ki,
                 &pdf_bin_accumulator, &kdf_weight_accumulator,
                 KDF_WIDTH_X, BIN_DX, PDF_DX,
                 X_AXIS);
    }
    for (ki=0;ki<=N_KDF_PART_POINTS_Y;ki++) {
        filter2d(histogram_array, pdf_idx, ki,
                 &pdf_bin_accumulator, &kdf_weight_accumulator,
                 KDF_WIDTH_Y, BIN_DY, PDF_DY,
                 Y_AXIS);
    }
    // Normalize the pdf point accumulation by dividing by the kdf weight accumulation
    //   and write to the return pdf_array element for this CL-kernel instance
    pdf_array[pdf_idx] = pdf_bin_accumulator/kdf_weight_accumulator;
    return;
}
#endif


#ifdef KERNEL_HISTOGRAM_UNIVARIATE
///
/// Compute the 1d raw (unnormalized) histogram of an @p sl_array data vector.
///
/// The single-column data vector @p sl_array spans [@p X_MIN,@p X_MAX] with
///   range @p X_RANGE.  Each kernel instance maps one element of this data
///   vector onto its matching @p histogram_array bin element through an @p atomic_inc
///   operation on that element.
///
/// Compiled if @p KERNEL_HISTOGRAM_UNIVARIATE is defined.
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
/// Filter and reduce a kd filter-span of a 1d @p histogram_array.
///
/// Stepping through the set of @p histogram_array bin elements encompassed
///   by the kernel filter span @p KDF_WIDTH_X at the kernel pdf element @p pdf_idx,
///   the kernel filter weight is obtained from  @p kdf_sample(), recorded cumulatively
///   in @p *kdf_weight_accumulator, applied to the @p histogram_array element value,
///   and recorded cumulatively in @p *pdf_bin_accumulator.
/// The accumulated filter weight in @p *kdf_weight_accumulator is used to normalize
///   the @p *pdf_bin_accumulator value: this is needed where the filter is clipped
///   at the data limits.
/// The kernel density filter function @p kdf_sample() is chosen and defined
///   through a compiler -D macro flag.
///
///
/// Compiled if @p KERNEL_PDF_UNIVARIATE is defined.
///
/// @param[in]     histogram_array        (uint *,   RO):
/// @param[in]     pdf_idx                (uint,     RO):
/// @param[in]     ki                     (uint,     RO):
/// @param[in,out] pdf_bin_accumulator    (double *, RW):
/// @param[in,out] kdf_weight_accumulator (double *, RW):
///
/// @returns void
///
/// @ingroup kde
///
static inline void filter1d(
        __global const uint *histogram_array, const uint pdf_idx,
        const uint ki, double *pdf_bin_accumulator, double *kdf_weight_accumulator)
{
    __private double kdf_bin_point, kdf_bin_edge, kdf_value, x;
    __private uint hi, lpki, rpki;
    __private uint h_left_idx, h_right_idx;

    // Set the number of histogram bins per pdf bin
    //   e.g., 2000 hist bins, 200 pdf point bins, 10 hist bins per pdf point bin
    const uint n_bins_per_point = N_HIST_BINS/N_PDF_POINTS;

    // Center of kdf bin
    kdf_bin_point = ((double)ki)*PDF_DX;
    // Left edge of kdf bin
    kdf_bin_edge = kdf_bin_point-PDF_DX/2.0f;
    // First right kdf-sampled pdf bin index...
    rpki = pdf_idx+ki;
    // ...then left kdf-sampled pdf bin index
    lpki = pdf_idx-ki+1;
    // Step through span of histogram bins for this pdf bin
    for (hi=0u;hi<n_bins_per_point;hi++) {
        // Center of histogram bin
        x = kdf_bin_edge+BIN_DX*((double)hi+0.5f);
        // Calculate the kdf (kernel density filter) weight for this pdf point
        kdf_value = kdf_sample(x,KDF_WIDTH_X);
        // Get histogram count for this hist bin
        //    and add to the pdf bin accumulator
        h_left_idx  = rpki*n_bins_per_point+hi;
        h_right_idx = lpki*n_bins_per_point-1-hi;
        *pdf_bin_accumulator
            += select((double)0.0f,
                      (double)histogram_array[h_right_idx]*kdf_value,
                      (unsigned long)isless(rpki,N_PDF_POINTS));
        *pdf_bin_accumulator
            += select((double)0.0f,
                      (double)histogram_array[h_left_idx]*kdf_value,
                      (unsigned long)(isnotequal(ki,0) & isgreater(lpki,0)) );
        // Add the kdf weight to the kdf-integral accumulator
        *kdf_weight_accumulator
            += select((double)0.0f, (double)kdf_value,
                      (unsigned long)isless(rpki,N_PDF_POINTS));
        *kdf_weight_accumulator
            += select((double)0.0f, (double)kdf_value,
                      (unsigned long)(isnotequal(ki,0) & isgreater(lpki,0)) );
    }
    return;
}
#endif

#ifdef KERNEL_PDF_UNIVARIATE
///
/// Use kernel-density estimation to map a 1d histogram into a smooth univariate pdf.
///
/// Compiled if @p KERNEL_PDF_UNIVARIATE is defined.
///
/// @param[in]     histogram_array (uint *,  RO):
/// @param[in,out] pdf_array       (float *, RW):
///
/// @returns void
///
/// @ingroup kde
///
__kernel void pdf_univariate(
        __global const  uint  *histogram_array,
        __global        float *pdf_array
   )
{
    // For every histogram bin...

    const uint global_id = get_global_id(0u)+get_global_id(1u)*get_global_size(0u);

    // Get this pdf point's index
    const uint pdf_idx = global_id;
    // Create a buffer variable for the pdf bin accumulator
    __private double pdf_bin_accumulator=0.0f;
    // Create a buffer variable for the kdf-integral accumulator
    __private double kdf_weight_accumulator=0.0f;
    __private uint ki;

    // Step through half kdf taking advantage of symmetry
    //   e.g., 0...2  for 5-point kdf
    for (ki=0;ki<=N_KDF_PART_POINTS_X;ki++) {
        filter1d(histogram_array, pdf_idx, ki,
                 &pdf_bin_accumulator, &kdf_weight_accumulator);
    }
    // Normalize the pdf point accumulation by dividing by the kdf weight accumulation
    //   and write to the return pdf_array element for this CL-kernel instance
    pdf_array[pdf_idx] = pdf_bin_accumulator/kdf_weight_accumulator;
    return;
}
#endif
