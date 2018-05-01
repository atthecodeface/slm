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

// All of the kernel-density filters below trap for bounds exceedance |@p x|>@p w.
// Shouldn't be necessary, but you never know.
// None are normalized, because each discretely sampled kd filter is integrated
//   and normalized.

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
static inline double k_sample(const double x, const double w) {
    return select((double)0.0f, (double)1.0f,
                  (unsigned long)isless(fabs(x),w)); //0.5f/w; hack
}
#endif

#ifdef KDF_IS_TRIANGLE
///
/// @details If @p KDF_IS_TRIANGLE is defined: Triangle kernel filter function.
/// Return the value of the filter at the sample point @p x given a bandwidth of @p w.
///
/// @ingroup kde
///
static inline double k_sample(const double x, const double w) {
    return fmax( 1.0f-fabs(x/w) , (double)0.0f );
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
static inline double k_sample(const double x, const double w) {
    return fmax( (1.0f-(x/w)*(x/w)), (double)0.0f); //0.75f*
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
static inline double k_sample(const double x, const double w) {
    return fmax( cos((M_PI_2*x)), (double)0.0f); //M_PI_4*
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
static inline double k_sample(const double x, const double w) {
    return exp(-0.5f*(x/(w*0.4f))*(x/(w*0.4f))); //M_SQRT1_2*
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
    const uint idx_x = min(max(0u,(uint)(bin_x*(float)(N_HIST_BINS-1u))),N_HIST_BINS-1u);
    const uint idx_y = min(max(0u,(uint)(bin_y*(float)(N_HIST_BINS-1u))),N_HIST_BINS-1u);
    atomic_inc(&histogram_array[idx_y+N_HIST_BINS*idx_x]);
    return;
}
#endif


#ifdef KERNEL_PDF_BIVARIATE_ROWS
///
/// Kernel-density filter and reduce a 2d @p histogram_array
///
///
///
/// Compiled if @p KERNEL_PDF_BIVARIATE_ROWS is defined.
///
/// @param[in]     histogram_array        (uint *,   RO):
/// @param[in]     hi_idx                 (uint,     RO):
/// @param[in]     bin_dx                 (double,   RO):
/// @param[in]     k_idx                  (uint,     RO):
/// @param[in]     k_width                (double,   RO):
/// @param[in,out] pdf_bin_accumulator    (double *, RW):
/// @param[in,out] k_weight_accumulator   (double *, RW):
///
/// @returns void
///
/// @ingroup kde
///
static inline void filter_along_rows(
        __global const uint *histogram_array,
        const uint hi_idx, const double bin_dx, const uint k_idx, const double k_width,
        double *pdf_bin_accumulator, double *k_weight_accumulator)
{
    // Figure out where we are on the coarse 2d pdf array (at the kdf center)
    const int hi_col = hi_idx % N_HIST_BINS;
    const int hi_row = hi_idx / N_HIST_BINS;
    // Figure out where the left & right kdf bins are on the coarse 2d pdf array
    // First right kdf-sampled pdf bin index...
    const int hi_col_rght = hi_col+(int)k_idx;
    // ...then left kdf-sampled pdf bin index
    // NB: add one here because histogram left-indexing starts at the right edge
    //     of the bin, aka the edge of the bin to the right ie with bin index plus one
    const int hi_col_left = hi_col-(int)k_idx+1;

    // Get the kdf weight
    const double k_value = k_sample((double)k_idx, k_width);

    const uint is_ok_right = isless(hi_col_rght,(int)N_HIST_BINS);
    const uint is_ok_left  = isnotequal(k_idx,0u) & isgreaterequal(hi_col_left,0);
    int hist_rght = hi_row*N_HIST_BINS+hi_col_rght;
    int hist_left = hi_row*N_HIST_BINS+hi_col_left;
    hist_rght = select(0u, (uint)hist_rght, is_ok_right);
    hist_left = select(0u, (uint)hist_left, is_ok_left);
    // Get histogram right & left counts for this hist bin
    //   and add to the pdf bin accumulator
    *pdf_bin_accumulator
        += select((double)0.0f, k_value*(double)histogram_array[hist_rght],
                  (unsigned long)is_ok_right);
    *pdf_bin_accumulator
        += select((double)0.0f, k_value*(double)histogram_array[hist_left],
                  (unsigned long)is_ok_left);
    // Add the kdf weight to the kdf-integral accumulator
    *k_weight_accumulator
        += select((double)0.0f, (double)k_value, (unsigned long)is_ok_right);
    *k_weight_accumulator
        += select((double)0.0f, (double)k_value, (unsigned long)is_ok_left );
    return;
}
#endif

#ifdef KERNEL_PDF_BIVARIATE_ROWS
///
/// Use kernel-density estimation to map a 2d histogram into a smooth 2d pdf: rows only.
///
/// Compiled if @p KERNEL_PDF_BIVARIATE is defined.
///
/// @param[in]     histogram_array   (uint *,  RO):
/// @param[in,out] partial_pdf_array (float *, RW):
///
/// @returns void
///
/// @ingroup kde
///
__kernel void pdf_bivariate_rows(
        __global const  uint  *histogram_array,
        __global        float *partial_pdf_array )
{
    // For every histogram bin...

    const uint global_id = get_global_id(0u)+get_global_id(1u)*get_global_size(0u);

    // Create a buffer variable for the pdf bin accumulator
    __private double pdf_bin_accumulator=0.0f;
    // Create a buffer variable for the kdf-integral accumulator
    __private double k_weight_accumulator=1.0f;
    __private uint k_idx;

    // Step through half kdf taking advantage of symmetry
    //   e.g., 0...2  for 5-point kdf
    // Smooth along rows (C-order, thus row-major, last (y) index changes faster)
    for (k_idx=0;k_idx<=N_KDF_PART_POINTS_Y;k_idx++) {
        filter_along_rows(histogram_array, global_id, BIN_DY,
                          k_idx, (double)(N_KDF_PART_POINTS_Y)+0.5f,
                          &pdf_bin_accumulator, &k_weight_accumulator);
    }
    // Normalize the pdf point accumulation by dividing by the kdf weight accumulation
    //   and write to the return pdf_array element for this CL-kernel instance
    partial_pdf_array[global_id] = (float)(pdf_bin_accumulator/k_weight_accumulator);
    return;
}
#endif


#ifdef KERNEL_PDF_BIVARIATE_COLS
///
/// Kernel-density filter and reduce a 2d @p histogram_array
///
///
///
/// Compiled if @p KERNEL_PDF_BIVARIATE_COLS is defined.
///
/// @param[in]     partial_pdf_array      (float *,  RO):
/// @param[in]     pdf_idx                (uint,     RO):
/// @param[in]     pdf_kdx                (double,   RO):
/// @param[in]     bin_dx                 (double,   RO):
/// @param[in]     k_idx                  (uint,     RO):
/// @param[in]     k_width                (double,   RO):
/// @param[in,out] pdf_bin_accumulator    (double *, RW):
/// @param[in,out] k_weight_accumulator   (double *, RW):
///
/// @returns void
///
/// @ingroup kde
///
static inline void filter_along_cols(
        __global const float *partial_pdf_array,
        const uint pdf_idx, const double bin_dx,
        const double pdf_kdx, const uint k_idx, const double k_width,
        double *pdf_bin_accumulator, double *k_weight_accumulator)
{
    __private double k_value, k_x;
    __private int hi_col, hi_row, hi_row_up, hi_row_dn;
    __private uint is_ok_down, is_ok_up;

    // Set the number of histogram bins per pdf bin
    //   e.g., 1000 hist bins, 100 pdf point bins, 10 hist bins per pdf point bin
    const uint n_bins_per_point = N_HIST_BINS/N_PDF_POINTS;

    // Figure out where we are on the coarse 2d pdf array (at the kdf center)
    const int pdf_col = pdf_idx % N_PDF_POINTS;
    const int pdf_row = pdf_idx / N_PDF_POINTS;

    // Figure out where the left & right kdf bins are on the coarse 2d pdf array
    // First right kdf-sampled pdf bin index...
    const int pdf_row_dn = pdf_row+(int)k_idx;
    // ...then left kdf-sampled pdf bin index
    const int pdf_row_up = pdf_row-(int)k_idx;

    // Figure out where we are on the fine 2d histogram array
    // First right kdf-sampled pdf bin index...
    const int hi_row_dn_offset \
        = (uint)(pdf_col+pdf_row_dn*N_HIST_BINS)*n_bins_per_point;
    // ...then left kdf-sampled pdf bin index
    const int hi_row_up_offset \
        = (uint)(pdf_col+pdf_row_up*N_HIST_BINS)*n_bins_per_point;

    // Step columnwise through all the histogram bins in this pdf bin
    // Apply this weight repeatedly across all the rows in this hist bin
    for (hi_row=0u;hi_row<(int)n_bins_per_point;hi_row++) {
        // Sample point on kernel-density filter
        k_x = (double)hi_row+((double)k_idx)*n_bins_per_point;
        // Get the filter weight at this point
        k_value = k_sample(k_x, k_width);
        // Apply this weight repeatedly across all the rows in this hist bin
        for (hi_col=0u;hi_col<(int)n_bins_per_point;hi_col++) {
            hi_row_dn = hi_row_dn_offset+hi_row*N_HIST_BINS+hi_col;
            hi_row_up = hi_row_up_offset+(n_bins_per_point-1-hi_row)*N_HIST_BINS+hi_col;
            is_ok_down = isless(hi_row_dn,(int)(N_HIST_BINS*N_HIST_BINS));
            is_ok_up   = isnotequal(k_idx,0) & isgreaterequal(hi_row_up,0);
            hi_row_dn = select(0u, (uint)hi_row_dn, is_ok_down);
            hi_row_up = select(0u, (uint)hi_row_up, is_ok_up);
            // Get histogram right & left counts for this hist bin
            //   and add to the pdf bin accumulator
            *pdf_bin_accumulator
                += select((double)0.0f, k_value*(double)partial_pdf_array[hi_row_dn],
                          (unsigned long)is_ok_down);
            *pdf_bin_accumulator
                += select((double)0.0f, k_value*(double)partial_pdf_array[hi_row_up],
                          (unsigned long)is_ok_up);
            // Add the kdf weight to the kdf-integral accumulator
            *k_weight_accumulator
                += select((double)0.0f, (double)k_value, (unsigned long)is_ok_down);
            *k_weight_accumulator
                += select((double)0.0f, (double)k_value, (unsigned long)is_ok_up);
        }
    }
    return;
}

#endif

#ifdef KERNEL_PDF_BIVARIATE_COLS
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
__kernel void pdf_bivariate_cols(
        __global const  float *partial_pdf_array,
        __global        float *pdf_array )
{
    // For every pdf point...

    const uint global_id = get_global_id(0u)+get_global_id(1u)*get_global_size(0u);

    // Get this pdf point's index
    const uint pdf_idx = global_id;
    // Create a buffer variable for the pdf bin accumulator
    __private double pdf_bin_accumulator=0.0f;
    // Create a buffer variable for the kdf-integral accumulator
    __private double k_weight_accumulator=1.0f;
    __private uint k_idx;

    // Step through half kdf taking advantage of symmetry
    //   e.g., 0...2  for 5-point kdf
    // Smooth along columns (C-order, thus row-major, first (x) index changes slower)
    for (k_idx=0;k_idx<=N_KDF_PART_POINTS_X;k_idx++) {
        filter_along_cols(partial_pdf_array, pdf_idx, PDF_DX, BIN_DX,
                          k_idx, (double)(N_KDF_PART_POINTS_X)+0.5f,
                          &pdf_bin_accumulator, &k_weight_accumulator);
    }
    // Normalize the pdf point accumulation by dividing by the kdf weight accumulation
    //   and write to the return pdf_array element for this CL-kernel instance
    pdf_array[pdf_idx] = (float)(pdf_bin_accumulator/k_weight_accumulator);
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
        __global          uint   *histogram_array )
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
///   the kernel filter weight is obtained from  @p k_sample(), recorded cumulatively
///   in @p *k_weight_accumulator, applied to the @p histogram_array element value,
///   and recorded cumulatively in @p *pdf_bin_accumulator.
/// The accumulated filter weight in @p *k_weight_accumulator is used to normalize
///   the @p *pdf_bin_accumulator value: this is needed where the filter is clipped
///   at the data limits.
/// The kernel density filter function @p k_sample() is chosen and defined
///   through a compiler -D macro flag.
///
///
/// Compiled if @p KERNEL_PDF_UNIVARIATE is defined.
///
/// @param[in]     histogram_array        (uint *,   RO):
/// @param[in]     pdf_idx                (uint,     RO):
/// @param[in]     k_idx                  (uint,     RO):
/// @param[in]     k_points               (uint,     RO):
/// @param[in,out] pdf_bin_accumulator    (double *, RW):
/// @param[in,out] k_weight_accumulator   (double *, RW):
///
/// @returns void
///
/// @ingroup kde
///
static inline void filter(
        __global const uint *histogram_array, const uint pdf_idx,
        const uint k_idx, const uint k_points,
        double *pdf_bin_accumulator, double *k_weight_accumulator)
{
    __private double k_bin_point, k_bin_edge, k_value, x;
    __private int pdf_col_left, pdf_col_rght;
    __private uint h_col, h_col_left, h_col_rght;
    __private uint is_ok_right, is_ok_left;

    // Set the number of histogram bins per pdf bin
    //   e.g., 2000 hist bins, 200 pdf point bins, 10 hist bins per pdf point bin
    const uint n_bins_per_point = N_HIST_BINS/N_PDF_POINTS;

    // Center of kdf bin
    k_bin_point = ((double)k_idx)*PDF_DX;
    // Left edge of kdf bin
    k_bin_edge = k_bin_point-PDF_DX/2.0f;
    // First right kdf-sampled pdf bin index...
    pdf_col_rght = pdf_idx+k_idx;
    // ...then left kdf-sampled pdf bin index
    // NB: add one here because histogram left-indexing starts at the right edge
    //     of the bin, aka the edge of the bin to the right ie with bin index plus one
    pdf_col_left = pdf_idx-k_idx+1;
    // Step through span of histogram bins for this pdf bin
    for (h_col=0u;h_col<n_bins_per_point;h_col++) {
        // Center of histogram bin
        x = k_bin_edge+BIN_DX*((double)h_col+0.5f);
        // Calculate the kdf (kernel density filter) weight for this pdf point
        k_value = k_sample(x,KDF_WIDTH_X);
        // Get histogram count for this hist bin on both sides of the symm filter
        // First check if the filter spills off the right or left array limits
        is_ok_right = isless(pdf_col_rght,(int)N_PDF_POINTS);
        // When checking left also suppress accumulation if at filter center
        //   (don't count twice at the x=0 pdf bin)
        is_ok_left  = isnotequal(k_idx,0) & isgreater(pdf_col_left,0);
        h_col_rght = (uint)(pdf_col_rght*n_bins_per_point+h_col);
        h_col_left = (uint)(pdf_col_left*n_bins_per_point-h_col-1u);
        // Map R&L col indexes into dummy value zero index if off grid
        h_col_rght = select(0u, h_col_rght, is_ok_right);
        h_col_left = select(0u, h_col_left, is_ok_left);
        // Add to the pdf bin accumulator
        //   - but don't bother if the array index is off grid
        *pdf_bin_accumulator
            += select((double)0.0f, (double)histogram_array[h_col_rght]*k_value,
                      (unsigned long)is_ok_right);
        *pdf_bin_accumulator
            += select((double)0.0f, (double)histogram_array[h_col_left]*k_value,
                      (unsigned long)is_ok_left);
        // Add the kdf weight to the kdf-integral accumulator
        *k_weight_accumulator
            += select((double)0.0f, (double)k_value, (unsigned long)is_ok_right);
        *k_weight_accumulator
            += select((double)0.0f, (double)k_value, (unsigned long)is_ok_left);
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
        __global        float *pdf_array )
{
    // For every pdf point...

    const uint global_id = get_global_id(0u)+get_global_id(1u)*get_global_size(0u);

    // Get this pdf point's index
    const uint pdf_idx = global_id;
    // Create a buffer variable for the pdf bin accumulator
    __private double pdf_bin_accumulator=0.0f;
    // Create a buffer variable for the kdf-integral accumulator
    __private double k_weight_accumulator=0.0f;
    __private uint k_idx;

    // Step through half kdf taking advantage of symmetry
    //   e.g., 0...2  for 5-point kdf
    for (k_idx=0;k_idx<=N_KDF_PART_POINTS_X;k_idx++) {
        filter(histogram_array, pdf_idx, k_idx, N_KDF_PART_POINTS_X*2u+1u,
               &pdf_bin_accumulator, &k_weight_accumulator);
    }
    // Normalize the pdf point accumulation by dividing by the kdf weight accumulation
    //   and write to the return pdf_array element for this CL-kernel instance
    pdf_array[pdf_idx] = (float)pdf_bin_accumulator/k_weight_accumulator;
    return;
}
#endif
