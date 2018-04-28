///
/// @file writearray.cl
///
/// Functions to write to slc & slc grid arrays using atomic ops (mask-checked and not)
///
/// @author CPS
/// @bug No known bugs
///

#ifdef KERNEL_INTEGRATE_FIELDS
///
/// Add the current streamline length (@p l_trajectory) to the current pixel of the
///    @p slt accumulation array.
/// Similarly, increment the streamline count at the current pixel of the
///    @p slc accumulation array.
/// Atomic operations are used since several kernel instances may need to write
///    to the same pixel effectively simultaneously.
///
/// Compiled if KERNEL_INTEGRATE_TRAJECTORY is defined.
///
/// @param[in,out] slc: grid recording accumulated count of streamline integration
///                         steps across each pixel (padded)
/// @param[in,out] slt: grid recording accumulated count of streamline segment lengths
///                         crossing each pixel (padded)
/// @param[in]     l_trajectory: total streamline distance so far
///
/// @returns void
///
/// @ingroup trajectoryfns
///
static inline void atomic_write_sl_data(__global uint *slt, __global uint *slc,
                                        const float l_trajectory) {
    // Add streamline length-so-far to total slt for this pixel
    //   - rounding up to & casting as 32bit int
    // There may be issues for short trajectories as a result.
    // Also if step distance is << pixel width.
    atomic_add(slt, (uint)(l_trajectory+0.5f));
    // Increment the 'visit' counter slc at this pixel.
    atomic_inc(slc);
}
#endif

#ifdef KERNEL_INTEGRATE_FIELDS
///
/// Extended version of atomic_write_sl_data() to include testing whether the
///    current pixel is masked, and an assignment of the previous pixel index
///    to the current pixel index.
///
/// Add the current streamline length (@p l_trajectory) to the current pixel of the
///    @p slt accumulation array.
/// Similarly, increment the streamline count at the current pixel of the
///    @p slc accumulation array.
/// Atomic operations are used since several kernel instances may need to write
///    to the same pixel effectively simultaneously.
///
/// Compiled if KERNEL_INTEGRATE_TRAJECTORY is defined.
///
/// @param[in]      idx: array index of pixel at current (x,y) position
/// @param[in,out]  prev_idx: array index of pixel at previous (x,y) position
/// @param[in]      mask_flag: whether current pixel is masked or not
/// @param[in,out]  slc: grid recording accumulated count of streamline integration
///                         steps across each pixel (padded)
/// @param[in,out]  slt: grid recording accumulated count of streamline segment lengths
///                         crossing each pixel (padded)
/// @param[in]      l_trajectory: total streamline distance so far
///
/// @returns void
///
/// @ingroup trajectoryfns
///
static inline void check_atomic_write_sl_data(const uint idx, uint *prev_idx,
                                              const bool mask_flag,
                                              __global uint *slt, __global uint *slc,
                                              const float l_trajectory) {
    if (idx<(NX_PADDED*NY_PADDED) && !mask_flag) {
        // Add streamline length-so-far to total slt for this pixel
        //   - rounding up to & casting as 32bit int
        // There may be issues for short trajectories as a result.
        // Also if step distance is << pixel width.
        atomic_add(slt, (uint)(l_trajectory+0.5f));
        // Increment the 'visit' counter slc at this pixel.
        atomic_inc(slc);
    }
    *prev_idx = idx;
}
#endif
