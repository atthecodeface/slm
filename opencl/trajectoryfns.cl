///
/// @file trajectoryfns.cl
///
/// Various functions to compute trajectories and record data along them
///
/// @author CPS
/// @bug No known bugs
///

///
/// @defgroup trajectoryfns Trajectory stepping and recording functions
/// Functions to carry out Runge-Kutta integration steps along streamlines & record them
///


#ifdef KERNEL_INTEGRATE_FIELDS
///
/// Update variables tracking trajectory length and integration step counter.
/// Write length and count data to global arrays.
///
/// Compiled if KERNEL_INTEGRATE_TRAJECTORY is defined.
///
/// @param[in]      dl: step distance
/// @param[in,out]  l_trajectory: total streamline distance so far
/// @param[in]      vec: current (x,y) coordinate vector at tip of streamline trajectory
/// @param[in]      prev_vec: previous (x,y) coordinate vector on streamline trajectory
/// @param[in,out]  n_steps: number of integration steps so far in streamline trajectory
/// @param[in,out]  idx: array index of pixel at current (x,y) position
/// @param[in,out]  prev_idx: array index of pixel at previous (x,y) position
/// @param[in]      mask_array: grid pixel mask (padded),
///                         with @p true = masked, @p false = good
/// @param[in,out]  slc_array: grid recording accumulated count of streamline integration
///                           steps across each pixel (padded)
/// @param[in,out]  slt_array: grid recording accumulated count of streamline segment
///                           lengths crossing each pixel (padded)
///
/// @returns void
///
/// @ingroup trajectoryfns
///
static inline void update_trajectory_write_sl_data(
        float  dl, float *l_trajectory, float2 vec, float2 prev_vec,
        uint *n_steps, uint *idx, uint *prev_idx,
        __global const bool *mask_array,
        __global uint *slt_array, __global uint *slc_array) {
    // Step to next point along streamline, adding to trajectory length
    //   and n_steps counter.
    // Compress step delta vector into fixed-point integer form & record in traj.
    // Write to slt, slc arrays to record passage of this streamline.
    *l_trajectory += dl;
    *n_steps += 1u;
    // Current pixel position in data array
    *idx = get_array_idx(vec);
    check_atomic_write_sl_data(*idx, prev_idx, mask_array[*idx],
                               &slt_array[*idx], &slc_array[*idx], *l_trajectory);
}
#endif

#ifdef KERNEL_INTEGRATE_TRAJECTORY
///
/// Update variables tracking trajectory length and integration step counter.
/// Record (to global array) a compressed version of the current trajectory step vector.
///
/// Compiled if KERNEL_INTEGRATE_TRAJECTORY is defined.
///
/// @param[in]      dl: step distance
/// @param[in,out]  l_trajectory: total streamline distance so far
/// @param[in]      vec: current (x,y) coordinate vector at tip of streamline trajectory
/// @param[in]      prev_vec: previous (x,y) coordinate vector on streamline trajectory
/// @param[in,out]  n_steps: number of integration steps so far in streamline trajectory
/// @param[in,out]  trajectory_vec: streamline trajectory record
///                                  (2d array of compressed (x,y) vectors)
///
/// @returns void
///
/// @ingroup trajectoryfns
///
static inline void update_record_trajectory(
        float dl, float *l_trajectory, float2 vec, float2 prev_vec,
        uint *n_steps, __global char2 *trajectory_vec) {
    // Step to next point along streamline, adding to trajectory length
    //   and n_steps counter.
    // Compress step delta vector into fixed-point integer form.
    *l_trajectory += dl;
    trajectory_vec[*n_steps] = compress(vec-prev_vec);
    *n_steps += 1u;
}
#endif

#ifdef KERNEL_CONNECT_CHANNELS
///
/// Update variables tracking trajectory length and integration step counter.
/// Record (to private array) a compressed version of the current trajectory step vector.
///
/// Compiled if KERNEL_INTEGRATE_TRAJECTORY is defined.
///
/// @param[in]      dl: step distance
/// @param[in,out]  l_trajectory: total streamline distance so far
/// @param[in]      vec: current (x,y) coordinate vector at tip of streamline trajectory
/// @param[in]      prev_vec: previous (x,y) coordinate vector on streamline trajectory
/// @param[in,out]  n_steps: number of integration steps so far in streamline trajectory
/// @param[in,out]  trajectory_vec: streamline trajectory record
///                                  (2d array of compressed (x,y) vectors)
///
/// @returns void
///
/// @ingroup trajectoryfns
///
static inline void update_record_private_trajectory(
        float dl, float *l_trajectory, float2 vec, float2 prev_vec,
        uint *n_steps, __private char2 *trajectory_vec) {
    // Step to next point along streamline, adding to trajectory length
    //   and n_steps counter.
    // Compress step delta vector into fixed-point integer form.
    *l_trajectory += dl;
    trajectory_vec[*n_steps] = compress(vec-prev_vec);
    *n_steps += 1u;
}
#endif

#ifdef KERNEL_HILLSLOPE_LENGTHS
///
/// Update variables tracking trajectory length and integration step counter.
///
/// Compiled if KERNEL_INTEGRATE_TRAJECTORY is defined.
///
/// @param[in]      dl: step distance
/// @param[in,out]  l_trajectory: total streamline distance so far
/// @param[in,out]  n_steps: number of integration steps so far in streamline trajectory
///
/// @returns void
///
/// @ingroup trajectoryfns
///
static inline void update_trajectory(
        float  dl, float *l_trajectory, uint *n_steps) {
    // Step to next point along streamline, adding to trajectory length
    //   and n_steps counter.
    // Compress step delta vector into fixed-point integer form.
    *l_trajectory += dl;
    *n_steps += 1u;
}
#endif

#ifdef KERNEL_INTEGRATE_TRAJECTORY
///
/// Record the (final) trajectory length and count of integration steps
///    to global arrays @p traj_length_array and @p traj_nsteps_array respectively.
/// This action takes place at the end of streamline tracing in each integration
///    kernel instance.
///
/// Compiled if KERNEL_INTEGRATE_TRAJECTORY is defined.
///
/// @param[in]   global_id: ID of the kernel instance
/// @param[in]   n_steps: number of integration steps so far in streamline trajectory
/// @param[in]   l_trajectory: total streamline distance so far
/// @param[out]  traj_nsteps_array: list of number of steps along each trajectory;
///                                 one per @p seed_point_array vector
/// @param[out]  traj_length_array: list of lengths of each trajectory;
///                                 one per @p seed_point_array vector
///
/// @returns void
///
/// @ingroup trajectoryfns
///
static inline void finalize_trajectory(const uint global_id, uint n_steps,
                                       float l_trajectory,
                                       __global ushort *traj_nsteps_array,
                                       __global float *traj_length_array) {
    // Record the total stream length in pixels and point count for rtn to CPU
    traj_nsteps_array[global_id] = n_steps;
    traj_length_array[global_id] = l_trajectory;
}
#endif
