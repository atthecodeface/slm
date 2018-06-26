///
/// @file integrationfns.cl
///
/// Adaptive 1st or 2nd order Runge-Kutta single-stepping functions
///
/// @author CPS
/// @bug No known bugs
///

///
/// @defgroup integrationfns Runge-Kutta integration step functions
/// Functions used to compute Runge-Kutta integration down and up streamlines
///

#ifdef KERNEL_INTEGRATE_TRAJECTORY
///
/// Compute a single step of 2nd-order Runge-Kutta numerical integration of
///    a streamline given precomputed 1st and 2nd order step vectors.
/// If the step is deemed too small, return @p true = stuck;
///    otherwise return @p false = ok.
/// Record this step in the global @p trajectory_vec (array of step vectors).
/// Also update the total streamline length @p l_trajectory and
///    increment the step counter @p n_steps.
/// In addition, push the current @p vec to the @p prev_vec, push the current @p idx to
///    the @p prev_idx, and update the @p idx of the current @p vec.
///
/// Compiled if KERNEL_INTEGRATE_TRAJECTORY is defined.
///
/// @param[in,out]  dt: delta time step
/// @param[in,out]  dl: step distance
/// @param[in,out]  l_trajectory: total streamline distance so far
/// @param[in,out]  dxy1_vec: R-K first order delta step vector
/// @param[in,out]  dxy2_vec: R-K second order delta step vector
/// @param[in,out]  vec: current (x,y) coordinate vector at tip of streamline trajectory
/// @param[in,out]  prev_vec: previous (x,y) coordinate vector on streamline trajectory
/// @param[in,out]  next_vec: next (x,y) coordinate vector on streamline trajectory
/// @param[in,out]  n_steps: number of integration steps so far in streamline trajectory
/// @param[in,out]  idx: array index of pixel at current (x,y) position
/// @param[in,out]  prev_idx: array index of pixel at previous (x,y) position
/// @param[in,out]  trajectory_vec: streamline trajectory record
///                                  (2d array of compressed (x,y) vectors)
///
/// @retval bool:
///               @p true if stuck (step length less than @p INTEGRATION_HALT_THRESHOLD);
///               @p false otherwise aka step computed well
///
/// @ingroup integrationfns
///
static inline bool runge_kutta_step_record(float *dt, float  *dl, float *l_trajectory,
                                           float2 *dxy1_vec, float2 *dxy2_vec,
                                           float2 *vec, float2 *prev_vec,
                                           float2 *next_vec,
                                           uint *n_steps, uint *idx, uint *prev_idx,
                                           __global char2 *trajectory_vec)
{
    const float step_error = fast_length((*dxy2_vec-*dxy1_vec)/GRID_SCALE);

    *dl = fast_length(*dxy2_vec);
    if (*dl<(INTEGRATION_HALT_THRESHOLD)) {
        *next_vec = *vec+*dxy1_vec;
        *dl = fast_length(*dxy1_vec);
    }
    *vec = *next_vec;
    *idx = get_array_idx(*next_vec);
    if (*dl<(INTEGRATION_HALT_THRESHOLD)) {
        update_record_trajectory(*dl,l_trajectory,*vec,*prev_vec,n_steps,trajectory_vec);
//        printf("runge_kutta_step_record: stuck\n");
        return true;
    }
    *dt = select( fmin(DT_MAX,(ADJUSTED_MAX_ERROR*(*dt))/(step_error)), DT_MAX,
                 isequal(step_error,0.0f) );
    update_record_trajectory(*dl,l_trajectory,*vec,*prev_vec,n_steps,trajectory_vec);
    *prev_vec = *vec;
//    *prev_idx = select(*prev_idx,*idx,isnotequal(*idx,*prev_idx));
    *prev_idx = *idx;
    return false;
}
#endif

#ifdef KERNEL_INTEGRATE_TRAJECTORY
/// Compute a single Euler integration step of of a streamline.
/// Record this step in the global @p trajectory_vec (array of step vectors).
/// Also update the total streamline length @p l_trajectory and
///    increment the step counter @p n_steps.
///
/// Compiled if KERNEL_INTEGRATE_TRAJECTORY is defined.
///
/// @param[in,out]  dt: delta time step
/// @param[in,out]  dl: step distance
/// @param[in,out]  l_trajectory: total streamline distance so far
/// @param[in]      uv_vec: flow velocity vector interpolated to current position
/// @param[in,out]  vec: current (x,y) coordinate vector at tip of streamline trajectory
/// @param[in]      prev_vec: previous (x,y) coordinate vector on streamline trajectory
/// @param[in,out]  n_steps: number of integration steps so far in streamline trajectory
/// @param[in,out]  trajectory_vec: streamline trajectory record
///                                  (2d array of compressed (x,y) vectors)
///
/// @retval bool:
///               @p true if stuck (step length less than @p INTEGRATION_HALT_THRESHOLD);
///               @p false otherwise aka step computed well
///
/// @ingroup integrationfns
///
static inline void euler_step_record(float *dt, float  *dl,
                                     float *l_trajectory, const float2 uv_vec,
                                     float2 *vec, const float2 prev_vec,
                                     uint *n_steps, __global char2 *trajectory_vec)
{
    const float2 sgnd_uv_vec = uv_vec*(float2)(DOWNUP_SIGN,DOWNUP_SIGN);
    const float dt_x = dt_to_nearest_edge((*vec)[0], sgnd_uv_vec[0]);
    const float dt_y = dt_to_nearest_edge((*vec)[1], sgnd_uv_vec[1]);

    *dt = minmag(dt_x,dt_y);
    *vec += sgnd_uv_vec*(*dt);
    *vec = approximate(*vec);
    *vec = (float2)( fmin(fmax((*vec)[0],-0.5f),NXF_MP5),
                     fmin(fmax((*vec)[1],-0.5f),NYF_MP5) );
    *dl = fast_length(*vec-prev_vec);
    update_record_trajectory(*dl,l_trajectory,*vec,prev_vec,n_steps,trajectory_vec);
 }
#endif

#ifdef KERNEL_INTEGRATE_FIELDS
///
/// Compute a single step of 2nd-order Runge-Kutta numerical integration of
///    a streamline given precomputed 1st and 2nd order step vectors.
/// If the step is deemed too small, return @p true = stuck;
///    otherwise return @p false = ok.
/// Record this step in the global @p slt_array and @p slc_array (streamline length total
///    and count total respectively) by updating both at the current pixel
///    (using @p mask_array to block where masked).
/// Also update the total streamline length @p l_trajectory and
///    increment the step counter @p n_steps.
/// In addition, push the current @p vec to the @p prev_vec, push the current @p idx to
///    the @p prev_idx, and update the @p idx of the current @p vec.
///
/// Compiled if KERNEL_INTEGRATE_FIELDS is defined.
///
/// @param[in,out]  dt: delta time step
/// @param[in,out]  dl: step distance
/// @param[in,out]  l_trajectory: total streamline distance so far
/// @param[in,out]  dxy1_vec: R-K first order delta step vector
/// @param[in,out]  dxy2_vec: R-K second order delta step vector
/// @param[in,out]  vec: current (x,y) coordinate vector at tip of streamline trajectory
/// @param[in,out]  prev_vec: previous (x,y) coordinate vector on streamline trajectory
/// @param[in]      next_vec: next (x,y) coordinate vector on streamline trajectory
/// @param[in,out]  n_steps: number of integration steps so far in streamline trajectory
/// @param[in,out]  idx: array index of pixel at current (x,y) position
/// @param[in,out]  prev_idx: array index of pixel at previous (x,y) position
/// @param[in]  mask_array: grid pixel mask (padded),
///                         with @p true = masked, @p false = good
/// @param[in,out] slc_array: grid recording accumulated count of streamline integration
///                           steps across each pixel (padded)
/// @param[in,out] slt_array: grid recording accumulated count of streamline segment
///                           lengths crossing each pixel (padded)
///
/// @retval bool:
///               @p true if stuck (step length less than @p INTEGRATION_HALT_THRESHOLD);
///               @p false otherwise aka step computed well
///
/// @ingroup integrationfns
///
static inline bool runge_kutta_step_write_sl_data(
                float *dt, float  *dl, float *l_trajectory,
                float2 *dxy1_vec, float2 *dxy2_vec,
                float2 *vec, float2 *prev_vec, const float2 next_vec,
                uint *n_steps, uint *idx,
                __global const bool *mask_array, __global uint *mapping_array,
                __global uint *slt_array, __global uint *slc_array)
{
    const float step_error = fast_length((*dxy2_vec-*dxy1_vec)/GRID_SCALE);

    *dl = fast_length(*dxy2_vec);
    *vec = next_vec;
    if ((*dl<(INTEGRATION_HALT_THRESHOLD)) ) {
        update_trajectory_write_sl_data(*dl,l_trajectory,*vec,*prev_vec,n_steps,
                                        idx, mask_array, slt_array, slc_array);
        return true;
    }
    *dt = select( fmin(DT_MAX,(ADJUSTED_MAX_ERROR*(*dt))/(step_error)), DT_MAX,
                 isequal(step_error,0.0f) );
    update_trajectory_write_sl_data(*dl,l_trajectory,*vec,*prev_vec,n_steps,
                                    idx, mask_array, slt_array, slc_array);
    *prev_vec = *vec;
    return false;
}
#endif

#ifdef KERNEL_INTEGRATE_FIELDS
///
/// Compute a single Euler integration step of of a streamline.
/// Record this step in the global @p trajectory_vec (array of step vectors).
/// Also update the total streamline length @p l_trajectory and
///    increment the step counter @p n_steps.
///
/// Compiled if KERNEL_INTEGRATE_FIELDS is defined.
///
/// @param[in,out]  dt: delta time step
/// @param[in,out]  dl: step distance
/// @param[in,out]  l_trajectory: total streamline distance so far
/// @param[in]      uv_vec: flow velocity vector interpolated to current position
/// @param[in,out]  vec: current (x,y) coordinate vector at tip of streamline trajectory
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
/// @retval bool:
///               @p true if stuck (step length less than @p INTEGRATION_HALT_THRESHOLD);
///               @p false otherwise aka step computed well
///
/// @ingroup integrationfns
///
static inline void euler_step_write_sl_data(float *dt, float *dl,
                                            float *l_trajectory,
                                            const float2 uv_vec, float2 *vec,
                                            const float2 prev_vec,
                                            uint *n_steps,
                                            uint *idx,
                                            __global const bool *mask_array,
                                            __global uint *slt_array,
                                            __global uint *slc_array)
{
    __private float2 sgnd_uv_vec;
    __private float dt_x, dt_y;

    sgnd_uv_vec = uv_vec*(float2)(DOWNUP_SIGN,DOWNUP_SIGN);
    dt_x = dt_to_nearest_edge((*vec)[0], sgnd_uv_vec[0]);
    dt_y = dt_to_nearest_edge((*vec)[1], sgnd_uv_vec[1]);
    *dt = minmag(dt_x,dt_y);
    *vec += sgnd_uv_vec*(*dt);
    *vec = approximate(*vec);
    *vec = (float2)( fmin(fmax((*vec)[0],-0.5f),NXF_MP5),
                     fmin(fmax((*vec)[1],-0.5f),NYF_MP5) );
    *dl = fast_length(*vec-prev_vec);
    update_trajectory_write_sl_data(*dl,l_trajectory,*vec,prev_vec,n_steps,
                                    idx, mask_array, slt_array, slc_array);
 }
#endif

#ifdef KERNEL_CONNECT_CHANNELS
///
/// Compute a single step of 2nd-order Runge-Kutta numerical integration of
///    a streamline given precomputed 1st and 2nd order step vectors.
/// If the step is deemed too small, return @p true = stuck;
///    otherwise return @p false = ok.
/// Record this step in the private @p trajectory_vec (array of step vectors).
/// Also update the total streamline length @p l_trajectory and
///    increment the step counter @p n_steps.
/// In addition, push the current @p vec to the @p prev_vec, push the current @p idx to
///    the @p prev_idx, and update the @p idx of the current @p vec.
///
/// Compiled if KERNEL_CONNECT_CHANNELS is defined.
///
/// @param[in,out]  dt: delta time step
/// @param[in,out]  dl: step distance
/// @param[in,out]  l_trajectory: total streamline distance so far
/// @param[in,out]  dxy1_vec: R-K first order delta step vector
/// @param[in,out]  dxy2_vec: R-K second order delta step vector
/// @param[in,out]  vec: current (x,y) coordinate vector at tip of streamline trajectory
/// @param[in,out]  prev_vec: previous (x,y) coordinate vector on streamline trajectory
/// @param[in,out]  next_vec: next (x,y) coordinate vector on streamline trajectory
/// @param[in,out]  n_steps: number of integration steps so far in streamline trajectory
/// @param[in,out]  idx: array index of pixel at current (x,y) position
/// @param[in,out]  prev_idx: array index of pixel at previous (x,y) position
/// @param[in,out]  trajectory_vec: streamline trajectory record
///                                  (2d array of compressed (x,y) vectors)
///
/// @retval bool:
///               @p true if stuck (step length less than @p INTEGRATION_HALT_THRESHOLD);
///               @p false otherwise aka step computed well
///
/// @ingroup integrationfns
///
static inline bool connect_runge_kutta_step_record(float *dt, float  *dl,
                                                   float *l_trajectory,
                                                   float2 *dxy1_vec, float2 *dxy2_vec,
                                                   float2 *vec, float2 *prev_vec,
                                                   float2 *next_vec,
                                                   uint *n_steps,
                                                   uint *idx, uint *prev_idx,
                                                   __private char2 *trajectory_vec)
{
    const float step_error = fast_length((*dxy2_vec-*dxy1_vec)/GRID_SCALE);

    *dl = fast_length(*dxy2_vec);
    if (*dl<(INTEGRATION_HALT_THRESHOLD)) {
        *next_vec = *vec+*dxy1_vec;
        *dl = fast_length(*dxy1_vec);
    }
    *vec = *next_vec;
    *idx = get_array_idx(*next_vec);
    if (*dl<(INTEGRATION_HALT_THRESHOLD)) {
        update_record_private_trajectory(*dl,l_trajectory,*vec,*prev_vec,n_steps,
                                         trajectory_vec);
        return true;
    }
    *dt = select( fmin(DT_MAX,(ADJUSTED_MAX_ERROR*(*dt))/(step_error)), DT_MAX,
                 isequal(step_error,0.0f) );
    update_record_private_trajectory(*dl,l_trajectory,*vec,*prev_vec,n_steps,
                                     trajectory_vec);
    *prev_vec = *vec;
    return false;
}
#endif

#ifdef KERNEL_MAP_CHANNEL_HEADS
///
/// Compute a single step of 2nd-order Runge-Kutta numerical integration of
///    a streamline given precomputed 1st and 2nd order step vectors.
/// Increment the step counter @p n_steps and update the @p idx of the new @p vec.
///
/// Compiled if KERNEL_INTEGRATE_TRAJECTORY is defined.
///
/// @param[in,out]  dt: delta time step
/// @param[in,out]  dl: step distance
/// @param[in,out]  dxy1_vec: R-K first order delta step vector
/// @param[in,out]  dxy2_vec: R-K second order delta step vector
/// @param[in,out]  vec: current (x,y) coordinate vector at tip of streamline trajectory
/// @param[in,out]  next_vec: next (x,y) coordinate vector on streamline trajectory
/// @param[in,out]  n_steps: number of integration steps so far in streamline trajectory
/// @param[in,out]  idx: array index of pixel at current (x,y) position
///
/// @retval bool:
///               @p true if stuck (step length less than @p INTEGRATION_HALT_THRESHOLD);
///               @p false otherwise aka step computed well
///
/// @ingroup integrationfns
///
static inline void channelheads_runge_kutta_step(float *dt, float *dl,
                                                 float2 *dxy1_vec, float2 *dxy2_vec,
                                                 float2 *vec, float2 *next_vec,
                                                 uint *n_steps, uint *idx)
{
    const float step_error = fast_length((*dxy2_vec-*dxy1_vec)/GRID_SCALE);

    *dl = fast_length(*dxy2_vec);
    if (*dl<(INTEGRATION_HALT_THRESHOLD)) {
        *next_vec = *vec+*dxy1_vec;
        *dl = fast_length(*dxy1_vec);
    }
    *vec = *next_vec;
    *idx = get_array_idx(*next_vec);
    *n_steps += 1u;
    *dt = select( fmin(DT_MAX,(ADJUSTED_MAX_ERROR*(*dt))/(step_error)), DT_MAX,
                 isequal(step_error,0.0f) );
    return;
}
#endif

#if defined(KERNEL_COUNT_DOWNCHANNELS) || defined(KERNEL_HILLSLOPES)
///
/// Compute a single step of 2nd-order Runge-Kutta numerical integration of
///    a streamline given precomputed 1st and 2nd order step vectors.
/// If the step is deemed too small, return @p true = stuck;
///    otherwise return @p false = ok.
/// In addition, flag the @p mapping_array pixel as @p IS_STUCK.
/// Update the @p idx of the current (new) @p vec.
///
/// Compiled if KERNEL_COUNT_DOWNCHANNELS or KERNEL_HILLSLOPES is defined.
///
/// @param[in,out]  dt: delta time step
/// @param[in,out]  dl: step distance
/// @param[in,out]  dxy1_vec: R-K first order delta step vector
/// @param[in,out]  dxy2_vec: R-K second order delta step vector
/// @param[in,out]  vec: current (x,y) coordinate vector at tip of streamline trajectory
/// @param[in,out]  next_vec: next (x,y) coordinate vector on streamline trajectory
/// @param[in,out]  idx: array index of pixel at current (x,y) position
/// @param[in,out]  mapping_array: flag grid recording status of each pixel (padded)
///
/// @retval bool:
///               @p true if stuck (step length less than @p INTEGRATION_HALT_THRESHOLD);
///               @p false otherwise aka step computed well
///
/// @ingroup integrationfns
///
static inline bool countlink_runge_kutta_step(float *dt, float *dl,
                                              float2 *dxy1_vec, float2 *dxy2_vec,
                                              float2 *vec, float2 *next_vec, uint *idx,
                                              __global uint  *mapping_array)
{
    const float step_error = fast_length((*dxy2_vec-*dxy1_vec)/GRID_SCALE);

    *dl = fast_length(*dxy2_vec);
    if (*dl<(INTEGRATION_HALT_THRESHOLD)) {
        *next_vec = *vec+*dxy1_vec;
        *dl = fast_length(*dxy1_vec);
    }
    *vec = *next_vec;
    *idx = get_array_idx(*next_vec);
    if (*dl<(INTEGRATION_HALT_THRESHOLD)) {
#ifdef DEBUG
        printf("Count-link @ %g,%g: stuck\n",(*vec)[0],(*vec)[1]);
#endif
//        atomic_or(&mapping_array[*idx],IS_STUCK);
        return true;
    }
    *dt = select( fmin(DT_MAX,(ADJUSTED_MAX_ERROR*(*dt))/(step_error)), DT_MAX,
                 isequal(step_error,0.0f) );
    return false;
}
#endif

#if defined(KERNEL_SEGMENT_HILLSLOPES) || defined(KERNEL_SUBSEGMENT_FLANKS) \
   || defined(KERNEL_FIX_RIGHT_FLANKS) || defined(KERNEL_FIX_LEFT_FLANKS)
///
/// Compute a single step of 2nd-order Runge-Kutta numerical integration of
///    a streamline given precomputed 1st and 2nd order step vectors.
/// If the step is deemed too small, return @p true = stuck;
///    otherwise return @p false = ok.
/// Increment the step counter @p n_steps and update the @p idx of the new (next) @p vec.
///
/// Compiled if KERNEL_SEGMENT_HILLSLOPES or KERNEL_SUBSEGMENT_FLANKS is defined.
///
/// @param[in,out]  dt: delta time step
/// @param[in,out]  dl: step distance
/// @param[in,out]  dxy1_vec: R-K first order delta step vector
/// @param[in,out]  dxy2_vec: R-K second order delta step vector
/// @param[in,out]  vec: current (x,y) coordinate vector at tip of streamline trajectory
/// @param[in,out]  next_vec: next (x,y) coordinate vector on streamline trajectory
/// @param[in,out]  n_steps: number of integration steps so far in streamline trajectory
/// @param[in,out]  idx: array index of pixel at current (x,y) position
///
/// @retval bool:
///               @p true if stuck (step length less than @p INTEGRATION_HALT_THRESHOLD);
///               @p false otherwise aka step computed well
///
/// @ingroup integrationfns
///
static inline bool segment_runge_kutta_step(float *dt, float *dl,
                                            float2 *dxy1_vec, float2 *dxy2_vec,
                                            float2 *vec, float2 *next_vec,
                                            uint *n_steps, uint *idx)
{
    const float step_error = fast_length((*dxy2_vec-*dxy1_vec)/GRID_SCALE);

    *dl = fast_length(*dxy2_vec);
    if (*dl<(INTEGRATION_HALT_THRESHOLD)) {
        *next_vec = *vec+*dxy1_vec;
        *dl = fast_length(*dxy1_vec);
    }
    *vec = *next_vec;
    *idx = get_array_idx(*next_vec);
    *n_steps += 1u;
    if (*dl<(INTEGRATION_HALT_THRESHOLD)) {
#ifdef DEBUG
        printf("Segment @ %g,%g: stuck\n",(*vec)[0],(*vec)[1]);
#endif
        return true;
    }
    *dt = select( fmin(DT_MAX,(ADJUSTED_MAX_ERROR*(*dt))/(step_error)), DT_MAX,
                 isequal(step_error,0.0f) );
    return false;
}
#endif

#ifdef KERNEL_HILLSLOPE_LENGTHS
///
/// Compute a single step of 2nd-order Runge-Kutta numerical integration of
///    a streamline given precomputed 1st and 2nd order step vectors.
/// If the step is deemed too small, return @p true = stuck;
///    otherwise return @p false = ok.
/// Also update the total streamline length @p l_trajectory and
///    increment the step counter @p n_steps.
/// In addition, push the current @p vec to the @p prev_vec and
///    update the @p idx of the current @p vec.
///
/// Compiled if KERNEL_HILLSLOPE_LENGTHS is defined.
///
/// @param[in,out]  dt: delta time step
/// @param[in,out]  dl: step distance
/// @param[in,out]  l_trajectory: total streamline distance so far
/// @param[in,out]  dxy1_vec: R-K first order delta step vector
/// @param[in,out]  dxy2_vec: R-K second order delta step vector
/// @param[in,out]  vec: current (x,y) coordinate vector at tip of streamline trajectory
/// @param[in,out]  prev_vec: previous (x,y) coordinate vector on streamline trajectory
/// @param[in,out]  next_vec: next (x,y) coordinate vector on streamline trajectory
/// @param[in,out]  n_steps: number of integration steps so far in streamline trajectory
/// @param[in,out]  idx: array index of pixel at current (x,y) position
///
/// @retval bool:
///               @p true if stuck (step length less than @p INTEGRATION_HALT_THRESHOLD);
///               @p false otherwise aka step computed well
///
/// @ingroup integrationfns
///
static inline bool lengths_runge_kutta_step(float *dt, float  *dl, float *l_trajectory,
                                            float2 *dxy1_vec, float2 *dxy2_vec,
                                            float2 *vec, float2 *prev_vec,
                                            float2 *next_vec,
                                            uint *n_steps, uint *idx)
{
    const float step_error = fast_length((*dxy2_vec-*dxy1_vec)/GRID_SCALE);

    *dl = fast_length(*dxy2_vec);
    if (*dl<(INTEGRATION_HALT_THRESHOLD)) {
        *next_vec = *vec+*dxy1_vec;
        *dl = fast_length(*dxy1_vec);
    }
    *vec = *next_vec;
    *idx = get_array_idx(*next_vec);
    if (*dl<(INTEGRATION_HALT_THRESHOLD)) {
        update_trajectory(*dl,l_trajectory,n_steps);
        return true;
    }
    *dt = select( fmin(DT_MAX,(ADJUSTED_MAX_ERROR*(*dt))/(step_error)), DT_MAX,
                 isequal(step_error,0.0f) );
    update_trajectory(*dl,l_trajectory,n_steps);
    *prev_vec = *vec;
    return false;
}
#endif
