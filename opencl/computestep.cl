///
/// @file computestep.cl
///
/// Compute single streamline integration step.
///
/// @author CPS
///

#if defined(KERNEL_INTEGRATE_TRAJECTORY) || defined(KERNEL_CONNECT_CHANNELS) \
    || defined(KERNEL_MAP_CHANNEL_HEADS) || defined(KERNEL_COUNT_DOWNCHANNELS) \
    || defined(KERNEL_LINK_HILLSLOPES)   || defined(KERNEL_SEGMENT_HILLSLOPES) \
    || defined(KERNEL_SUBSEGMENT_FLANKS) || defined(KERNEL_HILLSLOPE_LENGTHS)
///
/// Compute a 2nd-order Runge-Kutta integration step along a streamline.
///
/// Compiled if any of the following are defined:
///    - KERNEL_INTEGRATE_TRAJECTORY
///    - KERNEL_CONNECT_CHANNELS
///    - KERNEL_MAP_CHANNEL_HEADS
///    - KERNEL_COUNT_DOWNCHANNELS
///    - KERNEL_LINK_HILLSLOPES
///    - KERNEL_SEGMENT_HILLSLOPES
///    - KERNEL_SUBSEGMENT_FLANKS
///    - KERNEL_HILLSLOPE_LENGTHS
///
/// @param[in]      dt: delta time step
/// @param[in]      uv_array  (float *,  RO): gridded velocity vector components (u,v)
/// @param[in,out]  dxy1_vec: R-K first order delta step vector
/// @param[in,out]  dxy2_vec: R-K second order delta step vector
/// @param[in,out]  uv1_vec: flow velocity vector at current coordinate (at @p vec)
/// @param[in,out]  uv2_vec: flow velocity vector at RK1 stepped coordinate
///                           (at @p vec + @p dxy1_vec)
/// @param[in]      vec: current (x,y) coordinate vector at tip of streamline trajectory
/// @param[in,out]  next_vec: next (x,y) coordinate vector on streamline trajectory
/// @param[in,out]  idx: array index of pixel at current (x,y) position
///
/// @returns void
///
/// @ingroup trajectoryfns
///
static inline void compute_step_vec(const float dt,
                                    const __global float2 *uv_array,
                                    float2 *dxy1_vec, float2 *dxy2_vec,
                                    float2 *uv1_vec, float2 *uv2_vec,
                                    const float2 vec,
                                    float2 *next_vec, uint *idx) {
    // Calculate RK2 next pt vector and approx into a fixed-point-res vector.
    // Do this using randomly biased =jittered flow vector field.
    // Then get the next pixel's data array index.
    *uv1_vec = speed_interpolator(vec,uv_array);
    *dxy1_vec = approximate(*uv1_vec*COMBO_FACTOR*dt);
    *uv2_vec = speed_interpolator(vec+*dxy1_vec,uv_array);
    *dxy2_vec = approximate(0.5f*(*dxy1_vec+*uv2_vec*COMBO_FACTOR*dt));
    *next_vec = vec+*dxy2_vec;
    *idx = get_array_idx(*next_vec);
}
#endif

#if defined(KERNEL_INTEGRATE_TRAJECTORY) && defined(IS_RNG_AVAILABLE)
///
/// Compute a jittered 2nd-order Runge-Kutta integration step along a streamline.
/// Jittering is achieved by adding a uniform random vector to of the RK2 flow velocity
///   vectors, then normalizing to provide two RK2 unit step vectors.
/// The [0,1) uniform variates are scaled by JITTER_MAGNITUDE before addition
///   to the unit flow vectors.
///
/// Compiled if KERNEL_INTEGRATE_TRAJECTORY and IS_RNG_AVAILABLE is defined.
///
/// @param[in]      dt: delta time step
/// @param[in]      uv_array  (float *,  RO): gridded velocity vector components (u,v)
/// @param[in,out]  rng_state: RNG state (thus initally the seed) and RNG variate
/// @param[in,out]  dxy1_vec: R-K first order delta step vector
/// @param[in,out]  dxy2_vec: R-K second order delta step vector
/// @param[in,out]  uv1_vec: flow velocity vector at current coordinate (at @p vec)
/// @param[in,out]  uv2_vec: flow velocity vector at RK1 stepped coordinate
///                           (at @p vec + @p dxy1_vec)
/// @param[in]      vec: current (x,y) coordinate vector at tip of streamline trajectory
/// @param[in,out]  next_vec: next (x,y) coordinate vector on streamline trajectory
/// @param[in,out]  idx: array index of pixel at current (x,y) position
///
/// @returns void
///
/// @ingroup trajectoryfns
///
static inline void compute_step_vec_jittered(const float dt,
                                             const __global float2 *uv_array,
                                             uint *rng_state,
                                             float2 *dxy1_vec, float2 *dxy2_vec,
                                             float2 *uv1_vec, float2 *uv2_vec,
                                             const float2 vec, float2 *next_vec,
                                             uint *idx) {
    // Calculate RK2 next pt vector and approx into a fixed-point-res vector.
    // Do this using randomly biased aka jittered flow vector field.
    // Then get the next pixel's data array index.
    *uv1_vec = speed_interpolator(vec,uv_array);
    *uv1_vec += lehmer_rand_vec(rng_state)*JITTER_MAGNITUDE;
    *uv1_vec /= fast_length(*uv1_vec);
    *dxy1_vec = approximate(*uv1_vec*COMBO_FACTOR*dt);
    *uv2_vec = speed_interpolator(vec+*dxy1_vec,uv_array);
    *uv2_vec += lehmer_rand_vec(rng_state)*JITTER_MAGNITUDE;
    *uv2_vec /= fast_length(*uv2_vec);
    *dxy2_vec = approximate(0.5f*(*dxy1_vec+*uv2_vec*COMBO_FACTOR*dt));
    *next_vec = vec+*dxy2_vec;
    *idx = get_array_idx(*next_vec);
}
#endif
