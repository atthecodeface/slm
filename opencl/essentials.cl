/// @file essentials.cl
///
/// Essential functions for streamline trajectory integration
///

///
/// @defgroup utilities Utility functions
/// Functions used frequently by kernels
///

#ifdef C_ORDER
/// Bilinearly interpolate a velocity vector (choice of row-major or column-major arrays).
///
/// Perform a fast, simple bilinear interpolation at arbitrary position vector
/// from regular grid of velocity vectors in global uv_array with ordering type
/// set by macro definitions C_ORDER or F_ORDER.
///
/// ifdef C_ORDER: Assumes a C-order, row-major, last-index-fastest array type.
///
/// @param[in] vec      (float2 *, RO): real-valued vector position (x,y) onto which to
///                                     interpolate (u,v)
/// @param[in] uv_array (float2 *, RO): gridded velocity vector components (u,v)
///
/// @returns  normalized velocity vector (u,v) sampled at position vec (x,y)
///
/// @ingroup utilities
///
static float2 speed_interpolator(float2 vec, __global const float2 *uv_array)
{
    const uint x_lft = min( NX_PADDED-1, (uint)(max(0.0f, vec[0]+PAD_WIDTH_PP5)));
    const uint y_dwn = min( NY_PADDED-1, (uint)(max(0.0f, vec[1]+PAD_WIDTH_PP5)));
    const uint x_rgt = min( NX_PADDED-1, x_lft+1u );
    const uint y_upp = min( NY_PADDED-1, y_dwn+1u );

    // Get the fractional displacement of the sample point from the down-left vertex
    const float rx_weight = vec[0]-(float)x_lft+(float)PAD_WIDTH;
    const float lx_weight = 1.0f-rx_weight;
    const float uy_weight = vec[1]-(float)y_dwn+(float)PAD_WIDTH;
    const float dy_weight = 1.0f-uy_weight;

    // Use to weight the four corner values...
    const float2 uv_dwn = uv_array[NY_PADDED*x_lft+y_dwn]*lx_weight
                        + uv_array[NY_PADDED*x_rgt+y_dwn]*rx_weight;
    const float2 uv_upp = uv_array[NY_PADDED*x_lft+y_upp]*lx_weight
                        + uv_array[NY_PADDED*x_rgt+y_upp]*rx_weight;
    // Returns:
    //    interpolated 2d unit speed vector:
    return fast_normalize(uv_dwn*dy_weight+uv_upp*uy_weight);
}

/// Compute the array index of the padded grid pixel pointed to by
/// a float2 grid position vector (choice of row-major or column-major arrays).
///
/// ifdef C_ORDER: Assumes a C-order, row-major, last-index-fastest array type.
///
/// @param[in]  vec (float2 *, RO): real-valued vector position (x,y)
///
/// @returns  padded grid array index at position vec (x,y)
///
/// @ingroup utilities
///
static inline uint get_array_idx(float2 vec) {
    return          ( min( NY_PADDED-1, (uint)(max(0.0f, vec[1]+PAD_WIDTH_PP5)) )
           +NY_PADDED*min( NX_PADDED-1, (uint)(max(0.0f, vec[0]+PAD_WIDTH_PP5)) ) );
}
#endif

#ifdef F_ORDER
/// Bilinearly interpolate a velocity vector (choice of row-major or column-major arrays).
///
/// ifdef F_ORDER: Assumes a Fortran-order, column-major, first-index-fastest array type.
///
/// @param[in] vec      (float2 *, RO): real-valued vector position (x,y) onto which to
///                                     interpolate (u,v)
/// @param[in] uv_array (float2 *, RO): gridded velocity vector components (u,v)
///
/// @returns  normalized velocity vector (u,v) at position vec (x,y)
///
/// @ingroup utilities
///
static float2 speed_interpolator(float2 vec,
                                __global const float2 *uv_array)
{
    const uint x_lft = (uint)(max(0.0f, vec[0]+PAD_WIDTH_PP5));
    const uint y_dwn = (uint)(max(0.0f, vec[1]+PAD_WIDTH_PP5));
    const uint x_rgt = min( NX_PADDED-1, x_lft+1u );
    const uint y_upp = min( NY_PADDED-1, y_dwn+1u );

    // Get the fractional displacement of the sample point from the down-left vertex
    const float rx_weight = vec[0]-(float)x_lft+(float)PAD_WIDTH;
    const float lx_weight = 1.0f-rx_weight;
    const float uy_weight = vec[1]-(float)y_dwn+(float)PAD_WIDTH;
    const float dy_weight = 1.0f-uy_weight;

    // Use to weight the four corner values...
    const float2 uv_dwn = uv_array[x_lft+NX_PADDED*y_dwn]*lx_weight
                        + uv_array[x_rgt+NX_PADDED*y_dwn]*rx_weight;
    const float2 uv_upp = uv_array[x_lft+NX_PADDED*y_upp]*lx_weight
                        + uv_array[x_rgt+NX_PADDED*y_upp]*rx_weight;
    // Returns:
    //    interpolated 2d unit speed vector:
    return fast_normalize(uv_dwn*dy_weight+uv_upp*uy_weight);
}

/// Compute the array index of the padded grid pixel pointed to by
/// a float2 grid position vector (choice of row-major or column-major arrays).
///
/// ifdef F_ORDER: Assumes a Fortran-order, column-major, first-index-fastest array type.
///
/// @param[in]  vec (float2 *, RO): real-valued vector position (x,y)
///
/// @returns  padded grid array index at position vec (x,y)
///
/// @ingroup utilities
///
static inline uint get_array_idx(float2 vec) {
    return          ( min( NX_PADDED-1, (uint)(max(0.0f, vec[0]+PAD_WIDTH_PP5)) )
           +NX_PADDED*min( NY_PADDED-1, (uint)(max(0.0f, vec[1]+PAD_WIDTH_PP5)) ) );
}
#endif

/// Squish a float vector into a byte vector for O(<1 pixel) trajectory steps
/// Achieved through scaling by TRAJECTORY_RESOLUTION, e.g. x128,
/// and being content with 1/TRAJECTORY_RESOLUTION resolution.
///
/// @param[in]  raw_vector (float2): vector (trajectory step) to be compressed
///
/// @retval  char2 vector (trajectory step) in compressed (byte,byte) form
///
/// @ingroup utilities
///
static char2 compress(float2 raw_vector) {
    return (char2)(raw_vector[0]*TRAJECTORY_RESOLUTION,
                   raw_vector[1]*TRAJECTORY_RESOLUTION);
}

/// Unsquish a byte vector back to a float vector.
/// Used in connect_channels() to unpack a streamline trajectory.
///
/// @param[in]  compressed_vector (char2): vector (trajectory step) in
///                                        compressed (byte,byte) form
///
/// @retval  float2: uncompressed vector (trajectory step)
///
/// @ingroup utilities
///
static float2 uncompress(char2 compressed_vector) {
    return ((float2)(compressed_vector[0],
                     compressed_vector[1]))/TRAJECTORY_RESOLUTION;
}

/// Approximate a float vector at the resolution provided by a scaled byte vector.
/// Do this by compressing and then uncompressing at the TRAJECTORY_RESOLUTION,
///    which is usually 128.
/// This function is useful in trajectory step integration to make sure
///    that the progressively recorded, total trajectory length matches
///    the reduced resolution trajectory step sequence.
///
/// @param[in]  raw_position (float2): vector (trajectory step) to be approximated
///
/// @retval  float2: reduced-resolution vector
///
/// @ingroup utilities
///
static float2 approximate(float2 raw_position) {
    return ((float2)(
            (char)(raw_position[0]*TRAJECTORY_RESOLUTION),
            (char)(raw_position[1]*TRAJECTORY_RESOLUTION)
            ))/TRAJECTORY_RESOLUTION;
}

/// In Euler streamline integration (which is the last step), this function
///    provides the delta time required to reach the boundary precisely in one hop.
///
/// @param[in] x (float): position vector (the current point on the trajectory)
/// @param[in] u (float): flow speed vector (to be integrated to the grid boundary)
///
/// @retval  float: delta time that will integrate the flow vector onto the boundary
///
/// @ingroup utilities
///
static inline float dt_to_nearest_edge(float x,float u) {
    float dt = 0.0f;
    // going right?
    dt = select(dt,((int)(x+1.5f)-(x+0.5f))/u, isgreater(u,0.0f));
    // going left?
    dt = select(dt,-((x+0.5f)-(int)(x+0.5f))/u, isless(u,0.0f));
    // if dt=0 stuck
    return dt;
}
