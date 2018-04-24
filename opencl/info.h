/// @file info.h
///
/// Dummy example header that mimics the macro parameters passed to the CL compiler
///
/// In reality these macros are set in the Python wrapper files, e.g., @p integration.py,
/// in the variable @p compile_options, itself set by @p pocl.set_compile_options().
/// This example reflects the set of

#define macro flags passed when building
/// the kernel integrate_trajectory() in integration.cl.

///
/// @defgroup kernelflags Kernel instance control options
/// Basic kernel compilation and seed point offset information.
/// Communicated to CL kernels via compiler.

#define macro options.
///

/// @{
#define SEEDS_CHUNK_OFFSET  0u
#define KERNEL_INTEGRATE_TRAJECTORY
#define KERNEL_MAP_CHANNEL_HEADS
#define KERNEL_PRUNE_CHANNEL_HEADS
#define KERNEL_CONNECT_CHANNELS
#define KERNEL_PUSH_TO_EXIT
#define KERNEL_COUNT_DOWNCHANNELS
#define KERNEL_FLAG_DOWNCHANNELS
#define KERNEL_LINK_HILLSLOPES
#define KERNEL_SEGMENT_DOWNCHANNELS
#define KERNEL_SEGMENT_HILLSLOPES
#define KERNEL_SUBSEGMENT_CHANNEL_EDGES
#define KERNEL_SUBSEGMENT_FLANKS
#define KERNEL_HILLSLOPE_LENGTHS
/// @}

///
/// @defgroup intflags Streamline integration parameters
/// Parameters used to control R-K streamline integration behavior.
/// Communicated to CL kernels via compiler.

#define macro options.
///

/// @{
#define DOWNUP_SIGN  1
#define INTEGRATOR_STEP_FACTOR  0.5f
#define MAX_INTEGRATION_STEP_ERROR  0.029999999329447746f
#define ADJUSTED_MAX_ERROR  0.1472243219614029f
#define MAX_LENGTH  300.0f
#define INTEGRATION_HALT_THRESHOLD  0.009999999776482582f
#define GRID_SCALE  200.0f
#define COMBO_FACTOR  100.0f
#define DT_MAX  0.004999999888241291f
#define MAX_N_STEPS  600u
#define TRAJECTORY_RESOLUTION  128u
#define INTERCHANNEL_MAX_N_STEPS  200u
/// @}

///
/// @defgroup arrayflags Parameters describing grid array geometry, size, ordering
/// Parameters describing DTM grid array geometry, size, ordering, padding, etc.
/// Communicated to CL kernels via compiler.

#define macro options.
///

/// @{
#define C_ORDER
#define PIXEL_SIZE  1.0f
#define PAD_WIDTH  1u
#define PAD_WIDTH_PP5  1.5f
#define NX  200u
#define NY  200u
#define NXF  200.0f
#define NYF  200.0f
#define NX_PADDED  202u
#define NY_PADDED  202u
#define X_MAX    199.5f
#define Y_MAX  199.5f
/// @}

///
/// @defgroup trajflags Jittered trajectory integration control parameters
/// Parameters used to control sub-pixel, jittered streamline integration.
/// Communicated to CL kernels via compiler.

#define macro options.
///

/// @{
#define SUBPIXEL_SEED_POINT_DENSITY  5u
#define SUBPIXEL_SEED_HALFSPAN  0.4000000059604645f
#define SUBPIXEL_SEED_STEP  0.20000000298023224f
#define JITTER_MAGNITUDE  2.9000000953674316f
/// @}


///
/// @defgroup mapflags Mapping flags and control parameters
/// Mapping grid-pixel flags provided by @p mapping_array; mapping control parameters.
/// Communicated to CL kernels via compiler.

#define macro options.
///

/// @{
#define IS_CHANNEL  1u
#define IS_THINCHANNEL  2u
#define IS_INTERCHANNEL  4u
#define IS_CHANNELHEAD  8u
#define IS_CHANNELTAIL  16u
#define IS_MAJORCONFLUENCE  32u
#define IS_MINORCONFLUENCE  64u
#define IS_MAJORINFLOW  128u
#define IS_MINORINFLOW  256u
#define IS_LEFTFLANK  512u
#define IS_RIGHTFLANK  1024u
#define IS_MIDSLOPE  2048u
#define IS_RIDGE  4096u
#define IS_STUCK  8192u
#define IS_LOOP  16384u
#define IS_BLOCKAGE  32768u
#define LEFT_FLANK_ADDITION  2147483648u
#define SEGMENTATION_THRESHOLD  50u
/// @}



///
/// @defgroup kdeflags PDF kernel-density estimation flags and control parameters
/// PDF kernel-density estimation flags and control parameters.
/// Communicated to CL kernels via compiler.

#define macro options.
///

/// @{
#define KERNEL_HISTOGRAM_UNIVARIATE
#define KERNEL_HISTOGRAM_BIVARIATE
#define KERNEL_PDF_UNIVARIATE
#define KERNEL_PDF_BIVARIATE
#define KDF_BANDWIDTH=0.4000000059604645f
#define KDF_IS_TOPHAT
#define KDF_IS_TRIANGLE
#define KDF_IS_EPANECHNIKOV
#define KDF_IS_COSINE
#define KDF_IS_GAUSSIAN
#define N_DATA=202u
#define N_HIST_BINS=2000u
#define N_PDF_POINTS=200u
#define X_MIN=0.0f
#define X_MAX=4.368579864501953f
#define X_RANGE=4.368579864501953f
#define BIN_DX=0.00218428997322917f
#define PDF_DX=0.021842898800969124f
#define KDF_WIDTH_X=2.269500255584717f
#define N_KDF_PART_POINTS_X=51u
#define Y_MIN=-0.6931471824645996f
#define Y_MAX=5.776393413543701f
#define Y_RANGE=6.469540596008301f
#define BIN_DY=0.003234770381823182f
#define PDF_DY=0.032347701489925385f
#define KDF_WIDTH_Y=2.4467508792877197f
#define N_KDF_PART_POINTS_Y=37u
/// @}

