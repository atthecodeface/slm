/// @file info.h
///
/// Dummy example header that mimics the macro parameters passed to the CL compiler
///
/// In reality these macros are set in the Python wrapper files, e.g., @p integration.py,
/// in the variable @p compile_options, itself set by @p pocl.set_compile_options().
/// This example reflects the set of -D macro flags passed when building
/// the kernel integrate_trajectory() in integration.cl.

#define KERNEL_INTEGRATE_TRAJECTORY
#define C_ORDER
#define DOWNUP_SIGN  1
#define INTEGRATOR_STEP_FACTOR  0.5f
#define MAX_INTEGRATION_STEP_ERROR  0.029999999329447746f
#define ADJUSTED_MAX_ERROR  0.1472243219614029f
#define MAX_LENGTH  300.0f
#define PIXEL_SIZE  1.0f
#define INTEGRATION_HALT_THRESHOLD  0.009999999776482582f
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
#define GRID_SCALE  200.0f
#define COMBO_FACTOR  100.0f
#define DT_MAX  0.004999999888241291f
#define MAX_N_STEPS  600u
#define TRAJECTORY_RESOLUTION  128u
#define SEEDS_CHUNK_OFFSET  0u
#define SUBPIXEL_SEED_POINT_DENSITY  5u
#define SUBPIXEL_SEED_HALFSPAN  0.4000000059604645f
#define SUBPIXEL_SEED_STEP  0.20000000298023224f
#define JITTER_MAGNITUDE  2.9000000953674316f
#define INTERCHANNEL_MAX_N_STEPS  200u
#define SEGMENTATION_THRESHOLD  50u
#define LEFT_FLANK_ADDITION  2147483648u
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
