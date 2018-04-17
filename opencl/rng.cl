///
/// @file rng.cl
///
/// Lehmer linear-congruential random number generator functions (uint and float2 vector)
///
/// @author CPS
/// @bug No known bugs
///


/// Flag whether this RNG is included in the kernel source: the jittered streamline
/// integration function compute_step_vec_jittered() is not compiled unless the flag
/// is set.
#define IS_RNG_AVAILABLE 1

///
/// Generate a Lehmer (linear congruential) integer random variate.
/// The revised
/// <a href="http://www.firstpr.com.au/dsp/rand31/p1192-park.pdf">Park & Miller (1998)</a>
///  version is implemented here.
/// Refer to the <a href="https://en.wikipedia.org/wiki/Lehmer_random_number_generator">
///     Wikipedia page about this RNG</a> for more information.
///
/// The passed-in @p rng_state variable acts as a seed and a container to return
///    the subsequent state of the RNG aka the 32-bit unsigned integer random variate.
/// 64-bit arithmetic required to avoid overflow, although more labored 32-bit versions
///    exist.
/// The generator parameter used here is the prime modulus @f$2^{32}-5@f$.
///
/// @param[in,out] rng_state: pointer to the RNG state which is also the current integer
///                           variate
///
/// @retval uint: value of random 32-bit integer aka RNG state
///
/// @ingroup utilities
///
static uint lehmer_rand_uint(uint *rng_state)
{
    // Lehmer linear-congruential RNG (revised, 'extended' version)

    // Store the current number in the sequence and use as 'seed' next time
    *rng_state = (uint)( ((unsigned long)(*rng_state+1u)*279470273u) % 0xfffffffb );
    // Return as unsigned 32-bit integer
    return *rng_state;
}

///
/// Generate a Lehmer RNG float2 vector random variate @f$[-0.5,0.5)\times 2@f$.
///
/// The passed-in @p rng_state variable acts as a seed and a container to return
///    the subsequent state of the RNG.
/// The float2 vector random variate is returned explicitly.
///
/// @param[in,out] rng_state: pointer to the RNG state which is also the current integer
///                           variate
///
/// @retval float2: two random 32-bit float values each @f$[-0.5,0.5)@f$
///                 as a float2 vector
///
/// @ingroup utilities
///
static float2 lehmer_rand_vec(uint *rng_state)
{
    // Generate two uniform [-0.5,0.5) pseudo-random numbers in a float 2-vector
    return (float2)( (float)lehmer_rand_uint(rng_state)/(float)0xfffffffb-0.5f,
                     (float)lehmer_rand_uint(rng_state)/(float)0xfffffffb-0.5f);
}
