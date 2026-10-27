// rh.glsl: explicit f16 rounding (round to nearest, ties to even) for RADV.
//
// On RADV float(float16_t(x)) does not round: the f32 -> f16 -> f32 round trip is folded away
// (8059 of 8064 test values came back unrounded), and packHalf2x16 does not round to nearest even.
// Only a 16-bit memory store or load rounds. These helpers round with integer bit operations, so
// every f16 rounding point of NeuralHub's ISA (v_cvt_f16_f32, v_fma_f16, v_fmac_f16) is written out.
// sr_stage.comp carries the same rh(); this header is the shared copy for the other shaders.

#ifndef RH_GLSL
#define RH_GLSL

// rh(x): x rounded to the nearest f16 value, returned as f32. Normal range and subnormals; no inf or
// NaN handling (the port's values stay well inside the f16 range).
float rh(float x) {
    uint b = floatBitsToUint(x);
    uint sgn = b & 0x80000000u;
    float a = abs(x);
    if (a >= 6.103515625e-05) {                       // normal f16 (>= 2^-14): 11 significant bits
        uint ab = b & 0x7fffffffu;
        uint r = (ab + 0xfffu + ((ab >> 13u) & 1u)) & 0xffffe000u;
        return uintBitsToFloat(sgn | r);
    }
    float q = roundEven(a * 16777216.0) * 5.9604644775390625e-08;   // subnormal f16: multiples of 2^-24
    return (x < 0.0) ? -q : q;
}

vec3 rh3(vec3 v) {
    return vec3(rh(v.x), rh(v.y), rh(v.z));
}

#endif
