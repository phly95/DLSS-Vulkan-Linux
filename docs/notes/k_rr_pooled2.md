# Notes: `k_rr_pooled2`

## Recovered oracle launch

- **Launch**: frame 1, launch 1, `_ZN4rrio9k_pooled2EPKjiiPjii`, grid [12, 7, 1], block [16, 16, 1]. Frame 0 resets history and skips this kernel.
- **Arguments**: input E0c0e [u32], 384×216, 331,776 bytes; output E0c12 [u32], 192×108, 82,944 bytes. The arguments are pointer, input W/H, pointer, output W/H.
- **ISA evidence**: each output coordinate is expanded to a 2×2 input footprint. Each packed u32 is split into three 10-bit fields at bit offsets 0, 10, and 20, then each field is normalized by 1023; the two high bits are not read here. This is not R11G11B10F, despite RR's gather format code 1 being that DXGI format. The arithmetic clamps the tone-mapped channels, applies the inverse tone map, pools, then applies the forward transform and packs the result. The producer's exact format / alpha handling remains to be confirmed. The body is hundreds of instructions, so do not substitute a nearest sample or generic average without matching the order and conversions.
- **Initial port mismatch**: the old `shaders/rr/rr_pooled2.comp` declared f16vec4 input/output and read beyond this launch's true 4-byte-per-pixel input. It has been replaced with the packed-u32 implementation and the O7 fixture now uses the actual input/output element types and 2x dimensions.

## Attempt log

### Attempt 1: equal-weight inverse-tone-map pool
- **Hypothesis**: decode the three 10-bit channels, scale by 2.45, clamp, inverse-tone-map, average the 2x2 footprint equally, then apply the inverse function and quantize to 10-bit channels.
- **Implementation**: `shaders/rr/rr_pooled2.comp`, packed-u32 input/output, ISA unpack and inverse transform, equal-weight pooling, and the initial forward-transform attempt.
- **Test**: RADV RX 9070 XT `kernel_run` on frame 1 launch 1, using `diffdump.py inputs` for E0c0e [u32] and comparing E0c12 [u32].
- **Result**: 0/20,736 elements bit-exact; max raw-u32 absolute difference 441,908,672; max ULP 441,908,672; first mismatch index 0, port `0xc501f889`, oracle `0xc210cc38`.
- **Conclusion**: the inverse transform and equal-weight 2x2 pool were consistent with the ISA; the output re-encode was missing the reciprocal input scale 1/2.45. The first output's three channels differed by approximately 2.45x. All four footprint samples are in bounds for the measured launch.

### Attempt 2: apply the ISA's full forward-tone-map scale
- **Hypothesis**: after averaging the inverse-tone-mapped 2x2 samples, the output path multiplies `log2(100*x + 1)` in this order by `log10(2)`, `0.36`, and `1/2.45`, then rounds the three values to 10-bit channels. The last factor was missing from attempt 1.
- **Implementation**: `shaders/rr/rr_pooled2.comp`, using the ISA bit constants `0x3e9a209b`, `0x3eb851ec`, and `0x3ed0fac6` in that order.
- **Test**: RADV RX 9070 XT `kernel_run` on frame 1 launch 1 with exact prelaunch E0c0e [u32], comparing E0c12 [u32].
- **Result**: `DIFFDUMP: BIT-EXACT`; 20,736/20,736 elements (100.0000%), max absolute error 0, max ULP 0, no first mismatch.
- **Disposition**: `k_rr_pooled2` is EXACT for its observed frame 1 launch. Frame 0 skips this kernel; the restored oracle dump contains no additional pooled2 launches.
