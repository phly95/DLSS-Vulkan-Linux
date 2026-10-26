# Notes: `k_rr_pooled2`

## Recovered oracle launch

- **Launch**: frame 1, launch 1, `_ZN4rrio9k_pooled2EPKjiiPjii`, grid [12, 7, 1], block [16, 16, 1]. Frame 0 resets history and skips this kernel.
- **Arguments**: input E0c0e [u32], 384×216, 331,776 bytes; output E0c12 [u32], 192×108, 82,944 bytes. The arguments are pointer, input W/H, pointer, output W/H.
- **ISA evidence**: each output coordinate is expanded to a 2×2 input footprint. Each packed u32 is split into three 10-bit fields at bit offsets 0, 10, and 20, then each field is normalized by 1023; the two high bits are not read here. This is not R11G11B10F, despite RR's gather format code 1 being that DXGI format. The arithmetic clamps the tone-mapped channels, applies the inverse tone map, pools, then applies the forward transform and packs the result. The producer's exact format / alpha handling remains to be confirmed. The body is hundreds of instructions, so do not substitute a nearest sample or generic average without matching the order and conversions.
- **Initial port mismatch**: the old `shaders/rr/rr_pooled2.comp` declared f16vec4 input/output and read beyond this launch's true 4-byte-per-pixel input. It has been replaced with a packed-u32 placeholder and the O7 fixture now uses the actual input/output element types and 2x dimensions.

## Attempt log

### Attempt 1: equal-weight inverse-tone-map pool
- **Hypothesis**: decode the three 10-bit channels, scale by 2.45, clamp, inverse-tone-map, average the 2x2 footprint equally, then apply the inverse function and quantize to 10-bit channels.
- **Implementation**: `shaders/rr/rr_pooled2.comp` (placeholder; alpha/count normalization and exact output conversion not yet decoded).
- **Test**: RADV RX 9070 XT `kernel_run` on frame 1 launch 1, using `diffdump.py inputs` for E0c0e [u32] and comparing E0c12 [u32].
- **Result**: 0/20,736 elements bit-exact; max raw-u32 absolute difference 441,908,672; max ULP 441,908,672; first mismatch index 0, port `0xc501f889`, oracle `0xc210cc38`.
- **Conclusion**: this simple equal-weight transform is wrong. The ISA's alpha/count handling or output scaling/rounding is essential. No second hypothesis has been tested yet.
