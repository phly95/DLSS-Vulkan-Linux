# Notes: `k_rr_pooled2`

## Recovered oracle launch

- **Launch**: frame 1, launch 1, `_ZN4rrio9k_pooled2EPKjiiPjii`, grid [12, 7, 1], block [16, 16, 1]. Frame 0 resets history and skips this kernel.
- **Arguments**: input E0c0e [u32], 384×216, 331,776 bytes; output E0c12 [u32], 192×108, 82,944 bytes. The arguments are pointer, input W/H, pointer, output W/H.
- **ISA evidence**: each output coordinate is expanded to a 2×2 input footprint. Four packed RGB values are unpacked from R11G11B10; the arithmetic clamps the tone-mapped channels, applies the inverse tone map, pools, then applies the forward transform and packs the result. The body is hundreds of instructions and includes explicit exponent/mantissa conversion, so do not substitute a nearest sample or generic average without matching the order and conversions.
- **Current port mismatch**: `shaders/rr/rr_pooled2.comp` declares f16vec4 input/output, samples one coordinate, and multiplies by an x-dependent factor. That is not the oracle ABI or algorithm. No kernel comparison was run because the present shader would read beyond the true 4-byte-per-pixel input; first replace it with the packed-u32 2×2 transform, then compare frame 1 launch 1 using `diffdump.py inputs`.

No attempt budget has been spent on this mismatch yet; the next action is the first decoded implementation hypothesis, with the per-channel unpack/tonemap order taken from `build/rr_pooled2_isa.s`.
