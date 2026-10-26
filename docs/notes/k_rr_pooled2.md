# Notes: `k_rr_pooled2`

## Recovered oracle launch

- **Launch**: frame 1, launch 1, `_ZN4rrio9k_pooled2EPKjiiPjii`, grid [12, 7, 1], block [16, 16, 1]. Frame 0 resets history and skips this kernel.
- **Arguments**: input E0c0e [u32], 384×216, 331,776 bytes; output E0c12 [u32], 192×108, 82,944 bytes. The arguments are pointer, input W/H, pointer, output W/H.
- **ISA evidence**: each output coordinate is expanded to a 2×2 input footprint. Each packed u32 is split into three 10-bit fields at bit offsets 0, 10, and 20, then each field is normalized by 1023; the two high bits are not read here. This is not R11G11B10F, despite RR's gather format code 1 being that DXGI format. The arithmetic clamps the tone-mapped channels, applies the inverse tone map, pools, then applies the forward transform and packs the result. The producer's exact format / alpha handling remains to be confirmed. The body is hundreds of instructions, so do not substitute a nearest sample or generic average without matching the order and conversions.
- **Current port mismatch**: `shaders/rr/rr_pooled2.comp` declares f16vec4 input/output, samples one coordinate, and multiplies by an x-dependent factor. That is not the oracle ABI or algorithm. No kernel comparison was run because the present shader would read beyond the true 4-byte-per-pixel input; first replace it with the packed-u32 2×2 transform, then compare frame 1 launch 1 using `diffdump.py inputs`.

No attempt budget has been spent on this mismatch yet; the next action is the first decoded implementation hypothesis, with the per-channel 10-bit unpack and tonemap order taken from `build/rr_pooled2_isa.s`.
