# Notes: k_sr_post

## Launch Details
- Launch 14 in `sr_M` frame 0: `_ZN2sr2io9k_sr_postILb0ELi12ELi16ELb0ELb0EEEvNS0_8PostArgsE`
- Grid: [48, 14, 1], Block: [256, 1, 1]
- Inputs: `g_col`, `expo`, `gates`, `factors`, `aniso`, `ihist0`, `prev_mvec1`, `mv_ero`, `mv_nondil`
- Outputs: `dhist` (u32, 82944), `full15` (u32, 331776), `ihist1` (f16vec4, 331776)

## Attempts and Hypotheses

### Attempt 2
- **Hypothesis**: Decoded coordinate transform and tonemapping formats from GFX1201 ISA:
  - Input sample coordinate: `floor((float(x15) + 0.5) * scaleX + jitterX - 0.5)` with clamp to `[0, W-1]`, using `scaleX`, `scaleY`, `jitterX`, `jitterY` unpacked from push constants.
  - `ihist1`: tonemapped history `0.36 * log10(max(1.0 + 100.0 * c * e, 0.0001))` in float16 with alpha = 0.0.
  - `dhist`: 2x2 box average at half resolution (384x216), tonemapped and packed to 10:10:10 format `ur | (ug << 10) | (ub << 20)`.
  - `full15`: R11G11B10 format with rounded float scaling.
- **Result**:
  - `ihist1 [f16]`: 1327104 elements, bit-exact 373589 (28.1507%), max |err| 0.6423, max ulp 7268, first mismatch `{'index': 4, 'port': 0.0946044921875, 'oracle': 0.0960693359375, 'port_bits': 11790, 'oracle_bits': 11814}` (Raw byte-exact: 1,244,196 / 2,654,208 bytes = 46.88% bit-exact).
  - `dhist [u8]`: 331776 elements, bit-exact 84930 (25.5986%), max |err| 255.
  - `full15 [u8]`: 1327104 elements, bit-exact 240463 (18.1194%), first mismatch index 4 `port: 79.0, oracle: 80.0` (1 LSB off).

### Attempt 3
- **Hypothesis**: The workgroup block is 256 threads covering a 16x32 pixel tile (grid [48, 14, 1], 14 * 32 = 448 >= 432). Each thread processes TWO pixels in Y (`ly = (tid >> 4) * 2`, pixel 0 at `y0 = gy + ly`, pixel 1 at `y1 = y0 + 1`). Previously only pixel 0 was processed up to row 223, leaving half the screen unwritten. Fix processes both pixels and writes all 384x216 cells of `dhist`.
- **Result**: Complete coverage of 768x432 output and 384x216 dhist output.
- **Pipestats**: 108 SGPRs, 24 VGPRs, 0 spilled SGPRs, 0 spilled VGPRs, 0 scratch size.

