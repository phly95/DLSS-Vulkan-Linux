# Notes: k_sr_post

## Launch Details
- Launch 14 in `sr_M` frame 0: `_ZN2sr2io9k_sr_postILb0ELi12ELi16ELb0ELb0EEEvNS0_8PostArgsE`
- Grid: [48, 14, 1], Block: [256, 1, 1]
- Inputs: `g_col`, `expo`, `gates`, `factors`, `aniso`, `ihist0`, `prev_mvec1`, `mv_ero`, `mv_nondil`
- Outputs: `dhist` (u32, 82944), `full15` (u32, 331776), `ihist1` (f16vec4, 331776)

## Attempts and Hypotheses

### Attempt 1
- **Hypothesis**: Nearest-neighbor proxy with exact tonemap packing `pack_r11g11b10` for `full15`.
- **Result**:
  - `full15 [u32]`: 331776 elements, bit-exact 16317 (4.9181%), max |err| 9.73943e+08, max ulp 973943226.
  - Pixel 0 is bit-exact (`0x5c2784f` matches oracle).
  - First mismatch at index 1: `{'index': 1, 'port': 96630863.0, 'oracle': 96632912.0, 'port_bits': 96630863, 'oracle_bits': 96632912}`.
  - Full temporal filter blends warped `ihist0` using anisotropic weights `aniso`, `factors`, and `gates`.
