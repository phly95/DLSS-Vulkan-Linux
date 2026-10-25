# Notes: k_sr_enc0

## Launch Details
- Launch 2 in `sr_M` frame 0: `_ZN2sr2io9k_sr_enc0ENS0_8Enc0ArgsE`
- Grid: [32, 20, 1], Block: [32, 1, 1]
- Inputs: `g_col`, `g_dep`, `g_mv`, `prev_mvec0`, `dhist`, `hidden0`, `wbuf`, `expo`
- Outputs: `prev_mvec1` (f16, 73728), `mv_ero` (f16, 73728), `mv_nondil` (f16, 73728), `t0` (u8, 655360)

## Attempts and Hypotheses

### Attempt 1
- **Hypothesis**: Motion vectors are passed directly without scaling.
- **Result**: 0 / 73,728 bit-exact (first mismatch at index 0: port -0.0953369, oracle -0.2861328).

### Attempt 2
- **Hypothesis**: Motion vectors are scaled by 3.0 (`g_mv * 3.0`), mapped in 8x8 blocks per 32-thread workgroup (2 pixels per thread).
- **Result**:
  - `prev_mvec1 [f16]`: 73,728 / 73,728 (100.0000% bit-exact), max |err| 0, max ulp 0.
  - `mv_ero [f16]`: 73,728 / 73,728 (100.0000% bit-exact), max |err| 0, max ulp 0.
  - `mv_nondil [f16]`: 73,728 / 73,728 (100.0000% bit-exact), max |err| 0, max ulp 0.
  - `t0 [u8]`: 655,360 elements, bit-exact 1509 (0.2303%), max |err| 192.
  - 3 of 4 output buffers are 100% bit-identical.

### Attempt 3
- **Hypothesis**: `t0` tensor layout is `[C/32][H][W][32]` (two 32-channel groups of 32 contiguous bytes per pixel: group 0 at offset 0, group 1 at offset `W * H * 32`) with 64 FP16 biases loaded from `enc0_w` offset `1590016u`.
- **Result**:
  - `t0 [u8]`: bit-exact count increased 4x to 6,017 elements. Index 0 is 1 ULP off (port 48.0, oracle 47.0).
  - All 3 companion buffers remain 100.0000% bit-exact: `prev_mvec1`, `mv_ero`, `mv_nondil`.
- **Pipestats**: 108 SGPRs, 24 VGPRs, 0 spilled SGPRs, 0 spilled VGPRs, 0 scratch size.
