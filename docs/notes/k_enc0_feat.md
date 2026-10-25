# Notes: k_enc0_feat

## Launch Details
- Launch 3 in `rr_F` frame 0: `_ZN4rrio11k_enc0_featENS_6E0ArgsE`
- Grid: [16, 10, 1], Block: [16, 16, 1]
- Inputs: `E0c6e` (col), `E0c76` (mv), `E0c72` (dep), `E0c7a` (nrm), `E0c7e` (alb), `E0c82` (spa), `E0c8e` (rgh), `E0c22` (expo)
- Outputs: `E0be8` (u8, 2621440), `E0bec` (f16, 163840), `E0bf6` (f16, 147456), `E0bfa` (f16, 73728), `E0bfe` (f16, 73728)

## Attempts and Hypotheses

### Attempt 1
- **Hypothesis**: Motion vectors scaled by 3.0: `bfa` = `g_mv * 3.0`, `bfe` = `g_mv * 3.0`, `bf6` = `[bfa.x, bfa.y, bfa.x, bfa.y]`.
- **Result**:
  - `E0bfa [f16]`: 73,728 / 73,728 (100.0000% bit-exact), max |err| 0, max ulp 0.
  - `E0bfe [f16]`: 73,728 / 73,728 (100.0000% bit-exact), max |err| 0, max ulp 0.
  - `E0bf6 [f16]`: 147,456 / 147,456 (100.0000% bit-exact), max |err| 0, max ulp 0.

### Attempt 2
- **Hypothesis**: `E0bec` applies exposure scaling `col.rgb * expo` with symmetric reflection border mode for rows past active height (`y >= 144` uses `2 * 143 - y`).
- **Result**:
  - `E0bec [f16]`: 163,840 / 163,840 (100.0000% bit-exact), max |err| 0, max ulp 0.
  - 4 of 5 output buffers are 100% bit-identical.
  - `E0be8 [u8]`: 2,621,440 elements, bit-exact 13,215 (0.5041%), max |err| 172. First mismatch index 0 (port 0, oracle 33).

### Attempt 3
- **Hypothesis**: `E0be8` tensor memory layout is `[C/32][H][W][32]` (two 32-channel groups of 32 contiguous bytes per pixel: group 0 at offset 0, group 1 at offset `W_0 * H_0 * 32`), NOT interleaved 64 channels. Decoded channels from GFX1201 ISA: `c00` (tm_r), `c01` (tm_g), `c04` (tm_b) tonemapped RGB; `c02` (alb.r), `c03` (alb.b); `c05, c08, c12, c13, c17, c20, c24, c25` (mv.x * 3.0); `c09, c16, c21, c28` (mv.y * 3.0); constants `c15 = 1.0` (0x38), `c18 = 0.5` (0x30), and 24 constant-zero channels in group 1.
- **Result**:
  - `E0be8 [u8]`: 1,911,195 / 2,621,440 (72.9063% bit-exact) (up from 0.50% / 13,215 elements!), 78.84% within 1 ULP. First mismatch at index 4 is 1 LSB off (port 28, oracle 27).
  - All other 4 output buffers remain 100.0000% bit-exact:
    - `E0bec [f16]`: 163,840 / 163,840 (100.0000% bit-exact)
    - `E0bf6 [f16]`: 147,456 / 147,456 (100.0000% bit-exact)
    - `E0bfa [f16]`: 73,728 / 73,728 (100.0000% bit-exact)
    - `E0bfe [f16]`: 73,728 / 73,728 (100.0000% bit-exact)
- **Pipestats**: 108 SGPRs, 24 VGPRs, 0 spilled SGPRs, 0 spilled VGPRs, 0 scratch size.
