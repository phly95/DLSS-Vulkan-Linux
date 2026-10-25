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
