# Notes: k_sr_tube_split

## Launch Details
- Launch 5 in `sr_M` frame 0: `_ZN2sr15k_sr_tube_splitILi128ELi4ELb0EEEvNS_9StageArgsE`
- Grid: [9, 6, 1], Block: [64, 1, 1]
- Inputs:
  - `wbuf+235520`: Stage weights for `tube0` (RMSNorm1, Q/K/V per 4 heads, relative position bias tiles, attention out bias, RMSNorm2, 16x FC1/FC2 MLP blocks)
  - `t4`: from launch 4 (128 channels per pixel at 32x20, 81,920 bytes)
- Outputs:
  - `t5`: 81,920 bytes (128 channels per pixel at 32x20)

## Attempts and Hypotheses

### Attempt 1
- **Hypothesis**: Decoded full Swin Transformer Layer architecture from GFX1201 ISA and weight packer (`sr_pack.py`):
  - 4x4 pixel window (16 tokens), 64-thread workgroup mapping each thread to one token and one head.
  - Symmetrical reflect-101 border padding with shifted window coordinates (`x0 = gx*4 - shiftX`, `y0 = gy*4 - shiftY`).
  - RMSNorm 1 over 128 channels using baked `sqrt(C)` gain `s1b0` and epsilon $2^{-14}$ (`6.1035e-5`).
  - 4-head attention: Q (32x128), K (32x128), V (128x32) with 16x16 scaled relative position bias `v258` and non-overflowing `nr_swin_exp2` polynomial exponent.
  - Softmax quantization to FP8 E4M3 before context projection with V.
  - RMSNorm 2 using `s198`.
  - 16-block MLP expansion (128 -> 512) with Cubic SiLU activation and contraction (512 -> 128).
  - Residual connection and clamping to $[-448.0, 448.0]$.
  - Unpermuted channel mapping `P(p)` with store of in-bounds tokens.
- **Result**:
  - `t5 [u8]`: 81920 elements, bit-exact 2473 (3.0188%), max |err| 205, max ulp 205, first mismatch `{'index': 0, 'port': 204.0, 'oracle': 202.0, 'port_bits': 204, 'oracle_bits': 202}`.
  - Mean absolute error across all 81,920 elements is 2.87 on [0, 448] scale (93.5% within 5.0).

### Attempt 2
- **Hypothesis**: Eliminate private per-thread register arrays to prevent ACO compiler register spilling on GFX1201:
  - Move intermediate activation buffers into shared memory (LDS): `s_norm[16][128]`, `s_mlp[16][128]`, `s_Q[4][16][32]`, `s_K[4][16][32]`, `s_V[4][16][32]`, `s_h1[16][32]`.
  - LDS memory total: 25,600 bytes (within 32 KB limit).
  - Keeps all thread registers purely scalar.
- **Result**:
  - `pipestats build/shaders/sr_tube_split.spv`: sgprs 108, vgprs 168, spilled_sgprs 0, spilled_vgprs 0, scratch_size 0, lds_size 25600.
  - Native RX 9070 XT `kernel_run` against Launch 5:
    `t5 [fp8]: 81920 elements, bit-exact 2474 (3.0200%), max |err| 12, max ulp 148, first mismatch {'index': 0, 'port': -6.0, 'oracle': -5.0, 'port_bits': 204, 'oracle_bits': 202}`.
  - Full `tools/verify/offline_check.sh`: PASS all 27 pipelines (O1..O7 clean).
