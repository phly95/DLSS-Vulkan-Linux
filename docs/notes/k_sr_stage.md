# k_sr_stage Notes & Hypotheses

## Architecture & Implementation
`k_sr_stage` implements the outer Swin Transformer layers for SR presets M and L:
- Stage 0: `k_sr_stage<64, 2, 0, 96, 2, false>` (enc1: C=64, g=2, Cn=96, mode=0)
- Stage 1: `k_sr_stage<96, 4, 0, 128, 2, false>` (enc2: C=96, g=4, Cn=128, mode=0)
- Stage 8: `k_sr_stage<96, 4, 2, 128, 2, false>` (dec2: C=96, g=4, Cn=128, mode=2)
- Stage 9: `k_sr_stage<64, 2, 2, 96, 2, false>` (dec1: C=64, g=2, Cn=96, mode=2)

Workgroup structure:
- 64 threads covering an 8x8 pixel tile divided into four 4x4 windows (16 tokens per window).
- Shared memory: `s_kv[4][16][32]` (4096 bytes LDS) to buffer K and V tensors during window attention.
- Pre-RMSNorm 1 with scale `s1b0` (includes $\sqrt{C}$ factor baked in from `sr_pack.py`).
- 16-token window multi-head attention with relative position bias table (`v258`) and bit-built exponential:
  $y = \text{clamp}(h \cdot 0.0171814, -0.55615234, 0.55615234)$,
  $p = y \cdot (0.92724609 - y^2) + 1.375$,
  mantissa shift $(p_{\text{bits}} \ll 5) + 0x8000$, producing bit-exact positive float16 exponential.
- Attention context accumulation $Context = \sum P_q \cdot V$ using 8 f16 WMMAs.
- Post-attention residual add with $v270$ bias.
- Pre-RMSNorm 2 with scale `s198`.
- Multi-block MLP: $C/8$ blocks with FC1 ($32 \times C$), SiLU activation, and FC2 ($C \times 32$), followed by $v240$ bias add and residual 2.
- Mode 0 downsampling: 2x2 patch concatenation ($4C$ channels) projected by $W_{\text{proj}}$ ($Cn \times 4C$) to `out_act2`.

## Compilation & Resource Stats (ACO on RX 9070 XT GFX1201)
- SGPRs: 108
- VGPRs: 256
- Spilled SGPRs: 0
- Spilled VGPRs: 0
- Scratch size: 0
- LDS size: 40960 bytes

## Attempt Log

### Attempt 1: Full Collaborative Swin Transformer with LDS Staging
- **Hypothesis**: Replacing the 26-line pass-through placeholder with full 16-token window attention, bit-built exp2, LDS-buffered K/V staging, MLP blocks, and 2x2 downsample projection will bring the kernel into structural alignment with the NeuralHub code object and dramatically improve bit-exact count and error bounds.
- **Result**:
  - `t1 [fp8]`: 655,360 elements, bit-exact 7,515 (1.1467%), max |err| 17.1875, max ulp 152, first mismatch:
    `{'index': 0, 'port': 7.5, 'oracle': -0.625, 'port_bits': 79, 'oracle_bits': 178}`
    (Up from 116 bit-exact, 65x improvement over stub).
  - `t2 [fp8]`: 245,760 elements, bit-exact 2,630 (1.0701%), max |err| 8.5, max ulp 142, first mismatch:
    `{'index': 0, 'port': -0.75, 'oracle': -0.6875, 'port_bits': 180, 'oracle_bits': 179}`
    (Up from 53 bit-exact, 50x improvement over stub; max error reduced from `inf` to 8.5; **Pixel 0 is only 1 ULP off**).
- **Status**: APPROX (Pass 1 complete in breadth).

### Attempt 2: ISA Bit-Built FP16 Exponential
- **Hypothesis**: Replacing the placeholder linear exponential with the polynomial and bit construction visible in the SR stage ISA will make softmax probabilities align with the oracle.
- **Result** (frame 1, launch 3, `k_sr_stage<64,2,0,96,2,false>`):
  - `t1 [fp8]`: 655,360 elements, bit-exact 6,083 (0.9282%), max |err| 18.5, max ULP 157, first mismatch `{'index': 0, 'port': -3.25, 'oracle': 0.5625, 'port_bits': 197, 'oracle_bits': 49}`.
  - `t2 [fp8]`: 245,760 elements, bit-exact 3,322 (1.3517%), max |err| 9.75, max ULP 146, first mismatch `{'index': 0, 'port': -0.9375, 'oracle': -1.75, 'port_bits': 183, 'oracle_bits': 190}`.
  - `DIFFDUMP: DIFFERS`. t1 regressed versus attempt 1; t2 improved slightly.

### Attempt 3: Balanced FP16 Softmax Sum
- **Hypothesis**: The ISA uses packed fp16 additions and pair reductions for the softmax denominator; replacing the shader's f32 accumulation with an explicit balanced fp16 tree will reproduce its rounding points.
- **Result** (same launch):
  - `t1 [fp8]`: 655,360 elements, bit-exact 6,086 (0.9286%), max |err| 18.5, max ULP 157, first mismatch `{'index': 0, 'port': -3.25, 'oracle': 0.5625, 'port_bits': 197, 'oracle_bits': 49}`.
  - `t2 [fp8]`: 245,760 elements, bit-exact 3,327 (1.3538%), max |err| 9.75, max ULP 146, first mismatch `{'index': 0, 'port': -0.9375, 'oracle': -1.75, 'port_bits': 183, 'oracle_bits': 190}`.
  - `DIFFDUMP: DIFFERS`. The denominator hypothesis made only a tiny change and leaves large errors.
- **Status**: APPROX; continue broad work, revisit with fresh hypotheses after a pass over other families.

### Attempt 4 (2026-10-26): model-driven hypothesis sweep - the kernel is structurally wrong, not rounding-noisy
- **Method**: a NumPy replica of `sr_stage.comp` (C=64,g=2,mode=0 instantiation, oracle t0 input,
  packed `stage0.bin` weights) that reproduces the *port GPU shader* to 89.71% bit-exact (corr
  1.0; residue = fma/emulation detail), then structural variants scored against the oracle dump of
  frame 0 launch 3 (`build/work/stage_model.py`, git-ignored; the replica is the template).
- **Measured state** (kernel_run + diffdump on oracle inputs, frame 0 launch 3):
  t1 1.1595% bit-exact, t2 1.0478%; value correlation with the oracle only **0.4551** and port
  std 2.5x the oracle's - every output channel is wrong (per-channel corr scatters -0.5..0.9,
  means and scales off everywhere). This is a wrong function, not accumulation-order noise.
- **Refuted hypotheses** (corr vs oracle in parentheses; port baseline 0.4551): identity input
  permutation (0.2049), invP input permutation (0.1693), Q/K transposed weight read (0.4615),
  V transposed (0.4076), attention head swap (0.5054), no relative-position bias (0.4665),
  bias table transposed (0.4550) or rolled (0.4581), QKV-then-bias block order (0.0816 - worse),
  logit scale 0.5 (0.4569) / 0.25 (0.4608), softmax probs kept f32 (0.4554), Q/K rounded to fp8
  before the dot (0.4551), V rounded to fp8 (0.4551), exp approximimation constants (no change).
- **Layout facts recovered from the ISA load offsets** (`tools/re/fn.sh sr <stage<64,2,0,96>>`):
  weight regions at 0..128 (gains), 128..4224 (head0 Q+K), 4224..6784 (head0 bias+V),
  6784..13440 (head1), 13440..13568 (v270), 13568..13696 (s198), 13696..17856 (MLP block0:
  fc1 2048 + b1 64 + fc2 2048) - identical to the port's `cur_w` walk, so the block ORDER is right.
  The matmul is `v_wmma_f32_16x16x16_fp8_fp8` fed by `v_cvt_pk_fp8_f32` + `v_perm_b32 ... 0x3020104`
  operand assembly; logits are `v_fma_mix{lo,hi}_f16(dot, 1.0, bias)` - s11 is `s_mov 1.0`, so
  there is **no** 1/sqrt(d) logit scale in the oracle either.
- **Status**: APPROX; the remaining suspect is the WMMA operand/weight swizzle (`frag_off` is a
  valid bijection with the right 8-byte-lane/256-stride strip structure but has never been
  validated against a working matmul). Fastest next instrument: run the oracle kernel itself on
  *synthetic* packed weights via `tools/sr/srref` + `--dump` (srref takes any packed dir; the
  oracle may be used for measurement, never committed), which identifies each sub-block's exact
  math without reading 3.3k instructions of schedule.

### Attempt 5 (2026-10-26): synthetic-weight oracle probes - the interactive oracle works
srref runs NeuralHub's own kernel on **any** packed weight dir, so `tools/sr/srref` under podman
(`docker.io/rocm/dev-ubuntu-24.04:7.2.4`, see `tools/gpu/collect_oracle.sh`) plus a patched
`stage0.bin` turns the reference into an interactive function.  Smoke test: real weights,
1-frame frames dir -> dump `003.t1.bin` is **byte-identical** to the original oracle dump, so the
probe reproduces the original conditions exactly.  All probes below: frame 0 launch 3, output t1;
"identity" = t1 == t0 to 99.95% (the 0.05% are -0.0 -> +0.0 IEEE fixes, benign).

Probe results (stage0.bin regions per the port's `cur_w` walk; ones = fp8 0x38 = 1.0):

| probe | stage0.bin fill | oracle t1 |
|---|---|---|
| P0 | all zeros | **identity** - the output equation out = input + attention + mlp + bias is CONFIRMED |
| P2 | gains=f16 1.0, QK0+V0+bias0, head1 same, s198=1.0, MLP=0 | wild: mean 5.9, std 12.7, corr(t1,t0) 0.06 |
| w3 | QK region 128..4224 = 1.0 (gains zero) | identity (gains zero kills s_norm - expected) |
| w4 | bias+V region 4224..6784 = 1.0 (gains zero) | identity (same) |
| w5 | MLP block0 13696..17856 = 1.0 (gains zero) | mean +12.0, corr 0.89 - MLP region semantics MATCH the port's walk (b1 read as f16 pairs = 0.777 explains the offset) |
| r1/r2/r5 | head1 QK, head1 bias+V, tail 47104..71872 = 1.0 | identity |
| r3 | v270+s198 13440..13696 = 1.0 | constant +0.52 shift - **v270 region CONFIRMED** |
| sA | gains 1.0 + QK0 = 1.0 | identity |
| s4 | gains 1.0 + bias0+V0 = 1.0 | identity |
| sAB | gains 1.0 + QK0 = 1.0 + bias0+V0 = 1.0 | wild (mean 2.9, std 6.4) |

Conclusions:
1. The attention output is exactly zero unless **both** region A (128..4224) and region B
   (4224..6784) carry weights - A alone (Q/K with V=0) and B alone (V with Q/K=0) both give
   exactly identity, so B is not simply "V" and the port's "bias between K and V" reading is
   not what gates the output.
2. Under sAB the context (t1 - t0) is **per-window constant** (one 4x4 window = one value,
   first singular value 232 vs 2.5 for the rest) and ~16x larger than the port's
   softmax-normalized prediction (window means 0..15, positive-biased like sum(s_norm)).
   The port divides by sum(exp); the oracle's behaviour under constant weights looks like
   sum or window-size normalization.  Model variants no_div / div16 on the REAL weights did
   NOT improve (corr -0.04 / 0.11 vs 0.4551) - the normalization question must be settled in
   probe space (model on the sAB weights), not on real weights.
3. Next: run `build/work/stage_model.py` with STAGE_WDIR=build/work/probe/sAB (and w5, P2)
   to fit the exact attention arithmetic in probe space, then re-test on real weights.

Probe scaffolding (git-ignored): `build/work/srref` (hipcc build of tools/sr/srref/srref.cpp),
`build/work/probe/frames1` (1-frame dir), `build/work/probe/w*|r*|s*` (crafted weight dirs),
`build/work/stage_model.py` (NumPy replica; STAGE_WDIR/STAGE_ORC env overrides).
