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

### Attempt 5 (2026-10-27): softmax semantics settled in probe space; the Q5 x32 scale
Probe campaigns (all via `srref --dump` on crafted weight dirs + the NumPy replica):
- **Reciprocal/normalization function**: with constant-color frames and all-ones weights the
  logits are exactly equal within every window, and the oracle's attention output is NOT the
  softmax mean - it is large (means +20 .. -13.8 across colors 0.01..2.0) and even changes
  sign; g = ctx/S sampled at Sigma in {2.0, 2.69, 3.13, 3.25, 5.63} = {195, 85, 65, 61, 68}.
  Patching wbuf[0x20000..0x40000] with constant f16 changed nothing, so the ISA's
  `flat_load_u16` table index (wbuf-base + Sigma-bits<<1 + 0x20000) is anchored elsewhere
  (the prolog has a PC-relative `s_getpc + 0xe125c` base): the denominator's reciprocal is an
  **embedded f16 table** whose low-Sigma entries are non-recip (wild, sign-flipping).  The
  real-weights regime (large Sigma) is expected to be a clean 32/Sigma but is unverified.
- **Dead attention at exactly-equal logits**: q1 (Q=1, K=0) and s4/vs0 give EXACTLY zero
  context; the replica reproduces this at **100.0000% bit-exact** with
  `u = exp - max(exp); prob = fp8(32 * u / sum(u))` (any max-subtraction family), and plain
  softmax does NOT (it predicts uniform 1/16).  But every max-subtraction variant then
  *fails* the active-regime probes (sAB corr -0.5..0.1) where plain softmax x32 fits at
  corr 0.9983 (token-mean fit) / per-channel corr 0.998 (full replica).  The two regimes are
  not yet reconciled by one formula - the equal-logit regime is table-driven (above).
- **The x32 scale (Q5 probabilities)**: with active probs the oracle's context is
  **32 x sum_j softmax_j . V_j** at every regime, including one-hot (hot20: ctx/S0 = 32.15
  median; V-linearity probe h1 with 0.5-weights gives exactly half: 16.11).  The replica
  variant `prob = fp8(32 * exp / sum(exp))` reproduces sAB at 20.7% bit-exact (from 0.43%)
  and hot20 at 12.3% (from 0.24%); the residual is exactly the Q5 fp8 rounding (~7% at
  values 2-4, E4M3 ulp 0.25) plus a spatially-coherent 6.5% wobble.  Region bisect: the
  effective V matrix is fp8-linear and lives in the FIRST 1024 bytes of the port's bias+V
  region (4736..5760); the second 1024 is inert unless head1 is active.
- **Shader edit applied then reverted**: `inv_sum = 32.0 / ...` in sr_stage.comp +
  sr_tube_split.comp is live (kernel_run launch 3: 1.1595% -> 0.4305% bit-exact on real
  inputs - WORSE, because the dead-zone/denominator semantics are unsettled) and pipeline-
  inert (see `docs/notes/sr_chain_decoupling.md`: the chain's t1 is a dead end).  The fix
  lands together with the attention completion.
- **Status**: APPROX.  Next in this kernel: settle the equal-logit/table regime (find the
  embedded table in the .co at the PC-relative base), then land max-subtraction + Q5 + the
  mode-2 skip wiring as one change.

### Attempt 6 (2026-10-26): the probe regime does not span the real one - reconcile before landing

Instrument: `build/work/stage_model.py` (NumPy replica), scored by `corr vs oracle t0 -> t1`
against `sr_M/dump/frame0/003.t0.bin`/`003.t1.bin`, with `STAGE_WDIR`/`STAGE_ORC` pointed at a
crafted-weight probe to score the *same* variants in probe space.  **Cross-validation first:**
replica `port` = **0.4551**, and `tools/report/launch_report.py` on the real GPU launch 3 with
oracle inputs = **r 0.455124** (Step C baseline) - the offline replica and the GPU measurement
agree to four decimals, so this instrument may be trusted.

Variant sweep, `corr vs oracle`, one row per hypothesis:

| variant | real weights -> oracle t1 | probe `sAB` -> oracle t1 |
|---|---:|---:|
| `port` (plain softmax, as shipped) | **0.4551** | 0.2627 |
| `ctx32` (plain softmax x32) | 0.1003 | **0.6758** |
| `q5c` (x32 + fp8 probs) | 0.1002 | **0.6758** |
| `maxsub` (exp - max, then /sum) | -0.0537 | -0.5521 |
| `q5b` / `q5g` (maxsub + `max(u,0)`) | **0.5739** | 0.1071 |
| `q5` / `q5d` | -0.0537 | -0.0537 |
| `q5e` | 0.0516 | - |
| `no_div` | -0.0446 | - |
| `div16` | 0.1070 | - |
| `ctx_replace` | 0.2461 | - |
| `no_prob8` | 0.4554 | - |
| `qk_fp8` / `v_fp8` | 0.4551 / 0.4551 | - |

**The two spaces disagree completely.**  Real weights prefer `q5b` (0.4551 -> 0.5739) and hate
`ctx32`; probe space prefers `ctx32`/`q5c` (0.2627 -> 0.6758) and hates `q5b` (0.1071) and
`maxsub` (-0.5521).  No single tested variant wins both, which is the note's earlier
"two regimes are not yet reconciled" showing up as a measurement rather than a hunch.

**Why, measured** (`STAGE_STATS=1`, logit/pe statistics per head per token-batch):

| | logits min..max | mean | count `logit > 11` | pe max | non-finite |
|---|---|---:|---:|---:|---:|
| real weights | -22.1 .. **+33.4** | -9.9 | **64 / 1024** | **256** | 0 |
| probe `sAB` | -3.5 .. **+13.4** | 0.6 .. 7.9 | 0-7 / 1024 | **11.5** | 0 |

The real regime runs logits up to **33**, where `exp` saturates (pe pinned at 256); the sAB
probe never leaves the low-logit region (pe <= 11.5) and 6 of its 8 head-batches sit at
`logit == 0.0` exactly.  **A formula fitted on sAB is fitted on a regime the real kernel
never occupies, and vice versa** - which is why `ctx32` and `q5b` swap places between the
two columns.  This also explains the note's earlier contradiction (maxsub reproduces the
equal-logit dead-attention probes bit-exactly, yet fails every active-regime probe).

**Do not land any of these into `sr_stage.comp` yet** - the `inv_sum = 32.0` edit already
measured worse on real inputs when landed piecemeal (1.1595% -> 0.4305% bit-exact).

**Next step:** design probes that cover the *real* regime - push logits past 11 (large Q/K
weights or a high-contrast frame) so `exp` saturates, and sweep Sigma there, instead of
extending the low-logit sAB sweep.  Then one formula must fit both the saturation regime and
the equal-logit regime before anything lands.  Judge every result with
`tools/report/launch_report.py` (Step C): baselines st0 **4.9783 dB / r 0.455124**,
st8 7.8425 / 0.093988, st9 4.6123 / 0.058489.

Instrument change (uncommitted, `build/work/stage_model.py`, under `build/` like every other
probe): an env-guarded `STAGE_STATS=1` block right after `pe = nr_swin_exp(logits)` that prints
min/max/mean, the count above 11 and above 88, and pe's non-finite count and maximum.

### Attempt 7 (2026-10-26): probes pushed into saturation; normalization settled, content is the blocker

Three new instruments (all git-ignored, under `build/work/`):

- `probe_craft.py` - builds a `stage0.bin` from the region map (gain1 0..128, QK 128..4224 /
  6784..10880, bias 4224..4736 / 10880..11392, V 4736..6784 / 11392..13440, v270 13440..13568,
  gain2 13568..13696, MLP 13696..71872) into a copy of `build/re/packed/M`.  With `QK = 0`
  the logit is **exactly** `bias[tok,j]`, so the logit regime is placed on demand; with a
  uniform QK fill `f` the logits are exactly `32 f^2 * S_t * S_j`, i.e. `probe_fit.py`'s own
  model form with `alpha = 32 f^2`.
- `probe_ctx.py` - per-probe ctx = t1 - t0 with channel-uniformity and ctx/sumS diagnostics.
- `probe_one.py` - fits a least-squares **beta ONCE on one probe** and then holds it fixed
  while checking the others, which is what "one formula fits both regimes" has to mean.

Probe set and their logit regimes (measured by `STAGE_STATS=1` through `stage_model.py`):

| probe | construction | logits | >11 | pe max | oracle ctx |
|---|---|---|---|---|---|
| `sat_eq20` | QK 0, bias 20 | exactly 20 | 100% | 61 | **0.0000** |
| `sat_eq33` | QK 0, bias 33 | exactly 33 | 100% | 256 | **0.0000** |
| `sat_act` | QK 0, bias 8..33 | 8..33 varied | 88% | 256 | **0.0000** |
| `qk2` | QK 2.0 | -16.0..51.4 | 16.5% | 256 | mean **+15.05** |
| `qk1` | QK 1.0 | -4.0..12.9 | 0.01% | 9.75 | (not dumped) |
| `qk4.5` | QK 4.5 | -81..261 | 36.6% | 256 | (not dumped) |
| `sAB` (existing) | QK 1, bias+V 1 | -4.0..12.9 | ~0% | 9.75 | mean +3.02 |

(The real weights run -22.1..+33.4 with 64/1024 above 11 and pe at 256, so `qk2` and the
`sat_*` probes are the ones that occupy the regime the real kernel occupies.)

**Result 1 - the equal-logit dead-attention behaviour holds at every saturation level.**
`sat_eq20` and `sat_eq33` are exactly-equal logits *inside* the saturating regime and the
oracle's context is exactly `0.0000` (mean = min = max), while `qk2`/`sAB` on the same
harness give +15.05 / +3.02 - so the harness detects attention and the zero is real.
Consequence: **all six families `probe_fit.py` offered are refuted** - `softmax`, `div16`,
`none`, `onehot`, `fp8softmax`, `fp8div16` each predict a non-zero constant at equal logits,
and a beta fitted non-zero on the active probes cannot make them vanish.

**Result 2 - one formula does fit both regimes:**
`w = (pe - min(pe)) / sum(pe - min(pe))`, guarded so `0/0 -> 0`.  It is exactly 0 when every
logit is equal, and on the two *active* probes it reaches corr **0.9846** (saturating `qk2`)
and **0.9663** (low `sAB`) - the best of any family that passes the equal-logit requirement:

| family | corr qk2 | corr sAB | equal-logit |
|---|---:|---:|---|
| `softmax` / `fp8softmax` | **0.9965** | **0.9983** | FAIL (predicts 64.5, oracle 0) |
| `submax_logit` | 0.9945 | 0.9983 | FAIL |
| onehot | 0.9451 | 0.9001 | FAIL |
| **`submin_norm`** | **0.9846** | **0.9663** | **OK (0.000000)** |
| `submax_norm` | 0.7541 | 0.8203 | OK |
| `center` (= softmax - 1/16) | 0.7374 | 0.6296 | OK |
| `submax_raw` / `submin` / `submean` | 0.55..0.60 | 0.60..0.68 | OK |
| `submax0` (identically 0) | n/a | n/a | OK but degenerate |

fp8 quantisation of `w` and the x32 scale do not move it (`submin_fp8` 0.9844, `submin_fp8x32`
0.9844; beta absorbs scale), so the shape decision is settled.

**Result 3 - on real weights the normalization is NOT the error, the attention content is.**
Two independent sweeps, both through the replica whose `port` variant reproduces the measured
`r` (0.4551 vs `launch_report`'s 0.455124):

1. *ctx magnitude* (plain softmax, `STAGE_CTX_SCALE`): 0 -> **0.5739**, 0.125 -> 0.5690,
   0.25 -> 0.5604, 0.5 -> 0.5317, 1 -> 0.4551 (as shipped), 2 -> 0.3544, 4 -> 0.2826,
   32 -> 0.1003 - **strictly monotone, best at suppression**.  Negative scales beat their
   positive twins at every magnitude (-0.125 0.5782 > 0.5690, -0.25 **0.5793** > 0.5604,
   -0.5 0.5684 > 0.5317, -1 0.5299 > 0.4551, -2 0.4543 > 0.3544, -4 0.3656 > 0.2826) so
   there is a small negative component - but even the best signed scale (0.5793) barely beats
   doing nothing at all (0.5739): the port's attention explains **~0.6%** of the variance
   (0.5793^2 - 0.5739^2).
2. *normalization* on real weights: `subminnorm`'s scale sweep is **identical** to plain
   softmax's (0 -> 0.5739, 0.25 -> 0.5603 vs 0.5604, 1 -> 0.4551, 4 -> 0.2825 vs 0.2826),
   and `subminnorm` scores corr 0.4551 = baseline with oracle bit-exact 1.1401% vs 1.1606%.
   So the min-subtraction that the probes demand is **fp8-invisible on real weights** - it
   fixes an edge case, not the measured error.
3. *structure* on real weights: `head_swap` 0.5054 is the best variant tried, `no_bias`
   0.4665, `qk_transposed` 0.4615, `scale_quarter` 0.4608, `v_transposed` 0.4076 - **all
   below the no-attention 0.5739**.  `no_perm_load` 0.2049 and `invP_load` 0.1693 collapse,
   which confirms the current load permutation is right; `no_perm_store`/`invP_store` =
   0.4551 exactly, i.e. the store permutation is inert in this replica.

Two consequences worth stating plainly:

- `q5b`'s earlier 0.5739 is now explained: `max(u,0)` makes `prob` identically 0, so `q5b`
  was simply "attention removed" - confirmed independently by `STAGE_CTX_SCALE=0` giving
  the same 0.5739.  It was never a decode.
- **Nothing landed.**  `submin_norm` fits both probe regimes but is a no-op on real weights,
  and the attention content is where the error is; landing either alone would be exactly the
  partial-hypothesis edit the `inv_sum = 32.0` attempt already warned about.

**Next step:** decode the attention *content* rather than its normalization - with the
normalization shape now known (`submin_norm`), the remaining question is what the oracle's
Q/K/V actually are, since every structural variant tested (`qk_transposed`, `v_transposed`,
`head_swap`, `biasT*`, scale) stays below the attention-free ceiling.  The most specific
open lead from attempt 5 is that "the effective V matrix lives in the FIRST 1024 bytes of
the port's bias+V region (4736..5760) and the second 1024 is inert unless head1 is active":
bisect the bias and V regions with `probe_craft.py` (it can write arbitrary f16/f8 into any
region) against `probe_ctx.py`, which now has a known-good normalizer to compare against,
rather than reading the layout off the ISA.

### Attempt 8 (2026-10-26): the oracle's REAL attention isolated; weights right, V read wrong

Three findings force a re-reading of attempt 7's "content is the blocker", plus a set of
instrument bugs that had made two earlier conclusions invalid.

#### Instrument bugs found and fixed (all in git-ignored `build/work/`)

1. **`no_perm_store` / `invP_store` were literal no-ops.** `PS` was computed at line 163 but
   the store used a hardcoded `P[i]`, so attempt 7's "store permutation is inert in this
   replica" was never tested.  With `PS` actually used: `no_perm_store` **-0.1301**,
   `invP_store` **-0.1305** vs port **+0.2019** -> the port's store `P` is *correct*, now
   measured rather than assumed.
2. **`build_weights()` was called without its variant**, so the `bias_after_v` layout was
   never reachable - same class of dead variant as `PS`.  Fixed by passing `variant` through.
3. **`ulp()` returned the fp8 *value*, not the *step*.** The resulting "noise floor" (1.72)
   overstated the floor ~10x and would have wrongly declared the oracle's attention
   unmeasurable.  Corrected (`e>=1 -> 2^(e-10)`, `e==0 -> 2^-9`): floor is **0.1459**.
4. In `probe_w.py`, a token-coordinate typo (`tok % 4` for `tok // 4`), a missing load
   permutation (`s_act[tid, cg*32 + P]`), and a transposed least-squares system each had to
   be fixed before any solve meant anything.

#### New instruments (git-ignored)

- `probe_craft.py --isolate` - keeps **every** weight real (QK, bias, V, gains) and zeroes
  only `v270` + the MLP, so the oracle's dump *is its own real attention*.
- `probe_craft.py --zero-range LO:HI` - byte-liveness tests, composable with `--isolate`.
- `probe_attn.py` - scores **attention outputs**: `ctx = dequant(t1) - dequant(t0)` on each
  side, so the shared input `A` (which inflates any t1-vs-t1 correlation) is removed.
- `probe_w.py` - Exp A (free `w`, port's V) and Exp B (free `Wv`, port's weights), plus a
  swapped head->channel control; exports fitted `Wv`.
- `probe_multi.py` - fits ONE `w` across five probes that differ only in V bytes.
- `probe_vsweep.py` - ranks candidate V reads purely by correlation (scale-invariant).

#### Result 1 - the bias region does not drive the oracle's logits

| probe | construction | oracle ctx |
|---|---|---|
| `neg_var` | QK 0, bias **-33..-8** | **0.0000** |
| `neg_eq` | QK 0, bias **-20** | **0.0000** |
| `sat_act` | QK 0, bias 8..33 | **0.0000** |
| `qk2` | QK 2.0, bias 0 | **+15.05** |

Varying the bias - positive or negative, equal or spread - never activates attention, while
a non-zero QK fill always does.  So the oracle's logits are `Q.K` only; the port's
`logits[tid,j] = Q@K + bias[tok,j]` adds a term the oracle does not have.  Consistent with
attempt 7's `no_bias` = 0.4665 vs 0.4551 (small, so not the main error).

#### Result 2 - the oracle's real attention is now measured directly

With `--isolate`, both sides drop `v270` + MLP, so `ctx = t1 - t0` *is* the attention:

```
ctx_oracle: mean +0.1060  mean|ctx| 0.9300  max|ctx| 4.3125
input t0  : mean|a0| 0.4716  -> ctx is 197.2% of the input magnitude
noise floor (one fp8 step): mean 0.14592   -> mean|ctx|/noise = 6.37
```

The attention output is **twice the size of the input it is added to**, and it is 6.4x above
the fp8 differencing noise, so it is fully measurable.  On this metric:

| variant | corr(ctx) | corr(t1) | rRMSE(ctx) |
|---|---:|---:|---:|
| `port` (softmax) | **+0.2019** | +0.5447 | 1.2141 |
| `subminnorm` | +0.2019 | +0.5447 | 1.2142 |
| `no_prob8` | +0.2021 | +0.5450 | 1.2138 |
| attention removed (`q5b`) | n/a (ctx=0) | +0.5735 | **1.2055** |
| `maxsub` / `q5` / `q5d` | -0.1701 | -0.0594 | 464.3 |

`corr(t1)` alone is misleading: both sides share the input, so "no attention at all" scores
0.5735 there.  The honest number is **corr(ctx) = 0.2019**, and its rRMSE (1.2141) is barely
better than predicting zero attention (1.2055): the port's attention explains **~4%** of the
variance of the oracle's real attention.

#### Result 3 - all 4096 V bytes are live (attempt 5's "second 1024 inert" refuted)

Zeroing each 1024-byte half in turn, oracle vs `real_attn`:

| probe | bytes changed | \|Δctx\| | Δ/noise | corr(Δ, ctx) |
|---|---:|---:|---:|---:|
| `v0a` (4736..5760) | 317437 | 0.4906 | 3.36 | -0.4143 |
| `v0b` (5760..6784) | 322400 | 0.5033 | 3.45 | -0.3673 |
| `v1a` (11392..12416) | 315595 | 0.4246 | 2.91 | -0.3095 |
| `v1b` (12416..13440) | 312087 | 0.4466 | 3.06 | -0.2751 |

Every half changes ~half the output by ~3x the noise floor, so neither V region has an inert
second half.  (Verified the four probes differ from `real_attn` only in those exact ranges:
1014-1019 bytes each.)

#### Result 4 - the attention WEIGHTS are right; the V read is wrong

Because `w` depends only on logits (Q, K, bias - untouched by these probes), all five probes
share **one** `w` while exposing five different `V`.  So a single `w` must fit all of them:

| head | fit on 2 V sets | held-out (3rd probe) | fit all 3 | noise floor |
|---|---:|---:|---:|---:|
| 0 | 0.8344 | **0.9511** (corr +0.3247) | 0.8515 | 0.157 |
| 1 | 0.8008 | **1.1027** (corr -0.0163) | 0.8790 | 0.157 |

Residuals sit at 0.80-1.10 against a noise floor of 0.157, and recovered `w` values are
garbage (max 117598, 50% negative): **no attention weights can explain the oracle's output
through the port's V**.  That is a statement about V, not about `w`.

The weights are independently vindicated by Exp B controls (free `Wv`, port's `prob`):

| prob source | h0 explained | h1 explained |
|---|---:|---:|
| **port softmax** | **83.9%** | **50.7%** |
| `subminnorm` | 83.9% | 50.7% |
| `no_softmax` (unnormalised) | 79.9% | 22.6% |
| `div16` | 79.6% | 22.6% |
| uniform 1/16 | 61.4% | 47.8% |
| **shuffled port** | **49.8%** | 48.0% |
| random dirichlet | 59.8% | 46.2% |

The port's normalised softmax beats every control by ~34 points on head0, so the *weight
structure* is right while the *V read* is wrong - and the fitted `Wv` that works is **not a
rearrangement of the port's bytes**: overall corr(fitted, port) = 0.078 (h0) / 0.000 (h1),
best row permutation reaches only cosine 0.405 with 33/64 (h0) and 24/64 (h1) distinct
assignments (a true permutation would be 64/64), full rank 32/32.

#### Result 5 - neither the obvious alternatives nor the mapping explain it

- head->channel swap: h0 0.4011->0.3994, h1 0.7020->0.5850 - not the cause.
- V basis: `snorm` 83.9/50.7, `snorm_nogain` identical (gain1 is absorbed by the free `Wv`),
  `sact` 79.1/48.8 - not the cause.
- V layout sweep, predictive corr(ctx), port's read = 0.1884 (h0) / 0.2333 (h1):

| candidate | corr h0 | corr h1 |
|---|---:|---:|
| `frag(k,c,32)` **[port]** | **+0.1884** | +0.2333 |
| `frag, first 1024 only` | -0.0003 | **+0.3070** |
| `frag(P[k],c,32)` | +0.1142 | +0.2038 |
| `frag, e5m2 decode` | +0.1773 | +0.1554 |
| `rowmajor k*32+c` | +0.1186 | -0.1833 |
| `frag(k,P[c],32)` | +0.1113 | -0.1499 |
| V in bias region (`bias_after_v`) | +0.0866 | -0.0830 |
| `frag reversed` | +0.0432 | -0.0233 |
| `frag(k,c,16)` | +0.0090 | -0.0065 |
| `rowmajor c*64+k` | -0.1824 | -0.0544 |
| `frag(c,k,32)` | -0.3119 | -0.0948 |

No candidate decodes V: the port's own read remains the best for head0, and the only
"improvement" (head1 dropping to its first 1024 bytes) is an *ignore-the-broken-half* effect,
not a decode - since Result 3 proves those bytes are live.

**Nothing landed.**  All of these are measurements; no edit went into `sr_stage.comp`.

**Next step:** stop guessing swizzles and *measure* the byte->(k,c) mapping.  A byte `p` in
the V region is exactly one entry `Wv[k,c]`, so with `ctx[t,c] = Wv[k,c] * sum_j prob(t,j)
s_norm[j,k]`, a probe that activates a chosen input channel `k0` (via a non-uniform `gain1`)
and sets selected V bytes makes the *set of channels that respond* reveal `c(p)`, while the
response over queries reveals `k0`.  Bit-plane probes over `p` (8 probes) plus a handful of
`gain1` selections recover the mapping empirically instead of formula-guessing; the mapping
is a bijection (2048 bytes <-> 64x32), so a dozen well-chosen probes should pin the swizzle.
Head 1 (51% explained vs head0's 84%) is the more suspect of the two and the place to start.

### Attempt 9 (2026-10-26): V's byte layout DECODED - frag_off's arguments are swapped

Attempt 8 ended by proposing to stop guessing swizzles and measure the byte->(k,c) map.
That measurement now exists, it is validated, and it identifies the layout - with one
residual (a per-row sign) that still stops the fix landing on real weights.

#### Instruments

- `probe_craft.py --v-code low5|high5` writes V byte `i` (region-local) as fp8 code
  `0x28 + (i & 31)` / `0x28 + ((i >> 5) & 31)`, so the byte's own index bits are carried in
  its VALUE.  Codes 0x28..0x47 decode to 0.25..3.75 - large enough to survive a fit.
- `probe_craft.py --gain1-select K` zeros `gain1` except at channel `K`, so only ONE row of
  `s_norm` survives.  Attention then has **1 unknown per output channel** (64 equations,
  1 unknown) instead of 64, which removes the conditioning blow-up that defeated the first
  attempt at this decode (`probe_vdecode.py`: fit residual was at the noise floor while the
  entries came back 47% negative and mean |value-code| 1.46 - a least-squares solution
  amplified by kappa ~ 860, not a wrong layout).
- `probe_rowdec.py` - the row-wise decoder.

#### Validation (the decoder is exact before it is trusted)

Run against the MODEL's own noise-free `ctx` (exported via `STAGE_PROB_OUT`), where the
answer is known by construction:

```
g0_low5 k0=0 head0: |value-code| 0.000001  match vs frag_off 100.0%  resid 0.00001
g0_high5 k0=0 head1: |value-code| 0.000002  match vs frag_off 100.0%  resid 0.00001
g1_low5 k0=1 head0: |value-code| 0.000003  match vs frag_off 100.0%  resid 0.00001   (all 8 probes 100%)
```

#### Result 1 - the oracle's column bits land where the port's ROW bits land

Reading the swizzle's own bit dependencies, then the oracle's from the coded-probe
amplitudes (scale- and sign-free, so immune to the fit problem):

| arg bit | port's **first** arg (`r`) | port's **second** arg (`c`) | **oracle's output channel `c`** |
|---|---|---|---|
| 0 | p3 | p0 | **p3** |
| 1 | p5 | p2 | **p5** |
| 2 | p6 | p7 | **p6** |
| 3 | p9 | p8 | **p9** |
| 4 | p4 | p1 | **p4** |

The oracle's *output channel* drives exactly the bits the port's *first* argument drives -
all five agree.  So the oracle calls `frag_off(output_channel, input_row, ...)`, i.e. **the
two arguments are swapped** relative to the port's `frag_off(k, c, 32)`.

#### Result 2 - confirmed per row: shape corr 0.9994..1.0000 on every row tested

`--gain1-select` isolates one input row at a time; each probe carries the coded V bytes.
Predicted vs oracle per-channel amplitude shape:

| probe | k0 | **swap shape** | swap \|ctx\| | swap signed | port shape |
|---|---:|---:|---:|---:|---:|
| `g0_low5` | 0 | **+0.9999** | +0.9930 | -0.9930 | +0.4678 |
| `g1_low5` | 1 | **+0.9994** | +0.9390 | -0.9392 | +0.4705 |
| `g16_low5` | 16 | **+1.0000** | +0.6154 | -0.2845 | (n/a) |
| `g32_low5` | 32 | **+0.9997** | +0.9797 | **+0.9797** | +0.4614 |
| `g48_low5` | 48 | **+0.9999** | +0.7849 | -0.7811 | (n/a) |

Every row: transposed layout 0.9994-1.0000, port's layout 0.46-0.92.  The layout answer is
uniform across rows and is not a coincidence of one row.  Supporting checks:
- `frag_off(c, k, 32)` spans 0..1535 - a valid layout inside the 2048-byte region (and
  `frag_off(c, k, 64)` spans 0..2047);
- the low5 probe's predicted amplitude pattern for row 0 is `0.25 0.50 0.25 0.50 ... 1.0 2.0`
  and the oracle's is `0.124 0.247 0.124 0.247 ... 0.495 0.990` - identical shape.

#### Result 3 - the residual: a per-row SIGN, not explained by any single bit

Signed correlation on the same probes: -0.9930, -0.9392, -0.2845, **+0.9797**, -0.7811 for
k0 = 0, 1, 16, 32, 48.  Only k0=32 comes out positive, so it is *not* `bit5` (which would
predict + for 48), nor parity, nor any half/quarter split tested.  |ctx| correlation stays
high (0.61..0.99), i.e. magnitudes are right and the per-token sign is inverted for most rows.

This is exactly what makes the fix fail to transfer to real weights, where all 64 rows add
together and rows with opposite signs cancel:

| V read on REAL weights | corr h0 | corr h1 |
|---|---:|---:|
| port `frag(k,c,32)` (baseline) | +0.1884 | +0.2333 |
| **swap + negate** | **+0.3119** | +0.0948 |
| swap, no negate | -0.3119 | +0.0948 (sign of h0 flips) |

and on the all-rows coded probes the swap reaches only ~0.32 - consistent with per-row
cancellation, not with the layout being wrong (Result 2 fixes the layout).

#### Result 4 - still standing from attempt 8

- attention **weights right** (port softmax 83.9% of head0 variance vs 49.8% shuffled /
  61.4% uniform / 59.8% random controls);
- all 4096 V bytes live; **nothing landed** - no edit has gone into `sr_stage.comp`.

**Next step:** resolve the per-row sign, which is now the only thing between this decode and
a landing.  Two routes, in order: (a) map the sign over more rows (`--gain1-select 2,4,8,
24,40,56`, one oracle run each) and fit it against the row index / the byte's decoded value
/ the sign of `g = sum_j prob(t,j) s_norm[j,k0]` - if it is a property of `g` rather than of
the bytes, the model's `prob` for that probe is the thing to fix, not V; (b) if the sign map
is stable per row, build the corrected read as `Wv[k,c] = s_k * LUT[buf[vbase + frag_off(c,
k, 32)]]` with the measured `s_k`, verify real-weights corr(ctx) rises well past 0.2019, and
only then implement it in `shaders/sr/sr_stage.comp` and re-measure per-launch with
`launch_report.py` (provenance trio attached).
