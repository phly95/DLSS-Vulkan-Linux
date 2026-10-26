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
