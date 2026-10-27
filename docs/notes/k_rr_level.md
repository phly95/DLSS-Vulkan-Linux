# Notes: `k_level` / `rr_level` — where the two recorded rows came from

Question raised by `MIMO-ASSESSMENT-2026-10-25_230151.md` §5.6: `docs/PORT-STATUS.md`'s `k_level`
row reads **max |err| inf, 3824/1310720 (0.2917%) bit-exact**, while the same row in commit
`4ec6263` read **max |err| 1.25, 3789 (0.2891%)**. Same launch, or two? Written 2026-10-25 from
`git log -p`, `git show`, and the restored oracle dump; nothing here was guessed.

## Verdict

**Same oracle launch, different port measurements.** The two rows are two diffdumps of the same
oracle file against two *different* port outputs: the port's own first-element bytes differ
(0.140625 / bits 33 vs 1.125 / bits 57), so the port side was rebuilt and re-dumped between them —
it is not one measurement restated. The only committed change to `shaders/rr/rr_level.comp` in
between is `706a63e`'s rewrite (evidence below).

## Evidence: which commit pasted which number

| number | first commit that contains it | commit date/time | what that commit did |
| --- | --- | --- | --- |
| 3789 (0.2891%), max \|err\| 1.25, first mismatch port 0.140625 (bits 33) | `1695015` "docs/PORT-STATUS.md: paste diffdump numbers for k_sr_stage, k_sr_l0, k_level" | 2026-10-25 10:31 | docs-only: replaced `k_level`'s "Offline verification O7 passed" with the pasted diffdump |
| 3824 (0.2917%), max \|err\| inf, first mismatch port 1.125 (bits 57) | `706a63e` "rr_level: implement collaborative 128-thread Swin Transformer layer in LDS with 0 spills on GFX1201" | 2026-10-25 17:08 | rewrote `shaders/rr/rr_level.comp` (+168/-8) **and** updated the row in the same commit; the row's new Notes cell describes exactly that implementation (LDS staging, 2-head Swin attention, 48 VGPRs) |

`4ec6263` (2026-10-25 14:25) did **not** touch the `k_level` row: `git show 4ec6263 --
docs/PORT-STATUS.md` shows only the `k_sr_tube_split` and `k_sr_post` rows changed. Its version of
the row is inherited verbatim from `1695015`.

Between the two pastes (`1695015`, 10:31, and `706a63e`, 17:08) `shaders/rr/rr_level.comp` was
touched by exactly one of the 13 intervening commits — `706a63e` itself:
`git log --oneline 1695015..706a63e -- shaders/rr/rr_level.comp` prints only that commit, and
`git log --all --oneline -- shaders/rr/rr_level.comp` shows the file was created by the reverted
draft `560ca36` (reverted in `96ee848`), committed as breadth code in `6135cee` (00:32), and
rewritten in `706a63e`. So the port shader behind row 1.25 is the breadth version, and the only
committed change that can explain row `inf` is `706a63e`'s rewrite in that same commit.

## Evidence: same oracle launch

Both rows are for buffer `E0b54`, 1310720 fp8 elements, oracle first-mismatch value **0.1015625
(bits 29)** at index 0. In the restored dump (`build/oracle/oracle-8dbf027a9e-s20261025/rr_F/dump/`):

```bash
$ build/venv/bin/python -I tools/verify/diffdump.py launches \
      build/oracle/oracle-8dbf027a9e-s20261025/rr_F/dump/frame0 | sed -n '5p'
  4 _ZN3rrk7k_levelILi32ELi2ELi5ELi0ELi0ELi4EEEvNS_4ArgsE grid [33, 21, 1] block [32, 1, 1] wrote E0b54, E0ba8, E0bc0

$ od -An -tu1 -N1 build/oracle/oracle-8dbf027a9e-s20261025/rr_F/dump/frame0/004.E0b54.bin
 29
$ od -An -tu1 -N1 build/oracle/oracle-8dbf027a9e-s20261025/rr_F/dump/frame1/005.E0b54.bin
 17
$ cmp build/oracle/oracle-8dbf027a9e-s20261025/rr_F/dump/frame1/start.E0b54.bin \
      build/oracle/oracle-8dbf027a9e-s20261025/rr_F/dump/frame0/004.E0b54.bin && echo identical
identical
```

- frame 0, launch 4 is `rrk::k_level<32,2,5,0,0,4>`, the launch that writes `E0b54`; its output's
  first byte is 29 = 0.1015625 — the oracle value **both** rows report.
- frame 1, launch 5's `E0b54` starts with byte 17 = 0.06640625, which matches neither row → ruled out.
- `frame1/start.E0b54.bin` (the buffer's state before frame 1) is byte-identical to frame 0's
  launch-4 output, so if either measurement used the frame-1 copy it is the same content anyway.

The port side differs between the rows (0.140625/33 vs 1.125/57), i.e. two different port dumps of
that launch.

## Why one row says `max |err| inf`

`tools/verify/diffdump.py:123` maps NaN differences to infinity
(`d = np.where(np.isnan(d), np.inf, d)`; both-NaN counts as 0). The oracle file
`frame0/004.E0b54.bin` contains **zero** `0x7f` bytes (E4M3 NaN), measured:

```bash
$ build/venv/bin/python - <<'EOF'
import numpy as np
a = np.fromfile('build/oracle/oracle-8dbf027a9e-s20261025/rr_F/dump/frame0/004.E0b54.bin', dtype=np.uint8)
print(a.size, int((a == 0x7f).sum()))
EOF
1310720 0
```

so at least one `0x7f` (NaN) came from the **port** side of that measurement — the oracle file has
none. The earlier row (`1695015`) reported a finite max error, so the NaN belongs to a port dump
taken after that measurement; whether it came from `706a63e`'s rewrite or an intermediate
uncommitted state is not recoverable (below). (Both rows' bit-exact counts are ~0.29% either way;
the rewrite moved individual values a lot without changing how many words matched.)

## What could not be determined

- Which exact working tree produced the `inf` row's port dump. No port dump for frame 0 launch 4
  exists under `build/` and none was committed, so only the pasted row survives; `706a63e` is the
  only committed change consistent with it, but the diffdump itself may have run on that tree or on
  an uncommitted intermediate state — the history cannot tell which.
- Whether the rewrite made `E0b54` better or worse **overall**: re-running `diffdump.py compare` on
  the RX 9070 XT (port dump of frame 0 launch 4 vs the restored oracle) is required before either
  number is cited as the kernel's current state. Until then the row stays APPROX and both numbers
  stand as two dated measurements of two shader versions, not as a contradiction.

## Review R7 (2026-10-26): the k_level block, second hypothesis (`tools/rr/rr_level_model.py --set blk=v2`)

Read from the RR packer (`rr_pack.build_w`, docs/rr/LEVEL-LAYOUT.md) and the ISA of
`k_level<64,2,1,96,0,1>` (`tools/re/ssa.py`; listing `build/work/review/rr/u1.ssa`). This is
SR's block with these RR differences:

- **Windows are 8 x 8 (64 tokens)**, one per workgroup (grid (W+sx+7)/8). The packer's "gram"
  (g x 64 x 64, symmetrised and multiplied by 2*sqrt(2), the same factor SR applies to its 16 x 16
  v258) is the **relative-position bias** for those windows: it is translation-invariant with a
  constant diagonal. The logit is `v_fma_mix(relpos, 1.0, Q.K)` (a K = 32 fp8 WMMA pair).
- **No RMSNorm**: the RR packer packs no gains, the kernel has no rsqrt, and Q/K are
  `fp8(clamp(f16-chunked(W @ x), +-448))` of the stored input itself. `norm=none` beats both norm
  forms on every unit.
- **LayerScale** from Args +0xac / +0xb0 (f16 bits; `s1d2` / `s1d4`, about +-0.02..0.04):
  `x += s1d2 * f16(Wo ctx + v128)` and `x += s1d4 * f16(fc2 h + v158)` (ISA: `v_pk_mul_f16 s29 /
  s30, ...`). Without it every unit is at -24..-31 dB.
- Softmax entirely in f16: the same exp (constants 0x2466 / 0x3873 / 0x3b6b / 0x3d80, mantissa
  shift), an f16 sum (packed adds + `v_permlanex16`), `v_rcp_f16`, `prob = f16(pe * inv)`.
- The MLP activation is NeuralHub's f16 GELU spline (0xad30 / 0x3698, SR Review R4). V = K (f16).
  Q is rows 0..31 of the group and K rows 32..63 (swapping them drops r to about 0).
- Args +0xa8 (`s1d0`, about 2) and the +0x88 input are used only at the end, in the head /
  extra-output path (with x0.25 = 2 x 2 averages), not in the block.

Measured, frame 1, mode-0 units, whole block vs oracle (PSNR / r):

| launch | v1 | v2 (`norm=none act=nhgelu`) |
|---|---|---|
| 7 (unit 2, C96 g4) | -24.19 / -0.055 | 17.01 / 0.787 |
| 8 (unit 3, C128 g4) | -25.46 / -0.041 | 14.19 / 0.796 |
| 9 (unit 4, C160 g8) | -31.00 / -0.049 | 19.08 / 0.887 |
| 10 (unit 5, C160 g8) | -28.88 / 0.154 | 13.30 / 0.685 |

Attention isolated with an rrref probe (MLP items 4..7 zeroed in all 16 blobs;
`build/work/rrref` built from tools/rr/rrref/rrref.cpp, which reproduces all 63 frame-0 oracle
dumps byte-identically). Correlation of the attention delta (output - input), frame 0:
- decoder second blocks (units 9 / 11 / 13): 0.979 / 0.991 / 0.990, best scale about 0.97;
- encoders (units 2 / 3 / 4 / 5): 0.688 / 0.768 / 0.823 / 0.944;
- unit 7 (decoder): 0.896.

The QK logits are large (rms 27..294, max up to 1062), so the exp saturates and the relpos bias is
numerically inert at x0 and x4 alike. The f16 K-chunk accumulation (`acc16`) is also inert. The
encoder residual is uniform over windows and in-window positions, so it is not a border or
shift effect. Open: the encoders' attention gap (next instrument: patchprobe on rrref to read the
oracle's logits), the entry modes 1 / 5, mode 2's skip order, and the heads.

### Review R7b (2026-10-26): attention is Q.Q^T, the skip is scaled by s1d6. All 16 block outputs reach 34-45 dB per launch

Instrument: `tools/re/patchprobe.py --net rr` (rrref, rr.co) on `k_level<96,4,0,128,0,1>` (unit 2) at
0x10bc. It stores the logit WMMA output (v86..v100, packed f16) and then both WMMA operands
(A = v134..v149, B = v150..v153).

- **Both logit operands hold the same projection**: rows 0..31 of the head's group. Matched as row
  sets, 24 of 32 rows are exact, in the model's token order, with reflected border tokens
  reproduced. So **logits = Q.Q^T + relpos**, symmetric, which is why the packer symmetrises
  the relpos table. Rows 32..63 are the **value** projection.
- The oracle's logits: rms 4.2, max 36. They correlate 0.87 with relpos alone; the old Q.K
  hypothesis gave rms 7..10 and r 0.46.
- P @ V is an fp8 WMMA (the RR kernels have no f16 WMMA): `prob`, V to fp8 (`pv8=1`).
- Mode 2 (decoder entry): **x = up + s1d6 * skip**, with s1d6 = Args +0xb4 as f16 (-1.04..+0.45;
  negative for some units, hence the earlier anti-correlation). Up-projection rows are stored
  (q, c), pixel shuffle q = dx + 2 dy.

Attention only (MLP zeroed, rrref probe): the change correlates 0.9975-0.9994 with the oracle's on
every unit tested (encoders were 0.69-0.94).

**Per launch, whole unit, real oracle inputs (`rr_level_model.py --all`, defaults = this decode),
PSNR dB / r:**

| unit (mode) | frame 1 out | frame 0 out | down |
|---|---|---|---|
| 0 (5, entry) | 36.75 / 0.9954 | 34.65 / 0.9954 | - |
| 1 (1, merge) | 39.65 / 0.9979 | 39.12 / 0.9976 | 39.95 / 38.52 |
| 2 (0) | **44.36** / 0.9990 | **44.73** / 0.9990 | 43.03 / 43.94 |
| 3 (0) | **41.18** / 0.9991 | **43.51** / 0.9991 | 39.37 / 38.99 |
| 4 (0) | 39.89 / 0.9989 | **40.11** / 0.9988 | 37.00 / 37.53 |
| 5 (0, bottom) | 25.65 / 0.9757 | 34.44 / 0.9971 | - |
| 6 / 8 / 10 / 12 / 14 (2) | 35.95 / 37.66 / 37.49 / 39.52 / 39.64 | 35.65 / 38.51 / 37.67 / 38.41 / 40.09 | - |
| 7 / 9 / 11 / 13 (3) | 36.76 / 38.45 / 39.71 / **43.30** | 36.89 / **40.47** / **40.18** / **42.88** | - |

The 2026-11-02 milestone (>= 40 dB per launch on `k_rr_level`) is met on several launches.
**Not decoded:**
- the per-level head outputs (`o98`, 8 B/px, and `oa0`), all <= 23 dB;
- unit 15 (mode 4) and its render-resolution outputs E0c02 / E0c16 / E0c1a / E0c1e, which feed
  `k_post_prep` / `k_post`;
- the +0x88 / +0x90 extra inputs, which are used only in that head path;
- unit 5's frame-1 gap.

### Review R7c (2026-10-26): the per-level heads decoded (o98, oa0)

Read from the ISA of `k_level<96,4,0,128,0,1>` (encoder) and `k_level<160,8,3,0,0,1>` (decoder).
In each formula, h_r is the f16 chain `bias_r + K=32 chunks of m108[r] . fp8(out)` over the
unit's own fp8 block output, `sg = f16(sigmoid(h))` with an f32 exp and rcp, and outputs are clamped
to 65504 (NaN -> 65504):

- **Encoder head** (units 0-5, 3 rows): `o98_c = f16(f16(s1d0 * sg_c) * x88_c)` for c < 3,
  lane 3 = 0. `x88` is the previous level's `oa0`, and for unit 0 the enc0 feature E0bec.
- **oa0** (encoders, next-level resolution): the 2 x 2 average of o98, each tap x0.25 in f16,
  summed as `(p00 + p10) + (p01 + p11)`.
- **Decoder head** (mode-3 units 7/9/11/13, 4 rows):
  `o98_c = f16(f16(f16(f16(s1d0 * sg_c) * up_c) * sg_3) + f16(f16(1 - sg_3) * x88_c))`, a gated blend.
  `up` = a 2x bilinear upsample (weights 0.75 / 0.25 by parity, border tap weight 0) of `x90`,
  the coarser decoder's o98. `x88` is the same-level encoder head.

Frame 1, per launch vs oracle (PSNR / r / bit-exact):
- o98, units 0..5: 89.65 / 1.000000 / 96.9%, then 51.43, 54.56, 54.79, 53.13, 45.13 dB;
- oa0, units 0..4: 86.08 (96.2%), then 55.43, 59.23, 59.87, 56.97 dB;
- o98, decoders 7 / 9 / 11 / 13: 62.27 / 61.39 / 58.73 / 60.56 dB (r 0.99999).

Non-exactness follows from the about 99% of fp8 block-output codes the heads read. Remaining in
k_level: unit 15 (mode 4) and its render-resolution outputs E0c02 / E0c16 / E0c1a / E0c1e, which feed
post_prep / post.

### Review R9b (2026-10-27): the SR rounding closure transfers to k_level - units 2..15 at 99.7-99.9%

The same kernel family, so the same rounding points. Each was read in `build/work/review/rr/u2.ssa`
(`k_level<96,4,0,128,0,1>`) and adopted only where the ISA shows it:
- **QKV** (`acc16=1`): K=32 chains added through f16.
- **Softmax** (`smax=isa`): the 64-key f16 tree. Per lane, ((tile0 + tile1) + (tile2 + tile3)) for
  each packed row pair, + the permlanex16 partner, ((rp0 + rp1) + (rp2 + rp3)), then lo + hi; then
  f16 reciprocal, prob = f16(e * inv).
- **P @ V** (`ctx16=2`): two K=32 key chains, ctx = fp8(f16(f16(P[:, :32] V[:32]) + P[:, 32:] V[32:])).
  This was the largest single step (84.9 -> 99.8% on unit 2).
- **Heads** (`chain=1`): the head loop's accumulator, a = v128, then a = f16(a + Wo_h . ctx_h);
  X1 = f16(x + f16(s1d2 * a)).
- **MLP** (`chain=1`): per 32-hidden block, z = f16(f16(b1 + c0) + c1 ...), h = fp8(gelu(z)),
  m = f16(m + h . W2_b) from m = b2; out = clamp(f16(X1 + f16(s1d4 * m)), +-448).
- **Projections** (`chain_io=1`): the down projection, the mode-1 merge, the mode-5 entry and the
  mode-2 up-projection are all bias-first K=32 chains.
- **Mode 2** (`up8=1`, `upref=low`): exactly SR's decoder rules (fp8-staged up-projection,
  low-res border reflection).

Model vs oracle (both frames, `--all`):
- block outputs of units 2..15: **99.73-99.89% bit-exact, 57.1-63.5 dB** (were 34-45 dB);
- unit 15's render outputs E0c02 / E0c16 / E0c1a / E0c1e: **100.0000%**, and E0bd4 99.93%;
- the down projections: 91.4-94.1% (their chain is not fully closed).

Open:
- unit 0 (mode 5 entry, 31%) and unit 1 (mode 1 merge, 42%): their entry structure is still the
  GUESS of R7, and the ISA must be read;
- unit 5 (the C=160 bottleneck) is 99.86% on frame 0 but 81% on frame 1, a frame-dependent
  border case, as mode 2 was.

R9b addendum: the border rule. k_level (ISA 0x1d0..0x280 of `<160,8,0,0,0,1>`) and k_sr_stage
(0x78..0xdc) reflect a token coordinate p as a = |p|, b = 2(n-1) - a, r = min(a, b, n-1), and
r = n-1 when b < 0. The model's old single reflection clipped beyond the double fold to 0.
Unit 5 (8 x 5, frame-1 shift (0,4): rows 8..11) goes from 81.03% to **99.875%**. Both models now
use the ISA rule, which is identical to a single reflection whenever |p| <= 2(n-1).

R9b addendum 2: the entries. Mode 1's merge (unit 1) and mode 5's two-GEMM entry (unit 0) are
staged in LDS as **fp8** after a bias-first K=32 chain, exactly as the mode-2 up-projection:
- mode 1: ISA 0x1200..0x14f8 of `<64,2,1,96,0,1>`, clamp +-448, cvt_pk_fp8, ds_store; no activation;
- mode 5: the activation between its GEMMs is NeuralHub's f16 GELU spline (the 0xad30 / 0x3698
  constants at the start of `<32,2,5,0,0,4>`), then fp8 staging.

Unit 0 goes 31.3% -> **99.77%** and unit 1 42.2% -> **99.74%**. All 16 k_level block outputs are
now **99.73-99.89% bit-exact vs the oracle on frames 0 and 1**; the down projections are 91.4-94.1%.
