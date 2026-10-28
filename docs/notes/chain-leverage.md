# Chain leverage: which kernel's error actually reaches the image

Written 2026-10-26 from `DLSS5_AMD_INJECT_DIR` runs of `build/vk/vk_runner` measured with
`tools/report/pipeline_report.py` (frames 0 and 1 — the two frames the oracle dumps cover).
METHODOLOGY (MIMO-ASSESSMENT §7.3, item 3): a kernel's leverage is how much end-to-end dB its
error costs, not how many bits of its output differ.

## How to reproduce

`src/vk/pipeline.cpp`'s `set_injection` makes the chain, after a named dispatch point, record
`vkCmdCopyBuffer` from host-filled shadow buffers holding the oracle's own bytes for that launch
(`tools/verify/diffdump.py` dump layout, `build/oracle/oracle-8dbf027a9e-s20261025/<cell>/dump/`).
Everything upstream of the point still runs as the port; from the point on the chain *follows the
oracle*. The reported PSNR of frames 0/1 is therefore what the segment downstream of that point
contributes. Unset, the env vars change nothing.

```bash
D=build/oracle/oracle-8dbf027a9e-s20261025/sr_M/dump       # .../rr_F/dump for RR
mkdir -p build/work/lev
DLSS5_AMD_INJECT_DIR=$D DLSS5_AMD_INJECT_MASK=post \
  build/vk/vk_runner --mode sr --preset M --weights build/ngx-weights/M \
  --in build/oracle/oracle-8dbf027a9e-s20261025/frames/sr --out build/work/lev
build/venv/bin/python -I tools/report/pipeline_report.py build/work/lev \
  build/breadth-baseline/sr_M_ref --name lev_post
```

Sanity check: `DLSS5_AMD_INJECT_MASK=` (empty = every point) gives 120.00 dB on frames 0/1 —
the injected frames are the oracle's bytes; frames 2+ have no dump and stay port.

## SR M 256x144 -> 512x288, frame 0 / frame 1, PSNR dB (pipeline_report.py)

| inject from | f0 | f1 | reading |
|---|---|---|---|
| (none, baseline) | 28.15 | 29.50 | the port chain as it runs |
| gather | 28.15 | 29.50 | gather is EXACT: zero error, as expected |
| enc0 | 27.17 | 29.08 | replacing t0/mv with the oracle does not help: enc0's error is not what reaches the image (see k_sr_enc0.md attempt history) |
| stage0 | 28.11 | 28.06 | stage0's own error ~0 |
| stage1 | 28.14 | 27.95 | ~0 |
| tube0..tube5 | 26.8-28.2 | 27.7-28.9 | non-monotonic (error cancels between stages); the whole tube block is worth <1.3 dB either way |
| stage8 | 30.61 | 31.62 | the segment stage1..stage8 (t7's error) is 6.6e-4 MSE, **~43% of the total error** |
| stage9 | 36.87 | 35.30 | **stage9's own t8 error: another ~43%** (MSE 8.68e-4 -> 2.05e-4) |
| dec0 | 39.63 | 36.43 | dec0's gates/factors/aniso as consumed by post: ~6% (MSE 2.05e-4 -> 1.09e-4) |
| post | 80.79 | 80.60 | post's own arithmetic given oracle inputs: ~7% (MSE 1.09e-4 -> 8.3e-9) |
| downsample | 120.00 | 120.00 | **downsample's error is ~8e-9 MSE: 80.6 dB chain contribution >= 60 dB, attempts here are over** |

Work order this implies for SR: `sr_stage` (stages 1..9, same shader) and `sr_tube_split` first
(they carry ~86% of the error), then `sr_post`, then `sr_dec0`; `sr_downsample`, `sr_gather`,
`sr_exposure` and (measured) `k_sr_enc0`/`stage0` get no further attempts.

## RR F 256x144 -> 512x288, frame 0 / frame 1, PSNR dB (pipeline_report.py)

| inject from | f0 | f1 | reading |
|---|---|---|---|
| (none, baseline) | 27.68 | 27.22 | flat across all 9 frames |
| gather | 27.68 | 27.22 | EXACT, zero error |
| ae_block | 27.68 | 27.22 | ~0 |
| ae_final | 27.65 | 27.19 | ~0 with the port's post; the exposure *wiring* matters only once post is right (below) |
| enc0_feat | 25.45 | 25.32 | replacing bec with the oracle's does not help: the chain's cf2 is tonemap(bec) but the oracle's cf2 is tonemap(**U-Net output**), and the port's rr chain never runs the U-Net into its output path at all |
| post_prep | 27.37 | 25.86 | oracle cf2 (all U-Net effects folded in) + port post + port downsample = still ~27 dB -> **rr_post is the bottleneck kernel** |
| post | 27.08 | 27.21 | oracle cf6/cfa but the port's *frame-param* exposure (1.0) instead of the oracle's AE exposure (1.3994): rr_downsample mis-scales |
| ae_final,post | 79.55 | 79.61 | oracle exposure + oracle post outputs: **rr_downsample contributes 79.6 dB: done, no further attempts** |
| ae_final,post_prep | 40.87 | 33.00 | oracle exposure + oracle cf2 -> **rr_post given fully oracle inputs contributes only 33-41 dB**: the one RR kernel to fix before the U-Net |
| downsample | 120.00 | 120.00 | harness sanity |

Structural facts confirmed by the dumps while taking these numbers (frame 0, `launches.jsonl`):

- the oracle runs **16 `k_level` instantiations** (launches 4..19: a U-Net 32->64->96->128->160
  with down/up/skip modes); the port dispatches one `rr_level` whose output `t_act[1]` nothing
  consumes and whose shader ignores its weights.
- oracle `k_post_prep` (launch 20) reads **E0bd4/E0c16/E0c1e (the U-Net's outputs)**; the port's
  `rr_post_prep` is fed `bec/gates/aniso` (pre-network state).
- oracle `k_post` (launch 21) additionally reads E0c1a (U-Net), E0bf6/bfa/bfe (mv features from
  `enc0_feat`), E0c06 (previous history); the port's `rr_post` reads none of these.
- the oracle's chain exposure is `E0c22` written by `k_ae_final` ([1.3994, 1.0, 0.7146, 0] on
  frame 0); the port's RR chain keeps the frame parameter (1.0) everywhere — `rr_ae_final`'s
  exposure output (`ae_exp`) is written and never read.

RR work order: (1) wire the AE exposure into the chain, (2) fix `rr_post` given oracle inputs
(33-41 -> >= 60 dB), (3) the 16-level U-Net + the post_prep/post rewiring onto its outputs.

## 2026-10-27 addendum: the table's biggest caveat

The leverage table above attributes ~86% of the SR error to `sr_stage`/`sr_tube_split`, but a
follow-up showed the attribution is about *where the oracle's buffers differ*, not about what
moves the referee number: **the port's SR network is fully decoupled from the output image**
(enc0 emits a bias-pattern placeholder t0; the mode-2 stages never receive their skip inputs
t1/t3; the restored attempt-5 post ignores gates/factors/aniso).  Every row of this table that
injects inside the network (enc0..dec0) measures a segment whose output the live image path
never consumes - which is exactly why those injections moved the number little or oddly.
The live segment is post -> downsample (post-inject 80.79 dB, downsample-inject 120 dB).
See `docs/notes/sr_chain_decoupling.md` for the proofs and the corrected campaign order.

## 2026-10-27 re-measurement (post-wiring, post-R9, post-`5981a2e`): the first valid network table

Same method and commands as above, re-run by the orchestrator at `731d767` (shader hash
`1e42c285471471ad40cc207a50a413ec74bbf371048236b9c0ef48dcc7222566`), inputs
`build/oracle/oracle-8dbf027a9e-s20261025/{sr_M,rr_F}/dump` + `frames/{sr,rr}`, scored per frame
with `tools/report/pipeline_report.py` against `build/breadth-baseline/{sr_M,rr_F}_ref`. The
2026-10-26 addendum's caveat no longer applies: the network is wired (`f0c244f`), `sr_stage`/
`sr_tube_split` implement R9 (`11f3f49`), `sr_dec0` and `sr_post` are rewritten (`1056352`,
`5981a2e`), `rr_level` implements R9b (`e6755ab`), and `rr_post_prep` consumes the U-Net outputs.

### SR M, frames 0/1 (the two frames the oracle dumps cover; full-sequence is 32.25/33.31)

| inject from | f0 | f1 | reading |
|---|---:|---:|---|
| (none, baseline) | 37.89 | 34.48 | the port chain as it runs |
| gather | 37.89 | 34.48 | EXACT, zero error (unchanged) |
| enc0 | **53.41** | **55.81** | **enc0's error now costs +15.5/+21.3 dB** - the 2026-10-26 "enc0 gets no further attempts" conclusion is superseded (it held only while the stage was wrong) |
| stage0 | 53.61 | 56.00 | stage0's own error given oracle t0 ~= 0 (row ~= enc0 row) |
| stage1 | 38.56 | 38.88 | +0.7 / +4.4 |
| tube0 / tube5 | 37.65 / 37.65 | 36.75 / 36.83 | ~0 / +2.3 - the tube block is still small |
| stage8 | 38.50 | 38.88 | +0.6 / +4.4 (launch 11 given oracle t7) |
| stage9 | **106.75** | **77.96** | launch 12's segment (incl. its t1 skip path) is the largest single block: +68 / +39 dB |
| dec0 | 108.48 | 77.96 | dec0's own error ~ +1.7 / ~0 |
| post | 118.83 | 119.31 | post's own error given oracle inputs = 77.96-108.48 dB (matches `5981a2e`) |
| downsample | 120.00 | 120.00 | done |

**SR work order now:** (1) `k_sr_enc0` - the spec is in flight (`tools/sr/sr_enc0_model.py`), t0 is
0.92% bit-exact in the shader and removing its error is worth +15.5/+21.3; (2) the mode-2
decoder segment (launches 11/12 and the t1 skip) - its per-launch numbers are 61-69 dB but the
segment still carries the most image error once enc0 is oracle; (3) the mid-stage rounding
residual (the model's own 99.08-99.99% vs oracle); (4) post's 1-ULP hardware residual last.

### RR F, frames 0/1 (full-sequence is 31.56/32.98)

| inject from | f0 | f1 | reading |
|---|---:|---:|---|
| (none, baseline) | 36.05 | 32.82 | |
| gather / ae_block / ae_final | 36.05 | 36.05 -> 32.82 | all EXACT (unchanged) |
| enc0_feat | 40.69 | 33.01 | E0be8 (72.9% bit-exact) costs +4.6 / +0.2 |
| post_prep | 40.86 | 33.00 | post_prep's own error ~0 (it is bit-exact); this row is rr_post's own error given fully oracle inputs: **40.86 / 33.00 dB** |
| post | 118.83 | 117.13 | rr_downsample is clean behind it |
| downsample | 120.00 | 120.00 | done |

**RR work order now (unchanged in kind, confirmed with fresh numbers):** (1) `k_post` frame-1 spec
(`tools/rr/rr_post_model.py`, in flight) + rewrite `rr_post.comp` - post's own error is the
whole gap; (2) `k_enc0_feat`'s E0be8; (3) the U-Net is no longer a lever (R9b made every block
99.7-99.9% and the image moved ~0.2-4.8 dB).

## 2026-10-27: preset L's leverage table - and the L chain's structural gap

The L table (same method, `sr_L/dump`, refs `build/breadth-baseline/sr_L_ref`) is **mostly flat and
must not be read as leverage** until the host's injection indices are made preset-aware:

| mask -> actual L launch | f0 | f1 |
|---|---:|---:|
| (baseline) | 23.08 | 27.06 |
| stage0-mask -> 3 = `k_sr_l0<0>` | 23.08 | 29.40 |
| stage1-mask -> 4 = stage0 | 25.18 | 27.98 |
| tube0-mask -> 5 = stage1 | 24.44 | 29.52 |
| dec0-mask -> 13 = stage9 | 23.08 | 27.06 |
| post-mask -> 14 = `k_sr_l0<2>` | 23.08 | 27.06 |
| downsample-mask -> 15 = post | 23.08 | 36.70 |
| every other row | 23.08 | 27.06 |

Two reasons the rows are not what their names say:

1. **`sr_l[]` and the `st_pt` points are preset M's launch numbers** (`src/vk/pipeline.cpp:899`).
   L inserts `k_sr_l0` at launches 3 and 14, so every mask from `stage0` on is shifted (the correct
   mapping is in the table above). Additionally the L dump's launch 2 (`enc0L_in`) contains **no
   `t0.bin`** (only `l0_in` + the three MV outputs), and `l0_in` is not in `inj_targets`, so the
   `enc0` mask can inject nothing that reaches the image - hence that flat row.
2. **The port's L chain is structurally different from the oracle's** (all verified from
   `launches.jsonl` pointers and the sources today):
   - oracle: `enc0L_in`(2) writes `t0` + `l0_in`; `k_sr_l0<0,2>`(3) reads `l0_in`/`l0_skip` and
     writes **`t0`**; stage0(4) reads `t0`. The port: `sr_enc0L_in` writes only `l0_in` content
     into `t_act[0]`, **`shaders/sr/sr_l0.comp` is a pass-through stub** (`out_act[idx] =
     in_act[idx]`), and stage0 reads `t_act[0]` - so the real `k_sr_l0<0>` computation never
     happens.
   - oracle: stage9(13) -> **`k_sr_l0<2,2>`(14)** reads `t8`/`l0_skip` and writes
     `gates`/`factors`/`aniso`/`hidden0`; `k_sr_post<true,...>`(15) reads them (plus `g_mvhi` at
     0xe0). The port: **`sr_dec0` (M's kernel) is dispatched unconditionally** where the oracle has
     `l0<2>` (`pipeline.cpp` section 5 has no preset branch), and `sr_post` runs M's variant without
     `g_mvhi`. This is the L analogue of the old M decoupling: `docs/notes/k_sr_dec0.md` already
     says "Preset L has no dec0. Its equivalent heads come from `k_sr_l0` (launch 14), not yet
     decoded" - the chain substitution itself was not written down until here.

**Consequence for the campaign:** preset L (a V4 cell set: `sr_L_seq`, `sr_L_static`, `sr_L_q15`,
`sr_L_q15_ae`) cannot pass while `k_sr_l0` (both template variants) is a pass-through stub and the
chain runs M's dec0 in its place. Work order: decode `k_sr_l0<0,2>` (l0_in -> t0) and
`k_sr_l0<2,2>` (t8 -> heads) against the sr_L dump, then rewire the L branch (stage0 input from
l0's t0, dispatch l0<2> instead of sr_dec0, L's post + `g_mvhi`), and make the injection indices
preset-aware so L's leverage becomes measurable. The kernel itself is at 0.0266% bit-exact today
(PORT-STATUS `k_sr_l0` row).

## 2026-10-27 post-enc0 re-measurement (after `bffc60f`), SR M frames 0/1

| inject from | f0 | f1 | reading |
|---|---:|---:|---|
| (none, baseline) | **53.82** | **54.48** | enc0's error is gone (+15.5/+21.3 of the previous table recovered) |
| enc0 | 53.41 | 55.81 | ~0 (slight cancellation noise) - enc0 DONE |
| stage0 | 53.61 | 56.00 | ~0 |
| stage8 | **66.00** | **60.44** | the mode-2 pair (launches 11/12) is the top remaining lever: +12/+6 dB |
| stage9 | **106.75** | **77.96** | launch 12's segment: +41/+24 dB - GPU==model is 100%, so the residue is the MODEL's (99.62-99.92% vs oracle on launches 11/12) |
| dec0 | 108.48 | 77.96 | dec0 own ~+2/~0 - effectively closed |
| post | 118.83 | 119.31 | post's own given oracle: +10 dB f0, 77.96 dB-equivalent f1 (its history path) |

New work order (supersedes the 10:xx one): (1) **mode-2 closure** - find launches 11/12's remaining
model-vs-oracle rounding point (GPU is exact vs model, so this is pure decode), (2) post's f1
history-path residual, (3) preset-L rewiring (separate, V4 for L cells - spawned), (4) the
frames 4-8 degradation is NOT visible to this instrument (no oracle dumps past frame 1):
extending the dumps (`srref --dump-frames` supports it - instrument work, operator-visible
decision) is the prerequisite for attacking the full-sequence min (33.76).

### Instrument update (2026-10-27 15:40): multi-frame oracle dumps exist now

The "no oracle dumps past frame 1" gap above is closed. Fresh 9-frame collection
(`build/prof1080/bin/{srref,rrref} ... --dump-frames 0,1,2,3,4,5,6,7,8`, same frames dirs,
into `build/work/multidump/{sr_M,rr_F}/`, 889 MB, under build/ and never committed):
**all 9 out_<i>.f32 byte-identical (sha256) to build/breadth-baseline/{sr_M_ref,rr_F_ref}** on
both cells - the fresh dumps ARE the reference sequence. Per-launch dumps now exist for
frames 2..8 (same layout as the existing oracle: launches.jsonl + NNN.<name>.bin).

To use them for injection/launch_report on frames 2-8, two follow-ups (host code): lift
`FrameChain::inj_wanted`'s `fc >= 2` guard (pipeline.cpp:688-ish) for the new dir, and add the
per-frame dump dir to the injection/measure plumbing. That is the prerequisite for attacking the
full-sequence min (SR M 33.76 at frames 4-8, RR F 32.26 at frame 8).

### 2026-10-27 17:00: frames 4-8 made measurable - a host bug found, and what the table says

Host fix (the operator's host-fix purview, orchestrator): `inj_wanted`'s `fc >= 2` guard removed,
AND a real bug fixed - `inject_after` keyed the dump dir on `fc`, which is the **8-frame history
ring** (`(fc + 1) & 7`), so frame 8 injected under `/frame0/` (proven: injected `out_8` was
byte-identical to `ref_out_0`). Injection now keys on `FrameParams.frame_index` (monotonic, set by
the runner; `pipeline.h` + `runner.cpp` + `pipeline.cpp`), falling back to `fc` when unknown (bridge).
Sanity: no-injection SR M unchanged at 33.76/45.77; mask-ALL now **120.00 on all 9 frames**
(instrument fully validated end to end).

SR M, all 9 frames (`pipeline_report` vs `sr_M_ref`; multidump dir):

| port runs from row onward (oracle upstream + oracle history) | f0 | f1 | f2 | f3 | f4 | f5 | f6 | f7 | f8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline (all port) | 53.82 | 54.48 | 54.61 | 55.09 | **33.76** | 36.44 | 39.84 | 41.19 | 42.66 |
| from stage0 (oracle gather+enc0) | 53.41 | 55.81 | 56.72 | 56.99 | 16.43 | 17.95 | 19.15 | 20.12 | 21.46 |
| from tube8/stage8 | 90.01 | 77.20 | 76.09 | 67.57 | 16.43 | 17.94 | 19.17 | 20.14 | 21.49 |
| from stage9 | 106.75 | 77.96 | 76.65 | 67.77 | 16.43 | 17.94 | 19.17 | 20.14 | 21.49 |
| from dec0 | 108.48 | 77.96 | 76.65 | 67.77 | 16.43 | 17.94 | 19.17 | 20.14 | 21.49 |
| from post (oracle everything above) | 118.83 | 119.31 | 117.61 | 117.39 | 119.64 | 119.18 | 118.93 | 119.71 | 118.58 |
| mask-ALL (ds only downstream) | 120.00 | 120.00 | 120.00 | 120.00 | 120.00 | 120.00 | 120.00 | 120.00 | 120.00 |

Readings:
1. **The whole f4-8 gap flows through the port's `sr_post`.** Give the image oracle post outputs
   (any mask including post) and every frame is ~119 dB; give post ANY upstream (oracle gates,
   oracle t8, oracle t0 - all rows) and f4-8 collapses to 16-36 dB, WORSE than the pure-port
   baseline. f0-3 behave as expected (the rows rise as the boundary moves downstream).
2. **Single-point injection is invalid at f4+ for methodology reasons** - the second compensating
   error: the pure-port chain is self-consistent (port gates + port history cancel), so breaking
   either side with oracle data exposes the other. The 33-42 baseline is the honest referee
   number; the 16-21 rows are NOT a segment measurement, they are the compensation breaking.
3. Non-monotone f3->f4 collapse (67 -> 16) with oracle upstream means the post/history loop
   DIVERGES nonlinearly when fed mixed data. Decomposing f4-8 further needs a mask design the
   current inject-after plumbing cannot express (per-frame masks, or history-buffer injection
   independent of the points) - queued as instrument work if the post-side fixes below do not
   already lift f4-8.
4. What this implies for the work order: **post's own frame fidelity + the history loop IS the
   f4-8 problem** (mode-2 closure and enc0 already landed or in flight lift f0-3, which the
   table shows are nearly fine: 53-55).

### 2026-10-27 17:30: the `hist` pseudo-point - frames 4-8 decomposed

New mask token **`hist`** (host, orchestrator): before a frame's first dispatch it copies the
ORACLE temporal state (the dump's `start.<name>.bin`, the buffer as it stood before launch 0)
over the chain's - the only way after-point injection can supply oracle history to a segment
that writes that history itself. SR list: ihist0/1, dhist, full15, hidden0/1, prev_mvec0/1,
mv_ero, mv_nondil. Sanity: no-mask SR M unchanged 33.76/45.77, zero INJECTSTART lines.

SR M, all 9 frames (multidump dir, `pipeline_report` vs sr_M_ref):

| mask | f0 | f1 | f2 | f3 | f4 | f5 | f6 | f7 | f8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline (all port) | 53.82 | 54.48 | 54.61 | 55.09 | 33.76 | 36.44 | 39.84 | 41.19 | 42.66 |
| `hist` (oracle history only) | 53.82 | 57.79 | 59.34 | 63.01 | **62.83** | 64.25 | 65.84 | 64.57 | 64.87 |
| `hist,dec0` (+ oracle gates/factors/aniso) | 108.48 | 77.96 | 77.24 | 68.02 | 70.72 | 70.13 | 70.10 | 70.09 | 68.60 |
| `hist,dec0,stage9` | (identical to hist,dec0 - stage9 adds nothing beyond dec0) |
| post outputs oracle (lev7 `up_post`) | 118.83 | 119.31 | 117.61 | 117.39 | 119.64 | 119.18 | 118.93 | 119.71 | 118.58 |

**Readings - the f4-8 gap, ranked:**
1. **Oracle history alone is worth ~+29 dB on f4-8** (33-42 -> 62.8-64.9; f0 unchanged - it has
   no history yet, a built-in sanity). The port-written history (dhist/ihist/full15/hidden) is
   the dominant f4-8 error carrier - and it is WRITTEN by post+dec0, so their per-frame errors
   compound into it.
2. Oracle dec0 gates add ~+7-8 more (68-71); stage9 beyond dec0 adds nothing (matches the f0/f1
   table).
3. **The remaining ~48 dB cap is sr_post's own history-path fidelity**: with oracle history AND
   oracle gates every frame, the port's post only reaches 68-71 dB at f2+ (f0's reset path is
   108). This matches sr_post_model's own decoded PSNRs (f1 66.4 dB) - i.e. even the SPEC of the
   history blend path sits there, attributed in k_sr_post.md to hardware v_div/v_exp/v_log being
   1 ULP from numpy. Next investigation: why does GPU post on ~oracle inputs score only 68-77
   (some input still differs? the +0xe0/+0xf0 push fields l-rewire found in M's args too? the
   model's transcendental emulation), judged on the FULL 9-frame table per the
   compensating-errors rule.

**Post-residual update (17:05): the +0xe0/+0xf0 push-field hypothesis is REFUTED for M.**
srref.cpp:506-518 is the authoritative PostArgs map: +0xe0 = g_mvhi (pointer), +0xe8 = oW/oH,
+0xf0 = the F34-derived mv pair (1.5,1.5 on these frames), +0xf8 = mvoff. M's `<false,12,16>`
template never loads >= 0xe8 (its kernarg loads: b128@0x0, b128@0x18, b64@0x28, b128@0x48,
b64@0x60, b512@0x88, b96@0xc8 = up to 0xe7, b128@0x48; `build/k_sr_post.s`), so it cannot read
oW/oH/mv-scale/mvoff at all - the port pushing zeros there is equivalent for M. What M's template
DOES load beyond the port's documented set: the 0xc8..0xe7 window (log2_10, depinv, ldr, mvhi,
g_mvhi pointer - the mvhi=false template strips the deref). L's `<true>` template (l-post's job)
loads differently. => M's post residual under hist,dec0 (68-77 dB at f1+) is NOT the push; it is
the history-path blend arithmetic itself, where sr_post_model (99.8% f1) is the current cap and
GPU==model 99.99% - i.e. the deviation of BOTH from the oracle is in the spec'd blend/encode
chain (t1/t2 conf/u fma chains, the log tonemap). Next: an SR-side probe-bisect of the blend
intermediate (the rr-post-model-2 rrp2_* method), queued for a free slot.

### 2026-10-27 17:40: mode-2 leverage re-check (before the queued resume) + history split

Fresh inject-after rows at HEAD `d73d418` (the mode-2 rows predate the hist work, per Claude
Opus 5.5's tuning note): stage8/stage9 rows **reproduce exactly** (stage9 f0 106.75 / f1 77.96
vs baseline 53.82/54.48 = +52.9/+23.5 dB; M's chain is byte-identical since LP-4, so no shift).

History split (symlink dump forests - `hist` minus/only the dec0-written `hidden0/1`, no host
change; `build/agents/orch/lev10_*`):

| mask | f0 | f1 | f2 | f3 | f4 | f5 | f6 | f7 | f8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 53.82 | 54.48 | 54.61 | 55.09 | 33.76 | 36.44 | 39.84 | 41.19 | 42.66 |
| hist (all) | 53.82 | 57.79 | 59.34 | 63.01 | 62.83 | 64.25 | 65.84 | 64.57 | 64.87 |
| hist minus hidden | 53.82 | 56.02 | 57.83 | 61.24 | **33.74** | 40.21 | 44.46 | 47.77 | 53.70 |
| hist only hidden | 53.82 | 54.80 | 55.31 | 55.82 | **16.52** | 34.18 | 36.72 | 38.22 | 39.47 |

**Verdict (folds the tuning note into the queue):**
1. The full-hist +29 dB at f4 comes essentially entirely from `hidden0/1` (minus-hidden f4 =
   33.74 ~= baseline 33.76); the post-written history (dhist/ihist/full15) adds its share later
   (minus-hidden reaches 53.70 at f8 vs baseline 42.66).
2. **Hidden is written by dec0 FROM t8 = stage9's output - so mode-2's error is a root source of
   the f4-8 history carrier**, on top of its direct +52.9/+23.5 at f0/f1. Mode-2 is therefore
   STILL MATERIAL relative to the post/history chain: **the queued mode2-closure resume (X1-at-f16
   probe) proceeds as briefed when a slot frees.**
3. Caveat re-confirmed: subsets are non-additive (oracle-hidden ALONE reads 16.52 at f4, worse
   than baseline - the compensation effect again). Single-subset rows measure mixing, not shares;
   the additivity-invalid reading must not be quoted as "hidden's share is X dB".
4. Queue note (fresh-data discipline, per the same feedback): **before committing further RR
   effort** (enc0feat beyond its in-flight spec, or any RR chain work), re-run the RR inject-after
   leverage at current HEAD - the pre-port table (36.05/32.82 era) predates `5710959`'s rr_post
   port, and RR's flat per-frame numbers mean SR-style history reasoning does not transfer.

### 2026-10-27 18:00: the sr_post bil() fix landed - hist,dec0 row 68-71 -> 103.6-109.9

Commits `972efdc` (model: per-texel clamp law, f1 decoded 66.4 -> 97.58 dB) + `159a694` (shader:
x1/y1 from the UNCLIPPED index). Orchestrator re-runs: hist,dec0 row =
**108.48/109.94/104.49/103.95/103.90/104.39/106.58/103.59/103.64** (was .../68.02/70.72/70.13/
70.10/70.09/68.60) - min 68.02 -> **103.59**, mean 75.70 -> **105.44**. Full-9 judged frame-by-frame:
SR M mean 45.77 -> 45.79 (f3 +0.24, dips <=0.04), SR L mean 44.40 -> 44.41 (f3 +0.24, f4 -0.11 =
the compensating-errors effect). The SR f4-8 post/history chain is now capped only by the
hardware-transcendental 1-ULP class (104-119 vs the ~119 post-mask row); the remaining f4-8
baseline gap (33-42) is dominated by what the PORT's post WRITES into history when its other
inputs are port-class - next: fold the fix's per-frame gains into the work order (post's own
residual is closed as far as the spec can reach; upstream dec0/stage9 + l0's attention residue
feed the rest). RR carries the same double-clip (rr_post_model 365-366 + rr_post.comp) - queued
with the fresh-RR-leverage turn.

### 2026-10-27 19:00: fresh RR leverage (the queued re-table) + the reconciled SR L table

**RR F at HEAD (masks via multidump, all 9 frames, `pipeline_report` vs `rr_F_ref`):**

| mask | f0 | f1 | f2 | f3 | f4 | f5 | f6 | f7 | f8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 37.84 | 33.72 | 32.48 | 32.90 | 37.68 | 34.01 | 32.43 | 32.74 | 32.26 |
| gather / ae_block / ae_final | = baseline on every frame (exact) |
| **enc0_feat** | **54.31** | **55.51** | 55.99 | 56.42 | 54.61 | 55.44 | 56.11 | 56.38 | 56.51 |
| post_prep | 101.20 | 75.39 | 70.01 | 66.53 | 102.61 | 72.12 | 69.44 | 65.52 | 63.48 |
| post | 118.83 | 117.13 | 117.55 | 117.04 | 117.27 | 117.13 | 117.32 | 117.30 | 117.29 |
| downsample | 120.00 all frames |

- **enc0_feat is RR's TOP lever: +16.5/+21.8 dB at f0/f1** (the pre-port table's "+4.6" is
  superseded) - every frame reaches 54-56 with oracle E0be8. The port-fix child is in flight
  (three pinned bugs, `4d2b62c`).
- post_prep row (post's own given oracle cf2): after the RR bil fix (`c1d7850`) min 63.48 ->
  **70.01**, mean 76.26 -> **85.99** (f1 75.39 -> 101.59); the residual cap (f3/f5-f8 at 70-76)
  is the transcendental-ULP class + the unrenormalized two-candidate mix.
- gather/ae exact; downsample 120.

**SR L, reconciled (the named target; four commits of movement):**

| state | f0 | f1 | f2 | f3 | f4 | f5 | f6 | f7 | f8 | min / mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| dummy l0_in + l-post (`a58586b`/`d73d418`) | 36.09.. | | | | | | | | | 32.87 / 33.70 |
| real l0_in (`81b2015`) | 54.84 | 53.78 | 55.71 | 56.42 | **30.12** | 32.84 | 37.12 | 38.62 | 40.19 | 30.12 / 44.40 |
| + bil fix (`159a694`) | 54.84 | 53.86 | 55.72 | 56.66 | 30.01 | 32.80 | 37.07 | 38.57 | 40.18 | 30.01 / 44.41 |
| + sr_l0 ties-to-even (`fff6aaa`) | 55.04 | 54.16 | 55.84 | 56.74 | **30.08** | 32.82 | 37.11 | 38.60 | 40.19 | **30.08 / 44.51** |

**Named target: SR L's f4 min drain (30.08 vs the field's 54-57).** The l0 tie residue dominated
PRE-reset history and is fixed; after f4's reset the limiter is upstream - the mode-2/stage X1
f16 residue (`k_sr_stage.md`, mode-2 closure attempt 2) runs through L's stages with real
features now. L's mean is +10.8 dB over its pre-real-feature state; V4 needs min >= 50.

**Post-`1c6ba67` refresh (RR row note):** the fresh table's baseline row above is now the *before*
state - the enc0_feat port fix took the no-mask RR F chain to **34.77/40.33** (per-frame
54.31/38.15/35.52/36.33/54.61/38.39/35.54/35.36/34.77, every frame up, my re-run), with f0/f4
landing exactly on the enc0_feat-injection row (reset frames skip the history branch). The
residual to the oracle-perfect row on history frames is the port's MISSING history branch (F5
spec: gate=rst, E0c0e@C/2 + E0c12@C/4, pj, SC0*tm, c36; c22-family pending) plus post's
transcendental-ULP cap - the next RR task.
