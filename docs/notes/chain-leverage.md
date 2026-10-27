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
