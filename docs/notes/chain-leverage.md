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
