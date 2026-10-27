# Step D: how far is the port from NeuralHub's own speed on this card?

**Question (operator, 2026-10-26):** per dispatch and per frame, at the real 1080p -> 4K size,
excluding setup and uploads, with `srref`/`rrref` timed the same way on the same machine and sizes.
**Bar corrected by the operator:** distance from *NeuralHub's own* speed, not NVIDIA-class 1-3 ms -
`DEEPSEEK-ASSESSMENT.md` measured `rrref` at ~13 ms/frame at 1080p -> 4K on this machine, so
1-3 ms must not appear here as a target. Whether either side fits a real-time budget is the
operator's judgment from the tables below, not this note's conclusion.

**Answer in one line:** SR is **~205x** NeuralHub's own frame time and 96.5% of that gap is one
kernel (`sr_stage`, 623x on its own, scalar where the reference is WMMA); RR is **~1.8-2.0x**, but
that ratio flatters the port, which runs 8 dispatches against the reference's 24 - `rr_pooled2` is
never dispatched and the U-Net's 16 `k_level` launches collapse into one `rr_level` whose output
buffer nothing reads.

## Provenance (the trio, per the orchestrator gate)

- **commit SHA:** `4c9a082` (this instrumentation, on top of `a7a14e7`). Every source file whose
  runtime behaviour can affect these numbers is byte-identical to that tree.
- **SHA-256 of the shader binaries:** `sha256sum build/shaders/*.spv | sha256sum` =
  `7c4b05c0bb5453d6b5e69424ebf3ffd0bfb9b49fd5d21dc410fd7f24a03d8455`
  (individual: `sr_stage.spv` `2cbcf650d9ff90c7`, `sr_tube_split.spv` `dd2f2df51d822dca`,
  `sr_post.spv` `47c967b61cbeeec0`, `rr_post.spv` `6ee656789bf1f447`).
- **input directory:** `build/prof1080/sr` and `build/prof1080/rr`, both
  `{"render":[1920,1080],"output":[3840,2160],"frames":9}`, generated this session as:

  ```bash
  build/venv/bin/python -I tools/sr/gen_frames.py \
      reference-nr-vulkan/docs/ngx-verification/310.8.0/single-frame-inputs/3840x2160.png \
      build/prof1080/sr --render 1920 1080 --output 3840 2160 --frames 9 --seed 12345 --path
  build/venv/bin/python -I tools/rr/gen_rr_frames.py  \
      reference-nr-vulkan/docs/ngx-verification/310.8.0/single-frame-inputs/3840x2160.png \
      build/prof1080/rr  --render 1920 1080 --output 3840 2160 --frames 9 --seed 12345 --path
  ```

- **logs:** `build/prof1080/{port_sr,ctl,port_rr,ctl_rr,ref_sr,ctl_ref_sr,ref_rr,ctl_ref_rr}.log`
  (under `build/`, not committed).

## Method

| side | mechanism | window |
|---|---|---|
| port | Vulkan `vkCmdWriteTimestamp` at `VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT` before/after each `vkCmdDispatch` (`DLSS5_AMD_PROFILE=1`) | `GPU_FRAME_MS` brackets `chain->record()` only - the `slurp` of textures and the host-side `fill_*` memcpy happen before it, the D2H readback and file write after |
| `srref` / `rrref` | HIP event pair around every `hipModuleLaunchKernel` (`REF_PROFILE=1`) | `ev0` is recorded **after** the host->device copies, `ev1` **before** the device->host copy - render work only |

The two timestamp queries bracket `vkCmdDispatch` alone, so the memory barrier that follows each
dispatch is charged to neither neighbour, and `OTHER_MS = FRAME - SUM(dispatches)` reports the gap
(measured 2.4-2.6 ms/frame on SR - barriers and inter-dispatch bubbles, ~0.23% of the frame).

**Instrumentation overhead, same binary, same inputs, sequential runs, GPU idle** - controls have
the env vars unset, so they exercise the exact code path a normal run uses:

| | control (off) | instrumented (on) | delta |
|---|---|---|---|
| port SR frame | 1092.1858 ms | 1091.9477 ms | -0.02% |
| port RR frame | 23.6595 ms | 21.4139 ms | -9.9% (run-to-run spread, see note) |
| `srref` frame | 5.2200 ms | 5.4049 ms | +3.5% |
| `rrref` frame | 11.9013 ms | 12.1687 ms | +2.2% |

Medians over frames 1-8. Two caveats this table is meant to expose rather than hide:

**Port RR is bimodal, not a point.** Frames within a single run cluster at ~21.2 ms or ~23.9 ms
(one control run: 21.08, 23.42, 24.45, 23.91, 23.89, 21.17, 21.33, 23.91), and medians across the
four RR runs I took were 21.41 / 22.46 / 23.06 / 23.66 ms - a ~10% spread on identical inputs and
identical binaries. Port RR's frame time is therefore quoted as **21-24 ms** and its ratio as
**1.8-2.0x**; the -9.9% "delta" in the table above is inside that spread, not an effect of the
instrumentation (which only inserts timestamps around calls that were already there). SR shows no
such spread: back-to-back instrumented and control SR runs differ by 0.02%.

**One earlier pair of controls was invalid and is excluded.** I launched two controls in parallel
and they contended for the GPU, one reporting 8.99 ms where the same binary measured 5.22-5.40 ms
on its own. Every number in this note was re-taken sequentially on an idle GPU.

## SR, 1920x1080 -> 3840x2160, frames 1-8 averaged, ms per frame

| port dispatch | port | NeuralHub (`srref`) | ratio | oracle launches |
|---|---:|---:|---:|---:|
| `sr_gather` | 1.4968 | 0.0502 | 29.8x | 1 |
| `sr_exposure` | 0.0004 | 0.0074 | 0.1x | 1 |
| `sr_enc0` | 1.6921 | 0.4309 | 3.9x | 1 |
| `sr_stage` | **1054.0208** | **1.6907** | **623.4x** | **4** |
| `sr_tube_split` (6/frame) | 21.1037 | 0.7316 | 28.8x | 6 |
| `sr_dec0` | 2.5366 | 0.2194 | 11.6x | 1 |
| `sr_post` | 8.0082 | 1.6995 | 4.7x | 1 |
| `sr_downsample` | 3.4742 | 0.5082 | 6.8x | 1 |
| **frame** | **1092.3329** | **5.3379** | **204.6x** | **16 vs 16** |

`sr_stage` is **96.49%** of the port's SR frame. Medians: port 1091.95 ms, `srref` 5.40 ms
(instrumented); 1092.19 / 5.22 ms with profiling off -> 209x. So the frame ratio is **202-209x**
whichever variant is used.

**SR is work-equivalent:** the port issues 16 dispatches per frame and `srref` issues 16 launches
(4 `k_sr_stage` templates + 6 `k_sr_tube_split` + gather/exposure/enc0/dec0/post/downsample), so
every row above compares like with like. `sr_tube_split` is one name on both sides, issued 6 times
per frame on both - 21.1037 vs 0.7316 ms is six dispatches against six launches.

## RR, 1920x1080 -> 3840x2160, frames 1-8 averaged, ms per frame

| port dispatch | port | NeuralHub (`rrref`) | ratio | note |
|---|---:|---:|---:|---|
| `rr_gather` | 4.4534 | 0.3678 | 12.1x | |
| `rr_ae_block` | 0.3644 | 0.0527 | 6.9x | |
| `rr_ae_final` | 0.0026 | 0.0084 | 0.3x | |
| `rr_enc0_feat` | 3.9034 | 0.7528 | 5.2x | |
| `rr_post_prep` | 1.7008 | 0.1675 | 10.2x | |
| `rr_post` | 5.3925 | 1.9862 | 2.7x | |
| `rr_downsample` | 3.8279 | 0.4307 | 8.9x | |
| `rr_level` | 1.9497 | 8.2055 | n/a | **port: 1 dispatch; NeuralHub: 16 `k_level` launches; port output `t_act[1]` is never read** |
| `rr_pooled2` | never | 0.0810 | n/a | **never dispatched by the port** |
| **frame** | **21.5947** | **12.0524** | **1.8x** (1.8-2.0x across runs) | 8 dispatches vs 24 launches |

**The RR ratio is not a like-for-like comparison.** The port runs 8 dispatches where the reference
runs 24: `rr_pooled2` has a pipeline created (`add_pipe("rr_pooled2", ...)`) but no `dispatch()` call
anywhere, and the 16 `k_level` launches of the U-Net collapse to a single `rr_level` dispatch whose
output `t_act[1]` is written by that dispatch and read by no other - `rr_post_prep` instead consumes
`bec`, which `rr_enc0_feat` writes, so the U-Net's contribution never reaches the image. So the
port's 21.6 ms is measured on a chain that is **missing work the reference performs** - two
whole dispatch categories plus 15 of 16 level launches - rather than on the same chain run faster.
Compared only on the kernels the two actually share, the port is 2.7x-12.1x.

(`cf2` is *not* orphaned: `rr_post_prep` binding 3 is `writeonly OutTap` and writes it, and `rr_post`
reads it.)

## Why SR is 205x: `sr_stage` is scalar where the reference is WMMA

The two use identical launch geometry - both compute the grid as `(dimW + s0 + 7) >> 3` x
`(dimH + s1 + 7) >> 3` with block size 64, and both derive `dimW/dimH` from the same
`W0/H0` shifts (`L1w = W0>>1` etc.). The difference is the shader:

| | port `shaders/sr/sr_stage.comp` | reference `build/stage_isa.s` |
|---|---|---|
| `v_wmma_f32_16x16x16_fp8_fp8` | **0** (no `coopmat`/`cooperative`/`wmma` at all) | **152** |
| fp8 dequant | **11** `e4m3_lut[...]` table lookups in the hot loops | WMMA's fused fp8 input |
| shader instructions | 10293 | - |
| instruction mix | SALU 2945 > VALU 2439, VMEM 88, SMEM 362 | - |
| LDS | 40960 B -> **2 subgroups per SIMD** | - |
| VGPRs / spills | 256 / 0 | - |

Every GEMM in the port's `sr_stage` - Q/K projection, attention logits, V, the MLP's two
projections and the output projection - is a scalar `for k` loop with a 256-entry LUT lookup per
element (`sum_q += vnorm * float(e4m3_lut[uint(wbuf[off_q])])` and its four siblings). The
reference does the same 16x16 tile in one WMMA. On top of that, 40960 B of LDS per 64-thread
workgroup caps occupancy at 2 subgroups/SIMD, versus 32 for the port's LDS-free kernels - so the
scalar code also runs with no room to hide its latency. SALU exceeding VALU is the signature:
the kernel is bound by scalar address arithmetic, not by math.

The other port kernels are 3.9x-29.8x, not 623x. As arithmetic on the table above (not a
measurement, and with no claim about whether the result is desirable): replacing `sr_stage`'s
1054.02 ms with the reference's 1.69 ms leaves 40.00 ms/frame, still 7.5x; also replacing
`sr_tube_split`'s 21.10 with 0.73 leaves 19.63 ms, still 3.7x. The remaining ratio after both
would be carried by `sr_post` (4.7x), `sr_gather` (29.8x) and `sr_dec0` (11.6x).

## One honest paragraph

The port's SR budget is dominated to the point of irrelevance by one kernel: `sr_stage` is 96.5% of
frame time and 623x slower than NeuralHub's own `k_sr_stage`, because it implements fp8 GEMM as
scalar LUT loops instead of cooperative-matrix WMMA and then spends 40960 B of LDS to run at 2
subgroups per SIMD - a 1092 ms frame becomes ~40 ms if that one kernel matches the reference and
~20 ms if `sr_tube_split` does too, at which point the remaining 3.7-7.5x sits in `sr_post`,
`sr_gather` and `sr_dec0`. RR is 1.8-2.0x on frame time, but that number is measured against a
chain that omits `rr_pooled2` and collapses the U-Net's 16 level launches into one dispatch whose
output is never read, so it bounds the port's *current* kernels rather than a finished RR
implementation. **How far from NeuralHub these ratios leave the port, and whether that is close
enough for any target, is the operator's judgment**; the one thing this measurement settles
unambiguously is that no SR fidelity or parity work will be judged against a performance picture
in which `sr_stage` is still scalar.

## Re-measurement 2026-10-27 (idle GPU, after the cooperative-matrix rewrite and the Q.Q^T flip)

The gate had finished, so nothing else was loading the card. Same commands as above, medians over
the 9 frames of `build/prof1080/{sr,rr}`; per-frame totals, not per-dispatch means - the
per-dispatch spread on this machine is wide enough to be useless for claims (`sr_stage` individual
dispatches ran 16.45..71.70 ms in the same run), so the median over frames is what to compare.

| per-frame kernel total, ms | SR M | SR L | RR F |
|---|---:|---:|---:|
| `sr_stage` (4 dispatches) | **135.012** | 135.635 | - |
| `sr_tube_split` (6 dispatches, still scalar) | **90.253** | 91.004 | - |
| `sr_post` | 12.405 | 9.527 | - |
| `sr_downsample` | 3.482 | 3.481 | - |
| `sr_dec0` | 3.276 | 3.328 | - |
| `sr_enc0` / `sr_enc0L_in` | 1.589 | 0.485 | - |
| `sr_gather` | 1.837 | 1.489 | - |
| **whole SR frame** | **248.755** | **244.092** | - |
| `rr_level` (16 dispatches) | - | - | **1789.676** |
| `rr_gather` / `rr_enc0_feat` / `rr_downsample` / `rr_post` / `rr_post_prep` | - | - | 5.795 / 3.876 / 3.833 / 5.364 / 2.365 |
| `rr_ae_block` / `rr_ae_final` / `rr_pooled2` | - | - | 0.347 / 0.002 / 0.429 |
| **whole RR frame** | - | - | **1810.159** |

Readings, stated against the figures this file recorded before (`sr_stage` 96.49% of an
1091.95 ms SR frame; the stage family 1538.5752 -> 139.9240 ms/frame on the A2 check):

- `sr_stage` is now **135.012 ms/frame** (min 132.825, max 182.137 across the 9 frames), i.e. in line
  with - and marginally under - the 139.9240 ms the rewrite recorded. Part of the gap is plausibly
  the Q.Q^T flip dropping the `q8(K)` LDS stores, but that is **not attributed here**: a single run
  on this machine cannot separate ~5 ms from the noise that puts the frame max 59 ms above its median.
- The whole SR frame fell 1091.95 -> **248.755 ms** (4.4x), and `sr_stage` is no longer the whole
  frame: 54.3% of it, with `sr_tube_split` at 90.253 ms now the second term. **`sr_tube_split` is
  still scalar** - it was never rewritten for cooperative matrices - and is the obvious next target.
- **The A2 target of <= 50 ms/frame for `sr_stage` is not met and remains open**: 135.012 ms is
  2.70x the target. The 11.00x speedup over the scalar baseline is real; the target is separate.
- RR F is 1810.159 ms/frame and `rr_level` alone is **1789.676 ms of it (98.9%)** - 16 dispatches of
  a kernel whose block maths is scalar. RR's frame time will not move until that block does.

Caveat carried forward from this file: these are the port on an idle card, one run each, and the
per-frame maxima sit well above the medians (SR M frame max 338.110, RR F max 2168.596). Nothing
here is a comparison against `srref`; the reference-side numbers in this file were taken in the ROCm
container and were not re-taken.

## Reproduce

```bash
# port, per-dispatch (and the frame window is the same command without the env var)
DLSS5_AMD_PROFILE=1 build/vk/vk_runner --mode sr --preset M --weights build/ngx-weights/M \
    --in build/prof1080/sr --out build/prof1080/out_port_sr
DLSS5_AMD_PROFILE=1 build/vk/vk_runner --mode rr --preset F --weights build/rr/packed \
    --in build/prof1080/rr --out build/prof1080/out_port_rr

# references (ROCm container; REF_PROFILE unset = the control path)
podman run --rm --device /dev/kfd --device /dev/dri --group-add keep-groups \
    --security-opt label=disable -v "$PWD":/w docker.io/rocm/dev-ubuntu-24.04:7.2.4 \
    bash -c "cd /w && REF_PROFILE=1 build/prof1080/bin/srref \
        artifacts/neuralhub-v0.8.0/gfx1201/sr.gfx1201.co build/re/packed/M \
        build/prof1080/sr build/prof1080/out_ref_sr"
```

Note `build/ngx-weights/*` are absolute symlinks and do not resolve inside the container; the
container needs `build/re/packed/M` and `build/rr/packed` directly. Run the four cases
**sequentially** - two of my controls contended for the GPU and produced a 65% phantom gap before
they were re-taken.
