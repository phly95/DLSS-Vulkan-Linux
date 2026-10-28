# Handoff: state of the Linux/Proton port of NeuralHub (DLSS NR / SR / RR on RDNA4)

Read this first. It says what exists, how to reproduce it, what blocks the rest, and what must not
be retried.

## The goal (as given)

Port NeuralHub v0.8.0 (Windows: HIP kernels + D3D12 hooks) to Linux/Proton on SteamOS for the
RX 9070 XT (RADV GFX1201, Mesa 26.2+, FP8 cooperative matrices): NR, SR (DLSS presets M/L) and
RR (preset F) as Vulkan compute shaders, no HIP at run time, numerically matching the Windows build.
Seven deliverables in order: inventory, harness, NR, SR, RR, integration + installer, performance.
Constraint: the only Windows material is `NeuralHub_v0.8.0_setup.exe` (Mesa and other open tools
are allowed). The reference Vulkan NR port is `reference-nr-vulkan/` (MIT).

## Status

| # | deliverable | state |
| --- | --- | --- |
| 1 | kernel + interop inventory | **done**: `docs/inventory/KERNELS.md`, `docs/inventory/kernels-generated.md` |
| 2 | harness | **replaced in practice by running NeuralHub itself**: NeuralHub's code objects run unchanged under ROCm on this card (`tools/sr/srref`), so Windows dumps are no longer needed. `.hdump` + `tools/harness/compare.py` still exist (self-tested) for layer-by-layer compares. No Vulkan-side dumper yet. |
| 3 | NR | **runs on the RX 9070 XT with the real 310.8.0 model**: 45.59 / 47.97 / 49.00 dB vs NVIDIA's output at 1080p / 1440p / 4K (offline runner, `tools/hwtest/nr_gate.py`), the same as the reference's published figures. ≥ 49 dB at every size needs the 310.9.16 model, and the in-game runtime path is not measured yet. See `docs/hardware/RX9070XT-STEAMOS.md`. |
| 4 | SR | **in progress (breadth first)**. **Review R1 (2026-10-26): `k_sr_stage<64,2,0,96>` is decoded in the replica: per launch t1 34.63 dB / r 0.9973 and t2 30.58 dB, against the HEAD shader's 13.51 / 9.89. It is not in the shader yet; see "External review 2026-10-26" below, which supersedes this row's `sr_stage` description where they disagree.** All SR compute shaders implemented (`sr_gather`, `sr_exposure`, `sr_downsample`, `sr_post`, `sr_enc0`, `sr_dec0`, `sr_stage`, `sr_tube_split`, `sr_enc0L_in`, `sr_l0`). `sr_gather` and `sr_exposure` are EXACT (bit-identical). `sr_downsample` reached 75.05% bit-exact and 95.62% CLOSE against launch 15 via decoded 5-tap separable FIR filter. `sr_enc0` and `sr_enc0L_in` achieve 3 of 4 buffers 100% bit-exact with dynamic $W_{15}/W$ motion vector scaling. `sr_stage` implements full 16-token window multi-head attention with relative position bias (`v258`), 8 f16 WMMAs, MLP blocks, and 2x2 downsample projection in LDS with 0 spills, 0 scratch (108 SGPRs, 256 VGPRs). Attempts 2-3 added the ISA-derived fp16 exponential polynomial and balanced fp16 denominator tree, but frame 1 launch 3 still differs substantially: `t1` 6,086/655,360 exact (0.9286%), max error 18.5; `t2` 3,327/245,760 exact (1.3538%), max error 9.75. It remains APPROX; see `docs/notes/k_sr_stage.md`. `sr_tube_split` implements the bottleneck Swin Transformer layer in LDS with 0 spills, 0 scratch on GFX1201 (108 SGPRs, 256 VGPRs, 13-18 ms at 1080p). `sr_dec0` implements collaborative 16x128 shared memory staging with 0 spills, 0 scratch (108 SGPRs, 192 VGPRs, 4096 LDS) and row-major block indexing, achieving bit-exact factor channels (27, 5, 29) on pixel 0. As of Step B (commit `f0c244f`, 2026-10-26) **the SR network is wired end to end**: the mode-2 decoder stages now receive their skip inputs (`t3`/`t1`, oracle StageArgs offset 16) and `sr_post` reads `gates`/`factors`/`aniso`. The byte-identical wall is gone - injecting oracle buffers at all 12 network points used to leave all 9 `out_i.f32` byte-identical, now 9/9 differ (per link pinned with `DLSS5_AMD_DUMP_BUF`: `t7` moves with its skip `t3`, `full15` moves with `dec0`). `sr_post` is now a **hybrid (attempt 8)**: the decoded anisotropic 9-tap (`aniso`) and decoded gate/factor 3-way blend (`gates`/`factors`) carried over from attempt 6, on top of an invented stand-in bilinear MV warp and bilinear history sampler; `gates`/`factors` are also now `u8vec4`, matching what `sr_dec0` writes (the old `f16vec2` declaration was wrong about the type). The invented `0.90 - max_diff*2.0` floor and 2x2 colour box are gone. It is still a stand-in, never "the baseline" - the bilinear MV/history parts conflict with "Do not retry" below and are labelled as such in `docs/notes/k_sr_post.md` and `docs/PORT-STATUS.md`. Wiring it made `sr_post` faithfully consume `k_sr_dec0`'s still-wrong outputs, so the number fell **as pre-committed** (sr_M 32.20/33.76 -> 27.51/29.55); that drop is not to be reverted - see the no-revert rule in `docs/notes/sr_chain_decoupling.md`. Re-evaluating attempt 6's Catmull-Rom history is now unblocked but should wait until `k_sr_dec0` stops being wrong: a faithful consumer of wrong data has measured worse here twice. In `runner.cpp`, dedicated weight sub-buffers `w_enc0` and `w_dec0` eliminate cross-layer weight offset misalignment. Pipeline runner `vk_runner` runs SR M/L and RR F end-to-end natively on RX 9070 XT. See `docs/PORT-STATUS.md` and `docs/notes/`. **2026-10-27 (orchestrator, `11f3f49`): Review R9 + the R9 addendum are ported into `sr_stage.comp` / `sr_tube_split.comp`.** GPU vs model min 99.9544% (15/16 tensors 100.0000%) on launches 3,4,5,10,11,12 x frames 0,1 with oracle-input fixtures; GPU vs oracle 52.04-69.70 dB (r 0.999978-0.999999); model self-check PASS. Pipeline: SR M 32.25/33.31, SR L 22.88/28.18 (both HEAD baselines re-measured: 32.15/33.28, 22.70/28.31). 0 spills/0 scratch on device; O4/O7 green. **The 1080p->4K sr_stage timing budget is missed**: 234.187 ms/frame vs HEAD 144.468 same session (+62%; limit 162.014) - recorded in `docs/notes/perf_1080p4k.md`, register-chain optimization queued. Milestone re-verified: 8/8 k_sr_stage launches >= 52 dB per-launch. **Same day, later (`5177b5e`): the +62% timing regression is gone** - register-chained projections take 1080p->4K sr_stage to **92.700 ms** (budget 162.014; HEAD control 238.110; pre-R9 was 135.012) with fidelity bit-identical (min 99.9544%). **`k_sr_enc0` is decoded AND ported** (`b4180a2` model, `bffc60f` shader): t0 99.9638/99.9570% vs oracle (95.65/93.31 dB per-launch), MVs 100%, and SR M jumps to **min 33.76 / mean 45.77** (frames 0-3 at 53.8-55.1) - see the `k_sr_enc0` and `vk_runner` cells of `docs/PORT-STATUS.md`. |
| 5 | RR | **in progress (breadth first)**. **2026-10-26: the RR chain now issues NeuralHub's 24-launch structure.** It has 16 `rr_level` dispatches (one per `k_level` launch, with per-unit template ints, shifts and oracle buffer roles; table `kRrUnits` in `src/vk/pipeline.cpp`), `rr_pooled2` on non-reset frames, oracle-sized buffers, and dump/inject names for every E-buffer (inject points `L5`..`L20`; `DLSS5_AMD_TRACE=1` prints the dispatch list). **As pre-committed, the RR number fell to 1.80 / 1.82 dB** (from 26.78 / 27.17): `post_prep` now reads the U-Net outputs E0bd4/E0c16/E0c1e, which the not-yet-decoded `rr_level` leaves mostly zero. Do not revert this to recover the number. `k_level` per launch is now measurable (launch 5 baseline: -36.26 dB, r 0.027). RR shaders exist for gather, feature extraction, level, post-prep/post, downsample variants, pooling, and AE. `rr_gather`, `rr_ae_block`, `rr_post_prep`, and now `rr_pooled2` have measured exact launch results; pooled2 matches all 20,736 E0c12 words on its frame 1 steady-state launch. `rr_ae_final` matches the frame 0 reset launch exactly. A 7/8 target step now matches the frame 1 next-state half exactly (E0c2a 1/1), but E0c22 remains 3/4 exact with the temporal delta wrong; the family remains APPROX. See `docs/notes/k_ae_block.md`. `rr_enc0_feat` has 4 of 5 output buffers 100% bit-exact. `rr_level` remains APPROX. `rr_post` remains APPROX against launch 21: its packed 10:10:10:2 output is now wired through shader/runner/fixture, but attempt 1 differs at all 82,944 words (port 0xc89290a6 vs oracle 0xc1f0c032); current filtered RGB is not the source for that output. `rr_downsample` remains APPROX against its oracle launch. Current quality and timing evidence is in `docs/PORT-STATUS.md`; the latest 256x144 full-chain replay fails the referee's numerical target. **2026-10-27 (orchestrator, `e6755ab`): Review R9b is ported into `rr_level.comp`** (K=32 f16 QKV chains, the 64-key f16 softmax tree, two-chain P@V, head/MLP accumulators, bias-first fp8-staged entries incl. the mode-5 f16 GELU spline, mode-2 fp8 up + low-res reflection, the exact border rule; every rounding an explicit `rh()`). GPU vs model >= 99.9927% (frame 1) / 99.9976% (frame 0) over all 16 units x every buffer with per-unit oracle injection; GPU vs oracle k_level `out` 57.08-63.48 dB (30/30 >= 40 dB); 0 spills/0 scratch; rr_level 1080p->4K 1830.448 ms/frame vs HEAD 1822.050 (budget 2147.611, PASS). **The RR pipeline did not move** (31.56/32.98 vs HEAD 31.56/32.97): with every block output ~99.8% the gap is downstream - `rr_post` is the ceiling. **Ceiling removed (`b4cb89c` model + `5710959` shader):** k_post<24> fully ported - model frame 1 99.63-99.78%, shader GPU==oracle 99.85-99.998% both frames, RR F **31.56/32.98 -> 32.26/34.01**, timing +1.4..+11.4% (bar 20%), and the host's rr_post block now carries the oracle's 184-byte PostArgs, 12 bindings and 48x14 grid. |
| 6 | integration + installer | **installer implemented; both `EvaluateFeature` exports run the frame chain on the caller's resources**: `build/package/install.sh` supports install/remove, and `20261025_202951_2576121` passed V2, V5 and V7. V9 failed on the first Proton D3D12 case with 12 all-zero outputs because neither export did any work - that was true then, it is not true now: `8880861` bridges `NVSDK_NGX_VULKAN_EvaluateFeature` onto `ngxd::FrameChain` (image -> chain -> image, GENERAL in / GENERAL out, formats and subrect checked), and `341e9b3` does the same for `NVSDK_NGX_D3D12_EvaluateFeature` through vkd3d-proton's `ID3D12DXVKInteropDevice` / `ID3D12GraphicsCommandListExt`. Measured under Proton 10.0 with `ngx_client --api {vulkan,d3d12} --test-cycles`, 12 frames per cell, 0 GPUVM and 0 kernel page faults: RR F 1920x1080 -> 3840x2160 byte-identical to `vk_runner` on both APIs (largest per-frame gap 0.0000 dB), SR M Vulkan and D3D12 each equal to the native chain. **V9 is closed** (see `docs/OPEN.md`); what remains is listed there - a subrect smaller than the render target is refused, `FrameChain::record` resets its descriptor pool every frame, and two pre-existing shader/chain mismatches (`sr_exposure` bindings, `rr_level` push constants) are benign on RADV but wait on a measurement of their own. See `docs/PORT-STATUS.md`. |
| 7 | performance | **not passing the current target**: shader stats report 0 spills and 0 scratch, but the 256x144 oracle-sequence replay at commit `f0c244f` (2026-10-26, measured after the network was wired) gives SR M min/mean **27.51/29.55** dB, SR L **26.25/29.62** dB, RR F 26.78/27.17 dB (V4 requires >=50/55 dB; all three still fail). Median GPU times frames 1-8: SR M 60.176 ms, SR L 58.598 ms, RR F 0.760 ms. **The SR figure fell from 32.20/33.76 when the network was wired (commit `f0c244f`) - that fall was pre-committed, is not a regression, and must not be reverted to recover it** (the no-revert rule in `docs/notes/sr_chain_decoupling.md`); it is the expected cost of `sr_post` faithfully consuming still-wrong `k_sr_dec0`/`k_sr_stage` outputs. Earlier readings (32.20/33.76 pre-wiring, 27.79/30.06 with the a3a050b post) are superseded; this replay does not include ground truth. The run `20261025_202951_2576121` passed V2, V5 and V7 but failed V9 before oracle execution, so it produced no fresh V4/V8 result - **V9 has since closed** (row 6; `8880861` + `341e9b3`), but a full V1-V10 referee run has not been repeated since. The prior full run `20261025_142540_1926957` recorded 37.55 dB on `sr_M_seq`; old pipeline PSNR claims are superseded. See `docs/PORT-STATUS.md`. **Current (2026-10-27, orchestrator re-run at `e6755ab`, shader hash `1e42c285471471ad40cc207a50a413ec74bbf371048236b9c0ef48dcc7222566`, inputs `build/oracle/oracle-8dbf027a9e-s20261025/frames/{sr,rr}` vs `build/breadth-baseline/{sr_M,sr_L,rr_F}_ref`):** SR M min 32.25 / mean 33.31, SR L 22.88 / 28.18, RR F 31.56 / 32.98 - all with HEAD controls measured in the same session (32.15/33.28, 22.70/28.31, 31.56/32.97). 1080p->4K medians: sr_stage 234.187 ms (HEAD 144.468, +62% from the R9 chains) - **since fixed by `5177b5e`: sr_stage 92.700 ms** (same-session R9 control 238.110, ratio 0.389; sr_tube_split 130.985; rr_level 1830.448 vs HEAD 1822.050, PASS). **2026-10-27 after the sr_enc0 port:** SR M min **33.76** / mean **45.77** (was 32.25/33.31), SR L 22.88/28.18 (L chain still wrong - see OPEN.md), RR F **32.26/34.01** (was 31.56/32.98); shader hash `df5112610c76c10363d5a018563da7ed99e4ba3d3d051f82642ce7203f41c130`. V4 still fails every cell; the remaining SR min sits in frames 4-8 (history path).<br>**2026-10-28 (`9ff6ba9` reset-memset fix + `fb8e9a7` c22-family): SR M 53.82/55.64 V4 PASS, SR L 54.16/57.09 V4 PASS, RR F 54.02/54.46 (mean -0.54 under the bar) - 2/3 cells PASS.** Provenance and per-frame tables in docs/PORT-STATUS.md's V4 section; the f4-drain readings (33.76 f4, hist 62.83) are superseded - see the session 2026-10-28 section. |

## External review 2026-10-26 (read before resuming `k_sr_stage`)

An external review after attempt 15 (`dac531e`) measured the following. Details, tables, provenance
and reproduction commands are in `docs/notes/k_sr_stage.md`, "Review R1". No shader was edited.

- **The stage weight map is the packer's, not the probes'.** `docs/sr/STAGE-LAYOUT.md` (generated by
  `tools/sr/sr_layout.py` from `sr_pack.pack_layer`) gives every byte of every stage blob. Check
  hypotheses against it before crafting probes.
- **The attempt-15 "unread tail" lead is closed.** Those 24,768 bytes are the mode-0 downsample
  projection that produces **t2**, and the shader reads them. Zeroing them leaves t1 byte-identical,
  zeroes t2 and changes t3 (measured on the GPU).
- **t2 was never scored.** It is 9.89 dB / r 0.14 per launch at HEAD, and it is launch 4's input.
  Section 7 of `sr_stage.comp` feeds the (correct) projection the pre-MLP residual, with row-major
  token indices on a window-major tile and a raw-order bias.
- **Every per-channel f16 vector is read in the wrong order.** The packer stores them P-permuted;
  `read_f16(base + 2*c)` needs `invP`. `sr_tube_split.comp` has the same pattern.
- **Attention is V = K with a per-head output projection** (slot 0x60): 36.63 dB / r 0.9985
  attention-only against the oracle's own attention, where the port's is at the "no attention"
  level. The QK = 0 probes behind attempts 5-8's normalization conclusions had K = 0, hence V = 0.
- **The MLP is standard RMSNorm2 + GELU**: 34.42 dB given the oracle's input. `nr_swin_act` (the NR
  network's activation) is about 2x GELU. Attempt 14's "MLP = 76%" ceilings injected the oracle's
  attention downstream of the MLP.
- **Both landed gain-order commits are compensating errors.** `a903817` (gain1 first) and `83a5b91`
  (gain2 first) each offset a wrong attention or a 2x activation. The decode is gain-after for both
  norms, which agrees with the packer's own fp8 bound and with the 2026-10-24 ISA reading.
- Whole stage in the replica: 34.63 dB / r 0.9973 / 38.7% bit-exact on t1. The remaining ~5.4 dB to
  the 40 dB milestone is rounding closure (the f16 norm tree, the `g_sr_lut` rsqrt, the WMMA pairing
  through f16).
- **Landing caveat:** the per-launch t1 number is not monotone while parts are wrong (the decoded MLP
  with today's attention scores 11.07). Land these changes together, or judge each by its sub-block.

**Landed since (2026-10-26):**
- `shaders/sr/sr_stage.comp` and `sr_tube_split.comp` implement `tools/sr/sr_stage_model.py`. They
  are 99.73-100% bit-exact against the model on all ten stage launches, with per-launch PSNR equal
  to the model's (`docs/notes/k_sr_stage.md`, "Review R2").
- SR M replay 31.09 / 32.62 dB, SR L 32.13 / 33.00 (from 29.30 / 31.36 and 27.14 / 30.33).
- The RR chain runs the oracle's 24-launch structure (row 5).
- Delegated tasks follow `docs/agents/BRIEF-RULES.md`.

**RADV pitfall (measured 2026-10-26): `float(float16_t(x))` does not round.** RADV folds the f16
round trip away (8059 of 8064 test values came back unrounded), and `packHalf2x16` disagrees with
RNE on about half the values. Only a 16-bit memory store and load rounds. Every place a shader relies
on an f16 cast for rounding must use an explicit integer RNE, `rh()` in `sr_stage.comp`. The other
port shaders have not been audited for this.

**Corrections and open host bugs (2026-10-26):**
- **`tools/report/launch_report.py` printed a wrong bit-exact percentage before this date.** It
  compared the two byte strings as a whole, so every partial match read 0%. Earlier "bit-exact
  0.0000%" lines from it, such as Step C's baseline table, are wrong. PSNR, r and rRMSE were never
  affected. Fixed and covered by the self-test.
- **Host bug (FIXED 2026-10-27, `0e592f6`):** `rr_ae_final` was pushed 8 bytes (W, H), but the shader
  reads `scale` at push offset 24 and `reset` at 28. Its binding 1 (`previous_ae`) was bound to
  `expo`, and the injection names E0c22 / E0c2a looked swapped (`src/vk/pipeline.cpp` around the RR
  AE dispatch). Found by the f16 audit. The push now covers the fields the shader reads, `previous_ae`
  is bound correctly, the ping-pong order and reset seed follow `tools/rr/rrref/rrref.cpp`, and the
  frame-0 exposure is byte-exact against the oracle's own args
  (`[1.3994141, 1.0, 0.71458477, 0.0]`, both ping-pong slots). RR F mean 18.01/18.08 -> 18.67/18.76 dB
  (+0.68); SR M unchanged. `exposure[1]` on frame 1 remains the one gap: the undecoded temporal
  adaptation (the shader's `0.875` blend) leaves the port `-0.0055` against the oracle's `-0.0015`.
- **Host bug (FIXED 2026-10-27, `77fb6d6` + `1056352`):** the `sr_dec0` host push left Dec0Args
  +0x68..+0x9f zero, so the hidden-state warp ran with `r15w = r15h = 0` (px = mv alone, valid only
  where mv was 0) and with 1/W, 1/W15 and F28* all zero, reading the wrong tap everywhere. Every
  field is now reproduced from the oracle's own args on both dumped frames. The kernel itself was
  then rewritten against `tools/sr/sr_dec0_model.py` (Review R3b): GPU vs model
  99.99966..100.00000% bit-exact on gates/factors/aniso/hidden for both frames. SR M
  min 31.13 / mean 32.65 -> min 31.79 / mean 33.09 dB.
- **f16 audit:** explicit rounding (`shaders/include/rh.glsl`) brings `sr_downsample` from 75.30 to
  118.22 dB per launch and `rr_downsample` from 76.79 to 118.77 dB. `rr_post`, `rr_enc0_feat` and
  `sr_enc0` compute in f32 where NeuralHub uses f16 chains or a truncating quantizer, so they need
  their decodes, not a rounding fix.

Review recommendations, pending the operator: score every mode-0 stage on t1 **and** t2 plus three
sub-blocks (attention vs `--isolate`, MLP given the oracle's input, projection given the oracle's t1);
generate hypotheses from `sr_pack.py`, the decompile and the ISA, and use the srref probes to
validate rather than to explore.

## Session 2026-10-27: the stage family's logits, unit 15, and two closed host bugs

Nine commits, `57f5b62`..`HEAD`. Provenance for every number below is the commit named with it;
`sha256sum build/shaders/*.spv | sha256sum` is the shader identity, and the oracle inputs are
`build/oracle/oracle-8dbf027a9e-s20261025/{sr_M,rr_F}/dump` plus `frames/{sr,rr}`.

**The milestone is met.** `k_sr_stage`'s logits are Q.Q^T, not Q.K^T (`6a612b0`, Review R8): the
model default and both shaders changed together. Per-launch PSNR against the oracle, all twelve
stage launches of both frames, is now min 31.78 / mean 38.41 / max 42.96 dB with **12/24 launches
>= 40 dB**, against qq=0's 0/24 (min 18.66 / mean 27.41 / max 37.97). That is the 2026-11-02
milestone of >= 40 dB per-launch on `k_sr_stage` or `k_rr_level`. GPU fixtures confirm the operand
rather than merely tolerating it: 99.71..100.00% bit-exact against the qq=1 model, and 7.49..45.97%
against qq=0.

**Two host bugs closed.** `rr_ae_final`'s push now covers the fields its shader reads, with the
ping-pong order and reset seed from `rrref` (`0e592f6`); the frame-0 exposure is byte-exact against
the oracle's args. `sr_dec0`'s Dec0Args +0x68..+0x9f block is filled from the oracle's own args
(`77fb6d6`), and the kernel was then rewritten against its spec (`1056352`) to 99.99966..100.00000%
GPU-vs-model on both frames. Both are measured, not assumed: SR M went min 31.13/mean 32.65 ->
31.79/33.09 dB and RR F mean +0.68 dB.

**Two specs finished.** RR unit 15's dec0 heads are decoded (`e08ac56`): o98 68.15 dB, oc8 48.03,
od0 41.58, oc0 35.37 (signed i8 - it was being scored as unsigned, which cost ~12 dB for nothing),
od8 51.21, against 4.88 dB before. `k_sr_post` is complete (`6af7efc`): full15 99.9934%, ihist1
99.9895%, dhist 99.9976% on frame 0 and 99.80..99.85% on frame 1.

**Still open, stated plainly.** The V4 bar is not met and is not close: SR M is min 32.13 /
mean 33.61 dB against >= 50 / >= 55. `sr_tube_split` was never rewritten for cooperative matrices
and stays scalar. RR's `block_v2` residual (34-45 dB on every unit) is now the limiting factor for
the RR heads and was out of scope here. `rr_level.comp` still ports the old mode-4 guess, so unit
15's GPU/model parity follows from this session's spec but is not yet measured. `sr_post.comp` is
still the hybrid described in row 4; the spec it should now be rewritten against is committed.

**A cost the milestone does not show, measured both ways.** The qq flip moves the whole-pipeline
numbers differently from the per-launch ones, and the difference is not noise - both sides were run
twice and reproduce to the last digit:

| pipeline vs the oracle's own 9-frame output | qq=0 | qq=1 (`6a612b0`) | delta |
|---|---:|---:|---:|
| SR M min / mean | 31.79 / 33.09 | **32.13 / 33.61** | +0.34 / +0.52 |
| SR L min / mean | **24.10 / 26.26** | 23.78 / 25.84 | -0.32 / -0.42 |

So the flip costs SR L about 0.4 dB end to end. It is kept, for two reasons that are evidence and
not preference. First, per-launch against oracle inputs - the metric `launch_report.py` exists to
state, and the one the referee rules say governs kernel work - qq=1 wins on **both** presets, by a
wide margin: preset L's stage launches 4/5/12/13 go from 34.16/29.33/32.45/25.45 dB at qq=0 to
**44.48/37.84/44.37/33.87** at qq=1 (+7 to +12 dB), with r rising from ~0.993 to ~0.999. Preset L's
`k_sr_stage` is the same mangled kernel as preset M's, so the hardware cannot use one operand for
one preset and the other for the other. Second, Review R7b read this operand off the hardware. The
end-to-end L drop is the same effect already recorded for `sr_post` - a faithful consumer of an
upstream that is still wrong - and the chain's upstream (`enc0`, `dec0`, `post`) is what the stage's
per-launch measurement deliberately holds at oracle values. The next move on that number is to
fix the upstream, not to put a wrong operand back.

**Two tool fixes, both worth knowing about.** `tools/report/launch_report.py` gained an `i8` element
type (`8502fd2`) - signed activations scored as unsigned lose ~12 dB of apparent PSNR - and its
self-test had been failing since `1720a08` because it regexed `sr_stage.comp` for an `e4m3_lut` table
the cooperative-matrix rewrite replaced with bit arithmetic; it now cross-checks the shader's actual
`e4m3_to_f32`. The O7 fixture for `sr_dec0` was rewritten (`dd473af`): it had a 96-byte push for a
160-byte struct, a 4096-byte weight blob for an 8448-byte one, a 1-D grid for a 2-D tile kernel and
an all-zero input, any one of which makes every output constant.

**A measurement trap, recorded so it is not rediscovered.** `build/agents/stage-shader/fx/` feeds the
oracle's fragment-order `start.wbuf.bin`. Since `1720a08` the host repacks those matrices row-major,
and the stage kernels read the repacked copy (binding 5, `build/agents/stage-coopmat/fx/`). Any
shader run on the old fixtures now scores ~0.5% against the model **regardless of whether it is
correct**; use `build/agents/stage-coopmat/fx/` for `sr_stage`. `sr_tube_split` is scalar and still
reads the original layout, so the old fixtures remain right for it.

## Session 2026-10-27 (orchestrator continuation): R9/R9b landed in the shaders, the referee's red sr_post fixed, the milestone re-checked

Three commits, `9eb91dd` + `11f3f49` + `e6755ab` (pushed). Every number below was re-run by the
orchestrator; provenance is the commit named with it, `sha256sum build/shaders/*.spv | sha256sum` =
`1e42c285471471ad40cc207a50a413ec74bbf371048236b9c0ef48dcc7222566` for the combined tree (the
stage commit alone, `11f3f49`, is `62067af3cb872fdbc34c02382db709dd3325311a9453c3993fbe0d24b1caff46`),
inputs `build/oracle/oracle-8dbf027a9e-s20261025/{frames/{sr,rr},sr_M/dump,rr_F/dump}`,
`build/prof1080/{sr,rr}`, refs `build/breadth-baseline/{sr_M,sr_L,rr_F}_ref`.

- **Review R9 + R9 addendum -> the stage shaders** (`11f3f49`): GPU vs model 99.9544-100.0000%
  (15/16 tensors exact) on launches 3,4,5,10,11,12 x frames 0,1; GPU vs oracle 52.04-69.70 dB;
  pipeline SR M 32.25/33.31, SR L 22.88/28.18; 0 spills/0 scratch; O4/O7 green.
- **Review R9b -> `rr_level.comp`** (`e6755ab`): GPU vs model >= 99.9927% (frame 1) / 99.9976%
  (frame 0), 39 buffers x 16 units x 2 frames, none missing; GPU vs oracle k_level `out`
  57.08-63.48 dB (30/30 >= 40 dB); 0 spills/0 scratch; rr_level 1080p->4K 1830.448 ms vs HEAD
  1822.050 (PASS <= 2147.611).
- **RR end-to-end did not move** (31.56/32.98 vs HEAD 31.56/32.97): every k_level block output is
  now ~99.8% and the pipeline is unchanged, so the RR gap is downstream - `rr_post` is the ceiling
  (`docs/notes/chain-leverage.md`: 33-41 dB given oracle inputs). Highest-leverage RR step: finish
  `tools/rr/rr_post_model.py`'s frame-1 spec, then rewrite `rr_post.comp`.
- **The sr_stage timing budget is missed and recorded, not hidden**: 234.187 ms/frame at
  1080p->4K vs the 135.012 ms in `docs/notes/perf_1080p4k.md` (limit 162.014; same-session HEAD
  control 144.468, ratio 1.621; sr_tube_split 130.985 vs 100.622). The R9 chains cost an LDS
  store/barrier/combine per K=32 group. Fidelity-first is standing; queued optimization
  acceptance: GPU == model >= 99.9% unchanged, timing within budget, 0 spills.
- **A pre-existing referee failure was found and fixed** (`9eb91dd`): `offline_check` O7 had been
  red on `sr_post` since `5981a2e` - the fixture push predated PostArgs +0xa8..+0xcc, so the shader
  wrote constant zeros on RADV and lavapipe alike (HEAD's `sr_post.spv` byte-identical to the
  tree's: pre-existing, not from the R9 ports). The fixture now fills dec0_f1 / scaleX / scaleY /
  jitter / log2_10; `tools/verify/offline_check.sh` -> **PASS 27 pipelines**.
- **The 2026-11-02 milestone re-verified at this HEAD**: k_sr_stage 8/8 measured launches >= 52 dB
  and k_rr_level `out` 30/30 >= 57 dB per-launch vs the oracle (bar: >= 40 dB on either kernel).
- `REVIEW-BRIEF-2026-10-26.md` stays untracked scratch (its own header says it is not committed).

**Afternoon block (12:00-14:30, orchestrator; commits `9eb91dd`..`1b8498d`, all pushed):**
- `5177b5e` sr_stage register-chain optimization (92.700 ms, fidelity unchanged) + `d15197b` O4
  comment-lint fix (SPIR-V byte-identical).
- `b4cb89c` rr_post_model frame 1 (99.63-99.78%) -> `5710959` rr_post shader rewrite: P1
  99.85-99.998% vs oracle, RR F 32.26/34.01.
- `b4180a2` sr_enc0_model (A1-A4, double-verified) -> `bffc60f` sr_enc0 shader: t0 95.65/93.31 dB
  per-launch, SR M 33.76/45.77.
- `1b8498d` sr_l0_model: heads PASS (L2 8/8), **L1 frame-1 t0 at 95.94% documented as a GAP**
  (probe-bisected to k_sr_l0's attention; exact rounding point unbudgeted).
- The preset-L chain rewiring (l0<0,2> t0 into stage0, l0<2> instead of sr_dec0, L post + g_mvhi,
  preset-aware injection indices) is documented in OPEN.md and NOT yet started.
- Working tree clean at `1b8498d`; offline_check green (see PORT-STATUS referee line).

Next, in order: (1) ~~enc0 model + shader~~ DONE (`b4180a2`/`bffc60f`); (2) ~~rr_post model + shader~~ DONE
(`b4cb89c`/`5710959`); (3) ~~sr_stage timing~~ DONE (`5177b5e`); (4) **preset-L chain rewiring**
against `sr_l0_model.py` (the L cells are V4 cells); (5) re-measure chain leverage post-enc0 and
attack the SR frames 4-8 history path (post/dec0/ihist) - SR M frames 0-3 already sit at 53.8-55.1;
(6) RR: `k_enc0_feat` E0be8 (72.9%) per the leverage table; (7) `sr_tube_split` cooperative
matrices (perf, 130.985 ms).

## Session 2026-10-28 (orchestrator): the reset-memset bug fixed - SR M and SR L PASS V4; the c22-family decoded and ported - RR +14.87 dB mean

**The single biggest find: a verified HOST bug (`9ff6ba9`).** `FrameChain::record` memset
`prev_mvec`/`hidden`/`ihist` on EVERY reset frame. NeuralHub never does this (decompiled SR
launchers have zero memsets; reset reaches kernels only as an argument, Enc0Args +0xd8; srref
zeroes once at allocation). The oracle sequence resets at f0 AND f4, so the port wiped temporal
state at f4 that the oracle keeps - the reason SR's worst frame was always f4 with a slow f5-f8
recovery, and why the `hist` pseudo-point result (oracle start state lifting f4 33.76 -> 62.83)
and the f4-8 "compensating errors" decomposition were this bug showing through. `create_buffer`
zeroes at creation and the bridge runs frame_index = -1: do NOT re-gate the removal to frame 0.
Effect at `9ff6ba9` (orchestrator-verified, byte-identical 9/9 to the operator's independent
worktree recording): **SR M 45.79 -> 55.64 mean / 33.76 -> 53.82 min, SR L 44.51 -> 57.09 /
30.08 -> 54.16, both V4 PASS; RR byte-identical.** V4 is now 2/3 PASS; RR F sits at 54.02/54.46,
0.54 dB under the mean bar.

**The c22-family decoded and ported (`fb8e9a7`, agent e0be8-history).** The 8 open E0be8 channels
= `SC1 * bilinear(E0c02 ch/127)` sampled at `C/3 + pj_prev - 0.5` on the render grid (pj_prev =
the PREVIOUS frame's jitter, args +0x1a8/0x1ac - the missing piece vs the prior 71.7%-partial),
OOB-gated, dq'd; skipped on reset frames. Model F1 f0+f1 100.0000% (was 87.57%), shader H1
GPU==oracle==model 100% all 64 channels, RR chain 39.59 -> **54.46 mean (+14.87)**, every history
frame +16.9..+21.0 dB, f0/f4 exact, 0 spills, O4/O7 green. RR's next lever is the upstream
history/exposure ULP class (the c22 shader is spec-correct with oracle history inputs; the chain
sits just under the oracle-perfect 55.57-57.30 inject row) - operator-ordered PRE-requisite:
compare the port's per-frame start state vs the oracle `start.*.bin` dumps before more RR kernel
work.

**Also landed:** the fixed ctx instrument (`13eed3a`): probe patches at load-heavy points must
drain with `s_wait_loadcnt 0x0` FIRST (17 in-flight loads clobbered the patch's address VGPRs -
mode2's whole fault-address signature explained), use carry-free `v_mad_co_u64_u32` tails, and
pick scratch VGPRs >= the load destinations. First clean attention-boundary byte dump delivered;
the mode-2 X1-residue fragment-map closure is now offline-bounded work - but its leverage MUST be
re-measured on the reset-fixed chain first (old 33.76-era numbers superseded).

(1) ~~RR start-state comparison~~ DONE (`ce8e1d9`): no carry/clear/seed difference; the port's
mv-record carrier is byte-identical to the oracle's on all frames; the remaining RR delta is
in-flight ULP fidelity. (2) ~~mode-2 leverage re-measured on the fixed chain~~ DONE (`b11663b`):
stage9 inject = +40.45 min / +43.58 mean - launch 12's segment IS the entire remaining SR M
error. (3) ~~full `run_all.sh` referee~~ DONE (`56f6666`, 2026-10-28): V1/V2/V6/V7/V9/V10 PASS,
**V4 5/25 cells** (full matrix + failure classes in docs/PORT-STATUS.md "Referee run 20261028").
The pass revealed the largest named gap: **format-variant cells (col4-6, ldr1-3, rr fmt_a/b/c)
are catastrophic at 4-19 dB** - the port misreads non-default input bindings/formats; these
cells were never measured before. Then: (4) the format-variant class; (5) mode-2 X1 closure
(SR temporal cells 36-45 dB); (6) RR upstream ULP; (7) ae temporal state (fails only GT-delta,
by 0.001-0.109). V8 perf unchanged-fail (49-170x, OPEN.md).

## The real machine (where work happens now)

An RX 9070 XT under SteamOS 3.9 (`docs/hardware/RX9070XT-STEAMOS.md`). SteamOS's root is immutable and has
no compiler: everything is built in podman containers and the binaries run on the host. Everything
generated goes under `build/` (ignored by git); nothing from NVIDIA or NeuralHub may be committed except
the NeuralHub installer and the artifacts already listed below.

### Setting up from a fresh clone

```bash
git submodule update --init                       # reference-nr-vulkan (MIT), pinned at c71b0f4
(cd reference-nr-vulkan && bash fetch_deps.sh && python3 linux/build/build_network.py rdna4)
python3 -m venv build/venv && build/venv/bin/pip install numpy pillow pefile msgpack capstone
build/venv/bin/python -I tools/exe/extract.py artifacts/neuralhub-v0.8.0/NeuralHub_v0.8.0_setup.exe build/nh
#   -> build/nh/{neuralhub_host.dll, sr|rr|nr|fg.<target>.co}
podman pull docker.io/rocm/dev-ubuntu-24.04:7.2.4   # ROCm for the reference harness (/dev/kfd is usable as deck)
tools/hwtest/build_native.sh && tools/hwtest/build_plan.sh && tools/hwtest/build_probe.sh   # NR runner etc. -> build/hwtest/
```

Ghidra (for host-code questions; `tools/re/README.md`): Ghidra 12.1.4 unzipped to `build/ghidra/`, a JDK 21
image `localhost/ghidra:1` (`podman run --name t ubuntu:24.04 bash -c 'apt-get update && apt-get install -y
openjdk-21-jdk-headless' && podman commit t localhost/ghidra:1`), then the commands in `tools/re/README.md`.
A full decompile of `neuralhub_host.dll` takes about a minute; `tools/re/annotate.py` names the kernels in it.

### The private storage repository (DLLs and other inputs not in this repo)

**https://github.com/phly95/dlss5-private-storage** (private; the user's): every NVIDIA DLL below
(310.9.1.0, 310.9.16, the patched 310.8.0 NR, split into <100 MB parts), the Ghidra decompile of
`neuralhub_host.dll` (saves the Ghidra setup), and the rendered moving test sequence `sponza16`. Its README
has the restore commands into `build/` and the SHA-256 of every file. A fresh machine: clone it, run them,
then continue with the setup above. Put any new non-committable input there (split files over 90 MB) and
list it in its README. Never copy its contents into this repository.

### The user's DLLs (never commit them or anything extracted from them)

| what | where on this machine |
| --- | --- |
| `nvngx_dlss.dll` / `nvngx_dlssd.dll` 310.9.1.0 (SR / RR, the versions NeuralHub targets) | `~/.local/share/Steam/steamapps/common/Crimson Desert/bin64/` |
| `nvngx_dlssnr.dll` 310.8.0.0, community-patched (architecture check lifted) | downloaded from https://huggingface.co/Bandukids/DLSS-Runtimes (the user's pointer); weights verified identical to the reference's tested model |
| 310.9.16 `nvngx_dlss/dlssd/dlssg.dll` (no NR inside) | the user's zip; not needed so far |
| all of the above | also in the private storage repository (section above) |

NR model: `tools/hwtest/extract_nr_model.sh <nvngx_dlssnr.dll>` -> `build/model/dlssnr.bin` (the reference's
extractor with only the whole-file hash check relaxed; every weight entry is still checked), then
`build/venv/bin/python tools/hwtest/nr_gate.py --runner build/hwtest/nr_graph --plan-tool build/hwtest/nr_plan
--model build/model/dlssnr.bin --work build/hwtest/gate`.

## The cloud container (first sessions only; still useful for offline compiles)

- No GPU, no Windows. Ubuntu 24.04, 4 cores, 15 GB RAM.
- **Compile for the real card without it**: `bash tools/radv-offline/build.sh` (unpacks the prebuilt
  Mesa in `artifacts/mesa/`; `MESA_FROM_SOURCE=1` rebuilds Mesa main e2ab668 + libdrm b97cbde), then
  `tools/radv-offline/run.sh tools/radv-offline/pipestats [--isa DIR] file.spv ...`. RADV reports
  "AMD Radeon RX 9070 XT (RADV GFX1201)" through Mesa's amdgpu drm-shim; pipelines compile with ACO
  exactly as on the card; nothing executes.
- **CPU Vulkan**: the same Mesa build has lavapipe with `VK_KHR_cooperative_matrix` (f16×f16→f32,
  int8) but **no FP8** (`VK_EXT_shader_float8` is RADV GFX11.7+ only). The reference's FP8 network
  cannot run on it. E4M3 values are exact in f16, so an f16-widened shader variant could run there
  for functional tests; not written yet.
- Reference build: `cd reference-nr-vulkan && bash fetch_deps.sh && python3 linux/build/build_network.py rdna4`
  (pinned glslang 16.5.0, needed for byte-identical SPIR-V).

## Artifacts (private repository; binaries are intentional)

| path | what |
| --- | --- |
| `artifacts/neuralhub-v0.8.0/NeuralHub_v0.8.0_setup.exe` | the only Windows input (sha256 `aae53654…0b834`) |
| `artifacts/neuralhub-v0.8.0/gfx1201/{nr,sr,rr,fg}.gfx1201.co` | the RX 9070 XT code objects out of `neuralhub_host.dll`'s `.hip_fat` |
| `artifacts/neuralhub-v0.8.0/gfx1201/*.s.xz` | their `llvm-objdump --mcpu=gfx1201` disassembly |
| `artifacts/neuralhub-v0.8.0/gfx1201/kernels.json` | per-kernel metadata, kernarg reads, instruction profile (`tools/exe/kernels.py`) |
| `artifacts/neuralhub-v0.8.0/host_strings.txt.xz` | strings of `neuralhub_host.dll` (log messages, ini/env keys, source expressions of its error-check macro) |
| `artifacts/isa/amdgpu_isa_rdna4.xml.xz` | AMD's machine-readable RDNA4 ISA (GPUOpen): encodings + prose, no pseudocode, no WMMA lane map |
| `artifacts/isa/radv_nir_lower_cooperative_matrix.c` | Mesa source with the GFX12 WMMA lane layout comment |
| `artifacts/mesa/*.tar.xz` | prebuilt Mesa install (RADV, lavapipe, drm-shim) used by `build.sh` |
| `artifacts/reference-nr/` | ACO ISA + statistics of all 64 reference NR pipelines on GFX1201 |

`neuralhub.exe` (167 MB) and `neuralhub_host.dll` (98 MB) are not committed (GitHub's 100 MB limit);
`python3 -I tools/exe/extract.py artifacts/neuralhub-v0.8.0/NeuralHub_v0.8.0_setup.exe out/` recreates
them and every code object in seconds (needs `7z`, `pip install pefile msgpack`).

## Key findings (details in docs/inventory/KERNELS.md)

- NeuralHub's D3D12 backend runs the HIP gfx ISA through AMD's `amdxc64.dll` ElfHsa extension. Proton
  has no equivalent, so every kernel must become SPIR-V.
- SR weights: user's `nvngx_dlss.dll` 310.9.1.0, presets M/L only; RR: `nvngx_dlssd.dll` 310.9.1.0
  preset F; NR: `nvngx_dlssnr.dll` 310.8.0.0. All three are on this machine (table above); none may be committed.
- NeuralHub has built-in dumps usable on Windows: env `DLSSNR_DUMP_LAYERS`, `DLSSNR_DUMP_NOISE`, and
  `[Debug] DlssSrDumpNextFrame=1` (writes `sr_dump_<n>/stage_in_color.bin`, `stage_in_depth.bin`,
  `after_hidden0/1.bin`, `stage_output.bin`, `game_output.bin`; HIP backend only). Their file format is
  not yet known. RR has no dump.
- FP8 is E4M3, RNE, non-saturating (`v_cvt_pk_fp8_f32` without clamp); WMMA is E4M3×E4M3→F32.

## Session 2026-10-25 (cloud container, no GPU): where it stopped

- **verify v2 is frozen**: `1fea806` "verify: freeze v2 ..." fixes every defect listed under "Audit"
  below; `e4c9367` "verify: freeze v2 (correction ...)" fixes two bugs found in it before any port work
  (r2scan flagged its own pinned hash table; `rr_weights` needed `--dither`). `run_all.sh` V10 takes the
  newest "verify: freeze v2" commit, i.e. `e4c9367`. `tools/verify` must not change again; a real bug in it
  is a question for the user.
- `tools/verify/offline_check.sh` (GPU-less checks O1-O7): O1-O3 pass here; O4-O7 fail with "no shaders/
  directory" because no kernel exists yet. It does not pass until the first kernel is committed.
- `tools/gpu/collect_oracle.sh`: oracle dumps (srref M/L, rrref F, frames 0-1, seeded 256x144 -> 512x288)
  for the private repository. `--dry-run` checked here; the real run needs the card.
- `tools/re/slice.py` (def-use slices) and `tools/re/ssa.py` (straight-line SSA listing, `--expr` trees):
  reading aids for the ISA; text only, nothing evaluated.
- `k_sr_stage`: partial RE in `docs/sr/VULKAN.md`; no shader written. **Resume there**: the section
  "Seen but not yet decoded" and its softmax lead (unconfirmed), with
  `tools/re/fn.sh sr <symbol> > k.s; tools/re/ssa.py k.s --to 4660` (0x4660 is the first loop's start).
- The work was stopped at the user's request (goal cleared) with everything pushed.

**GPU-pending (run on the RX 9070 XT, from a clean checkout of this branch):**

    git fetch origin ccr-a7b81023-mipeai && git checkout ccr-a7b81023-mipeai
    # inputs: tools/verify/README.md "Restoring the inputs"; build/venv as in "Setting up from a fresh clone"
    python3 -I tools/exe/extract.py artifacts/neuralhub-v0.8.0/NeuralHub_v0.8.0_setup.exe build/nh
    bash tools/gpu/collect_oracle.sh
    # then follow the printed commands: copy build/oracle/<id>.tar.xz.part* and README into
    # phly95/dlss5-private-storage under oracle/, commit, push

With those dumps in the private repository, each kernel can be diffed offline (`tools/verify/diffdump.py`).
`run_all.sh` itself only makes sense once the port runner and package exist.

## Audit of the 2026-10-24 Vulkan attempt (`c229591`..`560ca36`)

A session asked to port SR/RR under a frozen verify script (`tools/verify/run_all.sh`, rules R1-R5,
checks V1-V10) produced `c229591` ("verify: freeze ...") and four port commits. The port commits are
reverted; they stay in history for reference. The freeze commit is left as is: changing anything in it
is the user's call (R4).

**Reverted** (`140482f`, `18bd87e`, `831401d`, `560ca36`): `shaders/`, `src/vk/`, `docs/sr/VULKAN.md`.
None of it matched the oracle and nothing compared it: there is no port-vs-`srref --dump` diff script,
and VULKAN.md's "verified / PASS" rows had no numbers. The GEMMs were scalar fp8->f16 loops (the
`coopmm.glsl` macros, copied from the reference, were included but unused); `sr_post` was an invented
bicubic + bilinear-history blend whose temporal weight read one `gates` byte (`/255`, clamped
0.05..0.95; 0.85 only as a fallback and 0 on reset) and never `factors`/`aniso`; `sr_downsample` plain bicubic; auto-exposure the
textbook 0.18 / exp(mean log luma); `sr_enc0`'s input-feature layout a guess; `rr_level.comp` a lightly
edited copy of `sr_stage.comp`. The runner had no barriers between dispatches, never filled `g_sr_lut`,
ignored `--mode` (RR cells ran the SR network), hard-coded the dec0 scales to 1.0 and did not enable
`VK_EXT_shader_float8`.

**Kept, with known defects** (`c229591`; all fixed in `1fea806` + `e4c9367`, "verify: freeze v2", above):
- V9: an absent `build/package/install.sh` is skipped and the run still passes; the NGX Vulkan path of
  `ngx_client` binds no images and reads nothing back; Proton outputs are never checked against the
  oracle; RR (feature 13) never runs; there is no missing / wrong DLL test.
- V5: the corrupted-weights and no-history runs come from the port's own flags; a 10 dB drop passes even
  if the degraded output would still pass V4 (`eval_verify.py:180`); SR M static only.
- V8: wall-clock of the whole process, the oracle's including podman start-up; under 50 ms exempt.
- V7: any log line with "RADV" passes, lavapipe is not excluded, the shaderstats table is not printed,
  and RADV's own "Spilled SGPRs: 0" lines probably trip the spill grep.
- V6: no SteamOS host check, no package scan, no extracted-weight hash scan; the ELF scan skips every
  path containing "nh".
- V1: falls back to `sponza16` (not held out, 16 frames) without Blender; ground truth is always off, so
  the 0.1 dB GT check never runs; the static generators draw the reset frame from 4-7, not 4-8.
- V3: preset L cells are skipped silently without L weights; the format cell sets `fmt_mv` 18 but no
  `mv_*.u32` is generated (srref exits, the cell cannot pass); the generators write 8-bit RGBA under
  colour code 5, which `docs/sr/HOST-RE.md` reads as RGBA32F; codes 16 / 18 are undocumented; srref's
  new auto-exposure arguments are guesses; rrref silently skips missing input files.
- V10: falls back to HEAD when no freeze commit is found; does not print the (empty) diff.

Usable from the freeze commit: the seeded camera / helmet paths and reset frames in the generators,
optional GT in `convert_seq.py`, srref's extra dumps (`g_col`, `g_dep`, `g_mv`, `g_mvhi`, preset L
buffers), the `/proc` maps / fd sampling loop, and the D3D12 path of `ngx_client`.

## How numbers are stated (2026-10-26)

- **Whole-pipeline PSNR and GPU time** come only from `tools/report/pipeline_report.py`
  (`build/venv/bin/python -I`). Nothing else in the tree may state a pipeline number.
- **Per-launch fidelity** for anything nontrivial comes only from `tools/report/launch_report.py`,
  measured against ORACLE inputs - upstream points forced with `DLSS5_AMD_INJECT_DIR` /
  `DLSS5_AMD_INJECT_MASK`, the kernel's own output taken with `DLSS5_AMD_DUMP_BUF` - and every
  claim states all three of: **peak = max|x| of the oracle tensor** (the tool prints it, and a
  per-launch tensor is fp8 up to +-448, so PSNR is meaningless without it), **PSNR**, and
  **Pearson r** with **relative RMSE**. Bit-exact % is recorded but is no longer the judgement.
  Definition, first baselines and the 2026-11-02 milestone are in
  `docs/notes/sr_chain_decoupling.md`, "Step C: the metric changed".
- Both scripts answer `--self-test`. A pasted number without its provenance trio - commit SHA,
  SHA-256 of the shader binaries, input directory - is not accepted.

## Do not retry

- ~~**Do not build a CPU emulator that executes NeuralHub's code objects.**~~ **Lifted 2026-10-26 by
  operator authorization.** Original rule text: "Do not build a CPU emulator that executes NeuralHub's
  code objects. That was attempted in the first session and was stopped by a safety check; the partial
  work was deleted. Do not resume it in another form." Origin: a first-session safety stop, partial work
  deleted. Lifted after an outside review flagged the ban as the project's main self-imposed limitation;
  the operator authorized it explicitly on 2026-10-26. The sanctioned line of work is
  `build/work/enc0emu.py`, a per-lane ISA emulator used as a *reading instrument* for `k_sr_enc0`
  (attempts 5-7 in `docs/notes/k_sr_enc0.md`); like every other probe it lives under `build/` and is
  never committed. It does not make a kernel verified: a decoded recipe still has to be implemented in
  `shaders/` and measured against the oracle before it counts.
- **Do not write approximations of NeuralHub kernels** (a generic TAA blend, bicubic resampling, textbook
  auto-exposure) or mark a kernel verified without a recorded per-buffer max error against `srref --dump`.
  That is what the reverted attempt did. **Known live exception**: the current `sr_post` (attempt 8,
  commit `f0c244f`) is part invented - a bilinear MV warp of `in_mv_nondil`, a bilinear
  `sample_history` instead of the decoded Catmull-Rom, and an invented history-discard test - on top
  of parts that *are* decoded (the anisotropic 9-tap from `aniso`, the gate/factor blend from
  `gates`/`factors`). The invented half is in the tree because the chain needs a post that runs and
  because that path measures best while `k_sr_dec0` is still wrong. It is a placeholder, not a port,
  is captioned as such in `docs/notes/k_sr_post.md` and `docs/PORT-STATUS.md`, and must be replaced
  once `k_sr_post` is decoded (see `docs/notes/sr_chain_decoupling.md`). Do not extend its invented
  logic to any other kernel, and do not cite the invented parts as decoded.
- **Do not re-land gain-before-RMSNorm** (either norm), `nr_swin_act` in an SR stage, or the attention
  topology `ctx = P @ (s_norm @ Wv)`. Each was refuted by Review R1 (`docs/notes/k_sr_stage.md`). Do
  not fit attention-normalization families on QK = 0 probes: K = 0 there makes V = 0. Do not build a
  ceiling by injecting oracle data downstream of the sub-block under test.
- (Resolved 2026-10-24) The user chose to port NeuralHub's SR/RR kernels to Vulkan; that is no longer blocked.

## Open questions for the user

1. (Answered 2026-10-24: port NeuralHub's kernels.)
2. (Answered: this machine is the RX 9070 XT. Windows dumps are no longer needed: NeuralHub's kernels run under ROCm here.)
3. NR at 310.9.16 (the 49 dB figure at 1080p) needs `nvngx_dlssnr.dll` 310.9.16.
4. (Answered 2026-10-25: re-freeze as "verify: freeze v2"; done in `1fea806` + `e4c9367`.)
5. Preset M's weight checksum mismatch (Known loose ends) is still unexplained; `collect_oracle.sh`
   requires only L's checksum. Is an M oracle from these weights acceptable as M's reference?
6. `k_sr_stage`'s second normalisation (`docs/sr/VULKAN.md`, "open") must be decoded exactly before a
   shader is written; until then there is no kernel to commit.

Open items are tracked in `docs/OPEN.md`; do not wait on answers before continuing independent kernel work.

## SR: how to continue (2026-10-24)

1. `python3 -I tools/exe/extract.py artifacts/neuralhub-v0.8.0/NeuralHub_v0.8.0_setup.exe build/nh` (code objects, host DLL).
2. Weights: `build/venv/bin/python -I tools/sr/sr_weights.py <Crimson Desert>/bin64/nvngx_dlss.dll --out build/re/weights`,
   then `tools/sr/sr_pack.py build/re/weights {M,L} build/re/packed/{M,L}`.
3. Reference: see `docs/sr/HOST-RE.md` (ROCm 7.2.4 in podman; `srref ... --dump` writes every stage tensor and the
   dec0 outputs). Every Vulkan kernel is to be checked against it, stage by stage. The diff script that does
   this (port dump vs `srref --dump`, per-buffer max error) does not exist yet; write it before the first kernel.
4. Port order: `k_sr_stage<64,2,0,96,2,false>` (encoder), `k_sr_tube_split<128,4,false>`, the decoder stages,
   `k_sr_enc0`, `k_sr_dec0`, `k_sr_post<false,12,16,false,false>`, `k_sr_downsample`, exposure; then preset L's
   `k_sr_enc0L_in` / `k_sr_l0`. On this card every stage is "fp8-safe", so only the `false` instantiations run.
   What is known of `k_sr_stage` so far: activations are `[C/32][H][W][32]` fp8; a workgroup (64 threads)
   covers an 8x8 pixel tile at grid*8 - shift, mirror-padded at the borders; RMSNorm = f16 sum of squares
   (fixed packed-add tree + a lane^16 exchange) + 2^-14, rsqrt from `g_sr_lut`, times the sqrt(C)-scaled gain,
   to fp8; GEMMs as pairs of K=32 WMMA chains whose f32 results are added through f16 (`v_fma_mixlo_f16`).
   No `exp` instruction anywhere, but see `docs/sr/VULKAN.md`: a bit-built exp2 and the `1/h` table suggest a
   softmax after all (unconfirmed).
5. Faster route to games, if wanted: NeuralHub's kernels run under ROCm, so a Vulkan layer could drive them
   through HIP external-memory interop. This breaks the goal's "no HIP at run time" and needs ROCm
   libraries; ask the user before going this way.

## Test content

- Static: the NR verification's 4K Tomb Raider frame (`reference-nr-vulkan/docs/ngx-verification/310.8.0/
  single-frame-inputs/`), jittered by `tools/sr/gen_frames.py` / `tools/rr/gen_rr_frames.py` (RR: + noise).
- Moving, path-traced: `tools/scenes/` (README there): Blender 4.2.3 Cycles renders Khronos's Sponza with the
  DamagedHelmet flying through it, a panning dolly camera, real low-sample noise, exact motion vectors
  (checked by `check_mv.py`), every RR guide, and denoised ground truth. Cycles runs on the 9070 XT through
  HIP inside a ROCm container (~49 s/frame, mostly the 512-spp ground truth). `sponza16` (16 frames,
  960x540 -> 1920x1080) is in the storage repo. Results: RR 33.2 dB (frame 0) -> 36.4 dB (frame 15)
  against ~22 dB for the noisy input; SR 33.0-33.4 dB vs bicubic 30.1-32.0 dB (its "clean" input still has
  256-spp noise and fireflies while the ground truth is denoised, so this caps SR's score).

## Next steps, in order

1. SR in Vulkan (section above): one kernel per commit, each compared with `srref --dump` output for the
   same inputs (bit-exact where the ISA's rounding points are mirrored; otherwise report the difference),
   compiled with zero spills (`RADV_DEBUG=shaderstats`, as in `docs/hardware/RX9070XT-STEAMOS.md`).
   Shaders follow the reference NR port's conventions (`reference-nr-vulkan/CLAUDE.md`: wave32 pinned,
   FP8 only where the original uses it, 16-byte aligned LDS coopmat loads). Then a Vulkan host that runs
   the whole SR frame and matches srref end to end (39 dB test above, and per-frame diff vs srref).
2. RR: the ROCm reference exists (`docs/rr/HOST-RE.md`); port its kernels to Vulkan the same way as SR's,
   checked against `rrref --dump` (it writes every engine buffer as `E<index>.bin`). The `rrk::k_level`
   family (16 units, 18 instantiations) is the bulk; `k_rr_gather` (84 KB of ISA) is a format converter.
3. Integration: the reference's Vulkan layer + OptiScaler route (`reference-nr-vulkan/linux/`), SR/RR
   behind NGX/FSR inputs; installer modelled on `reference-nr-vulkan/linux/package/install.sh`.
4. Performance on this card; NR's in-game runtime path vs NVIDIA (the gate so far used the offline runner).

## Known loose ends

- Preset M's raw weights do not reproduce NeuralHub's M checksum (L does); M still packs to the expected
  sizes and runs well under srref. Cause unknown (`docs/sr/HOST-RE.md`).
- srref runs NeuralHub's `k_sr_gather` with `--gather` (only the RGBA16F / RG16F / R32F inputs, code 0:
  bit-identical to the direct upload); the other formats, `k_sr_gather_mv` (output-resolution MVs) and the
  auto-exposure path are still unexercised. The Vulkan port needs them all.
- Coverage still missing in the references: RR's transparency / alpha / virtual-hit / SSS / mask inputs,
  UNORM normals, linear depth; SR preset L on the moving scene; a mid-sequence reset; longer sequences.
- `sponza16` frame 6 has one firefly (8688 in a pixel): fine for scoring, but clamp
  (`cycles.sample_clamp_direct`) if a future render needs clean warping checks.
- The decompiler shows raw float stores as `(int)` casts; check every scalar's type against the kernel ISA
  (this hid a bug once: dec0's first scalar).
