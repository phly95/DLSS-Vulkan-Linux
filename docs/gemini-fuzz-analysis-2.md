# Gemini analysis #2 — in-game fuzziness after the S8 perf fix (raw, unverified)

Verbatim copy of the second Gemini investigation (received 2026-11-01, after `80aa2f7` / the S8
FFX-seam + flash-probe fixes landed and the user reported "still objects far worse, unevenly, XeSS
sharp in the same scenes"). Verification status: NOT YET CHECKED — see `docs/gemini-claims-verdict.md`
for the method used on Gemini's first claim set (one row per claim, TRUE / PARTLY / FALSE / STALE
with file:line evidence read from the repo, not from Gemini's citations).

---

After the S8 FFX-seam and flash-probe fixes (80aa2f7 / S8-PERF-FIX), the in-game fuzziness comes down to a combination of temporal reprojection shortcuts taken during the 2026-10-28 perf pass, FFX-seam (src/ffx/ffx_front.cpp) plumbing bugs that degrade or mis-scale inputs, and missing < 2.0x upscale ratio kernel geometry in sr_post.comp.

## Summary of Root-Cause Theories

| Priority | Subsystem | Root Cause (file:line) | Visual Symptom in Game |
|---|---|---|---|
| 1 (Highest) | Temporal reprojection & disocclusion (sr_post.comp, sr_dec0.comp, sr_enc0.comp) | Deleted factors (3-candidate MV weights) and mv_ero (eroded MV) during the Oct 28 perf pass; sr_post.comp hardcodes nearest-neighbor in_mv_nondil and omits the 2nd Catmull-Rom history blend | Fuzzy halos, ghosting, and smeared silhouettes around moving characters, foliage, hair, and depth edges |
| 2 (High) | FFX seam quality ratio & preset (src/ffx/ffx_front.cpp) | ffxQuery casts kQualityRatio to (int) (1.5f -> 1, 1.7f -> 1), and ffxCreateContext hardcodes preset = 'M' even at Ultra Performance (3.0x) | Wrong render resolution or Preset M forced at 3.0x where DLSS 5 requires Preset L |
| 3 (High) | FFX seam MV & exposure plumbing (src/ffx/ffx_front.cpp, src/pe/nvngx.cpp) | Missing mv_scale == 0 guard, ignored FFX_UPSCALE_ENABLE_DISPLAY_RESOLUTION_MOTION_VECTORS (fp.mvhi stuck at 0), ignored d->exposure resource, and near-zero gameplay MV dumps (S6) | Uncompensated motion smear during pans, wrong MV sampling stride when display-res MVs are active, or exposure drift |
| 4 (Medium-High) | < 2.0x upscale ratio post-filter (sr_post.comp, srref.cpp) | NeuralHub switches to k_sr_post<isL, 15, 27, ...> when W15 / W < 3.0 (< 2.0x upscale), while sr_post.comp only implements <12, 16> and overflows its 20x20 LDS tile below 1.5x | Localized blurriness on Quality (1.5x) / Balanced (1.7x) and periodic 32x32 tile edge corruption on UltraQuality / DLAA |
| 5 (Medium) | U-Net / Swin precision & dither (sr_swin.comp, sr_enc0*.comp, sr_dec0.comp) | Fused sr_swin.comp drops R9's explicit rh() f16 rounding chains (sr_M_seq sits at 41.91 dB vs oracle, with stage9 mode-2 X1 residue dominating), and dq() stochastic FP8 dithering was removed | Slight softness in fine stochastic textures (grass, pores, stone) and temporal drift over multi-frame sequences |

## Detailed Theories and How to Fix Each

### 1. Missing 3-Candidate Motion Vector Selection and Second-History Blend in sr_post.comp

During the 2026-10-28 performance pass, three pieces of NeuralHub's temporal reprojection logic (documented in docs/notes/k_sr_post.md Review R5 and implemented in tools/sr/sr_post_model.py:130-185) were stripped out because the synthetic static oracle test frames had spatially constant motion vectors across the entire image (mv = (-0.2861328125, -0.075) everywhere), making mv_nondil == prev_mvec1 == mv_ero:

 * Deleted eroded MV (mv_ero): shaders/sr/sr_enc0.comp:71, 524 comments out the write to binding 7 (out_mv_ero) as a "dead store", even though sr_enc0.comp:307-318 still computes the 3x3 max-depth eroded coordinate (ex, ey).
 * Deleted candidate weights (factors): shaders/sr/sr_dec0.comp:32-34, 59, 310 removes the 3-channel factors output (fac0 = c16[51] + b16[51], fac1 = c16[52] + b16[52], fac2 = c16[53] + b16[53]) with the comment "no shader reads them".
 * Collapsed MV lookup & history fetch in sr_post.comp: shaders/sr/sr_post.comp:66-67, 467-475 leaves in_prev_mvec1 (binding 2), in_mv_ero (binding 3), and in_factors (binding 4) unreferenced. Instead of accumulating the 3 candidate scores over the spatial footprint, ranking [mv_nondil (bilinear), prev_mvec1 (dilated frontmost), mv_ero (eroded background)], and blending a second 5-tap Catmull-Rom history fetch when the runner-up candidate's confidence share clamp(1.1 * s - 0.05, 0.0, 1.0) > 1/255, sr_post.comp unconditionally uses a single nearest-neighbor load in_mv_nondil[my * W + mx].
 * Why this causes fuzzy regions in-game: At depth discontinuities (character silhouettes, foliage, hair, weapon edges, disocclusions), the center pixel's raw mv_nondil either belongs to the wrong depth layer or lacks subpixel bilinear filtering (mvntex), while prev_mvec1 and mv_ero hold the foreground and background velocities that factors learned to select and blend between.
 * How to fix:
   * Restore the out_mv_ero write in shaders/sr/sr_enc0.comp:524 (and verify sr_enc0L_in.comp:579, which still writes out_mv_ero).
   * Restore the out_factors computation (c16[51..53] + b16[51..53]) and buffer store in shaders/sr/sr_dec0.comp:310.
   * Port the R5 candidate selection and second Catmull-Rom history blend from tools/sr/sr_post_model.py:130-185 (and tools/sr/sr_postL_model.py) into shaders/sr/sr_post.comp:467-495, and switch in_mv_nondil from nearest-neighbor array indexing to bilinear sampling (mvntex).

### 2. FFX Seam Resolution Query Truncation and Hardcoded Preset M (src/ffx/ffx_front.cpp)

In src/ffx/ffx_front.cpp, two bugs directly affect upscale ratio and preset selection when Crimson Desert or OptiScaler queries/creates the FFX context:

 * Integer cast on quality ratio: In src/ffx/ffx_front.cpp:289-290 (FFX_API_QUERY_DESC_TYPE_UPSCALE_GETRENDERRESOLUTIONFROMQUALITYMODE and GETUPSCALEQUALITY), kQualityRatio ({1.0f, 1.5f, 1.7f, 2.0f, 3.0f}) is cast to (int) before dividing:

       *d->pOutRenderWidth  = d->displayWidth  / (int)kQualityRatio[d->qualityMode];
       *d->pOutRenderHeight = d->displayHeight / (int)kQualityRatio[d->qualityMode];

   Casting 1.5f (Quality) or 1.7f (Balanced) to (int) yields 1, returning full display resolution instead of 1.5x or 1.7x.

 * Hardcoded Preset M: In src/ffx/ffx_front.cpp:378, ffxCreateContext hardcodes c->fs.preset = 'M'; regardless of the upscale ratio or maxRenderSize vs maxUpscaleSize. At 3.0x Ultra Performance (1280x720 -> 3840x2160), DLSS 5 is trained for Preset L (sr_enc0L + 6-tap sr_post + sr_L weights), whereas running Preset M at 3.0x produces softer reconstruction.

 * How to fix:
   * In src/ffx/ffx_front.cpp:289-290 (and GETJITTEROFFSET), divide by kQualityRatio[d->qualityMode] in floating point: uint32_t(float(d->displayWidth) / kQualityRatio[d->qualityMode] + 0.5f).
   * In src/ffx/ffx_front.cpp:378 (and dynamically in ffxDispatch if rw * 5 <= ow * 2), select Preset 'L' when the upscale ratio is >= 2.5x (or allow an env override DLSS5_AMD_PRESET=L).

### 3. FFX / NGX Motion Vector Scaling, Display-Resolution MVs (mvhi), and Exposure Plumbing

Several host-to-bridge mismatches in src/ffx/ffx_front.cpp and src/pe/nvngx.cpp corrupt or drop temporal inputs:

 * Missing zero/NaN guard on motionVectorScale in FFX: src/pe/nvngx.cpp:482-483 guards `if (!std::isfinite(fp.mv_scale_x) || fp.mv_scale_x == 0.0f) fp.mv_scale_x = 1.0f;`, but src/ffx/ffx_front.cpp:512-513 assigns d->motionVectorScale.x and .y directly. If the caller passes 0.0f, all motion vectors are zeroed out.
 * fp.mvhi (display-resolution MVs) is never set on either bridge:
   * In src/ffx/ffx_front.cpp:365-380, 493-520, `c->flags & FFX_UPSCALE_ENABLE_DISPLAY_RESOLUTION_MOTION_VECTORS` (bit 1 << 2) is never checked, and `bi_mv.w == c->fs.ow` is never translated into fp.mvhi = 1. Worse, if bi_mv.w == ow, copy_row_texels sees img.w > W, falls back to a tight ow * oh copy into in_mv_, and then Pipeline::dispatch runs sr_gather (which indexes in_mv with render stride W) instead of sr_gather_mv (which indexes in_mv at output resolution oW x oH)!
   * Similarly, in src/pe/nvngx.cpp:1371, ins[k].w > chain_w fails the frame instead of allowing k == 1 (MV) to match out.w x out.h with fp.mvhi = 1.
 * In-game zero MVs (S6 finding): In tools/scenario/REPORT.md (S6), the dumped gameplay MV buffers (dlss5-amd-dump_mv_f10..f1800.bin) had 99.9% exact zeros (max|v| = 8.25e-5) during camera pans because Crimson Desert binds a blank/UI MV target on one of the dispatches or omits camera velocity from that pass. When MVs are zero during motion, sr_post reprojects static history at [x - jitter], blending unshifted previous frames into the current frame (classic TAA/DLSS motion blur/fuzziness).
 * Exposure resource ignored on FFX: src/ffx/ffx_front.cpp:516-517 hardcodes fp.auto_exposure = 1 and completely ignores d->exposure and d->preExposure even when FFX_UPSCALE_ENABLE_AUTO_EXPOSURE is not set in c->flags.
 * How to fix:
   * Add the std::isfinite / != 0.0f guard to fp.mv_scale_x and fp.mv_scale_y in src/ffx/ffx_front.cpp:512-513.
   * Set `fp.mvhi = ((c->flags & FFX_UPSCALE_ENABLE_DISPLAY_RESOLUTION_MOTION_VECTORS) != 0 || (bi_mv.w == c->fs.ow && bi_mv.h == c->fs.oh)) ? 1 : 0;` in src/ffx/ffx_front.cpp, and update copy_row_texels in src/pe/ngx_bridge_shared.h:355 so MV uploads use (mvhi ? oW : W) and (mvhi ? oH : H) as the expected extent.
   * If d->exposure.resource is non-null and !(c->flags & FFX_UPSCALE_ENABLE_AUTO_EXPOSURE), stage the 1x1 exposure texel (just like nvngx.cpp:1443-1469) instead of forcing fp.auto_exposure = 1.
   * Log bi_mv resource pointer, format, dimensions, d->motionVectorScale, and non-zero sample stats in ffx_front.cpp on the first few non-menu frames to verify whether OptiScaler's fsr31 path receives populated camera+object MVs in Crimson Desert.

### 4. Missing < 2.0x Upscale Kernel Specialization (<15, 27> vs <12, 16>) and LDS Tile Clamp in sr_post.comp

Look at how the NeuralHub oracle (tools/sr/srref/srref.cpp:211, 520-521) selects the k_sr_post kernel:

    const bool post_3_2 = (float(W15) / float(W) < 3.0f) || (float(H15) / float(H) < 3.0f);
    std::snprintf(sym, sizeof sym, "_ZN2sr2io9k_sr_postILb%dELi%sELb%dELb%dEEEvNS0_8PostArgsE",
                  int(presetL), post_3_2 ? "15ELi27" : "12ELi16", int(ldr), int(mvhi));

 * Whenever W15 / W < 3.0f—which is true for every upscale ratio below 2.0x, including Quality (1.5x, W15/W = 2.25), Balanced (1.7x, W15/W = 2.55), Ultra Quality (1.3x), and DLAA (1.0x, W15/W = 1.5)—NeuralHub dispatches k_sr_post<isL, 15, 27, ldr, mvhi> instead of <isL, 12, 16, ldr, mvhi>.
 * In shaders/sr/sr_post.comp:127-134, 334-335, the LDS staging tile is hardcoded to ST_C = 20, ST_R = 20 (ncol = min(X0hi - X0lo + kmax + 2, ST_C)). For a 32x32 workgroup in 1.5 * oW space, X0hi - X0lo = ceil(31 * W / W15):
   * At 2.0x Performance (W/W15 = 1/3), X0hi - X0lo + kmax + 2 is 14 (M) or 15 (L) <= 20.
   * At 1.5x Quality (W/W15 = 4/9), it reaches 17 (M) or 18 (L), and sr_post.comp still uses <12, 16> semantics instead of <15, 27>.
   * Below ~1.4x (Ultra Quality / DLAA), X0hi - X0lo + kmax + 2 exceeds 20 (up to 25 at DLAA), clamping ncol and nrow to 20 and causing threads on the right/bottom of every 32x32 workgroup to read wrapped/corrupted LDS rows in sA and sB!
 * How to fix:
   * Widen ST_C and ST_R in shaders/sr/sr_post.comp:127-128 from 20 to 27 (matching NeuralHub's <15, 27> tile constant) so < 1.5x ratios never clamp ncol/nrow and read out-of-bounds LDS.
   * Check k_sr_post<..., 15, 27, ...> in the NeuralHub .co disassembly (tools/re/fn.sh) to verify whether <15, 27> differs only in the LDS tile dimensions (15, 27) or also in filter tap scaling when W15 / W < 3.0.

### 5. Fused sr_swin.comp Rounding Drift & Removed Stochastic FP8 Dither (dq())

 * In tools/report/quality_guard.py:53-58, sr_M_static scores 57.13 dB against the oracle, whereas sr_M_seq drops to 41.91 dB (sr_L_seq 42.12 dB). As documented in docs/notes/chain-leverage.md:453-469 and docs/notes/k_sr_stage.md:1471-1495, two factors drive this multi-frame gap:
   * The shipped shaders/sr/sr_swin.comp (which replaced sr_stage.comp + sr_tube_split.comp in pipeline.cpp:1767) uses fused coopmat accumulators without R9's per-group rh() f16 rounding chains (k_sr_stage.md:1447-1464), and stage9 (mode-2 decoder launch 12) carries the post-attention X1 f16 residual into dec0's recurrent hidden state (hist0..7), compounding across frames.
   * Both shaders/sr/sr_enc0.comp:456 and shaders/sr/sr_enc0L_in.comp:434 dropped the stochastic blue-noise dither table dq(v, d) before FP8 quantization (to_fp8(v) direct). In dark or low-contrast gradients, omitting dq() causes FP8 E4M3 quantization banding/mushiness that the temporal loop accumulates.
 * How to fix:
   * Re-enable dq(v, d) (loading the 4-byte dither row per pixel) in sr_enc0.comp and sr_enc0L_in.comp—it costs only a single 16-entry LUT read per thread and prevents FP8 quantization banding in low-contrast regions.
   * Add an environment toggle (e.g. DLSS5_AMD_EXACT_STAGE=1) in src/vk/pipeline.cpp:1767 to bind the R9-exact sr_stage.spv + sr_tube_split.spv shaders instead of sr_swin.spv for A/B quality comparison.

## Modifications Needed in the Harnesses to Confirm Fixes Before Launching the Game

Right now, a fix to any of the issues above will not be caught or confirmed by tools/optiharness/ or tools/scenario/gate_cell.sh, because:

 * tools/optiharness/content_gate.py only checks non-emptiness (std > 0.01, nan == 0, nonzero_pct > 50%, corr(out, color) > 0.5). A heavily blurred or ghosted image passes content_gate.py at 100%.
 * tools/optiharness/opti_client.cpp drives the NGX path (NvLL_vk_Init_Ext), not dlss5-amd-ffx.dll (ffxCreateContext / ffxDispatch), and generates a synthetic checkerboard with a constant global MV (mv_vx = 2.0f, mv_vy = 1.0f) where foreground/background disocclusions do not exist.
 * tools/report/quality_guard.py sets B1 = 41.0 dB (tools/report/quality_guard.py:99), which is 9 dB below V4's 50.0 dB per-frame oracle bar (tools/verify/eval_verify.py:30), and only runs 2.0x cells (seq 960x540 -> 1920x1080 and static 1920x1080 -> 3840x2160).

| Harness File | Modification to Make | What It Confirms Headlessly |
|---|---|---|
| tools/optiharness/opti_client.cpp | Add a --ffx <path/to/dlss5-amd-ffx.dll> mode that loads ffxCreateContext, ffxQuery, and ffxDispatch directly over a D3D12 device, plus a --replay <dir> flag that loads real multi-frame .f32/.bin dumps (or the Blender seq cell) instead of the synthetic constant-MV checkerboard | Tests the exact dlss5-amd-ffx.dll binary, ffxQuery quality-mode math, motionVectorScale, mvhi, and exposure handling under Proton without launching Crimson Desert |
| tools/optiharness/opti_client.cpp (synthetic scene) | When not using --replay, give the moving foreground square a different velocity and depth (depth = 0.2, mv = (+4, -2)) from the background checkerboard (depth = 0.9, mv = (-1, +1)) | Creates true depth/MV disocclusions so mv_nondil != prev_mvec1 != mv_ero, immediately exposing whether sr_post candidate selection (factors) fixes edge fuzziness |
| tools/optiharness/content_gate.py | Add a --ref <oracle_or_gt_dir> mode that computes full-frame PSNR(out, ref) (using eval_verify.py's tone-mapped PSNR), edge-band PSNR (on pixels within 3 px of a depth/MV gradient), and a Laplacian variance sharpness ratio Var(Lap(out)) / Var(Lap(ref)) >= 0.90 | Turns optiharness from a "not black/NaN" smoke test into an actual sharpness and disocclusion-halo gate |
| tools/report/quality_guard.py | Add q15 (1280x720 -> 1920x1080, 1.5x Quality) and mvhi cells to CELLS (tools/report/quality_guard.py:86-89), and report an explicit disocclusion-mask PSNR (pixels where ||mv_nondil - mv_dil|| > 0.5 px in the seq cell) alongside B1 | Isolates the exact PSNR gain on moving object boundaries when factors + mv_ero + 2nd history fetch are restored, and catches < 2.0x (post_3_2) regressions |
| src/ffx/ffx_front.cpp + tools/scenario/gate_cell.sh | Add DLSS5_AMD_DUMP_DIR support to src/ffx/ffx_front.cpp (matching nvngx.cpp:940-965 so it dumps color, mv, depth, out, and params.json in vk_runner format on frames 10, 60, 300), and add a --check-mv-nonzero check in gate_cell.sh | Lets you capture 12 consecutive live frames from Crimson Desert once, then replay them infinitely through vk_runner and srref on the host to compare PSNR and inspect MV validity offline |

## Recommended Verification Workflow Before Launching the Game

 * Verify sr_post + sr_dec0 + sr_enc0 against the oracle on seq and q15: Restore mv_ero, factors, and the 2-candidate history blend in sr_post.comp, then run build/venv/bin/python -I tools/report/quality_guard.py --no-cache and build/venv/bin/python -I tools/report/psnr3x_gate.py. Look for sr_M_seq / sr_L_seq B1 (PSNR(port, ref)) jumping from ~41.9 dB toward the >= 50 dB V4 bar.
 * Verify the FFX DLL headlessly via optiharness: Fix the (int)kQualityRatio cast, mv_scale zero guard, Preset L selection, and mvhi handling in src/ffx/ffx_front.cpp, rebuild with bash src/ffx/build.sh, and run tools/optiharness/run_harness.sh with --ffx against the Blender seq inputs and the updated content_gate.py --ref.
 * Confirm in-game MV population on one short trace: With DLSS5_AMD_DUMP_DIR wired into ffx_front.cpp, check that the dumped mv buffer during a camera pan has non-zero velocities across the frame; if the game's FSR3.1 dispatch still passes zero camera MVs, toggle OptiScaler's MotionVectors / reactive mask settings in tools/scenario/inis/ffx1.ini before testing visual quality.