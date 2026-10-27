# Notes: k_sr_post

## Launch Details
- Launch 14 in `sr_M` frame 0: `_ZN2sr2io9k_sr_postILb0ELi12ELi16ELb0ELb0EEEvNS0_8PostArgsE`
- Grid: [48, 14, 1], Block: [256, 1, 1]
- Inputs: `g_col`, `expo`, `gates`, `factors`, `aniso`, `ihist0`, `prev_mvec1`, `mv_ero`, `mv_nondil`
- Outputs: `dhist` (u32, 82944), `full15` (u32, 331776), `ihist1` (f16vec4, 331776)

## Attempts and Hypotheses

### Attempt 2
- **Hypothesis**: Decoded coordinate transform and tonemapping formats from GFX1201 ISA:
  - Input sample coordinate: `floor((float(x15) + 0.5) * scaleX + jitterX - 0.5)` with clamp to `[0, W-1]`, using `scaleX`, `scaleY`, `jitterX`, `jitterY` unpacked from push constants.
  - `ihist1`: tonemapped history `0.36 * log10(max(1.0 + 100.0 * c * e, 0.0001))` in float16 with alpha = 0.0.
  - `dhist`: 2x2 box average at half resolution (384x216), tonemapped and packed to 10:10:10 format `ur | (ug << 10) | (ub << 20)`.
  - `full15`: R11G11B10 format with rounded float scaling.
- **Result**:
  - `ihist1 [f16]`: 1327104 elements, bit-exact 373589 (28.1507%), max |err| 0.6423, max ulp 7268, first mismatch `{'index': 4, 'port': 0.0946044921875, 'oracle': 0.0960693359375, 'port_bits': 11790, 'oracle_bits': 11814}` (Raw byte-exact: 1,244,196 / 2,654,208 bytes = 46.88% bit-exact).
  - `dhist [u8]`: 331776 elements, bit-exact 84930 (25.5986%), max |err| 255.
  - `full15 [u8]`: 1327104 elements, bit-exact 240463 (18.1194%), first mismatch index 4 `port: 79.0, oracle: 80.0` (1 LSB off).

### Attempt 3
- **Hypothesis**: The workgroup block is 256 threads covering a 16x32 pixel tile (grid [48, 14, 1], 14 * 32 = 448 >= 432). Each thread processes TWO pixels in Y (`ly = (tid >> 4) * 2`, pixel 0 at `y0 = gy + ly`, pixel 1 at `y1 = y0 + 1`). Previously only pixel 0 was processed up to row 223, leaving half the screen unwritten. Fix processes both pixels and writes all 384x216 cells of `dhist`.
- **Result**: Complete coverage of 768x432 output and 384x216 dhist output.
- **Pipestats**: 108 SGPRs, 24 VGPRs, 0 spilled SGPRs, 0 spilled VGPRs, 0 scratch size.

### Attempt 4
- **Hypothesis**: In GFX1201 ISA (`tools/re/fn.sh sr _ZN2sr2io9k_sr_postILb0ELi12ELi16ELb0ELb0EEEvNS0_8PostArgsE` 0x7738-0x77A8), each `g_col` input sample is individually tonemapped into logarithmic space ($0.36 \times \log_{10}(1.0 + 100.0 \times c \times \text{expo})$) before spatial reconstruction filtering. Doing bilinear interpolation in raw linear HDR space caused specular fireflies (e.g. 3,460.0) to smear into neighboring pixels (1,730+ vs ref max 13.75). Clamping to FP16 finite range (32752.0) and tonemapping each corner pixel before bilinear blending compresses specular peaks and prevents HDR smearing.
- **Result**:
  - `full15 [u8]`: bit-exact increased from 18.12% to 32.49% (431,169 / 1,327,104).
  - `dhist [u8]`: bit-exact increased from 25.60% to 40.63% (134,807 / 331,776). First mismatch index 1: port 30456887, oracle 30456888 (1 LSB off).
  - `ihist1 [u8]`: bit-exact increased from 46.88% to 56.52% (1,500,149 / 2,654,208).

### Attempt 5
- **Hypothesis**: History reprojection on moving frames suffered from four compounding errors:
  1. Jitter offset delta between frames: the reprojected history position must undo camera subpixel jitter `(jitter_prev - jitter) * (W15 / W)`. In `runner.cpp`, `jitter_prev` was omitted from push constants. Passing `mvoffx = (jpx - jx) * (W15 / W)` and `mvoffy = (jpy - jy) * (H15 / H)` in push constants (offset 248) eliminates subpixel jitter misalignment.
  2. Motion vectors were sampled with nearest-neighbor lookup, causing warping discontinuities. Bilinear interpolation of `in_mv_nondil` across continuous subpixel coordinates produces smooth warping.
  3. Disocclusion handling: previous shader clamped temporal weight to minimum 0.25, forcing 25% ghosting on disoccluded pixels. Lowering the clamp floor to 0.0 allows newly visible surfaces to discard invalid history.
  4. Local color bounding box clamping (`clamp(t_hist, cmin, cmax)`): restricts history samples to the local neighborhood color gamut, preventing ghosting artifacts.
- **Result**:
  - Minimum PSNR across `sr_M_seq` moving frames jumped by **+8.4 dB** (from 32.46 dB to 40.87 dB, with mean PSNR 42.40 dB).
  - Frames 3, 5, 6, and 9 achieved $|PSNR(\text{port}, GT) - PSNR(\text{ref}, GT)| \le 0.013 - 0.096\text{ dB}$, satisfying the referee's $\le 0.100\text{ dB}$ ground-truth delta threshold!
  - `sr_L_seq` achieves 42.67 dB mean PSNR (min 41.03 dB).
  - All 27 pipelines pass offline verification (O1..O7 clean).
  - Mesa `shaderstats` confirmed: 48 VGPRs, 108 SGPRs, 0 spilled SGPRs, 0 spilled VGPRs, 0 scratch size.

### Attempt 6
- **Hypothesis**: GFX1201 ISA analysis at 0xF8690-0xF8720 revealed history sampling uses 5 `image_sample_lz` instructions in a cross footprint (center, left, right, top, bottom) for Catmull-Rom bicubic reconstruction rather than 1-tap bilinear filtering. In addition, lower-bounding the spatial covariance diagonal ($A_{xx} \ge 3.0, A_{yy} \ge 3.0$) prevents degenerate box-blurring when intermediate MLPs output small eigenvalues.
- **Result**:
  - Launch 14 frame 0 diffdump:
    - `ihist1 [f16]`: bit-exact increased from 373,589 (28.15%) to 425,425 (32.06%), max error 0.5258, max ULP 6044, pixel 0 only 1 ULP off (`port_bits 11791, oracle_bits 11790`).
    - `full15 [u32]`: bit-exact jumped 4x from 8,865 (2.67%) to 33,880 (10.21%), pixel 1 only 1 LSB off (`port_bits 96632913, oracle_bits 96632912`).
    - `dhist [u32]`: bit-exact jumped 4x from 3,065 (3.70%) to 11,250 (13.56%).
  - Launch 14 frame 1 diffdump (moving frame with reprojected history):
    - `ihist0 [f16]`: 413,310 (31.14%) bit-exact, max error 0.5392.
    - `full15 [u32]`: 18,666 (5.63%) bit-exact.
    - `dhist [u32]`: 6,766 (8.16%) bit-exact.
  - Mesa `shaderstats`: 96 VGPRs, 108 SGPRs, 0 spilled SGPRs, 0 spilled VGPRs, 0 scratch size, 16 subgroups/SIMD.

### Attempt 7 (2026-10-26): attempt 6 was an end-to-end regression; reverted to attempt 5's shader
- **Method**: `DLSS5_AMD_INJECT_DIR` chain leverage (`docs/notes/chain-leverage.md`) showed `k_sr_post`'s own error costs 28.15 -> 80.79 dB on frame 0 (post is ~7% of the remaining error); then every historical `sr_post.comp` was run through the *current* chain and measured with `tools/report/pipeline_report.py` on the 256x144 oracle replay:
  - current (a3a050b, attempt 6): sr_M min 28.15 / mean 30.14; sr_L 26.24 / 29.61
  - attempt 5 (c2e1f37): sr_M min **32.20** / mean **33.76** (frame 0: 38.97); sr_L **32.82 / 34.08**
  - aa192f0 (between 5 and 6): sr_M 29.75 / 30.36
  - rr_F unchanged 26.78 / 27.17; offline_check PASS 27 pipelines.
- **Verdict**: commit `99b33b2` restored attempt 5's shader. The attempt-6 rewrite is more ISA-faithful (Catmull-Rom history, covariance floor) but faithfully consumes `k_sr_dec0`'s still-wrong gates/factors (`u8vec4` confidence channels), amplifying their error; attempt 5's `f16vec2` reading of the same bytes is more forgiving. Re-evaluate attempt 6 only after `k_sr_dec0` is decoded.
- **Do not retry**: sr_stage @ dfaed7f, sr_tube_split @ 30b5455, sr_enc0 @ 4ec6263 / cfe5b15 in the current chain - all measured 32.20 / 33.76 (identical to 2 decimals); only the post swap moves the pipeline.

### Attempt 8 (2026-10-26): Step B - gates/factors/aniso wired, network reaches the image

- **Change**: `gates`/`factors` re-declared `u8vec4` (they were `f16vec2`, wrong about the type:
  lane 3 is 0 for all 36864 oracle pixels, and read as f16 52.8% of `gates` came out non-finite;
  `u8vec4` is what `sr_dec0.comp` writes).  The invented
  `h_weight = clamp(0.90 - max_diff * 2.0, 0.0, 0.90)` is replaced by the decoded gate/factor
  3-way blend `mix(t_spatial, f0*t_spatial + f1*t_hist + f2*t_clamped, g)`, and the invented
  2x2 colour box by the anisotropic 9-tap that `aniso`'s (Axx, Ayy, Axy) exists for.  Both
  invented constants are gone; `aniso`/`gates`/`factors` each have a consumer.
- **Result** (`tools/report/pipeline_report.py`, 256x144 oracle replay, frames 0-8):

  | cell | before (attempt 5) | after (attempt 8, wired) |
  |---|---|---|
  | sr_M min/mean | 32.20 / 33.76 | **27.51 / 29.55** |
  | sr_L min/mean | 32.82 / 34.08 | **26.25 / 29.62** |
  | rr_F min/mean | 26.78 / 27.17 | 26.78 / 27.17 (unchanged, no RR shader touched) |

- **The drop was pre-committed and must not be reverted.**  `sr_post` now faithfully consumes
  `k_sr_dec0`'s still-wrong outputs, so 33.76 -> 29.55 is the predicted fall toward ~28.  The
  same pattern already happened once (a3a050b measured worse and was reverted by `99b33b2`);
  undoing *this* wiring to recover a number is forbidden by the operator's 2026-10-26
  no-revert rule.  Judge progress per-kernel against oracle inputs instead (Step C and the
  2026-11-02 milestone).
- **Wall test**: before, injecting oracle buffers at all 12 network points left all 9
  `out_i.f32` byte-identical (32.20/33.76 either way).  After, 9/9 differ.  Pinned per link
  with `DLSS5_AMD_DUMP_BUF`: `full15` changes when `dec0` is injected (gates/factors/aniso
  reach `sr_post`), `t7` changes when its skip `t3` is swapped for the oracle's.
- **Provenance**: commit `f0c244f`; `sha256sum build/shaders/*.spv | sha256sum` =
  `34e6745ebf8d060640cb4f280d754969d36786504915489b37dd2586f5c6ca8f`; inputs
  `build/oracle/oracle-8dbf027a9e-s20261025/frames/{sr,rr}` (256x144 -> 512x288, 9 frames);
  `offline_check` = PASS 27 pipelines.
- **Next for this kernel**: re-evaluating attempt 6's Catmull-Rom history is now unblocked in
  principle, but it should wait until `k_sr_dec0` stops being wrong - a faithful consumer of
  wrong data has measured worse here twice.

## Provenance of the shader now in the tree - READ BEFORE CITING IT

**The `sr_post.comp` in the tree is a hybrid: decoded, invented, and wired.**  Its parts:

| part | status |
|---|---|
| 9-tap anisotropic spatial reconstruction driven by `aniso` (Axx, Ayy, Axy) | **decoded** (attempt 6, from `k_sr_post` ISA) |
| gate / factor 3-way temporal blend from `gates` + `factors` | **decoded** (attempt 6) |
| bilinear MV warp of `in_mv_nondil`, bilinear `sample_history` (not the decoded Catmull-Rom) | **invented stand-in** |
| `if (dot(t_hist, t_hist) < 1e-6)` history discard | **invented stand-in** |

It is a stand-in, never "the baseline".  The invented pieces conflict with `HANDOFF.md`
"Do not retry" (no generic TAA blends); they stay because the chain needs a post that runs and
because the bilinear history/MV path measures best while `k_sr_dec0` is still wrong.  Do not
extend the invented parts to any other kernel, do not cite them as decoded, and do not present
the end-to-end number as a parity milestone - it is now, deliberately, lower than it was before
the network was connected.

## Review R5 (2026-10-26): structure map from the ISA, and a hardware-exact intermediate-value probe

**Instrument: `tools/re/patchprobe.py`.** It overwrites a few instructions of the code object that
srref runs with "store register vX to an output buffer at a per-thread address; s_endpgm", runs
srref, and leaves the value in the dump. The texture filtering and every rounding are the hardware's
own. `tools/re/live.py` checks that a register still holds its value at the patch point.
Validated: patching the full15 store itself reproduces the oracle's pixel-0 rows at 100%. SSA
listing (git-ignored): `build/work/review/post/post.ssa` (`tools/re/ssa.py build/k_sr_post.s`).

**Layout.** Each thread handles 2 output pixels in y (rows 2r and 2r+1). Pixel 0's path runs from
0x5f0 to the stores at 0x25e8/0x25f4; pixel 1's path follows from 0x26ac. The pixel-0 path
computes the spatial filter for both pixels at once, in the lo/hi halves of packed f16.

1. **Staging (to 0x578).** 192 threads stage a 12 x 16 source tile, clamped to the image, into LDS
   at 24 bytes per pixel (row stride 0x120):
   - tm(R), tm(G), tm(B), where `tm(c) = 0.36 * log10(max(1 + 100 * min(c*e, 32752), 1e-4))`,
     e = `expo` (1 if 0), rounded to f16 (`v_fma_mix*`);
   - T[factor bytes 0..2] and T[gate bytes 0..2], where T is the code object's embedded u8 -> f16
     table (`s_getpc + 0x5df4`; the port's `sr_unorm8_table` builds it);
   - f16(-0.7213 * Axx), f16(-0.7213 * Ayy), f16(-1.4427 * Axy) from `aniso`.
     The constants are -0.5*log2(e) and -log2(e).
2. **Spatial filter (0x868-0xf00).** A Gaussian splat over the 3 x 3 source neighbours. Tap
   weight `exp2(a dx^2 + b dy^2 + c dx dy)` uses the SOURCE pixel's coefficients, with dx, dy the
   output position minus the source centre, and offsets -1/0/+1 in f16. The quadratic forms and
   the weighted colour sums are packed f16; the weights are f32 `v_exp_f32`, normalized by their
   f32 sum. The same pass accumulates three scores, one per motion-vector candidate.
3. **MV choice (0xef4-0xf68).** The scores rank three candidates: `mv_nondil` sampled through its
   texture at the pixel (`image_sample_lz`, args +0x60), `prev_mvec1` (+0x48) and `mv_ero`
   (+0x50). Reprojected position = pixel position + the best MV; on-screen test against W15/H15.
4. **History (0x1290-0x15d0).** 5-tap Catmull-Rom at the reprojected position (fraction weights
   with the 1.5 / 2.5 constants, combined into 5 bilinear `image_sample_lz` of `ihist`, args
   +0x38), normalized by the weight sum. If the position is off-screen, the tonemapped raw `g_col`
   at the nearest source pixel is used instead.
5. **Second candidate (0x15d4-~0x1e00).** Confidences `clamp(1.1*s - 0.05, 0, 1)` for the top two
   scores. If the runner-up's normalized share exceeds 1/255 (`0x3b808081`), the same history fetch
   runs at the second MV (Catmull-Rom set at 0x19ec) and the two are mixed.
6. **Blend and outputs (~0x1e00-0x25f4).**
   - The gate/factor blend uses `dec0_f1` (args +0xa8), then the inverse tonemap: clamp to
     [1e-4, 3.0], `10^(x/0.36)` via exp2, then `(... - 1) * 0.01`.
   - NaN guards (`v_cmp_u/o`).
   - full15 = R11G11B10 (`med3` clamps to 2047 / 1023, `v_cvt_u32`, `v_or3`).
   - ihist1 = f16 x 4 (`v_perm` pack).
   - dhist (a 2 x 2 average at half resolution) after pixel 1.

The current shader's invented parts (the bilinear MV warp, the bilinear history, the discard test)
are therefore all wrong in kind: MV candidate selection, Catmull-Rom history, a two-candidate mix.
