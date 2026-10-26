# Notes: k_sr_enc0

## Launch Details
- Launch 2 in `sr_M` frame 0: `_ZN2sr2io9k_sr_enc0ENS0_8Enc0ArgsE`
- Grid: [32, 20, 1], Block: [32, 1, 1]
- Inputs: `g_col`, `g_dep`, `g_mv`, `prev_mvec0`, `dhist`, `hidden0`, `wbuf`, `expo`
- Outputs: `prev_mvec1` (f16, 73728), `mv_ero` (f16, 73728), `mv_nondil` (f16, 73728), `t0` (u8, 655360)

## Attempts and Hypotheses

### Attempt 1
- **Hypothesis**: Motion vectors are passed directly without scaling.
- **Result**: 0 / 73,728 bit-exact (first mismatch at index 0: port -0.0953369, oracle -0.2861328).

### Attempt 2
- **Hypothesis**: Motion vectors are scaled by 3.0 (`g_mv * 3.0`), mapped in 8x8 blocks per 32-thread workgroup (2 pixels per thread).
- **Result**:
  - `prev_mvec1 [f16]`: 73,728 / 73,728 (100.0000% bit-exact), max |err| 0, max ulp 0.
  - `mv_ero [f16]`: 73,728 / 73,728 (100.0000% bit-exact), max |err| 0, max ulp 0.
  - `mv_nondil [f16]`: 73,728 / 73,728 (100.0000% bit-exact), max |err| 0, max ulp 0.
  - `t0 [u8]`: 655,360 elements, bit-exact 1509 (0.2303%), max |err| 192.
  - 3 of 4 output buffers are 100% bit-identical.

### Attempt 3
- **Hypothesis**: `t0` tensor layout is `[C/32][H][W][32]` (two 32-channel groups of 32 contiguous bytes per pixel: group 0 at offset 0, group 1 at offset `W * H * 32`) with 64 FP16 biases loaded from `enc0_w` offset `1590016u`.
- **Result**:
  - `t0 [u8]`: bit-exact count increased 4x to 6,017 elements. Index 0 is 1 ULP off (port 48.0, oracle 47.0).
  - All 3 companion buffers remain 100.0000% bit-exact: `prev_mvec1`, `mv_ero`, `mv_nondil`.
- **Pipestats**: 108 SGPRs, 24 VGPRs, 0 spilled SGPRs, 0 spilled VGPRs, 0 scratch size.

### Attempt 4 (2026-10-27): t0 is a linear patch-embedding; frame0 decode closes at fp8 quantization
Context: `docs/notes/sr_chain_decoupling.md` proved the chain's t0 is the placeholder above and
that wiring real features is the SR campaign's first unblocker.  Method: least squares of the
oracle's frame-0 t0 (`dump/frame0/002.t0.bin`, value-decoded) against a candidate feature
basis built from the oracle's own gather outputs (gather is EXACT, so `000.g_*` are the true
inputs):
- Features tried: tonemapped colour `0.36*(log2(1+100*c*expo)*log10(2))` at the four 2x2
  subpixels x rgb (12), `mv*3` x/y, depth, constant 1 (16 total, render padded 144->160 rows
  for the H0/2 = 80 cell rows).
- Result: **every one of the 64 channels is a linear combination of this basis with mean
  residual 0.029 (max 0.050) = the fp8 half-ulp at the channels' value scale** - i.e. on
  frame 0 the embed is exactly linear in this feature span: t0[c] = fp8(sum_f W[c][f]*feat_f).
  No two channels are identical (no RR-style small template pool); per-channel means/stds
  vary smoothly (a learned 64 x F matrix, not a hand-coded pattern).
- The fitted W does NOT generalize to frame 1 (residual 0.40 = 13x quantization) - frame 1
  adds history dependence: `prev_mvec0` (frame-0's `mv*3`) is a real input on moving frames
  (the oracle's frame-1 enc0 launch reads `prev_mvec0` and writes it back), so the feature
  set must be extended with erosion/history terms (e.g. `mv*3 - prev_mv*3` and friends).
- The weight VALUES must come from `enc0.bin` at runtime (bytes 0..8192 precede the 64 f16
  biases at 8192..8320 that the current shader already reads) - matching the fitted W against
  the packed bytes was inconclusive (32% within 1e-3 of an fp8 value) because correlated
  subpixel proxies smear the fit; the next attempt should (a) add the history features and
  re-verify frame-1 linearity, then (b) identify the exact feature->byte mapping (value
  matching with a sparse/orthogonal fit, or the ISA of `_ZN2sr2io9k_sr_enc0...`), so the
  shader reads W from `enc0_w` and no fitted constant is ever committed (clean-room rule).
- **Status**: APPROX; the placeholder is still what ships.  This is the SR campaign's
  critical path (see `docs/notes/sr_chain_decoupling.md`).

- **Follow-up (same day)**: adding the obvious history features to the basis - `prev_mvec0`
  (start copy from the frame-1 dump), `hidden0` (start copy), `mv*3 - prev_mv*3` erosion
  terms - does NOT close frame 1: the frame-1-only fit residual stays 0.15 (5x the fp8
  half-ulp) and a joint frame0+frame1 fit gives 0.27/0.26 on both frames (vs 0.029 for
  frame0 alone on the 16-feature basis).  So the cross-frame dependence is NOT a linear
  function of (tm(col 2x2), mv*3, dep, prev_mv, hidden) - remaining candidates: `dhist`
  (10:10:10 u32 at hW15 x hH15 = 384x216, cell mapping unknown), nonlinear erosion terms
  (abs/max on the mv difference), or exposure/jitter-dependent preprocessing.  Do not
  repeat the plain-linear-with-these-features attempt; either add dhist + nonlinear mv
  terms next, or go straight to the ISA (`tools/re/fn.sh sr _ZN2sr2io9k_sr_enc0...`).

### Attempt 5 (2026-10-27): correct alignment, trunc-not-RNE store, the c20 recipe, v_log_f32 measured
Four findings that change the decode (all measured, probes in `build/work/probe/enc{0..7}`
= constant-color frame sets + `srref --dump`):
1. **The attempt-4 fits were mis-aligned**: `t0.flat.reshape(-1,64)` interleaves two adjacent
   cells' 32-byte groups; the cell-major layout is `t0.transpose(1,2,0,3).reshape(10240,64)`
   ([y][x][group][channel]).  With the correct alignment the frame-0 linear model
   {12x tm(col 2x2 subpixels), mv*3, dep, 1} closes at **fp8 half-ulp residual (median
   0.0159) and 50.26% bit-exact**; f16/fp8 weight snapping is neutral (weights are f16-grade);
   raw-col / 4-subpixel mv+dep extras do not help (48.7%).
2. **The store path truncates, it does not RNE**: every t0 value is built as
   `f16 add/sub chain -> v_and bits & 0xffffff80 (= clear 7 low mantissa bits: trunc toward
   zero onto the E4M3 grid) -> f32 -> v_med3(x, s39, 448.0) -> v_cvt_pk_fp8`.  Modeling
   truncation instead of RNE explains the 50% RNE ceiling; the residual off-by-one cases are
   the f16 intermediate roundings, which need the exact per-channel term order.
3. **The channel recipes are sparse, not dense**: ISA slice of convert site 0xC20 shows
   `v38 = f16(tm) ; v44 = tm_bits & 0xfc00 (power-of-two floor) ; v63 = v44 | (col_bits>>9) ;
   value = trunc_e4m3(f16(f16(tm)+v63) - v44)` - three terms with bit-level constructions,
   plus `0xfffffc00`-masked power-of-two constant channels.  The numpy reconstruction of
   this recipe reproduces ~16% of one channel bit-exactly (>>random 0.3%) - operand details
   (which subpixel feeds the tm vs the raw term, packed-half selection, the s39 lower
   clamp) still need decoding per channel.  There are 24 `v_cvt_pk_fp8` sites (+masked
   constant channels) to enumerate; `tools/re/slice.py <site>` gives each recipe.
4. **v_log_f32 is essentially exact** (measured on the GPU via inline-asm sweep,
   `build/work/logprobe/`: max relative error 5.8e-8 vs f64 log2) - the hardware log is NOT
   a source of error; the attempt-4 "log error" recovery was noise.  The tm constants are
   confirmed: `1+100*x` fma, min clamp **0x38d1b717** (not 1e-4 exactly), x log10(2)
   (0x3e9a209b), x 0.36 (0x3eb851ec), with NaN->0 and >=1024 guards.
- **Status**: APPROX; the placeholder still ships.  Next: enumerate the 24 convert-site
  recipes with slice.py (subpixel of the raw term, packed halves, s39), implement the
  exact per-channel recipes in `sr_enc0.comp` reading the raw f16 colors + computing tm
  with the confirmed constants, then extend to frame 1 (history features - `start.*` copies
  give the exact prev_mvec0/hidden0/dhist inputs; plain-linear history was refuted in
  attempt 4).

### Attempt 6 (2026-10-27): built a per-lane ISA emulator; found the real t0 path is in the EXECUTED region 0x13BC+, not the attempt-5 recipe site
`build/work/enc0emu.py` (git-ignored) now executes the enc0 disassembly with concrete inputs
(the dump's launch-2 args + the frame files): ~50 opcodes (dual-issues, packed f16, signed16
branch offsets, exec masking,64-bit address pairs, v_maxmin's TWO destinations, div/rcp/sqrt,
writelane/readlane).  Run: `python build/enc0emu.py build/enc0_isa.s`; it reaches the
MV-output stores (96 x b32, addresses decode against the correct buffer bases) and the t0
write LOOP, then dies on loop-exit semantics.  Findings that change the decode:
1. **`s_branch 1110` at 0xBF60 is UNCONDITIONAL**: the block 0xBF64..0x13BB - which contains
   the 24 v_cvt_pk_fp8 sites and the recipes decoded in attempt 5 - is DEAD CODE (an unchosen
   template path).  The executed t0 path starts at **0x13BC** and contains v_div_scale/
   v_div_fmas/v_div_fixup, v_sqrt, v_rcp - i.e. the real t0 features involve division/sqrt
   (norm-like terms), not just tm+trunc.  **Attempt 5's c20 recipe must not be relied on.**
2. **The kernel has a pixel/channel loop** entered by `s_cbranch_scc0 65317` at 0xFB20:
   the printed immediate is signed16 (= -219).  All branch offsets must be read signed-16;
   two anomalies (immediates 61991/61977 at ips 2614/2625 implying negative targets) suggest
   **embedded data is being disassembled as instructions inside the fn.sh slice**, shifting
   instruction indices after it - fn.sh/awk takes everything until the next symbol label.
   Next session: strip data blobs (or drive the emulator by the printed hex addresses rather
   than list indices) before trusting any ip arithmetic.
3. The MV store chain decodes: base = s80:81 (a pointer copied through s47..s50/s90..s93 from
   an args slot), offset = (y_reflected * pitch + x_reflected) << 2 via
   v_mad_co_i64_i32 + v_lshlrev_b64 - the reflect uses v_maxmin pairs feeding cndmask; my
   earlier -1/1 corruption here was the missing v_maxmin SECOND destination (fixed).
- **Status**: APPROX; emulator ~90% there.  Exact next steps:
  (a) make fn.sh-style parsing address-driven (use the printed 0x addresses, not list order)
      to eliminate the data-blob index drift;
  (b) finish the loop-exit semantics (the 0xFB20 loop's scc comes from the div/cmp chain);
  (c) identify the t0 b64-store site inside the loop and validate its bytes against
      002.t0.bin for wgid(0,0) - then all workgroups -> full t0 reproduction;
  (d) decode the executed-path recipes (0x13BC+): the div/sqrt terms first;
  (e) then implement in sr_enc0.comp (weights from enc0_w, tm constants confirmed in
      attempt 5: 1+100x fma, min-clamp 0x38d1b717, xlog10(2), x0.36; v_log_f32 ~ exact).
