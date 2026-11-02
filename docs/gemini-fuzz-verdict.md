# Gemini analysis #2 — verification verdict (every claim, evidence-first)

Verifier: subagent pass over `docs/gemini-fuzz-analysis-2.md`, 2026-11-01, on branch
`game-integration` at `2ff82f4` + this pass's host fixes.  Every citation in the Gemini document
was re-read from the repo (or from the in-game log, read-only); file:line below is what the tree
actually says at verification time, not what Gemini wrote.  Verdicts:

* **TRUE** — the claim as stated holds, at the cited location, with the cited substance.
* **PARTLY-TRUE** — substance holds, details (line, binding number, magnitude, scope) do not.
* **FALSE** — the committed code/docs do what Gemini says they do not.
* **STALE** — was once true, already fixed or superseded.
* **IRRELEVANT** — true as code but provably not exercised by this game / this symptom; stated
  so explicitly because Gemini's priority list implies causation for the user's report.

Key context Gemini did not have (from the in-game log, `bin64/dlss5-amd-ngx.log`, read-only):

* The **FFX surface runs the gameplay frames**: the last session's `[ffx]` chain is built from
  `ngx-weights/M` (`[ffx] dispatch: creating the chain 1280x720 -> 3840x2160 from
  .../ngx-weights/M`) and emits 44 227 `eval #N OK: id=0` frames before
  `[ffx] ffxDestroyContext: destroyed`.  The game runs **preset M at 3.0x on the FFX path**.
* The **NGX surface also exists in the same log** and chose **preset L** for the identical
  geometry: `D3D12 CreateFeature: id=1 feature=1 preset=L 1280x720 -> 3840x2160` →
  `SR preset L ...: chain ready ... weights .../ngx-weights/L`.  `nvngx.cpp:1782` maps the app's
  `DLSS.Hint.Render.Preset.Quality == 12` (ngx_abi.h:45, L) to `'L'`.
* Geometry in-game is only ever 1280x720 -> 3840x2160 (3.0x) and 1920x1080 -> 3840x2160 (2.0x,
  one earlier `preset=M` CreateFeature).  **No < 2.0x ratio ever runs.**
* Both surfaces get the same jitter (`jitter=(-0.25,0.166667)` on `[ffx] ffxDispatch: first
  frame`, identical `d5 diag: jx=-0.250000 jy=0.166667` on the NGX side), `pre_exp=1.0000`,
  `MV.Scale=(1280,-720)`, and every input pitch line reads `[every row aligned]`.

---

## The table (every claim in gemini-fuzz-analysis-2.md)

| # | Claim (as written) | Verdict | Evidence (file:line, this tree) | Action |
|---|---|---|---|---|
| 1 | `ffx_front.cpp:289-290` casts `kQualityRatio` to `(int)`, so 1.5f→1 returns full display res | **TRUE** (code) / **IRRELEVANT** (symptom) | was `displayWidth / (int)kQualityRatio[...]`; at this game's ratios (int)3.0f=3 and (int)2.0f=2, so 3840/3=1280 either way — the truncation only bites 1.5x/1.7x, which the game never requests | **FIXED** `ffx_front.cpp:297` — floating-point division with +0.5 rounding; bit-identical at 2.0x/3.0x |
| 2a | `ffx_front.cpp:378` hardcodes `preset='M'` even at 3.0x where L applies | **TRUE, LIVE** | was line 378 (`c->fs.preset='M'`) + line 424 (`root + "/M"`) + line 429 (`FrameChain::create(...,"M",...)`); in-game proof the FFX path ran M at 3.0x (`[ffx] ... from .../ngx-weights/M`, 44 227 frames) while the NGX path on the same build ran L for the same geometry (`CreateFeature ... preset=L 1280x720 -> 3840x2160`) | **FIXED** `ffx_front.cpp:389-401` — preset inferred from frame geometry (both per-axis ratios ≥ 2.5 → `'L'`), `DLSS5_AMD_PRESET=L\|M` env override for A/B; ratios ≤ 2.0x stay `'M'` (ffx1/ffx2 harness cells byte-identical) |
| 2b | "the game goes through the NGX surface and never ffxCreateContext" ⇒ FFX claim may be STALE | **FALSE (both directions)** | the game uses **both**: FFX ran the long gameplay stretch (id=0, preset M), NGX ran a short later stretch (id=1, preset L).  `[ffx] [S6] mv scale=(1280,-720)` exists in the log ⇒ FFX path is live, so claim 2a is **not** stale | — (recorded so the mixed topology is not re-litigated) |
| 3 | `ffx_front.cpp:512-513` assigns `motionVectorScale` with no isfinite/!=0 guard (contrast `nvngx.cpp:482-483`) | **TRUE** | guard existed only at `nvngx.cpp:482-483`; FFX assigned raw | **FIXED** `ffx_front.cpp:542-543` — identical guard (reference `DlssNr_Pipeline_Dx12.cpp:123-128`); in-game (1280,-720) unchanged |
| 4a | `FFX_UPSCALE_ENABLE_DISPLAY_RESOLUTION_MOTION_VECTORS` never handled on FFX | **TRUE** | ffx_front only ever tested `NON_LINEAR_COLORSPACE` (create); flag bit 1 (ffx_upscale.h:43) unread; in-game flags are 0x29/0x39 → bit 1 clear ⇒ not exercised in-game | **FIXED** `ffx_front.cpp:556-576` — flag logged, `fp.mvhi` derived from the fetched MV extent (dims win over the flag, with a mismatch log line) |
| 4b | `fp.mvhi` never set on the FFX path (nor anywhere on the bridge) | **TRUE** | `FrameParams.mvhi` (pipeline.h:41) zero-init everywhere; runner.cpp:222 sets it for cells only; nvngx.cpp:623 comment literally said "the bridge never sets it" | **FIXED** FFX `ffx_front.cpp:569`; NGX `nvngx.cpp:1371-1382` (below); comment updated `nvngx.cpp:622-626` |
| 4c | `copy_row_texels` falls back to a tight `ow*oh` copy when `img.w > W`, then `sr_gather` reads at render stride W (shear) | **TRUE** | `nvngx.cpp:675` returns 0 once `img.w > W` → `bufferRowLength=0` tight pack; `sr_gather` runs at `pmv = W*bpp` (pipeline.cpp:1857/1870) unless mvhi | **FIXED** by routing such images to mvhi (4a/4b/4d): the tight pack *is* the oW pitch `sr_gather_mv` reads (`pipeline.cpp:1878/1885`); `in_mv_` is sized `max(W,oW)*max(H,oH)*16` (pipeline.cpp:968) so the copy fits |
| 4d | `nvngx.cpp:1371` fails wide MV inputs instead of setting mvhi | **TRUE** (was an unconditional `NVSDK_NGX_Result_Fail` for any `img.w > W` with matching texel size) | old lines 1360-1378; now `nvngx.cpp:1371-1382`: SR + image exactly `s->ow x s->oh` → `fp.mvhi = 1` + log, no fail; every other wide-MV case keeps the refusal (RR unchanged — no `sr_gather_mv` twin there) | **FIXED**; harness cells feed render-sized MVs → `mvhi` stays 0 → byte-identical |
| 5a | `fp.auto_exposure` forced to 1 on the FFX path, `d->exposure` resource ignored | **PARTLY-TRUE / mis-cited** | there is **no `auto_exposure` field** in `FrameParams` (pipeline.h:30-60) — nothing "forced to 1"; `fp.exposure = 1.0f` at `ffx_front.cpp:532` is real, and `d->exposure` (the 1x1 FFX resource, ffx_upscale.h:89) is indeed never read.  But the NGX surface is equally exposed: it only reads `DLSS.Exposure.Scale` (nvngx.cpp:460-461, default 1.0), and in-game `d5 diag ... pre_exp=1.0000`.  Consuming the 1x1 texture needs a GPU-side copy+convert into the f16 parity `expo` buffer — shader-side work, out of this round's lane | **NO CODE CHANGE** — parity with the NGX surface is preserved; listed as a known limitation, not the symptom (exposure affects brightness, not sharpness distribution) |
| 5b | `d->preExposure` ignored on the FFX path | **FALSE** | `ffx_front.cpp:531` — `fp.pre_exposure = d->preExposure > 0.0f ? d->preExposure : 1.0f` (same effective rule as the NGX key read) | none |
| 5c | "In-game zero MVs (S6): 99.9% exact zeros (max\|v\|=8.25e-5) during camera pans ⇒ no reprojection ⇒ fuzz" | **PARTLY-TRUE / mis-quotes its own source** | `tools/scenario/REPORT.md:910` says **96.9 % pixels NONZERO**, 100 % finite, max\|v\| = 8.25e-5 *raw* (= 0.106 px after the ×1280 scale) — not "99.9 % zeros".  REPORT.md:915 leaves the question **pending**: a moving-camera scene with the S6 probe (`maxAbs` at frames 300/900/1800) decides "units confirmed" vs "SL's MV transfer broken upstream".  That probe has never come back (the dump marker is disarmed).  Direction check: near-zero MV hurts **moving** content reprojection (ghosting), while stills would have ≈0 MV legitimately — it does not by itself pick "stills worse" | **NO CODE CHANGE** (open measurement — see "How to read the running agent's 3.0x static cell" below) |
| 6a | `sr_enc0.comp:71,524` dead-stores `out_mv_ero` (binding 7) while `sr_enc0L_in.comp` still writes it | **TRUE** | `shaders/sr/sr_enc0.comp:71` ("binding 7 (mv_ero) removed 2026-10-28: dead") and `:524` (store removed); `shaders/sr/sr_enc0L_in.comp:83,493` still declares and writes it; removal commit **`20919d4`** (Wed Oct 28, "SR M feature-kernel tuning", a perf commit) | **PATCH PLAN P1** (shader lane, NOT applied) |
| 6b | `sr_dec0.comp:32-34,59,310` removed the factors output because "no shader reads them" | **TRUE** | `shaders/sr/sr_dec0.comp:33-35` (header), `:59` ("binding 4 (factors) removed 2026-10-28"), `:310` ("factors: REMOVED (dead store + dead softmax)"); same commit `20919d4` | **PATCH PLAN P1** (NOT applied) |
| 6c | `sr_post.comp` leaves `in_prev_mvec1 (binding 2)`, `in_mv_ero (binding 3)`, `in_factors (binding 4)` unreferenced | **PARTLY-TRUE — binding numbers are wrong** | the actual map (`sr_post.comp:66-71`, `pipeline.cpp:2251`): **binding 2 = gates (READ)**, 4 = aniso (READ), **3 = factors, 6 = prev_mvec, 7 = mv_ero** are the three bound-but-unread slots.  The substance (three temporal inputs unread) is TRUE | **PATCH PLAN P1** with corrected bindings (NOT applied) |
| 6d | `sr_post` uses a single nearest-neighbour `in_mv_nondil[my*W+mx]` and omits the 3-candidate ranking + second 5-tap Catmull-Rom blend + `clamp(1.1*s-0.05,0,1)` share test | **TRUE as a divergence from the real .co** | `shaders/sr/sr_post.comp:467-475` — nearest lookup + the honest comment "NOT DECODED (Review post-3)"; the real kernel's steps 3/5 (scores rank `mv_nondil`/`prev_mvec1`/`mv_ero`, runner-up share > 1/255 → second CR fetch) are `docs/notes/k_sr_post.md:148-161` (Review R5, ISA + patchprobe) | **PATCH PLAN P1** (NOT applied) |
| 6e | "…deleted during the 2026-10-28 perf pass" | **PARTLY-TRUE / mischaracterised** | what Oct 28 (`20919d4`) deleted were **dead stores of buffers no reader had**.  The *consumer* was never ported: `951cd0e` (Oct 27) rewrote `sr_post` against `sr_post_model.py`, whose docstring at `tools/sr/sr_post_model.py:57-60` says **"NOT MODELLED: the MV candidate selection and the second-candidate mix"**, and `k_sr_post.md:189` keeps "The MV choice, Catmull-Rom history and second candidate" as an **open** item (R6 #3).  So this is a never-ported feature, not a regression introduced by the perf pass | PATCH PLAN P1 (NOT applied); correction recorded here |
| 6f | "documented in `k_sr_post.md` R5 **and implemented in `sr_post_model.py:130-185`**" | **R5: TRUE.  Model: FALSE** | R5 documents the real kernel (`k_sr_post.md:148-161`) ✓.  But `sr_post_model.py:130-185` is the tail of the docstring + imports + constants; the model **explicitly does not implement** selection (`:57-60`) and takes `mv = mv_nondil[0,0]` directly (`:231`).  On the oracle's own frames all three candidates are identical and constant (`mv = (-0.2861328125, -0.075)`, `:57-59`, `:120-121`) | Correction recorded; drives the CRITICAL question below |
| 6g | **CRITICAL**: was the removal a provable no-op on the oracle's cells, and does it only matter on disoccluded content the cells lack? | **YES to both** | (i) mv_ero/factors: no reader existed ⇒ trivially no-op for port output; (ii) the *selection* would be a no-op on the oracle's frames anyway — all three buffers equal (`sr_post_model.py:57-59`), which is why the model never needed it; (iii) gates cover outputs, not internals: `quality_guard` B1/G1/G2/T1 and `psnr3x_gate` V4 all compare frames.  The removals **cannot move any current cell**; only content where `mv_nondil ≠ prev_mvec1 ≠ mv_ero` (depth edges/disocclusion, motion) can expose it — exactly the content none of the cells contain | PATCH PLAN P1 names the cell that must move to prove any restore |
| 6h | Therefore the fuzzy regions are these missing pieces at "character silhouettes, foliage, hair, depth edges" | **MECHANISM PLAUSIBLE, SYMPTOM MISMATCHED** | the mechanism only fires where the three MV candidates differ ⇒ moving/disoccluding content.  For **still** objects all three candidates are ≈0 and identical ⇒ selection is a no-op — the exact opposite of "stills far worse than moving".  Also refuted directionally by psnr3x cells D/E (3.0x L, moving corpus): port matches the oracle to 52-57 dB port/ref *with* the selection missing | Ranked below the FFX-preset finding in the judgment; shader work stays a patch plan |
| 7 | `srref.cpp:211,520-521` selects `k_sr_post<...,15,27,...>` when `W15/W < 3.0`, while `sr_post.comp:127-134` hardcodes `ST_C=ST_R=20` and can read OOB LDS below ~1.4x | **TRUE as arithmetic / IRRELEVANT at 3.0x** | `tools/sr/srref/srref.cpp:211` and `:521` verified verbatim.  Tile math (mine): need = `X0hi-X0lo + kmax + 2`, `sX = W/W15 = 1/(1.5R)`: 3.0x → need ≈ 10; 2.0x → 14/15; 1.5x → 17/18 (≤20, safe); 1.0x → 23/25 (>20 → `min()` clamps at `sr_post.comp:340-341`, readers `sA[(by+ky)*ST_C + bx + kx]` at `:408/421` are **not** clamped → rows ≥ 20 index past `sA[400]` ⇒ real OOB below ≈1.25-1.4x).  **But the game runs 3.0x: W15/W = 4.5 ≥ 3.0 ⇒ even the oracle picks `<12,16>`, and the needed tile (≈10) is half of 20.**  Cells C/D/E directly prove `sr_post` matches the oracle at 3.0x (min 52.4-57.3 dB port/ref).  The game never runs < 2.0x (log: only 1280x720 and 1920x1080 → 3840x2160) | **PATCH PLAN P2** (ST_C/ST_R 20→27, NOT applied) — a latent DLAA/1.0x bug, explicitly **not** a candidate for this symptom |
| 8 | `dq()` stochastic FP8 dither removed in `sr_enc0.comp:456`/`sr_enc0L_in.comp:434` | **TRUE, deliberate, gate-covered** | M: `sr_enc0.comp:452-456` early-return with the rationale in-comment ("cost 0.219 ms and the ground-truth guard passes without it (static G1 -0.041 vs -0.021 with dither)"), commit `20919d4` (Oct 28).  L: `sr_enc0L_in.comp:434` no-dither `DQ2` behind the RADV/non-F16 build, commit `c8c9f6d` (Oct 29, "the L guard decides").  Coverage: the ground-truth guard IS `quality_guard`'s G1/G2 on the static cells — fresh run `sr_M_static G1 -0.042`, `sr_L_static G1 +0.014` (both ≫ -0.20).  Not bit-exactness — a measured quality trade | **NO ACTION** — covered; a perceptual-softness effect on dark gradients is possible but small and would hit moving+still content alike |
| 9a | "`content_gate.py` only checks non-emptiness (std>0.01, nan==0, nonzero_pct>50%, corr>0.5)" | **FALSE** | the committed `tools/optiharness/content_gate.py` (g3b, `3fc1cea`) computes per-pair `mean |out_i - out_{i-1}|` and `frac(|Δ|>0.05)` and gates them against **calibrated floors** (0.001 / 0.01, frozen detector) and **scramble ceilings** (0.40 / 0.55) — three-population calibration in the docstring (lines 16-50), thresholds at `:71-73`.  No std/nan/nonzero/corr rule exists anywhere in the file | none — Gemini described a gate that is not this file |
| 9b | "A heavily blurred or ghosted image passes content_gate.py at 100%" | **TRUE (consequence of design)** | correct — the gate is deliberately oracle-free ("no reference image", `:3-4`): it detects frozen and scrambled outputs, not softness.  Sharpness gating needs a reference (`--ref`), which is a proposal, not a defect in the claimed code | noted; `--ref` PSNR/edge/Laplacian mode left as a proposal (would need oracle outputs for the synthetic scene, which do not exist) |
| 9c | "opti_client drives the NGX path, not dlss5-amd-ffx.dll" | **STALE (partly)** | default cell: yes, NGX (`g1 loaded_from_nvngx=1`).  But since S8 the harness has **two FFX cells** (`run_harness.sh --ini ffx1.ini` = stock Opti `FfxDx12SRPath`, `ffx2.ini` = fork entry) that load `dlss5-amd-ffx.dll` — REPORT.md S8 rows, both PASS x2/x3 | none |
| 9d | "opti_client … constant global MV … where foreground/background disocclusions do not exist" | **TRUE** (detail wrong: mvy = 0.25, not 1.0) | `opti_client.cpp:378-386` `synth_mv` writes the same `(2.0, 0.25)` (or `(2/w,-0.25/h)` UV) for **every pixel**; `synth_depth` `:392-398` is a plain y-gradient; the scene is a full-frame gradient+checker (`:362-372`) — there is no foreground object at all | **FIXED** — new default-off `--scene disoc` gives the foreground rect its own depth and its own (counter-panning) MV; default scene byte-identical, all calibrations untouched |
| 9e | "`quality_guard` only runs 2.0x cells (seq 960x540→1920x1080, static 1920x1080→3840x2160)" | **TRUE, by design** | `quality_guard.py:86-89` CELLS: seq 960x540→1920x1080 = 2.0x, static 1920x1080→3840x2160 = 2.0x.  The docstring `:34-36` names the delegate: "the 1.5x/2.0x/3.0x + UV-MV matrix gate is tools/report/psnr3x_gate.py" — and that gate exists (cells A-E, 1.5x/2.0x/3.0x + UV) | none — coverage gap already delegated |
| 9f | "quality_guard B1 = 41.0 (line 99), 9 dB below V4's 50.0 (eval_verify.py:30)" | **TRUE, documented as intentional** | `quality_guard.py:99` `B1 = 41.0`; the docstring `:21-36` states B1 is a *regression floor vs the oracle output*, never V4, and V4 belongs to `psnr3x_gate`/`eval_verify`.  Anchor: measured worst healthy cell 41.85 → bar 41.0 (`:41-49`) | none — but see the measured psnr3x result below: post-fix A/C PASS V4, B/D/E fail only the 0.1 dB GT-delta (0.175-0.180) — pre-existing, unrelated to this round |
| T1 | Theory-5 sub-claim: fused `sr_swin` drops R9's `rh()` chains, `sr_M_seq` 41.91 dB vs oracle, stage9 X1 residue compounding | **PARTLY-TRUE (numbers verified; mechanism per docs)** | `quality_guard.py:53-59` carries exactly `sr_M_seq 41.91 | sr_M_static 57.13 | sr_L_seq 42.12 | sr_L_static 55.42`.  The stage9 concentration is `docs/notes/chain-leverage.md:453-469` (inject table: stage9 row min 94.27 dB = "the entire remaining SR M error … concentrated in stage9") and `k_sr_stage.md:1471-1495` (post-attention X1 f16 residual, named+priced).  Not re-derived end-to-end this round | none this round — it is the standing accuracy item (B1's anchor row), not a new fuzz finding |
| T2 | Harness modification table (5 rows: `--ffx` mode, disoc scene, `--ref` sharpness gate, q15/mvhi cells, `DLSS5_AMD_DUMP_DIR` in ffx_front) | **mixed — proposals, not defects** | `--ffx` mode: STALE (ffx1/ffx2 exist).  disoc scene: **accepted** (9d).  `--ref` gate: reasonable, needs oracle outputs for the synthetic scene (does not exist) → proposal.  q15/mvhi cells in quality_guard: needs input dirs under `build/qguard` that the referee run never produced → psnr3x_gate already covers 1.5x; mvhi cells already exist in the frozen runner.  `DLSS5_AMD_DUMP_DIR` in ffx_front: the S7/S8 dump machinery lives in the **shared** `record_bridge` (nvngx.cpp:684 `dump_probe_on`, :713 probe) so parts already run on FFX frames; the `collect_frame`-side frame dumps don't — and the marker is deliberately disarmed | disoc scene applied; rest recorded as proposals |

---

## What was fixed (host lane), commit + build + gate status

Commits on `game-integration` (pushed): **`ef32de5`** (this verdict + the raw analysis),
**`d24803e`** (claims 1-4 host fixes), **`7888d29`** (harness `--scene disoc`, claim 9d),
**`317a7d4`** (mvhi log one-shot).

| Fix | File:line | Build | Gate |
|---|---|---|---|
| Float quality-ratio division (claim 1) | `src/ffx/ffx_front.cpp:293-301` | `bash src/ffx/build.sh` ✓ | bit-identical at 2.0x/3.0x (integer ratios) |
| Preset L inference + `DLSS5_AMD_PRESET` override (claim 2) | `src/ffx/ffx_front.cpp:389-401, 447-465` | ✓ | ffx1/ffx2 cells stay on `'M'` (2.0x); 3.0x → `'L'` = the NGX/reference combo; `psnr3x_gate --suffix 2` cells D/E are the 3.0x-L bar |
| mv_scale isfinite/!=0 guard (claim 3) | `src/ffx/ffx_front.cpp:542-543` | ✓ | identity on the measured (1280,-720) path |
| mvhi from flag+dims on FFX (claims 4a/4b) | `src/ffx/ffx_front.cpp:556-576` | ✓ | mvhi=0 for in-game render-sized MVs → byte-identical |
| mvhi instead of refusal for output-sized MVs on NGX (claim 4d) | `src/pe/nvngx.cpp:1360-1389` + comment `:622-626` | `bash src/vk/build.sh` ✓ | harness MVs are render-sized → untouched branch |

**Restaging note for the MAIN session:** `bin64/dlss5-amd-ffx.dll` and `bin64/_nvngx.dll` are
managed by the main session only.  The new builds are in `build/package/` — the preset-M→L
change is behaviour-visible in-game (it is the point of the fix), so **the game needs the main
session to restage the FFX dll** before the next in-game comparison.  Nothing in the game dir was
touched by this pass.

### Gate results measured for this pass (2026-11-01 evening, all green)

| Gate | Command | Result |
|---|---|---|
| FFX dll + runner builds | `bash src/ffx/build.sh`, `bash src/vk/build.sh` | both exit 0 (`dlss5-amd-ffx.dll`, `_nvngx.dll`, 64 shaders) |
| quality_guard (no shader/runner regression) | `build/venv/bin/python -I tools/report/quality_guard.py` | **PASS**, exit 0 — sr_M_seq 41.91 / sr_M_static 57.13 / sr_L_seq 42.12 / sr_L_static 55.42, G1/G2/T1/B1 all ok (digit-identical to the docstring's fresh run ⇒ nothing moved) |
| psnr3x matrix (scored, no regeneration) | `build/venv/bin/python -I tools/report/psnr3x_gate.py --suffix 2` | A 1.5x-M PASS; C 3.0x-M PASS; B/D/E fail ONLY the 0.1 dB GT-delta (0.175/0.180) — pre-existing property of the current tree, untouched by bridge-only edits |
| ffx1 harness cell (1920x1080→3840x2160, 2.0x) | `run_harness.sh --ini tools/scenario/inis/ffx1.ini` | **PASS** g1 create=1 sr_slot=1, g2 12/12, g3 nz=1.0, g3b CONTENT PASS, g4 median 4.46 ms — log shows `(ratio 2x2) preset M`, i.e. ≤2.0x behaviour byte-equivalent to the old hardcode |
| **ffx30 harness cell (NEW**: 1280x720→3840x2160 through `dlss5-amd-ffx.dll`) | `run_harness.sh --ini ffx1.ini --w 1280 --h 720 --ow 3840 --oh 2160 --out build/optiharness/harness_ffx30` | **PASS** all gates — log: `dispatch: creating the chain 1280x720 -> 3840x2160 (ratio 3x3) preset L from .../ngx-weights/L` + `chain ready ... 36 pipelines` (the exact chain the in-game NGX/L surface built) — the fixed preset path proven end-to-end at the game's geometry, never run before this pass |
| harness build | `bash tools/optiharness/build.sh` | exit 0 (`opti_client.exe` with `--scene disoc`) |

---

## Shader-lane patch plan — NOT APPLIED (other agent owns `shaders/` this round)

### P1 — restore the temporal-candidate path (claim 6) — order: enc0 → dec0 → post

**P1-a `shaders/sr/sr_enc0.comp` — restore the mv_ero store (binding 7).**
- line 71, old: `// binding 7 (mv_ero) removed 2026-10-28: dead (sr_post binds but never reads it)`
  new: `layout(set = 0, binding = 7) writeonly buffer OutMvEro    { f16vec2 out_mv_ero[]; };`
- line 524, old: `// mv_ero removed (dead store; the mx tournament value above is still used for dhist sample 2)`
  new: `out_mv_ero[idx0]    = f16vec2(mv15x * (mvoffx + m.mx.x), mv15y * (mvoffy + m.mx.y));`
  (keep the dhist-sample-2 comment — `m.mx` stays used either way; mirror `sr_enc0L_in.comp:493`
  which already writes exactly this form)
- host: `src/vk/pipeline.cpp:1992` already binds `&mv_ero` in the M enc0 list — verify slot 7 is
  in that vector (L at `:1963` proves the pattern).

**P1-b `shaders/sr/sr_dec0.comp` — restore the factors output (binding 4).**
- line 59, old: `// binding 4 (factors) removed 2026-10-28: dead (see header)`
  new: `layout(set = 0, binding = 4) writeonly buffer OutFactors { u8vec4 out_factors[]; };`
- line 310, old: `// ---- factors: REMOVED (dead store + dead softmax; see header) ----`
  new (restore pre-`20919d4` form, from `git show 20919d4^:shaders/sr/sr_dec0.comp`):
  ```glsl
  float fsum = f16sum(fac0...);   // exact pre-20919d4 text comes from the parent commit
  out_factors[idx] = u8vec4(...);
  ```
  Get the exact old text with `git show 20919d4^:shaders/sr/sr_dec0.comp | sed -n '25,60p;295,330p'`
  — do NOT reinvent it.
- header comment lines 33-35: drop "factors are NOT computed …".

**P1-c `shaders/sr/sr_post.comp` — 3-candidate selection + bilinear mv_nondil + 2nd CR blend.**
- binding table lines 66-71: declare `InPrevMv` (6), `InMvEro` (7), `InFactors` (3) again.
- lines 467-475 (the `NOT DECODED` block): replace the single nearest lookup with:
  1. bilinear sample of `in_mv_nondil` through `mvntex` (texture, args +0x60 as in the ISA —
     R5 step 3 says `image_sample_lz`), i.e. stop using `in_mv_nondil[my*W+mx]`;
  2. scores accumulated per candidate in the spatial-filter pass (R5 step 2: "the same pass
     accumulates three scores, one per motion-vector candidate" — this belongs in the tap loop
     at `:408-440`, using the same `wE/wO` weights);
  3. rank → best position → existing CR fetch; then confidences
     `clamp(1.1*s - 0.05, 0, 1)`; if runner-up share > 1/255 (`0x3b808081`) run the same 5-tap
     CR at the second MV and mix (R5 step 5).
- spec source for the mix weights: `docs/notes/k_sr_post.md:157-161` (R5 steps 3-5) — the python
  model does NOT have it (see verdict 6f), so **do not port from `sr_post_model.py`**; port from
  the R5 probe notes + (ideally) new patchprobe measurements.

**Gate that must move to prove P1:** none of the current cells can (all three MV buffers are
equal on every fixture — verdict 6g).  Required first: a **disocclusion cell** — either the new
`opti_client --scene disoc` run (once a `--ref` exists) or a Blender seq with depth-separated
motion, scored PSNR(port, oracle-with-selection) — where the restored path must IMPROVE and a
no-op on A/B/C/D/E cells must be **bit-identical** (they are the control: if A-E move, the restore
changed something the oracle's own content says it should not have).

### P2 — LDS tile width for < ~1.4x ratios (claim 7) — latent, NOT this symptom

- `shaders/sr/sr_post.comp:130-131`:
  old:
  ```glsl
  #define ST_C 20
  #define ST_R 20
  ```
  new:
  ```glsl
  #define ST_C 27
  #define ST_R 27
  ```
  (27 = NeuralHub's `<15, 27>` instantiation constant, srref.cpp:521; LDS cost:
  3 × 27² × 8 B = 17.5 KB → still under the 32 KB limit; verify against `docs/notes/k_sr_post.md`
  staging record size before applying.)
- The clamp at `:340-341` and the unclamped reader at `:408/421` stay as-is once 27 covers the
  worst real ratio; alternatively add `bx = min(bx, ncol-1-kx)`-style guards — the width fix is
  the honest one.
- **Gate:** only a < 1.4x cell can prove it (none exists; the game never runs one).  1.5x cell A
  and 3.0x cells C/D/E must stay **bit-identical** (need 17/18 and ~10 ≤ 20 ⇒ no behaviour change
  above 1.4x is expected — if A moves, the LDS size change altered staging order and the change
  is wrong).

---

## How to read the running agent's static 3.0x cell + spatial error maps

The other agent's `build/psnr3x` static 3.0x cell and spatial error maps can directly confirm or
refute the disocclusion theory (claim 6h) and the "input conventions" theory:

1. **Where the error energy sits.** If `|port − ref|` (or `|port − GT|`) hotspots are concentrated
   in narrow bands along **depth/motion silhouettes** of still geometry → supports the missing
   candidate-selection / second-history theory (6d) on *this* content.  If the error is
   **spatially uniform across all still texture** (flat map over grass/stone, no edge bias) →
   refutes 6h and points at preset/weights or accumulation (claim 2, theory T1).
2. **Still vs moving in one static cell.** A "static" cell under jitter has no moving objects;
   if the port's per-pixel error there is dominated by **history-accumulated** frames (error
   grows monotonically frame 0 → 11 while the oracle's stays flat), that is the jitter/history
   alignment signature — check the frame column: a rising `port/GT` deficit with flat
   `ref/GT` = our temporal loop drifting on stills, independent of MV candidates.
3. **Cross-read against psnr3x cells C vs D vs E (measured this pass, post-`80aa2f7`):**
   C (3.0x M) min 55.51 / mean 56.56 / Δmax 0.056 → PASS V4.  D/E (3.0x L, pixel/UV) min 52.41 /
   mean 55.52 / Δmax 0.180 → fail only the 0.1 dB GT-delta.  The port is *numerically excellent*
   at 3.0x on both presets vs its own oracle — so whatever is in-game is **not** reproduced by
   these cells: the delta between "sharp XeSS / fuzzy ours" lives in in-game inputs or preset
   choice, not in the kernels these cells exercise.
4. **The pending S6 probe** (moving-camera scene, `maxAbs` of the MV dump) remains the one
   measurement that decides whether the game's MVs carry real motion (verdict 5c).

---

## Ranked judgment — what actually explains "STILL objects fuzzy, unevenly, XeSS sharp at 3.0x"

1. **Preset M at 3.0x on the FFX path (Gemini claim 2 — proven TRUE and LIVE, now fixed
   host-side).**  The game's own NGX surface chose L for the identical geometry; real DLSS via SL
   asks for L; our FFX path silently ran the M weights and M kernels for 44 227 gameplay frames.
   This is the only claim in the document proven to be *actually happening in the user's session*.
   Caveat stated honestly: psnr3x shows port-M ≈ oracle-M at 3.0x (C passes V4), so M is not
   "broken" — but it is *not the model DLSS5 was asked to run at this ratio*, and M-vs-L differ
   in weights, level tables and post splat (3x3 vs 4x), which is exactly the kind of thing that
   reads as "different, uneven sharpness per texture class".  **Needs: restage the FFX dll, then
   an in-game A/B (`DLSS5_AMD_PRESET=M` reverts).**
2. **Input-side temporal conventions still unmeasured in-game (Gemini 5c, left open by S6).**
   Static-under-jitter cells are sharp (57.13 dB) with the *same* chain, so the code can produce
   sharp stills when fixture conventions hold; the in-game conventions (jitter handshake, MV
   content during pans — REPORT.md:915 still `pending`) have never been re-measured after the
   MV.Scale fix.  "Uneven among still objects" fits content-dependent temporal accumulation:
   objects with good history accumulate, objects near screen motion or resets don't.  **Next
   measurement: the S6 probe in a moving-camera scene, or the spatial maps above.**
3. **XeSS as the control.** XeSS receives the same game MVs and is sharp ⇒ if our MV decode or
   units were wrong, XeSS would misreproject too — unless our *interpretation* (scale, jitter
   phase, reset cadence) differs from XeSS's.  This is an argument for suspecting our side's
   handling rather than the game's MV content, i.e. rank 1 (preset/pipeline params) over "the
   game sends zero MVs".
4. **Missing 3-candidate selection + 2nd history blend (claim 6).**  Real divergence from the
   .co, worth doing (patch plan P1), but its physics only touch *moving/disoccluding* edges and
   it is provably a no-op where candidates coincide (stills).  **It does not fit this symptom**
   and Gemini's priority-1 placement is wrong on symptom direction as well as provenance
   (never ported, not deleted on Oct 28).
5. **`<2.0x` tile overflow (claim 7)** — arithmetically real below ≈1.3x, **provably not
   exercised**: the game runs 3.0x (`W15/W = 4.5`, oracle picks `<12,16>`, tile need ≈10 ≤ 20)
   and the 3.0x cells score 52-57 dB port/ref.  Plainly: **this theory cannot explain the user's
   symptom.**
6. **dq() dither removal (claim 8)** — deliberate, guard-covered, effects would be uniform
   (both still and moving), small; not the symptom.
7. **FFX plumbing claims 3/4/5** — true as code (now fixed/zero-impact), but every one of them
   is provably inert in this game: guard on a non-zero scale, mvhi on render-sized MVs,
   exposure at 1.0 on both surfaces.  None explains the symptom.

**Plain statement:** of Gemini's five theories, one (preset M at 3.0x) is proven live in the
user's session and fixed; one (missing MV-candidate selection) is proven real but provably
cannot produce "stills worse than moving"; one (<2.0x tile overflow) cannot execute in this
game at all; one (dq) is guard-covered; and the FFX plumbing bundle is true-but-inert here.
If the restaged preset-L build does not resolve the fuzziness, the next suspect is not in this
document at all — it is the still-unmeasured in-game temporal handshake (S6's pending MV-magnitude
probe and the jitter/reset cadence), because the identical chain renders sharp stills on every
fixture cell that mirrors the oracle's conventions.