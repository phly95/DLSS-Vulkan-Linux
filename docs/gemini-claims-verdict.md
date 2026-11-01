# Gemini's claims — verdict

**Date:** 2026-11-01 · **Branch:** `game-integration` · **Scope:** every claim in the inventory,
re-verified from primary sources (the cited line was *read*, never trusted; where the citation was
wrong the corrected location is given).

**Verdict vocabulary**

| verdict | meaning |
|---|---|
| `TRUE` | the claim is supported by the cited (or corrected) evidence, and it is still true today |
| `PARTLY-TRUE` | the substance is real but part of the claim — a number, a scope, an attribution or a citation — does not hold |
| `FALSE` | the cited evidence does not support the claim |
| `STALE` | was true, now superseded: fixed, re-measured or replaced by a later row/commit |

Nothing below was changed in order to make a claim pass. Bars, thresholds and frozen files are
reported as measured; where a bar was moved it is named explicitly, with the measurement that
justifies the move.

---

## §0 The decisive finding — it reframes LAYER 2

The oracle-parity matrix (`tools/report/psnr3x_gate.py` + `build/psnr3x/`, the other agent's work)
showed that the **41–44 dB ceiling on SR moving sequences was NOT primarily stage9 / mode-2 X1
residue**. The real cause was `shaders/sr/sr_enc0.comp` and `shaders/sr/sr_enc0L_in.comp` baking the
level-15 geometry ratios as the literals **`3.0` / `1` / `3`** — values correct only for the original
256×144→512×288 fixture where `W15/W == 3`, i.e. only at render **2.0×**. At 1.5× (true `W15/W =
2.25`) every history position and MV store was scaled **1.33× too far**; at 3.0× (true `4.5`) scaled
to **0.67×** of its correct value. That is exactly why the oracle-parity matrix passed at 2.0× and
diverged from frame 1 — the first history consumer — at 1.5×/3.0× while frame 0 always passed, and
why feature 15's position mask zeroed the right quarter of the frame from frame 1 on there.

**Fixed today in commit `80aa2f7`** ("sr: push the real level-15 ratios — stop baking the 2.0x
fixture's 3.0 into enc0", 2026-11-01 15:12:59): `src/vk/pipeline.cpp` now pushes the ratios from
the geometry (mirroring srref's `.f2(0x9c, fW/f15w, …)`) and both shaders read them instead of the
literals.

Measured effect (from the `80aa2f7` commit message, i.e. `build/psnr3x/logs/curve_*`):

| cell | parity mean before → after | max \|GT delta\| before → after | frames >0.1 dB before → after |
|---|---|---|---|
| **A** 1.5× M | 43.45 → **57.31** | 4.24 → 0.028 | 11/12 → **0/12** (V4 PASS) |
| **C** 3.0× M | 41.75 → **56.56** | 9.30 → 0.056 | 11/12 → **0/12** (V4 PASS) |
| **D** 3.0× L | 41.91 → **55.52** | 8.17 → 0.180 | 11/12 → **6/12** (V4 FAIL on the GT delta only — the port is *better* than the oracle vs GT there) |
| **B** 2.0× M | 58.17 (unchanged) | — | 5/12 at 0.175 (pre-existing, untouched) |

**Reading:** the scale-dependence, not stage9, was the dominant term at 1.5×/3.0× — frame 0 always
passed and frame 1 (the first history consumer) diverged, which is the signature of a baked
geometry ratio, not of a neural-residue. What remains is a **separate, much smaller gap** of
~0.175–0.180 dB on cells B and D. The stage9/+40.45 dB injection number (claim C5) stays a valid
*measurement*, but it is no longer the explanation of the seq ceiling.

---

## §1 LAYER 1 — inherent DLSS behaviour

| # | claim | verdict | evidence (corrected where needed) | action taken / why none |
|---|---|---|---|---|
| C1 | Recurrent feedback poisoning; memset of history on frame 4 (`HANDOFF:228-239`, `REPORT:895`, commit `9ff6ba9`) | **TRUE** — with a citation fault on the hash | `HANDOFF.md:228-239` (reset-frame host memset of `prev_mvec`/`hidden`/`ihist` on every reset frame; the oracle resets at f0 **and** f4, so the port wiped state the oracle keeps; effect SR M 45.79→55.64 / SR L 44.51→57.09, both V4 PASS). Second, independent poisoning path: `tools/scenario/REPORT.md:895` (S3-BYTES: `±inf` MV sentinels → f16 `inf` → history propagation → the user's blocky/shimmer/no-AA). **Hash fault: `9ff6ba9` does not exist in this repository** (`git cat-file -t 9ff6ba9` → *fatal: Not a valid object name*). The commit that matches the description and effect is **`bb791c9`** ("pipeline: drop the reset-frame host memset of prev_mvec/hidden/ihist — SR M 45.79 → 55.64, SR L 44.51 → 57.09 mean, both V4 PASS", 2026-10-28 00:13). The dead hash is cited in exactly 10 places that are not mine to edit — `docs/PORT-STATUS.md:44,52,65`, `docs/notes/chain-leverage.md:222,271,432,451`, `HANDOFF.md:25,228,236` (0 hits in `tools/scenario/REPORT.md`). | None — the bug is fixed and the numbers are recorded. Hash discrepancy recorded here; the docs that carry it belong to other agents/sessions. |
| C2 | FP8/FP16 quantization sensitivity incl. RADV rounding (`HANDOFF:66-70`, `k_sr_stage:198-205`) | **TRUE** (RADV half measured; the `k_sr_stage` citation is precision-regime evidence, not a quantization sweep) | `HANDOFF.md:66-70` — measured 2026-10-26: `float(float16_t(x))` does **not** round on RADV (8059 of 8064 values came back unrounded) and `packHalf2x16` disagrees with RNE on ~half the values; only a 16-bit memory store/load rounds → every rounding point must be an explicit integer RNE (`rh()`), and "the other port shaders have not been audited for this". `docs/notes/k_sr_stage.md:198-206` — measured with `STAGE_STATS=1`: real-weight logits run to **+33.4** with 64/1024 above `logit > 11`, `exp` saturates and `pe` pins at 256, vs the `sAB` probe's ≤11.5; "a formula fitted on sAB is fitted on a regime the real kernel never occupies". | None — kernel-work item, already owned by the SR kernel line (`rh()` rule in `sr_stage.comp`; the audit of the remaining shaders is an open note). |
| C3 | No fallback for UV-vs-pixel MVs **or** non-linear colour (`docs/notes/wronggpu-re.md:206-210`) | **PARTLY-TRUE** — non-linear colour `TRUE`, UV-vs-pixel MVs `STALE` | *Non-linear colour:* `docs/notes/wronggpu-re.md:205-210` — the FFX API delegates it (`FFX_UPSCALE_ENABLE_NON_LINEAR_COLORSPACE: AMD FSR handles it`) and the runtime string says why (`the game reports non-linear color; the network expects linear color`, spec 20). Our own front-end has no conversion path either: `src/ffx/ffx_front.cpp:176-177` — `NON_LINEAR_COLORSPACE refused (chain wants linear colour)`. *UV-vs-pixel MVs:* the citation says nothing about MV units (line fault). That half is **STALE**: S6 commit `e30a1ca` added `MV.Scale.X/.Y` consumption in the gather decode with the legacy `MV.Scale` X-fallback, a non-finite/0 → 1.0 guard, and the `--mv-units uv` harness pressure (`src/pe/nvngx.cpp:461-478`, `tools/scenario/REPORT.md:911`). | None — S6 already landed; the non-linear-colour half is a known, documented limitation of the FFX seam, not a regression. |

---

## §2 LAYER 2 — the shader port

| # | claim | verdict | evidence | action taken / why none |
|---|---|---|---|---|
| C4 | Non-default gather formats decode to 4.6–18.7 dB (`PORT-STATUS:100-105,115-119`) | **TRUE** | `docs/PORT-STATUS.md:100-105` — `sr_M_col4/6` 18.40, `sr_M_col5` 16.03, `sr_M_ldr1/2/3` 18.74, `rr_F_fmt_a` 7.98, `rr_F_fmt_b` 7.15, `rr_F_fmt_c` **4.61** → the quoted 4.6–18.7 range is exact (4.61 min, 18.74 max). `docs/PORT-STATUS.md:115-119` — failure class 3, "**CATASTROPHIC (4-19 dB, GT-deltas 33-42)** … the port renders plausible but WRONG images from non-default input bindings/formats … the largest named gap in the project". | None by me — still an open accuracy class (work-order item 1 for whoever owns the gather-input path). |
| C5 | SR moving sequences capped at 41.7–44.4 dB **because of stage9 mode-2 X1 residue**; `chain-leverage:453-470` says oracle injection of launch 12 gives **+40.45 dB** | **PARTLY-TRUE** — the cap and the +40.45 measurement are real; the **attribution is superseded** (§0) | Cap: `docs/PORT-STATUS.md:95-97` — `sr_M_seq` 41.74/42.90, `sr_L_seq` 42.02/42.81, `rr_F_seq` 42.26/**44.43** (the 41.7–44.4 span, though the 44.4 end is the RR cell, not SR — SR's own span is 41.7–42.9). Injection: `docs/notes/chain-leverage.md:451-470` — the re-measured table at the reset-fixed chain, `stage9` row `+40.45 / +43.58`, "the entire remaining SR M error … is concentrated in stage9's own segment (oracle launch 12)". Re-frame: §0 — `80aa2f7` shows the seq ceiling at 1.5×/3.0× was the baked `3.0` ratio in `sr_enc0`/`sr_enc0L_in`; a residual ~0.175–0.180 dB remains on cells B/D. | Recorded in §0. No bar moved; `chain-leverage`'s +40.45 stays as a measurement (it is another agent's/session's doc). |
| C6 | Unsanitized nearest-neighbour alpha in `sr_downsample` causes screen flashes; sanitize "in the working tree, uncommitted"; `nvngx.cpp` calls causation unproven / pending an S7 row | **PARTLY-TRUE** (suspect real, causation still unproven) and the *"uncommitted"* part is **STALE** | Suspect + instrumentation: `src/pe/nvngx.cpp:858-872` — "S7 FLASH hunt (2026-11-01): … `sr_downsample.comp` feeds the OUT plane's alpha as a nearest-neighbor fetch of a low-res input alpha, unsanitized and history-free. **If** the game composites out-alpha, junk frames READ as fullscreen flashes" + the tag/measure instrument. Causation is still open: there is **no S7 result row** in `tools/scenario/REPORT.md` (only S8's references at `:930,933-935`, and the observe-only commit `a1da855`). *Sanitize status:* **not** in the working tree any more — it is **committed in `80aa2f7`** ("`shaders/sr/sr_downsample.comp`: sanitise only NON-FINITE alpha to 1.0 (HOST-RE "alpha is 1.0 for formats without one"); finite alpha is byte identical, cell-A gate unchanged"). Working tree is clean for `shaders/`. | None by me — the sanitize landed with `80aa2f7`; the flash *causation* row is still owed by the scenario/S7 line (not my file). |
| C7 | Run-to-run drift originates at `t_act[4]` (`docs/OPEN.md:15`) | **TRUE** | `docs/OPEN.md:15` — two runs of the untouched pre-refactor `runner.cpp`: 12/12 files differ, frame 0 "4870 of 33177600 floats differ (0.0147%), max abs diff 0.0519753"; buffer hashing localises the first divergence to `t_act[4]` (sr_stage's second output at frame 0), everything after follows; **RR is stable** (0/33177600, 0/204 history hashes). Rule recorded there: never claim SR bit-reproducibility, judge SR by PSNR. | None — open item, deliberately unfixed (the `sr_stage` out2 path "waits for its own measurement"). |

---

## §3 LAYER 3 — the bridge

| # | claim | verdict | evidence | action taken / why none |
|---|---|---|---|---|
| C8 | Truthy `0xBAD00000` enum bug (`nvngx.cpp:1858-1868`) | **STALE** (was TRUE; fixed) | The bug: `NVSDK_NGX_Result` is an enum where Success `0x1` **and** Fail `0xBAD00000` are both non-zero, so the original `if (outcome) return Fail;` never fired — a refused frame fell into `record_bridge` with a null `VkImage`, then reported SUCCESS. Still documented in-tree at `src/pe/nvngx.cpp:1884-1895`, with the fix at `:1896-1897` and its Vulkan twin at `:2006-2010`: `if (!ngx_ok(frame_ok)) … return frame_ok;`. Fix commit: **`a4d5fc1`** ("port: accept the in-game formats (MV RG32F, Depth D32S8) and **fix the silent-success guard**"). The cited range `1858-1868` has drifted +~26 lines (later `DXGI 28` / diag edits; `a2efc80` documents that shift). | None — already fixed. |
| C9 | Port reads `"MV.Scale"` instead of `"MV.Scale.X"/"MV.Scale.Y"`; **claimed fixed by S5** (`nvngx.cpp:462-484`) | **PARTLY-TRUE** — the bug and its proof are real; S5 was *observability only*, **S6 is the fix** | Proof: `tools/scenario/REPORT.md:902` (P-S5, commit `834daf1`): "the port read the **non-existent `MV.Scale`** (my earlier H3 probe ⇒ the `mv_scale=1.0000` readings were defaults, not app values)". Current code `src/pe/nvngx.cpp:461-478` reads `MV.Scale.X` (legacy `MV.Scale` as X fallback) + `MV.Scale.Y` independently with the `isfinite/0 → 1.0` guard and the one-shot `[S6] mv scale=` line. Consumption in the chain (the part S5 did *not* do): `REPORT.md:911` (P-S6, commit `e30a1ca`) — `sr_gather` push 88→96 B, `sr_gather*.comp` convert every raw-MV decode branch by `clamp(san(raw)·scale, …)` before the f16 cast; harness `--mv-units uv`. Bit-exactness: `REPORT.md:913` (S6-PARITY, A vs B = 0 differing words of 33,177,600; control B vs C = 8.84 % differ, i.e. the harness discriminates). | None — both halves already landed (`834daf1` + `e30a1ca`). |
| C10 | Missing DXGI **16 / 19 / 28** (`check CURRENT ngxd_dxgi_to_vk at nvngx.cpp:1137`) | **STALE** — all three are present today | Current table `src/pe/nvngx.cpp:1143-1170`: `DXGI_FORMAT_R32G32_FLOAT` (16) → `VK_FORMAT_R32G32_FLOAT`; `19/20/21` → `VK_FORMAT_D32_SFLOAT_S8_UINT` (resolved exactly as vkd3d-proton resolves them); `DXGI_FORMAT_R8G8B8A8_UNORM` (**28**) → `VK_FORMAT_R8G8B8A8_UNORM`. The three are pinned against silent enum drift by `static_assert`s at `:1129-1141`, each with the in-game line that measured it. **DXGI 28 was fixed today in `c1ed0ed`** ("bridge: accept DXGI 28 (R8G8B8A8_UNORM) as RR code 5 for DiffuseAlbedo/SpecularAlbedo"), the referenced line number having moved +6 from the claim's `1137`. Still deliberately refused (documented at `:1172-1182`): D24/D16 depth — "refuse until an in-game line names it". | None — 28 fixed by `c1ed0ed` (that commit is not mine). |
| C11 | Missing Streamline keys / callbacks (`nvngx.cpp:1496-1636`) | **STALE** — present today | `src/pe/nvngx.cpp:1596-1644` `prime_params()` publishes the whole Opti `InitNGXParameters` recipe: `SuperSamplingDenoising.Available`/`FeatureInitResult`, `SuperSampling.Available` + the `#\x01` enum alias, `NeedsUpdatedDriver`, `SuperSampling.NeedsUpdatedDriver`, `MinDriverVersion 10.10`, `Snippet.*`, both optimal-settings callbacks + the `#\x3e` alias, `DLSSGetStatsCallback`, `Sharpness`, `MV.Scale/Offset.X/Y`, `PerfQualityValue`, `SizeInBytes`, `FreeMemOnReleaseFeature`, the six RR preset hints. Why each was needed is documented inline (`:1597-1616`: an *absent* key returns bare `0xBAD00000` → `sl.dlss_d`/`sl.dlss` die with "DLSSDContext cannot run"). Landed in **`5a588bd`** (core callbacks + `PopulateParameters_Impl`), **`894ce54`** (RR capability key), **`3110d8b`** (`SuperSampling.Available` — the in-game black-3D root cause). The `Init_Ext` / `Init_ProjectID` exports OptiScaler's proxy GetProcAddress-es are at `:1481-1506`. Claimed range `1496-1636` ≈ today's `1486-1644`. | None — already landed. |
| C12 | ReBAR debug readback + double `frame_idx` increment, **both fixed by S8, commit `6a337d5`** | **TRUE** | Commit **`6a337d5`** exists: "ffx: kill the in-game 47 s/frame — frame_idx parity + flash-read stride (S8)". ReBAR cost measured at `src/pe/nvngx.cpp:868-870`: "a FULL read of the 4K output (33.2M halfwords of `DEVICE_LOCAL|HOST_VISIBLE` staging = uncached ReBAR, ~600 ns/load) took **~47 s PER FRAME**"; fix = strided read (`stride = npx/4096`, ~4096 spatial samples). Double increment: `tools/scenario/REPORT.md:934` (S8-PERF-FIX) — "removed the duplicate `++c->fs.frame_idx` — `nvngx.cpp:818` is the single owner"; cause chain at `REPORT.md:933` (S8-PERF-CAUSE). | None — fixed and harness-verified. |

---

## §4 LAYER 4 — harness blindness

| # | claim | verdict | evidence | action taken / why none |
|---|---|---|---|---|
| C13 | `optiharness` / D1 assert only `OUT_NZ >= 0.10`, so a **scrambled** image passes (`optiharness README:67-71`, `REPORT:444-512`) | **TRUE** — and now **closed on the optiharness side** | At `HEAD`, `tools/optiharness/README.md:71`: "**g3 non-black**: every `out_<i>.f32` has `nz >= 0.10`". The D1 matrix's own rule: `tools/scenario/gate_cell.sh:217-225` — `awk '/^OUT_NZ /{ … if (a[2]+0 < 0.10) bad++ }'` → `G-OUT PASS "all frames nz>=0.10"`; nothing else inspects frame content. `tools/scenario/REPORT.md:444-512` is the D1 digest whose per-run evidence is exactly `client.log (OUT_NZ + EVAL_MS/GPU_EVAL_MS)`. Both failure modes satisfy `nz`: a frozen replay and a scrambled frame each sit at `nz = 1.000000` (measured, below). | **FIXED (my work)**: new `tools/optiharness/content_gate.py` (sub-gate **g3b**) wired into `tools/optiharness/run_harness.sh` right after g3, joining `HARNESS RESULT`. Not frozen, not oracle-based: two statistics per consecutive pair — `mean \|Δ\|` and `frac(Δ > 0.05)` — with NOT-FROZEN floors and NOT-SCRAMBLED ceilings. Measured distributions + all-mode results: `tools/optiharness/README.md` "g3 content". *Not claimed:* `tools/scenario/gate_cell.sh` (the D1 script proper) still asserts `nz` only — it is not my file. |
| C14 | `synth_mv` produces only clean, finite, constant pans (`README:344-354`) | **TRUE** — and that is exactly the blindness | At `HEAD`, `tools/optiharness/README.md:344-354` (now `:407-411` after the g3b section was inserted): "Nothing else changed in the byte stream: Colour (gradient+checker, per-frame), **MotionVectors (+2 px, +0.25 px pan)** and Depth (y-plane / 0.5 clear) generators are untouched", framed as "*frame-independent* like `synth_mv`/`synth_depth` — no `rand`, run-to-run input variance 0". Code: `tools/optiharness/opti_client.cpp:378-389` — a uniform `mvx/mvy` written to every texel (`2.0/0.25`, or `2/w / −0.25/h` under `--mv-units uv`); no `rand` anywhere (`:409-410`). The real stack carries neither: `REPORT.md:895` — "**Why D1/A06 missed it: the oracle corpus never carried non-finites**" (86.9 % `±inf` sentinels in-game). | None — the limitation is now *named and mitigated* rather than removed: `--mv-units uv` (`e30a1ca`) added the unit axis, and g3b (C13) added a content axis. Deliberately **not** made non-deterministic: `no rand` is a documented invariant (zero run-to-run input variance). |
| C15 | `tools/verify` covers only 1.5×/2.0×, **never 3.0×** (`REPORT:900-901`) | **TRUE** | `tools/scenario/REPORT.md:901` — "master-2 proved A06/row:204 numbers are **Millisecond TIMINGS** … ⇒ **H1 CONFIRMED: 3.0× has NEVER been PSNR-validated** (tools/verify PSNR cells cover only 1.5×/2.0× for M and L: `M{1.5,2.0}, L{1.5,2.0}`)". Confirmed against the frozen tree: `tools/verify/run_all.sh:19` ("V3 the matrix: SR M and L (moving, static, **ratio 1.5**)"), `:171-176` (the only cells: `sr_{M,L}_{seq,static,q15}` — `q15` = 1280×720→1920×1080 = 1.5×, `static` = 1920×1080→3840×2160 = 2.0×), `:163` (`gen_frames.py (ratio 1.5)`). Caveat for future readers: `tools/verify/eval_verify.py:30` also carries a `3.0` — `V8_RATIO` — but that is the **frame-time** ratio (port median ≤ 3× oracle median, `:132-134`), not a render scale. | None by me — `tools/verify/` is FROZEN (read/execute only). The 3.0×-capable gate lives **outside** it at `tools/report/psnr3x_gate.py`, committed as **`6b50a14`** ("report: psnr3x_gate — non-fail-fast V4 gate over the 1.5x/2.0x/3.0x+UV matrix"). |
| C16 | `quality_guard`'s **38 dB** bar sits below V4's 50/55 (`quality_guard:19-20,47-52`) | **TRUE** — **FIXED in this pass** | At `HEAD`: `tools/report/quality_guard.py:19-20` — "B1 backstop: every frame PSNR(port, ref) >= **38.0 dB**"; `:47` — `G1, G2, T1, B1 = -0.20, -0.50, -0.20, 38.0`. V4's authority, unchanged and read-only: `tools/verify/eval_verify.py:30` — `V4_MIN_DB, V4_MEAN_DB, GT_DELTA_DB = 50.0, 55.0, 0.1`. Defect quantified: 38.0 sat **3.85 dB** under the measured worst healthy cell (41.85), so the guard could lose 3.85 dB on it and still print `B1:ok`. | **B1 raised 38.0 → 41.0**, anchored 0.85 dB under the measured worst healthy cell (41.85, `sr_M_seq`) of the cells passing every other bar — the bar was **not** fitted to any run. Added a doc block that separates this script's bars (B1/G1/G2/T1) from the frozen V4 bars (50/55/0.1, authority `tools/verify/eval_verify.py:30`), and sharpened it: B1 shares V4's *first* term (per-frame PSNR vs the oracle output) but at a deliberately looser threshold (41.0 < 50.0) and without V4's mean/GT-delta terms — a green B1 is never a V4 pass. Fresh full run measured and pasted into the docstring, see §6. |

---

## §5 LAYER 5 — the external stack

| # | claim | verdict | evidence | action taken / why none |
|---|---|---|---|---|
| C17 | 86.9 % `±inf` MV sentinels on menus (`REPORT:895`) | **TRUE** (and already fixed) | `tools/scenario/REPORT.md:895` (S3-BYTES): `dlss5-amd-dump_mv.bin` 1920×1080 RG32F = **86.9 % non-finite channel words** (≈14.3k `−inf` / 42.8k `+inf` per 57k sample; first pixels `(-inf, inf)`); finite magnitudes pixel-scale `x=-484..511`; cause chain `sl.common` carries `consts.motionVectorsInvalidValue`, `sl.dlss` never transfers ⇒ raw sentinels into `g_mv` → f16 `inf` → history propagation. Fixed by S4: `REPORT.md:903` (D5.3-S4-GATE) — "4147200/4147200 channel words finite (**100.0 %**) … vs S3's 86.9 % non-finite". | None — fixed at S4 (`6c8aed3` staging + the sanitizer). |
| C18 | "99.9 % zero MVs in gameplay" (`src/pe/nvngx.cpp:941`, `REPORT` pending row ~915) | **NOT PROVEN / STALE-pending** — the report still marks it pending, and the two in-tree numbers disagree | The comment: `src/pe/nvngx.cpp:942-948` — "the S6 frames answered 'no motion' (**99.9 % exact zeros** during pans)", followed by the ORDERING-vs-NEVER discriminator it added. The measured gameplay row: `REPORT.md:910` (S6-CAUSE) — dumps f10/f60 at 1280×720: **100.0 % finite, 96.9 % pixels NONZERO**, `max|v| = 8.25e-5` raw ⇒ ≈0 motion *as chain-plane pixels*. **99.9 % exact zeros and 96.9 % nonzero are not the same measurement**, and neither was taken under a moving camera. The discriminating capture is still open: `REPORT.md:915` — "pending: user relaunch + DLSS toggle and **a MOVING camera scene**: S6 probe stats for frames 300/900/1800 decide the UV-unit hypothesis". Per the report, **that capture has not happened**, so no capture has ever PROVED zeros during a moving camera. | None — resolving it needs the in-game capture (user/game lane, off-limits here: the game dir and its processes are out of scope for this pass). Recorded so the claim is not repeated as established. |
| C19 | WinVerifyTrust + Turing arch gate block RR (`REPORT:849-852`) | **TRUE** (both measured; both now routed around) | `tools/scenario/REPORT.md:849` — D5 proof cell: `getNGXFeatureRequirements ngxResult failed 0xbad00005` ×2, and the triangulation "returned by the GENUINE core only when adapter arch ≥TU100; at arch 0x130 the core blesses init but SL's static TU100 floor zeroes the masks. **No arch value satisfies both** … all 8 `nvngx_dlss.dll` on box are Turing-stamped ⇒ no newer-stamped core lever exists on this box." `REPORT.md:850-851` — P-N3, the 2-byte NOP of the WinVerifyTrust pre-load gate at `sl.common.dll:0x8c15e`, **measured FAILED and REVERTED** ("the 2-byte NOP … breaks SL's own load of its host library"). `REPORT.md:852` — P-SIGN-D: a properly code-signing-cert'd port **passes** WinVerifyTrust (`Loaded L"…harness\nvngx.dll" … native`, `g5 … c1..c6 all ok`, `SL-RR-ENABLE rc=0x00000000 ctx=loaded`); `REPORT.md:853` — the signed core was then installed game-dir (G1). | None — resolved by signing, with full provenance rows. |
| C20 | OptiScaler `fsr21` fallback / `DxgiFactoryWrapping` / AutoExposure 4-frame warmup (`REPORT:287-293`, `:20`, `:921`) | **TRUE** (three independent, measured facts) | *fsr21:* `REPORT.md:291-295` — "`xess`/`fsr31` are announced correctly, then Opti logs `CheckUpscalerFiles libxess.dll not found!` and `ChangeFeature changing backend to fsr21 … init successful for fsr21` … the companion libs live in the game's `bin64` but not in `$STAGE` ⇒ **NEED N5**" (later closed by N5's companion staging; a second, independent instance at `REPORT.md:921`: `Dx12Upscaler=ffx` has no RTTI case and silently substitutes `FSR2FeatureDx12_212`). *DxgiFactoryWrapping:* `REPORT.md:20` (P1) — `[Spoofing] DxgiFactoryWrapping=true -> false`, because the wrapped factory's `QueryInterface` does not expose `IDXGIVkInteropFactory1/2` + `IDXGIVkInteropAdapter`. *AutoExposure warmup:* `REPORT.md:921` (S8-STOCK-FACTS b) — with AutoExposure off and no ExposureTexture, `FSR31Feature::Evaluate` sets `autoExposure+changeBackend` and returns **without dispatching** (frame 0), then `ChangeFeature` eats frames 1–3 (1000 ms sleep + recreate + init) — "**measured OUT_NZ 0-3 = 0.000000 before the fix, 1.000000 ×12 after** (`[InitFlags] AutoExposure=true`)". | None — all three are recorded and handled in their own lanes (N5 staging, the ini lever, the staged ini flag). |
| C21 | Proton 11 hangs (`docs/OPEN.md:14`) | **TRUE** (was) — **FIXED** | `docs/OPEN.md:14` — every process started by `proton run` on Proton 11.0 / Proton-Experimental blocks before loading anything (`anon_pipe_read`, `do_epoll_wait`); timed probe `timeout 90 … --nonsense-arg` → `RC=124` for Experimental and 11.0, `RC=1` for Proton 10.0. Fix recorded in the same row: **`774fd8a`** (operator-authorized exception to the frozen rule) — `run_all.sh` probes newest-first with a 20 s health probe instead of `sort -V \| tail -1`, pasted selection run in `build/proton-fix/sim/`, `PROTON=<path>` skips it. | None — fixed and documented. |

---

## §6 The gate I own, validated in this pass

### 6.1 `tools/report/quality_guard.py` — full run to completion

`build/venv/bin/python -I tools/report/quality_guard.py`, 2026-11-01 15:13–15:28, whole tree as of
`80aa2f7` (shader `spv` built 15:11–15:12, `vk_runner` 15:12 — one consistent build for all four
cells; no cache hits, every cell actually re-ran).

| cell | verdict | G1 (mean dGT ≥ −0.200) | G2 (min dGT ≥ −0.500) | T1 (mean dT ≥ −0.200) | B1 (min vsNH ≥ **41.0**) |
|---|---|---|---|---|---|
| `sr_M_seq` | **PASS** | +2.841 ok | +0.403 ok | +2.935 ok | **41.91** ok |
| `sr_M_static` | **PASS** | −0.042 ok | −0.106 ok | −0.036 ok | **57.13** ok |
| `sr_L_seq` | **PASS** | +2.557 ok | +0.306 ok | +2.722 ok | **42.12** ok |
| `sr_L_static` | **PASS** | +0.014 ok | −0.082 ok | +0.010 ok | **55.42** ok |
| **overall** | **`QUALITY GUARD: PASS`** (exit 0) | | | | worst cell **41.91**, margin **0.91 dB** |

**B1 justification (unchanged):** the bar is `41.0`, anchored **0.85 dB below the measured worst
healthy cell of 41.85** (`sr_M_seq` in the pre-raise run) — the worst cell among those passing
every *other* bar. It is a regression floor, not a quality certification: the pre-raise 38.0 sat
3.85 dB under that anchor, i.e. it could lose 3.85 dB on the worst cell and still print `B1:ok`.
The fresh run confirms the anchor still holds (worst healthy cell 41.91, 0.91 dB over the bar) and
**the bar was not moved** to fit it.

**Docstring correction (measured reality wins):** the draft claimed
`sr_M_seq 41.85 | sr_M_static 57.13 | sr_L_seq 42.11 | sr_L_static 55.42`. Fresh measurement:
**41.91 | 57.13 | 42.12 | 55.42**. Two static cells reproduce to the digit; the two seq cells moved
by +0.06 and +0.01 dB (the draft's run predated `80aa2f7` reaching the shared tree). The
docstring now carries the corrected numbers, the per-cell G/T/B detail, the tree state, and an
explicit note that no bar moved. (`sr_L_seq` also recovered 33.12 → 42.12 across the same span,
G1 −1.983 → +2.557 — another session's accuracy fix, not this bar.)

### 6.2 `tools/optiharness/content_gate.py` — measured distributions

Calibration dir `build/optiharness/out` (12 frames, 1920×1080 → 3840×2160, 11 pairs), plus two
fixtures built from those frames. All numbers below are this script's own `--report` output,
2026-11-01:

| population | mean_i min..max | frac_i min..max | verdict |
|---|---|---|---|
| healthy (real run) | 0.00779327 .. 0.15699814 | 0.03951296 .. 0.33365218 | **PASS** |
| healthy, second dir (`sl-baseline_out`, byte-identical frames — `md5 out_0/out_5` equal) | identical to the digit | identical | **PASS** |
| frozen (`out_1..11` → symlinks to `out_0`) | 0.0 .. 0.0 | 0.0 .. 0.0 | **FAIL** (both floors) |
| scrambled-A (per-frame RGB permutation, `default_rng(20261101)`) | 0.19201234 .. 0.19498892 | 0.73412584 .. 0.74969425 | **FAIL** (`frac` ceiling) |
| scrambled-B (IID draws from the pooled 12-frame distribution) | 0.21419674 .. 0.21428631 | 0.82184992 .. 0.82211018 | **FAIL** (`frac` ceiling) |

**Thresholds (`content_gate.py:68-70`: `0.001 / 0.40 / 0.01 / 0.55`) are sane and were NOT moved.**

* `min mean > 0.001` — healthy 0.0078 = **7.8×** over; frozen exactly 0.
* `min frac > 0.01` — healthy 0.0395 = **4.0×** over; frozen exactly 0.
* `max mean < 0.40` — healthy 0.157 = **2.5×** headroom. Deliberately *not* the discriminator:
  neither scramble trips it (0.195 and 0.214 both sit under 0.40).
* `max frac < 0.55` — healthy 0.334 = **1.65×** headroom; both scrambles sit **above** it
  (0.734 and 0.822). The bar lies between them: 1.65× over the healthy maximum and 1.34× under
  the weaker scramble (midpoint 0.534), so it keeps one step of extra room on the healthy side.

All modes exercised standalone (rc): healthy [0] PASS · frozen [1] FAIL `FROZEN` · scrambled-A/B
[1] FAIL `SCRAMBLED` · `--expect-fail` frozen [0] PASS / healthy [1] FAIL · `--ceilings-only`
scrambled-A [1] FAIL · `--frames 1` [0] `SKIP reason=needs>=2-frames` · missing dir [2]
`CONTENT_GATE ERROR reason=missing:…` (fail closed). The calibration block, the fixture recipes and
this mode table are in `tools/optiharness/README.md` ("g3 content") and in the module docstring.

**g3b end-to-end inside `run_harness.sh`** — `bash tools/optiharness/run_harness.sh --out
build/optiharness/out_g3b`, 2026-11-01 15:43, post-`80aa2f7` shaders:

```
g3 non-black  : PASS   threshold 'every frame nz >= 0.10', viol=0/12, min nz=1.000000
g3b content   : PASS   CONTENT_GATE PASS n=11 mean_min=0.00778715 mean_max=0.15700708
                       frac_min=0.03947752 frac_max=0.33365138 … (not frozen; not scrambled)
HARNESS RESULT: PASS  (opti_client rc=0, exe summary RESULT=PASS)
```

A second independent healthy sample: every statistic is within 1e-5…1e-3 of the calibration run
(margins unchanged). The **FAIL** side of the wiring was also exercised end-to-end, on the two runs
preceding it: `g3b content : FAIL  CONTENT_GATE ERROR … reason=missing:…/out_0.f32` printed in the
summary and `HARNESS RESULT: FAIL` followed — exit 2 maps to `g3b FAIL`, i.e. fail closed.

Those two runs failed for a reason worth recording: a *relative* `--out` reached the PE as the
drive-relative `Z:build\…` (`winpath()` is a pure text transform), so the exe opened nothing and
exited `rc=1` after eval #1 with `out_*.f32`, `client.log`, `eval_ms.csv` and `summary.txt` all
absent while the port log — written into the CWD by the dll — still looked healthy. `run_harness.sh`
now absolutizes `$OUTD` right after `mkdir`, before anything uses it.

---

## §7 Citation faults found while verifying

Recorded because they will otherwise be cited again:

1. **`9ff6ba9` does not exist in this repository.** `git cat-file -t 9ff6ba9` → *fatal: Not a
   valid object name*. The commit described by `HANDOFF.md:228-239` (reset-memset fix) is
   **`bb791c9`** ("pipeline: drop the reset-frame host memset of prev_mvec/hidden/ihist", 2026-10-28
   00:13, subject and effect matching the docs exactly). The dead hash appears 10 times —
   `docs/PORT-STATUS.md:44,52,65`, `docs/notes/chain-leverage.md:222,271,432,451`,
   `HANDOFF.md:25,228,236` — none of which are my files, so it is reported here rather than edited.
2. **Line numbers drift with every bridge edit** — `nvngx.cpp:1858-1868` → `1884-1897`;
   `nvngx.cpp:1137` → `1143-1170`; `nvngx.cpp:1496-1636` → `1486-1644`; `optiharness README:67-71`
   → `:71-78`, `README:344-354` → `:407-411`. Always re-anchor by symbol/text, not by number
   (`a2efc80` already records one such +1/+18 shift).
3. **Two different "3.0"s.** `eval_verify.py:30`'s `V8_RATIO = 3.0` is a *frame-time* ratio; it is
   not evidence that `tools/verify` covers a 3.0× render cell (it does not — C15).
4. **`quality_guard`'s 38 dB bar is not V4's bar.** They share their first term (per-frame PSNR vs
   the oracle output) but V4's threshold is 50.0 and V4 adds a mean ≥ 55.0 and a per-frame
   |GT delta| ≤ 0.1; `quality_guard` enforces none of those. Saying "the guard is 12 dB below V4"
   compares two different contracts — which is exactly why the doc block in `quality_guard.py`
   now spells out both.