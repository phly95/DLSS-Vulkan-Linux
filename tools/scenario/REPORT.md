# tools/scenario — D1 scenario replication + D2 RR root-cause + D3 in-game assurance

Owner: orchestrator session `ses_e6fd51294ffey5ZMwuIL2arg9H` (nested agent). This report is the
running provenance + result log for the DLSS/RR replication deliverables (D1-D4).

Hard rules in force: `tools/verify/**` FROZEN, `tools/optiharness/**` owned by the sibling
harness agent (do not edit), backups before every binary/ini/XML touch, every claim carries a
file:line or a log quote, the permanent Steam launch line must be the final state:

```
WINEDLLOVERRIDES=winmm,dxgi=n,b PROTON_FORCE_NVAPI=1 DLSS5_AMD_WEIGHTS=/home/deck/DLSS-Vulkan-Linux/build/ngx-weights %command%
```

---

## Provenance log (file-level interventions, newest first)

| # | time (EDT) | file touched | backup | change | why |
|---|---|---|---|---|---|
| P1 | 2026-10-31 00:31 | `bin64/OptiScaler.ini` | `bin64/OptiScaler.ini.bak-prelever1` (cp -p) | `[Spoofing] DxgiFactoryWrapping=true -> false` | top-level ranked lever #1: wrapping mode's `WrappedIDXGIFactory7::QueryInterface` does not expose `IDXGIVkInteropFactory1/2` + `IDXGIVkInteropAdapter`, which dxvk-nvapi's `NvAPI_Initialize` needs (see /tmp/opencode/dxvknvapi_readme.md) |
| P2 | 2026-10-31 00:31 | `bin64/dxgi.dll.optiscaler-off` -> `bin64/dxgi.dll` | (rename, no data change) | restore OptiScaler into the load path | baseline (no-Opti) verification run of 23:17 was finished |

Prior interventions (before this session's compaction) are recorded in
`bin64/runs/levers-round1.txt` and the `.bak-*` files in `bin64/`:

- `OptiScaler.ini.bak-spoof`, `.bak-diag`, `.bak-prethread` — earlier lever rounds.
- falsified already: `[NvApi] OverrideNvapiDll=false`, `[Spoofing] Registry=true`.
- `_nvngx.dll.bak-{preinit,diag,f32,mvfix}` — port DLL iteration history.

## Key evidence (unchanged, from archived runs in `bin64/runs/`)

```
runs/OptiScaler.log.slgate-fail-20261030:144  getSystemCaps] Enumerating up to 8 adapters ...
runs/OptiScaler.log.slgate-fail-20261030:153  NVAPI failed to initialize, please update your driver...
runs/OptiScaler.log.slgate-fail-20261030:176  sl.dlss requires driver 512.15.0 or newer - detected 0.0.0 - sl.dlss will be disabled
runs/OptiScaler.log.stallsite-20261030:459    Failed to initialize NGX, any SL feature requiring NGX will be unloaded and disabled
```

## Nested subagent reports (this session)

- `/tmp/opencode/d2-evidence.md` — D2 causal chain: game uses Streamline feature 1001
  `kFeatureDLSS_RR` via plugin `sl.dlss_d.dll` (no `sl.dlss_rr` anywhere); stock `sl.dlss_d`
  needs driver >= 535.15 in SystemCaps -> `supported=false` -> `Ignoring plugin 'sl.dlss_d'`;
  Opti refuses to spoof DLSSD (`Streamline_Hooks.cpp:448-452`); our port already answers
  `GetFeatureRequirements` with `Supported` for every feature id (`src/pe/nvngx.cpp:1379-1395`).
- `/tmp/opencode/opti-build-feasibility.md` — OptiScaler v0.9.4 is MSVC/vcxproj-only
  (BUILDABLE-AFTER-TOOLCHAIN-INSTALL; recommended route fork + GitHub Actions `build.yml`),
  `isNvidia()` init branch at `dllmain.cpp:1749-1756`, and a ranked file-level alternative:
  `[NvApi] OverrideNvapiDll=true` makes `LoadNvApi()` serve `bin64\fakenvapi.dll` (already
  present) -> fakenvapi reports driver 99999 + count=1 without init and sets
  `isUsingFakenvapi()` so `Streamline_Hooks.cpp:415` writes `driverVersionMajor=999`.

## Lever experiments

### Lever 1 (P1): `DxgiFactoryWrapping=false`
- Applied in the GAME ini at 00:31 (backup P1 above), `dxgi.dll` restored into the load path.
- **In-game verification NOT yet obtained**: the Steam client died between runs
  (`bootstrap_log.txt`: `[01:18:22] Shutdown`, and again after the 01:22 start), and
  `steam-runtime-steam-remote` then reports `E: Steam is not running: No such device or address`,
  so `steam://rungameid/3321460` / `-applaunch` cannot reach a client. Console log proves the
  required launch line is intact and was what the 23:17 baseline run used:
  `console_log.txt [01:10:51] Game process removed ... "WINEDLLOVERRIDES=winmm,dxgi=n,b
  PROTON_FORCE_NVAPI=1 DLSS5_AMD_WEIGHTS=... %command%"`.
  Launch line in `userdata/49277565/config/localconfig.vdf:2894` UNCHANGED (final-state requirement
  met — never edited this session).

### Harness lever matrix (`run_harness.sh --sl`, nested agent `ses_e6f8f698effeGaYRt1CehOtjUX`)
Report: `/tmp/opencode/lever-runner.md`; logs `/tmp/opencode/slrun_{C0,L1,L2,L2b,L3}.log`.

| lever | getSystemCaps | verdict |
|---|---|---|
| C0 control | `NVIDIA driver 999.99` (arch 0x130) | caps already OK headlessly: the harness has **no** `DxgiFactoryWrapping` key, i.e. the Opti default (false) |
| L1 `DxgiFactoryWrapping=false` | byte-identical to C0 | no-op in the harness (key absent = false already) ⇒ **L1 is only a GAME-side lever**, where the inode value is `true` |
| L2 `OverrideNvapiDll=true` | `LoadNvApi Load fakenvapi.dll` but harness stages no `fakenvapi.dll` → `nvapi64.dll loaded from system!` | inert unless fakenvapi is staged |
| L2b/L3 (fakenvapi staged) | `fakenvapi initialized successfully`, `Adapter 0 architecture 0x190` (was 0x130) | proves the fakenvapi path works; driver number unchanged (already 999.99) |

g5 FAILs identically in every cell: c2 `Failed to initialize NGX` x2, c4 `Missing NGX context` x2,
and **RR stays shut**: `Ignoring plugin 'sl.dlss_d' ... adapter mask 0x0`.

### NEW ROOT-CAUSE EVIDENCE for RR (harness `build/optiharness/harness/OptiScaler.log:222-250`)
```
[01:42:25.373651] commonEntry.cpp:1093[getNGXFeatureRequirements] ngxResult failed 0xbad00002
[01:42:25.376551] dlss_dEntry.cpp:210[updateEmbeddedJSON] DLSSD feature is not supported.
                              Please check if you have a valid nvngx_dlssd
```
`0xBAD00002 = NVSDK_NGX_Result_FAIL_PlatformError` (`nvsdk_ngx_defs.h:94`). Three facts narrow
whoever answers it:
- our port answers `GetFeatureRequirements` with `Supported` for every feature id and **exports it**
  (`src/pe/nvngx.cpp:1379-1395`, `objdump -p _nvngx.dll`), but its once-per-process log line never
  appears → **nobody asks our DLL**.
- the genuine `nvngx_dlss.dll` / `nvngx_dlssd.dll` ARE in the game dir and export
  `NVSDK_NGX_D3D12_GetFeatureRequirements` (objdump) — NVIDIA-only payloads; SL's message
  ("check if you have a valid nvngx_dlssd") is exactly what it says when *that* query fails.
- `NVSDK_NGX_D3D12_Init_with_ProjectID` (SL's init call, `commonEntry.cpp:1499-1540`) exists in the
  game dir only in Opti's `dxgi.dll` — the genuine model DLLs and our port (which has
  `NVSDK_NGX_D3D12_Init_ProjectID`, no `with_`) do NOT export it.

⇒ **next experiment (file-level, cheapest)**: in the harness stage, replace the two genuine
`nvngx_dlss*.dll` symlinks with copies of our `_nvngx.dll` (originals are symlinks, nothing lost)
and re-run `--sl`; if `0xbad00002` disappears and `sl.dlss_d` joins the execution order, apply the
same substitution to `bin64/` (backs up first) and then verify in-game.

### D2 causal chain (current best, nested agents' reports below)
game hides RR because (1) in-game `DxgiFactoryWrapping=true` lets Opti's wrapped factory starve
dxvk-nvapi of `IDXGIVkInterop*` → `NvAPI_Initialize` fails → caps `0.0.0` → `sl.dlss`
(needs 512.15) AND `sl.dlss_d` (needs 535.15) are both gated off; and (2) even with caps fixed
(harness proves 999.99), SL's `getNGXFeatureRequirements` fails with `PlatformError` — answered, as
far as the evidence goes, by the genuine NVIDIA `nvngx_dlssd.dll` rather than our port — so
`sl.dlss_d` gets `adapter mask 0x0` and is dropped. Our port already returns `Supported` and has no
arch/driver minimum (`MinHWArchitecture=0`).

---

## D4 — consolidated report + master handoff (MASTER-2, 2026-10-31 02:2x EDT)

**Takeover provenance.** The previous master session `ses_e6fd51294ffey5ZMwuIL2arg9H` died at
02:05 on `provider.quota` (`/tmp/opencode/failover_runbook.md:5,7`). Replacement:
`ses_e6f7044eeffeSSITmFI4a0g4Mb`, running on `opencode/mimo-v2.6-flash-free`. All of its on-disk
work was recovered **from this file and from disk** — the provenance table (P1/P2), the lever matrix,
the nested-agent reports and the D2 chain above were already written when it died. Nothing above
this line was edited or removed (append-only ownership rule).

### Deliverable status at 02:2x EDT

| D | state | artifact / evidence |
|---|---|---|
| **D1** scenario replication | **design complete, 0/33 cells executed** (no number exists yet — none is invented) | `D1-DESIGN.md` (factors, methodology, run plan), `D1-GATES.md` (gate catalogue + harness mapping), `D1-MATRIX.csv` (33 cells: 7 READY, 19 BLOCKED-N1, 7 BLOCKED-N2), `run_cell.sh`, `gate_cell.sh` (both `bash -n` clean; refusal paths smoke-tested: unknown cell rc=2, BLOCKED cell rc=3) |
| **D2** RR/black root cause | running with the EXECUTOR; **L4 in flight**: staged harness `nvngx_dlss.dll` replaced by our port copy 01:56 to test whether `0xbad00002 FAIL_PlatformError` clears and `sl.dlss_d` (feature 1001) joins | this file: the `OptiScaler.log:222-250` quote above; `/tmp/opencode/d2-evidence.md`; `tools/scenario/levers/LEVER-RUN1.md` |
| **D3** in-game assurance | lever `DxgiFactoryWrapping=false` **applied** to the game ini (P1) but **not verified in-game**: Steam client was down (`bootstrap_log.txt [01:18:22] Shutdown`, `E: Steam is not running`); fakenvapi staged and proven in-harness (`fakenvapi initialized successfully`, caps 999.99) | P1/P2 rows; `L2b/L3` rows above; launch line verified UNCHANGED (`localconfig.vdf:2894`) |
| **D4** consolidated report | **this section** — index in `tools/scenario/README.md`, updated as evidence accrues | `README.md` |

### Needs (nothing below is unblocked by this session alone)

| id | need | owner |
|---|---|---|
| N1 | ~~`--ini FILE`~~ → **LANDED** (`3795535`: `run_harness.sh:71` parse, `:85-87` pre-stage validation, `:127` staging). 19 former BLOCKED-N1 cells are READY. | harness owner — done |
| N2 | ~~stage `nvngx_dlssd.dll`~~ → **LANDED** (`3795535`: `run_harness.sh:102` preflight die, `:117` symlink). 7 former BLOCKED-N2 cells are READY; sizes verified by `G-STAGE`/`G-PORT-RR` from a deref'd manifest. | harness owner — done |
| N3 | **STILL HELD**: released execution slot (prefix 3321460 free, no live harness — shared `$STAGE`, `run_harness.sh:46` — game exited, executor quiet; in-game session live as of 02:24) | top-level / EXECUTOR |

### Next lever round — HYPOTHESES ONLY (analysis; nothing here was measured in this session)

Ranked by cost; each remains a single-variable experiment, backup-first, one released slot each.

- **H-1 / L5 — export-name gap (cheapest, port-side).** SL's init call is
  `NVSDK_NGX_D3D12_Init_with_ProjectID`; Opti's `dxgi.dll` exports it, the genuine NVIDIA model DLLs
  do not, and our port exports `NVSDK_NGX_D3D12_Init_ProjectID` (no `with_`) — quoted above under
  "NEW ROOT-CAUSE EVIDENCE". Proposal: add the `Init_with_ProjectID` (+ D3D11/Vulkan siblings,
  mirroring the existing alias block at `src/pe/nvngx.cpp:1327-1345`) as an export alias and re-run
  `--sl`. **Deprioritise if L4's file-level substitution clears `0xbad00002`** (that would prove the
  caller's target, not the name, was the problem).
- **H-2 / L6 — architecture spoof.** fakenvapi reports AD100 in its own source, yet caps read
  `Adapter 0 architecture 0x190` (`levers` L2b/L3 rows above; `LEVER-RUN1.md:91` reads `0x130` for
  the control, so the two rows disagree — resolve that discrepancy first, it may be two different
  binaries in the load path before designing any new lever). RR's minimum architecture question
  (Ada+) hangs on this number, and `DXVK_NVAPI_DRIVER_VERSION` is already falsified as a lever here.
- **H-3 / L7 — OptiScaler fork patch (source is patchable per user).** Opti refuses to spoof DLSSD
  (`Streamline_Hooks.cpp:448-452`) while the client's `isNvidia()` branch is at `dllmain.cpp:1749-1756`;
  MSVC-only build → fork + GitHub Actions `build.yml` route is documented in
  `/tmp/opencode/opti-build-feasibility.md`. Only worth doing after H-1/H-2 fail, because it changes a
  third-party build we then have to maintain.
- **H-4 / L8 — in-game confirmation of P1** (`DxgiFactoryWrapping=false`): requires Steam to be
  restarted (EXECUTOR's slot), and is currently the only blocked step between "caps fixed in the
  harness" and "caps fixed in the game".

Decision rule (unchanged): one variable per lever, `.bak-*` before any binary/xml/ini touch,
ship-line never edited, every claim re-cited to `file:line` or a log quote.

---

## D1 wave 1 — MEASURED (2026-10-31 02:49:15–03:10:55 EDT, slot released by top-level 02:48)

21 runs total (6 tier-A cells × 3 reps + close-out sentinel rep4 + E01/E02 (`--sl`) + B03 × 3).
All logs kept per rep under `tools/scenario/runs/<cell>/rep<k>/` (`OptiScaler.log`,
`dlss5-amd-ngx.log`, `client.log`, `eval_ms.csv`, `summary.txt`, `g5_evidence.txt`,
`stage_manifest.txt`). Frames 1..59 per run (frame 0 excluded repo-wide).
**Median of the 3 run-medians** is reported; sentinel A03 defines the noise floor.

### Noise floor (sentinel A03, 4 reps, dlss / Quality / 2560x1440 -> 3840x2160)

| rep | EVAL_MS med / p95 | GPU_EVAL_MS med / p95 | gates |
|---|---|---|---|
| rep1 | 4.4711 / 6.1431 | 10.6174 / 13.0206 | PASS 7/7 |
| rep2 | 4.3936 / 6.2472 | 10.6106 / 12.9222 | PASS 7/7 |
| rep3 | 4.4760 / 7.8884 | 10.5804 / 12.7740 | PASS 7/7 |
| rep4 (close-out, after 5 cells) | 4.4479 / 7.8100 | 10.6455 / 12.8483 | PASS 7/7 |

ranges: EVAL 0.0824 ms, GPU 0.0651 ms ⇒ per G-EFFECT the reporting threshold is
**2× = 0.165 ms (EVAL), 0.130 ms (GPU)**. Close-out rep4 sits mid-range ⇒ no drift over wave 1.

### Tier A: dlss × 6 quality levels (RR=off), 3 reps each — all 18 runs PASS 7/7 gates

| cell | render -> out | EVAL med (p95) ms | GPU med (p95) ms | gates |
|---|---|---|---|---|
| A01 DLAA | 3840x2160 -> 3840x2160 | 4.4663 (7.5120) | **19.7137** (20.7421) | PASS ×3 |
| A02 Ultra Quality | 2954x1662 -> 3840x2160 | 4.4171 (8.6545) | **13.3211** (15.8435) | PASS ×3 |
| A03 Quality *(sentinel)* | 2560x1440 -> 3840x2160 | 4.4711 (6.2472) | **10.6106** (12.9222) | PASS ×3 |
| A04 Balanced | 2228x1254 -> 3840x2160 | 4.4367 (7.4179) | **8.9988** (11.3753) | PASS ×3 |
| A05 Performance | 1920x1080 -> 3840x2160 | 4.4357 (8.2525) | **8.1175** (9.1530) | PASS ×3 |
| A06 Ultra Performance | 1280x720 -> 3840x2160 | 4.3522 (6.8823) | **5.5119** (6.9321) | PASS ×3 |

Every run: `G-ART PASS OUT_NZ x60`, `G-STAGE nvngx_dlss.dll=58956912`, `G-IDENT 'Creating new
dlss upscaler'x1 fsr_fallback=0 nvngx_loaded=1 Running on Nvidia`, `G-PORT ok=60/60 fmt col=0
mv=1 dep=0`, `G-PORT-SR no feature=13`, `G-OUT min nz=1.000000`, harness rc=0.

**What the numbers say (and don't):**
- **GPU is resolvable and monotone in render resolution**: 19.71 → 13.32 → 10.61 → 9.00 → 8.12 →
  5.51 ms across DLAA→UltraPerf. Every adjacent gap is ≥ 0.879 ms, i.e. ≥ 6.8× the 0.130 ms
  threshold. The biggest absolute savings: DLAA→Quality −9.10 ms; the smallest, Balanced→Performance
  −0.88 ms (the 4K render is already GPU-bound before the render resolution drops).
- **CPU-side (EVAL) shows NO resolvable effect of the quality preset**: cell medians span
  4.3522–4.4711 = 0.119 ms < 0.165 ms threshold ⇒ reported as *below the noise floor*, no ranking.
  Consistent with a fixed per-evaluate overhead dominating (Opti entry + copy setup), stated as
  observation, not proven mechanism.
- **One outlier observed, kept, not smoothed**: A05 rep2 p95 = 54.3534 ms (a single ~54 ms frame)
  while its median = 4.4380 — p95 over 59 samples is dominated by one hitch; the table uses the
  median-of-medians, so it is unaffected. Recorded so nobody "cleans" it later.

### Tier E (`--sl`, Streamline booted headless) — D3-grade evidence, captured at first hit

**E01 (dlss / Quality / RR off / `--sl`) — G-RR-ASSURE FAIL (3 of 6 subchecks), everything else
PASS.** Harness g5: `c1_getSystemCaps_nvapi=ok  c2_no_failed_NGX=bad(x2)  c3_driver=ok
c4_no_missing_ctx=bad(x2)  c5_sl.dlss_loaded=ok`. Full lines, `runs/E01/rep1/g5_evidence.txt`:

```
:16 commonInterface.cpp:290[getSystemCaps] NVIDIA driver 999.99
:17 commonInterface.cpp:315[getSystemCaps] Adapter 0 architecture 0x130 implementation 0x2 revision 0xffffffff - bit 0x1 - LUID 0.1010
:g5 | nvapi: ... sl-requirements: driverDetected=999.99.0 driverRequired=512.15.0
:42 pluginManager.cpp:907[mapPlugins] Loaded plugin 'sl.dlss' - version 2.14.1.98614dad6 PRODUCTION - id 0 - priority 100 - adapter mask 0x1
:47 pluginManager.cpp:907[mapPlugins] Loaded plugin 'sl.dlss_d' - version 2.14.1.98614dad6 PRODUCTION - id 1001 - priority 100 - adapter mask 0x0
:68 pluginManager.cpp:1090[loadPlugins] Ignoring plugin 'sl.dlss_d' since it is not supported on this platform   [gate counts 6 lines matching `Ignoring plugin 'sl.dlss_d'` OR `adapter mask 0x0`]
:83 commonEntry.cpp:1748[slOnPluginStartup] Failed to initialize NGX, any SL feature requiring NGX will be unloaded and disabled
:114 dlssEntry.cpp:923[slOnPluginStartup] Missing NGX context - DLSSContext cannot run
```

**This is the in-game doom trio reproduced headlessly — while SR itself is fully healthy**
(60/60 port evals OK, `HARNESS RESULT: FAIL (opti_client rc=0, exe summary RESULT=PASS, g5=FAIL)`).
The clean implication, and the strongest new D3-grade fact of the session: **the caps gate can be
GREEN (driver 999.99 ≥ 512.15, architecture present) and `Failed to initialize NGX` / `Missing NGX
context` / `sl.dlss_d` exclusion still occur.** So the `0.0.0`-caps fix that the in-game
`DxgiFactoryWrapping=false` lever targets is *necessary but not sufficient* — the remaining root is
downstream of caps, matching H-1' (Opti's own `HookNgxApi`/`GetFeatureRequirements` surface, L4).
E01 timing (SR + SL, 1 rep): EVAL 4.4764 / p95 5.3243, GPU 10.4249 — vs sentinel 4.4711 / 10.6106:
EVAL +0.0053 ms (below the 0.165 floor ⇒ no effect claim permitted; and G-EFFECT forbids claims
across a FAILing cell anyway). Gate counts: 6 `Ignoring plugin 'sl.dlss_d'` / `adapter mask 0x0`
matching lines, 2× `Failed to
initialize NGX`, 2× `Missing NGX context`.

### Tier B RR (`Dx12Upscaler=dlssd`, `--ini inis/dlssd.ini`) — **BLOCKED(N4), reclass 03:11**

B03 × 3 and E02 × 1, identity each time: `G-IDENT 'Creating new dlssd upscaler'x1 fsr_fallback=0
nvngx_loaded=1`. Port side, `runs/B03/rep1/dlss5-amd-ngx.log`:

```
[    614 ms] RR 2560x1440 -> 3840x2160: creating the frame chain from /home/deck/DLSS-Vulkan-Linux/build/ngx-weights/rr
[  14042 ms] RR preset F 2560x1440 -> 3840x2160: chain ready in 13428 ms, 28 pipelines, weights .../ngx-weights/rr
[  14046 ms] D3D12 CreateFeature: id=1 feature=13 preset=F 2560x1440 -> 3840x2160 [diag-v1]
[  14137 ms] eval #1 FAIL: id=1 feature=13 preset=F n=3 slot=GBuffer.Normals why=GetD3D12Resource returned null for this key ctx=[no ID3D12Resource]
```
Opti echoes `_EvaluateFeature result: BAD00000` / `Upscaling done: false` ×60.

⇒ **RR create + RR chain build WORK headlessly through our port** (feature 13, preset F, rr weights);
the only blocker was the harness binding none of the port's **four** mandatory RR inputs
(`src/pe/nvngx.cpp:1133-1136` = `GBuffer.Normals` + `DLSS.Input.DiffuseAlbedo` +
`DLSS.Input.SpecularAlbedo` + `GBuffer.Roughness`, fetched at `:1152-1161`) ⇒ **need N4**
(D1-DESIGN §6). *Correction 04:1x: this paragraph first said "GBuffer.Normals/GBuffer.Roughness"
only — 2 of 4; the true set is the four above, equal to the frozen referee's `kRrInputs`
(`tools/verify/ngx_client/main.cpp:91-98`).* Gates then changed to `BLOCKED` (never FAIL)
with precedence kept so a real FAIL still wins; refusal ms (EVAL ~2.49, GPU ~0.002) are excluded
from all tables. E02 additionally failed G-RR-ASSURE exactly as E01 (same three subchecks).
N4 landed after wave 1, so these cells get re-run below (§ "D1 wave 3 — RR tier"), with two gate
changes first: `fmt col=0 mv=10 dep=0` for feature 13 (`nvngx.cpp:1177`), and the four-input
binding assertion.

### Wave 2 — upscaler type axis (C/D), 03:53:36–04:11:22 EDT, + fsr22 sentinel 04:1x

24 runs: C01/C02/C03 ×3, D06-D10 ×3, C02 rep4 (fsr22 sentinel). Honouring the single-runner rule
required waiting out the N4 harness agent 3× (`REFUSE: another run_harness.sh is live`, 60 s back-off
×3 before C01's first run) — the retry logic did exactly its job.

**Rule fixes that these runs forced (each measured, not speculative):**

1. **Port log is not universal.** `dlss5-amd-ngx.log` exists only when `_nvngx.dll` runs; for Opti
   internal backends its absence is by design ⇒ G-ART no longer requires it (else every type-axis
   cell would read "G-ART FAIL missing dlss5-amd-ngx.log", which is what C01-C03 first printed) and
   G-PORT / G-PORT-SR print `n/a` instead of a vacuous PASS off an absent file.
2. **Backend swap after a successful create (the identity trap).** `xess`/`fsr31` are announced
   correctly, then Opti logs `[W] Config::CheckUpscalerFiles libxess.dll not found!` and
   `FeatureProvider_Dx12::ChangeFeature changing backend to fsr21 … init successful for fsr21`
   (`runs/C01/rep1/OptiScaler.log`, same in all 3+3 runs) — the companion libs live in the game's
   `bin64` but not in `$STAGE` ⇒ **NEED N5**. G-IDENT now detects `changing backend to != cell
   backend` ⇒ `BLOCKED(N5)`, G-TIME marks those ms as *not the axis* and excludes them. This is
   exactly the substitution the old fsr-fallback grep could not see (it watches `falling back to FSR`,
   which in Opti's `ChangeFeature` domain is never printed).
3. **Warmup window for Opti-internal backends**: frames 0..3 black in **all 18** xess/fsr* runs,
   then `nz=1.000000` through frame 59 ⇒ G-OUT allows *≤5 leading black frames and 0 afterwards* for
   those backends, and reports `warmup=K`. DLSS cells keep the strict all-frames rule (they never
   used it: `min nz=1.000000` from frame 0).

**BUT the logged missing lib quoted by G-IDENT is cosmetic**: it is the *first* `Can't find…`
(`amd_fidelityfx_denoiser_dx12.dll`), not the deciding one — the deciding evidence is the
`changing backend to fsr21` line itself. Recorded here so nobody edits the message into a claim the
log does not support.

**fsr22 — valid measurements (Opti-internal, unaffected by N5). Median of 3 run-medians →**

| cell | render -> 4K | EVAL med ms | GPU med ms | gates |
|---|---|---|---|---|
| D06 DLAA | 3840x2160 | 0.0941 | **2.5593** | PASS ×3 |
| D07 UltraQ | 2954x1662 | 0.0949 | **2.1114** | PASS ×3 (rep2 EVAL p95=41.0394 outlier, logged) |
| C02 Quality *(sentinel)* | 2560x1440 | **0.0946** | **1.7158** | PASS ×4 (rep4 0.0929 / 1.8829) |
| D08 Balanced | 2228x1254 | 0.0864 | **1.7553** | PASS ×3 |
| D09 Performance | 1920x1080 | 0.0869 | **1.6753** | PASS ×3 |
| D10 UltraPerf | 1280x720 | 0.0872 | **1.5202** | PASS ×3 |

fsr22 noise floor (C02 ×4): EVAL range 0.0023 (⇒ threshold 0.0046), GPU range 0.2344 (⇒ threshold
**0.469 ms** — fsr22's own run-to-run spread is far wider than dlss's 0.065). Applying G-EFFECT:
GPU **DLAA vs Quality 0.844 is resolvable**; UQ↔Q↔Balanced↔Perf↔UltraPerf adjacent gaps (0.04–0.40)
are **below the floor → no ranking** for them. EVAL: 0.0946 (Q/UQ) vs 0.0864–0.0872 (Bal/Perf/UPerf)
= 0.0074–0.0082 > 0.0046 ⇒ the CPU side does separate the two res-groups, by ~8 µs.

**xess / fsr31 → BLOCKED(N5) ×6 runs — no number may be quoted for those cells.** What WOULD have
been quoted (excluded now): EVAL med ~0.103 / GPU med ~1.67–1.93 — those are fsr21's, not the axis'.

**Cross-backend observation (Quality res, 2560x1440 → 3840x2160):**

| path | EVAL med (CPU) | GPU med |
|---|---|---|
| dlss → our port (A03) | 4.4711 ms | 10.6106 ms |
| fsr22 (C02) | 0.0946 ms | 1.7158 ms |
| ratio | **47 ×** | **6.2 ×** |

vs the fsr22 noise floors (0.0046 / 0.469) these gaps are enormous ⇒ the *cost difference is real*.
Two honest caveats, both recorded so the number is not overclaiming: (i) the paths are not like-for-
like — our evaluate performs its own input copies + record + output copy around the chain, Opti's
FSR path does neither; (ii) where the cost sits on OUR side is **not yet root-caused**: the existing
`--direct` A/B (`README:275-277`) puts EVAL ≈ 4.3 ms with OptiScaler removed, i.e. the CPU number
belongs to the port, not to Opti. Candidate mechanisms (per-frame copies / a sync inside
`record_bridge`) are hypotheses for a later lever round — **not measured here**.

**Wave-2 scoreboard:** fsr22 ×5 cells (15 runs) PASS + sentinel C02 rep4 PASS; xess/fsr31 cells
C01/C03 (6 runs) → BLOCKED(N5); still unrun: D01-D05 (xess) + D11-D15 (fsr31) = N5-blocked,
B01/B02/B04-B06 = N4-blocked, E03 (N1-ready, not scheduled).
`out_*.f32`: purged automatically by `run_cell.sh` after gating (0 files on disk; `tools/scenario/runs`
stays ≤ 6 MB).

### Wave 3 — RR tier after N4 + the type axis after N5 (04:2x–09:35 EDT)

Commits in between: `9fc35a5` (N4 = four RR input slots) and `b685fcf` (N5 = stage the upscaler
companions). 55 runs this wave: B01-B06 ×3 (B03 appended as rep4-6), 12 xess/fsr31 cells ×3
(C01/C03 appended), sentinels C01 rep7 + C03 rep7, E02 rep2.

**Gate fixes the runs forced, both measured:**
- **`mv` is per-feature**: feature 13 prints `fmt col=0 mv=10 dep=0` (`nvngx.cpp:1177`
  `fp.fmt_mv = rr ? 10 : 1`), so the hardcoded `mv=1` failed every healthy RR cell until
  `EXPECT_FMT` became backend-dependent (`MV_EXPECT=10` for `dlssd`). Harness g1/g2 had already
  learned the same thing (`run_harness.sh:242-249`).
- **name-vs-key false FAIL**: my four-input assertion grepped `GBuffer.Roughness`, but the port logs
  `Slot.name` = `Roughness` (`nvngx.cpp:1136` pairs `{"GBuffer.Roughness","Roughness"}`) ⇒ all six
  B cells FAILed on a slot that was bound all along (`eval #1 OK … n=7 … [6 Roughness …]`). Fixed to
  match either spelling **plus** require `n=7` (3 common + 4 RR) on every RR OK line.
- **4-slot correction**: DESIGN/GATES/REPORT said N4 needed 2 inputs; the truth is four —
  `GBuffer.Normals`, `DLSS.Input.DiffuseAlbedo`, `DLSS.Input.SpecularAlbedo`, `GBuffer.Roughness`
  (`nvngx.cpp:1133-1136`, fetched `:1152-1161`, = referee `kRrInputs`, `ngx_client/main.cpp:91-98`).
  All three places corrected in place with an explicit "this row first said 2" marker.

**N5 confirmation (the evidence you asked for, `runs/C01/rep4`):**
`Creating new xess upscaler` ×2, `fsr_fallback=0`, **no** `CheckUpscalerFiles … not found`, and the
`ChangeFeature changing backend to …` line now names **xess itself** (`:217 … to xess`,
`init successful for xess`), with 8 companion DLLs in `stage_listing.txt`. Same for fsr31
(`:215 … to fsr31`). The pre-N5 runs (C01/C03 rep1-3) stay on disk as BLOCKED(N5) provenance.

**RR tier — dlssd, 60 frames, 3 reps each (median of run-medians):**

| cell | render -> 4K | EVAL med ms | GPU med ms | gates |
|---|---|---|---|---|
| B01 DLAA | 3840x2160 | 7.0886 | **50.7897** | PASS ×3 |
| B02 UltraQ | 2954x1662 | 7.0330 | **31.3426** | PASS ×3 |
| B03 Quality | 2560x1440 | 7.0358 | **24.7600** | PASS ×3 (rep4-6; rep1-3 = pre-N4 BLOCKED, kept) |
| B04 Balanced | 2228x1254 | 7.0365 | **20.0992** | PASS ×3 |
| B05 Performance | 1920x1080 | 7.0147 | **15.7362** | PASS ×3 |
| B06 UltraPerf | 1280x720 | 6.9438 | **9.0559** | PASS ×3 |

every run: `G-IDENT 'Creating new dlssd upscaler'`, `G-PORT ok=60/60 'fmt col=0 mv=10 dep=0'x60`,
`G-PORT-RR … all n=7 … OUTPUT ORACLE-PENDING`, `G-OUT nz=1.000000` with the oracle caveat,
`G-STAGE nvngx_dlssd.dll=48343664`.
RR floor (B03 ×3): EVAL range 0.099 → threshold **0.198**; GPU range 0.248 → threshold **0.495**.
⇒ EVAL across the RR tier spans 0.160 ms ⇒ **below the floor, no CPU ranking**; GPU gaps 4.4–14.1 ms
are all resolvable (and exceed even the widest intra-cell spread seen, 3.76 ms in B02).
RR-vs-SR at 2560×1440 (B03 vs A03): CPU 7.0358 vs 4.4711 (**+2.56 ms, ×1.57**), GPU 24.7600 vs
10.6106 (**+14.15 ms, ×2.33**) — independently corroborated by the N4 agent's 12-frame smoke
(EVAL 7.1221 / GPU 24.1245 vs SR 4.4160 / 10.2298). Chain build stays one-time (~13.3 s).

**E02 rep2 = the D3-grade decoupling result**: with the SL stack booted, RR still runs 60/60 through
our port (`G-PORT-RR PASS … n=7`, EVAL 7.0371 / GPU 24.5138 — same numbers as the non-SL B03),
while `G-RR-ASSURE` still FAILs on the same trio (`init-ngx`, `missing-ctx`, `sl.dlss_d-dropped(x6)`).
⇒ **Streamline's refusal and our RR capability are now provably independent**: the port+Opti dlssd
path serves RR; what stays broken is the SL-side gating (H-1' territory), in the harness and
in-game alike.

**Type axis completed — xess (6 cells ×3, sentinels C01 rep7) and fsr31 (6 ×3, sentinel C03 rep7):**

| backend / quality | EVAL med ms | GPU med ms |
|---|---|---|
| xess DLAA / UQ / Q / Bal / Perf / UPerf | 0.1864 / 0.1841 / 0.1832 / 0.1722 / 0.1717 / 0.1741 | 4.2261 / 4.4986 / 3.8601 / 3.8401 / 3.7039 / 3.7143 |
| fsr31 DLAA / UQ / Q / Bal / Perf / UPerf | 0.1271 / 0.1267 / 0.1267 / 0.1180 / 0.1164 / 0.1181 | 3.0274 / 2.2870 / 2.1540 / 1.8114 / 1.9028 / 1.9088 |

Floors: xess GPU 0.908 ms (rep range 0.454!), fsr31 GPU 0.265 ms; EVAL floors 0.007 / 0.006.
G-EFFECT applied: **xess GPU gaps are ALL below its 0.908 floor ⇒ the xess quality axis is
unranked** (it also shows no resolution trend — worth a future look, not a claim); xess EVAL
Q(0.1832) vs Bal/Perf(≈0.172) = ~0.011 > 0.007 ⇒ resolvable. fsr31 GPU: DLAA→UQ 0.740 ✓ and
Q→Bal 0.343 ✓ resolvable, the rest below 0.265 ✗; fsr31 EVAL Q vs Bal 0.0087 > 0.006 ✓.

**Comparative table at 2560×1440 → 3840×2160 (Quality), median of run-medians, all cells identity-
gated:**

| path | EVAL ms | GPU ms | n |
|---|---|---|---|
| fsr22 (Opti internal) | 0.0946 | 1.7158 | 4 |
| fsr31 (Opti + amd dll) | 0.1267 | 2.1540 | 3+sent |
| xess (Opti + libxess) | 0.1832 | 3.8601 | 3+sent |
| **dlss → our port** | 4.4711 | 10.6106 | 4 |
| **dlssd (RR) → our port** | 7.0358 | 24.7600 | 3 |

Same caveats as wave 2, now with a fifth row: the port paths carry their own copies + record and are
not like-for-like against Opti's internal backends; the CPU gap (×47 vs fsr22) is already attributed
to the port by the `--direct` A/B; **mechanism still unrooted** ⇒ candidate for a later lever round,
and this table is the first full-baseline that such a round can be judged against.

**Disk:** auto-purge held across all 55 runs — 0 `out_*.f32`, `runs/` = 12 MB.

**Matrix status: 33/33 cells have run** (B03/E02 carry both their pre-N4 BLOCKED and post-N4 PASS
reps; C01/C03 likewise pre-N5/N5). Only E01/E02 remain non-PASS, both on `G-RR-ASSURE` — the
in-game SL trio, which is D2/H-1' work, not D1's.

### Environment side-effect to flag (executor/D3 relevant)

Every wave-1 run used the game prefix un-cloned (game not running ⇒ `game prefix free`), and Proton
10.0 printed on first touch:

```
Proton: Upgrading prefix from GE-Proton11-6 to 10.1000-105 (/home/deck/.local/share/Steam/steamapps/compatdata/3321460/)
Proton: Prefix has an invalid version?! You may want to back up user files and delete this prefix.
```
(`runs/A03/rep1/run.stdout.log:1-2`) — i.e. **the shared prefix's Proton version marker was flipped
10.1000-105 by the harness, and Proton itself called it invalid.** The executor should expect the
game's next GE-Proton11-6 launch to re-assert its own version; no user files were written by us and
`tools/verify/**` stayed untouched. Re-verified: launch line in `localconfig.vdf:2894` never edited
this session.

### Wave-1 scoreboard

| tier | cells run | runs | PASS | FAIL | BLOCKED |
|---|---|---|---|---|---|
| A (dlss × 6 quality, RR off) | 6 | 19 (incl close-out sentinel) | 19 | 0 | 0 |
| E (`--sl`) | E01, E02 | 2 | 0 | 2 (G-RR-ASSURE, expected current state) | 0 |
| B (RR on via `dlssd`) | B03 | 3 | 0 | 0 | 3 (N4 — RR create itself PASSED) |

Cells remaining unrun: B01,B02,B04,B05,B06 (N4), C01-C03 + D01-D15 (18, N1-ready, not yet scheduled),
E03 (N1-ready).

**Disk provenance:** wave 1 produced 1440 read-back frames `out_*.f32` × 127 MB = **178 GB** in
`tools/scenario/runs/` (4K RGB32F × 60 × reps). Purged at 03:1x with
`find tools/scenario/runs -name 'out_*.f32' -delete` (evidence retained: 2.9 MB of logs/CSV/manifests;
freed 178 GB, `/home` 87%→82% `df`). Nothing in any gate or number above depends on them (`OUT_NZ` is
a client.log line), each is re-derivable in ~25 s by re-running its cell, and `run_cell.sh` now purges
them after gating (`KEEP_FRAMES=1` opts out) so no later session repeats this.

### D4 addendum — L4 verdict folded in (received 02:3x from top-level; report `/tmp/opencode/lever-runner-L4.md`)

Measured verdict: substituting the staged `nvngx_dlss*.dll` with our port changed **nothing** —
g5 byte-for-byte identical, and our port's `GetFeatureRequirements -> Supported` line
(`src/pe/nvngx.cpp:1389-1394`) appears **0 times in all four cells**: nobody asks the module.
`WINEDEBUG=+loaddll` puts the three failing `getNGXFeatureRequirements` calls at **+0.13..0.17 s
while no `nvngx_dlss*.dll` is mapped**, answered instead by **Opti's own `dxgi.dll`** (53 NVSDK
exports incl. `NVSDK_NGX_D3D12_GetFeatureRequirements` + `..._Init_with_ProjectID`, plus
`HookNgxApi ... found, hooking!` installed on our export 104 ms before SL asks); first genuine
`nvngx_dlss` load is +6.7 s (ReadVersion probe). Byte-pattern check: `0xBAD00002` occurs x1 in
Opti's `dxgi.dll`, x145 in the genuine (then-unmapped) `nvngx_dlss.dll` ⇒ **Opti's NGX surface is
the only possible source of the PlatformError**, and since `..._Init_with_ProjectID` exists *only*
in Opti's `dxgi.dll` while `slOnPluginStartup` still fails, the failure is downstream of that
surface, not a stageable-export problem.

Consequences recorded against the levelling list above:

- **H-1 demoted, H-1' becomes primary:** the fix must change *Opti's* `HookNgxApi` detour / its own
  `GetFeatureRequirements` answer so it either (a) forwards to a real provider, (b) refuses to
  answer (letting the call fail over), or (c) offers an option disabling its NVSDK export surface —
  i.e. exactly H-3's territory (patch OptiScaler source, fork + GH Actions per
  `/tmp/opencode/opti-build-feasibility.md`). An export alias in our port is *not* the lever L4
  showed; it stays only as a follow-up if (c) is unavailable.
- **N2 unchanged and cheap:** `no change` ⇒ staging `nvngx_dlssd.dll` remains harmless-baseline; keep.
- **New side-finding baked into the D1 gates:** a plain `cp` of the port over `nvngx_dlss.dll` KILLS
  SR (L4a: `CreateFeature refused: nvngx_dlss.dll is missing or not its expected size`,
  g2/g3 FAIL) because of the port's own size gate (`src/pe/nvngx.cpp:187-191`); the padded
  variant went green. `gate_cell.sh` now checks both model-DLL sizes in the stage manifest, so a
  mis-staged copy reads `BLOCKED/FAIL` instead of a mysterious SR outage. Evidence kept by the
  executor at `/tmp/opencode/{L3,L4a,L4control}_evidence/`, `slrun_L4*.log`; staging restored to
  baseline, game dir untouched.
- N3 remains held: in-game verification ran 02:24 (game pid 3282740; Xalia `No displays available`
  fatal under examination) — release needs game exited + N1/N2 agent done + executor quiet.