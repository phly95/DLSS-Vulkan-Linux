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

> **FINAL-REPORT SKELETON NOW LIVES IN [`D4-FINAL.md`](D4-FINAL.md)** (assembled ~11:0x; 33-cell
> digest with superseded rows marked, lever provenance P1..WARN, h1prime §4 pointer, upstream-diff
> verdict, RR-oracle scope line, and the open **D3 slot** awaiting the executor's in-game evidence).
> This section below remains the original status log.

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
reps; C01/C03 likewise pre-N5/N5; E03 run at 09:38 as the last cell). The only non-PASS cells are
the three `--sl` ones — **E01/E02/E03 all fail `G-RR-ASSURE` on the identical trio** (`init-ngx`,
`missing-ctx`, `sl.dlss_d-dropped(x6)`), i.e. the failure is *backend-independent*: xess+SL (E03:
EVAL 0.1835 / GPU 3.8250 vs non-SL C01 0.1832 / 3.8601 ⇒ SL's presence costs nothing measurable on
the type axis either) fails it exactly as dlss+SL (E01) and dlssd+SL (E02) do. That trio is D2/H-1'
work, not D1's.

## D1 FINAL RESULTS DIGEST — the 33-cell deliverable table (compiled 09:4x, read-only from `runs/`)

Median-of-run-medians over the **current-era reps** listed (frame 0 excluded; EVAL/GPU p95 are the
median of per-run p95s). *(E01/E02/E03 rows are **superseded** by the post-R1/L-C re-gate in
`## P-RECTOR` below — those three cells are now PASS; their wave-1..4 FAIL rows are kept as
provenance.)* Per-run evidence for every row: `tools/scenario/runs/<cell>/rep<k>/` =
`client.log` (OUT_NZ + EVAL_MS/GPU_EVAL_MS), `eval_ms.csv`, `summary.txt`, `OptiScaler.log`,
`dlss5-amd-ngx.log` (port cells), `g5_evidence.txt` (`--sl` cells), `harness_invoke.log` (g1-g5),
`stage_manifest.txt` + `stage_listing.txt` (sizes/inventory). Verdicts are `gate_cell.sh`'s own
(they can be re-derived at any time with `bash tools/scenario/gate_cell.sh <repdir> <backend> 60 [--sl]`).

| cell | axis | backend | quality | reps | EVAL med | EVAL p95 | GPU med | GPU p95 | verdict | era / provenance |
|---|---|---|---|---|---|---|---|---|---|---|
| A01 | SR | dlss | DLAA | 3 | 4.4663 | 7.5120 | 19.7137 | 20.7421 | PASS ×3 | wave1 |
| A02 | SR | dlss | UltraQuality | 3 | 4.4171 | 8.6545 | 13.3211 | 15.8435 | PASS ×3 | wave1 |
| **A03 (sentinel)** | SR | dlss | Quality | **4** | **4.4595** | 7.0286 | **10.6140** | 12.8852 | PASS ×4 | wave1 + close-out rep4 |
| A04 | SR | dlss | Balanced | 3 | 4.4367 | 7.4179 | 8.9988 | 11.3753 | PASS ×3 | wave1 |
| A05 | SR | dlss | Performance | 3 | 4.4357 | 8.2525 | 8.1175 | 9.0680 | PASS ×3 | wave1 (rep2 p95 54.35 outlier logged) |
| A06 | SR | dlss | UltraPerformance | 3 | 4.3522 | 6.8823 | 5.5119 | 6.9244 | PASS ×3 | wave1 |
| B01 | RR | dlssd | DLAA | 3 | 7.0886 | 12.5470 | 50.7897 | 51.2028 | PASS ×3 | wave3 post-N4 |
| B02 | RR | dlssd | UltraQuality | 3 | 7.0330 | 12.1755 | 31.3426 | 31.6116 | PASS ×3 | wave3 post-N4 |
| B03 | RR | dlssd | Quality | 3 | 7.0358 | 12.9891 | 24.7600 | 25.4000 | PASS ×3 | wave3 post-N4; **rep1-3 = pre-N4 BLOCKED kept** |
| B04 | RR | dlssd | Balanced | 3 | 7.0365 | 13.4226 | 20.0992 | 20.3519 | PASS ×3 | wave3 post-N4 |
| B05 | RR | dlssd | Performance | 3 | 7.0147 | 12.9220 | 15.7362 | 18.4865 | PASS ×3 | wave3 post-N4 |
| B06 | RR | dlssd | UltraPerformance | 3 | 6.9438 | 12.6514 | 9.0559 | 11.1308 | PASS ×3 | wave3 post-N4 |
| C01 | type | xess | Quality | 4 | 0.1832 | 0.2676 | 3.8677 | 4.5293 | PASS ×4 | wave4 post-N5; **rep1-3 = pre-N5 BLOCKED kept**; rep7 sentinel |
| C02 | type | fsr22 | Quality | 4 | 0.0945 | 1.2185 | 1.7747 | 2.1014 | PASS ×4 | wave2; rep4 sentinel |
| C03 | type | fsr31 | Quality | 4 | 0.1270 | 1.4366 | 2.1564 | 2.4224 | PASS ×4 | wave4 post-N5; **rep1-3 = pre-N5 BLOCKED kept**; rep7 sentinel |
| D01 | type | xess | DLAA | 3 | 0.1864 | 0.2255 | 4.2261 | 4.9240 | PASS ×3 | wave4 |
| D02 | type | xess | UltraQuality | 3 | 0.1841 | 0.2354 | 4.4986 | 4.7004 | PASS ×3 | wave4 |
| D03 | type | xess | Balanced | 3 | 0.1722 | 0.2154 | 3.8401 | 4.5154 | PASS ×3 | wave4 |
| D04 | type | xess | Performance | 3 | 0.1717 | 0.2831 | 3.7039 | 4.3715 | PASS ×3 | wave4 |
| D05 | type | xess | UltraPerformance | 3 | 0.1741 | 0.2846 | 3.7143 | 5.4852 | PASS ×3 | wave4 |
| D06 | type | fsr22 | DLAA | 3 | 0.0941 | 1.1999 | 2.5593 | 2.7722 | PASS ×3 | wave2 |
| D07 | type | fsr22 | UltraQuality | 3 | 0.0949 | 1.2713 | 2.1114 | 2.3437 | PASS ×3 | wave2 (rep2 p95 41.04 outlier logged) |
| D08 | type | fsr22 | Balanced | 3 | 0.0864 | 1.1944 | 1.7553 | 1.9947 | PASS ×3 | wave2 |
| D09 | type | fsr22 | Performance | 3 | 0.0869 | 1.1866 | 1.6753 | 1.6963 | PASS ×3 | wave2 |
| D10 | type | fsr22 | UltraPerformance | 3 | 0.0872 | 1.1843 | 1.5202 | 2.2325 | PASS ×3 | wave2 |
| D11 | type | fsr31 | DLAA | 3 | 0.1271 | 1.4540 | 3.0274 | 3.4386 | PASS ×3 | wave4 |
| D12 | type | fsr31 | UltraQuality | 3 | 0.1267 | 1.4193 | 2.2870 | 2.6574 | PASS ×3 | wave4 |
| D13 | type | fsr31 | Balanced | 3 | 0.1181 | 1.4189 | 1.8114 | 2.2606 | PASS ×3 | wave4 |
| D14 | type | fsr31 | Performance | 3 | 0.1164 | 1.4013 | 1.9028 | 2.1437 | PASS ×3 | wave4 |
| D15 | type | fsr31 | UltraPerformance | 3 | 0.1181 | 1.5115 | 1.9088 | 2.3421 | PASS ×3 | wave4 |
| E01 | SL diag | dlss | Quality | 1 | 4.4764 | 5.3243 | 10.4249 | 12.8407 | **FAIL (G-RR-ASSURE only)**; all others PASS | wave1 `--sl`; D3-grade trio captured |
| E02 | SL diag + RR | dlssd | Quality | 1 | 7.0371 | 10.4226 | 24.5138 | 25.2236 | **FAIL (G-RR-ASSURE only)**; RR gates PASS | wave3 post-N4 `--sl`; rep1 = pre-N4 BLOCKED kept |
| E03 | SL diag | xess | Quality | 1 | 0.1835 | 0.2256 | 3.8250 | 4.5408 | **FAIL (G-RR-ASSURE only)** | wave4 `--sl` |

**Sentinel rows / noise floors (the numbers every claim above is gated against, G-EFFECT):**

| sentinel | reps | EVAL range → ×2 floor | GPU range → ×2 floor |
|---|---|---|---|
| A03 dlss/Quality | 4 | 0.0824 → **0.165 ms** | 0.0651 → **0.130 ms** |
| C02 fsr22/Quality | 4 | 0.0023 → **0.0046 ms** | 0.2344 → **0.469 ms** |
| C01 xess/Quality | 4 | 0.0035 → **0.007 ms** | 0.4542 → **0.908 ms** |
| C03 fsr31/Quality | 4 | 0.0030 → **0.006 ms** | 0.1323 → **0.265 ms** |
| B03 dlssd/Quality (RR) | 3 | 0.0992 → **0.198 ms** | 0.2478 → **0.496 ms** |

Rounding note: wave-3's text wrote "0.495"; the exact double of 0.2478 is 0.4956 → 0.496.

**Digest reconciliation (so no two tables in this file disagree):** A03/C01/C02/C03 medians here
include their sentinel rep (4 reps), while the wave-1/2/3 tables quoted the first 3 — e.g. A03
4.4595 here vs 4.4711 in wave 1 (same runs, different rep count). Both are correct for their
stated rep sets; this digest is the authoritative one because it names its reps.

**Verdict summary:** 30 cells fully PASS; 3 cells (E01/E02/E03) PASS on every gate except
`G-RR-ASSURE`, i.e. the in-game SL trio, reproduced headlessly and backend-independently. No cell is
currently BLOCKED; the BLOCKED-era archives (B03 rep1-3, C01/C03 rep1-3, E02 rep1) are retained as
provenance for N4/N5.

### Post-R1 re-gate plan for E01/E02/E03 (TURNKEY — written 09:4x, **not executed**; slot to be released explicitly)

Context: the executor's R1 gives the port a PE version resource so Streamline accepts it as
NGXCore; acceptance is `/tmp/opencode/h1prime-report.md` §4. Nothing below has been run.

```bash
cd /home/deck/DLSS-Vulkan-Linux

# 0) PRECONDITIONS (no harness run):
#    P1 the harness copies build/package/_nvngx.dll (run_harness.sh:51, :120) -> it MUST be the R1 build
stat -c '%y %s' build/package/_nvngx.dll            # newer than 9fc35a5, size will differ from 871812
objdump -x build/package/_nvngx.dll | grep -iA1 '.rsrc' | head    # PE .rsrc present (version resource)
#    P2 SL's NGXCore loader wants the NAME nvngx.dll in the stage; run_harness stages only
#       _nvngx.dll (line 120), so this file must exist at run time - SEE NEED R1-a below. Check after
#       the first run (stage_listing.txt is copied into the rep dir):
#         grep -qx 'nvngx.dll' tools/scenario/runs/E01/rep2/stage_listing.txt || echo "MISSING nvngx.dll"
#    P3 released slot (top-level) + no other runner: run_cell.sh enforces both, fail-closed.

# 1) RUN, one cell at a time, append-only (never overwrites an existing rep):
bash tools/scenario/run_cell.sh E01 --reps 1      # -> runs/E01/rep2   (--sl, dlss)
bash tools/scenario/run_cell.sh E02 --reps 1      # -> runs/E02/rep3   (--sl, dlssd + RR)
bash tools/scenario/run_cell.sh E03 --reps 1      # -> runs/E03/rep2   (--sl, xess)

# 2) RE-GATE each new rep (also re-derives every other gate, archive-safe, no run):
for r in E01/rep2 E02/rep3 E03/rep2; do
  be=dlss; case $r in E02/*) be=dlssd;; E03/*) be=xess;; esac
  bash tools/scenario/gate_cell.sh tools/scenario/runs/$r "$be" 60 --sl \
    | tee tools/scenario/runs/$r/gate.regate-R1.txt
done
#    PASS criterion for all three: `GATE G-RR-ASSURE PASS` (today: FAIL init-ngx missing-ctx
#    sl.dlss_d-dropped(x6)) and `CELL VERDICT: PASS`. Everything else already passed.

# 3) ACCEPTANCE GREPS, h1prime-report.md §4, against the rep dirs (logs are copied OUT of the wiped
#    staging dir by run_cell.sh, so these stay valid after later runs):
R=tools/scenario/runs/E01/rep2
grep -E 'g5 SL boot|c2_no_failed_NGX|c4_no_missing_ctx'  $R/harness_invoke.log   # want: ok / PASS
grep -qF 'Failed to initialize NGX'                       $R/g5_evidence.txt && echo BAD-c2
grep -qF 'Missing NGX context'                            $R/g5_evidence.txt && echo BAD-c4
grep -qF '0xbad00002'                                     $R/g5_evidence.txt && echo BAD-platform
grep -qF "Ignoring plugin 'sl.dlss_d'"                    $R/g5_evidence.txt && echo BAD-rr-drop
grep -qF 'not supported'                                  $R/g5_evidence.txt && echo BAD-rr-notsup
grep -E 'Loaded NGXCore from path|NGX loaded - app id'    $R/g5_evidence.txt || echo "no NGXCore line (only if SL routed it)"
grep -E 'P1001-sl\.dlss_d|Loaded plugin .sl\.dlss_d'      $R/g5_evidence.txt
grep -E "context is missing"                              $R/g5_evidence.txt && echo BAD-ctx-missing
#    (all BAD-* must print NOTHING; the last dlss_d grep must be non-empty)
#    loaddll sanity (`Loaded L"...harness\nvngx.dll"`) needs WINEDEBUG=+loaddll, which run_cell does
#    not set - that evidence stays with the executor's L-runs; my cells prove the g5/Opti side.

# 4) On success: append the three verdict flips to tools/scenario/REPORT.md (D1 closes 33/33 PASS,
#    G-RR-ASSURE historical FAIL rows stay as provenance) and re-gate nothing else.
```

**NEED R1-a (blocking for step 0/P2):** `run_harness.sh` stages `_nvngx.dll` only; the acceptance
gloss in h1prime §4 expects `nvngx.dll` in the stage (the executor's L-runs injected it with a keeper
loop). Either the harness owner stages `nvngx.dll` as a copy beside `_nvngx.dll` (mirroring the
`nvngx_dlss.dll` waypoint, `run_harness.sh:115-117`) or R1's own acceptance path must declare which
name SL is expected to load. Until that is settled my three cells would test R1 only partially.

### ORACLE-PENDING — what RR *output* verification still owes (design only, no work done)

D1's RR numbers are **performance-only**: with N4's frame-independent synthetics (normals = analytic
unit-normal gradient, `DiffuseAlbedo`=0.5, `SpecularAlbedo`=0.04, `Roughness`=0.5), a green `G-OUT`
proves feature 13 *ran and wrote pixels*, not that anything was reconstructed correctly, so no RR
*quality* claim is in scope yet. Three candidate referee routes and why none exists today: (a) the
frozen `tools/verify/**` referee already speaks `kRrInputs` (`ngx_client/main.cpp:91-98`) but its RR
cells (`rr_F_fmt_a`, `run_all.sh:192`) drive the port's **native** runner path, not the
Opti-in-the-loop path D1 measures — re-pointing it at `opti_client` is a new fixture and the verify
tree is frozen, so it would have to live out-of-tree (e.g. under `tools/scenario/`); (b) NVIDIA's
genuine `nvngx_dlssd.dll` cannot be a ground-truth generator here (it needs a real NVIDIA D3D12
driver); (c) there is no analytic truth — a constant-roughness synthetic scene has no closed-form RR
answer. A cheap interim check that needs no referee and is fully specified: **stability** — with
`KEEP_FRAMES=1` on one designated RR cell, the same synthetic inputs must yield byte-identical
`out_*.f32` across reps (they are frame-independent by construction), which would catch nondeterminism
but not wrongness. Recommended scope line for the final report: *"D1 establishes RR presence,
correct-input wiring, and ms costs; RR output fidelity is pending an out-of-tree referee against the
same synthetic inputs, and is explicitly not claimed."*

## D1/D2 REFERENCE PACKAGE — OptiScaler newer-than-v0.9.4: channels, per-area diff, upgrade verdict

*(analysis only, no runs; compiled 10:0x from GitHub APIs + `git diff v0.9.4..FETCH_HEAD` in
`/tmp/opencode/OptiScaler`, which is a full clone of `optiscaler/OptiScaler`. Our checkout = tag
**v0.9.4**, `7534ad0`, released 2026-07-18 — measured by the releases API, `published_at
2026-07-18T21:46:07Z`.)*

### Where newer builds live (how to obtain, all verified this session)

| channel | contents | obtain |
|---|---|---|
| **`optiscaler/OptiScaler-nightly`** (org repo, pushed 2026-10-31T10:17Z) | **85 releases `nightly-YYYYMMDD`**, oldest `nightly-20260807`, newest `nightly-20261031` (2026-10-31T10:17:07Z), each with one public asset, latest `OptiScaler_v10.0.0-pre1_20261031.7z` | **public, no auth**: `https://github.com/optiscaler/OptiScaler-nightly/releases/download/nightly-20261031/OptiScaler_v10.0.0-pre1_20261031.7z` (asset names: v10.0.0-pre1 for recent, v0.9.5-pre*/v0.10.0-pre1 in Aug) |
| **Actions artifacts on `optiscaler/OptiScaler`** | 17 artifacts, newest `OptiScaler_v10.0.0-pre1_20260925.7z` (2026-09-25, 55 732 614 B), older `v0.9.5-pre1..pre4` (Aug-Sep), all `expired=False` | **auth required**: `GET .../actions/artifacts/<id>/zip` with `Accept: application/octet-stream` returned **401** unauthenticated (measured) ⇒ GitHub token or Actions UI |
| **source (the diff basis)** | `git fetch origin master` → `FETCH_HEAD = 6a7b0336` (2026-10-30) = **4712 commits** since v0.9.4 | already fetched into `/tmp/opencode/OptiScaler`; `git diff v0.9.4..FETCH_HEAD -- <paths>` |
| releases API | newest official = **v0.9.4** (then v0.9.3, v0.9.2…); also prereleases `nightly` and `0.7-old_nightly` | confirms "no official >0.9.4" |

**"0.9.5.4" (WrongGPU's tested build) matches no tag or asset name** in either repo (checked all 85
nightly releases + 10 releases + 17 artifacts). Nearest candidate: Actions
`OptiScaler_v0.9.5-pre4_20260905.7z` — pre4 ↔ internal `.4` is a **hypothesis, not established**.
WrongGPU itself (`/tmp/opencode/wronggpu/RELEASE-NOTES.md:37`) only says "all through OptiScaler
0.9.5.4"; it ships no OptiScaler bits ("OptiScaler is supplied separately", `:52`).

### Per-area diff, the five surfaces we fight

| area | v0.9.4 (ours) | master (FETCH_HEAD) | verdict |
|---|---|---|---|
| **(i) nvngx.dll / NGXCore provisioning** | `NVNGX_Proxy.h:446 dllNames={_nvngx.dll,nvngx.dll}`; `LibraryLoad_Hooks.cpp:65-68` "nvngx call: … returning this dll!"; `Kernel_Hooks.cpp:191-226` GetFileAttributesW/CreateFileW override so a missing `nvngx.dll` resolves to a *signed* dll; `Crypt32_Hooks.h` / `Wintrust_Hooks.h` signature bypass (1 / 16 refs); `Util.cpp` reads `Services\nvlddmkm\NGXCore\NGXPath` (7 refs) | `NVNGX_Proxy.h:599` same vector; `LibraryLoad_Hooks.cpp:72-75` same "returning this dll!"; `Kernel_Hooks.cpp:250-287` same overrides (now `State::Instance().nvngxReplacement` instead of a file-static); **counts identical** (dllNames 7/7, NGXCore 7/7, Wintrust 16/16); **new**: `Streamline_Hooks.cpp:246-251` compares PE **file versions** of `sl.*.dll` (local vs game, picks the newer into `pathsToPlugins`) | **no new core shipped/forwarded.** Upstream never grew an `nvngx.dll` provider — the machinery is v0.9.4-shaped. The only new version-resource *use* is for SL plugin selection, not for an NGXCore. R1's insight is not mirrored upstream. |
| **(ii) `Streamline_Hooks.cpp` dlssd-spoof refusal** | `:441-451` "Don't change arch for DLSS/DLSSD…" + `:448 // Don't spoof DLSSD at all { return; }` | `:922-933` byte-equivalent logic (`// Don't spoof DLSSD at all` → `return;`), only refactors around it: `setArch(maxArch, altSystemCaps)`, `activeFgNvngx != FGNvngxReplacement::None`, `fakenvapi::isUsingAsMainNvapi()` | **UNCHANGED.** Upgrading does not remove the DLSSD spoof refusal. |
| **(iii) HookNgxApi / GetFeatureRequirements** | `NVNGX_Proxy.h:91-135` detours D3D11/D3D12/Vulkan `…GetFeatureRequirements`; export `inputs/NVNGX_DLSS_Dx12.cpp:770-812`: `SuperSampling` → `Supported`/`MinHWArchitecture=0`, else forward to real nvngx, else `AdapterUnsupported` + `FAIL_FeatureNotSupported` | `NVNGX_Proxy.h:148-200` identical + **two new Vulkan extension hooks** (`GetFeatureInstanceExtensionRequirements`, `GetFeatureDeviceExtensionRequirements`); export `:899-947` same structure, but provider-init now gated on **`IdentifyGpu::getPrimaryGpu().dlssCapable`** (`:925, :936`); fallback still `FAIL_FeatureNotSupported` (**never `0xbad00002`**) | **No new answer to our PlatformError.** Master additionally *conditions* forwarding on a new GPU-identification module — on a non-Nvidia/dlss-incapable primary GPU it will skip `InitNVNGX()` entirely (different, not better, for us). |
| **(iv) `dllmain` isNvidia / init branch** | `isNvidia()` `dllmain.cpp:1691`, early return `:1712`, final `return nvidiaDetected;` `:1765` (real-NvAPI branch `:1745-1757` = the "1749-1756" the D3 notes cite); three branches: dxvk-nvapi → `NvAPI_Initialize`; fakenvapi (`QueryInterface(0x21382138)`/`Fake_InformFGState`) → `nvidiaDetected=false`; real NvAPI → init() == OK | refactored to lambda `isNvidiaViaNvapi` `dllmain.cpp:1687-1749` + `bool possibleNvidia = isNvidiaViaNvapi();` `:1752`; same three branches; fakenvapi branch now logs **`spdlog::error("Using fakenvapi as nvapi64.dll, remove it!")`** and still `nvidiaDetected=false`; adds `CheckQuirks(bool isNvidia)` `:1313` | **Same outcome** (fakenvapi ⇒ not NVIDIA), louder log. Our `Running on Nvidia` gate behaves the same. |
| **(v) `NGXCore` / `nvngx.dll` / `getNGXFeatureRequirements` references** | `Util.cpp` NGXCore registry (7), `NVNGX_Proxy.h` (3), `Crypt32` (1), `Config.cpp` `nvngxExists` block `:1507-1520` | identical counts; `Config.cpp:2098-2118` same block; **`getNGXFeatureRequirements` (SL's symbol) appears in NEITHER tree** — it lives in `sl.common.dll`, consistent with h1prime | no upstream change on this axis |

**New risk found while diffing (matters to our whole approach):** master introduces
`misc/IdentifyGpu.cpp` with `gpuInfo.dlssCapable = architecture_id >= NV_GPU_ARCHITECTURE_TU100`
(`:287`) and gates Streamline config-JSON surgery on
`primaryGpu.vendorId == VendorId::Nvidia && dlssCapable` at
`Streamline_Hooks.cpp:987, :1114, :1218, :1507` (clearing SL's advertised VK instance/device
extensions + 1.2 features when the primary GPU isn't Nvidia-capable). **On our AMD primary GPU that
gate changes exactly the caps/extension surface we currently spoof** — direction untested.
Config churn also bit us once already: **`OverrideNvapiDll` no longer exists in master** (3 refs → 0;
successor is `[Libraries] NvapiPath`, `Config.cpp:822`), while `DxgiFactoryWrapping`,
`NvngxPath`, `StreamlineSpoofing` survive (3/3/3) and `Dx12Upscaler` grew (2 → 8 refs).

### Verdict — is upgrading cheaper than N3 (patching)?

**No, not as a fix for c2/c4 — keep v0.9.4 as the measured baseline and pursue R1 + the N3 patch
route; treat an upgrade as its own re-validation project.** Reasons, all from the table above:
(1) the four surfaces we are fighting are **unchanged or riskier** — the dlssd-spoof refusal is
verbatim, `GetFeatureRequirements` still cannot produce anything but `FAIL_FeatureNotSupported`
(no PlatformError fix, no new NGXCore), and the new `IdentifyGpu` gate plausibly *tightens* rather
than relaxes caps handling on an AMD primary; (2) our D1 digest (33 cells, this file) is measured on
v0.9.4 — any upgrade invalidates it and costs a full re-run (~40 min) plus an ini-key audit
(`OverrideNvapiDll` is already gone, so keys must be re-mapped, and our harness template + the
game's ini both live in that space); (3) 4712 commits / 3 months of delta in a `dllmain.cpp` that
grew by 1505 lines and a `Streamline_Hooks.cpp` by 2068 lines is a large blast radius for a lever
with no evidence it touches our root. **Where an upgrade becomes the cheaper path:** if R1 and the
N3 patch both fail, or if we later want upstream bugfixes for the game itself — WrongGPU validated
Crimson Desert on the 0.9.5.x generation (`RELEASE-NOTES.md:37`), so that generation is known-good
for our title; the obtainable artifacts are public nightly zips (table above), and the cheapest
first experiment would be one harness-only `--ini`-compatible smoke on `nightly-20261031` gated by
the same G-IDENT/G-PORT suite, not a game-side swap.

## P-RECTOR — post-R1 + L-C re-gate of E01/E02/E03 (executed 10:54:02–10:59:24, slot released by top-level)

Slot preconditions: R1 landed (`1fd8c6c`, port PE version resource; staged `_nvngx.dll` rebuilt
09:57:37, 873 366 B), L-C first-class + g5 header fix landed (`a349cfb`), `pgrep` clean before each
cell, executor stood by for this wave. Turnkey plan executed as written at `ce0e0cb`.

### Verdicts — **all three cells PASS; `G-RR-ASSURE` flipped FAIL → PASS**

| cell / rep | backend | `G-RR-ASSURE` evidence (gate line) | EVAL med / p95 ms | GPU med / p95 ms | cell verdict | harness header |
|---|---|---|---|---|---|---|
| **E01/rep2** | dlss + `--sl` | `caps=10 init-ngx=0 drv0=0 ctx=0 plugin_sl.dlss=x2 plugin_sl.dlss_d=x2 dropped=0` | **4.4290 / 8.3448** (n=59) | **10.5915 / 12.9216** | **PASS** (7/7) | `HARNESS RESULT: PASS … g5=PASS` (`c1..c5 = ok`) |
| **E02/rep3** | dlssd + `--sl` (RR) | same six-way PASS | **7.0051 / 9.5282** | **24.4524 / 24.6942** | **PASS** (all gates incl. `G-PORT-RR … all n=7`) | `HARNESS RESULT: PASS … g5=PASS` |
| **E03/rep2** | xess + `--sl` | same six-way PASS | **0.1833 / 0.2137** | **4.0977 / 4.5688** | **PASS** (scenario gates) | `g5 SL boot : PASS …` but `HARNESS RESULT: FAIL` — **known type-axis divergence, see below** |
| E03/rep3 | xess + `--sl` (confirmation run) | same six-way PASS | 0.1826 (n=60 incl. f0) | — | PASS | `g5=PASS`, same `HARNESS RESULT: FAIL` |

**D1 is now 33/33 PASS** — the three cells that carried the only FAIL verdicts in the matrix have
flipped; the earlier FAIL rows stay as provenance (digest table above now points here).

### §4 acceptance greps (`h1prime-report.md` §4), against the copied rep dirs

All three cells, every negative check **0 occurrences**: `Failed to initialize NGX`,
`Missing NGX context`, `0xbad00002`, `Ignoring plugin 'sl.dlss_d'`,
`not supported on this platform`, `context is missing`.
Presences (each cell): 

```
g5 SL boot : PASS  c1_getSystemCaps_nvapi=ok c2_no_failed_NGX=ok c3_driver=ok c4_no_missing_ctx=ok c5_sl.dlss_loaded=ok
pluginManager.cpp:907[mapPlugins] Loaded plugin 'sl.dlss_d' ... id 1001 - priority 100 - adapter mask 0x1   <- mask was 0x0 pre-fix
pluginManager.cpp:1193[loadPlugins] P100 - sl.dlss_d                                                        <- in the execution order
commonEntry.cpp:1694[slOnPluginStartup] NGX loaded - app id 101640911 - application data path Z:\...\harness\
```
- `Loaded NGXCore from path (%ls)`: **absent** in all three — §4 lists it as "if routed", so it is
  recorded, not treated as a miss.
- **loaddll sanity is not available from my runner** (run_cell does not set `WINEDEBUG=+loaddll`);
  that half of §4 stays with the executor's L-runs, as the plan already stated.
- `nvngx.dll` (the L-C file) is in the stage for E01/E02 (manifest `58956912` = byte-copy of the
  signed `nvngx_dlss.dll`), and in E03/rep3 — see the rep2 anomaly below.

### Watch item: is the SL path's `EVAL med 56.9 ms` (L-C, 12 frames) real? — **No.**

60-frame methodology runs say the SL path adds **no measurable per-frame cost**:

| cell | with `--sl` (this wave) | without SL (same res, same gates) | Δ | floor (2×sentinel) | verdict |
|---|---|---|---|---|---|
| E01 vs A03 | EVAL 4.4290 / GPU 10.5915 | EVAL 4.4595 / GPU 10.6140 | −0.0305 / −0.0225 | 0.165 / 0.130 | **below floor → no effect** |
| E02 vs B03 | EVAL 7.0051 / GPU 24.4524 | EVAL 7.0358 / GPU 24.7600 | −0.0307 / −0.3076 | 0.198 / 0.496 | **below floor → no effect** |
| E03 vs C01 | EVAL 0.1833 / GPU 4.0977 | EVAL 0.1832 / GPU 3.8677 | +0.0001 / +0.2300 | 0.007 / 0.908 | **below floor → no effect** |

And the 56.9 ms figure does not exist anywhere in the 60-sample window: E01/rep2's **max single
frame = 8.8762 ms**, frame 0 = 9.2078 ms, and the *first 11 samples* (the closest I can come to a
12-frame view of my own data, frame 0 excluded) have median **4.4024 ms**. Reported as measured, no
smoothing: **L-C's 56.9 ms did not reproduce at 60 frames — short-run/init noise** (its 12-frame
window cannot be reconstructed from here; if needed, the executor's L-C log can be re-cut the same
way).

### Anomaly + fix (recorded, not smoothed)

E03/**rep2**'s post-run `stage_listing.txt` lacked `nvngx.dll`, while its own
`OptiScaler.log:1 [10:55:05] Config::CheckUpscalerFiles nvngx.dll found in game folder` proves the
file existed **during** the run — and rep3, identical command, staged it normally
(`stage_listing.txt:14`). Diagnosis: the copy-out raced something re-staging the shared
`$STAGE` after our process returned (single-runner discipline covers concurrent *runs*, not this
sub-second window). `run_cell.sh` now prints a WARN when an `--sl` run's listing lacks
`nvngx.dll`, pointing at the Opti log line for runtime truth, so a future reader cannot mistake a
race for a missing precondition. Rep2's own numbers are unaffected (its g5/§4 evidence and rep3
agree).

### Two notes for the harness owner (informational, not needs)

1. `HARNESS RESULT: FAIL` on **E03** is `g1 create=0` (its literal is `Creating new dlss upscaler`,
   xess logs `Creating new xess upscaler`), `g2 ok=0/60` (no port log by design for Opti-internal
   backends) and `g3 viol=4/60` (the 4-frame warmup) — the divergence D1-GATES "Harness mapping"
   already documents as expected; **`g5=PASS` inside that header is the fix top-level asked about,
   and it did land**. If the harness ever wants a green exit for non-dlss cells, g1/g3 need the
   same backend/warmup generalization my scenario gates have (g2 cannot apply without a port log).
2. The g5 header assignment fix (`a349cfb`) behaves exactly as specified on all three cells
   (`g5 SL boot : PASS`, `c1..c5 = ok`, folded into the exit verdict).

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
### R1/R2 cycle (L15-L20, 2026-10-31 10:10 EDT) — full detail in /tmp/opencode/h1prime-report.md §6
| # | time | what | result / revert |
|---|---|---|---|
| P-PORT-R1 | 10:57 EDT | **port source**: new `src/pe/nvngx_version.rc` + `src/vk/build.sh` windres wiring; rebuilt `build/package/_nvngx.dll` (873,366 B, md5 `19942e53`, backup of old DLL `/tmp/opencode/_nvngx.dll.preR1`) | kept; harness `--sl` L15: c2/c4 unchanged, core still not loaded ⇒ version-resource gate falsified |
| P-R2a | 10:01 EDT | prefix `drive_c/windows/system32/nvngx.dll` (NEW file, `find` showed none existed) | REJECTED (never loaded); **removed** in revert |
| P-R2b | 10:04-10:06 EDT | registry: `Services\nvlddmkm\{NGXCore,Parameters\NGXCore}\NGXPath`, `SOFTWARE\...\NGXCore\{FullPath,NGXPath}` | REJECTED for c2/c4 + flipped `isNvidia` true→false (regression: g2/g3 red) ⇒ **fully reverted** (keys/values deleted, snapshot `/tmp/opencode/system.reg.preR2revert`); L20 confirmation run: g1-g4 PASS + `Running on Nvidia` restored |
| WARN | - | never create keys under `Services\nvlddmkm` (breaks Opti/dxvk NVIDIA detection, `Advapi32_Hooks.h:26`); `SOFTWARE\...\NGXCore\FullPath` is Opti-owned | - |

### N1+N2+L-C cycle (2026-10-31 10:40 EDT) - **c2/c4 FLIPPED + sl.dlss_d LOADED** (detail: /tmp/opencode/h1prime-report.md §7)
| cell | staged harness/nvngx.dll | g5 c2/c4 | decisive evidence |
|---|---|---|---|
| LA1 | our port (unsigned, R1 version resource) | bad/bad | Wintrust.dll+crypt32.dll loaded in 012c (first ever), file opened x2 by verifier, module NEVER loaded |
| N2 | absent (+WINEDEBUG=+reg) | bad/bad | SL consults exactly `nvlddmkm\Parameters\NGXCore` -> `nvlddmkm\NGXCore` -> `SOFTWARE\...\NGXCore\FullPath` in retry cycles + app-local stat |
| N2b | system32 copy, registry untouched | bad/bad | zero opens of the file (dir-valued registry result rejected pre-open) |
| **L-C** | **genuine SIGNED nvngx_dlss.dll (58,956,912 B)** | **ok/ok** | `Loaded plugin 'sl.dlss_d' id 1001` + `loadPlugins] P100 - sl.dlss_d`; `Failed to initialize NGX`=0, `Missing NGX context`=0, `ngxResult failed`=0 (was 3); g1-g4 PASS 12/12 nz=1.0; g5 header prints `n/a` despite all five sub-gates ok (harness-quirk flagged, file untouched) |
| N1 | disassembly (slcommon.asm) | - | pre-load gate `0x18008d0d0`->`0x18008ce00` resolves WinVerifyTrust+3 WTHelper (wintrust) + 9 cert fns (crypt32); contract `0x18008d374 mov %edi,%eax` (0=fail+SetLastError(esi), 1=pass); LoadLibraryExW `0x18008cd86` only on pass |

**WINNER = file-level: staged `nvngx.dll` = byte-copy of genuine signed `nvngx_dlss.dll`.** Game-dir plan
(`bin64/nvngx.dll` <- copy of `bin64/nvngx_dlss.dll`, NEW file + provenance row) now UNBLOCKED per the
"flip happens in harness first" condition - **awaiting top-level approval before touching bin64**.
Registry chain reverted (L20 baseline proven); `Services\nvlddmkm` writes forbidden; port R1 `.rc` kept (commit 1fd8c6c).
| G1 | 2026-10-31 11:0x EDT | `bin64/nvngx.dll` **NEW file** (find/ls proved no predecessor, no backup possible) = byte-copy of `bin64/nvngx_dlss.dll` (cmp OK, md5 source==copy `a0dac8ab01c00f56a1de1bc383f4147c`, 58,956,912 B, Authenticode Security Directory 0x3837400+0x2870 present) | APPROVED by top-level after E01-E03 GREEN (a349cfb first-class staging); harness-proven winner (L-C) applied game-side; levers G2 `DxgiFactoryWrapping=false` + Opti dxgi.dll IN + fakenvapi.dll present all left as-is |

### D3 execution (2026-10-31 11:25 EDT) — G1 kept in place; in-game SL boot still blocked by LAUNCH PATH (detail h1prime §9)
- D3a direct chain (cwd=bin64, full Steam env): Opti boots (+0.3 s) then engine-level stall, no `hkD3D12CreateDevice`/`slInit`/`getSystemCaps` for 3 min (archive `runs/OptiScaler.log.D3a-stall-111908`) — G1 inert on this layer, as §8.2 predicted.
- D3b/D3c Steam URL forward x3: client assert-crash each time (dumps `assert_20261031112059_9`, `...112450_{3,6,9}`), no game process.
- Verified byte-exact launch line before+after; `bin64/nvngx.dll` intact (byte-copy, md5 `a0dac8ab...`); all other levers untouched; zero leftovers. Pending: a working launch path (proposed: xdotool UI-Play / Proton-10 direct A-B).

### N-B + N-A (2026-10-31 11:48 EDT) — D3 closed as "harness-proven, in-game launch-path-blocked" (h1prime §10)
- N-B direct + Proton 10.0: stall reproduces at `RunVersionCheck`+0.3 s (`hkD3D12CreateDevice`=0) ⇒ runtime-independent engine-startup/launch-context issue; Xalia further along under P10; archive `runs/OptiScaler.log.NB-stall-p10-*`; prefix regs restored from pre-NB snapshot (md5-verified).
- N-A UI click: client healthy but no Steam UI window mapped (6 polls) + no screenshot tooling (Spectacle no DBus object) ⇒ blocked; no forward issued.
- FINAL STATE (deliberate, per top-level verdict): G1 `bin64/nvngx.dll` KEPT (signed byte-copy, cmp-verified after every attempt); `DxgiFactoryWrapping=false` kept; `OverrideNvapiDll=auto`; launch line byte-exact before+after; prefix regs GE-Proton11-native; G1 provenance row current.
### D5 / S1 prep (2026-10-31 13:15 EDT)
- Port commit (see git log): RR capability key `SuperSamplingDenoising.Available=1` in AllocateParameters (A4(i)), MV.Scale read+default (H3), 1 Hz `[diag-d5]` jitter/mv_scale line (H2). Validated headless FIRST: **D5m1 `--direct --sl` = HARNESS RESULT rc=0, g1 n/a, g2 12/12, g3 nz=1.0, g4 4.27 ms, g5 SL boot PASS c1-c5** — first all-green standalone (no-Opti) cell; d5 diag shows real jitter (±0.4375) and mv_scale=1.0000.
- S1 game-dir state: `bin64/dxgi.dll` → `dxgi.dll.optiscaler-off-s1` (byte-identical off-name backup exists; Opti OUT of load path; file preserved). G1 `nvngx.dll` signed core intact; fakenvapi.dll present; `DxgiFactoryWrapping=false` kept (inert with Opti out); launch line verified count==1. **READY FOR RELAUNCH (user manual desktop-shortcut launch)** for the S1 reference rung; dump probe armed via `bin64/dlss5-amd-dump.enable` if desired on that rung.
- Addendum (corrected): `dxgi.dll.optiscaler-off` (the older off copy seen in the 14:43 listing) was **already absent** when I moved today's IN-path build (cmp guard fired on a missing file, not on a byte difference). Current state: exactly one preserved Opti build = `dxgi.dll.optiscaler-off-s1` (md5 `d917da31633f309195916d8ed0570d82`, 25,379,632 B), no `dxgi.dll` in the load path, zero deletions by me this cycle (mv semantics).
### D5 RR headless cycle 2026-10-31 13:38-13:50 (cells rr1-script-die → rr4; detail in shell log + d5-state)
| cell | lever | g5 delta | verdict |
|---|---|---|---|
| rr1 | first `--sl-rr` | died pre-run: staging wants harness-convention `dxgi.dll.optiscaler-off` (my `-s1` name) | renamed to convention (bytes md5 d917da31); script-lane nit recorded |
| rr1b | standalone+RR | c6 bad(doom0xbad00000=**0**, not_loaded=1, ctx_missing=2); **arch 0x130**, `sl.dlss_d adapter mask 0x0`, `dlss_dEntry:210 not supported` | **A4 key fix VERIFIED** (0xbad00000=0); blocker = arch < TU100 |
| rr2/rr3 | fakenvapi → prefix `system32/nvapi64.dll` | arch still 0x130 | **inert: GE-Proton `proton:1417-1426` redeploys its `files/lib/wine/nvapi/x86_64-windows/nvapi64.dll` on every `PROTON_FORCE_NVAPI=1` launch** (overwrote swap) |
| rr4 | fakenvapi swapped AT BOTH proton sources (sudo; backups `.bak-dxvknvapi` GE=bdcd58b8 / P10=035510e7) | **arch 0x190 Ada ✓, ALL plugin masks 0x1 ✓** — but NEW: `getNGXFeatureRequirements 0xbad00005` ×2 → c2 `Failed to initialize NGX` ×1, c4 `Missing NGX context` ×1, c6 still bad | fakenvapi's NVAPI subset breaks the genuine CORE's requirements query (works only on dxvk-nvapi) |

**RESOLUTION REQUESTED (launch-line class ⇒ top-level/user):** both halves are individually proven —
dxvk-nvapi ⇒ NGX core OK (E-cells) but arch 0x130 masks fail; fakenvapi ⇒ arch 0x190 masks pass but core
0xbad00005. The single satisfying lever = **keep dxvk-nvapi + `DXVK_NVAPI_GPU_ARCH=AD100`** (dxvk-nvapi's
documented arch override) ⇒ (1) add `SL_NVAPI_ARCH` passthrough to run_harness (harness lane, requested) →
one headless cell to prove `arch 0x190 + masks + NGX-OK + c6`, (2) append the var to the permanent launch
line (user discussion) → READY FOR RELAUNCH. **Proton sources reverted to dxvk-nvapi originals now**
(backups kept); prefix `system32/nvapi64.dll` restored to dxvk-nvapi; my `fakenvapi.ini` removed.
| P-LAUNCH-ARCH | 2026-10-31 14:0x EDT | `userdata/49277565/config/localconfig.vdf:2894` launch options += ` DXVK_NVAPI_GPU_ARCH=AD100` (backup `.bak-archlever`; new-string grep==1, old-string grep==0) | **USER-APPROVED rule amendment**, authorized verbatim by top-level: "USER APPROVED the launch-line append (asked via structured question 13:5x EDT, answer = 'Append the env var (Recommended)'; approval recorded here as provenance)" — permanent line, never touched beyond this |
- P-LAUNCH-ARCH live-client caveat: Steam client (pid 4778) was UP during the edit ⇒ per authorization, **re-verify after next client start** is on record (first immediate re-check after edit: new==1/old==0).
- **D5 proof cell = PARTIAL (both arch settings triangulate the dial).** AD100 (13:55): `g5 FAIL c1=ok c2_no_failed_NGX=bad(x1) c3=ok c4_no_missing_ctx=bad(x1) c5=ok c6_rr_enable=bad(doom0xbad00000=0 not_loaded=1 ctx_missing=2)`; `Adapter 0 architecture 0x190 ✓`; **6× `adapter mask 0x1` ✓**; but `getNGXFeatureRequirements ngxResult failed 0xbad00005` ×2 → `Failed to initialize NGX` ×1 → `Missing NGX context` ×1 → marker `SL-RR-ENABLE rc=0x00000022 ctx=missing`. TU100 (13:57): arch `0x160 ✓`, masks **5×0x1 + 1×0x0**, identical `0xbad00005`×2/c2/c4/c6. Combined with rr1b (arch 0x130 ⇒ c2/c4 OK but masks 0×1): **`0xBAD00005 = NVSDK_NGX_Result_FAIL_InvalidParameter` (`nvsdk_ngx_defs.h:103`) is returned by the GENUINE core only when adapter arch ≥TU100; at arch 0x130 the core blesses init but SL's static TU100 floor zeroes the masks. No arch value satisfies both while the genuine core answers. All 8 nvngx_dlss.dll on box are Turing-stamped (`NGXGpuArchitecture=NVSDK_NGX_GPU_Arch_Turing`; ZZZ/HL2RTX/PRAGMATA/Crimson/stage2 = 324,x / 310,3,0,0) ⇒ **no newer-stamped core lever exists on this box**.
| P-N3 | 2026-10-31 14:2x | `bin64/sl.common.dll` 2-byte gate bypass: file 0x8c15e `74 2f` → `90 90` (NOP of `je 0x18008cd8f` after `call 0x18008d0d0; test %eax,%eax` — the single integration point of the WinVerifyTrust pre-load gate; gate result now ignored ⇒ `LoadLibraryExW` always runs for a valid path). Backup `sl.common.dll.bak-n3`; `cmp -l` proves ONLY those 2 bytes differ; disasm-after verified (`nop/nop` at 0x18008cd5e/f); no other sl.* touched; launch line untouched | **N3 AUTHORIZED by top-level verbatim: "N3 IS NOW SIGNED OFF (user granted software-patching broadly; I authorize as top-level): patch sl.common.dll's pre-load gate so the contract at 0x18008ce00 always returns 1 ... STRICT TERMS: same-dir backup .bak-n3, provenance row quoting this authorization, patch only that gate (minimal byte edit, verify with disasm before+after), NO other sl.* file touched"** — implemented at the caller (single consumer) rather than inside 0x18008ce00 to avoid null-ptr risk on its other cached users; contract EFFECT (gate always pass) achieved with the minimal 2-byte edit. Recorded risk: patched sl.common carries broken Authenticode (Proton does no load-time sig check) = accepted per authorization. A-route note: `pacman -Ss osslsigncode` = empty on this box → tool unavailable → straight to B per instruction. |
| P-N3-OUTCOME | 14:09-14:15 | **N3 measured FAILED + REVERTED.** Patched cell: `pluginManager.cpp:917 Failed to load plugin '.../harness/sl.common.dll - last error Invalid handle'` (full text, NOT a signature message — 0 hits for 'NOT correctly signed') → JSON `key 'external' not found` cascade ×N → mini-dumps ×2 → `SL slInit result=24`, marker `SL-RR-ENABLE rc=0xSKIP ctx=skip(slInit=0x00000018)`. A/B control (sl.common restored, everything else identical incl. port-core substitute): `slInit result=0 (eOk)`, `Invalid handle`=0, `Failed to load sl.common`=0; restored gate then correctly rejects the unsigned port core (`0xbad00002`×3 → c2/c4/c6 red = the known pre-N3 deadlock). ⇒ the 2-byte NOP at 0x8c15e breaks SL's own load of its host library ⇒ any sl.common edit is blocked by the library's load-critical role (not by Authenticode — that string never fired). **A-route: tool unavailable (`pacman -Ss osslsigncode` empty). B-route: measured fail. Per authorization C-rule: STOP.** File state: `sl.common.dll` md5 `08d257f8...` == `.bak-n3` (kept for record); no other sl.* touched; keeper killed; stage will re-stage clean next run. |
| P-SIGN-D | 14:2x-14:36 | **D SEQUENCE GREEN.** (i) osslsigncode 2.15-dev built FROM SOURCE into `build/pe/tools/` (host has no cmake/cc/ssl-headers → built inside podman `nrbuild:2` with container-only `apt libssl-dev`, no host changes; clone `868d6e9`). (ii) cert regenerated with **Code Signing EKU** (`build/pe/sign/port-core.cert.pem`, CA:FALSE+codeSigning; sha1 A9948E7A...8AF8) → signed D5 port: `build/package/_nvngx.signed.dll` 875,728 B md5 `1bca310f`, digest verify Current==Calculated ✓. (iii) trust anchor: **wine-staging 11's certutil.exe is a STUB** (`fixme:certutil:wmain stub:`) and `proton run`'s steam.exe-shim swallows certutil → wrote Root store directly: `HKLM\Software\Microsoft\SystemCertificates\Root\Certificates\<thumb>` default value = 922 B raw DER via wine `reg add` → "operation completed successfully" (backup `pfx/system.reg.bak-certstore` md5 11fa0d39; deviation note: GE-Proton11's wine used for the store write since Proton 10.0 ships no certutil.exe — store is runtime-agnostic registry state). (iv) **PROOF CELL (`SL_NVAPI_ARCH=AD100 --direct --sl-rr`, signed port as harness core via keeper, WINEDEBUG=+loaddll):** `g5 SL boot : PASS` c1..**c6 all ok**; arch `0x190` ✓; `6× adapter mask 0x1` ✓; `0xbad00005=0 0xbad00002=0 Failed-to-init=0 Missing-ctx=0` ✓; **`SL-RR-ENABLE rc=0x00000000 ctx=loaded`** + `slInit result=0 (eOk)` ✓; **load decision captured: `Loaded L"...harness\nvngx.dll" at 00006FFFFC600000: native`** (WinVerifyTrust ACCEPTED the signed port — the unlock). Note: one non-gate `0xBAD00000` hit elsewhere (c6's archived pattern absent ⇒ gate ok); port `feature=13` not exercised headlessly (smoke runs pre-tag path) — in-game toggle is the RR-eval proof. **FALLBACK WORDING NOW RESOLVED — no longer contingent.** |
| P-G1S | 2026-10-31 14:4x | `bin64/nvngx.dll`: G1 genuine copy → **SIGNED PORT** (`_nvngx.signed.dll`, 875,728 B). Backup `nvngx.dll.bak-g1-genuine` (cp -p, md5 `a0dac8ab01c00f56a1de1bc383f4147c`); source md5 `1bca310fe5a33f5e4676c88765a7bec0`; installed md5 `1bca310fe5a33f5e4676c88765a7bec0`; `cmp` == signed source ✓ | **APPROVED by top-level: "G1->signed-core game-dir swap = APPROVED now (condition 'flip in harness first' met with all-six g5)"** — makes in-game SL core == the green-cell core (passing WinVerifyTrust + arch-blind requirements + our capability key). G1-genuine preserved; S1 Opti-out + all other levers untouched; launch line untouched. |
| STANDING-NOTE-D5 | — | D4 wording (keep verbatim): **headless evidence carries one non-gate `0xBAD00000` hit elsewhere (c6's archived pattern absent ⇒ gate ok) and `feature=13` was NOT exercised by the pre-tag smoke — the in-game RR toggle is the eval proof** | folds top-level's required honest-note wording into D5 rows |
### D5.3 live capture — 2026-10-31 15:33-15:40 (rows continue below)
| item | evidence |
|---|---|
| LAUNCH PATH (final, for D4) | **URL-forward dead**: 15:25 forward → client death again (no dumps, console_log froze at forward time) ⇒ user-facing trigger unusable (consistent with the 15+ `CCrossProcessPipe` cycles). **Working path = top-level's VERBATIM SteamLaunch chain**: `reaper SteamLaunch AppId=3321460 → SteamLinuxRuntime_4/_v2-entry-point --verb=waitforexitandrun → GE-Proton11-6/proton waitforexitandrun <exe>`, **cwd=bin64**, session env `DISPLAY=:0 XAUTHORITY=/run/user/1000/xauth_* WAYLAND_DISPLAY=wayland-0` (lesson: first tries failed on missing XAUTHORITY), launch options incl. `DXVK_NVAPI_GPU_ARCH=AD100`, **plus the two client-injected envs STEAM_COMPAT_DATA_PATH + STEAM_COMPAT_CLIENT_INSTALL_PATH** (first attempt died `Proton: No compat data path?`, second up in 15 s). Client pid 69996 alive; game pid 72127 from 15:33:51. |
| capture @+3:52 (read-only) | `sl.interposer.dll` mapped ✓; `dxgi.dll` in maps = **PREFIX `system32/dxgi.dll` (DXVK, not Opti — Opti out confirmed live)**; **signed core `nvngx.dll` not yet mapped** (SL NGX stage not reached); game port log `dlss5-amd-ngx.log` untouched since 12:08 (= baseline 7,480,927 B — no evaluates yet, `diag-d5`=0, `feature=13`=0 both EXPECTED pre-world/pre-toggle); main thread `ntsync_schedule`, workers futex, 0.1 %CPU, **GPU busy 0 %**, no ≥400 px X window via xwininfo (xwininfo is X11-only; a Wayland-driver window would be invisible ⇒ window check INCONCLUSIVE), 5 render fds + vulkan loader mapped. |
| pending | user ground-truth (what is on screen) via top-level relay → then SR-quality + RR-toggle capture (`feature=13`, RR frame lines, no-black, model/preset, `diag-d5`). |
### D5 black-3D root cause #2 (SR twin) + fix 3110d8b — rows P-SRKEY / D5-SR-DIAG (2026-10-31 17:0x-17:5x)
| row | evidence / action |
|---|---|
| D5-SR-DIAG | **In-game S-standalone sessions 17:05/17:08**: `dlssEntry.cpp:937[slOnPluginStartup] NGX parameter indicating DLSSContext support cannot be found (0xbad00000)` (`Launcher_2026_10_31_17_08_25_336.log:23`) → `'kFeatureDLSS' context is missing` ×4 (`sl.cpp:179`, archive lines `:55-61`). Port log for the session = only `GetFeatureRequirements -> Supported` + `Init_Ext` ×2 (+432 B over baseline), **zero CreateFeature/evals** ⇒ black-3D. Missing key = **`SuperSampling.Available`** (string sits next to `SuperSamplingDenoising.Available` in `sl.dlss.dll`); 894ce54 primed only the RR twin ⇒ `ParametersImpl::find` → bare `NVSDK_NGX_Result_Fail`. History: the 937-line is in **NONE of the 21 `Launcher_*.log` files nor any `runs/` archive** — SL's parameter surface initialized in-game ONLY with Opti's dxgi as that surface (10-30 `ver=1` runs from 14:57); pre-Opti runs `sl_total=0`; the 23:17 no-Opti baseline `ver=0`. |
| P-SRKEY | Fix **commit `3110d8b`** = port `AllocateParameters` now primes **both** capability keys (`SuperSampling.Available=1u` + `SuperSamplingDenoising.Available=1u`, evidence comments) + harness **g5 c7** (SR twin of c6: greps `dlssEntry:937` doom + `'kFeatureDLSS'` ctx-missing on every `--sl` run). **Note-format deviation recorded as ordered: c7 suffix appears only under `--sl`.** Staged game-side (atomic rename-swap while game ran, lands next launch): `bin64/nvngx.dll` ← `build/package/_nvngx.signed.dll` **md5 `eb9f3fe64fafa0cd51fba9de18725524`, 875,688 B, cmp-identical to src**, signed 17:46 with the SAME cert (sha1 `A9:94:8E:7A:...` ⇒ Root-store anchor intact); unsigned build `bb5cbb6c` (873,992 B); backups `nvngx.dll.bak-pre-srkey` (= `1bca310f`, cp -p) + `nvngx.dll.bak-g1-genuine` (= `a0dac8ab`) both verified. Game dir otherwise unchanged: Opti out, `DxgiFactoryWrapping=false`, launch line `new==1/old==0` re-verified at 17:5x. Executor re-verified every md5/cmp/commit claim read-only before writing this row. |
| D5-SR-CAPTURE (pending) | On next user Steam-UI relaunch: absence of `dlssEntry:937` + no `'kFeatureDLSS' ctx missing` in the new `Launcher_*.log`; port log `D3D12 CreateFeature: id=1 feature=1` + `SR frame 0` + `eval #1 OK`; user SR image judgement; then RR toggle → `feature=13`/RR frame lines + no-black. Top-level watcher status file: `/tmp/opencode/srfix_watch.status`. |
### D5 black-3D layer #2 (S2) — rows D5-SRDIAG-2 / P-S2 (2026-10-31 17:51-18:0x)
| row | evidence / action |
|---|---|
| D5-SRDIAG-2 | Relaunch #2 (17:51, pid 122593) **PROVES the SR-key fix**: new `Launcher_2026_10_31_17_51_43_328.log` = **ZERO `dlssEntry:937` + ZERO `'kFeatureDLSS'` context-missing**; SL boots (DLSS 2.14.1, RTX 4090/999.99). Next layer fails: **`dlssEntry.cpp:819[slGetData] DLSSContext 'getOptimalSettings' callback is missing` ×4** (log `:52-58`); port log still **zero CreateFeature** ⇒ black persists one layer deeper. The 819-line is **new-today-only** (0 hits across all 21 `Launcher_*.log` + archives) — i.e., every earlier session died before this layer. |
| P-S2 | Fix commit `port: publish the core callbacks + PopulateParameters_Impl export (S2)`: `prime_params` mirrors Opti `InitNGXParameters` (`NVNGX_Parameter.cpp:715+`) — callbacks `DLSSOptimalSettingsCallback`/`DLSSGetStatsCallback`/`DLSSDOptimalSettingsCallback` (+alias `#(3e)`), SuperSampling feature/driver keys + UE `MinDriverVersion` 10.10, `EParameter` alias `#(01)`, PQ/Sharpness/MV/Exposure/SizeInBytes/RayReconstruction hint defaults; `port_get_optimal_settings` implements the strict Width/Height/PQ contract + SDK scale table (MaxPerf=0..DLAA=5) + DRS band, dual-writes string+alias keys, logs `[S2]` line; **new exports `NVSDK_NGX_D3D12_PopulateParameters_Impl` + VULKAN twin** (both were in the export-diff vs Opti's dxgi; `Init_with_ProjectID` deliberately NOT added — no evidence needed). Staged game-side (executor re-verified read-only): `bin64/nvngx.dll` = signed **md5 `097a802d498921f2598f23618f489d50`, 879,528 B, cmp==package signed** (digest verified by top-level), unsigned `19bb84ae`; **three backups**: `.bak-pre-cbpop` = `eb9f3fe6` (S1), `.bak-pre-srkey` = `1bca310f`, `.bak-g1-genuine` = `a0dac8ab` (58,956,912 genuine); `[S2]/PopulateParameters/DLSSOptimalSettingsCallback` strings present in staged binary ✓; line `new==1/old==0`, Opti out ✓. Watcher `/tmp/opencode/s2_watch.status` for relaunch #3. |
| D5-S2-CAPTURE (pending) | `[S2] optimal settings cb` line + `D3D12 CreateFeature: id=1` + `SR frame 0` / `eval #1 OK`; user visual verdict; then RR toggle → `feature=13` + RR frame + no-black. |
### D5.3 SR-QUALITY CAPTURE — relaunch #3 (2026-10-31 18:17 session, read-only capture; RR = KNOWN-OPEN/deferred per user)
| fact | verbatim evidence |
|---|---|
| session | `Launcher_2026_10_31_18_17_19_328.log` (289,446 B, mtime 18:19:57), game pid 131208; port log `bin64/dlss5-amd-ngx.log` grew 16.2 MB → 22.8 MB → streaming (mtime 18:26:59) |
| layers cleared (S2 in-game proof) | `dlssEntry.cpp:937`=**0**, `'getOptimalSettings' callback is missing`=**0**, `'context is missing'`=**0** in the #3 log (937 was 0 since #2; 819 was ×4 in #2's log `Launcher_2026_10_31_17_51_43_328.log`) |
| **[S2] pq-probe sequence** | port log `:34881-:34885` — `optimal settings cb (dlssEntry:819): 3840x2160 pq=3 -> 1280x720 scale=0.3333` / `pq=0 -> 1920x1080 0.5000` / `pq=1 -> 2258x1270 0.5882` / `pq=2 -> 2560x1440 0.6667` / `pq=3 -> 1280x720 0.3333` (@18,859-18,919 ms; re-probes @134,151 / 192,492 ms) — **confirms game requests Ultra Performance (pq=3, 33.3%)** |
| CreateFeature + chain | `[37351 ms] D3D12 CreateFeature: id=1 feature=1 preset=L 1280x720 -> 3840x2160 [diag-v1]` ; `[34774 ms] SR 1280x720 -> 3840x2160: creating the frame chain from /home/deck/DLSS-Vulkan-Linux/build/ngx-weights/L` |
| first eval | `[37370 ms] SR frame 0: 3 images in, 1280x720 -> 3840x2160, recorded in 1 ms [eval #1 id=1 diag-v1]` + `eval #1 OK` |
| throughput | **57,236 `eval #N OK`** over 37,370→678,120 ms ≈ **89 evals/s (~89 fps)**; ids 1 and 3 observed (`eval #41115 OK ... id=3` @678,120 ms) |
| diag-d5 (jitter/params) | first: `jx=-0.250000 jy=0.166667 prev=(0,0) mv_scale=1.0000 reset=1 pre_exp=1.0000` @22,697 ms; last sampled: `jx=0.281250 jy=-0.092593 prev=(-0.218750,-0.425926) mv_scale=1.0000 reset=0` @578,300 ms — app jitter flows, prev-pair tracks, mv_scale constant 1.0 |
| user verdict | SR renders (**not black**) but "does not apply the proper algorithm" (visual) — quantitative A/B PENDING master-2's PSNR cell config (executor runs nothing until relayed) |
| RR = KNOWN-OPEN (deferred by user, NO action taken) | #3 log carries **1,312 × `commonEntry.cpp:899[evaluateNGXFeature] [sl.dlss_d] NGX evaluate feature failed 0xbad00000`** (samples 18:19:45, 18:19:50) ⇒ the game IS issuing RR evaluates; they fail at the DLSSD context/evidence layer ⇒ RR black persists one layer below the cleared startup gates. Recorded as open item; RR untouched per user instruction. |
### D5.3 S3 dump capture + measured eliminations (2026-10-31 18:38-18:41, read-only capture)
| row | evidence |
|---|---|
| P-S3MV (staging, executor re-verified) | commit `338d902`; `bin64/nvngx.dll` signed **md5 `8faa241dd113a55a64f72c0a51a6f908`, 880,040 B**; backup chain now `.bak-pre-s3mv` = `097a802d` (S2 stage) + `.bak-pre-cbpop`/`.bak-pre-srkey`/`.bak-g1-genuine`; marker `bin64/dlss5-amd-dump.enable` armed 18:38 (0 B) |
| D5.3-S3-CAPTURE (user quit+relaunch observed mid-poll; game fresh 00:04→) | **dump set written at chain frame 2**: `dlss5-amd-dump_in.bin` **33,177,600 B** md5 `4973dcf9abf0a431` (1920×1080×16 RGBA16F) @18:40:59; `dlss5-amd-dump_mv.bin` **16,588,800 B** md5 `4c69b24294fbcbea` (1920×1080×8) @18:41:00; `dlss5-amd-dump_dep.bin` **8,294,400 B** md5 `1a11eb994502d39d` (1920×1080×4) @18:41:00; `dlss5-amd-dump_out.f32` **132,710,400 B** md5 `b788b936e1de9d85` (3840×2160×16) @18:40:59. Port log verbatim: **`[68948 ms] dumped chain out/in of frame 0 (out 3840x2160, in 1920x1080) + mv/dep planes`**. Marker DELETED post-capture per S3 instruction (no re-dump on next launch unless re-armed). Byte analysis = top-level's lane (their relay) |
| ELIMINATED-MEASURED (top-level, folded as ordered) | (a) jitter correct per frame: **657 `[diag-d5]` lines**, Halton-like ±0.5 with `prev=` trails; (b) `mv_scale=1.0000 pre_exp=1.0000`; (c) **eval fingerprint byte-identical to the 10-30 `slgate-fail` archive** (Color `vk=97 dxgi=10`, MV `vk=103 dxgi=16`, Depth `vk=130 dxgi=19`, `fmt col=0 mv=1 dep=0`); (d) game `dlss5-amd/shaders` vs `build/shaders` **md5-identical ×65**; (e) no shader source change since **10-30 13:04**; (f) weights repack stable since **10-24**; (g) **H1 ("3.0x never validated") REFUTED** by D1 A06 row (REPORT :204) — exact in-game geometry (1280x720→3840x2160 UltraPerf ×3) already passed headless. pq-probe + S2 rows at REPORT :878-:883 stand |
| standing | master-2 SR-quality analysis in flight (enum/preset-12/source-history); NeuralHub subagent in flight; **RR parked by user decision**; single-runner = no harness runs by me |
### D5.3 S3 byte-analysis + S4 staging & capture (2026-10-31 18:41-19:1x)
| row | evidence |
|---|---|
| S3-BYTES (top-level analysis, folded) | `dlss5-amd-dump_mv.bin` 1920×1080 RG32F = **86.9 % non-finite channel words** (histogram ≈14.3k `-inf` / 42.8k `+inf` per 57k sample; first pixels `(-inf, inf)`); finite content only in a mid-frame band; finite magnitudes pixel-scale `x=-484..511, y=-14..0.18` = OUR convention (`check_mv.py`: `prev = cur + mv`, pixel units); Depth in same dump **clean [0,1] frac=1.0**; Color/out fine. Cause chain: `sl.common` carries `consts.motionVectorsInvalidValue` (game-declared sentinel) which `sl.dlss` never transfers ⇒ raw sentinels entered `g_mv` → f16 inf → history propagation ⇒ user's blocky/shimmer/no-AA. **Why D1/A06 missed it: the oracle corpus never carried non-finites.** |
| P-S4 (staging, executor re-verified) | commit `6c8aed3`; `bin64/nvngx.dll` signed **md5 `a8a8b9f1373f3c55b827e6724856fd7a`**, backup `.bak-pre-s4mv` = `8faa241d` (S3) ✓; **game-dir shader refresh**: `dlss5-amd/shaders/sr_gather.spv` = `9d5b403ce1490b91` and `sr_gather_mv.spv` = `863d9f8b37ddfa71` — **both md5-identical to `build/shaders/` (SYNC confirmed)**; mirror-sync fact: all 65 game↔build `.spv` pairs were md5-identical BEFORE this refresh (port prefers game-dir per `pipeline.cpp:796`), so only these 2 changed content-wise with S4. Marker `dlss5-amd-dump.enable` still armed. RR shaders untouched (parked). |
| D5.3-S4-CAPTURE (partial — session predates S4) | Poll 19:0x for 240 s on the RUNNING session (etime 27:45→31:38 ⇒ started ≈18:40 = the S3 session): **`[S4] mv frame` lines = 0, `dlss5-amd-dump_mv_f10/_f60.bin` = absent** despite 203,685 OK evals (far past frame 60) ⇒ current process still runs the S3 build/shaders; **S4 needs the next relaunch** (marker+files+new dll staged and verified). Session sanity: evals streaming ≈660/10 s ≈ 66/s, total 203,693 OK; `D3D12 CreateFeature id=1 feature=1 preset=M 1920x1080 -> 3840x2160` @23,068 ms + second create `id=3 preset=M` @196,295 ms (**note: render/preset now 1920x1080/M — changed vs session #3's 1280x720/L; flagged for the quality A/B attribution**). Visual verdict gate = user (pending). |
### D5.3 — CORRECTION + S5 staging + S4 gate capture + RR known-cause (2026-10-31 22:29-22:4x)
| row | evidence |
|---|---|
| **CORRECTION (supersedes the earlier "H1 REFUTED by A06 :204" claim — that claim was WRONG)** | master-2 proved A06/row:204 numbers are **Millisecond TIMINGS** (paired with 6.8823 etc = pass times), not PSNR ⇒ **H1 CONFIRMED: 3.0× has NEVER been PSNR-validated** (tools/verify PSNR cells cover only 1.5×/2.0× for M and L: `M{1.5,2.0}, L{1.5,2.0}`). Also folded: preset hint **12 = Preset L** (`ngx_abi.h:45`; read site `nvngx.cpp:1586`, origin `4739ca6`); pq enum verified; the sessions in play are **preset=M 1920×1080 = 2.0× = the validated cell** (⇒ L-out-of-matrix does NOT explain tonight's static shimmer); **retracted suspects**: `4b4651a` (numbers are ms, not coefficients), `a4d5fc1` (formats = validated set), `894ce54` (diag-only). Full analysis: `/tmp/opencode/sr-quality-analysis.md` (master-2) |
| P-S5 (staging, executor re-verified) | commit **`834daf1`** (observability only): port now reads canonical **`MV.Scale.X`** (fallback Y, legacy) + **`Sharpness`** with set-flag into d5 diag. Plumbing bug **PROVEN**: harness writes `MV.Scale.X/Y` at `opti_client:1048-56`, `sl.dlss` owns those strings, port read the **non-existent `MV.Scale`** (my earlier H3 probe ⇒ the `mv_scale=1.0000` readings were defaults, not app values). Staged: `bin64/nvngx.dll` signed **md5 `b623c7ca42a246217580a437b4b719d2`**, backup `.bak-pre-s5` = `a8a8b9f1` (S4) ✓, marker re-armed ✓. `dump_probe_on()` was LATCHED FALSE in the pre-S5 process (marker absent at start) ⇒ dumps needed a fresh process — which the user then did (captured below) |
| D5.3-S4-GATE (fresh session captured live) | Relaunch observed mid-poll (etime dropped → new process). **Sanitizer VERIFIED: `S4 mv frame 10: 4147200/4147200 channel words finite (100.0%) -> dlss5-amd-dump_mv_f10.bin [S4]` @65,533 ms; `frame 60: 4147200/4147200 ... (100.0%)` @71,866 ms; third at 1280×720 `1843200/1843200 ... (100.0%)` @215,369 ms** — i.e. **100.0 % finite vs S3's 86.9 % non-finite**. Dump files: `dlss5-amd-dump_mv_f10.bin` + `_f60.bin` 16,588,800 B each (22:29:43 / 22:29:49) + fresh `dep.bin` 8,294,400 B (22:29:37). (Pattern note: log format is `S4 mv frame` — no bracket — which a `S4\]` poll missed; loose grep used.) |
| D5.3-S5-DIAG | New diag fields live: `d5 diag: ... mv_scale=1920.0000 mvY=-1080.0000 sharp=0.000(+1) ...` @161,542/162,553 ms (render 1920×1080) and `mv_scale=1280.0000 mvY=-720.0000 ... reset=1` @166,547 ms (render switched to 1280×720 mid-session) ⇒ canonical scale semantics = **X=width, Y=−height** (negative Y), app-set Sharpness **0.0 with set-flag present**. The pre-S5 `mv_scale=1.0000` lines are superseded as defaults-of-a-never-existing key. |
| RR-OPEN-KNOWN-CAUSE (recorded, parked per user) | master-2: RR blocker already identified — **1,312 × FAIL `id=2 feature=13, slot=DiffuseAlbedo`, `DXGI format is not one this bridge decodes fmt=28 (R8G8B8A8_UNORM)`**; port `extra[]` demands RGBA16F (`nvngx.cpp:1160`); fix = **accept R8G8B8A8 with conversion**. No action taken (RR parked by user decision). |
| pending | user visual verdicts: sharpness-slider A/B (running now) + **static-shimmer verdict after this S4/S5 relaunch** (sanitizer live since shader refresh; S5 diag live since b623c7ca). |
### D5 SR-QUALITY S6 — MV.Scale consumption + gameplay MV magnitude probe (2026-10-31 23:0x-23:3x)
| row | evidence / action |
|---|---|
| S6-CAUSE (root-cause evidence, measured) | Session 22:32 (bin64 `dlss5-amd-ngx.log` 145 MB): `d5 diag: ... mv_scale=1280.0000 mvY=-720.0000 ...` — SL writes `MV.Scale.X` = the MV plane WIDTH and `MV.Scale.Y` = the NEGATIVE plane HEIGHT = the UV-normalized-MV signature (1.0 = full plane, Y mirrored). Measured gameplay MV contents (`dlss5-amd-dump_mv_f10/f60.bin`, 1280×720 RG32F): **100.0 % finite, 96.9 % pixels nonzero, max \|v\| = 8.25e-5 raw** — fed as chain-plane pixels that is ≈0 motion ⇒ temporal history never reprojects ⇒ the user-visible shimmer/blocky/no-AA. Menu frame-2 dump same session: max\|v\| 1.26e-3 × 1280 = 1.6 px = plausible menu motion. S4's sanitizer was correct but orthogonal (its predicate — 86.9 % non-finite at menu frames — measured 100 % finite in gameplay); **units were the remaining gap** |
| P-S6 (commit, port+shader+probe+harness) | commit **`e30a1ca`** "port: S6 - consume MV.Scale.X/.Y in the gather decode + harness uv unit pressure". Port: `FrameParams.mv_scale` → `mv_scale_x/.y` (`pipeline.h`); `read_frame_params` reads `MV.Scale.X` (legacy `MV.Scale` as X fallback) + `MV.Scale.Y` independently, guard = `DlssNr_Pipeline_Dx12.cpp:123-128` (non-finite \|\| 0 → 1.0); d5 diag keeps its format, now from the real per-axis fields; **ONE `[S6] mv scale=(%g,%g)` line per feature** at the first non-identity frame. Chain: `sr_gather` push 88→96 B (`mv_scale_x@84, mv_scale_y@88`), `sr_gather_mv` stays 32 B with the free +16/+20 slots; shaders `sr_gather*.comp` convert every raw-MV decode branch by `mv_pair = clamp(san(raw)·scale, ±65504, 65504)` **before** the f16 cast (scale 1280 can overflow); enc0's hardcoded 3.0 output-ratio factor untouched — single application point; `sr_post` push unchanged (oracle ISA proves +0xf0 unread). Probe: S6 frames **300/900/1800** keep the S4 dump pattern (`dlss5-amd-dump_mv_f<N>.bin`) and log `maxAbs` + fraction `\|v\|>0.02`, all behind `dump_probe_on()`. Harness: **`--mv-units uv`** in `opti_client` (`MV.Scale.X/.Y = (float planeW, −float planeH)` exactly as in-game SL; `synth_mv` writes `(2/W, −0.25/H)` so `uv·scale` reproduces the proven `(2.0, 0.25)` px pan), threaded through `run_harness.sh`, mode in client.log/summary.txt + README row |
| S6-HARNESS (4 cells, all gates green) | build: `bash src/vk/build.sh` (mingw + toolchain glslang; in-repo toolchain verified) → `build/package/_nvngx.dll` md5 `8ccdc888d1d62a6fbb23366a888a9a8c`, `sr_gather.spv` `c8472fdff4612351eae6086dca5f2f1d`, `sr_gather_mv.spv` `6418b3dcd0f706c86255628dc5b7680a`. Pre-S6 snapshot: `build/optiharness/_nvngx-preS6.dll` `ddb24b44c2d1b8cce9c5e547127537cf` (= git-ignored build artifact, never committed). Cells (`DLSS5_AMD_DUMP=1`, dump files in staging; stage dir is wiped per run — dump_out copied out to `build/optiharness/s6runs/` first): **A new-DLL px** = `HARNESS PASS` g1 create=1 loaded=1 bad00000=0, **g2 ok=12/12 fail=0 'fmt col=0 mv=1 dep=0'×12**, g3 nz=1.0/12, g4 EVAL_MED 12.40 ms GPU_MED 8.66 ms, RESULT=PASS, **no `[S6]` line (identity — correct)**; **B new-DLL uv** = PASS same gates + **`[S6] mv scale=(1920,-1080)`** logged; **C pre-DLL uv (pre shaders — inconsistent-combo cell discarded, see note)**; **D pre-DLL px (pre shaders)** = PASS. Note: the first C attempt ran pre-DLL with NEW shaders (old host builds an 88-byte push; new shader reads to +95 ⇒ undefined push words) — redone with pre-S6 `.spv` momentarily overlaid from the game mirror into `build/shaders` (then restored, md5s re-verified) — control validity discipline |
| S6-PARITY (the decisive numbers, pure-python f16 comparator) | 3840×2160×4 = 33,177,600 f32 words per dump: **A(new px) vs B(new uv): max abs diff = 0.000e+00, mean = 0, differing words 0/33,177,600 (0.0000 %)** — BIT-identical, better than the required ≤2/65504-f16-step bound (uv·scale = (2/1920)·1920 exact). **A(new px) vs D(pre px): max abs 0.000e+00** — the (1,1) default is EXACTLY yesterday's behavior. **CONTROL B(new uv) vs C(pre uv): max abs 2.866e-01, mean abs 2.312e-03, 2,932,338 words differ (8.84 %)** — the harness discriminates the bug, and the fix's effect is measurable on the same cell. Comparator: `/tmp/opencode/s6compare.py` (no numpy on this host), input MV sanity: `A_in_mv.bin` first px (2.0, 0.25), max 2.0/0.25 over 2,073,600 px |
| P-S6-STAGE (game dir, executor re-verified) | Signed: `osslsigncode-bin sign -h sha256 -certs port-core.cert.pem -key port-core.key.pem` (2.15-dev `-certs` syntax) → `build/package/_nvngx.signed.dll` md5 `040c9870303d186dccd4e2396792f747` 881,008 B; digest verify **Current == Calculated `AFB1ED28…CE8` ✓** (self-signed verify error = expected). Game dir: only `pgrep`-verified NO CrimsonDesert.exe / NO wineserver at staging time (only the checking shell matched); backup `bin64/nvngx.dll.bak-pre-s6` **= `b623c7ca` (S5) cp -p ✓**; atomic swap (`.nvngx.dll.s6tmp` → `mv`); installed `nvngx.dll` md5 `040c9870303d186dccd4e2396792f747`, `cmp` IDENTICAL to signed source. **Shader mirror**: full diff game↔build = exactly 2 files (`sr_gather.spv`, `sr_gather_mv.spv`); backups `bin64/dlss5-amd/shaders.bak-pre-s6/` = `9d5b403c`/`863d9f8b` (S4-era); installed = build md5 `c8472fdff4612351eae6086dca5f2f1d` / `6418b3dcd0f706c86255628dc5b7680a`, cmp ✓. No other game file touched. Marker `bin64/dlss5-amd-dump.enable` already armed (22:23) |
| pending | user relaunch + DLSS toggle + a MOVING camera scene: S6 probe stats for frames 10/60/**300/900/1800** decide the UV-unit hypothesis — gameplay `maxAbs` ~1e-2..1e-1 (raw UV: 1 px = 7.8e-4 on a 1280-wide plane) = units confirmed and fix in-path; pinned ≤1e-5 across pans = SL's MV transfer itself broken upstream (next target). If puzzles persist visually, replay S4 vs S6 A/B via `nvngx.dll.bak-pre-s6` |
