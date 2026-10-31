# D1 — scenario-replication design (upscaler type x quality preset x RR toggle)

Owner: `ses_e6f7044eeffeSSITmFI4a0g4Mb` (MASTER-2, `opencode/mimo-v2.6-flash-free`).
Status: **DESIGN FINAL / EXECUTION NOT STARTED** (all runs wait for a released slot; this session
does not launch anything — see "Slot discipline").

Every factual claim below carries a `file:line`. Anything not yet measured is marked **unproven**
or **blocked** — no result is pre-filled.

---

## 1. The question

Replicate the user-facing Crimson Desert scenario space — *upscaler type x quality preset x Ray
Reconstruction toggle* — in the headless harness, and produce methodology-grade millisecond
numbers per cell, so that:

- **A (assurance)**: each cell proves, by log identity, that the expected backend ran *through our
  `_nvngx.dll` port* (or, for the RR cells, through the RR create/evaluate path) — the same
  assurance D3 wants in-game, obtained without owning the game lifecycle.
- **B (performance)**: EVAL_MS (CPU wall) and GPU_EVAL_MS (GPU timestamp span) are reported per
  cell with a defensible statistic, an explicit drift control, and a stated noise floor.

The in-game half of D3 (`sl.dlss_d` joining, RR visible in the game menu) stays with the EXECUTOR;
D1 supplies the headless counterpart plus the gate definitions they can reuse.

## 2. Factors and levels (each grounded)

### F1 — upscaler type (`[Upscalers] Dx12Upscaler`, OptiScaler)

Accepted D3D12 values, exhaustive for this build (OptiScaler source `/tmp/opencode/OptiScaler`,
v0.9.4):

| value | evidence |
|---|---|
| `xess` | `OptiScaler/upscalers/FeatureProvider_Dx12.cpp:24-29` |
| `fsr21` | `:29-32` — **also the silent failure fallback** (`:57-60` unknown/backend-missing → `FSR2FeatureDx12_212`; `:69-73` module-not-loaded → `upscalerName = "fsr21"`) ⇒ never a planned cell |
| `fsr22` | `:34-37` |
| `fsr31` | `:39-42` |
| `dlss`  | `:47-51`, gated on `DLSSEnabled` + `NVNGX_DLSS_Path` |
| `dlssd` | `:52-56`, gated on `DLSSEnabled` + `NVNGX_DLSSD_Path` (resolved at `dllmain.cpp:1966/1968/2006`) |

A successful construct writes the answer back into the config, including the `dlssd` → `dlss`
rename (`FeatureProvider_Dx12.cpp:76`, `:84-86`), so the log line, not the ini, is the identity
record.

The key is read from the ini next to `dxgi.dll` (`Config.cpp:62 readString("Upscalers","Dx12Upscaler")`,
ini located at `Config.cpp:45`). Default is `xess` (`Config.h:392`); the harness ini forces `dlss`
(`tools/optiharness/OptiScaler.harness.ini` `[Upscalers] Dx12Upscaler=dlss`).

### F2 — quality preset (6 levels: Ultra Quality / Quality / Balanced / Performance / Ultra
Performance / DLAA)

The harness client derives `PerfQualityValue` from the output/render ratio, nearest-of-six
(`tools/optiharness/opti_client.cpp:734-739`):

```
{1.0, NGX_PQ_DLAA} {1.3, NGX_PQ_ULTRA_QUALITY} {1.5, NGX_PQ_MAX_QUALITY}
{1.724, NGX_PQ_BALANCED} {2.0, NGX_PQ_MAX_PERF} {3.0, NGX_PQ_ULTRA_PERF}
```

and pins every `DLSS.Hint.Render.Preset.*` to `NGX_PRESET_M` (`opti_client.cpp:742-743`), so the
preset axis is configuration-only: **no executable change, no `tools/optiharness` edit** — D1 cells
select the level by choosing `--w/--h/--ow/--oh`. Output fixed at 3840x2160 (the client's default
output, `opti_client.cpp:170`); render dims chosen even and nearest to each nominal ratio (DLSS needs even
dims):

| level | ratio (ration) | render | out | ratio check (ow/w) |
|---|---|---|---|---|
| DLAA | 1.000 | 3840x2160 | 3840x2160 | 1.0000 |
| Ultra Quality | 1.300 | 2954x1662 | 3840x2160 | 1.3000 |
| Quality | 1.500 | 2560x1440 | 3840x2160 | 1.5000 |
| Balanced | 1.724 | 2228x1254 | 3840x2160 | 1.7235 (nearest to 1.724; next is 1.5/2.0) |
| Performance | 2.000 | 1920x1080 | 3840x2160 | 2.0000 |
| Ultra Performance | 3.000 | 1280x720 | 3840x2160 | 3.0000 |

(Ultra Quality at 4K nominal is 2880x1620 = 1.333; we take 1.300 because the client's classifier
bin is 1.3 — the cell must land in the bin that sets `NGX_PQ_ULTRA_QUALITY`, and `opti_client.cpp:734-738`
picks the nearest of the six. This is recorded here so nobody "fixes" it to 2880x1620 later and
silently gets bin 1.300 vs 1.333 confusion: 3840/2880 = 1.3333, whose nearest bin is **also** 1.3 —
both dims are acceptable; 2954x1662 is used for exactness to the classifier constant.)

### F3 — Ray Reconstruction toggle (2 levels)

- **RR = off** → `Dx12Upscaler=dlss`: straight SR through the port (game scenario: DLSS on, RR off).
- **RR = on** → `Dx12Upscaler=dlssd`: OptiScaler's expressible "RR is active" form. `FeatureProvider_`
  then constructs `DLSSDFeatureDx12`, whose `_CreateFeature` calls
  `NVNGXProxy::D3D12_CreateFeature()(cmd, NVSDK_NGX_Feature_RayReconstruction, ...)`
  (`OptiScaler/upscalers/dlssd/DLSSDFeature_Dx12.cpp:52`) — i.e. **feature id 13**
  (`NVSDK_NGX_Feature_RayReconstruction = 13`, `tools/verify/ngx_client/ngx_abi.h:31`). Success is
  logged `_CreateFeature result: NVSDK_NGX_Type... Success` (`DLSSDFeature_Dx12.cpp:63`).
- The port handles feature 13 natively: session tag `RR` (`src/pe/nvngx.cpp:425`),
  RR weights `DLSS5_AMD_WEIGHTS/.../rr` (`nvngx.cpp:443`: `w = root + (rr ? "/rr" : ...)`;
  tree present: `build/ngx-weights/rr`), preset forced `'F'` (`nvngx.cpp:1440`), and its per-feature
  companion-DLL check expects **`nvngx_dlssd.dll` of exactly 48 343 664 B** next to `_nvngx.dll`
  (`nvngx.cpp:187-189`).

**Mapping question — SETTLED 2026-10-31 by the background probe** (`/tmp/opencode/d1-rr-mapping.md`,
read-only, every claim cited there): in-game RR is **simultaneous SR + RR**, not "`dlssd` as the
upscaler" — SR is created from `Dx12Upscaler` (`NVNGX_DLSS_Dx12.cpp:621-640`) and RR as an
*independent* NGX feature from `GetFeature("dlssd")` (`:642-652`), and OptiScaler exposes **no RR
enable key at all** (`[DLSSD]` is presets-only, `Config.cpp:300-333`). The game-side option is
`_enableRayReconstruction` (present as a string in `CrimsonDesert.exe`; **absent** from
`user_engine_option_save.xml`, so its stored state is unknown) → Streamline `kFeatureDLSS_RR = 1001`
→ shipped `sl.dlss_d.dll` → feature 13. `auto` never picks `dlssd` (`Config.cpp:1561-1562`).
Consequences for this design: (i) `Dx12Upscaler=dlssd` is kept as the RR=on level but is **labelled
an approximate encoding** in every artifact — a faithful replication of the game would need SR *and*
RR simultaneously, which `opti_client.cpp` cannot request today (it creates exactly one feature,
`NVSDK_NGX_Feature_SuperSampling`, `opti_client.cpp:759`); (ii) the RR cells' meaning is "does our
port serve feature 13", which is the assurance question underneath the game toggle.

## 3. Cell registry

Machine-readable: [`D1-MATRIX.csv`](D1-MATRIX.csv). Tiers:

| tier | cells | meaning | gate status |
|---|---|---|---|
| A | 6 | dlss x 6 quality x RR=off | **READY** (ini already `dlss`; executable in any released slot) |
| B | 6 | dlss x 6 quality x RR=on | **BLOCKED-N2** (runner must stage `nvngx_dlssd.dll`) |
| C | 3 | xess/fsr22/fsr31 at Quality, RR=off | **BLOCKED-N1** (runner needs an ini override) |
| D | 15 | xess/fsr22/fsr31 x the other 5 quality levels | **BLOCKED-N1** (release condition: Tier C spread > noise floor) |
| E | 3 | diagnostic `--sl` runs (game's Streamline stack booted headless) | BLOCKED-N2 for the RR row; READY otherwise |

Cell id: `<tier><nn>` (e.g. `A03`, `B06`). 33 rows total.

## 4. Measurement methodology (the "methodology-grade" contract)

1. **Frames**: `--frames 60` per run (harness default is 12, `run_harness.sh:48`). Statistics skip
   frame 0 repo-wide (`tools/optiharness/README.md:49-52`) ⇒ n = 59 samples per rep; Opti's 500 ms
   first `CreateFeature` sleep (`DLSSFeature_Dx12.cpp:38`, cited `README.md:51`) and the port's
   ~3.8 s chain build are outside the measure anyway (they are not in `EVAL_MS`).
2. **Repetitions**: **R = 3 independent runs per cell** (each run re-stages from scratch,
   `run_harness.sh:111 rm -rf "$STAGE"` ⇒ no log leakage, `README.md:260-261`). Reported number =
   median of the 3 run-medians; also min/max across reps. Per-run median/p95 come from
   `EVAL_MS` / `GPU_EVAL_MS` (`README.md:35-36`).
3. **Two clocks, always both**: `EVAL_MS` (QPC around `NVSDK_NGX_D3D12_EvaluateFeature`) and
   `GPU_EVAL_MS` (D3D12 timestamp pair, `README.md:35-36`). A cell is reportable only if both are
   present; `n/a` GPU stamps = degraded, reported, never silently dropped (contract `README.md:71-72`).
4. **Drift control (sentinel)**: cell `A03` (dlss / Quality / RR=off) is re-run at the start and
   after every 5th cell. The **observed sentinel spread is the noise floor**: no cross-cell
   difference is reported as an effect unless it exceeds 2x the sentinel max-min spread (coefficient
   defined in D1-GATES §"G-EFFECT"). Uncontrolled variables we cannot pin on this machine: GPU
   clocks/thermals, `wineserver`/prefix contention with the game (prefix clone path exists,
   `run_harness.sh:168-188`), and background GUI load. The sentinel measures their net effect rather
   than pretending they are absent.
5. **Inputs are deterministic**: `opti_client.cpp` contains no `rand/srand/mt19937` (grep, 0 hits) —
   inputs are synthetic gradient+checker indexed by frame number, so run-to-run input variance is 0
   and all spread is system noise.
6. **Identity before ms** (the anti-fallback rule): OptiScaler silently substitutes `fsr21` when the
   asked-for backend fails to construct (`FeatureProvider_Dx12.cpp:57-60`, `:69-73`) and rewrites the
   config to match (`:84-90`). A cell that measures without first passing the **identity gate**
   (D1-GATES G-IDENT) would report fsr21 as if it were DLSS. G-IDENT runs first and aborts the cell.
7. **Artefact retention**: every rep saves `OptiScaler.log`, `dlss5-amd-ngx.log`, `client.log`,
   `run.stdout.log`, `summary.txt`, `eval_ms.csv` + a stage manifest (`ls -l` with byte sizes) under
   `tools/scenario/runs/<cell>/rep<k>/` (stage dir is wiped by the next run, so copy-out is
   mandatory). `tools/scenario/**` is this session's area; `build/optiharness/**` and `tools/
   optiharness/**` stay the harness owner's.

## 5. Slot discipline (partition rules this design obeys)

- This session **never** launches the game, never writes `bin64/`, never runs the harness
  *concurrently* with the EXECUTOR's L4 experiment. The shared `STAGE` dir
  (`run_harness.sh:46`) means harness runs are serial-by-construction.
- `tools/scenario/run_cell.sh` is the only entry point it defines; it refuses to start when
  (a) `ps` shows a CrimsonDesert/GE-Proton wineserver on prefix 3321460, or (b) another
  `run_harness.sh` is live — both checks are hard-coded, fail-closed.
- Permanent Steam launch line stays untouched (final-state requirement, quoted in `REPORT.md`).
- Refusal is **fail-closed by construction**: since the two live checks above only read `/proc` and
  `pgrep`, they cannot be satisfied by a stale state file — and the third condition (game exited,
  executor quiet) is checked by the top-level "SLOT RELEASED" handshake, never self-judged.

## 6. Needs (things this design cannot unblock alone)

| id | need | owner | why |
|---|---|---|---|
| N1 | **LANDED 2026-10-31 02:4x (commit `3795535`)** — `run_harness.sh --ini <path>` stages a caller-chosen `OptiScaler.ini`, validated pre-stage (missing file / no `[section]` ⇒ `HARNESS FAIL: --ini:` rc 1), absent = byte-for-byte old behaviour, printed as `ini=<path>` | harness owner — done | tiers C/D unblocked. Cell inis live in `tools/scenario/inis/` (one per backend: `dlssd/xess/fsr22/fsr31.ini`), each a copy of the harness template with `[Upscalers] Dx12Upscaler` as the single difference; `run_cell.sh` passes `--ini` for every non-`dlss` cell. |
| N2 | **LANDED 2026-10-31 02:4x (commit `3795535`)** — `nvngx_dlssd.dll` symlinked at `run_harness.sh:117` beside `nvngx_dlss.dll` (`:115`), source missing ⇒ hard preflight die (`:102`) | harness owner — done | tier B/E-RR unblocked on staging. The port still demands exactly 48 343 664 B (`nvngx.cpp:187-189`) ⇒ `G-PORT-RR` verifies the size from a deref'd manifest; L4's side-finding (a plain `cp` of the port over `nvngx_dlss.dll` breaks SR, `nvngx.cpp:187-191`) is what `G-STAGE` guards. |
| N4 | **harness RR inputs**: `opti_client.cpp` must create and bind `GBuffer.Normals` (`VK_FORMAT_R16G16B16A16_SFLOAT`) and `GBuffer.Roughness` (`VK_FORMAT_R32_SFLOAT`) for `dlssd` cells (port's mandatory RR slots, `src/pe/nvngx.cpp:1133-1136`) | harness owner (`tools/optiharness/**`) | **measured 03:09 (B03/E02)**: RR *create succeeds* — `RR 2560x1440 -> 3840x2160: creating the frame chain from .../ngx-weights/rr`, `D3D12 CreateFeature: id=1 feature=13 preset=F`, chain ready 13 428 ms / 28 pipelines — then all 60 evaluates refuse `slot=GBuffer.Normals why=GetD3D12Resource returned null for this key`. Gate class = BLOCKED (harness input gap), never FAIL; the refusals' ms (EVAL ~2.5 / GPU ~0.002) are excluded from every table. Precedent: issue 1's Depth slot, which the harness likewise learned to feed (`README:96`). |
| N3 | released execution slot: harness agent (N1/N2) done **+** game exited **+** executor quiet; top-level sends "SLOT RELEASED" | top-level / EXECUTOR | **LANDED 02:49; wave 1 ran then** (02:49–03:11). |

## 7. Run plan (execution order once N1/N2/slots clear)

```
1.  sentinel A03 (establish noise floor)          x3 reps
2.  A01..A06  dlss x 6 quality (RR off)           x3 reps each  (READY now)
3.  sentinel A03
4.  B01..B06  dlssd x 6 quality (RR on)           x3 reps each  (needs N2)
5.  E01..E03  --sl diagnostics                     x1 rep each   (RR row needs N2)
6.  sentinel A03
7.  C01..C03  xess/fsr22/fsr31 @ Quality          x3 reps each  (needs N1)
8.  compute noise floor; if Tier C spread > floor -> D01..D15 (needs N1), else record "not
    released: below noise floor"
9.  final sentinel A03 -> close-out drift check
```

Estimated cost: ~90 s startup-dominated per run (Proton probe ~3 s + chain build 3.8 s + 60 frames
at ~4-8 ms) ⇒ on the order of 30-45 min for the whole plan; cells are independent, so the plan can
be executed in pieces without invalidating completed cells.

## 8. What would falsify this design

- Sentinel spread comparable to the effects we care about ⇒ numbers are not separable, report as
  "harness cannot resolve X at this resolution" instead of ranking cells.
- G-IDENT failing on a READY cell (fsr fallback under `Dx12Upscaler=dlss`) ⇒ the cell's own backend
  story is broken; halt Tier A.
- RR cells creating feature 13 but the port logging `eval ... FAIL` ⇒ the rr weights/shaders path
  (`nvngx.cpp:443`, `build/ngx-weights/rr`) has a gap; D1 records it, D2 owns the fix.