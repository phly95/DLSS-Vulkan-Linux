# D1 — gate definitions

Owner: `ses_e6f7044eeffeSSITmFI4a0g4Mb`. Companion to `D1-DESIGN.md` and `D1-MATRIX.csv`.
Evaluator: `tools/scenario/gate_cell.sh <repdir> <backend> <frames>` (module from
`run_cell.sh`, can be re-run over any archived rep dir).

Conventions:

- **G-**, cell-level gates (every rep, evaluated in the order below).
- **harness gates g1-g5** (`tools/optiharness/run_harness.sh:244-328`, `opti_client`'s own
  `GATE eval/nonblack/timing/port`) keep running underneath every cell. Where they diverge from a
  scenario cell, **the scenario gate is authoritative** and the divergence is written down explicitly
  (see "Harness mapping").
- Every gate line printed: `GATE <name>: PASS|FAIL|BLOCKED — <evidence>` where evidence is a quoted
  log string or a count, so the assertion can be re-checked by hand (claim rule: file:line or log
  quote for anything asserted).

---

## G-ART artefact integrity (run before anything else)

- `<repdir>` contains `OptiScaler.log`, `dlss5-amd-ngx.log`, `client.log` (or `run.stdout.log`),
  `summary.txt`, `eval_ms.csv`, `stage_manifest.txt`.
- `OUT_NZ` line count == `frames` (`run_harness.sh:288` parses the same token).
- `eval_ms.csv` has `frames` rows (+ header).
- FAIL ⇒ nothing downstream of it is trusted (a truncated copy-out is indistinguishable from a
  short run).

## G-IDENT backend identity (primary anti-fallback gate — must precede any ms number)

Rationale: OptiScaler silently falls back to `fsr21` when the requested provider fails and then
writes that choice back into its config (`/tmp/opencode/OptiScaler/OptiScaler/upscalers/`
`FeatureProvider_Dx12.cpp:57-60` and `:69-73`, log line `LOG_INFO("Creating new {} upscaler",
State::Instance().newBackend)` at `:177`). So "output is non-black" never implies "our backend ran".

| check | expression | source of truth |
|---|---|---|
| requested backend announced | `grep -cF "Creating new <backend> upscaler"` `OptiScaler.log` >= 1, where `<backend>` ∈ {dlss, dlssd, xess, fsr22, fsr31} | `FeatureProvider_Dx12.cpp:177`; harness's own use of the same literal at `run_harness.sh:250` |
| no silent substitution | `grep -cE "falling back to FSR|Upscaler can't created"` == 0 | `run_harness.sh:253` |
| our port was the DLSS provider (dlss/dlssd cells) | `grep -cE "NVNGXProxy::InitNVNGX Loaded from .*_nvngx\.dll"` >= 1 | `run_harness.sh:251` |
| Opti's NVIDIA gate held | `grep -cF "Running on Nvidia"` >= 1 (evidence line, not pass/fail: `isNvidia()`, `dllmain.cpp:1978`) | `run_harness.sh:252` |
| FS fallback config rewrite absent | no `Dx12Upscaler=fsr21` (or changed value) in the manifest's ini copy | `FeatureProvider_Dx12.cpp:76`, `:84-86` |
| model DLLs sized correctly in stage | `stage_manifest.txt`: `nvngx_dlss.dll` == 58 956 912 B (and, for RR cells, `nvngx_dlssd.dll` == 48 343 664 B). Wrong/absent ⇒ `BLOCKED` for RR cells or `FAIL` for SR cells, never a silent "SR outage". | port's own gate `src/pe/nvngx.cpp:187-191`; L4 side-finding: a plain `cp` of the port over `nvngx_dlss.dll` produced `CreateFeature refused: nvngx_dlss.dll is missing or not its expected size` (L4a), padded copy green |

Expected-backend cross-check: the cell's `<backend>` comes from `D1-MATRIX.csv`, never from the
log — otherwise the gate degenerates to "whatever ran, passed".

## G-PORT the port's own verdict for every frame

Same numbers the harness g2 uses (`run_harness.sh:267-284`):

- `grep -cE 'eval #[0-9]+ OK:' dlss5-amd-ngx.log >= frames`
- `grep -cE 'eval #[0-9]+ FAIL:' == 0`
- `grep -cF 'fmt col=0 mv=1 dep=0' >= 1` (default `rg32` MV ⇒ re-derived code 1;
  `MV_CODE` logic `run_harness.sh:239-242`; port triplet emitter `nvngx.cpp`)
- `grep -cF "Depth: DXGI format 19 is not one this bridge decodes" == 0` (issue-1 refusal must not
  be present, `README.md:64-65`)

### G-PORT-RR (RR cells only — `backend == dlssd`)

- every frame's OK line must be the RR session: `grep -cE 'eval #[0-9]+ OK: id=[0-9]+ feature=13'
  >= frames` (feature id 13 = `NVSDK_NGX_Feature_RayReconstruction`, `ngx_abi.h:31`; emitter
  `nvngx.cpp:1248` `eval #%llu OK: id=%u feature=%d ...`).
- the RR chain must have been built from the rr weights: `grep -E 'RR [0-9]+x[0-9]+: creating the
  frame chain from .*\/rr'` present (`feature_name()` returns `"RR"` for feature 13,
  `nvngx.cpp:425`; weights selector `root + (rr ? "/rr" : ...)` `nvngx.cpp:443`; tree exists
  `build/ngx-weights/rr`).
- companion DLL present and correct size in the stage manifest: `nvngx_dlssd.dll` == 48 343 664 B
  (`src/pe/nvngx.cpp:187-189`). **Missing → gate yields `BLOCKED-N2`, not FAIL** (environment
  not yet provisioned — never report a provisioning gap as a product result).

### Re-classification rule (added after the first RR cell): `rr_gap` ⇒ BLOCKED(N4), not FAIL

Measured 03:09 on B03/E02: the port is asked for `feature=13`, **creates** it and builds the RR
chain from `.../ngx-weights/rr` (create line `D3D12 CreateFeature: id=1 feature=13 preset=F`), but
the harness supplies only Color/MotionVectors/Depth, so every evaluate refuses at the port's first
mandatory RR input — `slot=GBuffer.Normals why=GetD3D12Resource returned null for this key`
(`src/pe/nvngx.cpp:1133-1136`: `GBuffer.Normals` R16G16B16A16_SFLOAT, `GBuffer.Roughness`
R32_SFLOAT). `gate_cell.sh` therefore computes `rr_gap` (chain built **and** `null`-refusals ≥
frames **and** zero OKs) and reports G-PORT / G-PORT-RR / G-OUT / G-TIME as **BLOCKED**, with the
refusal's ms explicitly excluded from any table (EVAL ~2.5 ms = Opti entry, GPU ~0.002 ms = no GPU
work). Gate precedence: a real FAIL from another gate still wins, so this can never mask a defect
(`gate()` keeps `RC=1` over `RC=3`). Not triggered by a *create* failure — that stays FAIL.

### G-PORT-SR (RR=off cells)

- no RR evaluates leaked in: `grep -cE 'feature=13'` == 0 in `dlss5-amd-ngx.log`
  (otherwise a "SR" cell silently measured RR).

## G-OUT the upscaled output is real

- every `OUT_NZ` line has `nz >= 0.10` (`run_harness.sh:295-301`; justification: healthy cells sit
  at `nz ≈ 1.000000`, threshold is a "never written" detector).

## G-TIME timing present and sane

- `EVAL_MS` n >= frames-1 and median > 0 over frames 1.. (`run_harness.sh:317-323`)
- `GPU_EVAL_MS`: n >= frames-1 with median > 0 ⇒ `PASS`; all `n/a` ⇒ **`DEGRADED`** (recorded, not
  failed — contract `tools/optiharness/README.md:71-72`); mixed n/a ⇒ `DEGRADED` with counts.
- sanity-only (reported, never fails the cell): p95/median <= 3.0; any `nan`/`inf` token in the CSV
  ⇒ FAIL (encoding bug, not a result).

## G-RR-ASSURE RR-visibility diagnostics (Tier E `--sl` runs only)

The headless analogue of D3's in-game assurance; these reuse the harness g5 subchecks
(`tools/optiharness/README.md:73-85`) plus the RR one:

1. `getSystemCaps` present, no `NVAPI failed to initialize`; reported driver >= required one
   (`README:77-78`).
2. no `Failed to initialize NGX` (`README:79`).
3. no `requires driver 512.15.0 or newer - detected 0.0.0` unless detected >= required (`README:80-81`).
4. no `Missing NGX context` (`README:82`).
5. `Loaded plugin 'sl.dlss'` present (`README:83`).
6. **RR**: `Loaded plugin 'sl.dlss_d'` present **and** no `Ignoring plugin 'sl.dlss_d'` /
   `adapter mask 0x0` (the failure seen in-game, quoted in `REPORT.md` and `tools/scenario/`
   lever log). Emitted from the same gate family as check 5 — absence of loaded-plugin = FAIL for
   Tier E-RR rows, `BLOCKED-N2` when L4/substitution state is unknown (gate can consult
   `REPORT.md`'s current lever state, it is never guessed).

g5 joins the exit verdict only under `--sl` (`README:84-85`); scenario gates always apply 6 to
Tier E rows.

---

## Harness mapping (where the harness's own gates disagree with a scenario cell)

| harness gate | applies unchanged? | what the scenario does |
|---|---|---|
| g1 `Creating new dlss upscaler` (`run_harness.sh:250`) | **only for `dlss` cells** | for `xess/fsr22/fsr31/dlssd` cells the literal differs, so harness g1 fails *by construction* while G-IDENT passes. The cell records `harness_g1=FAIL(expected-literal-mismatch)` explicitly; harness non-zero exit is treated as expected for those cells, never as the verdict. |
| g1 `Loaded from ... _nvngx.dll` | **not asserted for xess/fsr cells** — those never touch `_nvngx.dll`; G-IDENT asserts it only for `dlss`/`dlssd` cells. | |
| g2/g3/g4 | yes, verbatim | mirrored as G-PORT / G-OUT / G-TIME with identical thresholds so a result can be double-read by both parsers |
| g5 | only for Tier E (`--sl`) | extended by G-RR-ASSURE as above |

## Cell verdicts

- `PASS` — G-ART, G-IDENT, G-PORT (+G-PORT-RR/-SR), G-OUT, G-TIME all PASS; G-TIME may be
  `DEGRADED`; RR rows also need G-RR-ASSURE in Tier E.
- `BLOCKED-<need>` — an environment prerequisite is absent (`BLOCKED-N1` ini knob never needed here —
  cells that need N1 are never *run*; `BLOCKED-N2` for the `nvngx_dlssd.dll`/L4 dependency). A
  BLOCKED cell contributes **no** number to any table.
- `FAIL` — a measured contradiction. Distinguished from BLOCKED so a provisioning gap can never be
  written up as a defect and a defect can never be hidden as "environment".

## G-EFFECT (cross-cell reporting rule)

Effect claim for difference `d` between two cells is allowed only after:

1. sentinel `A03` ran at least twice (start + close-out) or three times (every 5th cell);
2. `d > 2 * (max(sentinel run-medians) - min(sentinel run-medians))`;
3. both cells `PASS` with same `frames`, same `reps`.

Otherwise the line reads `below noise floor (sentinel spread X ms)` — no ranking, no "faster".