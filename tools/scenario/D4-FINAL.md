# D4 — final consolidated report (skeleton: every input exists except D3)

**Owner:** `ses_e6f7044eeffeSSITmFI4a0g4Mb` (master-2). **Assembled 2026-10-31 ~11:0x EDT, no runs.**
**Status:** D1 DONE (33/33 PASS) · D2 closed to "SL gate cracked headlessly" · **D3 = the only slot
below that is unfilled** · D4 = this document. The running log this document consolidates is
[`REPORT.md`](REPORT.md); the machine-readable matrix is [`D1-MATRIX.csv`](D1-MATRIX.csv).

> FILL RULE: the D3 section is the executor's (he fills it or hands rows to top-level). Every other
> section is final unless a D3 result contradicts it — if it does, fix the contradiction here and in
> `REPORT.md`, never by deleting a measured row.

---

## 1. Executive summary

1. **The port is exonerated for both original failures.** In-harness g1–g4 PASS, 12/12 (+60-frame)
   OK evaluates per run, `fmt col=0 mv=1|10 dep=0`, and the two in-game issues (Depth-slot refusal;
   XeSS-black from a forced `Dx12Upscaler=dlss`) are reproduced *without the game* by staged
   archives (`README.md` § reproductions).
2. **SR is measured end-to-end through OptiScaler on this machine**: 6 quality levels × 3 reps,
   identity-gated (no silent fsr21 substitution anywhere in the digest's current-era reps).
3. **RR (feature 13) is served by the port**: create + RR chain (`.../ngx-weights/rr`, preset F,
   28 pipelines) + 60/60 evaluates with all four synthetic inputs bound — output correctness is
   **ORACLE-PENDING** (§6).
4. **The Streamline doom loop is cracked headlessly**: after R1 + the L-C `nvngx.dll` staging,
   the game's own SL v2.14.1 stack boots green on **all three** `--sl` cells —
   `g5 SL boot : PASS c1..c5=ok`, `sl.dlss` (id 0) *and* `sl.dlss_d` (id 1001, adapter mask
   **0x1**) loaded and in the execution order (`P100 - sl.dlss_d`), zero
   `Failed to initialize NGX` / `Missing NGX context` / `0xbad00002`, while the port does 60/60 SR
   *and* 60/60 RR evaluates in the same process (§ `REPORT.md` § P-RECTOR).
5. **The SL path costs nothing measurable**: vs non-SL twins, all EVAL/GPU deltas sit below the
   sentinel floors (§3); L-C's 56.9 ms did not reproduce (max frame 8.8762 ms).
6. **What remains:** the in-game shot (D3, §5) and the RR referee (§6).

### The D3-grade claim, stated precisely (one sentence)

E01/E02/E03 prove that **inside one Wine process the game's own Streamline stack — same 11
`sl.*.dlls`, same app-id 101640911, same prefs — boots, trusts an NGX core, reports SR *and* RR
feature-available against caps 999.99/≥512.15 (`sl.dlss` mask 0x1 + `sl.dlss_d` mask 0x1), and
serves 60/60 SR (dlss and xess hosts) and 60/60 RR evaluates through our port**, while what still
needs the in-game shot is that *Crimson Desert's own host* reaches that same state under the real
launch line — its own `slInit` preferences and host callbacks, its menu options actually flipping
`_enableRayReconstruction`, and the renderer's real resource bindings feeding the RR inputs — which
only a game session can show.

---

## 2. D1 results — 33-cell digest (authoritative table lives in `REPORT.md`)

*Authoritative 33-row table: `REPORT.md` § "D1 FINAL RESULTS DIGEST" — note its superseded-row
markers: E01/E02/E03 rows there read "FAIL (G-RR-ASSURE only)" and are **superseded** by
`REPORT.md` § "P-RECTOR" (now PASS); B03/C01/C03/E02 keep their pre-N4/pre-N5 BLOCKED reps as
provenance.*

Consolidated shape (median of run-medians, frames 1..59, frame 0 excluded everywhere):

| axis | cells | verdicts | headline numbers (Quality res, 2560x1440→4K) |
|---|---|---|---|
| SR / dlss (our port) | A01–A06 (+sentinel A03 ×4) | PASS ×19 runs | EVAL 4.4595 · GPU 10.6140 *(digest's 4-rep A03)* |
| RR / dlssd (our port) | B01–B06 | PASS ×18 runs | EVAL 7.0358 · GPU 24.7600 |
| type: fsr22 | C02 + D06–D10 | PASS ×19 runs | EVAL 0.0945 · GPU 1.7747 |
| type: xess | C01 + D01–D05 | PASS ×19 runs | EVAL 0.1832 · GPU 3.8677 |
| type: fsr31 | C03 + D11–D15 | PASS ×19 runs | EVAL 0.1270 · GPU 2.1564 |
| SL diagnostics | E01 ×2, E02 ×3, E03 ×3 (8 runs on disk) | **4 current-era PASS** post R1+L-C (`E01/rep2`, `E02/rep3`, `E03/rep2+rep3`); earlier 4 = pre-fix FAIL/BLOCKED provenance, kept | 4.4290/10.5915 · 7.0051/24.4524 · 0.1833/4.0977 (PASS reps, frames 1..59) |

Noise floors (2× sentinel range, the gate on every effect claim, G-EFFECT): A03 `0.165/0.130`,
C02 `0.0046/0.469`, C01 `0.007/0.908`, C03 `0.006/0.265`, B03 `0.198/0.496` (EVAL/GPU ms).
**Consequences already stated in the digest:** dlss CPU-side differences are below floor (unranked);
xess's whole GPU axis is below its own 0.908 floor (unranked, no resolution trend — flagged, not
claimed); GPU resolution scaling is resolvable for dlss (19.71→5.51) and partially for fsr22/fsr31;
the port costs ~47× CPU / ~6× GPU vs fsr22 at equal res with the not-like-for-like caveat and the
`--direct` A/B attribution (mechanism unrooted ⇒ lever-round candidate).

## 3. What SL costs (the watch item, decided)

E01 vs A03, E02 vs B03, E03 vs C01: Δ EVAL/GPU = −0.0305/−0.0225, −0.0307/−0.3076, +0.0001/+0.2300 —
**all below their sentinel floors ⇒ no measurable per-frame cost from booting Streamline**; L-C's
56.9 ms (12f) not reproduced (E01/rep2 max frame 8.8762, first-11 median 4.4024).

## 4. Lever provenance chain (P1..WARN, newest of each class; full detail in `REPORT.md`)

| id | change / experiment | status & verdict |
|---|---|---|
| P1 | game `OptiScaler.ini` `[Spoofing] DxgiFactoryWrapping=true→false` (`.bak-prelever1`) | **applied, in-game verdict = D3 slot**; in-harness it is a no-op (defaults false) |
| P2 | `bin64/dxgi.dll.optiscaler-off → dxgi.dll` (rename back) | done (baseline run finished); no data change |
| — | falsified earlier: `[NvApi] OverrideNvapiDll=false`, `[Spoofing] Registry=true`, env `DXVK_NVAPI_DRIVER_VERSION` | **dead levers** — don't retry |
| L1–L3 | harness lever matrix incl. fakenvapi staged (`nvapi64`): caps 999.99, arch `0x190` | recorded in `REPORT.md` § lever matrix; game-side twin of L3 not yet exercised |
| L4 | staged `nvngx_dlss*.dll` ⇄ our port substitution | **`no change`** — SL asks nothing of the module at that time (+0.13 s vs first load +6.7 s) ⇒ killed H-1 as a stageable export fix |
| L5, L9–L14 | h1prime source/strace cycle (LogLevel=2, template restored byte-identical, `.bak-L5/L9` kept) | executor's; see §4b pointer |
| **L-C** | stage `<appdir>\nvngx.dll` = byte-copy of the game's **signed** `nvngx_dlss.dll` (SL's pre-LoadLibrary gate = WinVerifyTrust) | **WINNER**, first-class in `a349cfb`; flips c2/c4 + pulls `sl.dlss_d` |
| **R1** | port PE version resource (`1fd8c6c`) | necessary (L15: resource-less core refused pre-load) but **not sufficient** alone; combined with L-C ⇒ green |
| N1/N2 | `run_harness.sh --ini`, staged `nvngx_dlssd.dll` (`3795535`) | landed ⇒ 19 cells unblocked |
| N4 | four RR input slots in `opti_client.cpp` (`9fc35a5`) | landed ⇒ RR cells unblocked |
| N5 | stage upscaler companions (`b685fcf`) | landed ⇒ xess/fsr31 cells became real |
| **WARN** | E03/rep2 stage-race: post-run listing lacked `nvngx.dll` while Opti's log proves it present at runtime | benign race on shared `$STAGE`; `run_cell.sh` now WARNs (evidence: Opti log line + rep3 confirmation) |

### 4b. H-1′ appendix pointer (source kept by the executor)

`/tmp/opencode/h1prime-report.md` — **the executor keeps that file**; this report cites its §4
(acceptance greps), which D1 executed in `REPORT.md` § P-RECTOR: all six negative greps 0× in
E01/rep2, E02/rep3, E03/rep2 (+ rep3 confirm), presences `Loaded plugin 'sl.dlss_d' … mask 0x1`,
`P100 - sl.dlss_d`, `NGX loaded - app id 101640911`; `Loaded NGXCore from path` absent (its §4
marks that line conditional), and the `WINEDEBUG=+loaddll` half remains the executor's evidence
(the D1 runner does not set it). If that file moves, re-home this pointer — the acceptance *results*
are duplicated here so they survive.

## 5. ▓▓ D3 SLOT — IN-GAME EVIDENCE (EXECUTOR TO FILL) ▓▓

Status: **PENDING**. Rows below are the required evidence; each needs `file:line` or a log quote.

| # | claim to prove in-game | evidence required | status |
|---|---|---|---|
| 1 | launch line final state unchanged | `userdata/49277565/config/localconfig.vdf:2894` exact string = `WINEDLLOVERRIDES=winmm,dxgi=n,b PROTON_FORCE_NVAPI=1 DLSS5_AMD_WEIGHTS=/home/deck/DLSS-Vulkan-Linux/build/ngx-weights %command%` | PENDING |
| 2 | caps live (the P1 lever's whole point) | `OptiScaler.log` `getSystemCaps` — driver ≥ 512.15, **not** `0.0.0`; no `NVAPI failed to initialize` | PENDING |
| 3 | SR available & rendering | `Loaded plugin 'sl.dlss' … adapter mask 0x1`; port `eval #K OK: … feature=1` counts; no black screen (game frame / user report) | PENDING |
| 4 | RR available (menu path) | menu/save: `_enableRayReconstruction` present **and** set in `user_engine_option_save.xml`; Opti `Creating DLSSD feature` / `GetFeature("dlssd")`; `Loaded plugin 'sl.dlss_d' … adapter mask 0x1`, **no** `Ignoring plugin 'sl.dlss_d'` | PENDING |
| 5 | SL doom trio stays absent in-game | zero `Failed to initialize NGX` · `Missing NGX context` · `0xbad00002` in the session log | PENDING |
| 6 | RR actually evaluates with the game's real G-buffer | port `eval #K OK: … feature=13 n=7` (or the game's real slot set) — non-black, still ORACLE-PENDING per §6 | PENDING |
| 7 | timing sanity vs harness baselines | optional: one in-game EVAL/GPU figure, compared against §2 without claiming harness/game equivalence | PENDING |

*(Rows 1-7 are the template; attach `bin64/runs/<date>-*` log names or screenshot paths in the
status column as they land.)*

## 6. RR output verification — scope line (final-report wording, unchanged from `REPORT.md`)

*"D1 establishes RR presence, correct-input wiring, and ms costs; RR output fidelity is pending an
out-of-tree referee against the same synthetic inputs, and is explicitly not claimed."* (Why: the
inputs are frame-independent synthetics — analytic unit-normal gradient, DiffuseAlbedo 0.5,
SpecularAlbedo 0.04, Roughness 0.5 — the frozen `tools/verify/**` referee drives the native runner
path, not the Opti-in-the-loop path; NVIDIA's `nvngx_dlssd.dll` cannot run here; no analytic truth
exists. Interim referee-free check specified in `REPORT.md`: `KEEP_FRAMES=1` byte-stability across
RR reps — catches nondeterminism, not wrongness.)

## 7. Open items / explicitly NOT claimed

- In-game SR+RR availability → §5 only. RR quality → §6 only.
- xess GPU quality axis: below its own noise floor ⇒ unranked (plus no resolution trend observed —
  a question for a later session, not a result).
- Port evaluate cost vs Opti-internal backends: ~47×/~6× observed, `--direct` A/B attributes the
  CPU half to the port, **mechanism unrooted** ⇒ candidate lever round.
- Harness `g1`/`g3` still fail non-dlss cells by construction (literal + warmup) — informational
  ask to the harness owner; scenario gates are authoritative (`D1-GATES.md` § Harness mapping).
- Shared-prefix Proton version flip (`GE-Proton11-6 → 10.1000-105`, `Prefix has an invalid
  version?!`, `runs/A03/rep1/run.stdout.log:1-2`) — flagged for the executor's next game launch.
- Upstream OptiScaler: newer builds exist (nightly channel) but the 5-surface diff says **don't
  upgrade as a lever** — `REPORT.md` § reference package, verdict + obtain paths.
