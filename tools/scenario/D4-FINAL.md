# D4 — final consolidated report (skeleton: every input exists except D3)

**Owner:** `ses_e6f7044eeffeSSITmFI4a0g4Mb` (master-2). **Assembled 2026-10-31 ~11:0x EDT, no runs.**
**Status:** D1 DONE (33/33 PASS) · D2 closed to "SL gate cracked headlessly" · **RE-SCOPED 2026-10-31
(user decision): the OptiScaler path is ARCHIVED for in-game use; the new target is standalone
(game direct + port) = D5 (§8 slot open, executor)** · D3 = LAUNCH-PATH-
BLOCKED after 3 attempts; G1 kept game-side with provenance; two approved one-shot levers (N-B, then
N-A) outstanding, hard stop after — see §5; its boxed closing sentence is **superseded-pending-D5** · D4 = this document.
The running log this document consolidates is
[`REPORT.md`](REPORT.md); the machine-readable matrix is [`D1-MATRIX.csv`](D1-MATRIX.csv).

> FILL RULE: the D3 section (§5) and the D5 section (§8) are the executor's (he fills them or hands
> rows to top-level). Every other section is final unless a D3/D5 result contradicts it — if it does,
> fix the contradiction here and in `REPORT.md`, never by deleting a measured row.

---

## 0. RE-SCOPE (2026-10-31, user decision — read before citing anything below)

The **OptiScaler path is abandoned for in-game use and is ARCHIVED as an experimental path.** What
survives, stated precisely:

- **Measured and still true (archived Opti-in-the-loop configuration):** the SL/NGX gate chain was
  cracked via **L-C + G1** (signed `nvngx.dll` byte-copy; game-side provenance kept) together with
  **R1** (port PE version resource); **D1's 33/33 methodology numbers stand** for exactly what they
  measured — OptiScaler-in-the-loop ms and gates on this machine — and every gate/identity rule
  (`G-IDENT` no-silent-backend-swap, warmup window, `mv=10` per feature 13, the four-RR-input
  binding check, all five sentinel floors) transfers to any configuration that reuses them.
- **Also measured, and the reason for the re-scope (executor/top-level brief, not independently
  re-measured here):** in-game the port produced **~17 000 OK SR evaluates through Opti** while the
  **user judged the image quality defective**, and **RR was black with `DLSSDContext 0xbad00000`**
  ⇒ the Opti path's *assurance* did not translate into user-acceptable in-game behaviour.
- **No longer describes the target:** the §2/§3 timings and gates describe the archived path. They
  remain valid as a baseline for comparing the standalone configuration (§8) and as methodology —
  not as a statement about what ships.
- **New target:** standalone (game direct + port), §8.

---

## 1. Executive summary

*(Findings 1–5 are **archived-path (OptiScaler-in-the-loop)** results unless stated; see §0.)*

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
6. **What remains:** the standalone target (D5, §8 — now the thing that ships), the in-game shot
   (D3, §5, launch-path-blocked), and the RR referee (§6).

### The D3-grade claim, stated precisely (one sentence)

E01/E02/E03 prove that **inside one Wine process the game's own Streamline stack — same 11
`sl.*.dlls`, same app-id 101640911, same prefs — boots, trusts an NGX core, reports SR *and* RR
feature-available against caps 999.99/≥512.15 (`sl.dlss` mask 0x1 + `sl.dlss_d` mask 0x1), and
serves 60/60 SR (dlss and xess hosts) and 60/60 RR evaluates through our port**, while what still
needs the in-game shot is that *Crimson Desert's own host* reaches that same state under the real
launch line — its own `slInit` preferences and host callbacks, its menu options actually flipping
`_enableRayReconstruction`, and the renderer's real resource bindings feeding the RR inputs — which
only a game session can show. *(This claim is scoped to the archived OptiScaler path, §0; the
shipping configuration is standalone, §8 — the same sentence must be re-proved there.)*

---

## 2. D1 results — 33-cell digest (authoritative table lives in `REPORT.md`)

**Archived-path scope:** everything in §2–§4 measures the *OptiScaler-in-the-loop* configuration
(§0) — the numbers are final for that configuration and are the **baseline** against which §8's
standalone measurements will be compared.

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

**STATUS (2026-10-31 11:1x, from the executor's report via top-level; evidence rows still his):**

- **G1 applied and kept game-side**: staged `nvngx.dll` = byte-copy of the game's signed
  `nvngx_dlss.dll` in `bin64/` — `cmp`-identical, Authenticode **Security Directory** verified,
  and the permanent Steam launch line byte-exact before/after (provenance = executor's report;
  not independently re-measured by this session).
- **In-game verification = LAUNCH-PATH-BLOCKED after 3 attempts**: Steam URL assert-wall ×2, plus
  one direct `ntsync` stall **pre-`slInit`** ×1 ⇒ the money-shot items (a)–(e) below were **never
  reached**, so no in-game log claim exists either way.
- **Two approved one-shot levers remain**, hard stop after: **N-B** (Proton 10.0 direct A/B, first)
  then **N-A** (xdotool `Play`, second). Their outcomes land as rows here.

> **THE D3 SENTENCE — ⚑ SUPERSEDED-PENDING-D5 (kept verbatim):** it remains the closing wording for
> **D3's own** launch-path story if N-B and N-A fail, but it **no longer closes D4** — the final
> report's closing now follows the standalone outcome (§8). Original text, unmodified:
>
> *"harness-side SR+RR-via-Streamline assurance is PROVEN (E01/E02/E03
> `G-RR-ASSURE` PASS, `g5 SL boot : PASS` c1–c5, `sl.dlss` + `sl.dlss_d` mask 0x1, zero
> `Failed to initialize NGX` / `Missing NGX context` / `0xbad00002`, 60/60 SR and 60/60 RR
> evaluates through the port); in-game confirmation is pending a working launch path, **not**
> pending the fix."**

Rows below are the required in-game evidence; each needs `file:line` or a log quote.

| # | claim to prove in-game | evidence required | status |
|---|---|---|---|
| 1 | launch line final state unchanged | `userdata/49277565/config/localconfig.vdf:2894` exact string = `WINEDLLOVERRIDES=winmm,dxgi=n,b PROTON_FORCE_NVAPI=1 DLSS5_AMD_WEIGHTS=/home/deck/DLSS-Vulkan-Linux/build/ngx-weights %command%` | **PROVEN (reported, executor)** — byte-exact before/after the G1 intervention |
| 1b | G1 file provenance | `cmp`-identical to game `nvngx_dlss.dll` + Authenticode Security Directory verified | **PROVEN (reported, executor)** |
| 2 | caps live (the P1 lever's whole point) | `OptiScaler.log` `getSystemCaps` — driver ≥ 512.15, **not** `0.0.0`; no `NVAPI failed to initialize` | BLOCKED (launch path) |
| 3 | SR available & rendering | `Loaded plugin 'sl.dlss' … adapter mask 0x1`; port `eval #K OK: … feature=1` counts; no black screen (game frame / user report) | BLOCKED (launch path) |
| 4 | RR available (menu path) | menu/save: `_enableRayReconstruction` present **and** set in `user_engine_option_save.xml`; Opti `Creating DLSSD feature` / `GetFeature("dlssd")`; `Loaded plugin 'sl.dlss_d' … adapter mask 0x1`, **no** `Ignoring plugin 'sl.dlss_d'` | BLOCKED (launch path) |
| 5 | SL doom trio stays absent in-game | zero `Failed to initialize NGX` · `Missing NGX context` · `0xbad00002` in the session log | BLOCKED (launch path) |
| 6 | RR actually evaluates with the game's real G-buffer | port `eval #K OK: … feature=13 n=7` (or the game's real slot set) — non-black, still ORACLE-PENDING per §6 | BLOCKED (launch path) |
| 7 | timing sanity vs harness baselines | optional: one in-game EVAL/GPU figure, compared against §2 without claiming harness/game equivalence | BLOCKED (launch path) |

*(Attempts log so far: 3 — Steam URL assert-wall ×2, direct ntsync stall pre-`slInit` ×1. Pending:
N-B then N-A; hard stop after those. Fill rows with `bin64/runs/<date>-*` log names or screenshot
paths as they land; if N-B and N-A fail, close D4 with the boxed sentence above.)*

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
- **Re-scope caveat:** the in-game Opti-path observations (`~17k` OK SR evals with user-judged
  defective quality; RR black + `DLSSDContext 0xbad00000`) are recorded from the executor/top-level
  brief in §0 — attributed, not independently re-measured by this session.

## 8. ▓▓ D5 — STANDALONE SLOT (new target: game direct + port; EXECUTOR TO FILL) ▓▓

**Status: PENDING (rows come later — this session runs nothing).** Re-scoped 2026-10-31: the
OptiScaler path is archived (§0) and standalone is what ships. Validation surface per the brief:
executor's `bin64/` standalone staging → `opti_client --direct --sl` harness validation →
user-relaunch verification. The D1 gate suite is reusable as-is (identity/port/non-black/timing +
`G-RR-ASSURE` under `--sl`), with the §2 archived numbers as the comparison baseline — a standalone
cell is *expected* to differ from them (no Opti host layer), so differences must be read as
configuration deltas, not regressions, unless they breach the same sentinel floors.

| # | claim to establish for standalone | evidence required | status |
|---|---|---|---|
| 1 | standalone staging correct in `bin64/` | executor provenance row: files staged, `.bak-*` present, launch line byte-exact (`localconfig.vdf:2894`) | PENDING |
| 2 | harness `--direct --sl` boots and gates | `HARNESS RESULT` with g1 (`--direct` ⇒ g1 n/a by construction) + g2/g3/g4 + `g5 SL boot : PASS` c1–c5 | PENDING |
| 3 | SR present standalone | port `eval #K OK: … feature=1` counts, `fmt col=0 mv=1 dep=0`, non-black outputs | PENDING |
| 4 | RR present standalone | `feature=13` OK lines with `n=7` (or the game's real slot set), RR chain from `ngx-weights/rr`, `G-RR-ASSURE` PASS under `--sl` | PENDING |
| 5 | **user-relaunch verification (the acceptance that matters)** | user launches the game, **judges quality acceptable**, DLSS + RR selectable in menu, `_enableRayReconstruction` set, no black screen | PENDING |
| 6 | timing vs archived baseline | standalone EVAL/GPU medians compared against §2 cells at the same resolution, floors applied, deltas labelled *configuration delta* | PENDING |

*(Fill rule: same as §5 — rows carry `file:line` or log quotes; contradictions are fixed by editing,
never by deleting a measured row. If D5 succeeds, D4's closing becomes "standalone SR+RR accepted by
the user", superseding both the §5 boxed sentence and the §0 archived-path assurance.)*
