# tools/scenario — DLSS/RR deliverables (D1-D4), owner MASTER-2

`ses_e6f7044eeffeSSITmFI4a0g4Mb` (`opencode/mimo-v2.6-flash-free`) owns this directory.
Partition: this session does **not** launch the game, does **not** write `bin64/`, and does not run
the harness while another session holds the slot. `tools/verify/**` is frozen; `tools/optiharness/**`
belongs to the harness owner.

## Deliverable index

| D | what | artifact | state (2026-10-31 02:2x) |
|---|---|---|---|
| **D1** | scenario replication: upscaler x quality x RR matrix, gates, run plan | `D1-DESIGN.md`, `D1-GATES.md`, `D1-MATRIX.csv`, `inis/*.ini`, `run_cell.sh`, `gate_cell.sh`, `runs/*/` | **33/33 CELLS RUN** (waves 1-3, 02:49-09:35, inside released slots). Tier A dlss ×6 quality PASS; RR tier dlssd ×6 PASS post-N4 (4 RR inputs bound, `mv=10`, outputs ORACLE-PENDING); type axis fsr22/xess/fsr31 ×6 each PASS post-N5 (backend-swap detector + 4-frame warmup rule); three `--sl` cells (E01/E02/E03) reproduce the in-game SL trio while the port serves RR 60/60 (decoupling proof) — **then post-R1+L-C all three flipped to PASS** (§ "P-RECTOR": `g5 SL boot: PASS`, six absences 0, `sl.dlss_d id 1001 adapter mask 0x1`, `P100 - sl.dlss_d`, SL adds no measurable ms, 56.9 ms not reproduced). **Final: 33/33 cells PASS.** Full medians/floors: `REPORT.md` § waves 1-4 + P-RECTOR. |
| **D2** | RR / black-screen root cause | `REPORT.md` (lever log + causal chain), `notes/rr-gating-web.md`, `notes/optiscaler-upgrade-path.md` | running with EXECUTOR; latest: `0xbad00002 FAIL_PlatformError` + `sl.dlss_d adapter mask 0x0`, L4 substitution in flight |
| **D3** | in-game assurance (DLSS + RR selectable & working in Crimson Desert) | `REPORT.md` provenance table (P1/P2…) + `levers/LEVER-RUN1.md` | blocked on L4 outcome; game-side lever `DxgiFactoryWrapping=false` applied but unverified in-game |
| **D4** | consolidated report / documentation | **`D4-FINAL.md`** (final skeleton, assembled 11:0x: exec summary incl. the one-sentence D3-grade claim from E01/E02/E03, digest with superseded rows marked, lever provenance P1..WARN, h1prime §4 pointer, upstream-diff verdict, RR-oracle scope line, open items) + `REPORT.md` § "D4"/"P-RECTOR" + this index | **skeleton COMPLETE; §5 D3 slot = PENDING (executor fills)** |

Cross-references (read-only): `tools/optiharness/README.md` (harness gates/timing grammar),
`/tmp/opencode/{d2-evidence,opti-build-feasibility,lever-runner,failover_runbook}.md`,
`bin64/runs/` (archived game logs).

## Running D1 (ONLY in a released slot, prefix 3321460 idle, no other harness live)

    bash tools/scenario/run_cell.sh A03            # one cell, reps from D1-MATRIX.csv
    bash tools/scenario/gate_cell.sh tools/scenario/runs/A03/rep1 dlss 60   # re-gate an archive

Both scripts are refused (exit non-zero) when the game's wineserver holds prefix 3321460 or another
`run_harness.sh` is already running. Status columns in `D1-MATRIX.csv` are authoritative: cells
marked `BLOCKED-N1`/`BLOCKED-N2` must not be run (see `D1-DESIGN.md` §6 needs).