# tools/scenario — DLSS/RR deliverables (D1-D4), owner MASTER-2

`ses_e6f7044eeffeSSITmFI4a0g4Mb` (`opencode/mimo-v2.6-flash-free`) owns this directory.
Partition: this session does **not** launch the game, does **not** write `bin64/`, and does not run
the harness while another session holds the slot. `tools/verify/**` is frozen; `tools/optiharness/**`
belongs to the harness owner.

## Deliverable index

| D | what | artifact | state (2026-10-31 02:2x) |
|---|---|---|---|
| **D1** | scenario replication: upscaler x quality x RR matrix, gates, run plan | `D1-DESIGN.md`, `D1-GATES.md`, `D1-MATRIX.csv`, `inis/*.ini`, `run_cell.sh`, `gate_cell.sh`, `runs/*/` | **WAVE 1 RUN 2026-10-31 02:49–03:11** under a released slot: tier A (dlss × 6 quality) **19/19 runs PASS**, sentinel noise floor 0.082/0.065 ms, GPU resolves resolution (19.71→5.51 ms) while EVAL is below the floor; `--sl` cells **reproduce the in-game trio headlessly with caps GREEN** (`g5_evidence.txt`); RR create **works** (`feature=13`, rr-chain, preset F) but evaluates are BLOCKED(N4 = harness supplies no GBuffer inputs). Numbers + quotes: `REPORT.md` § "D1 wave 1". Unrun: B01/B02/B04-B06 (N4), C/D/E03. |
| **D2** | RR / black-screen root cause | `REPORT.md` (lever log + causal chain), `notes/rr-gating-web.md`, `notes/optiscaler-upgrade-path.md` | running with EXECUTOR; latest: `0xbad00002 FAIL_PlatformError` + `sl.dlss_d adapter mask 0x0`, L4 substitution in flight |
| **D3** | in-game assurance (DLSS + RR selectable & working in Crimson Desert) | `REPORT.md` provenance table (P1/P2…) + `levers/LEVER-RUN1.md` | blocked on L4 outcome; game-side lever `DxgiFactoryWrapping=false` applied but unverified in-game |
| **D4** | consolidated report / documentation | `REPORT.md` § "D4" + this index | appended as evidence accrues; no fabricated results |

Cross-references (read-only): `tools/optiharness/README.md` (harness gates/timing grammar),
`/tmp/opencode/{d2-evidence,opti-build-feasibility,lever-runner,failover_runbook}.md`,
`bin64/runs/` (archived game logs).

## Running D1 (ONLY in a released slot, prefix 3321460 idle, no other harness live)

    bash tools/scenario/run_cell.sh A03            # one cell, reps from D1-MATRIX.csv
    bash tools/scenario/gate_cell.sh tools/scenario/runs/A03/rep1 dlss 60   # re-gate an archive

Both scripts are refused (exit non-zero) when the game's wineserver holds prefix 3321460 or another
`run_harness.sh` is already running. Status columns in `D1-MATRIX.csv` are authoritative: cells
marked `BLOCKED-N1`/`BLOCKED-N2` must not be run (see `D1-DESIGN.md` §6 needs).