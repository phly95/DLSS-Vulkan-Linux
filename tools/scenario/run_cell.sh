#!/usr/bin/env bash
# tools/scenario/run_cell.sh — run ONE D1 cell from D1-MATRIX.csv, in a released slot only.
#
#   bash tools/scenario/run_cell.sh <cell_id> [--reps N] [--frames N] [--force-rep k]
#
# Safety rules baked in (D1-DESIGN §5):
#   * refuses when the game's wineserver holds compatdata/3321460 (prefix contention, run_harness
#     "version mismatch" hazard) and when another run_harness.sh is already live (shared $STAGE,
#     run_harness.sh:46) — from `ps aux`; nobody else is pinged, we just stop.
#   * refuses cells whose D1-MATRIX status is not READY (BLOCKED-N1/N2 must not be run and must
#     never yield a number).
#   * NEVER touches bin64/ or the Steam launch line; writes only under tools/scenario/runs/ and
#     reuses tools/optiharness/run_harness.sh unmodified (it owns build/optiharness/**).
#   * order/front: run in a released slot; re-run the sentinel A03 per D1-DESIGN §6/§7.
# Exits: 0 all reps gated PASS/BLOCKED (see gate_cell.sh), 1 gate FAIL, 3 cell BLOCKED,
#        2 usage/precondition refusal.
set -uo pipefail

ROOT=$(cd -- "$(dirname -- "$0")/../.." && pwd)
cd "$ROOT" || exit 2
CSV=tools/scenario/D1-MATRIX.csv
RUNS=tools/scenario/runs
HARN=tools/optiharness/run_harness.sh

CELL=${1:-}; [[ -n $CELL && -f $CSV ]] || { echo "usage: run_cell.sh <cell_id>" >&2; exit 2; }
shift || true
REPS_OVERRIDE=; FRAME_OVERRIDE=
while [[ $# -gt 0 ]]; do
    case "$1" in
        --reps) REPS_OVERRIDE=$2; shift 2;;
        --frames) FRAME_OVERRIDE=$2; shift 2;;
        *) echo "unknown arg $1" >&2; exit 2;;
    esac
done

row=$(awk -F, -v c="$CELL" '$1==c' "$CSV")
[[ -n $row ]] || { echo "REFUSE: cell '$CELL' not in $CSV" >&2; exit 2; }
IFS=, read -r _ TIER BACKEND RR QUALITY W H OW OH RATIO FRAMES REPS SL STATUS BLOCKED NOTES <<<"$row"
FRAMES=${FRAME_OVERRIDE:-$FRAMES}
REPS=${REPS_OVERRIDE:-$REPS}

# ---- precondition 1: not READY? stop (BLOCKED cells are never executed) -----------------------
if [[ $STATUS != READY ]]; then
    echo "REFUSE: $CELL status=$STATUS blocked_by=$BLOCKED (D1-DESIGN §6 - no number may be collected)" >&2
    exit 3
fi

# ---- cell ini (N1: run_harness.sh --ini <path>, commit 3795535) --------------------------------
# dlss -> no --ini: the harness template already forces Dx12Upscaler=dlss (its [Upscalers] block).
# every other backend gets its own ini from tools/scenario/inis/<backend>.ini; missing file = refuse.
INI_CASE=""
if [[ $BACKEND != dlss ]]; then
    INI_CASE=tools/scenario/inis/$BACKEND.ini
    [[ -f $INI_CASE ]] || { echo "REFUSE: no cell ini $INI_CASE (create it; see tools/scenario/inis/dlssd.ini)" >&2; exit 2; }
fi
INI_ARG=(); [[ -n $INI_CASE ]] && INI_ARG=(--ini "$INI_CASE")

# ---- precondition 2: prefix 3321460 must be free ---------------------------------------------
PREFIX="$HOME/.local/share/Steam/steamapps/compatdata/3321460"
busy=0
for p in $(pgrep -x wineserver 2>/dev/null); do
    tr '\0' '\n' < "/proc/$p/environ" 2>/dev/null | grep -qF "STEAM_COMPAT_DATA_PATH=$PREFIX" && busy=1
done
if [[ $busy -eq 1 ]]; then
    echo "REFUSE: game wineserver holds prefix 3321460 (EXECUTOR slot not released)" >&2
    exit 2
fi

# ---- precondition 3: no other harness run ----------------------------------------------------
# Precise match on purpose: a naive `pgrep -f run_harness.sh` matches ANY process whose cmdline even
# quotes the name - my own wrapper shell, the top-level scheduler's prompt-posting shell, etc.
# (measured 02:52: it refused against the scheduler's own message). Real invocations have an argv
# element that IS the script path (bash <path>/run_harness.sh --out ...), so require THAT shape.
harness_live=""
for p in $(pgrep -f 'run_harness\.sh' 2>/dev/null); do
    [[ $p -eq $$ ]] && continue
    if tr '\0' '\n' < "/proc/$p/cmdline" 2>/dev/null | grep -qE '(^|.*/)run_harness\.sh$'; then
        harness_live="$harness_live $p"
    fi
done
if [[ -n $harness_live ]]; then
    echo "REFUSE: another run_harness.sh is live (shared \$STAGE): pids$harness_live - wait or ping top-level" >&2
    exit 2
fi
if pgrep -f 'opti_client\.exe' >/dev/null 2>&1; then
    echo "REFUSE: an opti_client.exe process is already running" >&2
    exit 2
fi

SL_ARG=(); [[ $SL == yes ]] && SL_ARG=(--sl)
echo "== D1 cell $CELL  tier=$TIER backend=$BACKEND rr=$RR quality=$QUALITY ${W}x${H}->${OW}x${OH} frames=$FRAMES reps=$REPS sl=$SL ini=${INI_CASE:-<harness template>}"

overall=0; PASSN=0; FAILN=0; BLOCKN=0
# --reps N means "N MORE rep dirs, appended": existing rep* dirs are never overwritten (a sentinel
# re-run must keep its earlier samples, or the drift check compares against erased history).
START=$(( $(ls -d "$RUNS/$CELL"/rep* 2>/dev/null | wc -l) + 1 ))
[[ $START -gt 1 ]] && echo "  (appending at rep$START: $((START-1)) existing rep dir(s) kept)"
for k in $(seq "$START" $((START + REPS - 1))); do
    # ABSOLUTE path, deliberately: run_harness.sh converts --out with winpath() (run_harness.sh:224
    # `Z:${1//\//\\}`), so a relative path becomes the drive-relative `Z:tools\scenario\...` and the
    # exe's fopen(o.odir + "/client.log") (opti_client.cpp:430) resolves against the STAGE cwd -> it
    # fails, g_msg stays null, and EVERY [opti_client] line is silently lost (measured 02:49: A03 rep1
    # looked like `rc=1` with no message at all). Never pass --out/--csv relative from here.
    OUT=$ROOT/$RUNS/$CELL/rep$k
    mkdir -p "$OUT"
    echo "-- rep $k/$REPS -> ${OUT#$ROOT/}"
    bash "$HARN" --out "$OUT" --frames "$FRAMES" \
        --w "$W" --h "$H" --ow "$OW" --oh "$OH" "${SL_ARG[@]}" "${INI_ARG[@]}" \
        > "$OUT/harness_invoke.log" 2>&1
    rc=$?
    if [[ $BACKEND == dlss ]]; then
        echo "   harness rc=$rc (a non-zero rc for a dlss cell means a real gate failure - see below)"
    else
        echo "   harness rc=$rc (non-zero is EXPECTED for non-dlss backends: g1's literal is"
        echo "   'Creating new dlss upscaler' - run_harness.sh:250; see D1-GATES 'Harness mapping')"
    fi

    # ---- copy the stage out (the next run wipes it, run_harness.sh:111) -------------------------
    STAGE=$ROOT/build/optiharness/harness
    for f in OptiScaler.log dlss5-amd-ngx.log; do
        [[ -f $STAGE/$f ]] && cp -f "$STAGE/$f" "$OUT/" 2>/dev/null
    done
    # stage manifest: file names + byte sizes (G-PORT-RR reads nvngx_dlssd.dll from it)
    # NOTE -l (DEREFERENCE) is mandatory here, not cosmetic: every model/sl DLL in the stage is a
    # symlink into the game dir, and plain `stat -c %s` reports the LINK length (~82 B) - so
    # G-STAGE / G-PORT-RR would compare 82 against 58956912 / 48343664 forever. Anti-regression:
    # do not "simplify" this back to `stat -c %s`; gate_cell.sh additionally flags link-length
    # artifacts as INVALID rather than as a product failure.
    ( cd "$STAGE" 2>/dev/null && find . -maxdepth 1 -type f -o -maxdepth 1 -type l | sed 's|^\./||' \
        | while read -r f; do printf '%s %s\n' "$f" "$(stat -Lc %s "$f" 2>/dev/null || echo NA)"; done ) \
        > "$OUT/stage_manifest.txt"
    # keep the ini that was in effect (provenance for the identity gate)
    [[ -f $STAGE/OptiScaler.ini ]] && cp -f "$STAGE/OptiScaler.ini" "$OUT/OptiScaler.ini.in-effect"

    bash tools/scenario/gate_cell.sh "$OUT" "$BACKEND" "$FRAMES" $([[ $SL == yes ]] && echo --sl)
    grc=$?
    if [[ $grc -eq 1 ]]; then FAILN=$((FAILN+1)); overall=1
    elif [[ $grc -eq 3 ]]; then BLOCKN=$((BLOCKN+1)); [[ $overall -eq 0 ]] && overall=3
    else PASSN=$((PASSN+1))
    fi

    # Purge the read-back frames AFTER gating: they are 127 MB EACH at 4K x 60 frames x reps (wave 1
    # ended at 1440 files = 178 GB in tools/scenario/runs for 9 cells). Nothing a gate reads lives
    # in them - OUT_NZ, EVAL_MS/GPU_EVAL_MS, the port verdicts and the harness's own gates all come
    # from client.log/eval_ms.csv/the port+Opti logs, which are kept. Re-derivable in ~25 s per rep
    # by re-running the cell. --keep-frames turns the purge off for a pixel forensics need.
    if [[ ${KEEP_FRAMES:-0} != 1 ]]; then
        n_purged=$(find "$OUT" -maxdepth 1 -name 'out_*.f32' | wc -l)
        find "$OUT" -maxdepth 1 -name 'out_*.f32' -delete
        [[ $n_purged -gt 0 ]] && echo "   purged $n_purged read-back .f32 from ${OUT#$ROOT/} (KEEP_FRAMES=1 to keep)"
    fi
done

echo "== cell $CELL done: PASS=$PASSN FAIL=$FAILN BLOCKED=$BLOCKN"
case $overall in 0) v=OK;; 3) v=BLOCKED;; *) v=FAIL;; esac
echo "== remember: sentinel discipline — re-run A03 after every 5th cell (D1-DESIGN §4.4/§7)"
exit $overall