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
if pgrep -f "run_harness.sh" >/dev/null 2>&1; then
    echo "REFUSE: another run_harness.sh is live (shared staging dir)" >&2
    exit 2
fi

SL_ARG=(); [[ $SL == yes ]] && SL_ARG=(--sl)
echo "== D1 cell $CELL  tier=$TIER backend=$BACKEND rr=$RR quality=$QUALITY ${W}x${H}->${OW}x${OH} frames=$FRAMES reps=$REPS sl=$SL ini=${INI_CASE:-<harness template>}"

overall=0
for k in $(seq 1 "$REPS"); do
    OUT=$RUNS/$CELL/rep$k
    mkdir -p "$OUT"
    echo "-- rep $k/$REPS -> $OUT"
    bash "$HARN" --out "$OUT" --frames "$FRAMES" \
        --w "$W" --h "$H" --ow "$OW" --oh "$OH" "${SL_ARG[@]}" "${INI_ARG[@]}" \
        > "$OUT/harness_invoke.log" 2>&1
    rc=$?
    echo "   harness rc=$rc (non-zero is EXPECTED for non-dlss backends: g1's literal is"
    echo "   'Creating new dlss upscaler' - run_harness.sh:250; see D1-GATES 'Harness mapping')"

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
    [[ $grc -eq 1 ]] && overall=1
    [[ $grc -eq 3 && $overall -eq 0 ]] && overall=3
done

echo "== cell $CELL done, overall=$overall (0 pass/blocked, 1 fail, 3 blocked)"
case $overall in 0) v=OK;; 3) v=BLOCKED;; *) v=FAIL;; esac
echo "== remember: sentinel discipline — re-run A03 after every 5th cell (D1-DESIGN §4.4/§7)"
exit $overall