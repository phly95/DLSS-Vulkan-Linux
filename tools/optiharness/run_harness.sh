#!/usr/bin/env bash
# tools/optiharness/run_harness.sh: stage and run the OptiScaler-in-the-loop harness, gate the
# results, print HARNESS SUMMARY, exit non-zero when any gate fails.
#
#   bash tools/optiharness/run_harness.sh                     # the fixed port, --opti (default)
#   bash tools/optiharness/run_harness.sh --direct            # A/B: skip OptiScaler entirely
#   bash tools/optiharness/run_harness.sh \
#       --nvngx "/path/to/_nvngx.dll.bak-mvfix" --expect-fail  # issue-1 reproduction demo
#
# What it stages (build/optiharness/harness/, all git-ignored):
#   _nvngx.dll            copied from --nvngx (default build/package/_nvngx.dll - the freshly
#                          rebuilt Depth fix; NEVER rebuilt by this script) or an archived build
#   dxgi.dll, nvngx_dlss.dll   SYMLINKS into the game folder (read-only usage: OptiScaler
#                          v0.9.4's proxy dxgi.dll; the port's CreateFeature size check needs
#                          nvngx_dlss.dll = 58 956 912 bytes next to _nvngx.dll, nvngx.cpp:189)
#   dlss5-amd/shaders     -> build/shaders when its .spv files exist, else the game's copy
#   OptiScaler.ini        generated from tools/optiharness/OptiScaler.harness.ini (the GAME's ini
#                          is never read: Config.cpp:45 loads the ini next to the dxgi.dll)
#   opti_client.exe       the harness executable (built here when missing/stale)
# Logs land together: OptiScaler.log next to the exe, dlss5-amd-ngx.log in the CWD (the port,
# nvngx.cpp:255) - both in the staging dir, this script's stdout in the output dir.
set -uo pipefail

ROOT=$(cd -- "$(dirname -- "$0")/../.." && pwd)
cd "$ROOT" || exit 1
TOOL=$ROOT/tools/optiharness
STEAM=$HOME/.local/share/Steam
GAME="$STEAM/steamapps/common/Crimson Desert/bin64"
PREFIX="$STEAM/steamapps/compatdata/3321460"     # the GAME's prefix: its nvapi64.dll passes
                                                 # OptiScaler's isNvidia() DLSS gate (dllmain.cpp:1978)
NVNGX="$ROOT/build/package/_nvngx.dll"
EXE=$ROOT/build/optiharness/opti_client.exe
STAGE=$ROOT/build/optiharness/harness
OUTD=$ROOT/build/optiharness/out
FRAMES=12
EXPECT=0
MODE=opti
EXTRA=()

usage() {
    echo "usage: tools/optiharness/run_harness.sh [--nvngx PATH] [--expect-fail] [--frames N]"
    echo "      [--direct] [--out DIR] [--mv-fmt rg32|rg16] [--dep-fmt d32s8|r32f]"
    echo "      [--corpus DIR] [--w W] [--h H] [--ow W] [--oh H]"
}
die() { echo "HARNESS FAIL: $*" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --nvngx) NVNGX=$2; shift 2;;
        --expect-fail) EXPECT=1; shift;;
        --frames) FRAMES=$2; shift 2;;
        --direct) MODE=direct; shift;;
        --out) OUTD=$2; shift 2;;
        --help|-h) usage; exit 0;;
        --mv-fmt|--dep-fmt|--w|--h|--ow|--oh|--corpus) EXTRA+=("$1" "$2"); shift 2;;
        *) echo "unknown argument: $1" >&2; usage; exit 2;;
    esac
done

# ---- preflight ----------------------------------------------------------------------------------
[[ -f "$NVNGX" ]] || die "no port DLL at $NVNGX (this script never builds the port: build/package/_nvngx.dll must pre-exist)"
if [[ ! -f "$EXE" || "$TOOL/opti_client.cpp" -nt "$EXE" || "$TOOL/build.sh" -nt "$EXE" ]]; then
    bash "$TOOL/build.sh" || die "build.sh"
fi
[[ -f "$GAME/dxgi.dll" ]] || die "no OptiScaler dxgi.dll in $GAME"
[[ -f "$GAME/nvngx_dlss.dll" ]] || die "no nvngx_dlss.dll in $GAME (the port's CreateFeature check refuses without it)"
# shaders: prefer the port's own build output; fall back to the game's installed copy
SHDIR=
if compgen -G "$ROOT/build/shaders/*.spv" > /dev/null; then SHDIR=$ROOT/build/shaders
elif compgen -G "$GAME/dlss5-amd/shaders/*.spv" > /dev/null; then SHDIR=$GAME/dlss5-amd/shaders
else die "no .spv shaders in build/shaders or $GAME/dlss5-amd/shaders"
fi

# ---- stage --------------------------------------------------------------------------------------
rm -rf "$STAGE"
mkdir -p "$STAGE/dlss5-amd"
cp "$NVNGX" "$STAGE/_nvngx.dll"
if [[ $MODE == opti ]]; then ln -s "$GAME/dxgi.dll" "$STAGE/dxgi.dll"; fi
ln -s "$GAME/nvngx_dlss.dll" "$STAGE/nvngx_dlss.dll"
ln -s "$SHDIR" "$STAGE/dlss5-amd/shaders"
cp "$TOOL/OptiScaler.harness.ini" "$STAGE/OptiScaler.ini"
cp "$EXE" "$STAGE/opti_client.exe"
mkdir -p "$OUTD"
rm -f "$OUTD"/out_*.f32 "$OUTD/eval_ms.csv" "$OUTD/summary.txt" "$OUTD/client.log" "$OUTD/run.stdout.log"
echo "  staged $STAGE (nvngx=$(basename "$NVNGX"), shaders=$SHDIR, mode=$MODE, frames=$FRAMES)"

# ---- Proton selection: copied from tools/verify/run_all.sh:282-301 (V9 probe block) ---------------
# Probe newest-that-works only with `timeout 20 ... --nonsense-arg`: a healthy build rejects it
# with exit 1, a wedged Wine-11 build returns 124 (docs/MEASUREMENT.md:252-256 pins Proton 10.0
# because Wine 11 builds hang before loading anything, so 10.0 is probed FIRST here), no Traceback.
RADV_ICD=$(ls /usr/share/vulkan/icd.d/radeon_icd.x86_64.json 2>/dev/null | head -1)   # run_all.sh:91
PROTON_ENV=(env -u LD_PRELOAD VK_ICD_FILENAMES="$RADV_ICD" VK_DRIVER_FILES="$RADV_ICD" MESA_SHADER_CACHE_DISABLE=true)
PROBE_N=0
probe_one() {   # $1 = proton path; 0 = good (accept)
    PROBE_N=$((PROBE_N + 1))
    mkdir -p "$ROOT/build/optiharness/prefix_probe_$PROBE_N"
    timeout 20 "${PROTON_ENV[@]}" \
        STEAM_COMPAT_DATA_PATH="$ROOT/build/optiharness/prefix_probe_$PROBE_N" \
        STEAM_COMPAT_CLIENT_INSTALL_PATH="$STEAM" \
        "$1" run "$STAGE/opti_client.exe" --nonsense-arg \
        > "$ROOT/build/optiharness/proton_probe_$PROBE_N.log" 2>&1
    local rc=$?
    echo "  Proton probe rc=$rc: $1"
    [[ $rc -eq 1 ]] && ! grep -q "Traceback (most recent call last)" "$ROOT/build/optiharness/proton_probe_$PROBE_N.log"
}
if [[ -n ${PROTON:-} ]]; then
    echo "  Proton: $PROTON (set by the caller, probe skipped)"
else
    P10="$STEAM/steamapps/common/Proton 10.0/proton"
    if [[ -x "$P10" ]] && probe_one "$P10"; then
        PROTON=$P10
    else
        while IFS= read -r cand; do
            [[ "$cand" == "$P10" ]] && continue
            if probe_one "$cand"; then PROTON=$cand; break; fi
        done < <(ls -d "$STEAM"/compatibilitytools.d/*/proton "$STEAM"/steamapps/common/Proton*/proton 2>/dev/null | sort -Vr)
    fi
fi
[[ -n ${PROTON:-} && -x ${PROTON:-} ]] || die "no Proton under $STEAM starts the harness (see $ROOT/build/optiharness/proton_probe_*.log)"
echo "  Proton: $PROTON"

# ---- run ----------------------------------------------------------------------------------------
winpath() { echo "Z:${1//\//\\}"; }    # a Linux path as the Wine process sees it (run_all.sh:313)
EXPECT_ARGS=()
[[ $EXPECT -eq 1 ]] && EXPECT_ARGS=(--expect-fail)
STDOUT_LOG=$OUTD/run.stdout.log
echo "  running ($(date +%H:%M:%S), timeout 900 s)..."
(
    cd "$STAGE" || exit 90
    # env: the literal set the diagnostics established for loading OptiScaler inside a Wine
    # process (winmm/dxgi native-first), the game's launch.env companions (PROTON force-nvapi,
    # packed weights location) and the game's compatdata prefix; VK_ICD/VK_DRIVER_FILES only as
    # run_all.sh's PROTON_ENV carries them (RADV_ICD above).
    exec env -u LD_PRELOAD \
        WINEDLLOVERRIDES=winmm,dxgi=n,b \
        PROTON_FORCE_NVAPI=1 \
        DLSS5_AMD_WEIGHTS="$ROOT/build/ngx-weights" \
        STEAM_COMPAT_DATA_PATH="$PREFIX" \
        STEAM_COMPAT_CLIENT_INSTALL_PATH="$STEAM" \
        VK_ICD_FILENAMES="$RADV_ICD" \
        VK_DRIVER_FILES="$RADV_ICD" \
        MESA_SHADER_CACHE_DISABLE=true \
        timeout 900 "$PROTON" run "./opti_client.exe" \
            "--$MODE" --frames "$FRAMES" \
            --out "$(winpath "$OUTD")" --csv "$(winpath "$OUTD/eval_ms.csv")" \
            "${EXPECT_ARGS[@]}" "${EXTRA[@]}"
) > "$STDOUT_LOG" 2>&1
RC=$?
echo "  opti_client rc=$RC ($(date +%H:%M:%S)); logs in $STAGE"

# ---- gates --------------------------------------------------------------------------------------
OLOG=$STAGE/OptiScaler.log
PLOG=$STAGE/dlss5-amd-ngx.log
# The PE's stdout/stderr do not reach the shell under `proton run` (the frozen V9 logs prove it),
# so opti_client mirrors every line into <out>/client.log - g3/g4 parse THERE, falling back to
# run.stdout.log (which at least carries Proton's own noise) when client.log is absent.
PARSE=$OUTD/client.log
[[ -f "$PARSE" ]] || PARSE=$STDOUT_LOG

# which fmt triplet the port must print (it re-derives only the MV code from the resource):
MV_CODE=1                                       # rg32 (default) == in-game R32G32_FLOAT -> code 1
if [[ " ${EXTRA[*]-} " == *" --mv-fmt rg16 "* ]] || [[ " ${EXTRA[*]-} " == *" --corpus "* ]]; then MV_CODE=0; fi
EXPECT_FMT="fmt col=0 mv=$MV_CODE dep=0"

# --- g1: OptiScaler served OUR dlss provider from the staged _nvngx.dll ---
g1=PASS; g1note=""
if [[ $MODE == direct ]]; then
    g1note="n/a (--direct: OptiScaler not loaded)"
else
    if [[ -f "$OLOG" ]]; then
        n_create=$(grep -cF "Creating new dlss upscaler" "$OLOG")
        n_load=$(grep -cE "NVNGXProxy::InitNVNGX Loaded from .*_nvngx\.dll" "$OLOG")
        n_nvidia=$(grep -cF "Running on Nvidia" "$OLOG")
        n_gone=$(grep -cE "falling back to FSR|Upscaler can't created" "$OLOG")
        n_bad0=$(grep -cF "_EvaluateFeature result: BAD00000" "$OLOG")
        if [[ $EXPECT -eq 1 ]]; then
            # inverted: the refusal (BAD00000) must COME BACK through OptiScaler, no FSR fallback
            if (( n_create < 1 || n_load < 1 || n_gone > 0 || n_bad0 < 1 )); then g1=FAIL; fi
        else
            if (( n_create < 1 || n_load < 1 || n_gone > 0 || n_bad0 > 0 )); then g1=FAIL; fi
        fi
        g1note="create=$n_create loaded_from_nvngx=$n_load running_on_nvidia=$n_nvidia fsr_fallback=$n_gone bad00000=$n_bad0"
    else
        g1=FAIL; g1note="no OptiScaler.log in $STAGE"
    fi
fi

# --- g2: the port's own verdict for every frame ---
g2=PASS; g2note=""
if [[ -f "$PLOG" ]]; then
    n_ok=$(grep -cE 'eval #[0-9]+ OK:' "$PLOG")
    n_fail=$(grep -cE 'eval #[0-9]+ FAIL:' "$PLOG")
    n_fmt=$(grep -cF "$EXPECT_FMT" "$PLOG")
    n_d19=$(grep -cF "Depth: DXGI format 19 is not one this bridge decodes" "$PLOG")
    n_slotd=$(grep -cE 'eval #1 FAIL:.*slot=Depth' "$PLOG")
    if [[ $EXPECT -eq 1 ]]; then
        # reproduction: EVERY presented frame refused on the Depth slot, with issue 1's exact line
        if (( n_d19 < 1 || n_slotd < 1 || n_ok > 0 || n_fail < FRAMES - 2 )); then g2=FAIL; fi
    else
        if (( n_ok < FRAMES || n_fail > 0 || n_fmt < 1 )); then g2=FAIL; fi
    fi
    g2note="ok=$n_ok/$FRAMES fail=$n_fail '$EXPECT_FMT'x$n_fmt depth19_refusal=$n_d19 slotdepth@eval1=$n_slotd"
else
    g2=FAIL; g2note="no dlss5-amd-ngx.log in $STAGE"
fi

# --- g3: the upscaled output is real (every frame's non-zero fraction) ---
g3=PASS; g3note=""
n_out=$(grep -cE '^OUT_NZ [0-9]+ nz=' "$PARSE" || true)
if [[ $n_out -eq $FRAMES ]]; then
    if [[ $EXPECT -eq 1 ]]; then
        # inverted: the archived DLL must yield BLACK frames (nothing ever writes the output)
        viol=$(awk '/^OUT_NZ /{ split($3, a, "="); if (!(a[2] >= 0.0 && a[2] < 0.10)) bad++ } END{ print bad+0 }' "$PARSE")
        thr="every frame nz < 0.10 (black expected)"
    else
        # gate 0.10: the synthetic colour is gradient+checker with all channels > 0, so a healthy
        # evaluate sits near 1.00; 0.10 is the conservative floor that still catches "all black"
        viol=$(awk '/^OUT_NZ /{ split($3, a, "="); if (a[2] < 0.10) bad++ } END{ print bad+0 }' "$PARSE")
        thr="every frame nz >= 0.10"
    fi
    [[ $viol -eq 0 ]] || g3=FAIL
    g3note="threshold '$thr', viol=$viol/$FRAMES, min nz=$(awk 'BEGIN{hit=0} /^OUT_NZ /{ split($3,a,"="); v=a[2]+0; if(!hit || v<m) m=v; hit=1 } END{ if(hit) printf "%.6f", m }' "$PARSE")"
else
    g3=FAIL; g3note="OUT_NZ frames $n_out/$FRAMES"
fi

# --- g4: timings, frames 1..N (frame 0 excluded, the repo convention) ---
stats_sorted() {  # stdin: numbers already ascending; prints "n median p95"
    awk '{ a[NR-1] = $1 } END {
        n = NR; if (n == 0) { print "0 -1 -1"; exit }
        med = (n % 2) ? a[int(n/2)] : (a[n/2 - 1] + a[n/2]) / 2;
        k = int(0.95 * n); if (0.95 * n > k) k++; if (k < 1) k = 1; if (k > n) k = n;
        printf "%d %.4f %.4f\n", n, med, a[k-1]
    }'
}
# frame 0 excluded from every statistic (the repo convention: runner timestamps, eval_verify.py,
# the exe's own summary all skip it)
EV_STATS=$(grep -E '^EVAL_MS [0-9]+ [0-9]+(\.[0-9]+)?$' "$PARSE" | awk '$2 != "0" {print $3}' | sort -g | stats_sorted)
read -r EV_N EV_MED EV_P95 <<< "$EV_STATS"
GP_STATS=$(grep -E '^GPU_EVAL_MS [0-9]+ [0-9]+(\.[0-9]+)?$' "$PARSE" | awk '$2 != "0" {print $3}' | sort -g | stats_sorted)
read -r GP_N GP_MED GP_P95 <<< "$GP_STATS"
GP_NA=$(grep -cE '^GPU_EVAL_MS [0-9]+ n/a' "$PARSE" || true)
g4=PASS
(( EV_N >= FRAMES - 1 && $(awk -v m="$EV_MED" 'BEGIN{print (m>0)?1:0}') == 1 )) || g4=FAIL
if (( GP_N > 0 )); then
    g4note="EVAL_MS n=$EV_N median=$EV_MED ms p95=$EV_P95 ms | GPU_EVAL_MS n=$GP_N median=$GP_MED ms p95=$GP_P95 ms (frames 1..)"
else
    g4note="EVAL_MS n=$EV_N median=$EV_MED ms p95=$EV_P95 ms | GPU_EVAL_MS n/a x$GP_NA (timestamps unsupported - degraded, per contract)"
fi
EXE_RESULT=$(awk -F= '/^RESULT=/{print $2}' "$OUTD/summary.txt" 2>/dev/null)

# ---- HARNESS SUMMARY -----------------------------------------------------------------------------
echo
echo "HARNESS SUMMARY"
echo "  mode=$MODE expect_fail=$EXPECT frames=$FRAMES nvngx=$(basename "$NVNGX")"
echo "  g1 OptiScaler : $g1   $g1note"
echo "  g2 port log   : $g2   $g2note"
echo "  g3 non-black  : $g3   $g3note"
echo "  g4 timing     : $g4   $g4note"
if [[ -f "$OUTD/summary.txt" ]]; then
    echo "  --- opti_client summary.txt ---"
    sed 's/^/  /' "$OUTD/summary.txt"
fi
ALL=PASS
for x in "$g1" "$g2" "$g3" "$g4"; do [[ $x == PASS ]] || ALL=FAIL; done
[[ $RC -eq 0 ]] || ALL=FAIL
echo "HARNESS RESULT: $ALL  (opti_client rc=$RC, exe summary RESULT=${EXE_RESULT:-n/a})"
[[ $ALL == PASS ]] || exit 1
exit 0