#!/usr/bin/env bash
# tools/optiharness/run_harness.sh: stage and run the OptiScaler-in-the-loop harness, gate the
# results, print HARNESS SUMMARY, exit non-zero when any gate fails.
#
#   bash tools/optiharness/run_harness.sh                     # the fixed port, --opti (default)
#   bash tools/optiharness/run_harness.sh --direct            # A/B: skip OptiScaler entirely
#   bash tools/optiharness/run_harness.sh \
#       --nvngx "/path/to/_nvngx.dll.bak-mvfix" --expect-fail  # issue-1 reproduction demo
#   bash tools/optiharness/run_harness.sh --sl                # ALSO boot the game's Streamline stack
#                                                             # (gates g1-g4 + g5; SL_NVAPI_VER=<n>
#                                                             #  adds DXVK_NVAPI_DRIVER_VERSION=<n>)
#   bash tools/optiharness/run_harness.sh \
#       --ini "/path/to/cell_A03.ini"                         # stage THAT ini as OptiScaler.ini
#                                                             # (default: OptiScaler.harness.ini)
#
# What it stages (build/optiharness/harness/, all git-ignored):
#   _nvngx.dll            copied from --nvngx (default build/package/_nvngx.dll - the freshly
#                          rebuilt Depth fix; NEVER rebuilt by this script) or an archived build
#   dxgi.dll, nvngx_dlss.dll   SYMLINKS into the game folder (read-only usage: OptiScaler
#                          v0.9.4's proxy dxgi.dll; the port's CreateFeature size check needs
#                          nvngx_dlss.dll = 58 956 912 bytes next to _nvngx.dll, nvngx.cpp:189)
#   nvngx_dlssd.dll       SYMLINK (absolute, same waypoint) to the game's real NVIDIA DLSS-D
#                          companion, 48 343 664 bytes - the RR cells need it next to _nvngx.dll
#                          (nvngx.cpp:187-189); absent source dies in preflight like nvngx_dlss.dll
#   upscaler companions    SYMLINKS (absolute, same waypoint, N5) to the upscaler libs OptiScaler
#                          probes next to its own dxgi.dll: amd_fidelityfx_loader_dx12/dx12/
#                          upscaler_dx12/framegeneration_dx12/denoiser_dx12/radiancecache_dx12,
#                          libxess.dll, libxess_dx11.dll, libxess_fg.dll, libxell.dll - 9 of the
#                          10 exist in the game's bin64, the absent one is skipped with a note
#                          (OPTIONAL by design, never fatal: without them OptiScaler silently
#                          reclassifies the xess/fsr31 cells to fsr21 - N5 detail in README)
#   dlss5-amd/shaders     -> build/shaders when its .spv files exist, else the game's copy
#   OptiScaler.ini        generated from tools/optiharness/OptiScaler.harness.ini, or from the
#                          --ini <path> file when given (the GAME's ini is never read: Config.cpp:45
#                          loads the ini next to the dxgi.dll)
#   sl.interposer.dll + 10 sl.*.dlls  SYMLINKS (all 11 game plugins; read-only) - the Streamline
#                          stack the --sl mode boots (app-id/flags/features pulled from the
#                          runs/OptiScaler.slayer3* archive; see README)
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
SLMODE=0
EXTRA=()
INI_OVERRIDE=           # empty = stage the default OptiScaler.harness.ini (byte-for-byte as before)

usage() {
    echo "usage: tools/optiharness/run_harness.sh [--nvngx PATH] [--expect-fail] [--frames N]"
    echo "      [--direct] [--sl] [--out DIR] [--ini PATH] [--mv-fmt rg32|rg16] [--dep-fmt d32s8|r32f]"
    echo "      [--corpus DIR] [--w W] [--h H] [--ow W] [--oh H]"
    echo "      env: SL_NVAPI_VER=<n>  also sets DXVK_NVAPI_DRIVER_VERSION=<n> for the run"
}
die() { echo "HARNESS FAIL: $*" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --nvngx) NVNGX=$2; shift 2;;
        --expect-fail) EXPECT=1; shift;;
        --frames) FRAMES=$2; shift 2;;
        --direct) MODE=direct; shift;;
        --sl) SLMODE=1; shift;;
        --out) OUTD=$2; shift 2;;
        --ini) INI_OVERRIDE=$2; shift 2;;
        --help|-h) usage; exit 0;;
        --mv-fmt|--dep-fmt|--w|--h|--ow|--oh|--corpus) EXTRA+=("$1" "$2"); shift 2;;
        *) echo "unknown argument: $1" >&2; usage; exit 2;;
    esac
done

# ---- preflight ----------------------------------------------------------------------------------
# --ini <path>: stage THAT file as the harness OptiScaler.ini instead of the default template.
# Validated HERE - before anything is built, staged or run - so a bad path can never boot a run on
# the wrong config: the file must exist and must look like an ini (at least one [section] header).
INI=$TOOL/OptiScaler.harness.ini
if [[ -n $INI_OVERRIDE ]]; then
    INI=$INI_OVERRIDE
    [[ -f "$INI" ]] || die "--ini: no such file: $INI"
    grep -qE '^[[:space:]]*\[[^]]+\]' "$INI" || \
        die "--ini: $INI does not look like an ini (no [section] header)"
fi
[[ -f "$NVNGX" ]] || die "no port DLL at $NVNGX (this script never builds the port: build/package/_nvngx.dll must pre-exist)"
if [[ ! -f "$EXE" || "$TOOL/opti_client.cpp" -nt "$EXE" || "$TOOL/build.sh" -nt "$EXE" ]]; then
    bash "$TOOL/build.sh" || die "build.sh"
fi
# The OptiScaler proxy dll: preference is dxgi.dll, but the game folder is shared with live
# diagnostics sessions that may park it as dxgi.dll.optiscaler-off (measured 2026-10-31: present,
# 25 379 632 B = the same v0.9.4 proxy this harness booted yesterday); fall back to that name so
# a parallel experiment can turn OptiScaler "off" in the game without breaking this harness.
DXGI_SRC="$GAME/dxgi.dll"
[[ -f "$DXGI_SRC" ]] || DXGI_SRC="$GAME/dxgi.dll.optiscaler-off"
[[ -f "$DXGI_SRC" ]] || die "no OptiScaler dxgi.dll in $GAME (nor dxgi.dll.optiscaler-off)"
[[ -f "$GAME/nvngx_dlss.dll" ]] || die "no nvngx_dlss.dll in $GAME (the port's CreateFeature check refuses without it)"
# same failure semantics as nvngx_dlss.dll above: a missing source is fatal, never a dangling link
[[ -f "$GAME/nvngx_dlssd.dll" ]] || die "no nvngx_dlssd.dll in $GAME (the D1 RR cells need the DLSS-D companion beside _nvngx.dll)"
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
if [[ $MODE == opti ]]; then ln -s "$DXGI_SRC" "$STAGE/dxgi.dll"; fi
ln -s "$GAME/nvngx_dlss.dll" "$STAGE/nvngx_dlss.dll"
# same waypoint, same ABSOLUTE style as nvngx_dlss.dll: the real NVIDIA DLSS-D (RR) companion
ln -s "$GAME/nvngx_dlssd.dll" "$STAGE/nvngx_dlssd.dll"
# N5 (D1-DESIGN.md:179, REPORT.md:279-295): OptiScaler probes these upscaler companion dlls
# beside its own dxgi.dll on EVERY start (Config::CheckUpscalerFiles, FfxApi_Proxy.h:211/368/539/
# 677/819, XeSS_Proxy.h:247) and silently rewrites the backend when the one its chosen upscaler
# needs is absent - measured in C01/C03: "Config::CheckUpscalerFiles libxess.dll not found!" ->
# "FeatureProvider_Dx12::ChangeFeature changing backend to fsr21" -> "init successful for fsr21",
# i.e. the xess/fsr31 cells measured fsr21, not their axis label (6 runs BLOCKED(N5)). Same
# waypoint, same ABSOLUTE symlink style as nvngx_dlss.dll above - but NOT fatal: these are optional
# backends and an absent variant is normal (OptiScaler logs "Can't find ..." and continues), so a
# missing source prints one note line and staging continues. nvngx_dlss.dll/nvngx_dlssd.dll stay
# fatal: without them the experiment itself is broken, which is a different kind of missing.
UPSCALER_DLLS=(amd_fidelityfx_loader_dx12.dll amd_fidelityfx_dx12.dll amd_fidelityfx_upscaler_dx12.dll
               amd_fidelityfx_framegeneration_dx12.dll amd_fidelityfx_denoiser_dx12.dll
               amd_fidelityfx_radiancecache_dx12.dll
               libxess.dll libxess_dx11.dll libxess_fg.dll libxell.dll)
for s in "${UPSCALER_DLLS[@]}"; do
    if [[ -f "$GAME/$s" ]]; then
        ln -s "$GAME/$s" "$STAGE/$s"
    else
        echo "  note: optional upscaler companion absent in game dir, not staged: $s"
    fi
done
# Streamline: all 11 of the game's sl.*.dlls (read-only symlinks), so sl.interposer finds its
# plugins in the exe dir exactly like the game's findPlugins in bin64 (archive:545/309).
SL_DLLS=(sl.interposer.dll sl.common.dll sl.dlss.dll sl.dlss_d.dll sl.dlss_g.dll sl.reflex.dll
         sl.pcl.dll sl.nis.dll sl.nvperf.dll sl.deepdvc.dll sl.directsr.dll)
for s in "${SL_DLLS[@]}"; do
    [[ -f "$GAME/$s" ]] || die "missing game SL dll $GAME/$s"
    ln -s "$GAME/$s" "$STAGE/$s"
done
ln -s "$SHDIR" "$STAGE/dlss5-amd/shaders"
cp "$INI" "$STAGE/OptiScaler.ini"     # $INI = --ini <path>, or the harness template when absent
cp "$EXE" "$STAGE/opti_client.exe"
mkdir -p "$OUTD"
rm -f "$OUTD"/out_*.f32 "$OUTD/eval_ms.csv" "$OUTD/summary.txt" "$OUTD/client.log" "$OUTD/run.stdout.log"
echo "  staged $STAGE (nvngx=$(basename "$NVNGX"), ini=$INI, shaders=$SHDIR, mode=$MODE, frames=$FRAMES)"

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

# ---- prefix: never race another session's live game --------------------------------------------
# Measured 2026-10-31 00:14: a GE-Proton11-6 wineserver (pid 3046121) held compatdata/3321460
# with CrimsonDesert.exe RUNNING in a concurrent session; a second Proton on the same prefix dies
# immediately with "wine client error: version mismatch 935/856" (client = Proton 10.0's wineserver
# protocol, server = GE's 935) and the exe never starts.  The game must not be started/stopped by
# this harness, so when the game's wineserver is alive: run against a one-time read-only CLONE of
# the prefix (478 MB, cp -a) - the original is only ever READ from that point on.
RUN_PREFIX=$PREFIX
prefix_busy() {
    local p
    for p in $(pgrep -x wineserver 2>/dev/null); do
        tr '\0' '\n' < "/proc/$p/environ" 2>/dev/null | grep -qF "STEAM_COMPAT_DATA_PATH=$PREFIX" && return 0
    done
    return 1
}
if prefix_busy; then
    CLONE=$ROOT/build/optiharness/prefix_3321460.clone
    echo "  game prefix BUSY (another session's wineserver on 3321460) - cloning to $CLONE"
    rm -rf "$CLONE"
    mkdir -p "$CLONE"
    cp -a "$PREFIX"/. "$CLONE"/ || die "cannot clone the game prefix"
    RUN_PREFIX=$CLONE
else
    echo "  game prefix free: $RUN_PREFIX (default game compatdata)"
fi

# ---- run ----------------------------------------------------------------------------------------
winpath() { echo "Z:${1//\//\\}"; }    # a Linux path as the Wine process sees it (run_all.sh:313)
EXPECT_ARGS=()
[[ $EXPECT -eq 1 ]] && EXPECT_ARGS=(--expect-fail)
SL_ARGS=()
[[ $SLMODE -eq 1 ]] && SL_ARGS=(--sl)
# Optional one-line lever: SL_NVAPI_VER -> DXVK_NVAPI_DRIVER_VERSION (dxvk-nvapi wants a DECIMAL
# number 100..99999, e.g. 57602 for "576.02" - dxvk-nvapi README), passed through verbatim.
SL_ENV=()
[[ -n ${SL_NVAPI_VER:-} ]] && SL_ENV=("DXVK_NVAPI_DRIVER_VERSION=$SL_NVAPI_VER")
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
        STEAM_COMPAT_DATA_PATH="$RUN_PREFIX" \
        STEAM_COMPAT_CLIENT_INSTALL_PATH="$STEAM" \
        VK_ICD_FILENAMES="$RADV_ICD" \
        VK_DRIVER_FILES="$RADV_ICD" \
        MESA_SHADER_CACHE_DISABLE=true \
        "${SL_ENV[@]}" \
        timeout 900 "$PROTON" run "./opti_client.exe" \
            "--$MODE" --frames "$FRAMES" \
            --out "$(winpath "$OUTD")" --csv "$(winpath "$OUTD/eval_ms.csv")" \
            "${EXPECT_ARGS[@]}" "${SL_ARGS[@]}" "${EXTRA[@]}"
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
# N4 (2026-10-31) RR cells: a staged ini with [Upscalers] Dx12Upscaler=dlssd runs feature 13, whose
# eval-OK line re-derives the SAME in-game RG32F MotionVectors to code 10, not 1 - the port prints
# `fmt col=0 mv=10 dep=0` (nvngx.cpp:1177 `fp.fmt_mv = rr ? 10 : 1`, so an RR eval can never print
# mv=1/MV stays 0 for rg16).  Non-RR runs read here, fail the grep and keep the old expectation.
if [[ $MV_CODE -eq 1 ]] && grep -qiE '^[[:space:]]*Dx12Upscaler[[:space:]]*=[[:space:]]*dlssd([[:space:]]*;.*)?[[:space:]]*$' "$INI"; then
    MV_CODE=10
fi
EXPECT_FMT="fmt col=0 mv=$MV_CODE dep=0"

# --- g1: OptiScaler served OUR dlss provider from the staged _nvngx.dll ---
g1=PASS; g1note=""
if [[ $MODE == direct ]]; then
    g1note="n/a (--direct: OptiScaler not loaded)"
else
    if [[ -f "$OLOG" ]]; then
        # `dlssd?`: an RR cell logs "Creating new dlssd upscaler" (FeatureProvider_Dx12.cpp:177 with
        # newBackend = "dlssd"); the optional d leaves every SR count byte-identical to before.
        n_create=$(grep -cE "Creating new dlssd? upscaler" "$OLOG")
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

# --- g5: Streamline boot (only under --sl): the game's exact doom lines, checked one by one ---
# Evidence = our [sl-cb] callback lines + SLDLSS probe lines (client.log) concatenated with
# OptiScaler's own [streamline] echo (OptiScaler.log) - whichever registered first carried the
# lines, both sources are searched so the gate is about Streamline, not about who echoed it.
g5=n/a; g5note=""; g5detail=""
ver_ge() { [[ $(printf '%s\n%s\n' "$2" "$1" | sort -V | head -1) == "$2" ]]; }  # $1 >= $2
if [[ $SLMODE -eq 1 ]]; then
    G5EV=$OUTD/g5_evidence.txt
    { cat "$PARSE" 2>/dev/null; echo "--- optiscaler echo ---"; cat "$OLOG" 2>/dev/null; } > "$G5EV"
    n_caps=$(grep -cF 'getSystemCaps' "$G5EV")
    n_nvfail=$(grep -cF 'NVAPI failed to initialize' "$G5EV")
    n_failngx=$(grep -cF 'Failed to initialize NGX' "$G5EV")
    n_missing=$(grep -cF 'Missing NGX context' "$G5EV")
    n_dlss=$(grep -cE "Loaded plugin 'sl\.dlss'" "$G5EV")
    c1=ok; c2=ok; c3=ok; c4=ok; c5=ok
    (( n_caps >= 1 && n_nvfail == 0 )) || c1="bad(caps=$n_caps nvapi_fail=$n_nvfail)"
    (( n_failngx == 0 )) || c2="bad(x$n_failngx)"
    (( n_missing == 0 )) || c4="bad(x$n_missing)"
    (( n_dlss >= 1 )) || c5="bad(x$n_dlss)"
    # c3: the "requires driver ... detected ..." line (exactly what the game prints for
    # sl.dlss); absent = ok, present = the DETECTED version must satisfy the required one
    drv=$(grep -oE 'requires driver [0-9]+\.[0-9]+\.[0-9]+ or newer - detected [0-9]+\.[0-9]+\.[0-9]+' "$G5EV" | head -1)
    if [[ -z "$drv" ]]; then
        c3=ok; drv_note="no 'requires driver' line"
    else
        need=${drv#requires driver }; need=${need%% or newer*}
        got=${drv##* - detected }
        if ver_ge "$got" "$need"; then c3=ok; drv_note="detected $got >= required $need"
        else c3="bad(detected $got < required $need)"; drv_note="detected $got < required $need"; fi
    fi
    for c in "$c1" "$c2" "$c3" "$c4" "$c5"; do [[ $c == ok ]] || g5=FAIL; done
    g5note="c1_getSystemCaps_nvapi=$c1  c2_no_failed_NGX=$c2  c3_driver=$c3  c4_no_missing_ctx=$c4  c5_sl.dlss_loaded=$c5"
    # the EXACT measured nvapi/driver numbers, whatever they are (game archive: NVAPI failed +
    # detected 0.0.0; headless it can differ - that is data, reported either way)
    nvapi_drv=$(grep -oE 'getSystemCaps\] NVIDIA driver [0-9]+\.[0-9.]+' "$G5EV" | head -1)
    nvapi_fail_txt=$(grep -F 'NVAPI failed to initialize' "$G5EV" | head -1)
    isnv=$(grep -E 'isNvidia Detected|NVIDIA driver [0-9]' "$PARSE" | head -1)
    sldrv=$(grep -E '^SLDLSS slGetFeatureRequirements' "$PARSE" | grep -oE 'driverDetected=[0-9.]+ driverRequired=[0-9.]+' | head -1)
    g5detail="nvapi: ${nvapi_drv:-${nvapi_fail_txt:-(no NVAPI line from getSystemCaps)}}; ${drv_note}; sl-requirements: "
    g5detail+="${sldrv:-n/a}; evidence: [sl-cb]=$(grep -c '^\[sl-cb\]' "$PARSE" 2>/dev/null) "
    g5detail+="SLDLSS=$(grep -c '^SLDLSS' "$PARSE" 2>/dev/null) slInit=$(grep -c '^SL slInit' "$PARSE" 2>/dev/null)"
    g5detail+=" isNvidia='${isnv:-n/a}' files=$G5EV"
fi

# ---- HARNESS SUMMARY -----------------------------------------------------------------------------
echo
echo "HARNESS SUMMARY"
echo "  mode=$MODE expect_fail=$EXPECT sl=$SLMODE frames=$FRAMES nvngx=$(basename "$NVNGX")"
echo "  g1 OptiScaler : $g1   $g1note"
echo "  g2 port log   : $g2   $g2note"
echo "  g3 non-black  : $g3   $g3note"
echo "  g4 timing     : $g4   $g4note"
[[ $SLMODE -eq 1 ]] && { echo "  g5 SL boot    : $g5   $g5note"; echo "                   $g5detail"; }
if [[ -f "$OUTD/summary.txt" ]]; then
    echo "  --- opti_client summary.txt ---"
    sed 's/^/  /' "$OUTD/summary.txt"
fi
ALL=PASS
for x in "$g1" "$g2" "$g3" "$g4"; do [[ $x == PASS ]] || ALL=FAIL; done
[[ $RC -eq 0 ]] || ALL=FAIL
TAIL="opti_client rc=$RC, exe summary RESULT=${EXE_RESULT:-n/a}"
# g5 joins the verdict only when the caller asked for the Streamline boot (--sl): the baseline
# without --sl stays exactly the old four-gate run, and with --sl a g5 FAIL means "the game's
# Streamline doom reproduced headlessly" -> the run exits non-zero like every other failed gate
# (flip it green after an SL/NVAPI fix by the same evidence, no game launch needed).
if [[ $SLMODE -eq 1 ]]; then
    [[ $g5 == PASS ]] || ALL=FAIL
    TAIL="$TAIL, g5=$g5"
fi
echo "HARNESS RESULT: $ALL  ($TAIL)"
[[ $ALL == PASS ]] || exit 1
exit 0