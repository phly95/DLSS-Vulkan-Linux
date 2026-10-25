#!/usr/bin/env bash
# tools/verify/run_all.sh: the referee for the Vulkan port of NeuralHub v0.8.0's DLSS-SR (presets M, L)
# and DLSS-RR (preset F). Runs on the RX 9070 XT under SteamOS only. Contract: tools/verify/README.md.
#
#   bash tools/verify/run_all.sh            (PORT_RUNNER=<path> overrides build/vk/vk_runner)
#
# Exit 0 and "VERIFY RESULT: PASS <run id>" only when every check passes; otherwise
# "VERIFY RESULT: FAIL <reason>" at the first failure.
#
#   V10 provenance: run from a clean tree; tools/verify identical to the newest "verify: freeze v2" commit
#   V6  SteamOS host; no ROCm / HIP library, /dev/kfd, NeuralHub code object, drm-shim or software
#       Vulkan driver in any port process; the port's build, its package and an installed game folder
#       free of NVIDIA / NeuralHub content (r2scan.py)
#   V7  the port runs on "AMD Radeon RX 9070 XT (RADV GFX1201)" through the system's RADV ICD only, and
#       every pipeline it creates has 0 spills and 0 scratch (Mesa's own RADV_DEBUG=shaderstats)
#   V1  fresh inputs from a /dev/urandom seed: a newly rendered moving scene (Blender, ground truth on)
#       and seeded static sequences (pan, reset frame 4..8, ground truth)
#   V2  every port run happens before any oracle run: no oracle output exists while the port runs
#   V3  the matrix: SR M and L (moving, static, ratio 1.5), SR gather formats (HDR, LDR, motion vectors,
#       depth), output-resolution motion vectors, auto exposure; RR F (moving, static, native
#       resolution, gather formats)
#   V4  per cell, port vs oracle: >= 50 dB every frame, >= 55 dB mean, finite; with ground truth
#       |PSNR(port, GT) - PSNR(oracle, GT)| <= 0.1 dB every frame
#   V5  the referee corrupts copies of weights and inputs; each corrupted port run must FAIL V4 against
#       the oracle on the uncorrupted inputs (as must plain bicubic)
#   V8  per cell, the port's median GPU frame time (frames 1..) <= 3x the oracle's (both from
#       per-frame GPU timestamps)
#   V9  the package (build/package/install.sh) installs into a scratch game folder, ngx_client.exe drives
#       SR (M, L) and RR through NGX under Proton with D3D12 and Vulkan, its outputs pass V4 against the
#       oracle, create / release / resize cycles work, CreateFeature refuses without the NVIDIA DLLs and
#       with the wrong ones, and uninstalling restores the folder byte for byte
set -uo pipefail

ROOT=$(cd -- "$(dirname -- "$0")/../.." && pwd)
cd "$ROOT" || exit 1
RUN_ID=$(date +%Y%m%d_%H%M%S)_$$
SCRATCH=$ROOT/build/verify/run_$RUN_ID
PY=$ROOT/build/venv/bin/python
EV="$PY -I $ROOT/tools/verify/eval_verify.py"
MC="$PY -I $ROOT/tools/verify/make_cells.py"
ROCM_IMAGE=docker.io/rocm/dev-ubuntu-24.04:7.2.4
DEVICE="AMD Radeon RX 9070 XT (RADV GFX1201)"
PORT_RUNNER=${PORT_RUNNER:-$ROOT/build/vk/vk_runner}
FRAMES=12

fail() {
    echo
    echo "================================================================================"
    echo "VERIFY RESULT: FAIL $1"
    echo "================================================================================"
    exit 1
}
run() { echo "  \$ $*"; "$@"; }
cpath() { [[ "$1" == "$ROOT"* ]] && echo "/w${1#$ROOT}" || fail "path $1 is outside the repository"; }
rocm() {   # rocm <log> <command...>: run in the ROCm container, where the oracles live
    local log=$1; shift
    podman run --rm --device /dev/kfd --device /dev/dri --group-add keep-groups --security-opt label=disable \
        -v "$ROOT":/w "$ROCM_IMAGE" bash -c "cd /w && $*" > "$log" 2>&1
}

echo "================================================================================"
echo "DLSS5 Vulkan port referee, run $RUN_ID"
echo "================================================================================"

# ---- V10 ----------------------------------------------------------------------------------------
echo "[V10] provenance"
HEAD=$(git rev-parse HEAD) || fail "V10: not a git checkout"
FREEZE=$(git log --grep='^verify: freeze v2' -n 1 --format=%H)
[[ -n "$FREEZE" ]] || fail "V10: no commit whose message starts with 'verify: freeze v2'"
echo "  HEAD   $HEAD"
echo "  freeze $FREEZE $(git log -1 --format=%s "$FREEZE")"
DIFF=$(git diff --stat "$FREEZE" HEAD -- tools/verify)
echo "  git diff --stat $FREEZE HEAD -- tools/verify:"
echo "${DIFF:-  (empty)}"
[[ -z "$DIFF" ]] || fail "V10: tools/verify changed since the freeze commit"
DIRTY=$(git status --porcelain --untracked-files=all)
[[ -z "$DIRTY" ]] || { echo "$DIRTY"; fail "V10: the working tree is not clean"; }
mkdir -p "$SCRATCH"
# oracle outputs of earlier runs must not be readable while this run's port runs
find "$ROOT/build/verify" -maxdepth 2 -name out_ref -type d -prune -exec rm -rf {} + 2>/dev/null

# ---- V6: the host ------------------------------------------------------------------------------
echo "[V6] host"
grep -qx 'ID=steamos' /etc/os-release || fail "V6: not SteamOS ($(grep '^ID=' /etc/os-release))"
echo "  $(grep '^PRETTY_NAME=' /etc/os-release)"
[[ -x "$PY" ]] || fail "no $PY (HANDOFF.md, setup)"
command -v podman >/dev/null || fail "podman missing"

# ---- V7: the device the port is given ----------------------------------------------------------
echo "[V7] device"
RADV_ICD=$(ls /usr/share/vulkan/icd.d/radeon_icd.x86_64.json 2>/dev/null | head -1)
[[ -f "$RADV_ICD" ]] || fail "V7: no system RADV ICD in /usr/share/vulkan/icd.d"
# the port runner: the system RADV ICD only, no preloads or implicit layers, Mesa's statistics, no cache
PORT_ENV=(env -u LD_PRELOAD -u VK_LAYER_PATH -u VK_ADD_LAYER_PATH VK_ICD_FILENAMES="$RADV_ICD" VK_DRIVER_FILES="$RADV_ICD"
          VK_LOADER_LAYERS_DISABLE='~implicit~' RADV_DEBUG=shaderstats MESA_SHADER_CACHE_DISABLE=true)
# under Proton: the same driver, but layers stay as the package's launch environment sets them
PROTON_ENV=(env -u LD_PRELOAD VK_ICD_FILENAMES="$RADV_ICD" VK_DRIVER_FILES="$RADV_ICD" MESA_SHADER_CACHE_DISABLE=true)
podman image exists localhost/nrbuild:2 || bash tools/hwtest/build_native.sh >/dev/null 2>&1 || true
podman run --rm -v "$ROOT":/w localhost/nrbuild:2 bash -c \
    "g++ -std=c++17 -O1 -I/w/reference-nr-vulkan/toolchain/Vulkan-Headers/include /w/tools/verify/vkprobe.cpp -o /w/build/verify/run_$RUN_ID/vkprobe -lvulkan" \
    || fail "V7: cannot build vkprobe (localhost/nrbuild:2, tools/hwtest/build_native.sh)"
"${PORT_ENV[@]}" "$SCRATCH/vkprobe" > "$SCRATCH/vkprobe.log" 2>&1 || { cat "$SCRATCH/vkprobe.log"; fail "V7: vkprobe failed"; }
cat "$SCRATCH/vkprobe.log"
[[ $(grep -c '^VK_PROBE_DEVICE:' "$SCRATCH/vkprobe.log") -eq 1 ]] || fail "V7: the port's environment offers more or fewer than one device"
grep -q "^VK_PROBE_DEVICE: $DEVICE | driverID 3 | radv .* | fp8 1 | coopmat 1$" "$SCRATCH/vkprobe.log" \
    || fail "V7: the device is not $DEVICE through RADV (driverID 3) with FP8 and cooperative matrices"

# ---- inputs the referee prepares itself --------------------------------------------------------
echo "[setup] weights from the user's DLLs, oracles"
for d in build/re/nvngx_dlss_310.9.1.dll build/re/nvngx_dlssd_310.9.1.dll build/re/nvngx_dlss_310.9.16.dll build/re/nvngx_dlssd_310.9.16.dll; do
    [[ -f "$d" ]] || fail "missing $d (tools/verify/README.md: restoring the inputs)"
done
W=$SCRATCH/weights
run "$PY" -I tools/sr/sr_weights.py build/re/nvngx_dlss_310.9.1.dll --out "$W/sr_raw" > "$SCRATCH/weights.log" 2>&1 || fail "sr_weights.py"
run "$PY" -I tools/sr/sr_pack.py "$W/sr_raw" M "$W/sr_M" >> "$SCRATCH/weights.log" 2>&1 || fail "sr_pack.py M"
run "$PY" -I tools/sr/sr_pack.py "$W/sr_raw" L "$W/sr_L" >> "$SCRATCH/weights.log" 2>&1 || fail "sr_pack.py L"
run "$PY" -I tools/rr/rr_weights.py build/re/nvngx_dlssd_310.9.1.dll --out "$W/rr_raw" >> "$SCRATCH/weights.log" 2>&1 || fail "rr_weights.py"
run "$PY" -I tools/rr/rr_pack.py "$W/rr_raw" "$W/rr_F" >> "$SCRATCH/weights.log" 2>&1 || fail "rr_pack.py"
grep -q "preset L: .*NeuralHub checksum matches" "$SCRATCH/weights.log" || fail "SR preset L weights do not match NeuralHub's checksum"
grep -q "fingerprint .*: matches NeuralHub" "$SCRATCH/weights.log" || fail "RR weights do not match NeuralHub's fingerprint"
SR_CO=artifacts/neuralhub-v0.8.0/gfx1201/sr.gfx1201.co
RR_CO=artifacts/neuralhub-v0.8.0/gfx1201/rr.gfx1201.co
mkdir -p "$SCRATCH/bin"
rocm "$SCRATCH/oracle_build.log" "hipcc -O2 -std=c++17 tools/sr/srref/srref.cpp -o $(cpath "$SCRATCH/bin")/srref && \
    hipcc -O2 -std=c++17 tools/rr/rrref/rrref.cpp -o $(cpath "$SCRATCH/bin")/rrref" \
    || { cat "$SCRATCH/oracle_build.log"; fail "cannot build the oracles"; }

# ---- V1: fresh inputs --------------------------------------------------------------------------
echo "[V1] fresh inputs"
SEED=$(od -An -N4 -tu4 < /dev/urandom | tr -d ' ')
echo "  seed $SEED"
SEQ=$SCRATCH/seq
BLENDER=$ROOT/build/blender/blender-4.2.3-linux-x64/blender
[[ -x "$BLENDER" && -d build/scenes/Sponza && -d build/scenes/DamagedHelmet ]] \
    || fail "V1: Blender 4.2.3 or the Sponza / DamagedHelmet assets are missing (tools/scenes/README.md)"
podman image exists localhost/rocm-blender:1 || fail "V1: no localhost/rocm-blender:1 image (tools/scenes/README.md)"
mkdir -p build/blender/kcache "$SEQ"
podman run --rm --device /dev/kfd --device /dev/dri --group-add keep-groups --security-opt label=disable \
    -v "$ROOT":/w -v "$ROOT/build/blender/kcache":/root/.cache/cycles localhost/rocm-blender:1 bash -c "cd /w && \
    $(cpath "$BLENDER") -b -P tools/scenes/render_seq.py -- $(cpath "$SEQ") --sponza build/scenes/Sponza \
    --helmet build/scenes/DamagedHelmet --frames $FRAMES --render 960 540 --output 1920 1080 --noisy-spp 2 \
    --clean-spp 256 --gt-spp 512 --device HIP --seed $SEED" > "$SCRATCH/render.log" 2>&1 \
    || { tail -30 "$SCRATCH/render.log"; fail "V1: render_seq.py"; }
"$BLENDER" -b -P tools/scenes/convert_seq.py -- "$SEQ" > "$SCRATCH/convert.log" 2>&1 || { tail -20 "$SCRATCH/convert.log"; fail "V1: convert_seq.py"; }
for i in $(seq 0 $((FRAMES - 1))); do [[ -f "$SEQ/gt/gt_$i.f32" ]] || fail "V1: no ground truth for frame $i"; done
"$PY" -I tools/scenes/check_mv.py "$SEQ/sr" | head -3
PIC=reference-nr-vulkan/docs/ngx-verification/310.8.0/single-frame-inputs/3840x2160.png
[[ -f "$PIC" ]] || fail "V1: missing $PIC"
B=$SCRATCH/base
run "$PY" tools/sr/gen_frames.py "$PIC" "$B/sr" --render 1920 1080 --output 3840 2160 --frames $FRAMES --seed "$SEED" --path \
    --gt "$B/sr_gt" --mv-out > /dev/null || fail "V1: gen_frames.py"
run "$PY" tools/sr/gen_frames.py "$PIC" "$B/sr_q15" --render 1280 720 --output 1920 1080 --frames $FRAMES --seed "$SEED" --path \
    --gt "$B/sr_q15_gt" > /dev/null || fail "V1: gen_frames.py (ratio 1.5)"
run "$PY" tools/rr/gen_rr_frames.py "$PIC" "$B/rr" --render 1920 1080 --output 3840 2160 --frames $FRAMES --seed "$SEED" --path \
    --gt "$B/rr_gt" > /dev/null || fail "V1: gen_rr_frames.py"
run "$PY" tools/rr/gen_rr_frames.py "$PIC" "$B/rr_native" --render 1920 1080 --output 1920 1080 --frames $FRAMES --seed "$SEED" --path \
    --gt "$B/rr_native_gt" > /dev/null || fail "V1: gen_rr_frames.py (native)"
grep -l '"reset": true' "$B"/sr/params_*.json | sed 's/^/  reset: /'

# ---- V3: the matrix (name mode preset input-dir gt-dir) ------------------------------------------
C=$SCRATCH/cells
mkdir -p "$C"
CELLS=()
cell() { CELLS+=("$*"); }
for P in M L; do
    cell "sr_${P}_seq sr $P $SEQ/sr $SEQ/gt"
    cell "sr_${P}_static sr $P $B/sr $B/sr_gt"
    cell "sr_${P}_q15 sr $P $B/sr_q15 $B/sr_q15_gt"
done
mk() { "$@" > /dev/null || fail "make_cells.py $*"; }
for c in 4 5 6; do mk $MC fmt "$B/sr" "$C/sr_col$c" --mode sr --col $c; cell "sr_M_col$c sr M $C/sr_col$c $B/sr_gt"; done
for c in 1 2 3; do mk $MC fmt "$B/sr" "$C/sr_ldr$c" --mode sr --col $c --ldr; cell "sr_M_ldr$c sr M $C/sr_ldr$c $B/sr_gt"; done
mk $MC fmt "$B/sr" "$C/sr_mv1_dep1" --mode sr --mv 1 --dep 1; cell "sr_M_mv1_dep1 sr M $C/sr_mv1_dep1 $B/sr_gt"
mk $MC fmt "$B/sr" "$C/sr_mv2" --mode sr --mv 2; cell "sr_M_mv2 sr M $C/sr_mv2 $B/sr_gt"
for c in 0 1; do mk $MC mvhi "$B/sr" "$C/sr_mvhi$c" --mv $c; cell "sr_M_mvhi$c sr M $C/sr_mvhi$c $B/sr_gt"; done
mk $MC params "$B/sr" "$C/sr_ae" auto_exposure=1; cell "sr_M_ae sr M $C/sr_ae $B/sr_gt"
mk $MC params "$B/sr" "$C/sr_ae_exact" auto_exposure=1 ae_exact_sum=1; cell "sr_M_ae_exact sr M $C/sr_ae_exact $B/sr_gt"
mk $MC params "$B/sr_q15" "$C/sr_L_q15_ae" auto_exposure=1; cell "sr_L_q15_ae sr L $C/sr_L_q15_ae $B/sr_q15_gt"
cell "rr_F_seq rr F $SEQ/rr $SEQ/gt"
cell "rr_F_static rr F $B/rr $B/rr_gt"
cell "rr_F_native rr F $B/rr_native $B/rr_native_gt"
mk $MC fmt "$B/rr" "$C/rr_fmt_a" --mode rr --col 11 --mv 10 --dep 16 --rgh 4; cell "rr_F_fmt_a rr F $C/rr_fmt_a $B/rr_gt"
mk $MC fmt "$B/rr" "$C/rr_fmt_b" --mode rr --col 1 --nrm 14 --normals-unorm --alb 5 --spa 7; cell "rr_F_fmt_b rr F $C/rr_fmt_b $B/rr_gt"
mk $MC fmt "$B/rr" "$C/rr_fmt_c" --mode rr --col 12 --alb 6 --dep 3 --rgh 8; cell "rr_F_fmt_c rr F $C/rr_fmt_c $B/rr_gt"

# ---- V5: corrupted copies (name mode preset input-dir weights-dir base-cell) ----------------------
BAD=()
mk $MC corrupt-weights "$W/sr_M" "$W/sr_M_corrupt" stage4.bin
mk $MC corrupt-weights "$W/rr_F" "$W/rr_F_corrupt" blob5.bin
mk $MC params "$B/sr" "$C/sr_reset_all" reset=true
mk $MC params "$B/rr" "$C/rr_reset_all" reset=true
mk $MC zero-mv "$B/sr" "$C/sr_zero_mv"
mk $MC zero-mv "$SEQ/sr" "$C/sr_seq_zero_mv"
mk $MC zero-mv "$SEQ/rr" "$C/rr_seq_zero_mv"
BAD+=("v5_sr_weights sr M $B/sr $W/sr_M_corrupt sr_M_static")
BAD+=("v5_sr_reset sr M $C/sr_reset_all $W/sr_M sr_M_static")
BAD+=("v5_sr_zero_mv sr M $C/sr_zero_mv $W/sr_M sr_M_static")
BAD+=("v5_sr_seq_zero_mv sr M $C/sr_seq_zero_mv $W/sr_M sr_M_seq")
BAD+=("v5_rr_weights rr F $B/rr $W/rr_F_corrupt rr_F_static")
BAD+=("v5_rr_reset rr F $C/rr_reset_all $W/rr_F rr_F_static")
BAD+=("v5_rr_seq_zero_mv rr F $C/rr_seq_zero_mv $W/rr_F rr_F_seq")
weights_for() { case "$1$2" in srM) echo "$W/sr_M";; srL) echo "$W/sr_L";; rrF) echo "$W/rr_F";; esac; }

# ---- the port (V2: before any oracle run), with V6 sampling and V7 per run ------------------------
[[ -x "$PORT_RUNNER" ]] || fail "port runner not found at $PORT_RUNNER (the Vulkan port is not built)"
V6LOG=$SCRATCH/v6_monitor.log
: > "$V6LOG"
monitor() {   # monitor <log> <command...>: run, sampling /proc/<pid>/maps and fds of it and its children
    local log=$1; shift
    "$@" > "$log" 2>&1 &
    local pid=$! rc
    while kill -0 "$pid" 2>/dev/null; do
        # the whole process tree (Proton starts wine, which starts the program)
        local tree=$pid frontier=$pid kids
        while [[ -n "$frontier" ]]; do
            kids=$(for q in $frontier; do pgrep -P "$q" 2>/dev/null; done)
            tree="$tree $kids"; frontier=$kids
        done
        for p in $tree; do
            if [[ -r /proc/$p/maps ]]; then
                grep -E 'libamdhip64|libhsa-runtime|libamd_comgr|librocm|libhiprtc|drm_shim|noop_drm|libvulkan_lvp|llvmpipe|\.co$|/build/nh/' \
                    "/proc/$p/maps" 2>/dev/null | sed "s|^|V6 VIOLATION pid $p maps: |" >> "$V6LOG"
            fi
            for fd in /proc/$p/fd/*; do
                t=$(readlink "$fd" 2>/dev/null) || continue
                case "$t" in
                    /dev/kfd|*.co|*/build/nh/*|*/out_ref/*|*/build/oracle/*) echo "V6 VIOLATION pid $p fd: $t" >> "$V6LOG";;
                esac
            done
        done
        sleep 0.01
    done
    wait "$pid"; rc=$?
    if grep -q "V6 VIOLATION" "$V6LOG"; then sort -u "$V6LOG" | head -20; fail "V6: forbidden library, device or file in a port process ($log)"; fi
    return $rc
}
port_run() {   # port_run <name> <mode> <preset> <in> <weights>
    local n=$1 mode=$2 preset=$3 in=$4 w=$5
    mkdir -p "$SCRATCH/out_port/$n"
    echo "  [port] $n"
    monitor "$SCRATCH/port_$n.log" "${PORT_ENV[@]}" "$PORT_RUNNER" --mode "$mode" --preset "$preset" --weights "$w" \
        --in "$in" --out "$SCRATCH/out_port/$n"
}
echo "[V2] port runs (no oracle output exists yet)"
for c in "${CELLS[@]}"; do
    read -r n mode preset in gt <<< "$c"
    port_run "$n" "$mode" "$preset" "$in" "$(weights_for "$mode" "$preset")" || { tail -30 "$SCRATCH/port_$n.log"; fail "port runner failed on $n"; }
    $EV check-radv "$SCRATCH/port_$n.log" --device "$DEVICE" > "$SCRATCH/v7_$n.txt" || { cat "$SCRATCH/v7_$n.txt"; fail "V7 on $n"; }
done
cat "$SCRATCH/v7_${CELLS[0]%% *}.txt"
for b in "${BAD[@]}"; do
    read -r n mode preset in w base <<< "$b"
    port_run "$n" "$mode" "$preset" "$in" "$w" || echo "  ($n: the port exited non-zero; V5 still needs its output to fail V4)"
done

# ---- V9: the package under Proton (still before any oracle run) --------------------------------
echo "[V9] package under Proton"
INSTALLER=$ROOT/build/package/install.sh
[[ -f "$INSTALLER" ]] || fail "V9: no package at $INSTALLER"
bash tools/verify/ngx_client/build.sh "$SCRATCH/ngx_client.exe" > "$SCRATCH/ngx_build.log" 2>&1 || { cat "$SCRATCH/ngx_build.log"; fail "V9: ngx_client build"; }
STEAM=$HOME/.local/share/Steam
PROTON=$(ls -d "$STEAM"/compatibilitytools.d/*/proton "$STEAM"/steamapps/common/Proton*/proton 2>/dev/null | sort -V | tail -1)
[[ -x "$PROTON" ]] || fail "V9: no Proton under $STEAM"
echo "  Proton: $PROTON"
game_dir() {   # game_dir <dir> <dll set: good | none | wrong>
    mkdir -p "$1"
    cp "$SCRATCH/ngx_client.exe" "$1/"
    case "$2" in
        good) cp build/re/nvngx_dlss_310.9.1.dll "$1/nvngx_dlss.dll"; cp build/re/nvngx_dlssd_310.9.1.dll "$1/nvngx_dlssd.dll";;
        wrong) cp build/re/nvngx_dlss_310.9.16.dll "$1/nvngx_dlss.dll"; cp build/re/nvngx_dlssd_310.9.16.dll "$1/nvngx_dlssd.dll";;
    esac
    (cd "$1" && find . -type f -exec sha256sum {} + | sort) > "$1.before"
    run bash "$INSTALLER" "$1" optiscaler > "$1.install.log" 2>&1 || { cat "$1.install.log"; fail "V9: install into $1"; }
}
winpath() { echo "Z:${1//\//\\}"; }   # a Linux path as the Wine process sees it
proton_run() {   # proton_run <game dir> <log> <ngx_client args...>; the package's launch environment applies
    local g=$1 log=$2 rc; shift 2
    local envs=()
    if [[ -f "$g/dlss5-amd-launch.env" ]]; then mapfile -t envs < <(grep -E '^[A-Za-z_][A-Za-z0-9_]*=' "$g/dlss5-amd-launch.env"); fi
    mkdir -p "$SCRATCH/prefix_$(basename "$g")" "$SCRATCH/cwd"
    pushd "$SCRATCH/cwd" > /dev/null || fail "pushd"     # NGX writes its logs to the working directory, not the game folder
    monitor "$log" "${PROTON_ENV[@]}" "${envs[@]}" STEAM_COMPAT_DATA_PATH="$SCRATCH/prefix_$(basename "$g")" \
        STEAM_COMPAT_CLIENT_INSTALL_PATH="$STEAM" "$PROTON" run "$g/ngx_client.exe" "$@"
    rc=$?
    popd > /dev/null || fail "popd"
    return $rc
}
uninstall_check() {
    run bash "$INSTALLER" "$1" remove > "$1.remove.log" 2>&1 || { cat "$1.remove.log"; fail "V9: uninstall from $1"; }
    (cd "$1" && find . -type f -exec sha256sum {} + | sort) > "$1.after"
    diff -u "$1.before" "$1.after" || fail "V9: $1 is not restored byte for byte after uninstalling"
}
G=$SCRATCH/game
game_dir "$G" good
"$PY" -I tools/verify/r2scan.py --dir "$G" --allow-file "$G/nvngx_dlss.dll" --allow-file "$G/nvngx_dlssd.dll" | tail -5
[[ ${PIPESTATUS[0]} -eq 0 ]] || fail "V6: the installed game folder holds NVIDIA / NeuralHub content"
V9CELLS=()
for api in d3d12 vulkan; do
    for spec in "sr M sr_M_static $B/sr" "sr L sr_L_static $B/sr" "rr F rr_F_static $B/rr"; do
        read -r feat preset base in <<< "$spec"
        n=v9_${api}_${base}
        echo "  [proton] $n"
        mkdir -p "$SCRATCH/out_port/$n"
        args=(--api "$api" --feature "$feat" --frames "$(winpath "$in")" --out "$(winpath "$SCRATCH/out_port/$n")" --test-cycles)
        [[ "$feat" == sr ]] && args+=(--preset "$preset")
        proton_run "$G" "$SCRATCH/$n.log" "${args[@]}" || { tail -30 "$SCRATCH/$n.log"; fail "V9: $n"; }
        V9CELLS+=("$n $base")
    done
done
uninstall_check "$G"
for set in none wrong; do
    g=$SCRATCH/game_$set
    game_dir "$g" "$set"
    for api in d3d12 vulkan; do
        for feat in sr rr; do
            n=v9_refuse_${set}_${api}_${feat}
            echo "  [proton] $n"
            proton_run "$g" "$SCRATCH/$n.log" --api "$api" --feature "$feat" --frames "$(winpath "$B/$feat")" --expect-create-fail \
                || { tail -20 "$SCRATCH/$n.log"; fail "V9: CreateFeature did not refuse ($set NVIDIA DLLs, $api, $feat)"; }
        done
    done
    uninstall_check "$g"
done

# ---- V6: the port's build and package ----------------------------------------------------------
echo "[V6] scan of the port's build and package"
for d in build/vk build/shaders build/package; do [[ -d "$d" ]] || fail "V6: $d missing"; done
"$PY" -I tools/verify/r2scan.py --dir build/vk --dir build/shaders --dir build/package | tail -8
[[ ${PIPESTATUS[0]} -eq 0 ]] || fail "V6: NVIDIA / NeuralHub content in the port's build or package"

# ---- the oracle (only now) -----------------------------------------------------------------------
echo "[V2] oracle runs"
for c in "${CELLS[@]}"; do
    read -r n mode preset in gt <<< "$c"
    mkdir -p "$SCRATCH/out_ref/$n"
    echo "  [oracle] $n"
    if [[ "$mode" == sr ]]; then
        rocm "$SCRATCH/ref_$n.log" "$(cpath "$SCRATCH/bin")/srref $SR_CO $(cpath "$(weights_for sr "$preset")") $(cpath "$in") \
            $(cpath "$SCRATCH/out_ref/$n") $([[ $preset == L ]] && echo --L)" || { tail -20 "$SCRATCH/ref_$n.log"; fail "oracle srref failed on $n"; }
    else
        rocm "$SCRATCH/ref_$n.log" "$(cpath "$SCRATCH/bin")/rrref $RR_CO $(cpath "$W/rr_F") $(cpath "$in") $(cpath "$SCRATCH/out_ref/$n")" \
            || { tail -20 "$SCRATCH/ref_$n.log"; fail "oracle rrref failed on $n"; }
    fi
    cp "$in/frame.json" "$SCRATCH/out_ref/$n/frame.json"
done

# ---- V4, V8 per cell ---------------------------------------------------------------------------
echo "[V4, V8] cells"
for c in "${CELLS[@]}"; do
    read -r n mode preset in gt <<< "$c"
    $EV check-cell "$SCRATCH/out_port/$n" "$SCRATCH/out_ref/$n" --gt "$gt" --name "$n" || fail "V4 on $n"
    $EV check-timing "$SCRATCH/port_$n.log" "$SCRATCH/ref_$n.log" --frames $FRAMES --name "$n" || fail "V8 on $n"
done

# ---- V5 ----------------------------------------------------------------------------------------
echo "[V5] corrupted runs must fail V4"
for b in "${BAD[@]}"; do
    read -r n mode preset in w base <<< "$b"
    $EV check-degraded "$SCRATCH/out_port/$n" "$SCRATCH/out_ref/$base" --name "$n" || fail "V5: $n passes V4"
done
"$PY" -I - "$B/sr" "$SCRATCH/out_port/v5_bicubic" <<'EOF' || fail "V5: bicubic"
import json, os, sys
import numpy as np
from PIL import Image
src, out = sys.argv[1:3]
fj = json.load(open(os.path.join(src, "frame.json")))
(w, h), (ow, oh) = fj["render"], fj["output"]
os.makedirs(out, exist_ok=True)
for i in range(fj["frames"]):
    c = np.fromfile(os.path.join(src, f"col_{i}.f16"), np.float16).reshape(h, w, 4).astype(np.float32)
    up = [np.asarray(Image.fromarray(c[..., k], mode="F").resize((ow, oh), Image.BICUBIC)) for k in range(3)]
    np.concatenate([np.stack(up, -1), np.ones((oh, ow, 1), np.float32)], -1).astype(np.float32).tofile(os.path.join(out, f"out_{i}.f32"))
EOF
$EV check-degraded "$SCRATCH/out_port/v5_bicubic" "$SCRATCH/out_ref/sr_M_static" --name v5_bicubic || fail "V5: bicubic passes V4"

# ---- V9: Proton outputs against the oracle -------------------------------------------------------
echo "[V9] Proton outputs"
for v in "${V9CELLS[@]}"; do
    read -r n base <<< "$v"
    gtd=$B/sr_gt; [[ $base == rr_* ]] && gtd=$B/rr_gt
    $EV check-cell "$SCRATCH/out_port/$n" "$SCRATCH/out_ref/$base" --gt "$gtd" --name "$n" || fail "V9: V4 on $n"
done

echo
echo "================================================================================"
echo "VERIFY RESULT: PASS $RUN_ID"
echo "================================================================================"
exit 0
