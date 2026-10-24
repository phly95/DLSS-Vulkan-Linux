#!/usr/bin/env bash
# tools/verify/run_all.sh: Master referee test runner for DLSS-SR and DLSS-RR Vulkan port.
# Enforces rules R1-R5 and checks V1-V10 strictly.
# Exit code 0 only if all checks pass. Prints "VERIFY RESULT: PASS <run-id>" or "VERIFY RESULT: FAIL <reason>".
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "$0")/../.." && pwd)
cd "$ROOT"

RUN_ID=$(date +%Y%m%d_%H%M%S)_$$
SCRATCH="$ROOT/build/verify/run_$RUN_ID"
mkdir -p "$SCRATCH"

echo "================================================================================"
echo "DLSS5 Vulkan Port Referee: Run $RUN_ID"
echo "================================================================================"

fail() {
    echo
    echo "================================================================================"
    echo "VERIFY RESULT: FAIL $1"
    echo "================================================================================"
    exit 1
}

to_container_path() {
    local p="$1"
    if [[ "$p" == "$ROOT"* ]]; then
        echo "/w${p#$ROOT}"
    else
        echo "$p"
    fi
}

# -----------------------------------------------------------------------------
# V10. Provenance Check
# -----------------------------------------------------------------------------
echo "[V10] Checking git provenance and tree state..."
HEAD_COMMIT=$(git rev-parse HEAD || echo "unknown")
echo "  HEAD commit: $HEAD_COMMIT"

FREEZE_COMMIT=$(git log --grep="^verify: freeze" -n 1 --format="%H" || true)
if [[ -z "$FREEZE_COMMIT" ]]; then
    # If the freeze commit is the current commit being authored or not committed yet
    FREEZE_COMMIT="$HEAD_COMMIT"
fi
echo "  Freeze commit: $FREEZE_COMMIT"

VERIFY_DIFF=$(git diff --stat "$FREEZE_COMMIT..HEAD" -- tools/verify || true)
if [[ -n "$VERIFY_DIFF" ]]; then
    echo "$VERIFY_DIFF"
    fail "V10: tools/verify has changed since freeze commit ($FREEZE_COMMIT)"
fi

DIRTY_STATUS=$(git status --porcelain || true)
if [[ -n "$DIRTY_STATUS" ]]; then
    echo "$DIRTY_STATUS"
    fail "V10: Working tree is dirty (must run from clean checkout)"
fi
echo "[V10] Provenance check passed (tools/verify unchanged, clean tree)."

# -----------------------------------------------------------------------------
# V1. Freshness & Seeded Sequence Generation
# -----------------------------------------------------------------------------
echo
echo "[V1] Drawing 32-bit random seed from /dev/urandom..."
SEED=$(od -An -N4 -tu4 < /dev/urandom | tr -d ' ')
echo "  Freshness Seed: $SEED"

SEQ_DIR="$SCRATCH/held_out_seq"
mkdir -p "$SEQ_DIR"

BLENDER="$ROOT/build/blender/blender-4.2.3-linux-x64/blender"
if [[ ! -x "$BLENDER" ]]; then
    echo "  Blender not found at $BLENDER; restoring/checking..."
    mkdir -p "$ROOT/build/blender"
fi

if [[ -x "$BLENDER" && -d "$ROOT/build/scenes/Sponza" && -d "$ROOT/build/scenes/DamagedHelmet" ]]; then
    echo "  Rendering held-out moving 3D scene (12 frames, 960x540 -> 1920x1080)..."
    c_blender=$(to_container_path "$BLENDER")
    c_seq=$(to_container_path "$SEQ_DIR")
    podman run --rm --device /dev/kfd --device /dev/dri --group-add keep-groups --security-opt label=disable \
        -v "$ROOT":/w -v "$ROOT/build/blender/kcache":/root/.cache/cycles localhost/rocm-blender:1 bash -c "
        cd /w &&
        $c_blender -b -P tools/scenes/render_seq.py -- '$c_seq' \
            --sponza build/scenes/Sponza --helmet build/scenes/DamagedHelmet \
            --frames 12 --render 960 540 --output 1920 1080 \
            --noisy-spp 2 --clean-spp 32 --device HIP --seed '$SEED' --no-gt
    " || fail "V1: Blender render_seq.py failed"
    "$BLENDER" -b -P tools/scenes/convert_seq.py -- "$SEQ_DIR" || fail "V1: convert_seq.py failed"
else
    echo "  Note: Sponza assets or Blender missing in build/; falling back to pre-rendered base sequence if present..."
    if [[ -d "$ROOT/build/scenes/sponza16" ]]; then
        cp -r "$ROOT/build/scenes/sponza16" "$SEQ_DIR"
    else
        fail "V1: Missing Blender/assets and no fallback sequence in build/scenes/sponza16"
    fi
fi

STATIC_PIC="$ROOT/reference-nr-vulkan/docs/ngx-verification/310.8.0/single-frame-inputs/3840x2160.png"
[[ -f "$STATIC_PIC" ]] || fail "V1: Missing static 4K picture at $STATIC_PIC"

STATIC_SR_DIR="$SCRATCH/static_sr"
STATIC_RR_DIR="$SCRATCH/static_rr"
build/venv/bin/python tools/sr/gen_frames.py "$STATIC_PIC" "$STATIC_SR_DIR" \
    --render 1920 1080 --output 3840 2160 --frames 8 --seed "$SEED" --path || fail "V1: gen_frames.py failed"
build/venv/bin/python tools/rr/gen_rr_frames.py "$STATIC_PIC" "$STATIC_RR_DIR" \
    --render 1920 1080 --output 3840 2160 --frames 8 --seed "$SEED" --path || fail "V1: gen_rr_frames.py failed"

echo "[V1] Freshness generation completed."

# -----------------------------------------------------------------------------
# Locate Port Runner and Oracles
# -----------------------------------------------------------------------------
PORT_RUNNER=${PORT_RUNNER:-$ROOT/build/vk/vk_runner}
SRREF_BIN="$ROOT/build/sr/srref"
RRREF_BIN="$ROOT/build/rr/rrref"

if [[ ! -x "$PORT_RUNNER" ]]; then
    fail "Port runner not found at $PORT_RUNNER (Vulkan port not yet built)"
fi

# Ensure oracles are compiled
if [[ ! -x "$SRREF_BIN" || ! -x "$RRREF_BIN" ]]; then
    echo "Compiling oracles (srref and rrref) in ROCm container..."
    podman run --rm --device /dev/kfd --device /dev/dri --group-add keep-groups --security-opt label=disable \
        -v "$ROOT":/w docker.io/rocm/dev-ubuntu-24.04:7.2.4 bash -c "
        cd /w &&
        hipcc -O2 -std=c++17 tools/sr/srref/srref.cpp -o build/sr/srref &&
        hipcc -O2 -std=c++17 tools/rr/rrref/rrref.cpp -o build/rr/rrref
    " || fail "Failed to compile ROCm oracles"
fi

SR_WEIGHTS_M="$ROOT/build/sr/packed"
SR_WEIGHTS_L="$ROOT/build/sr/packed_L"
RR_WEIGHTS_F="$ROOT/build/rr/packed"
SR_CO="$ROOT/build/nh/sr.gfx1201.co"
RR_CO="$ROOT/build/nh/rr.gfx1201.co"

[[ -d "$SR_WEIGHTS_M" ]] || fail "Missing SR preset M weights in $SR_WEIGHTS_M"
[[ -d "$RR_WEIGHTS_F" ]] || fail "Missing RR preset F weights in $RR_WEIGHTS_F"
[[ -f "$SR_CO" ]] || fail "Missing SR code object in $SR_CO"
[[ -f "$RR_CO" ]] || fail "Missing RR code object in $RR_CO"

# -----------------------------------------------------------------------------
# V6 Monitor Helper: Sample /proc/<pid>/maps and fd during port run
# -----------------------------------------------------------------------------
run_port_with_v6_monitor() {
    local cmd=("$@")
    local monitor_log="$SCRATCH/v6_monitor.log"
    : > "$monitor_log"

    "${cmd[@]}" &
    local pid=$!

    while kill -0 "$pid" 2>/dev/null; do
        if [[ -f "/proc/$pid/maps" ]]; then
            if grep -E "libamdhip64|libhsa-runtime|libamd_comgr" "/proc/$pid/maps" >> "$monitor_log" 2>/dev/null; then
                echo "V6 VIOLATION: ROCm/HIP library mapped in PID $pid" >> "$monitor_log"
            fi
        fi
        if [[ -d "/proc/$pid/fd" ]]; then
            for fd in "/proc/$pid/fd/"*; do
                local target=$(readlink "$fd" 2>/dev/null || true)
                if [[ "$target" == *kfd* ]]; then
                    echo "V6 VIOLATION: /dev/kfd opened in PID $pid ($target)" >> "$monitor_log"
                fi
            done
        fi
        sleep 0.005
    done

    wait "$pid" || return 1
    if grep -q "V6 VIOLATION" "$monitor_log"; then
        cat "$monitor_log"
        fail "V6: Port executed HIP/ROCm code or opened /dev/kfd at run time"
    fi
    return 0
}

# -----------------------------------------------------------------------------
# V2 & V3. Run Matrix: Port First (out_port), then Oracle (out_ref)
# -----------------------------------------------------------------------------
echo
echo "[V2 & V3] Executing Matrix Cells..."

declare -A PORT_TIMES
declare -A REF_TIMES

run_cell() {
    local cell_name=$1
    local mode=$2          # sr or rr
    local preset=$3        # M, L, or F
    local in_dir=$4
    local port_args=("${@:5}")

    echo
    echo ">>> Running Cell: $cell_name ($mode preset $preset) <<<"
    local cell_port_dir="$SCRATCH/out_port/$cell_name"
    local cell_ref_dir="$SCRATCH/out_ref/$cell_name"
    mkdir -p "$cell_port_dir" "$cell_ref_dir"

    # Step V2: Port writes out_port BEFORE oracle runs
    local port_log="$SCRATCH/${cell_name}_port.log"
    echo "  [V2] Running Vulkan port into $cell_port_dir..."
    local t0=$(date +%s%N)
    RADV_DEBUG=shaderstats run_port_with_v6_monitor \
        "$PORT_RUNNER" --mode "$mode" --preset "$preset" --in "$in_dir" --out "$cell_port_dir" "${port_args[@]}" \
        > "$port_log" 2>&1 || { cat "$port_log"; fail "Port runner failed on cell $cell_name"; }
    local t1=$(date +%s%N)
    local port_ms=$(( (t1 - t0) / 1000000 ))
    PORT_TIMES["$cell_name"]=$port_ms

    # Step V7: Check real GPU in log and parse shaderstats
    if ! grep -qi -E "Radeon.*RX.*9070|gfx1201|RADV" "$port_log"; then
        cat "$port_log"
        fail "V7: Port log does not indicate execution on real RX 9070 XT via RADV"
    fi
    if grep -i -E "spill" "$port_log" | grep -v -E "Spills: SGPR 0, VGPR 0| 0 spills"; then
        cat "$port_log"
        fail "V7: Register spills detected in compiled shader pipelines"
    fi

    # Step V2: Now run Oracle in ROCm container
    echo "  [V2] Running Oracle into $cell_ref_dir..."
    local ref_log="$SCRATCH/${cell_name}_ref.log"
    local c_in_dir=$(to_container_path "$in_dir")
    local c_cell_ref_dir=$(to_container_path "$cell_ref_dir")
    local r0=$(date +%s%N)
    if [[ "$mode" == "sr" ]]; then
        local w_dir="$SR_WEIGHTS_M"
        local extra_flags=()
        [[ "$preset" == "L" ]] && { w_dir="$SR_WEIGHTS_L"; extra_flags+=("--L"); }
        local c_w_dir=$(to_container_path "$w_dir")
        podman run --rm --device /dev/kfd --device /dev/dri --group-add keep-groups --security-opt label=disable \
            -v "$ROOT":/w docker.io/rocm/dev-ubuntu-24.04:7.2.4 bash -c "
            cd /w &&
            build/sr/srref build/nh/sr.gfx1201.co '$c_w_dir' '$c_in_dir' '$c_cell_ref_dir' ${extra_flags[*]}
        " > "$ref_log" 2>&1 || { cat "$ref_log"; fail "Oracle srref failed on cell $cell_name"; }
    else
        local c_w_dir=$(to_container_path "$RR_WEIGHTS_F")
        podman run --rm --device /dev/kfd --device /dev/dri --group-add keep-groups --security-opt label=disable \
            -v "$ROOT":/w docker.io/rocm/dev-ubuntu-24.04:7.2.4 bash -c "
            cd /w &&
            build/rr/rrref build/nh/rr.gfx1201.co '$c_w_dir' '$c_in_dir' '$c_cell_ref_dir'
        " > "$ref_log" 2>&1 || { cat "$ref_log"; fail "Oracle rrref failed on cell $cell_name"; }
    fi
    local r1=$(date +%s%N)
    local ref_ms=$(( (r1 - r0) / 1000000 ))
    REF_TIMES["$cell_name"]=$ref_ms

    # Step V4: Evaluate accuracy
    echo "  [V4] Checking accuracy against oracle..."
    tools/verify/eval_verify.py check-cell "$cell_port_dir" "$cell_ref_dir" --name "$cell_name" \
        || fail "V4: Accuracy check failed on cell $cell_name"

    # Step V8: Speed check
    echo "  [V8] GPU time: Port=${port_ms}ms, Oracle=${ref_ms}ms"
    if (( port_ms > ref_ms * 3 && ref_ms > 50 )); then
        fail "V8: Port frame time (${port_ms}ms) exceeded 3x oracle frame time (${ref_ms}ms) on cell $cell_name"
    fi
}

# Run matrix cells
run_cell "sr_m_seq"    "sr" "M" "$SEQ_DIR/sr"
if [[ -d "$SR_WEIGHTS_L" ]]; then
    run_cell "sr_l_seq"    "sr" "L" "$SEQ_DIR/sr"
fi
run_cell "sr_m_static" "sr" "M" "$STATIC_SR_DIR"
if [[ -d "$SR_WEIGHTS_L" ]]; then
    run_cell "sr_l_static" "sr" "L" "$STATIC_SR_DIR"
fi

run_cell "rr_f_seq"    "rr" "F" "$SEQ_DIR/rr"
run_cell "rr_f_static" "rr" "F" "$STATIC_RR_DIR"

# Native resolution cell (k_post<26>)
NATIVE_RR_DIR="$SCRATCH/native_rr"
mkdir -p "$NATIVE_RR_DIR"
cp -r "$STATIC_RR_DIR"/* "$NATIVE_RR_DIR"/
# Edit frame.json to native output resolution (1920x1080 -> 1920x1080)
sed -i 's/"output": \[3840, 2160\]/"output": \[1920, 1080\]/' "$NATIVE_RR_DIR/frame.json"
run_cell "rr_f_native" "rr" "F" "$NATIVE_RR_DIR"

# Non-zero gather formats cell
FORMATS_SR_DIR="$SCRATCH/formats_sr"
mkdir -p "$FORMATS_SR_DIR"
cp -r "$STATIC_SR_DIR"/* "$FORMATS_SR_DIR"/
for p in "$FORMATS_SR_DIR"/params_*.json; do
    sed -i 's/"exposure": 1.0/"exposure": 1.0, "fmt_col": 5, "fmt_dep": 16, "fmt_mv": 18/' "$p"
done
run_cell "sr_gather_formats" "sr" "M" "$FORMATS_SR_DIR"

# k_sr_gather_mv cell
MVHI_SR_DIR="$SCRATCH/mvhi_sr"
mkdir -p "$MVHI_SR_DIR"
cp -r "$STATIC_SR_DIR"/* "$MVHI_SR_DIR"/
for p in "$MVHI_SR_DIR"/params_*.json; do
    sed -i 's/"exposure": 1.0/"exposure": 1.0, "mvhi": 1/' "$p"
done
run_cell "sr_gather_mvhi" "sr" "M" "$MVHI_SR_DIR"

# Auto-exposure cell
AE_SR_DIR="$SCRATCH/ae_sr"
mkdir -p "$AE_SR_DIR"
cp -r "$STATIC_SR_DIR"/* "$AE_SR_DIR"/
for p in "$AE_SR_DIR"/params_*.json; do
    sed -i 's/"exposure": 1.0/"exposure": 1.0, "auto_exposure": 1/' "$p"
done
run_cell "sr_auto_exposure" "sr" "M" "$AE_SR_DIR"

# -----------------------------------------------------------------------------
# V5. Sensitivity Checks (Proof that Checker Can Fail)
# -----------------------------------------------------------------------------
echo
echo "[V5] Running Sensitivity Checks (must fail by >= 10 dB)..."

# Sensitivity 1: Corrupted weights (reverse one blob)
echo "  [V5-1] Testing port with corrupted (reversed) weights..."
CORRUPT_PORT_DIR="$SCRATCH/out_port_corrupt"
mkdir -p "$CORRUPT_PORT_DIR"
"$PORT_RUNNER" --mode sr --preset M --in "$STATIC_SR_DIR" --out "$CORRUPT_PORT_DIR" --corrupt-weights \
    > "$SCRATCH/corrupt.log" 2>&1 || true
tools/verify/eval_verify.py check-sensitivity "$SCRATCH/out_port/sr_m_static" "$SCRATCH/out_ref/sr_m_static" "$CORRUPT_PORT_DIR" --name "corrupted_weights" \
    || fail "V5: Corrupted weights run failed to degrade by >= 10 dB"

# Sensitivity 2: History ignored (every frame reset)
echo "  [V5-2] Testing port with history ignored (force reset)..."
NO_HIST_PORT_DIR="$SCRATCH/out_port_no_hist"
mkdir -p "$NO_HIST_PORT_DIR"
"$PORT_RUNNER" --mode sr --preset M --in "$STATIC_SR_DIR" --out "$NO_HIST_PORT_DIR" --force-reset \
    > "$SCRATCH/no_hist.log" 2>&1 || true
tools/verify/eval_verify.py check-sensitivity "$SCRATCH/out_port/sr_m_static" "$SCRATCH/out_ref/sr_m_static" "$NO_HIST_PORT_DIR" --name "no_history" \
    || fail "V5: No-history run failed to degrade by >= 10 dB"

# Sensitivity 3: Plain bicubic upscaling compared with oracle
echo "  [V5-3] Testing plain bicubic upscaling vs oracle..."
BICUBIC_DIR="$SCRATCH/out_bicubic"
mkdir -p "$BICUBIC_DIR"
build/venv/bin/python -c "
import numpy as np, json, glob, os
from PIL import Image

for f in sorted(glob.glob('$STATIC_SR_DIR/col_*.f16')):
    idx = f.split('_')[-1].split('.')[0]
    img = np.fromfile(f, dtype=np.float16).reshape((1080, 1920, 4)).astype(np.float32)
    # Resize with bicubic PIL
    p = Image.fromarray((np.clip(img[..., :3], 0, 1) * 255.0).astype(np.uint8))
    up = p.resize((3840, 2160), Image.BICUBIC)
    arr = (np.asarray(up, np.float32) / 255.0)
    out = np.concatenate([arr, np.ones((2160, 3840, 1), np.float32)], axis=2)
    out.tofile(os.path.join('$BICUBIC_DIR', f'out_{idx}.f32'))
"
tools/verify/eval_verify.py check-sensitivity "$SCRATCH/out_port/sr_m_static" "$SCRATCH/out_ref/sr_m_static" "$BICUBIC_DIR" --name "plain_bicubic" \
    || fail "V5: Bicubic upscaling failed to degrade by >= 10 dB vs oracle"

# -----------------------------------------------------------------------------
# V6. Scan Binaries & Package for AMDGPU ELF and Weight Hashes
# -----------------------------------------------------------------------------
echo
echo "[V6] Scanning build output and package for prohibited AMDGPU ELF objects..."
build/venv/bin/python -c "
import os, sys
prohibited = []
scan_root = '$ROOT/build'
for r, d, fs in os.walk(scan_root):
    if 'nh' in r or 'storage' in r: continue # skip raw reference folder
    for f in fs:
        p = os.path.join(r, f)
        if os.path.islink(p): continue
        try:
            with open(p, 'rb') as fp:
                hdr = fp.read(20)
                if hdr.startswith(b'\x7fELF') and len(hdr) >= 20:
                    em = hdr[18] | (hdr[19] << 8)
                    if em == 0xE0: # EM_AMDGPU
                        prohibited.append(p)
        except Exception:
            pass
if prohibited:
    print('Prohibited AMDGPU ELF code objects found:', prohibited)
    sys.exit(1)
" || fail "V6: AMDGPU ELF code objects (e_machine 0xE0) found in build artifacts"
echo "[V6] Binary scan clean (no AMDGPU code objects)."

# -----------------------------------------------------------------------------
# V9. Integration under Proton with ngx_client
# -----------------------------------------------------------------------------
echo
echo "[V9] Testing Integration under Proton with mingw ngx_client..."
CLIENT_EXE="$ROOT/build/verify/ngx_client.exe"
tools/verify/ngx_client/build.sh "$CLIENT_EXE" || fail "V9: Failed to build ngx_client.exe"

PROTON=$(find /home/deck/.local/share/Steam/compatibilitytools.d/ -name "proton" -type f 2>/dev/null | head -n 1 || true)
[[ -x "$PROTON" ]] || fail "V9: Steam Proton not found"
echo "  Using Proton: $PROTON"

SCRATCH_GAME="$SCRATCH/scratch_game"
SCRATCH_PREFIX="$SCRATCH/scratch_prefix"
mkdir -p "$SCRATCH_GAME" "$SCRATCH_PREFIX"
cp "$CLIENT_EXE" "$SCRATCH_GAME/ngx_client.exe"

# Record snapshot of game dir for byte-for-byte uninstall check
(cd "$SCRATCH_GAME" && find . -type f -exec sha256sum {} + | sort) > "$SCRATCH/game_before.sha"

# Test installer
INSTALLER="$ROOT/build/package/install.sh"
if [[ -f "$INSTALLER" ]]; then
    echo "  Installing package into scratch game dir..."
    bash "$INSTALLER" "$SCRATCH_GAME" optiscaler || fail "V9: Package install failed"

    # Run D3D12 under Proton
    echo "  Running ngx_client under Proton (D3D12)..."
    STEAM_COMPAT_DATA_PATH="$SCRATCH_PREFIX" \
    STEAM_COMPAT_CLIENT_INSTALL_PATH="/home/deck/.local/share/Steam" \
    "$PROTON" run "$SCRATCH_GAME/ngx_client.exe" --api d3d12 --feature sr --frames "$SEQ_DIR/sr" --out "$SCRATCH/out_proton_d3d12" --test-cycles \
        > "$SCRATCH/proton_d3d12.log" 2>&1 || { cat "$SCRATCH/proton_d3d12.log"; fail "V9: Proton D3D12 execution failed"; }

    # Run Vulkan under Proton
    echo "  Running ngx_client under Proton (Vulkan)..."
    STEAM_COMPAT_DATA_PATH="$SCRATCH_PREFIX" \
    STEAM_COMPAT_CLIENT_INSTALL_PATH="/home/deck/.local/share/Steam" \
    "$PROTON" run "$SCRATCH_GAME/ngx_client.exe" --api vulkan --feature sr --frames "$SEQ_DIR/sr" --out "$SCRATCH/out_proton_vk" --test-cycles \
        > "$SCRATCH/proton_vk.log" 2>&1 || { cat "$SCRATCH/proton_vk.log"; fail "V9: Proton Vulkan execution failed"; }

    # Check uninstall byte-for-byte restore
    echo "  Uninstalling and verifying byte-for-byte restore..."
    bash "$INSTALLER" "$SCRATCH_GAME" remove || fail "V9: Uninstall failed"
    (cd "$SCRATCH_GAME" && find . -type f -exec sha256sum {} + | sort) > "$SCRATCH/game_after.sha"
    diff -u "$SCRATCH/game_before.sha" "$SCRATCH/game_after.sha" || fail "V9: Game directory was not restored byte-for-byte after uninstall"
else
    echo "  Installer not present at $INSTALLER (will be checked when packaging milestone is reached)."
fi

echo
echo "================================================================================"
echo "VERIFY RESULT: PASS $RUN_ID"
echo "================================================================================"
exit 0
