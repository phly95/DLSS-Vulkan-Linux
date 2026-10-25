#!/usr/bin/env bash
# tools/verify/offline_check.sh: everything about the port that can be checked without the RX 9070 XT.
# It is not the referee (run_all.sh is, on the GPU machine) and never prints "VERIFY RESULT".
#
#   bash tools/verify/offline_check.sh
#
# Steps (any failure is reported, the rest still run, the last line is the verdict):
#   O1  setup: venv, reference deps (pinned glslang, Vulkan headers), the offline Mesa (RADV on a
#       shimmed GFX1201 + lavapipe), NeuralHub's extracted binaries, weights from the user's DLLs
#       (restored from the private storage repository into build/re per HANDOFF.md)
#   O2  the referee's own tools: self-tests of texfmt, eval_verify (V4/V5/V7/V8 logic; V7's parser also
#       on a real RADV_DEBUG=shaderstats log), diffdump; srref / rrref syntax (declaration-only HIP stub,
#       nothing linked or run), vkprobe, ngx_client (mingw), every shell script (bash -n)
#   O3  R2: tools/verify/r2scan.py over every tracked file
#   O4  R1 lint: no shader under shaders/ names a cooperative matrix or an FP8 type or extension, none
#       copies the reference's includes; shaders/include/nh_mm.glsl is the one wrapper, includes the
#       reference's coopmm.glsl and selects the f16 build by NH_F16
#   O5  every pipeline in shaders/pipelines.json built from its source twice with the pinned glslang:
#       the RADV build and the lavapipe build (-DNH_F16=1), nothing else differing; every .comp under
#       shaders/ is listed; the lavapipe build carries no Float8 capability
#   O6  RADV / ACO on the shimmed RX 9070 XT (tools/radv-offline): every pipeline 0 spilled SGPRs,
#       0 spilled VGPRs, 0 scratch
#   O7  lavapipe: every pipeline's f16 build run twice by build/vk/kernel_run on the inputs its test
#       maker writes (python3 shaders/tests/make_test.py NAME DIR), on llvmpipe only: both runs
#       bit-identical, every output finite and not constant
# Last line: "OFFLINE RESULT: PASS <pipelines> pipelines" or "OFFLINE RESULT: FAIL <failed steps>".
set -uo pipefail
ROOT=$(cd -- "$(dirname -- "$0")/../.." && pwd)
cd "$ROOT" || exit 1
PY=$ROOT/build/venv/bin/python
OUT=$ROOT/build/offline
GLSLANG=$ROOT/reference-nr-vulkan/toolchain/glslang/bin/glslang
REFINC=$ROOT/reference-nr-vulkan/linux/shaders/rdna4/include
LVP=$ROOT/toolchain/mesa/install/share/vulkan/icd.d/lvp_icd.x86_64.json
FAILED=()
fail() { echo "  FAIL: $*"; FAILED+=("$CUR"); }
step() { CUR=$1; echo; echo "==== $1: $2"; }

echo "offline_check: HEAD $(git rev-parse HEAD) ($(git log -1 --format=%s | cut -c1-70))"
echo "  tree: $([[ -z "$(git status --porcelain)" ]] && echo clean || echo DIRTY)"
echo "  freeze: $(git log --grep='^verify: freeze v2' -n 1 --format='%H %s' || true)"
rm -rf "$OUT"; mkdir -p "$OUT"

# ---------------------------------------------------------------------------------------------
step O1 "setup"
if [[ ! -x "$PY" ]]; then
    python3 -m venv build/venv && build/venv/bin/pip install -q numpy pillow pefile msgpack capstone \
        || fail "venv"
fi
[[ -f reference-nr-vulkan/linux/shaders/rdna4/include/coopmm.glsl ]] || git submodule update --init || fail "submodule"
[[ -x "$GLSLANG" && -f reference-nr-vulkan/toolchain/Vulkan-Headers/include/vulkan/vulkan.h ]] \
    || (cd reference-nr-vulkan && bash fetch_deps.sh) || fail "reference-nr-vulkan/fetch_deps.sh"
"$GLSLANG" --version | head -1
[[ -f toolchain/mesa/install/share/vulkan/icd.d/radeon_icd.x86_64.json && -x tools/radv-offline/pipestats ]] \
    || bash tools/radv-offline/build.sh || fail "tools/radv-offline/build.sh"
[[ -f build/nh/sr.gfx1201.co ]] \
    || "$PY" -I tools/exe/extract.py artifacts/neuralhub-v0.8.0/NeuralHub_v0.8.0_setup.exe build/nh >/dev/null || fail "extract.py"
if [[ ! -f build/re/nvngx_dlss_310.9.1.dll || ! -f build/re/nvngx_dlssd_310.9.1.dll ]]; then
    fail "build/re/nvngx_dlss{,d}_310.9.1.dll missing: restore them from phly95/dlss5-private-storage (its README)"
else
    [[ -d build/re/packed/M && -d build/re/packed/L ]] || {
        "$PY" -I tools/sr/sr_weights.py build/re/nvngx_dlss_310.9.1.dll --out build/re/weights >/dev/null &&
        "$PY" -I tools/sr/sr_pack.py build/re/weights M build/re/packed/M >/dev/null &&
        "$PY" -I tools/sr/sr_pack.py build/re/weights L build/re/packed/L >/dev/null; } || fail "SR weights"
    [[ -d build/rr/packed ]] || {
        "$PY" -I tools/rr/rr_weights.py build/re/nvngx_dlssd_310.9.1.dll --out build/rr/weights >/dev/null &&
        "$PY" -I tools/rr/rr_pack.py build/rr/weights build/rr/packed >/dev/null; } || fail "RR weights"
fi
echo "  weights: SR $(ls build/re/packed/M 2>/dev/null | wc -l) + $(ls build/re/packed/L 2>/dev/null | wc -l) files, RR $(ls build/rr/packed 2>/dev/null | wc -l) files"

# ---------------------------------------------------------------------------------------------
step O2 "the referee's tools"
"$PY" -I tools/verify/texfmt.py --self-test | tail -1 || fail "texfmt self-test"
"$PY" -I tools/verify/eval_verify.py --self-test | tail -16 || fail "eval_verify self-test"
"$PY" -I tools/verify/diffdump.py --self-test | grep -E "^  [a-z0-9]|SELF-TEST" || fail "diffdump self-test"
{ cat > "$OUT/t.comp" <<'EOF'
#version 460
layout(local_size_x = 64) in;
layout(std430, binding = 0) buffer B { float d[]; };
void main() { d[gl_GlobalInvocationID.x] *= 2.0; }
EOF
  "$GLSLANG" -V --target-env vulkan1.3 "$OUT/t.comp" -o "$OUT/t.spv" >/dev/null &&
  RADV_DEBUG=shaderstats MESA_SHADER_CACHE_DISABLE=true tools/radv-offline/run.sh tools/radv-offline/pipestats "$OUT/t.spv" \
      > "$OUT/shaderstats.log" 2>&1 &&
  { echo "VK_DEVICE: $(sed -n 's/^device: \(.*\) (radv)$/\1/p' "$OUT/shaderstats.log" | head -1)"; echo "PIPELINES_CREATED: 1"
    cat "$OUT/shaderstats.log"; } > "$OUT/shaderstats_v7.log" &&
  "$PY" -I -c "import sys; sys.path.insert(0, 'tools/verify'); import eval_verify as e
b = e.shader_stats(open('$OUT/shaderstats.log').read())
assert len(b) == 1 and all(k in b[0] for k in ('SGPRs', 'VGPRs', 'Spilled SGPRs', 'Spilled VGPRs', 'Scratch size')), b
print('  V7 parser on a real Mesa shaderstats log:', b[0])"; } || fail "V7 parser on real RADV output"
if "$PY" -I tools/verify/eval_verify.py check-radv "$OUT/shaderstats_v7.log" --device "AMD Radeon RX 9070 XT (RADV GFX1201)" \
        > "$OUT/v7_shim.txt" 2>&1; then
    fail "V7 accepted the drm-shim compile log for the RX 9070 XT"
else
    echo "  V7 rejects the drm-shim compile log: $(grep 'V7 FAIL' "$OUT/v7_shim.txt" | head -1 | sed 's/^ *//')"
fi
for o in tools/sr/srref/srref.cpp tools/rr/rrref/rrref.cpp; do
    g++ -std=c++17 -fsyntax-only -Wall -Werror -Itools/verify/hipstub "$o" && echo "  $o: syntax ok (HIP stub)" || fail "$o syntax"
done
g++ -std=c++17 -O1 -Wall -Werror -Ireference-nr-vulkan/toolchain/Vulkan-Headers/include tools/verify/vkprobe.cpp \
    -o "$OUT/vkprobe" -lvulkan && echo "  vkprobe: builds" || fail "vkprobe build"
if command -v x86_64-w64-mingw32-g++ >/dev/null; then
    bash tools/verify/ngx_client/build.sh "$OUT/ngx_client.exe" >/dev/null && echo "  ngx_client.exe: builds (mingw)" || fail "ngx_client build"
else
    fail "x86_64-w64-mingw32-g++ missing (apt-get install g++-mingw-w64-x86-64)"
fi
for s in tools/verify/*.sh tools/verify/ngx_client/*.sh tools/gpu/*.sh src/vk/*.sh; do
    [[ -f "$s" ]] || continue
    bash -n "$s" && echo "  $s: bash -n ok" || fail "bash -n $s"
done

# ---------------------------------------------------------------------------------------------
step O3 "R2 scan of the tracked files"
"$PY" -I tools/verify/r2scan.py --tracked | tail -12
[[ ${PIPESTATUS[0]} -eq 0 ]] || fail "r2scan"

# ---------------------------------------------------------------------------------------------
step O4 "R1 lint of shaders/"
if [[ ! -d shaders ]]; then
    fail "no shaders/ directory"
else
    hits=$(grep -rnE 'coopmat|floate4m3|fe4m3|bfloat|float_e4m3|float_e5m2|GL_KHR_cooperative_matrix|gl_CooperativeMatrix|coopMat(Load|Store|MulAdd)|saturatedConvert' shaders || true)
    [[ -z "$hits" ]] && echo "  no cooperative matrix / FP8 spelling in shaders/" || { echo "$hits"; fail "R1: coopmat / FP8 named outside the reference's coopmm.glsl"; }
    for f in coopmm.glsl coherent_act.glsl; do
        [[ -z "$(find shaders -name "$f")" ]] || fail "R1: shaders/ has its own $f"
    done
    "$PY" -I - <<'EOF' || fail "R1: shaders/ copies text of the reference's includes"
import glob, os, sys
ref = {p: open(p, encoding="utf-8", errors="replace").read() for p in glob.glob("reference-nr-vulkan/linux/shaders/rdna4/include/*.glsl")}
lines = {}
for p, t in ref.items():
    for l in t.splitlines():
        s = l.strip()
        if len(s) >= 60 and not s.startswith("//"):
            lines[s] = p
bad = 0
for p in glob.glob("shaders/**/*", recursive=True):
    if os.path.isfile(p):
        for l in open(p, encoding="utf-8", errors="replace"):
            if l.strip() in lines:
                print(f"  {p}: copies a line of {lines[l.strip()]}: {l.strip()[:80]}")
                bad += 1
print(f"  {bad} copied lines of the reference's includes")
sys.exit(1 if bad else 0)
EOF
    w=shaders/include/nh_mm.glsl
    if [[ -f "$w" ]] && grep -q '#include "coopmm.glsl"' "$w" && grep -q 'NH_F16' "$w"; then
        echo "  $w includes the reference's coopmm.glsl and switches on NH_F16"
    else
        fail "R1: $w missing, or it does not include coopmm.glsl / select on NH_F16"
    fi
    others=$(grep -rln 'coopmm.glsl\|NH_F16' shaders --include='*.comp' --include='*.glsl' | grep -v "^$w$" || true)
    [[ -z "$others" ]] && echo "  no other shader includes coopmm.glsl or tests NH_F16" || { echo "$others"; fail "R1: only $w may include coopmm.glsl or test NH_F16"; }
fi

# ---------------------------------------------------------------------------------------------
step O5 "build every pipeline twice"
NPIPE=0
if [[ ! -f shaders/pipelines.json ]]; then
    fail "no shaders/pipelines.json"
else
    "$PY" -I - "$GLSLANG" "$REFINC" "$OUT/spv" > "$OUT/build.log" 2>&1 <<'EOF'
import json, os, subprocess, sys
glslang, refinc, out = sys.argv[1:4]
m = json.load(open("shaders/pipelines.json"))
os.makedirs(out, exist_ok=True)
names, srcs, bad = set(), set(), 0
for e in m["pipelines"]:
    n, src, defs = e["name"], os.path.join("shaders", e["source"]), e.get("defines", [])
    if n in names or "/" in n:
        print(f"bad or duplicate pipeline name {n}"); bad += 1; continue
    names.add(n); srcs.add(os.path.normpath(src))
    if any(d.startswith("NH_F16") for d in defs):
        print(f"{n}: NH_F16 is offline_check's to set"); bad += 1; continue
    for tag, extra in (("", []), (".f16", ["NH_F16=1"])):
        cmd = [glslang, "-V", "--target-env", "vulkan1.3", "-Ishaders/include", "-I" + refinc] + \
              ["-D" + d for d in defs + extra] + [src, "-o", os.path.join(out, n + tag + ".spv")]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode:
            print(f"{n}{tag}: glslang failed\n{r.stdout}{r.stderr}"); bad += 1
        else:
            print(f"{n}{tag}: built from {src} {' '.join('-D' + d for d in defs + extra)}")
for dp, _, fs in os.walk("shaders"):
    for f in fs:
        if f.endswith(".comp") and os.path.normpath(os.path.join(dp, f)) not in srcs:
            print(f"{os.path.join(dp, f)}: not in shaders/pipelines.json"); bad += 1
print(f"PIPELINES {len(names)}")
sys.exit(1 if bad or not names else 0)
EOF
    rc=$?
    cat "$OUT/build.log"
    [[ $rc -eq 0 ]] || fail "shader build"
    NPIPE=$(grep -oP '^PIPELINES \K\d+' "$OUT/build.log" || echo 0)
    for f in "$OUT"/spv/*.f16.spv; do
        [[ -f "$f" ]] || continue
        if spirv-dis "$f" | grep -qE 'Float8EXT|Float8CooperativeMatrixEXT'; then fail "$(basename "$f"): the lavapipe build uses FP8"; fi
    done
fi

# ---------------------------------------------------------------------------------------------
step O6 "RADV / ACO on the shimmed RX 9070 XT"
fp8=()
for f in "$OUT"/spv/*.spv; do [[ -f "$f" && "$f" != *.f16.spv ]] && fp8+=("$f"); done
if (( ${#fp8[@]} == 0 )); then
    fail "no pipeline to compile"
else
    tools/radv-offline/run.sh tools/radv-offline/pipestats --isa "$OUT/isa" "${fp8[@]}" > "$OUT/pipestats.txt" 2> "$OUT/pipestats.err"
    rc=$?
    grep -v "amdgpu.ids\|not a conformant" "$OUT/pipestats.err" | head -5
    grep "^device:" "$OUT/pipestats.err" "$OUT/pipestats.txt" | head -1
    "$PY" -I - "$OUT/pipestats.txt" "${#fp8[@]}" <<'EOF' || fail "ACO statistics"
import re, sys
rows = [l.split() for l in open(sys.argv[1]) if " vgprs " in l]
bad = 0
print(f"  {'pipeline':28s} {'vgprs':>6s} {'sgprs':>6s} {'spill_v':>8s} {'spill_s':>8s} {'scratch':>8s} {'lds':>7s} {'waves':>6s} {'insts':>7s}")
for r in rows:
    kv = dict(zip(r[1::2], r[2::2]))
    g = lambda k: int(kv.get(k, -1))
    print(f"  {r[0]:28s} {g('vgprs'):6d} {g('sgprs'):6d} {g('spilled_vgprs'):8d} {g('spilled_sgprs'):8d} {g('scratch_size'):8d} "
          f"{g('lds_size'):7d} {g('subgroups_per_simd'):6d} {g('instructions'):7d}")
    if g("spilled_vgprs") != 0 or g("spilled_sgprs") != 0 or g("scratch_size") != 0:
        print(f"  {r[0]}: spills or scratch"); bad += 1
if len(rows) != int(sys.argv[2]):
    print(f"  {len(rows)} statistics lines for {sys.argv[2]} pipelines"); bad += 1
sys.exit(1 if bad else 0)
EOF
    [[ $rc -eq 0 ]] || fail "pipestats exit $rc"
fi

# ---------------------------------------------------------------------------------------------
step O7 "lavapipe runs of the f16 builds"
if (( NPIPE == 0 )); then
    fail "no pipeline to run"
elif [[ ! -f src/vk/build.sh ]]; then
    fail "no src/vk/build.sh (builds build/vk/kernel_run)"
else
    bash src/vk/build.sh > "$OUT/vkbuild.log" 2>&1 || { tail -20 "$OUT/vkbuild.log"; fail "src/vk/build.sh"; }
    for f in "$OUT"/spv/*.f16.spv; do
        [[ -f "$f" ]] || continue
        n=$(basename "$f" .f16.spv)
        t=$OUT/test/$n
        mkdir -p "$t"
        if ! "$PY" -I shaders/tests/make_test.py "$n" "$t" > "$t/make.log" 2>&1; then
            tail -5 "$t/make.log"; fail "$n: make_test.py"; continue
        fi
        ok=1
        for run in 1 2; do
            VK_ICD_FILENAMES=$LVP VK_DRIVER_FILES=$LVP build/vk/kernel_run --spv "$f" --launch "$t/launch.json" --out "$t/run$run" \
                > "$t/run$run.log" 2>&1 || { tail -5 "$t/run$run.log"; fail "$n: kernel_run (run $run)"; ok=0; break; }
        done
        (( ok )) || continue
        grep -q "^VK_DEVICE: llvmpipe" "$t/run1.log" || { fail "$n: not run on llvmpipe ($(grep '^VK_DEVICE' "$t/run1.log"))"; continue; }
        "$PY" -I - "$t" "$n" <<'EOF' || fail "$n: lavapipe outputs"
import hashlib, json, os, sys
import numpy as np
t, n = sys.argv[1:3]
L = json.load(open(os.path.join(t, "launch.json")))
outs = [b for b in L["bindings"] if b.get("output")]
bad = 0
if not outs:
    print(f"  {n}: the launch has no output binding"); bad += 1
for b in outs:
    name, typ = b["output"], b.get("type", "u8")
    a, c = (open(os.path.join(t, r, name + ".bin"), "rb").read() for r in ("run1", "run2"))
    v = np.frombuffer(a, {"u8": np.uint8, "f16": np.float16, "f32": np.float32, "u32": np.uint32}[typ]).astype(np.float64)
    fin = bool(np.all(np.isfinite(v)))
    const = bool(v.size == 0 or np.all(v == v.flat[0]))
    same = a == c
    print(f"  {n} {name} [{typ}]: {v.size} values, sha256 {hashlib.sha256(a).hexdigest()[:16]}, runs identical {same}, "
          f"finite {fin}, constant {const}, range [{np.nanmin(v) if v.size else 0:.4g}, {np.nanmax(v) if v.size else 0:.4g}]")
    bad += (not same) + (not fin) + const
sys.exit(1 if bad else 0)
EOF
    done
fi

echo
if (( ${#FAILED[@]} == 0 )); then
    echo "OFFLINE RESULT: PASS $NPIPE pipelines"
    exit 0
fi
echo "OFFLINE RESULT: FAIL $(printf '%s\n' "${FAILED[@]}" | sort -u | tr '\n' ' ')"
exit 1
