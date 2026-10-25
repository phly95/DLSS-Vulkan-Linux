#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "$0")/../.." && pwd)
mkdir -p "$ROOT/build/vk" "$ROOT/build/shaders"

GLSLANG="$ROOT/reference-nr-vulkan/toolchain/glslang/bin/glslang"
REFINC="$ROOT/reference-nr-vulkan/linux/shaders/rdna4/include"

echo "Compiling shaders..."
python3 - "$GLSLANG" "$REFINC" "$ROOT/shaders/pipelines.json" "$ROOT/build/shaders" <<'EOF'
import json, os, subprocess, sys
glslang, refinc, pfile, outdir = sys.argv[1:5]
os.makedirs(outdir, exist_ok=True)
m = json.load(open(pfile))
for e in m["pipelines"]:
    n, src = e["name"], os.path.join("shaders", e["source"])
    defs = e.get("defines", [])
    cmd = [glslang, "-V", "--target-env", "vulkan1.3", "-Ishaders/include", "-I" + refinc] + \
          ["-D" + d for d in defs] + [src, "-o", os.path.join(outdir, n + ".spv")]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        print(f"FAILED {n}: {r.stderr}")
        sys.exit(1)
print(f"Compiled {len(m['pipelines'])} shaders to {outdir}")
EOF

RPATH='$ORIGIN/../../toolchain/mesa/install/lib/x86_64-linux-gnu'

mkdir -p "$ROOT/build/package"
if [[ -f "$ROOT/src/pe/nvngx.cpp" ]]; then
    if command -v x86_64-w64-mingw32-g++ >/dev/null; then
        x86_64-w64-mingw32-g++ -std=c++17 -shared -O2 \
            -I"$ROOT/reference-nr-vulkan/toolchain/Vulkan-Headers/include" \
            "$ROOT/src/pe/nvngx.cpp" -o "$ROOT/build/package/_nvngx.dll" \
            -ld3d12 -ldxgi -lole32 -static -static-libgcc -static-libstdc++
    fi
fi

if command -v g++ >/dev/null; then
    g++ -std=c++17 -O2 -I"$ROOT/reference-nr-vulkan/toolchain/Vulkan-Headers/include" \
        -Wl,-rpath,"$RPATH" \
        "$ROOT/src/vk/kernel_run.cpp" -o "$ROOT/build/vk/kernel_run" -lvulkan
    if [[ -f "$ROOT/src/vk/runner.cpp" ]]; then
        g++ -std=c++17 -O2 -I"$ROOT/reference-nr-vulkan/toolchain/Vulkan-Headers/include" \
            -Wl,-rpath,"$RPATH" \
            "$ROOT/src/vk/runner.cpp" -o "$ROOT/build/vk/vk_runner" -lvulkan
    fi
else
    podman run --rm --memory=8g -v "$ROOT":/w localhost/nrbuild:2 bash -c \
        "g++ -std=c++17 -O2 -I/w/reference-nr-vulkan/toolchain/Vulkan-Headers/include \
            -Wl,-rpath,'$RPATH' \
            /w/src/vk/kernel_run.cpp -o /w/build/vk/kernel_run -lvulkan && \
         if [ -f /w/src/vk/runner.cpp ]; then \
            g++ -std=c++17 -O2 -I/w/reference-nr-vulkan/toolchain/Vulkan-Headers/include \
                -Wl,-rpath,'$RPATH' \
                /w/src/vk/runner.cpp -o /w/build/vk/vk_runner -lvulkan; \
         fi"
fi
