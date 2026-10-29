#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "$0")/../.." && pwd)
mkdir -p "$ROOT/build/vk" "$ROOT/build/shaders"
# M2: OUT_F16=1 (default) = final out as RGBA16F; 0 = the old f32 path. Propagated to every
# shader (-DOUT_F16) and every C++ TU (-DDLSS5_AMD_OUT_F16) - keep in step with
# tools/verify/ngx_client/build.sh (the harness side of the Output-format contract).
export OUT_F16=${OUT_F16:-1}

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
    defs = e.get("defines", []) + ["OUT_F16=" + os.environ.get("OUT_F16", "1")]
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
if [[ -f "$ROOT/src/package/install.sh" ]]; then
    cp "$ROOT/src/package/install.sh" "$ROOT/build/package/install.sh"
    chmod +x "$ROOT/build/package/install.sh"
fi
if [[ -f "$ROOT/src/pe/nvngx.cpp" ]]; then
    if command -v x86_64-w64-mingw32-g++ >/dev/null; then
        # The DLL links no Vulkan: with VK_NO_PROTOTYPES the frame chain calls the entry points that
        # NVSDK_NGX_VULKAN_Init / vkd3d-proton hand over (src/vk/vkfuncs.h), not libvulkan.
        x86_64-w64-mingw32-g++ -std=c++17 -shared -O2 -DVK_NO_PROTOTYPES -DDLSS5_AMD_OUT_F16=$OUT_F16 \
            -I"$ROOT/reference-nr-vulkan/toolchain/Vulkan-Headers/include" \
            "$ROOT/src/vk/pipeline.cpp" "$ROOT/src/pe/nvngx.cpp" -o "$ROOT/build/package/_nvngx.dll" \
            -ld3d12 -ldxgi -lole32 -static -static-libgcc -static-libstdc++
    fi
fi

if command -v g++ >/dev/null; then
    g++ -std=c++17 -O2 -I"$ROOT/reference-nr-vulkan/toolchain/Vulkan-Headers/include" \
        -Wl,-rpath,"$RPATH" \
        "$ROOT/src/vk/kernel_run.cpp" -o "$ROOT/build/vk/kernel_run" -lvulkan
    if [[ -f "$ROOT/src/vk/runner.cpp" ]]; then
        g++ -std=c++17 -O2 -I"$ROOT/reference-nr-vulkan/toolchain/Vulkan-Headers/include" \
            -Wl,-rpath,"$RPATH" -DDLSS5_AMD_OUT_F16=$OUT_F16 \
            "$ROOT/src/vk/pipeline.cpp" "$ROOT/src/vk/runner.cpp" -o "$ROOT/build/vk/vk_runner" -lvulkan
    fi
else
    podman run --rm --memory=8g -v "$ROOT":/w localhost/nrbuild:2 bash -c \
        "g++ -std=c++17 -O2 -I/w/reference-nr-vulkan/toolchain/Vulkan-Headers/include \
            -Wl,-rpath,'$RPATH' \
            /w/src/vk/kernel_run.cpp -o /w/build/vk/kernel_run -lvulkan && \
         if [ -f /w/src/vk/runner.cpp ]; then \
            g++ -std=c++17 -O2 -I/w/reference-nr-vulkan/toolchain/Vulkan-Headers/include \
                -Wl,-rpath,'$RPATH' \
                /w/src/vk/pipeline.cpp /w/src/vk/runner.cpp -o /w/build/vk/vk_runner -lvulkan; \
         fi"
fi
