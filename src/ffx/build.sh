#!/usr/bin/env bash
# src/ffx/build.sh: dlss5-amd-ffx.dll - the FFX surface in front of the SAME bridge the NGX
# surface runs.  Links src/ffx/ffx_front.cpp + src/vk/pipeline.cpp + src/pe/nvngx.cpp (the
# shared-header refactor: ngx_bridge_shared.h externs the bridge internals, one interop dance,
# one record path).  Deliberately does NOT touch build/vk or build/shaders (the PSNR matrix's
# territory): shaders are loaded at runtime via DLSS5_AMD_SHADER_DIR / the dll-dir tail.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "$0")/../.." && pwd)
mkdir -p "$ROOT/build/package"
# the NGX TU needs the PE version resource when it ships as _nvngx.dll; the FFX dll's engine is
# the same code, so build the .res once from src/pe/nvngx_version.rc (same recipe as src/vk/build.sh).
RES=""
if command -v x86_64-w64-mingw32-windres >/dev/null; then
    mkdir -p "$ROOT/build/pe"
    x86_64-w64-mingw32-windres -O coff "$ROOT/src/pe/nvngx_version.rc" -o "$ROOT/build/pe/nvngx_version.res" \
        && RES="$ROOT/build/pe/nvngx_version.res"
fi
x86_64-w64-mingw32-g++ -std=c++17 -shared -O2 -DVK_NO_PROTOTYPES -DDLSS5_AMD_OUT_F16=1 \
    -I"$ROOT/reference-nr-vulkan/toolchain/Vulkan-Headers/include" \
    "$ROOT/src/vk/pipeline.cpp" "$ROOT/src/pe/nvngx.cpp" "$ROOT/src/ffx/ffx_front.cpp" $RES \
    -o "$ROOT/build/package/dlss5-amd-ffx.dll" \
    -ld3d12 -ldxgi -lole32 -static -static-libgcc -static-libstdc++
echo "built $ROOT/build/package/dlss5-amd-ffx.dll"
