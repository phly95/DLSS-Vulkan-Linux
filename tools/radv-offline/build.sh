#!/usr/bin/env bash
# Build the offline RDNA4 toolchain: Mesa (RADV + lavapipe + the amdgpu drm-shim) and pipestats.
#
#   bash tools/radv-offline/build.sh            -> toolchain/mesa/, tools/radv-offline/pipestats
#   tools/radv-offline/run.sh pipestats a.spv   (or any Vulkan program) on a shimmed RX 9070 XT
#
# The drm-shim fakes an amdgpu kernel driver for CHIP_GFX1201, so RADV compiles pipelines with ACO
# exactly as it would on the card; nothing executes. Needs meson, ninja, llvm-dev, a C/C++ compiler.
set -euo pipefail
cd -- "$(dirname -- "$0")/../.."
R=$PWD
MESA_COMMIT=e2ab6687d5ad700cc2b856af684aa014f9dae943     # Mesa main, 2026-10-24 (27.0.0-devel)
LIBDRM_COMMIT=b97cbde15c5c3abfe44d78e8f57139e50f612fec   # Mesa main needs libdrm >= 2.4.133
T=$R/toolchain/mesa
mkdir -p "$T"

# Prebuilt copy of exactly this build (Ubuntu 24.04 x86_64, LLVM 18 shared): unpack instead of building.
# The ICD files carry the install's absolute path, so they are rewritten to this checkout's.
PREBUILT=$R/artifacts/mesa/mesa-e2ab668-libdrm-b97cbde-ubuntu24.04-x86_64.tar.xz
if [[ ! -d "$T/install" && -f "$PREBUILT" && "${MESA_FROM_SOURCE:-0}" != 1 ]]; then
    mkdir -p "$T/install"
    tar -C "$T/install" -xJf "$PREBUILT"
    for j in "$T"/install/share/vulkan/icd.d/*.json; do
        sed -i -E "s#\"library_path\": \"[^\"]*/lib/#\"library_path\": \"$T/install/lib/#" "$j"
    done
fi

repo() {  # <dir> <url> <commit>
    [[ -d "$1" ]] && return
    git init -q "$1"
    git -C "$1" fetch -q --depth 1 "$2" "$3"
    git -C "$1" checkout -q FETCH_HEAD
}
if [[ ! -f "$T/install/share/vulkan/icd.d/radeon_icd.x86_64.json" ]]; then
    repo "$T/libdrm-src" https://gitlab.freedesktop.org/mesa/drm.git "$LIBDRM_COMMIT"
    repo "$T/mesa-src" https://gitlab.freedesktop.org/mesa/mesa.git "$MESA_COMMIT"
fi
if [[ ! -f "$T/install/share/vulkan/icd.d/radeon_icd.x86_64.json" && ! -f "$T/install/lib/x86_64-linux-gnu/pkgconfig/libdrm.pc" && ! -f "$T/install/lib/pkgconfig/libdrm.pc" ]]; then
    meson setup "$T/libdrm-build" "$T/libdrm-src" --prefix="$T/install" -Dintel=disabled -Dradeon=disabled \
        -Dnouveau=disabled -Dvmwgfx=disabled -Dtests=false -Dman-pages=disabled -Dvalgrind=disabled \
        -Dcairo-tests=disabled
    ninja -C "$T/libdrm-build" install
fi
export PKG_CONFIG_PATH=$(dirname "$(find "$T/install" -name libdrm.pc | head -1)")
if [[ ! -f "$T/install/share/vulkan/icd.d/radeon_icd.x86_64.json" ]]; then
    meson setup "$T/mesa-build" "$T/mesa-src" --prefix="$T/install" -Dbuildtype=release \
        -Dvulkan-drivers=amd,swrast -Dgallium-drivers=llvmpipe -Dplatforms= -Dglx=disabled -Degl=disabled \
        -Dgles1=disabled -Dgles2=disabled -Dopengl=false -Dllvm=enabled -Dshared-llvm=enabled \
        -Dvideo-codecs= -Dtools=drm-shim
    ninja -C "$T/mesa-build" install
fi
H=$R/reference-nr-vulkan/toolchain/Vulkan-Headers/include
[[ -d "$H" ]] || { echo "run reference-nr-vulkan/fetch_deps.sh first (Vulkan-Headers)" >&2; exit 1; }
g++ -std=c++17 -O1 -I"$H" tools/radv-offline/pipestats.cpp -o tools/radv-offline/pipestats -lvulkan
echo "ready: tools/radv-offline/run.sh tools/radv-offline/pipestats <spv>..."
