#!/usr/bin/env bash
# Build ngx_client.exe (Windows x86-64, mingw-w64) for run_all.sh V9.
#   tools/verify/ngx_client/build.sh [out.exe]
# Uses x86_64-w64-mingw32-g++ when the host has it (the cloud container), else a podman image with
# mingw-w64 built once (SteamOS has no compiler). Vulkan headers: reference-nr-vulkan's pinned copy
# (reference-nr-vulkan/fetch_deps.sh).
set -euo pipefail
here=$(cd -- "$(dirname -- "$0")" && pwd)
root=$(cd -- "$here/../../.." && pwd)
out=${1:-$root/build/verify/ngx_client.exe}
mkdir -p "$(dirname -- "$out")"
out=$(cd -- "$(dirname -- "$out")" && pwd)/$(basename -- "$out")
vkh=reference-nr-vulkan/toolchain/Vulkan-Headers/include
[[ -f "$root/$vkh/vulkan/vulkan.h" ]] || { echo "missing $vkh: run reference-nr-vulkan/fetch_deps.sh" >&2; exit 1; }
# M2: FMT16=1 (default) = app Output RGBA16F, matching OUT_F16=1 in src/vk/build.sh.
# (Named FMT16, not OUT_F16: this script substitutes the literal token OUT with the output path.)
cmd=(x86_64-w64-mingw32-g++ -std=c++17 -O2 -Wall -DFMT16=${FMT16:-1} -I"$vkh" tools/verify/ngx_client/main.cpp
     -o OUT -ld3d12 -ldxgi -lole32 -static -static-libgcc -static-libstdc++)
if command -v x86_64-w64-mingw32-g++ >/dev/null; then
    (cd "$root" && "${cmd[@]/OUT/$out}")
else
    image=localhost/nrmingw:1
    if ! podman image exists "$image"; then
        podman run --name nrmingw-tmp docker.io/library/ubuntu:24.04 bash -c \
            'apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq g++-mingw-w64-x86-64'
        podman commit nrmingw-tmp "$image"
        podman rm nrmingw-tmp
    fi
    podman run --rm -v "$root":/w:ro -v "$(dirname -- "$out")":/out "$image" bash -c \
        "cd /w && ${cmd[*]/OUT//out/$(basename -- "$out")}"
fi
echo "built $out"
