#!/usr/bin/env bash
# Build the reference's native NR runner (nr_graph.cpp's own main) for this machine.
#
# SteamOS has no compiler in its immutable root, so the build runs in a podman container and the
# binary is written to the host, where it runs against the host's Mesa - the driver under test.
# Nothing NVIDIA is involved; the model still has to come from the user's nvngx_dlssnr.dll (or be
# synthesized with synth_model.py for a bring-up run).
#
#   tools/hwtest/build_native.sh [out dir] [reference checkout]
#
# Needs podman. The first run builds a small Ubuntu 24.04 image with g++/cmake/Vulkan headers.
set -euo pipefail

out=${1:-$(cd -- "$(dirname -- "$0")/../.." && pwd)/build/hwtest}
ref=${2:-$(cd -- "$(dirname -- "$0")/../.." && pwd)/reference-nr-vulkan}
image=localhost/nrbuild:2
[[ -d "$ref/linux/src/core" ]] || { echo "no reference checkout at $ref" >&2; exit 1; }
[[ -f "$ref/toolchain/Vulkan-Headers/include/vulkan/vulkan.h" ]] || {
    echo "missing $ref/toolchain/Vulkan-Headers; run reference-nr-vulkan/fetch_deps.sh there" >&2; exit 1; }
command -v podman >/dev/null || { echo "podman is required" >&2; exit 1; }

if ! podman image exists "$image"; then
    echo "building $image (once)"
    podman run --name nrbuild-tmp docker.io/library/ubuntu:24.04 bash -c \
        'apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
         build-essential cmake ninja-build pkg-config git ca-certificates python3 libvulkan-dev'
    podman commit nrbuild-tmp "$image"
    podman rm nrbuild-tmp
fi

mkdir -p -- "$out"; out=$(cd -- "$out" && pwd)
podman run --rm -v "$ref":/w:ro -v "$out":/out "$image" bash -c '
set -e
cd /w
source linux/build/arch/rdna4.sh
g++ -std=c++17 -O2 -Itoolchain/Vulkan-Headers/include -Ilinux/src -Ilinux/src/core \
    "${NR_PRODUCT_DEFINES[@]}" linux/src/core/nr_graph.cpp -o /out/nr_graph -lvulkan -lpthread -ldl'
echo "nr_graph: $out/nr_graph"
