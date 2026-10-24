#!/usr/bin/env bash
# Build `nr_plan` (tools/hwtest/nr_plan.cpp) against the reference sources.
#
# SteamOS has no compiler in its immutable root, so the build runs in the same podman container
# build_native.sh uses and the binary is written to the host. nr_native_plan.cpp is self-contained
# (no Vulkan), so this needs nothing but a C++17 compiler.
#
#   tools/hwtest/build_plan.sh [out dir] [reference checkout]
set -euo pipefail

out=${1:-$(cd -- "$(dirname -- "$0")/../.." && pwd)/build/hwtest}
here=$(cd -- "$(dirname -- "$0")" && pwd)
ref=${2:-$(cd -- "$here/../.." && pwd)/reference-nr-vulkan}
image=localhost/nrbuild:2
[[ -f "$ref/linux/src/core/nr_native_plan.cpp" ]] || { echo "no reference checkout at $ref" >&2; exit 1; }
command -v podman >/dev/null || { echo "podman is required" >&2; exit 1; }

mkdir -p -- "$out"; out=$(cd -- "$out" && pwd)
podman run --rm -v "$ref":/w:ro -v "$here":/t:ro -v "$out":/out "$image" bash -c '
set -e
cd /w
g++ -std=c++17 -O2 -Ilinux/src -Ilinux/src/core \
    /t/nr_plan.cpp linux/src/core/nr_native_plan.cpp -o /out/nr_plan'
echo "nr_plan: $out/nr_plan"
