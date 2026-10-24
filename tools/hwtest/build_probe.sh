#!/usr/bin/env bash
# Build one of tools/hwtest's single-file tools in the podman container used by build_native.sh and
# write the binary to the host, so it runs against the host's Mesa. SteamOS has no compiler of its
# own; see build_native.sh.
#
#   tools/hwtest/build_probe.sh [source.cpp] [out dir]
#
# The Vulkan headers come from the reference checkout's pinned Vulkan-Headers (the distribution's are
# too old for VK_EXT_shader_float8), the loader library from the container.
set -euo pipefail

here=$(cd -- "$(dirname -- "$0")" && pwd)
src=${1:-$here/hwprobe.cpp}
out=${2:-$(cd -- "$here/../.." && pwd)/build/hwtest}
image=localhost/nrbuild:2
ref=$(cd -- "$here/../.." && pwd)/reference-nr-vulkan
[[ -f "$src" ]] || { echo "no such source: $src" >&2; exit 1; }

"$here/build_native.sh" "$out" >/dev/null   # makes sure the image and out dir exist
name=$(basename -- "$src" .cpp)
podman run --rm -v "$here":/src:ro -v "$ref/toolchain/Vulkan-Headers":/vkh:ro -v "$out":/out "$image" bash -c \
    "g++ -std=c++17 -O2 -I/vkh/include /src/$(basename -- "$src") -o /out/$name -lvulkan"
echo "$name: $out/$name"
