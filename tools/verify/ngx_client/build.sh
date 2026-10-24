#!/usr/bin/env bash
# Build ngx_client.exe using podman and localhost/nrbuild:2
set -euo pipefail
here=$(cd -- "$(dirname -- "$0")" && pwd)
root=$(cd -- "$here/../../.." && pwd)
out=${1:-$root/build/verify/ngx_client.exe}
rel_out=$(realpath --relative-to="$root" "$out")
mkdir -p "$(dirname -- "$out")"

podman run --rm -v "$root":/w localhost/nrbuild:2 bash -c "
set -e
cd /w
x86_64-w64-mingw32-g++ -std=c++17 -O2 \
    -Itoolchain/Vulkan-Headers/include \
    -Itools/verify/ngx_client \
    tools/verify/ngx_client/main.cpp \
    -o '$rel_out' \
    -ld3d12 -ldxgi -lole32 -static -static-libgcc -static-libstdc++
"

echo "Built: $out"
