#!/usr/bin/env bash
# Build Vulkan compute shaders and native vk_runner executable
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "$0")/../.." && pwd)
cd "$ROOT"

mkdir -p build/vk/spv

echo "Building Vulkan compute shaders..."
python3 shaders/build_shaders.py

echo "Building vk_runner in build container..."
podman run --rm -v "$ROOT":/w localhost/nrbuild:2 bash -c "
set -e
cd /w
g++ -std=c++17 -O2 \
    -Itoolchain/Vulkan-Headers/include \
    -Isrc/vk \
    src/vk/runner.cpp \
    -o build/vk/vk_runner \
    -lvulkan -lpthread -ldl
"

echo "vk_runner built: build/vk/vk_runner"
