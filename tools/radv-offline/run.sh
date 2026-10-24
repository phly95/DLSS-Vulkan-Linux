#!/usr/bin/env bash
# Run a Vulkan program against RADV on a shimmed RX 9070 XT (GFX1201): pipelines compile, nothing runs.
#   tools/radv-offline/run.sh <program> [args...]
# MESA_PREFIX overrides the Mesa install (default toolchain/mesa/install from build.sh).
set -euo pipefail
R=$(cd -- "$(dirname -- "$0")/../.." && pwd)
P=${MESA_PREFIX:-$R/toolchain/mesa/install}
L=$(dirname "$(find "$P" -name libamdgpu_noop_drm_shim.so | head -1)")
export LD_LIBRARY_PATH=$L${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export LD_PRELOAD=$L/libamdgpu_noop_drm_shim.so
export AMDGPU_GPU_ID=gfx1201
export VK_ICD_FILENAMES=$P/share/vulkan/icd.d/radeon_icd.x86_64.json
exec "$@"
