#!/usr/bin/env bash
# Extract dlssnr.bin from an nvngx_dlssnr.dll 310.8.0 whose whole-file hash the reference's
# extract_model.sh does not know (e.g. the community-patched build with NVIDIA's architecture check
# lifted, SHA256 8270b350...44cc206). Only that one check is relaxed: every extracted weight entry is
# still verified byte for byte against the reference's model-files.sha256, so a DLL with different
# weights still fails.
#
#   tools/hwtest/extract_nr_model.sh <nvngx_dlssnr.dll> [out dir, default build/model]
set -euo pipefail
here=$(cd -- "$(dirname -- "$0")" && pwd)
root=$(cd -- "$here/../.." && pwd)
mt=$root/reference-nr-vulkan/linux/package/model-tools
dll=${1:?usage: extract_nr_model.sh <nvngx_dlssnr.dll> [out dir]}
out=${2:-$root/build/model}
mkdir -p -- "$out"
sha=$(sha256sum -- "$dll" | cut -d' ' -f1)
tmp=$(mktemp -d)
trap 'rm -rf -- "$tmp"' EXIT
sed -e "s#^here=.*#here=$mt#" -e "s/^sha_3108=.*/sha_3108=$sha/" "$mt/extract_model.sh" > "$tmp/extract.sh"
bash "$tmp/extract.sh" "$dll" "$out"
