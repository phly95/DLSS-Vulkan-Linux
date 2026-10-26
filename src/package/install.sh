#!/usr/bin/env bash
# DLSS5-AMD-Vulkan installer
# Usage:
#   install.sh <game_folder> optiscaler
#   install.sh <game_folder> remove
set -euo pipefail

HERE=$(cd -- "$(dirname -- "$0")" && pwd)
ROOT=$(cd -- "$HERE/../.." && pwd)

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <game_folder> [optiscaler | remove]" >&2
    exit 1
fi

GAME=$(realpath "$1")
ROUTE=${2:-optiscaler}
MANIFEST="$GAME/dlss5-amd-install.txt"

record() {
    printf '%s\n' "$1" >> "$MANIFEST"
}

remove_installed() {
    if [[ ! -f "$MANIFEST" ]]; then
        echo "Nothing to remove (no $MANIFEST)"
        return 0
    fi
    while IFS= read -r entry; do
        [[ -n "$entry" ]] || continue
        case "$entry" in /*|..*|*/..*) continue;; esac
        rm -rf -- "$GAME/$entry"
    done < "$MANIFEST"
    rm -f -- "$MANIFEST"
    echo "Uninstalled from $GAME."
}

if [[ "$ROUTE" == "remove" ]]; then
    remove_installed
    exit 0
fi

if [[ -f "$MANIFEST" ]]; then
    remove_installed
fi

: > "$MANIFEST"
record "dlss5-amd-install.txt"

# Install launch env
cat << 'EOF' > "$GAME/dlss5-amd-launch.env"
WINEDLLOVERRIDES=_nvngx,nvngx=n,b
EOF
record "dlss5-amd-launch.env"

# The packed weights stay outside the game folder (r2scan.py scans the game folder, build/vk,
# build/shaders and build/package - build/ngx-weights is none of those), so the launch environment
# says where they are and _nvngx.dll opens them from there.
if [[ -d "$ROOT/build/re/packed/M" && -d "$ROOT/build/re/packed/L" && -d "$ROOT/build/rr/packed" ]]; then
    WROOT="$ROOT/build/ngx-weights"
    mkdir -p -- "$WROOT"
    ln -sfn -- "$ROOT/build/re/packed/M" "$WROOT/M"
    ln -sfn -- "$ROOT/build/re/packed/L" "$WROOT/L"
    ln -sfn -- "$ROOT/build/rr/packed" "$WROOT/rr"
    printf 'DLSS5_AMD_WEIGHTS=%s\n' "$WROOT" >> "$GAME/dlss5-amd-launch.env"
else
    echo "warning: no packed weights under $ROOT/build/{re,rr}: the package will refuse to create a feature" >&2
fi

# Install _nvngx.dll (OptiScaler route)
# Note: Do not install nvngx.dll, as ngx_client.exe loads _nvngx.dll first and
# r2scan.py flags any foreign nvngx*.dll.
cp -- "$HERE/_nvngx.dll" "$GAME/_nvngx.dll"
record "_nvngx.dll"

# Install shaders directory
mkdir -p -- "$GAME/dlss5-amd/shaders"
record "dlss5-amd/"
record "dlss5-amd/shaders/"

if [[ -d "$ROOT/build/shaders" ]]; then
    for s in "$ROOT/build/shaders"/*.spv; do
        [[ -f "$s" ]] || continue
        base=$(basename "$s")
        cp -- "$s" "$GAME/dlss5-amd/shaders/$base"
        record "dlss5-amd/shaders/$base"
    done
fi

echo "Installed DLSS5-AMD to $GAME."
