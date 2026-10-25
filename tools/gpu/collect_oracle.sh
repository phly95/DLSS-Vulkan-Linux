#!/usr/bin/env bash
# tools/gpu/collect_oracle.sh: on the RX 9070 XT, run NeuralHub's own SR (presets M, L) and RR (preset F)
# kernels under ROCm (srref, rrref) on one fixed, seeded, small sequence with per-launch dumps, so every
# Vulkan kernel can be checked alone, offline, against what NeuralHub computed (tools/verify/diffdump.py).
#
#   bash tools/gpu/collect_oracle.sh [--dry-run]
#
# Writes build/oracle/<id>/ (id = oracle-<HEAD short hash>-s<seed>):
#   frames/{sr,rr}/          the inputs (tools/sr/gen_frames.py, tools/rr/gen_rr_frames.py, seed 20261025)
#   sr_M/, sr_L/, rr_F/      out_<i>.f32, oracle.log (GPU_FRAME_MS lines), dump/ (frames 0 and 1:
#                            start.*, NNN.*, launches.jsonl, buffers.json)
#   README.md                what this is, how it was made, the SHA-256 of every file
# and build/oracle/<id>.tar.xz.partNN (< 90 MB each) for the private storage repository. It prints the
# commands that add them there. Nothing it writes may go into this repository (R2).
#
# --dry-run does everything but the two ROCm runs (no GPU needed): sequences, weights, README layout.
# Needs: podman with docker.io/rocm/dev-ubuntu-24.04:7.2.4, /dev/kfd, build/re/nvngx_dlss{,d}_310.9.1.dll,
# build/venv (HANDOFF.md).
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "$0")/../.." && pwd)
cd "$ROOT"
DRY=0
[[ "${1:-}" == --dry-run ]] && DRY=1
PY=$ROOT/build/venv/bin/python
SEED=20261025
ID=oracle-$(git rev-parse --short=10 HEAD)-s$SEED
O=$ROOT/build/oracle/$ID
ROCM_IMAGE=docker.io/rocm/dev-ubuntu-24.04:7.2.4
PIC=reference-nr-vulkan/docs/ngx-verification/310.8.0/single-frame-inputs/3840x2160.png
# small enough for a git repository, large enough for every level (W0, H0 multiples of 32, level 3 = /8)
RW=256 RH=144 OW=512 OH=288 FRAMES=9 DUMP_FRAMES=0,1

[[ -n "$(git status --porcelain)" ]] && { echo "the tree is not clean: the id names HEAD" >&2; exit 1; }
for f in build/re/nvngx_dlss_310.9.1.dll build/re/nvngx_dlssd_310.9.1.dll "$PIC" "$PY"; do
    [[ -e "$f" ]] || { echo "missing $f (HANDOFF.md; tools/verify/README.md, restoring the inputs)" >&2; exit 1; }
done
[[ -e "$O" ]] && { echo "$O exists" >&2; exit 1; }
mkdir -p "$O"
echo "collecting $O"

W=$O/weights.tmp
"$PY" -I tools/sr/sr_weights.py build/re/nvngx_dlss_310.9.1.dll --out "$W/sr_raw" > "$O/weights.log" 2>&1
"$PY" -I tools/sr/sr_pack.py "$W/sr_raw" M "$W/sr_M" >> "$O/weights.log" 2>&1
"$PY" -I tools/sr/sr_pack.py "$W/sr_raw" L "$W/sr_L" >> "$O/weights.log" 2>&1
"$PY" -I tools/rr/rr_weights.py build/re/nvngx_dlssd_310.9.1.dll --dither "$W/sr_raw/dither.bin" --out "$W/rr_raw" >> "$O/weights.log" 2>&1
"$PY" -I tools/rr/rr_pack.py "$W/rr_raw" "$W/rr_F" >> "$O/weights.log" 2>&1
grep -q "preset L: .*NeuralHub checksum matches" "$O/weights.log" || { echo "SR L weights: checksum differs" >&2; exit 1; }
grep -q "fingerprint .*: matches NeuralHub" "$O/weights.log" || { echo "RR weights: fingerprint differs" >&2; exit 1; }

"$PY" tools/sr/gen_frames.py "$PIC" "$O/frames/sr" --render $RW $RH --output $OW $OH --frames $FRAMES --seed $SEED --path \
    --gt "$O/frames/sr_gt" > /dev/null
"$PY" tools/rr/gen_rr_frames.py "$PIC" "$O/frames/rr" --render $RW $RH --output $OW $OH --frames $FRAMES --seed $SEED --path \
    --gt "$O/frames/rr_gt" > /dev/null

c() { echo "/w${1#$ROOT}"; }
SCRIPT="cd /w && set -e
hipcc -O2 -std=c++17 tools/sr/srref/srref.cpp -o $(c "$O/bin")/srref
hipcc -O2 -std=c++17 tools/rr/rrref/rrref.cpp -o $(c "$O/bin")/rrref
for p in M L; do
    mkdir -p $(c "$O")/sr_\$p
    $(c "$O/bin")/srref artifacts/neuralhub-v0.8.0/gfx1201/sr.gfx1201.co $(c "$W")/sr_\$p $(c "$O/frames/sr") \\
        $(c "$O")/sr_\$p \$( [ \$p = L ] && echo --L ) --dump $(c "$O")/sr_\$p/dump --dump-frames $DUMP_FRAMES \\
        > $(c "$O")/sr_\$p/oracle.log 2>&1
done
mkdir -p $(c "$O")/rr_F
$(c "$O/bin")/rrref artifacts/neuralhub-v0.8.0/gfx1201/rr.gfx1201.co $(c "$W")/rr_F $(c "$O/frames/rr") $(c "$O")/rr_F \\
    --dump $(c "$O")/rr_F/dump --dump-frames $DUMP_FRAMES > $(c "$O")/rr_F/oracle.log 2>&1
rm -rf $(c "$O/bin")"
if (( DRY )); then
    printf 'DRY RUN: would run in %s:\n%s\n' "$ROCM_IMAGE" "$SCRIPT" | tee "$O/dry_run_script.txt"
    bash -n -c "$SCRIPT" && echo "(container script: bash -n ok)"
else
    mkdir -p "$O/bin"
    podman run --rm --device /dev/kfd --device /dev/dri --group-add keep-groups --security-opt label=disable \
        -v "$ROOT":/w "$ROCM_IMAGE" bash -c "$SCRIPT"
    for d in sr_M sr_L rr_F; do
        [[ $(grep -c '^GPU_FRAME_MS' "$O/$d/oracle.log") -eq $FRAMES ]] || { tail "$O/$d/oracle.log"; echo "$d: incomplete" >&2; exit 1; }
        for f in ${DUMP_FRAMES//,/ }; do
            [[ -s "$O/$d/dump/frame$f/launches.jsonl" ]] || { echo "$d: no dump of frame $f" >&2; exit 1; }
        done
    done
fi
rm -rf "$W"

{
    echo "# $ID"
    echo
    echo "NeuralHub v0.8.0's own SR (presets M, L) and RR (preset F) kernels, run under ROCm 7.2.4 on the RX 9070 XT"
    echo "by \`tools/gpu/collect_oracle.sh\` at DLSS5-AMD-Linux $(git rev-parse HEAD), $(date -u +%Y-%m-%dT%H:%MZ)."
    echo "Weights from the user's nvngx_dlss.dll / nvngx_dlssd.dll 310.9.1.0. NVIDIA- and NeuralHub-derived:"
    echo "private storage only, never the main repository."
    echo
    echo "- inputs: \`frames/sr\`, \`frames/rr\` ($RW x $RH -> $OW x $OH, $FRAMES frames, seed $SEED, seeded pan and"
    echo "  reset frame $(grep -l '"reset": true' "$O"/frames/sr/params_*.json | sed 's/.*params_\([0-9]*\).json/\1/' | sort -n | tr '\n' ' ')),"
    echo "  ground truth \`frames/{sr,rr}_gt\`"
    echo "- \`sr_M\`, \`sr_L\`, \`rr_F\`: outputs \`out_<i>.f32\`, \`oracle.log\`, per-launch dumps of frames $DUMP_FRAMES in"
    echo "  \`dump/\` (\`tools/verify/oracle_dump.hpp\`; read with \`tools/verify/diffdump.py\`)"
    (( DRY )) && echo "- DRY RUN: the ROCm runs were skipped; this directory is not an oracle"
    echo
    echo "Restore: \`cat $ID.tar.xz.part* | tar xJ -C <DLSS5-AMD-Linux>/build/oracle\`"
    echo
    echo "## SHA-256"
    echo
    echo '```'
    (cd "$O" && find . -type f ! -name README.md -print0 | sort -z | xargs -0 sha256sum)
    echo '```'
} > "$O/README.md"

tar -C "$ROOT/build/oracle" -cJf - "$ID" | split -b 90M -d - "$ROOT/build/oracle/$ID.tar.xz.part"
ls -la "$ROOT/build/oracle/$ID".tar.xz.part*
(cd "$ROOT/build/oracle" && sha256sum "$ID".tar.xz.part*) > "$ROOT/build/oracle/$ID.parts.sha256"
cat <<EOF

Done: $O ($(du -sh "$O" | cut -f1))
To store it (the private repository, phly95/dlss5-private-storage):
  S=/path/to/dlss5-private-storage
  mkdir -p \$S/oracle && cp $ROOT/build/oracle/$ID.tar.xz.part* $ROOT/build/oracle/$ID.parts.sha256 \$S/oracle/
  cp $O/README.md \$S/oracle/$ID.README.md
  cd \$S && git add oracle && git commit -m "Oracle dumps $ID" && git push
EOF
