#!/usr/bin/env bash
# tools/exe/fetch_neuralhub.sh [DEST_DIR=build/nh]: bring your own NeuralHub v0.8.0 inputs.
#
# NeuralHub's license forbids redistributing it, so this repository holds neither its installer nor
# anything extracted from it; every tool that needs NeuralHub's code objects reads build/nh, and this
# script is what fills it:
#
#   tools/exe/fetch_neuralhub.sh                                  # download, then unpack into build/nh
#   NH_INSTALLER=/path/to/NeuralHub_v0.8.0_setup.exe \
#       tools/exe/fetch_neuralhub.sh [DEST_DIR]                   # a copy already on disk
#
# Installer: the official release https://github.com/danielblnc/DLSS-NR-on-AMD/releases/tag/v0.8.0
# (asset NeuralHub_v0.8.0_setup.exe). Its SHA-256 is pinned below and checked before anything is
# unpacked; a mismatch prints expected and actual and stops (exit 1).
#
# What it writes, into DEST_DIR only (always git-ignored: a DEST the repository tracks is refused):
#   <net>.gfx1201.co, neuralhub_host.dll, manifest.json   tools/exe/extract.py (fatal if it fails)
#   kernels.json                                         tools/exe/kernels.py (best effort)
#   nr|sr|rr|fg.s.xz                                     llvm-objdump --mcpu=gfx1201 (best effort)
#   host_strings.txt.xz                                 strings of neuralhub_host.dll (best effort)
# Steps whose tool this machine has not got print the exact command to run later and continue: the
# script exits 0 once the installer was verified and unpacked. It is idempotent: a second run finds
# its own output and does nothing.
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "$0")/../.." && pwd)
cd "$ROOT"

DEST=${1:-build/nh}
SETUP_SHA256=aae53654001bf7345dc5b2e621a3f9018f4a5735760a1572a7532180249b0834
URL=https://github.com/danielblnc/DLSS-NR-on-AMD/releases/download/v0.8.0/NeuralHub_v0.8.0_setup.exe
ROCM_IMAGE=docker.io/rocm/dev-ubuntu-24.04:7.2.4
PY=$ROOT/build/venv/bin/python
NETS=(nr sr rr fg)

say() { echo "fetch_neuralhub: $*"; }
warn() { echo "fetch_neuralhub: WARNING: $*" >&2; }
die() { echo "fetch_neuralhub: ERROR: $*" >&2; exit 1; }

# --- DEST: canonical for the checks, repository-relative for the container that mounts ROOT at /w ---
# realpath -m resolves the build/ symlink of a shared scratch: what it points at is outside this
# checkout, and writing there is not writing a tracked path of this repository.
ABS=$(realpath -m -- "$DEST")
[[ $ABS != "$ROOT" ]] || die "DEST must be a directory below $ROOT, not the repository itself"
if [[ $ABS == "$ROOT"/* ]]; then
    git -C "$ROOT" check-ignore -q -- "${ABS#"$ROOT"/}" \
        || die "DEST=$ABS is inside the repository and not git-ignored: refusing to write tracked paths"
    DEST=${ABS#"$ROOT"/}
else
    DEST=$ABS
fi
mkdir -p "$DEST"

# --- the installer -------------------------------------------------------------------------------
if [[ -n ${NH_INSTALLER:-} ]]; then
    SETUP=$NH_INSTALLER
    [[ -f $SETUP ]] || die "NH_INSTALLER=$SETUP is not a file"
    say "using the installer you pointed at: $SETUP"
else
    SETUP=build/NeuralHub_v0.8.0_setup.exe   # build/ is git-ignored: a download never lands in git
    if [[ -f $SETUP ]]; then
        say "using the copy already in build/: $SETUP"
    else
        say "downloading NeuralHub_v0.8.0_setup.exe"
        say "  $URL"
        if ! curl -fL --retry 3 --connect-timeout 20 -o "$SETUP.part" "$URL"; then
            rm -f "$SETUP.part"
            die "download failed; run with NH_INSTALLER=/path/to/NeuralHub_v0.8.0_setup.exe instead"
        fi
        mv "$SETUP.part" "$SETUP"
    fi
fi

ACTUAL=$(sha256sum "$SETUP" | awk '{print $1}')
if [[ $ACTUAL != "$SETUP_SHA256" ]]; then
    echo "fetch_neuralhub: ERROR: SHA-256 mismatch, refusing to unpack $SETUP" >&2
    echo "  expected $SETUP_SHA256  (NeuralHub v0.8.0, $URL)" >&2
    echo "  actual   $ACTUAL" >&2
    exit 1
fi
say "verified sha256 $ACTUAL"

# --- unpack it -----------------------------------------------------------------------------------
REGEN=0   # 1 when this run unpacked: the reference inputs derived from it are stale until regenerated
if [[ -f $DEST/manifest.json && -f $DEST/sr.gfx1201.co ]] && grep -qF "$SETUP_SHA256" "$DEST/manifest.json"; then
    say "$DEST already holds this installer's extraction"
else
    [[ -x $PY ]] || die "no build/venv/bin/python (python3 -m venv build/venv && \
build/venv/bin/pip install pefile msgpack)"
    say "extracting into $DEST (tools/exe/extract.py)"
    "$PY" -I tools/exe/extract.py "$SETUP" "$DEST" || die "tools/exe/extract.py failed"
    REGEN=1
fi
for f in manifest.json neuralhub_host.dll sr.gfx1201.co rr.gfx1201.co nr.gfx1201.co fg.gfx1201.co; do
    [[ -f $DEST/$f ]] || die "$DEST/$f missing after extraction"
done

# --- an llvm-objdump that knows gfx1201 (LLVM 18+) -----------------------------------------------
# $LLVM_OBJDUMP, then PATH, then toolchain/, then the ROCm container this repository documents (that
# image keeps it at /opt/rocm/llvm/bin, off PATH -- probe both names).
# A container only sees what is mounted: mount the repository at /w, and -- since build/ is a symlink
# into another checkout here and DEST may live outside -- also mount whatever those paths cross into,
# at the same absolute path they have on the host (HANDOFF.md's container recipe).
VMOUNTS=("$ROOT:/w")
BUILD_REAL=$(realpath -m -- "$ROOT/build")
[[ $BUILD_REAL == "$ROOT/build" ]] || VMOUNTS+=("$BUILD_REAL:$BUILD_REAL")
DEST_REAL=$(realpath -m -- "$DEST")
case $DEST_REAL in
    "$ROOT"/*|"$BUILD_REAL"/*) : ;;   # already reachable through VMOUNTS
    /*) VMOUNTS+=("$DEST_REAL:$DEST_REAL") ;;
esac
PODMAN_OBJDUMP=llvm-objdump   # the name inside the image, probed below
podman_objdump() {            # podman_objdump <binary in the image> [llvm-objdump args...]
    local bin=$1; shift
    local a=() x
    for x in "${VMOUNTS[@]}"; do a+=(-v "$x"); done
    podman run --rm "${a[@]}" -w /w "$ROCM_IMAGE" "$bin" "$@"
}

OBJDUMP_KIND=""   # host | podman
OBJDUMP=""
for cand in ${LLVM_OBJDUMP:-} llvm-objdump toolchain/*/llvm-objdump toolchain/*/*/llvm-objdump; do
    [[ -n ${cand:-} ]] || continue
    if [[ $cand == */* ]]; then
        [[ -x $cand ]] || continue
    else
        cand=$(command -v "$cand" || true)
        [[ -n $cand ]] || continue
    fi
    if "$cand" -d --mcpu=gfx1201 --no-show-raw-insn "$DEST/sr.gfx1201.co" >/dev/null 2>&1; then
        OBJDUMP_KIND=host OBJDUMP=$cand
        say "objdump: $OBJDUMP"
        break
    fi
done
if [[ -z $OBJDUMP_KIND ]] && command -v podman >/dev/null 2>&1 \
        && podman image exists "$ROCM_IMAGE" 2>/dev/null; then
    for cb in llvm-objdump /opt/rocm/llvm/bin/llvm-objdump; do
        if podman_objdump "$cb" -d --mcpu=gfx1201 --no-show-raw-insn "$DEST/sr.gfx1201.co" >/dev/null 2>&1; then
            OBJDUMP_KIND=podman PODMAN_OBJDUMP=$cb
            say "objdump: $ROCM_IMAGE $cb (podman)"
            break
        fi
    done
fi
OBJDUMP_BIN=$OBJDUMP
if [[ $OBJDUMP_KIND == podman ]]; then    # kernels.py wants a command it can exec
    OBJDUMP_BIN=$DEST/llvm-objdump.sh
    { printf '#!/usr/bin/env bash\nexec podman run --rm'
      for x in "${VMOUNTS[@]}"; do printf ' -v %q' "$x"; done
      printf ' -w /w %q %q "$@"\n' "$ROCM_IMAGE" "$PODMAN_OBJDUMP"; } > "$OBJDUMP_BIN"
    chmod +x "$OBJDUMP_BIN"
fi

dis() {
    case $OBJDUMP_KIND in
        host)   "$OBJDUMP" -d --mcpu=gfx1201 "$@" ;;
        podman) podman_objdump "$PODMAN_OBJDUMP" -d --mcpu=gfx1201 "$@" ;;
        *)      return 127 ;;
    esac
}

missing_objdump() {
    warn "no llvm-objdump with --mcpu=gfx1201 here (LLVM 18+): $* skipped. Run this later, from the repository root:"
    echo "  build/venv/bin/python -I tools/exe/kernels.py $DEST/nr.gfx1201.co $DEST/sr.gfx1201.co \\"
    echo "      $DEST/rr.gfx1201.co $DEST/fg.gfx1201.co > $DEST/kernels.json"
    for n in "${NETS[@]}"; do
        echo "  llvm-objdump -d --mcpu=gfx1201 $DEST/$n.gfx1201.co | xz -c > $DEST/$n.s.xz"
    done
    local line="  # same thing in the ROCm container (mount both paths, see above):"
    for x in "${VMOUNTS[@]}"; do line+=" -v $x"; done
    echo "$line -w /w $ROCM_IMAGE $PODMAN_OBJDUMP -d --mcpu=gfx1201 $DEST/sr.gfx1201.co | xz -c > $DEST/sr.s.xz"
    echo "  # (the image's llvm-objdump lives in /opt/rocm/llvm/bin; a host one: LLVM_OBJDUMP=/path)"
}

# --- reference inputs, best effort ----------------------------------------------------------------
if [[ -z $OBJDUMP_KIND ]]; then
    missing_objdump "kernels.json and the *.s.xz disassembly"
elif [[ ! -f $DEST/kernels.json || $REGEN -eq 1 ]]; then
    say "kernels.json (tools/exe/kernels.py)"
    if LLVM_OBJDUMP=$OBJDUMP_BIN "$PY" -I tools/exe/kernels.py \
            "$DEST/nr.gfx1201.co" "$DEST/sr.gfx1201.co" "$DEST/rr.gfx1201.co" "$DEST/fg.gfx1201.co" \
            > "$DEST/kernels.json.tmp"; then
        mv "$DEST/kernels.json.tmp" "$DEST/kernels.json"
    else
        rm -f "$DEST/kernels.json.tmp"
        warn "kernels.py failed; run it yourself with LLVM_OBJDUMP=<llvm-objdump> (see its --help)"
    fi
fi

if [[ -n $OBJDUMP_KIND ]]; then
    for n in "${NETS[@]}"; do
        if [[ -f $DEST/$n.s.xz && $REGEN -eq 0 ]]; then continue; fi
        if dis "$DEST/$n.gfx1201.co" | xz -c > "$DEST/$n.s.xz.tmp"; then
            mv "$DEST/$n.s.xz.tmp" "$DEST/$n.s.xz"
            say "$n.s.xz"
        else
            rm -f "$DEST/$n.s.xz.tmp"
            warn "$n.s.xz not written (llvm-objdump failed)"
        fi
    done
fi

if [[ ! -f $DEST/host_strings.txt.xz || $REGEN -eq 1 ]]; then
    if command -v strings >/dev/null 2>&1 && command -v xz >/dev/null 2>&1 \
            && strings -a -n 5 "$DEST/neuralhub_host.dll" | xz -c > "$DEST/host_strings.txt.xz.tmp"; then
        mv "$DEST/host_strings.txt.xz.tmp" "$DEST/host_strings.txt.xz"
        say "host_strings.txt.xz"
    else
        rm -f "$DEST/host_strings.txt.xz.tmp"
        warn "host_strings.txt.xz not written; run later: strings -a -n 5 $DEST/neuralhub_host.dll | xz -c > $DEST/host_strings.txt.xz"
    fi
fi

# --- what is there now --------------------------------------------------------------------------
echo "fetch_neuralhub: $DEST"
for f in nr.gfx1201.co sr.gfx1201.co rr.gfx1201.co fg.gfx1201.co manifest.json neuralhub_host.dll \
         kernels.json host_strings.txt.xz nr.s.xz sr.s.xz rr.s.xz fg.s.xz; do
    if [[ -f $DEST/$f ]]; then
        printf '  ok      %-22s %s bytes\n' "$f" "$(stat -c%s "$DEST/$f")"
    else
        printf '  absent  %-22s %s\n' "$f" "run the command printed above"
    fi
done
