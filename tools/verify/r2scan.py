#!/usr/bin/env python3
"""Scan files for NVIDIA- or NeuralHub-derived content (R2; run_all.sh V6; offline_check.sh).

    r2scan.py --tracked                 every file git tracks in this repository
    r2scan.py --dir D [--dir D ...]     every file under D (the port's build output, the package, an
                                        installed game folder), following no symlinks
    --secrets S [--secrets S ...]       files or directories whose content must not appear (default:
                                        the user's DLLs, extracted and packed weights, NeuralHub's
                                        extracted binaries, oracle dumps, wherever they exist in build/)
    --allow-file PATH                   (with --dir) skip exactly this file: the user's own DLL a game
                                        folder legitimately holds

A scanned file fails when
  * its SHA-256 equals a secret file's, or it contains a 64-byte window of a secret file (a window
    every 512 bytes of every secret file, every 64 KiB of a DLL, where the bytes are not repetitive),
    or the hex spelling of such a window;
  * it is text and contains the hex SHA-256 of a secret file (no committed hashes of weights), except
    the PREEXISTING integrity checks below, each pinned to the one file that held it before R2;
  * it is an AMDGPU ELF (e_machine 0xE0), or is named nvngx*.dll / *.co;
  * with --tracked: it lies under artifacts/ and is not one of the pinned files below with exactly
    this SHA-256 (the intentional binaries the repository already holds; changing them is the user's
    call), or under build/ or toolchain/.
Last line: "R2SCAN: PASS <files>" or "R2SCAN: FAIL <n> findings".
"""
import argparse
import hashlib
import struct
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ARTIFACTS = {
    "artifacts/isa/amdgpu_isa_rdna4.xml.xz": "1e61ec15026d239d4e9571b1df97e443995a758a4a3a877d8217f566b1a94b4d",
    "artifacts/isa/radv_nir_lower_cooperative_matrix.c": "dae99f350528dc9f15d3e3dedc8540bccf71ad67b05601b278acdaa18cea82cf",
    "artifacts/mesa/mesa-e2ab668-libdrm-b97cbde-ubuntu24.04-x86_64.tar.xz": "c3087150c6d9e93aca33605c4d9e062075fabcffde59bb8f091cd05125801419",
    "artifacts/neuralhub-v0.8.0/NeuralHub_v0.8.0_setup.exe": "aae53654001bf7345dc5b2e621a3f9018f4a5735760a1572a7532180249b0834",
    "artifacts/neuralhub-v0.8.0/gfx1201/fg.gfx1201.co": "f77362c3b821c71a754f2a6b2f668bd94b8180fdd10437ea79da57c088a570eb",
    "artifacts/neuralhub-v0.8.0/gfx1201/fg.s.xz": "ca9de44017f8f5ba2c45cf13e80b61dc7827d6651684248da38ca5c2af5e11fb",
    "artifacts/neuralhub-v0.8.0/gfx1201/kernels.json": "e5d3495150eaf79b4c5f60b6040386ac48543c84c49d7bb1d449dd942f45dbf0",
    "artifacts/neuralhub-v0.8.0/gfx1201/manifest.json": "813788b2156e5c0a4580766213bfe86b43969efac756f348a8725f5fd9cae972",
    "artifacts/neuralhub-v0.8.0/gfx1201/nr.gfx1201.co": "0566a2d1cf71e29d3921aa075afac176e36dbf0364d88aa7de025cf9d2637340",
    "artifacts/neuralhub-v0.8.0/gfx1201/nr.s.xz": "9bd629a2d79529ea43045334f1766a9bc4ae668876dafe33028d498dba3b848e",
    "artifacts/neuralhub-v0.8.0/gfx1201/rr.gfx1201.co": "8fc1a110f5dbe4e938fb0010f6608ae9fb1b59aa511becfcd08c1794d0035dfd",
    "artifacts/neuralhub-v0.8.0/gfx1201/rr.s.xz": "421ecf6b2422e7d5eee691a032771fe362067758aac9f63f2dec7ee172deb914",
    "artifacts/neuralhub-v0.8.0/gfx1201/sr.gfx1201.co": "ad772d3f563130edd159a332421f5cb455437fe10386bac77208a6072f2df558",
    "artifacts/neuralhub-v0.8.0/gfx1201/sr.s.xz": "76d631a2bea4ff0a5d3a13d70aaaaa22cbcaab2db8391ffb2085a680a63fa726",
    "artifacts/neuralhub-v0.8.0/host_strings.txt.xz": "a979514774caf35be50e37ce341a05753d19d65fcea3550e2a12f8c80c30fd50",
    "artifacts/reference-nr/aco-isa-gfx1201.tar.xz": "34b8b3e30150bc838638041196a55d48ba3ebbdbda8cfa5f02322ac99641478d",
    "artifacts/reference-nr/pipestats-gfx1201.txt": "f6221e7e76cd6397fd548fa64c7c6933488ba33a0895eb73400b93deb2e69a32",
}
# Hashes of the user's inputs that the extractors already compared against when R2 was set (they check
# that the DLL handed to them is 310.9.1.0); allowed in exactly these files, nowhere else.
PREEXISTING = {
    ("tools/sr/sr_weights.py", "4707682a85d793cafcfe746e70d181190ba59bd989b88e27dee9edf1f20305b8"),
    ("tools/rr/rr_weights.py", "4707682a85d793cafcfe746e70d181190ba59bd989b88e27dee9edf1f20305b8"),
    ("tools/rr/rr_weights.py", "4bc7ea5fcb2f32cf86bc2cb072e8d2860914a0be511cbdcb2fc206c1d9d83b80"),
    ("tools/rr/rr_weights.py", "8f4c75775480450c05aae2e0a37cd64582834babba85346d99a0bbcd2ef88c7e"),
}
DEFAULT_SECRETS = ["build/re", "build/dlls", "build/rr/weights", "build/rr/packed", "build/nh", "build/oracle",
                   "build/verify/weights"]


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def files_under(d):
    for r, ds, fs in os.walk(d, followlinks=False):
        ds.sort()
        for f in sorted(fs):
            p = os.path.join(r, f)
            if os.path.isfile(p) and not os.path.islink(p):
                yield p


def windows(data, step, w=64):
    """a 64-byte window every `step` bytes, where its bytes are binary data (>= 16 distinct values, at
    most 40 printable ASCII: names and messages also present in a program's own source are not
    evidence of a copy)"""
    out = []
    for o in range(0, len(data) - w + 1, step):
        win = data[o:o + w]
        if len(set(win)) >= 16 and sum(32 <= c < 127 for c in win) <= 40:
            out.append(win)
    return out


def find_probes(data, index):
    """the first probe (window, source) found in data: candidate offsets by the probe's first 8 bytes
    (every byte alignment), then a full compare"""
    if len(data) < 64 or not index:
        return None
    keys = np.fromiter(index.keys(), np.uint64, len(index))
    buf = np.frombuffer(data, np.uint8)
    for k in range(8):
        n8 = (len(buf) - k) // 8
        if n8 <= 0:
            continue
        v = buf[k:k + 8 * n8].view(np.uint64)
        for j in np.flatnonzero(np.isin(v, keys)):
            off = k + 8 * int(j)
            for w, src in index[int(v[j])]:
                if data[off:off + 64] == w:
                    return src
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tracked", action="store_true")
    ap.add_argument("--dir", action="append", default=[])
    ap.add_argument("--secrets", action="append", default=None)
    ap.add_argument("--allow-file", action="append", default=[])
    a = ap.parse_args()
    if not a.tracked and not a.dir:
        ap.error("--tracked or --dir")
    findings = []
    secrets = a.secrets if a.secrets is not None else [os.path.join(ROOT, s) for s in DEFAULT_SECRETS]
    sfiles = []
    for s in secrets:
        if os.path.isfile(s):
            sfiles.append(s)
        elif os.path.isdir(s):
            sfiles.extend(files_under(s))
    shas, index, hexes, nprobes = {}, {}, [], 0
    for p in sfiles:
        if os.path.getsize(p) < 128:
            continue
        shas[sha(p)] = p
        data = open(p, "rb").read()
        # dense windows in extracted weights and dumps; sparse ones in whole programs (DLLs, NeuralHub's
        # extracted binaries), whose copies would also be caught by name, hash or ELF header
        whole = p.lower().endswith((".dll", ".exe", ".co")) or "/build/nh/" in p
        for w in windows(data, 65536 if whole else 512):
            index.setdefault(struct.unpack("<Q", w[:8])[0], []).append((w, p))
            hexes.append((w[:32].hex().encode(), p))
            nprobes += 1
    print(f"r2scan: {len(sfiles)} secret files, {len(shas)} hashes, {nprobes} probes")

    if a.tracked:
        out = subprocess.run(["git", "-C", ROOT, "ls-files", "-z"], check=True, capture_output=True).stdout
        targets = [os.path.join(ROOT, p) for p in out.decode().split("\0") if p]
    else:
        targets = [p for d in a.dir for p in files_under(d)]
    allow = {os.path.realpath(p) for p in a.allow_file}
    seen_art = set()
    n = 0
    for p in targets:
        if os.path.realpath(p) in allow:
            continue
        rel = os.path.relpath(p, ROOT)
        if not os.path.isfile(p):
            continue
        n += 1
        name = os.path.basename(p).lower()
        if a.tracked and rel.split("/")[0] in ("build", "toolchain"):
            findings.append(f"{rel}: tracked under {rel.split('/')[0]}/")
        if a.tracked and rel.startswith("artifacts/"):
            seen_art.add(rel)
            if ARTIFACTS.get(rel) != sha(p):
                findings.append(f"{rel}: under artifacts/ but not a pinned file with its pinned SHA-256")
            continue          # the pinned binaries hold NeuralHub's code objects on purpose
        if name.startswith("nvngx") and name.endswith(".dll"):
            findings.append(f"{rel}: an NVIDIA NGX DLL name")
        if name.endswith(".co"):
            findings.append(f"{rel}: a code object name")
        data = open(p, "rb").read()
        if len(data) >= 20 and data[:4] == b"\x7fELF" and (data[18] | data[19] << 8) == 0xE0:
            findings.append(f"{rel}: AMDGPU ELF (e_machine 0xE0)")
        h = hashlib.sha256(data).hexdigest()
        if h in shas:
            findings.append(f"{rel}: identical to {os.path.relpath(shas[h], ROOT)}")
        src = find_probes(data, index)
        if src:
            findings.append(f"{rel}: contains bytes of {os.path.relpath(src, ROOT)}")
        if b"\0" not in data[:8192]:
            low = data.lower()
            for hx, src in hexes:
                if hx in low:
                    findings.append(f"{rel}: contains the hex spelling of bytes of {os.path.relpath(src, ROOT)}")
                    break
            for hh, src in shas.items():
                if hh.encode() in low and (rel, hh) not in PREEXISTING:
                    findings.append(f"{rel}: contains the SHA-256 of {os.path.relpath(src, ROOT)}")
    if a.tracked:
        for rel in sorted(set(ARTIFACTS) - seen_art):
            findings.append(f"{rel}: pinned artifact missing")
    for f in findings:
        print("  R2:", f)
    print(f"R2SCAN: {'PASS' if not findings else 'FAIL'} {n if not findings else len(findings)} "
          f"{'files' if not findings else 'findings'}")
    sys.exit(0 if not findings else 1)


if __name__ == "__main__":
    main()
