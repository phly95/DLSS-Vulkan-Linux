#!/usr/bin/env python3
"""Unpack the NeuralHub Windows installer into the pieces the port is checked against.

    extract.py NeuralHub_v0.8.0_setup.exe OUTDIR

Writes OUTDIR/neuralhub.exe (the Tauri manager, out of the NSIS archive), OUTDIR/neuralhub_host.dll
(the in-game DLL the manager embeds), OUTDIR/<net>.<target>.co (every HIP code object of the host's
.hip_fat section, one per network and GPU target) and OUTDIR/manifest.json.

Nothing written here may be committed: it is NeuralHub's code. Run with `python3 -I`, since the input
is untrusted data.
"""
import hashlib
import json
import os
import struct
import subprocess
import sys

import pefile

EXPECTED_SETUP_SHA256 = "aae53654001bf7345dc5b2e621a3f9018f4a5735760a1572a7532180249b0834"

# Which network a bundle holds, told apart by a kernel name only that bundle defines.
NET_MARKERS = [
    (b"k_reg_swin", "nr"),
    (b"k_sr_stage", "sr"),
    (b"k_rr_gather", "rr"),
    (b"k_DL1Net_Input", "fg"),
]


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def unpack_nsis(setup, out):
    exe = os.path.join(out, "neuralhub.exe")
    if not os.path.exists(exe):
        subprocess.run(["7z", "e", "-y", "-o" + out, setup, "neuralhub.exe"],
                       check=True, stdout=subprocess.DEVNULL)
    return open(exe, "rb").read()


def find_pe(data, name):
    """Return the embedded PE whose export directory is named `name` (the host renames itself
    version.dll in its exports, so the NGX exports are what identify it)."""
    pos = 1
    while True:
        pos = data.find(b"This program cannot be run in DOS mode", pos)
        if pos < 0:
            return None
        start = data.rfind(b"MZ", max(0, pos - 0x80), pos)
        pos += 1
        if start <= 0:
            continue
        try:
            pe = pefile.PE(data=data[start:start + 0x10000000], fast_load=True)
            end = max(s.PointerToRawData + s.SizeOfRawData for s in pe.sections)
            img = data[start:start + end]
            pe = pefile.PE(data=img)
        except pefile.PEFormatError:
            continue
        exports = [e.name for e in getattr(pe, "DIRECTORY_ENTRY_EXPORT", None).symbols] \
            if hasattr(pe, "DIRECTORY_ENTRY_EXPORT") else []
        if name in exports:
            return img


def offload_bundles(blob):
    """Yield (target_id, code_object) from every clang offload bundle in blob."""
    magic = b"__CLANG_OFFLOAD_BUNDLE__"
    pos = 0
    while True:
        pos = blob.find(magic, pos)
        if pos < 0:
            return
        count = struct.unpack_from("<Q", blob, pos + 24)[0]
        p = pos + 32
        entries = []
        for _ in range(count):
            off, size, idlen = struct.unpack_from("<QQQ", blob, p)
            tid = blob[p + 24:p + 24 + idlen].decode()
            p += 24 + idlen
            entries.append((tid, blob[pos + off:pos + off + size]))
        yield pos, entries
        pos += len(magic)


def main():
    setup, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    raw = open(setup, "rb").read()
    if sha256(raw) != EXPECTED_SETUP_SHA256:
        print("warning: installer hash differs from v0.8.0 - the inventory was made from v0.8.0",
              file=sys.stderr)
    exe = unpack_nsis(setup, out)
    host = find_pe(exe, b"NeuralHub_HostStart")
    if host is None:
        sys.exit("neuralhub_host.dll not found in neuralhub.exe")
    open(os.path.join(out, "neuralhub_host.dll"), "wb").write(host)
    pe = pefile.PE(data=host, fast_load=True)
    fat = next(s for s in pe.sections if s.Name.rstrip(b"\0") == b".hip_fat")
    fat_data = fat.get_data()
    manifest = {"setup_sha256": sha256(raw), "host_sha256": sha256(host), "code_objects": []}
    for off, entries in offload_bundles(fat_data):
        net = "unknown"
        for tid, co in entries:
            for marker, n in NET_MARKERS:
                if marker in co:
                    net = n
        for tid, co in entries:
            if not co:
                continue  # empty host entry
            target = tid.split("--")[-1]
            fn = f"{net}.{target}.co"
            open(os.path.join(out, fn), "wb").write(co)
            manifest["code_objects"].append({"net": net, "target": target, "file": fn,
                                             "bundle_offset": off, "size": len(co),
                                             "sha256": sha256(co)})
    json.dump(manifest, open(os.path.join(out, "manifest.json"), "w"), indent=1)
    for c in manifest["code_objects"]:
        print(f"{c['net']:3} {c['target']:20} {c['size']:9}  {c['file']}")


if __name__ == "__main__":
    main()
