#!/usr/bin/env python3
"""tools/re/patchprobe.py: read any intermediate value of a NeuralHub SR kernel, hardware-exact, by
patching the code object that srref runs (a measurement probe, like the synthetic-weight probes).
The patched .co and every output stay under build/work/review/patch/runs/ (git-ignored); they are
only ever run by srref, and nothing they produce is committed.

    build/venv/bin/python -I tools/re/patchprobe.py --kernel SYMBOL --at OFFSET \
        --asm "INSTR; INSTR; ...; s_endpgm" --name RUN

run(..., tolerant=True) returns {dump, rc, stdout, stderr} instead of raising on a nonzero
exit - for fault experiments where the ROCm "Memory access fault ... on address 0x..." line
is the measurement (fault_addr() extracts it). The default return value is unchanged.

OFFSET is kernel-relative (the address column of tools/re/ssa.py). The bytes at OFFSET are
overwritten with the assembled instructions (llvm-mc gfx1201 in the ROCm container). Nothing after
them runs, so the patch must end with s_endpgm. The run's dump (srref --dump, frame 0, the 1-frame
directory build/work/probe/frames1 by default, or --frames DIR; packed M weights) goes to
build/work/review/patch/runs/RUN_dump/frameF/. In a multi-frame run every frame is patched.

Writing a patch:
- Store the register that holds the value you want (tools/re/ssa.py's listing names each value's
  destination register; tools/re/live.py checks it has not been overwritten by OFFSET) to an output
  buffer the dump will capture, at a per-thread address, then end the program.
- Use only registers that are intact at OFFSET: the kernarg pointer s[0:1] is reused late in some
  kernels. For k_sr_post<false,12,16,false,false> at OFFSET >= 0x15a0, s[50:51] = full15,
  s12 = W15, v68 = x, v78 = pixel-0 row:
      s_wait_kmcnt 0x0; v_mad_u32_u24 v0, v78, s12, v68; v_lshlrev_b32 v0, 2, v0;
      global_store_b32 v0, vX, s[50:51]; s_endpgm
  At OFFSET < 0x1ea8, s[0:1] is still the kernarg pointer, so load the buffer pointer from it
  (s_load_b64 s[100:101], s[0:1], 0x78 ...).
- Validated: patching the full15 store itself (0x25f4, v1) reproduces the oracle's pixel-0 rows
  100% (docs/notes/k_sr_post.md, Review R5).
"""
import argparse
import os
import re
import struct
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
# NeuralHub's code objects are bring-your-own: tools/exe/fetch_neuralhub.sh unpacks them into
# build/nh (DLSS5_AMD_NH overrides it); this repository holds none, its license forbids that.
NH = Path(os.environ.get("DLSS5_AMD_NH", "build/nh"))
NH = NH if NH.is_absolute() else REPO / NH
CO = NH / "sr.gfx1201.co"
NETS = {"sr": dict(co=NH / "sr.gfx1201.co", ref="build/work/srref",
                   weights="build/re/packed/M", frames="build/work/probe/frames1"),
        "rr": dict(co=NH / "rr.gfx1201.co", ref="build/work/rrref",
                   weights="build/rr/packed", frames="build/work/review/rrframes1")}
OUT = REPO / "build/work/review/patch/runs"
IMG = "docker.io/rocm/dev-ubuntu-24.04:7.2.4"


def elf_symbol(data, name):
    """(file offset, size) of a function symbol in an ELF64 code object."""
    shoff, = struct.unpack_from("<Q", data, 0x28)
    shentsize, shnum, shstrndx = struct.unpack_from("<HHH", data, 0x3A)
    secs = [struct.unpack_from("<IIQQQQIIQQ", data, shoff + i * shentsize) for i in range(shnum)]
    for (nm, typ, flags, addr, off, size, link, info, align, entsize) in secs:
        if typ != 2:                                   # SHT_SYMTAB
            continue
        strtab = secs[link]
        for j in range(size // entsize):
            st_name, st_info, st_other, st_shndx, st_value, st_size = struct.unpack_from("<IBBHQQ", data, off + j * entsize)
            s = data[strtab[4] + st_name:data.index(b"\0", strtab[4] + st_name)].decode()
            if s == name:
                sec = secs[st_shndx]
                return st_value - sec[3] + sec[4], st_size
    raise SystemExit(f"symbol {name} not found")


def assemble(asm):
    src = "\n".join(a.strip() for a in asm.split(";") if a.strip()) + "\n"
    r = subprocess.run(["podman", "run", "--rm", "-i", IMG, "/opt/rocm/llvm/bin/llvm-mc",
                        "-triple=amdgcn-amd-amdhsa", "-mcpu=gfx1201", "-show-encoding"],
                       input=src, capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(r.stderr)
    out = bytearray()
    for line in r.stdout.splitlines():
        m = re.search(r"encoding: \[([^\]]*)\]", line)
        if m:
            out += bytes(int(b, 16) for b in m.group(1).split(","))
    return bytes(out)


def run(kernel, at, asm, name, frames=None, weights=None, net="sr", tolerant=False):
    cfg = NETS[net]
    if not cfg["co"].exists():
        raise SystemExit(f"{cfg['co']} missing: run tools/exe/fetch_neuralhub.sh (bring your own "
                         f"NeuralHub v0.8.0; tools/verify/README.md)")
    frames, weights = frames or cfg["frames"], weights or cfg["weights"]
    data = bytearray(cfg["co"].read_bytes())
    off, size = elf_symbol(data, kernel)
    code = assemble(asm)
    if at + len(code) > size:
        raise SystemExit(f"patch of {len(code)} bytes at {at:#x} runs past the kernel ({size:#x} bytes)")
    data[off + at:off + at + len(code)] = code
    OUT.mkdir(parents=True, exist_ok=True)
    co = OUT / f"{name}.co"
    co.write_bytes(bytes(data))
    rel = lambda p: str(Path(p).relative_to(REPO))
    cmd = ["podman", "run", "--rm", "--device", "/dev/kfd", "--device", "/dev/dri", "--group-add", "keep-groups",
           "--security-opt", "label=disable", "-v", f"{REPO}:/w", IMG, "bash", "-c",
           f"cd /w && mkdir -p {rel(OUT / (name + '_dump'))} && {cfg['ref']} {rel(co)} {weights} {frames} "
           f"{rel(OUT / (name + '_out'))} --dump {rel(OUT / (name + '_dump'))}"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode and not tolerant:
        raise SystemExit(f"{cfg['ref']} failed:\n{r.stdout[-1500:]}\n{r.stderr[-1500:]}")
    print(f"patched {len(code)} bytes at {kernel}+{at:#x}; dump {OUT / (name + '_dump')}")
    if tolerant:                       # fault experiments: caller parses rc/stderr itself
        return dict(dump=OUT / f"{name}_dump", rc=r.returncode, stdout=r.stdout, stderr=r.stderr)
    return OUT / f"{name}_dump"


def fault_addr(text):
    """The GPU fault address ROCm prints ("... on address 0x..."), or None."""
    m = re.search(r"on address (0x[0-9a-fA-F]+)", text)
    return int(m.group(1), 16) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", required=True)
    ap.add_argument("--at", required=True, type=lambda s: int(s, 0))
    ap.add_argument("--asm", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--net", default="sr", choices=list(NETS), help="sr (srref, sr.co) or rr (rrref, rr.co)")
    ap.add_argument("--frames", default=None,
                    help="frames dir (repo-relative); default: a 1-frame dir of the net's oracle sequence")
    ap.add_argument("--weights", default=None)
    a = ap.parse_args()
    run(a.kernel, a.at, a.asm, a.name, a.frames, a.weights, a.net)


if __name__ == "__main__":
    main()
