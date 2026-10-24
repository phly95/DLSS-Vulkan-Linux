#!/usr/bin/env python3
"""Read the DLSS-RR preset F weight table out of the user's nvngx_dlssd.dll (310.9.1.0).

NVIDIA's DLL fills one table object in a straight-line function (`xor r8d, r8d; mov [rdx+0x48], r8;
mov [rdx+0x40], r8d; ...; ret`): tensor k gets an element count at [k] (dword) and a data pointer at
[k+8] (`lea rax, [rip+X]; mov [rdx+k+8], rax`); a few f16 scalars are stored as immediates. NeuralHub
finds the function by its first 18 bytes and replays the stores into a 0x2000-byte image; this does
the same with capstone. The DLL is only read, never loaded or run. Values are f16.

    rr_weights.py nvngx_dlssd.dll --list        # tensor table
    rr_weights.py nvngx_dlssd.dll --out DIR     # DIR/table.bin (the replayed 0x2000-byte object),
                                                # DIR/written.bin (its byte-written mask),
                                                # DIR/t_XXXX.f16 per tensor, DIR/index.json, DIR/dither.bin,
                                                # DIR/bicubic.bin (2052 f32 for the post pass)

NeuralHub's fingerprint of preset F (`a05b941a...`) is SHA-256 over every tensor's values in key
order, then the f16 scalars at +0x220..+0x236 of each of the 11 level blocks (0x238 bytes each),
then the 4096-byte dither table; checked here.

Nothing this writes may be committed: the values are NVIDIA's.
"""
import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

import capstone
import pefile

PATTERN = bytes.fromhex("4533c04c894248448942404c898298000000")
DLL_SHA = "4bc7ea5fcb2f32cf86bc2cb072e8d2860914a0be511cbdcb2fc206c1d9d83b80"   # 310.9.1.0
F_SHA = "a05b941a52679b76436a7c38df60a194b513948b7b1bf89b9f7b34133ea6fe8b"
DITHER_SHA = "4707682a85d793cafcfe746e70d181190ba59bd989b88e27dee9edf1f20305b8"   # same table as SR's
N_TENSORS, N_VALUES = 186, 3193136
TABLE = 0x2000
BLOCK = 0x238


def find_init(pe):
    hits = []
    for s in pe.sections:
        if s.Name.rstrip(b"\0") != b".text":
            continue
        d = s.get_data()
        i = d.find(PATTERN)
        while i >= 0:
            hits.append(s.VirtualAddress + i)
            i = d.find(PATTERN, i + 1)
    if len(hits) != 1:
        raise SystemExit(f"F weight-table initialiser: {len(hits)} candidates (expected 1)")
    return hits[0]


def replay(pe, rva):
    """Returns (table bytes, written mask, {offset: rva} of pointer stores)."""
    ib = pe.OPTIONAL_HEADER.ImageBase
    md = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)
    md.detail = True
    X = capstone.x86
    table = bytearray(TABLE)
    written = bytearray(TABLE)
    ptrs = {}
    regs = {"r8": (0, False), "rax": (0, False)}   # value, is-pointer (an RVA)
    for ins in md.disasm(pe.get_data(rva, 0x10000), ib + rva):
        m, ops = ins.mnemonic, ins.operands
        if m == "ret":
            return bytes(table), bytes(written), ptrs
        if m == "xor":
            regs["r8"] = (0, False)
            continue
        if m == "lea":
            regs["rax"] = (ins.address + ins.size + ops[1].mem.disp - ib, True)
            continue
        if m == "mov" and ops[0].type == X.X86_OP_REG:
            # mov eax, imm / mov rax, [rip+x] (NeuralHub takes 0) / mov rax, rcx (return value)
            regs["rax"] = (ops[1].imm & 0xFFFFFFFF, False) if ops[1].type == X.X86_OP_IMM else (0, False)
            continue
        if m != "mov" or ops[0].type != X.X86_OP_MEM:
            raise SystemExit(f"unexpected instruction at {ins.address:#x}: {m} {ins.op_str}")
        mem = ops[0].mem
        if mem.base == X.X86_REG_RCX:      # the returned object, not the table
            continue
        if mem.base != X.X86_REG_RDX or mem.disp < 0:
            raise SystemExit(f"unexpected store at {ins.address:#x}: {ins.op_str}")
        k, n = mem.disp, ops[0].size
        if ops[1].type == X.X86_OP_IMM:
            v, isptr = ops[1].imm, False
        else:
            name = ins.reg_name(ops[1].reg)
            v, isptr = regs["r8" if name.startswith("r8") else "rax"]
        if isptr:
            if n != 8:
                raise SystemExit(f"pointer stored as {n} bytes at {ins.address:#x}")
            ptrs[k] = v
        else:
            ptrs.pop(k, None)
        if k + n > TABLE:
            raise SystemExit(f"store outside the table at {ins.address:#x}")
        table[k:k + n] = (v & ((1 << (8 * n)) - 1)).to_bytes(n, "little")
        written[k:k + n] = b"\1" * n
    raise SystemExit("F initialiser: no ret found")


def tensors(table, written, ptrs):
    """{key: (count, rva)} the way NeuralHub collects them (step 8)."""
    out = {}
    for k in range(0, TABLE - 0x10 + 1, 8):
        cnt = struct.unpack_from("<I", table, k)[0]
        if k + 8 in ptrs and all(written[k:k + 4]) and 0 < cnt < 0x1000000:
            out[k] = (cnt, ptrs[k + 8])
    return out


def scalars(table, written):
    out = b""
    for blk in range(11):
        for o in range(0x220, BLOCK, 2):
            k = blk * BLOCK + o
            if written[k] and written[k + 1]:
                out += table[k:k + 2]
    return out


BICUBIC_NAME = b"kBicubicHwPhaseWeightsGlobal\0"
BICUBIC_SHA = "8f4c75775480450c05aae2e0a37cd64582834babba85346d99a0bbcd2ef88c7e"


def find_bicubic(raw):
    """The 0x2010-byte `kBicubicHwPhaseWeightsGlobal` from the CUDA ELF images inside the DLL (every
    copy is identical; NeuralHub requires that and checks its SHA-256)."""
    copies = set()
    i = raw.find(b"\x7fELF\x02\x01")
    while i >= 0:
        if struct.unpack_from("<H", raw, i + 0x3A)[0] == 0x40:
            shoff = struct.unpack_from("<Q", raw, i + 0x28)[0]
            shnum = struct.unpack_from("<H", raw, i + 0x3C)[0]
            if 0 < shnum < 0x1000 and i + shoff + shnum * 0x40 <= len(raw):
                sh = [struct.unpack_from("<IIQQQQIIQQ", raw, i + shoff + k * 0x40) for k in range(shnum)]
                for s in sh:
                    if s[1] != 2:   # SHT_SYMTAB
                        continue
                    strtab = sh[s[6]]
                    for e in range(0, s[5], 0x18):
                        nm, _, _, shndx, val, sz = struct.unpack_from("<IBBHQQ", raw, i + s[4] + e)
                        n0 = i + strtab[4] + nm
                        if sz == 0x2010 and shndx < shnum and raw[n0:n0 + len(BICUBIC_NAME)] == BICUBIC_NAME:
                            o = i + sh[shndx][4] + val
                            copies.add(raw[o:o + 0x2010])
        i = raw.find(b"\x7fELF\x02\x01", i + 1)
    if len(copies) != 1:
        raise SystemExit(f"bicubic phase table: {len(copies)} distinct copies (expected 1)")
    return copies.pop()


def find_dither(pe, data):
    hits = []
    for s in pe.sections:
        d = s.get_data()
        i = d.find(data)
        while i >= 0:
            hits.append(s.VirtualAddress + i)
            i = d.find(data, i + 1)
    return hits


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dll", type=Path)
    ap.add_argument("--dither", type=Path, default=Path("build/re/weights/dither.bin"),
                    help="the 4096-byte dither table from sr_weights.py (the same table is in nvngx_dlssd.dll)")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    raw = a.dll.read_bytes()
    if hashlib.sha256(raw).hexdigest() != DLL_SHA:
        print("warning: not nvngx_dlssd.dll 310.9.1.0; continuing, the fingerprint decides", file=sys.stderr)
    pe = pefile.PE(data=raw, fast_load=True)
    table, written, ptrs = replay(pe, find_init(pe))
    t = tensors(table, written, ptrs)
    total = sum(n for n, _ in t.values())
    print(f"F weight table: {len(t)} tensors, {total} f16 values "
          f"(NeuralHub expects {N_TENSORS}, {N_VALUES})", file=sys.stderr)
    dither = a.dither.read_bytes()
    if hashlib.sha256(dither).hexdigest() != DITHER_SHA:
        raise SystemExit("dither table: wrong file")
    hits = find_dither(pe, dither)
    print(f"dither table at RVA {', '.join(hex(h) for h in hits)}", file=sys.stderr)
    h = hashlib.sha256()
    for k in sorted(t):
        n, p = t[k]
        h.update(pe.get_data(p, 2 * n))
    h.update(scalars(table, written))
    h.update(dither)
    ok = h.hexdigest() == F_SHA
    bicubic = find_bicubic(raw)
    ok_b = hashlib.sha256(bicubic).hexdigest() == BICUBIC_SHA
    print(f"bicubic phase table: {'matches NeuralHub' if ok_b else 'DIFFERS from NeuralHub'}", file=sys.stderr)
    ok = ok and ok_b
    print(f"fingerprint {h.hexdigest()}: {'matches NeuralHub' if ok else 'DIFFERS from NeuralHub'}", file=sys.stderr)
    if a.list:
        for k in sorted(t):
            n, p = t[k]
            print(f"{k:#06x} block {k // BLOCK:2d} +{k % BLOCK:#05x} {n:8d} {p:#x}")
    if a.out:
        a.out.mkdir(parents=True, exist_ok=True)
        (a.out / "table.bin").write_bytes(table)
        (a.out / "written.bin").write_bytes(written)
        (a.out / "dither.bin").write_bytes(dither)
        (a.out / "bicubic.bin").write_bytes(bicubic)
        idx = {}
        for k in sorted(t):
            n, p = t[k]
            blob = pe.get_data(p, 2 * n)
            (a.out / f"t_{k:04x}.f16").write_bytes(blob)
            idx[f"{k:#06x}"] = dict(count=n, rva=p)
        (a.out / "index.json").write_text(json.dumps(idx, indent=1))
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
