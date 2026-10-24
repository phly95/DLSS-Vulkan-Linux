#!/usr/bin/env python3
"""Read the DLSS-SR preset M / L weight tables out of the user's nvngx_dlss.dll (310.9.1.0).

NVIDIA's DLL fills a table object in one function: `switch (preset) { case 0xc (L): ...; case 0xd
(M): ... }`, where every slot at byte offset k of the object gets an element count at [k] (a
`mov dword [rcx+k], imm32`) and a pointer to the data at [k+8] (`lea rax, [rip+X]; mov [rcx+k+8], rax`).
The function is found by its prologue (`sub rsp, 0x38; cmp edx, 0xc; jne ...; lea rax, [rip+X]`),
exactly one match in 310.9.1.0, the same pattern NeuralHub looks for. The DLL is only read, never
loaded or run. Values are f16.

    sr_weights.py nvngx_dlss.dll --list            # slot table of both presets
    sr_weights.py nvngx_dlss.dll --out DIR         # DIR/{M,L}/slot_XXXX.f16 + index.json, DIR/dither.bin

Nothing this writes may be committed: the values are NVIDIA's.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import capstone
import pefile

PRESETS = {0xC: "L", 0xD: "M"}
PROLOGUE = bytes.fromhex("4883ec3883fa0c0f85")


def find_init(pe):
    hits = []
    ib = pe.OPTIONAL_HEADER.ImageBase
    for s in pe.sections:
        if s.Name.rstrip(b"\0") != b".text":
            continue
        d = s.get_data()
        i = d.find(PROLOGUE)
        while i >= 0:
            if d[i + 13:i + 16] == bytes.fromhex("488d05"):
                hits.append(ib + s.VirtualAddress + i)
            i = d.find(PROLOGUE, i + 1)
    if len(hits) != 1:
        raise SystemExit(f"weight-table initialiser: {len(hits)} candidates (expected 1)")
    return hits[0]


def parse(pe, start):
    """Follow the initialiser: returns {preset: {slot: [count, va]}}."""
    ib = pe.OPTIONAL_HEADER.ImageBase
    md = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)
    md.detail = True
    tables = {}
    pending = [(start, None, None)]
    seen = set()
    while pending:
        va, preset, rax = pending.pop()
        if (va, preset) in seen:
            continue
        seen.add((va, preset))
        for ins in md.disasm(pe.get_data(va - ib, 0x2000), va):
            m, ops = ins.mnemonic, ins.operands
            if m == "sub" or m == "add":
                continue
            if m == "cmp" and ops[0].type == capstone.x86.X86_OP_REG:
                cmp_imm = ops[1].imm
                continue
            if m == "jne":
                # fall-through is the case `edx == cmp_imm`; the branch tests the next case
                pending.append((ops[0].imm, None, None))
                preset = cmp_imm
                tables.setdefault(preset, {})
                continue
            if m == "lea":
                rax = ins.address + ins.size + ops[1].mem.disp
                continue
            if m == "mov" and ops[0].type == capstone.x86.X86_OP_MEM and ops[0].mem.base == capstone.x86.X86_REG_RCX:
                k = ops[0].mem.disp
                if ops[0].size == 4:
                    tables[preset].setdefault(k, [None, None])[0] = ops[1].imm
                elif ops[0].size == 8:
                    tables[preset].setdefault(k - 8, [None, None])[1] = rax
                continue
            if m == "jmp":
                pending.append((ops[0].imm, preset, rax))  # the tail stores rax too
                break
            if m == "ret" or m == "call" or (m == "mov" and ops[0].type == capstone.x86.X86_OP_MEM):
                break
            raise SystemExit(f"unexpected instruction at {ins.address:#x}: {m} {ins.op_str}")
    return {PRESETS[p]: t for p, t in tables.items() if p in PRESETS}


DITHER_FNV = 0xB647896B35218409   # FNV-1a-64 (basis below) of the dither table's first 16 bytes
DITHER_SHA = "4707682a85d793cafcfe746e70d181190ba59bd989b88e27dee9edf1f20305b8"
PRESET_SHA = {"L": "a9b038908c6781c104fc7aca8e59004afd8c6ce4c29f12b356ddeb53ce456d6e",
              "M": "0ff032b7fb90f49513dc0d4d101a52266a78cecf4499bfd0ea3ecaeb72c64155"}


def fnv(b):
    h = 0x14650FB0739D0383
    for x in b:
        h = ((h ^ x) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def find_dither(pe):
    """The 4096-byte dither table (NeuralHub hashes it with the weights). Searched in every
    section but .text (slow: about a minute; only when the 310.9.1.0 address does not hold it)."""
    hits = []
    for s in pe.sections:
        if s.Name.rstrip(b"\0") == b".text":
            continue
        d = s.get_data()
        for i in range(0, len(d) - 0x1000 + 1, 4):
            if fnv(d[i:i + 16]) != DITHER_FNV:
                continue
            if hashlib.sha256(d[i:i + 0x1000]).hexdigest() == DITHER_SHA:
                hits.append(s.VirtualAddress + i)
    return hits


def checksum(pe, t, name, dither):
    ib = pe.OPTIONAL_HEADER.ImageBase
    h = hashlib.sha256()
    for k in sorted(t):
        n, p = t[k]
        pad = 16 if (k == 0x200 and name == "M") else 0
        h.update(pe.get_data(p - ib, 2 * (n + pad)))
    h.update(dither)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dll", type=Path)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--dither-rva", type=lambda v: int(v, 0), default=0x135290,
                    help="where 310.9.1.0 keeps the dither table; checked by its SHA-256")
    a = ap.parse_args()
    pe = pefile.PE(str(a.dll), fast_load=True)
    ib = pe.OPTIONAL_HEADER.ImageBase
    tables = parse(pe, find_init(pe))
    dither = pe.get_data(a.dither_rva, 0x1000)
    if hashlib.sha256(dither).hexdigest() != DITHER_SHA:
        hits = find_dither(pe)
        if len(hits) != 1:
            raise SystemExit(f"dither table not found ({len(hits)} candidates)")
        dither = pe.get_data(hits[0], 0x1000)
    if a.out:
        a.out.mkdir(parents=True, exist_ok=True)
        (a.out / "dither.bin").write_bytes(dither)
    for name, t in sorted(tables.items()):
        missing = [hex(k) for k, (n, p) in t.items() if n is None or p is None]
        total = sum(n for n, p in t.values() if n)
        ok = checksum(pe, t, name, dither) == PRESET_SHA[name]
        print(f"preset {name}: {len(t)} slots, {total} f16 values, incomplete: {missing}, "
              f"NeuralHub checksum {'matches' if ok else 'differs'}", file=sys.stderr)
        if a.list:
            for k in sorted(t):
                n, p = t[k]
                print(f"{name} {k:#06x} {n} {p:#x}" if n and p else f"{name} {k:#06x} {n} {p}")
        if a.out:
            d = a.out / name
            d.mkdir(parents=True, exist_ok=True)
            idx = {}
            for k in sorted(t):
                n, p = t[k]
                if not (n and p):
                    continue
                blob = pe.get_data(p - ib, 2 * n)
                (d / f"slot_{k:04x}.f16").write_bytes(blob)
                idx[f"{k:#06x}"] = dict(count=n, rva=p - ib, sha256=hashlib.sha256(blob).hexdigest())
            (d / "index.json").write_text(json.dumps(idx, indent=1))


if __name__ == "__main__":
    main()
