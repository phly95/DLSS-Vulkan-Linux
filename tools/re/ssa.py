#!/usr/bin/env python3
"""Straight-line SSA view of a kernel's disassembly (tools/re/fn.sh output), for reading: every
instruction becomes `%N = mnemonic operands` with each register operand replaced by the SSA value that
last wrote it, and `--expr` prints a value as an expression tree. It parses text and never evaluates an
instruction or a value (R1: nothing of NeuralHub's runs).

    ssa.py k.s [--from ADDR] [--to ADDR]          SSA listing of [ADDR, ADDR)
    ssa.py k.s --expr ADDR [--depth N]            the value ADDR defines, as a tree N levels deep

Branches are ignored (textual order); a register read before any write in the range is `vN@in`.
"""
import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from slice import parse, defs_uses  # noqa: E402


def name(v):
    return f"%{v}" if isinstance(v, int) or not str(v).endswith("@in") and re.fullmatch(r"\d+\.\d+", str(v)) else str(v)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file")
    ap.add_argument("--from", dest="lo", default="0")
    ap.add_argument("--to", dest="hi", default="ffffffff")
    ap.add_argument("--expr")
    ap.add_argument("--depth", type=int, default=6)
    a = ap.parse_args()
    lo, hi = int(a.lo, 16), int(a.hi, 16)
    ins = [(o, t) for o, t in parse(a.file) if lo <= o < hi]
    cur = {}      # register -> SSA id
    node = {}     # SSA id -> (addr, mnemonic, [operand strings with SSA refs])
    at = {}       # addr -> SSA id
    for o, t in ins:
        d, u = defs_uses(t)
        mn, _, rest = t.partition(" ")
        ref = {r: cur.get(r, f"{r}@in") for r in u}

        def sub(m):
            tok = m.group(0)
            mm = re.fullmatch(r"([vs])\[(\d+):(\d+)\]", tok)
            if mm:
                rs = [f"{mm.group(1)}{i}" for i in range(int(mm.group(2)), int(mm.group(3)) + 1)]
                return "[" + ",".join(name(ref.get(r, r)) for r in rs) + "]"
            return name(ref.get(tok, tok))
        dst, _, srcs = rest.partition(",")
        body = re.sub(r"[vs]\[\d+:\d+\]|\b[vs]\d+\b", sub, srcs if d else rest)
        if not d:
            print(f"{o:5x}        {mn} {body.strip()}")
            continue
        n = len(node)
        node[n] = (o, mn, body.strip(), dst.strip())
        at[o] = n
        for i, r in enumerate(d):
            cur[r] = n if len(d) == 1 or r in ("vcc_lo", "scc", "exec_lo") else f"{n}.{i}"
        if not a.expr:
            print(f"{o:5x} %{n:<5} = {mn} {body.strip()}    ; {dst.strip()}")
    if a.expr:
        def show(n, depth, ind):
            o, mn, body, dst = node[n]
            print(f"{'  ' * ind}%{n} @{o:x} {mn} {body}    ; {dst}")
            if depth:
                for m in dict.fromkeys(int(x) for x in re.findall(r"%(\d+)", body)):
                    show(m, depth - 1, ind + 1)
        show(at[int(a.expr, 16)], a.depth, 0)


if __name__ == "__main__":
    sys.exit(main())
