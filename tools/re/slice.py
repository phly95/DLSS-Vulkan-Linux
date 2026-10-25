#!/usr/bin/env python3
"""Def-use slices over a kernel's disassembly text (tools/re/fn.sh output): which earlier instruction
defined each register an instruction reads, followed back to a given depth. A reading aid for the
interleaved schedules of NeuralHub's kernels; it parses text and never evaluates an instruction or
a value (R1: nothing of NeuralHub's runs).

    fn.sh sr <symbol> > k.s
    slice.py k.s ADDR [ADDR ...] [--depth N]     the instructions ADDR reads from, N levels back
    slice.py k.s --uses ADDR                      the later instructions that read what ADDR wrote
    slice.py k.s --list PATTERN                   instructions whose text matches PATTERN (a regex)

ADDR is the offset from the kernel's first instruction (hex), as `--list` prints it. Branches are
ignored: within a loop body the slice follows the textual order, which is the first iteration's;
`--list s_cbranch` shows where loops are. Partial writes (v_fma_mix{lo,hi}, v_cvt_pk_fp8 with op_sel,
d16 loads, v_perm with a merged operand) also read the destination, so the slice shows both halves.
"""
import argparse
import re
import sys

PARTIAL = re.compile(r"^(v_fma_mixlo|v_fma_mixhi|v_cvt_pk_fp8|v_cvt_pk_bf8|v_cvt_sr_|.*_d16|v_mov_b16|v_cndmask_b16)")
NO_DEF = re.compile(r"^(global_store|buffer_store|flat_store|scratch_store|ds_store|ds_write|s_barrier|s_wait|s_cbranch|s_branch|"
                    r"s_endpgm|s_clause|s_setprio|s_sendmsg|s_cmp|v_cmpx)")


def regs(tok):
    """registers named by one operand: ['v5'], ['v12', 'v13'] for v[12:13], ['s4'], ['vcc_lo'], [] for literals"""
    tok = tok.strip()
    tok = re.sub(r"^(neg|abs|-)\(?", "", tok).rstrip(")")
    m = re.fullmatch(r"([vs])\[(\d+):(\d+)\]", tok)
    if m:
        return [f"{m.group(1)}{i}" for i in range(int(m.group(2)), int(m.group(3)) + 1)]
    m = re.fullmatch(r"([vs])(\d+)", tok)
    if m:
        return [tok]
    if tok in ("vcc_lo", "vcc_hi", "vcc", "exec_lo", "exec", "scc", "m0"):
        return [tok]
    return []


def parse(path):
    out = []
    base = None
    for line in open(path):
        m = re.search(r"//\s*([0-9A-F]+):", line)
        if not m:
            continue
        a = int(m.group(1), 16)
        base = a if base is None else base
        text = line.split("//")[0].strip()
        if not text or text.startswith(("s_delay_alu", "s_wait_alu", "s_nop", "s_code_end")):
            continue
        out.append((a - base, text))
    return out


def defs_uses(text):
    parts = [p.strip() for p in text.split("::")]
    defs, uses = [], []
    for p in parts:
        mn, _, rest = p.partition(" ")
        ops = [o for o in re.split(r",\s*", rest) if o and not re.match(r"^(op_sel|op_sel_hi|offset|off$|neg_lo|neg_hi|clamp|sc0|sc1|nt|dlc|glc|slc)", o)]
        ops = [re.sub(r"\s+(offset|offset0|offset1|op_sel|op_sel_hi)[:0-9a-fx\[\],]*.*$", "", o) for o in ops]
        if NO_DEF.match(mn):
            for o in ops:
                uses += regs(o)
            if mn.startswith(("s_cmp", "v_cmpx")):
                defs.append("scc" if mn.startswith("s_cmp") else "exec_lo")
            continue
        if not ops:
            continue
        d = regs(ops[0])
        defs += d
        srcs = ops[1:]
        if mn.startswith("v_") and re.search(r"_co_|_co_ci_", mn) and srcs and (srcs[0] == "null" or regs(srcs[0]) and regs(srcs[0])[0][0] == "s" or srcs[0].startswith("vcc")):
            defs += regs(srcs[0])
            srcs = srcs[1:]
        for o in srcs:
            uses += regs(o)
        if PARTIAL.match(mn) or (mn.startswith("v_cvt_pk_fp8") ):
            uses += d
        if mn.startswith("s_") and not mn.startswith(("s_load", "s_mov", "s_getpc")):
            defs.append("scc")
        if mn.startswith("v_cndmask") and "vcc" not in rest:
            pass
        if mn.startswith("v_cndmask_b32_e32"):
            uses.append("vcc_lo")
    return defs, uses


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file")
    ap.add_argument("addr", nargs="*")
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--uses")
    ap.add_argument("--list")
    a = ap.parse_args()
    ins = parse(a.file)
    idx = {off: i for i, (off, _) in enumerate(ins)}
    du = [defs_uses(t) for _, t in ins]
    if a.list:
        for off, t in ins:
            if re.search(a.list, t):
                print(f"{off:5x} {t}")
        return
    if a.uses:
        i = idx[int(a.uses, 16)]
        live = set(du[i][0])
        for j in range(i + 1, len(ins)):
            hit = live & set(du[j][1])
            if hit:
                print(f"{ins[j][0]:5x} {ins[j][1]}    <- {', '.join(sorted(hit))}")
            live -= set(du[j][0]) - hit
            if not live:
                break
        return

    def definer(i, r):
        for j in range(i - 1, -1, -1):
            if r in du[j][0]:
                return j
        return None

    seen = set()

    def show(i, depth, ind):
        off, t = ins[i]
        print(f"{'  ' * ind}{off:5x} {t}")
        if depth == 0 or i in seen:
            if i in seen and depth:
                print(f"{'  ' * (ind + 1)}(shown above)")
            return
        seen.add(i)
        by_def = {}
        for r in dict.fromkeys(du[i][1]):
            j = definer(i, r)
            if j is None:
                print(f"{'  ' * (ind + 1)}{r}: (kernel entry)")
            else:
                by_def.setdefault(j, []).append(r)
        for j in sorted(by_def, reverse=True):
            show(j, depth - 1, ind + 1)

    for s in a.addr:
        show(idx[int(s, 16)], a.depth, 0)
        print()


if __name__ == "__main__":
    sys.exit(main())
