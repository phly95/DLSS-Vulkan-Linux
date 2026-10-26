#!/usr/bin/env python3
"""tools/sr/sr_layout.py: the byte map of every SR stage blob, read off NeuralHub's own packer.

    build/venv/bin/python -I tools/sr/sr_layout.py [--markdown]

Runs `sr_pack.pack_layer` (FUN_180111850) on zero-filled slots of the sizes it requires and
records every Blob call it makes, so the map cannot drift from the packer: if `pack_layer`
changes, the label sequence below no longer matches and this script fails loudly instead of
printing a stale map. No weight value is read; nothing NVIDIA's is involved.

Two encoding rules apply to every region (Blob in sr_pack.py, LAYOUT = 1):
  frag  element (r, c) of a rows x cols fp8 matrix is at byte frag_offsets(rows, cols)[r, c].
        With LAYOUT = 1 the `op` argument ("A"/"B") is ignored.
  f16   a vector of n halfs is stored PERMUTED when n % 32 == 0: blob position i holds
        logical element P[i] within each group of 32, P = 0,1,16,17,2,3,18,19,... (the same
        P the shaders apply to activations), i.e. logical element c is at blob index invP[c].
        Otherwise it is stored plain.
  f16_block (the 16x16 relative-position table) is stored plain with LAYOUT = 1.

Offsets, sizes, shapes and encodings come from the packer. The note column's semantics (gain
applied after the norm, V = K, the 0x60 slot as a per-head output projection, the down_proj as
the second output) were measured against the oracle: docs/notes/k_sr_stage.md, "Review R1".
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sr_pack  # noqa: E402


class _ZeroSlots:
    """Slots stand-in: every slot has exactly the element count pack_layer asks for, all zero."""

    def __init__(self, need):
        self.need = need

    def count(self, k):
        return self.need.get(k, 0)

    def f16(self, k, pad=0):
        return np.zeros(self.need[k] + pad, np.uint16)


def _need(layer):
    _, b, C, g, Cn, mode = layer
    n = {0x40: C, 0x50: C * g * 64, 0x60: C * g * 32, 0x70: C, 0x80: g * 256, 0x90: C,
         0xA0: 4 * C * C, 0xB0: 4 * C, 0xC0: 4 * C * C, 0xD0: C}
    if mode == 0:
        n.update({0xE0: Cn * 4 * C, 0xF0: Cn})
    if mode == 2:
        n.update({0x100: Cn * 4 * C, 0x110: 4 * C})
    return {b + k: v for k, v in n.items()}


def _labels(layer):
    """What pack_layer writes, in its order: (name, slot offset, note)."""
    _, _, C, g, Cn, mode = layer
    out = []
    if mode == 2:
        out += [("up_proj.W", 0x100, f"{4 * C}x{Cn}, mode-2 upsample projection"),
                ("up_proj.b", 0x110, f"{4 * C} halfs")]
    out.append(("gain1 (s1b0)", 0x40, "x sqrt(C); RMSNorm1 gain, applied AFTER x/||x||"))
    for h in range(g):
        out += [(f"head{h}.QK", 0x50, f"rows 0..31 Wq, 32..63 Wk (slot rows {h * 32}.. and {g * 32 + h * 32}..)"),
                (f"head{h}.relpos (v258)", 0x80, "16x16 f16, x sqrt(32) x 0.5"),
                (f"head{h}.Wo", 0x60, f"Cx32 = slot cols {h * 32}..{h * 32 + 31}; per-head output projection, V = K")]
    out += [("v270", 0x70, "added to the residual after attention"),
            ("gain2 (s198)", 0x90, "x sqrt(C); RMSNorm2 gain, applied AFTER x/||x||")]
    for j in range(C // 8):
        out += [(f"mlp{j}.fc1", 0xA0, f"32x{C} = slot rows {j * 32}..{j * 32 + 31}"),
                (f"mlp{j}.b1", 0xB0, "32 halfs"),
                (f"mlp{j}.fc2", 0xC0, f"{C}x32 = slot cols {j * 32}..{j * 32 + 31}")]
        if j == 0:
            out.append(("fc2_bias (v240)", 0xD0, f"{C} halfs, written once after block 0"))
    if mode == 0:
        out += [("down_proj.W", 0xE0, f"{Cn}x{4 * C}, mode-0 2x2 patch-merge projection -> second output"),
                ("down_proj.b", 0xF0, f"{Cn} halfs")]
    return out


def layer_map(layer):
    calls = []
    orig = sr_pack.Blob.frag, sr_pack.Blob.f16, sr_pack.Blob.f16_block

    def frag(self, m, op="A"):
        calls.append(("frag fp8", len(self.b), m.rows * m.cols, f"{m.rows}x{m.cols}"))
        orig[0](self, m, op)

    def f16(self, u16):
        n = np.asarray(u16).size
        calls.append(("f16 " + ("PERMUTED" if n % 32 == 0 else "plain"), len(self.b), 2 * n, f"{n}"))
        orig[1](self, u16)

    def f16_block(self, u16):
        calls.append(("f16 block plain", len(self.b), 512, "16x16"))
        orig[2](self, u16)

    sr_pack.Blob.frag, sr_pack.Blob.f16, sr_pack.Blob.f16_block = frag, f16, f16_block
    try:
        blob, _ = sr_pack.pack_layer(_ZeroSlots(_need(layer)), layer)
    finally:
        sr_pack.Blob.frag, sr_pack.Blob.f16, sr_pack.Blob.f16_block = orig
    labels = _labels(layer)
    if len(labels) != len(calls):
        raise SystemExit(f"{layer[0]}: pack_layer made {len(calls)} writes, the label table has "
                         f"{len(labels)} - sr_pack.py changed; update _labels()")
    return [(lab[0], lab[1], c[0], c[1], c[2], c[3], lab[2]) for lab, c in zip(labels, calls)], len(blob)


def main():
    md = "--markdown" in sys.argv
    seen = {}
    for i, layer in enumerate(sr_pack.LAYERS):
        rows, size = layer_map(layer)
        if size != sr_pack.STAGE_BYTES[i]:
            raise SystemExit(f"{layer[0]}: {size:#x} bytes, NeuralHub requires {sr_pack.STAGE_BYTES[i]:#x}")
        name, _, C, g, Cn, mode = layer
        title = f"stage{i}.bin = {name}: C={C} g={g} Cn={Cn} mode={mode}, {size} bytes"
        key = tuple(r[2:6] for r in rows)
        if key in seen:
            print(f"\n{'### ' if md else ''}{title}: same layout as stage{seen[key]}.bin")
            continue
        seen[key] = i
        print(f"\n### {title}\n" if md else f"\n{title}")
        if md:
            print("| region | slot | encoding | offset | bytes | shape | note |\n|---|---|---|---:|---:|---|---|")
        for lab, slot, enc, off, n, shape, note in rows:
            if md:
                print(f"| {lab} | +{slot:#x} | {enc} | {off} | {n} | {shape} | {note} |")
            else:
                print(f"  {off:7d} {n:7d}  {enc:16s} {shape:9s} {lab:22s} {note}")


if __name__ == "__main__":
    main()
