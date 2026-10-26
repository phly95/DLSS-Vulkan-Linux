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

enc0.bin and dec0.bin (presets M and L) are mapped from sr_pack.build() itself: build() runs on
zero-filled slots that answer its own count checks (_Expect), the dither is zero bytes of the size
build() leaves for it, and every write build() makes is recorded (_Recorder). The table of
build()'s own writes (ENC0_TOP, DEC0_TOP) is checked against that recording, and the pack_layer
blobs it makes must equal layer_map()'s standalone record.
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


class _Expect(int):
    """build() checks `S.count(k) != n`. This answers every count with a value whose `!=` records
    the packer's n and passes, so the element counts are the packer's own, not copied here."""

    def __new__(cls, rec, k):
        obj = int.__new__(cls, 0)
        obj.rec, obj.k = rec, k
        return obj

    def __ne__(self, n):
        self.rec.expect[self.k] = int(n)
        return False


class _RecSlots:
    """Slots stand-in for sr_pack.build(): zero values of the counts the packer checked, f16(k, pad)
    honours pad. Preset L reads slots 0xf60/0xf70/0xf80 without a count check: one value is given."""

    def __init__(self, rec):
        self.rec = rec

    def count(self, k):
        return _Expect(self.rec, k)

    def f16(self, k, pad=0):
        rec = self.rec
        if k in rec.expect:
            n = rec.expect[k]
        elif rec.depth == 0:
            n = 1
            rec.unchecked.add(k)
        else:
            raise SystemExit(f"slot {k:#x} read inside pack_layer without a count check")
        if rec.depth == 0:
            rec.pending.append((k, pad))
        return np.zeros(n + pad, np.uint16)


class _Recorder:
    """Patches sr_pack.Blob and sr_pack.pack_layer while build() runs, and logs every write: the Blob
    (inst), the pack_layer depth and layer, encoding, byte offset, length, shape and, for depth-0
    writes (build()'s own), the slots read since the previous depth-0 write."""

    def __init__(self):
        self.expect, self.unchecked, self.pending = {}, set(), []
        self.depth, self.layers, self.insts, self.writes = 0, [], [], []

    def _write(self, s, enc, n, shape):
        slots = []
        if self.depth == 0:
            slots, self.pending = self.pending, []
        self.writes.append(dict(inst=s._inst, depth=self.depth, layer=self.layers[-1] if self.layers else None,
                                enc=enc, off=len(s.b), n=n, shape=shape, slots=slots))

    def __enter__(self):
        B = sr_pack.Blob
        self.saved = (B.__init__, B.frag, B.f16, B.f16_block, B.raw, sr_pack.pack_layer)
        init, frag, f16, blk, raw, pl = self.saved
        rec = self

        def new(s):
            init(s)
            s._inst = len(rec.insts)
            rec.insts.append(rec.depth)

        def frag_(s, m, op="A"):
            rec._write(s, "frag fp8", m.rows * m.cols, f"{m.rows}x{m.cols}")
            frag(s, m, op)

        def f16_(s, u16):
            n = np.asarray(u16).size
            rec._write(s, "f16 PERMUTED" if n % 32 == 0 else "f16 plain", 2 * n, str(n))
            f16(s, u16)

        def blk_(s, u16):
            rec._write(s, "f16 block plain", 512, "16x16")
            blk(s, u16)

        def raw_(s, b):
            rec._write(s, "raw", len(bytes(b)), "raw")
            raw(s, b)

        def pl_(S, layer):
            rec.layers.append(layer[0])
            rec.depth += 1
            try:
                return pl(S, layer)
            finally:
                rec.depth -= 1
                rec.layers.pop()

        B.__init__, B.frag, B.f16, B.f16_block, B.raw = new, frag_, f16_, blk_, raw_
        sr_pack.pack_layer = pl_
        return self

    def __exit__(self, *exc):
        B = sr_pack.Blob
        B.__init__, B.frag, B.f16, B.f16_block, B.raw, sr_pack.pack_layer = self.saved
        return False


# build()'s own enc0 / dec0 writes, in order: (region, encoding, slot, shape, note). The slots and
# shapes are checked against the recording; "sub" rows are pack_layer's blobs, placed at their offset.
DITHER = ("dither (name in sr_pack)", "raw", None, "raw",
          "bytes of dither.bin; the size is what build() leaves over (not checked by sr_pack); "
          "semantics not yet decoded")
ENC0_TOP = {
    "M": [DITHER,
          ("matrix (no name in sr_pack)", "frag fp8", 0x200, "64x128",
           "element (r, c) = slot 0x200[(c & 31) + 20 * (c >> 5) + 80 * r] (to_e4m3), 16 halfs read past "
           "the count; semantics not yet decoded"),
          ("vector (no name in sr_pack)", "f16 PERMUTED", 0x210, "64", "semantics not yet decoded")],
    "L": [DITHER,
          ("matrix (no name in sr_pack)", "frag fp8", 0x120, "32x32",
           "element (r, c) = slot 0x120[20 * r + c] for c < 20, 0 for c >= 20; semantics not yet decoded"),
          ("vector (no name in sr_pack)", "f16 PERMUTED", 0x130, "32", "semantics not yet decoded"),
          ("matrix (no name in sr_pack)", "frag fp8", 0x140, "32x32",
           "element (r, c) = slot 0x140[32 * r + c]; semantics not yet decoded"),
          ("vector (no name in sr_pack)", "f16 PERMUTED", 0x150, "32", "semantics not yet decoded"),
          ("sub", "pack_layer", "enc0L", None, "pack_layer(ENC0L): the stage labels below")],
}
DEC0_TOP = {
    "M": [("matrix (no name in sr_pack)", "frag fp8", 0xF40, "128x64",
           "element (r, c) = slot 0xf40[64 * r + c] (to_e4m3); semantics not yet decoded"),
          ("vector (no name in sr_pack)", "f16 PERMUTED", 0xF50, "128", "semantics not yet decoded")],
    "L": [("sub", "pack_layer", "dec0L", None, "pack_layer(DEC0L): the stage labels below")],
}
SCALARS = [0xF60, 0xF70, 0xF80]


def _sub_rows(layer, base):
    """pack_layer(layer)'s rows (layer_map, the stage labels) with slot ids and offsets made absolute."""
    rows, size = layer_map(layer)
    return [(lab, f"{layer[1] + slot:#x}", enc, base + off, n, shape, note)
            for lab, slot, enc, off, n, shape, note in rows], size


def _check_stage_cross(rec, preset):
    """build()'s own pack_layer writes (depth >= 1) must equal layer_map's standalone record."""
    seen = {}
    for w in rec.writes:
        if w["depth"] >= 1:
            seen.setdefault(w["layer"], []).append((w["enc"], w["n"], w["off"]))
    for layer in list(sr_pack.LAYERS) + [sr_pack.ENC0L, sr_pack.DEC0L]:
        if layer[0] not in seen:
            continue
        rows, _ = layer_map(layer)
        want = [(enc, n, off) for _, _, enc, off, n, _, _ in rows]
        if want != seen[layer[0]]:
            raise SystemExit(f"{preset}: build() writes of {layer[0]} differ from layer_map()")


def _record_top(preset):
    """Runs build() twice (the first pass finds the dither size build() leaves room for, the second is the
    real run) and returns (dither size, {"enc0": (writes, bytes), "dec0": (...)}, blobs)."""
    rec = _Recorder()
    try:
        with rec:
            sr_pack.build(_RecSlots(rec), preset, b"")
    except SystemExit as e:
        if not str(e).startswith("enc0/dec0"):
            raise
    enc_inst = next(i for i, d in enumerate(rec.insts) if d == 0)
    rest = sum(w["n"] for w in rec.writes if w["inst"] == enc_inst)
    dither_n = sr_pack.ENC0_BYTES[preset] - rest
    if dither_n <= 0:
        raise SystemExit(f"{preset}: enc0 has no room for the dither ({rest} bytes of other writes)")
    rec = _Recorder()
    with rec:
        blobs, _ = sr_pack.build(_RecSlots(rec), preset, bytes(dither_n))
    tops = [i for i, d in enumerate(rec.insts) if d == 0]
    if len(tops) != 2:
        raise SystemExit(f"{preset}: expected two depth-0 Blobs (enc0, dec0), found {len(tops)}")
    out = {}
    for name, inst in (("enc0", tops[0]), ("dec0", tops[1])):
        ws = [w for w in rec.writes if w["inst"] == inst and w["depth"] == 0]
        end = 0
        for w in ws:
            if w["off"] != end:
                raise SystemExit(f"{preset} {name}: write at {w['off']} does not follow {end}")
            end += w["n"]
        out[name] = (ws, end)
    if not rec.unchecked <= set(SCALARS):
        raise SystemExit(f"{preset}: unexpected unchecked slot reads {sorted(rec.unchecked)}")
    leftover = [k for k, _ in rec.pending]
    if leftover != SCALARS:
        raise SystemExit(f"{preset}: reads after the last write are {leftover}, expected {SCALARS}")
    _check_stage_cross(rec, preset)
    return dither_n, out, blobs


def enc_dec_rows(preset):
    """Rows for enc0.bin and dec0.bin of one preset, from build()'s recording, and the blobs."""
    dither_n, recs, blobs = _record_top(preset)
    result = {}
    for name in ("enc0", "dec0"):
        ws, end = recs[name]
        spec = (ENC0_TOP if name == "enc0" else DEC0_TOP)[preset]
        size = (sr_pack.ENC0_BYTES if name == "enc0" else sr_pack.DEC0_BYTES)[preset]
        if end != size or len(blobs[name]) != size:
            raise SystemExit(f"{preset} {name}: {end} written, {len(blobs[name])} returned, {size} required")
        if len(spec) != len(ws):
            raise SystemExit(f"{preset} {name}: build() made {len(ws)} top-level writes, the table has {len(spec)}")
        rows = []
        for (region, enc, slot, shape, note), w in zip(spec, ws):
            if region == "sub":
                sub_layer = sr_pack.ENC0L if slot == "enc0L" else sr_pack.DEC0L
                if w["enc"] != "raw":
                    raise SystemExit(f"{preset} {name}: the pack_layer blob is not a raw write")
                srows, sub_size = _sub_rows(sub_layer, w["off"])
                if sub_size != w["n"]:
                    raise SystemExit(f"{preset} {name}: {slot} is {sub_size} bytes, build() wrote {w['n']}")
                rows += srows
                continue
            got = [k for k, _ in w["slots"]]
            if w["enc"] != enc or w["shape"] != shape or got != ([] if slot is None else [slot]):
                raise SystemExit(f"{preset} {name}: build() wrote {w['enc']} {w['shape']} from {got}, "
                                 f"the table says {enc} {shape} from {slot}")
            slot_txt = "-" if slot is None else f"{slot:#x}"
            rows.append((region, slot_txt, w["enc"], w["off"], w["n"], w["shape"], note))
        result[name] = rows
    return result, dither_n, blobs


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
    enc_dec_sections(md)


ENC_DEC_INTRO = (
    "sr_pack.build() writes enc0.bin and dec0.bin for each preset. This map is recorded from build() itself: "
    "its slot counts are the packer's own checks, the slots are zero-filled to those counts (slot 0x200 with the "
    "16 values the packer reads past its end), and the dither is zero-filled to the size build() leaves for it. "
    "pack_layer(ENC0L) and pack_layer(DEC0L) are stage-like (C = 32) and use the stage labels; their slot ids are "
    "absolute (base 0x120 for ENC0L, 0xe40 for DEC0L). Semantics are not decoded unless sr_pack names them.")


def enc_dec_sections(md):
    print("\n## enc0.bin and dec0.bin (presets M and L)\n\n" + ENC_DEC_INTRO if md
          else "\nenc0.bin and dec0.bin (presets M and L)\n" + ENC_DEC_INTRO)
    for preset in ("M", "L"):
        rows_by, _, blobs = enc_dec_rows(preset)
        for name in ("enc0", "dec0"):
            size = len(blobs[name])
            title = f"{preset}: {name}.bin, {size} bytes"
            if md:
                print(f"\n### {title}\n")
                print("| region | slot | encoding | offset | bytes | shape | note |\n|---|---|---|---:|---:|---|---|")
            else:
                print(f"\n{title}")
            for region, slot, enc, off, n, shape, note in rows_by[name]:
                if md:
                    print(f"| {region} | {slot} | {enc} | {off} | {n} | {shape} | {note} |")
                else:
                    print(f"  {off:7d} {n:7d}  {enc:16s} {shape:9s} {region:22s} {note}")
        if md:
            print("\nNot in dec0.bin: slots 0xf60, 0xf70 and 0xf80 are read after the blob is written. "
                  "params.json holds dec0_s0 = min(exp(f16 0xf60[0]), 8), dec0_f0 = exp(0xf70[0]) and "
                  "dec0_f1 = exp(0xf80[0]) as f32. Preset M checks their counts; preset L reads them unchecked.")


if __name__ == "__main__":
    main()
