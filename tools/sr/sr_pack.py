#!/usr/bin/env python3
"""Build the DLSS-SR GPU weight blobs exactly as NeuralHub v0.8.0 does (FUN_1801091c0 and
FUN_180111850 of neuralhub_host.dll; docs/sr/HOST-RE.md), from the slot files `sr_weights.py --out`
writes.

    sr_pack.py WEIGHTS_DIR {M|L} OUT_DIR

writes OUT_DIR/stage{0..9}.bin, enc0.bin, dec0.bin and params.json (per-stage fp8 bound flags and
the decoder scalars). Every blob size is checked against the size NeuralHub requires. Nothing this
writes may be committed: the values are NVIDIA's.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

LAYERS = [  # name, slot base, C, g, Cnext, mode (layer table at 0x1801b9c00)
    ("enc1", 0x270, 64, 2, 96, 0),
    ("enc2", 0x3C0, 96, 4, 128, 0),
    ("tube0", 0xF90, 128, 4, 0, 1),
    ("tube1", 0x10E0, 128, 4, 0, 1),
    ("tube2", 0x1230, 128, 4, 0, 1),
    ("tube3", 0x1380, 128, 4, 0, 1),
    ("tube4", 0x14D0, 128, 4, 0, 1),
    ("tube5", 0x1620, 128, 4, 0, 1),
    ("dec2", 0xBA0, 96, 4, 128, 2),
    ("dec1", 0xCF0, 64, 2, 96, 2),
]
ENC0L = ("enc0L", 0x120, 32, 2, 64, 0)
DEC0L = ("dec0L", 0xE40, 32, 2, 64, 2)
STAGE_BYTES = [0x118C0, 0x27F00] + [0x2D000] * 6 + [0x28100, 0x11A00]
ENC0_BYTES = {"M": 0x3080, "L": 0x7700}
DEC0_BYTES = {"M": 0x2100, "L": 0x5F00}
FP8_MAX = np.float32(448.0)
LAYOUT = 1  # FUN_1801091c0's second argument: both call sites pass 1


class Slots:
    def __init__(self, root: Path):
        self.root = root
        self.idx = {int(k, 16): v for k, v in json.loads((root / "index.json").read_text()).items()}

    def count(self, k):
        return self.idx[k]["count"] if k in self.idx else 0

    def f16(self, k, pad=0):
        """FUN_180113b50: count + pad values from the DLL. `pad` reads past the slot, into the next
        one: sr_weights.py writes each slot alone, so the next slot in the DLL's address order
        supplies them."""
        a = np.fromfile(self.root / f"slot_{k:04x}.f16", np.uint16)[: self.idx[k]["count"]]
        if pad:
            rva = self.idx[k]["rva"] + 2 * self.idx[k]["count"]
            nxt = next(s for s, v in self.idx.items() if v["rva"] == rva)
            a = np.concatenate([a, np.fromfile(self.root / f"slot_{nxt:04x}.f16", np.uint16)[:pad]])
        return a


def f16_to_f32(u16):
    return np.asarray(u16, np.uint16).view(np.float16).astype(np.float32)


def to_e4m3(x32):
    """FUN_180114320's f32 -> OCP e4m3: |x| clamped to 448, RNE by bit arithmetic on normals,
    nearbyint(|x| * 512) on subnormals (|x| < 2^-6), NaN -> 0x7f."""
    x32 = np.asarray(x32, np.float32)
    bits = x32.view(np.uint32)
    sign = ((bits >> 24) & 0x80).astype(np.uint8)
    a = np.minimum(np.abs(x32), FP8_MAX)
    ab = a.view(np.uint32).astype(np.uint64)
    normal = ((ab + 0x407FFFF + ((ab >> 20) & 1)) >> 20).astype(np.uint8)
    sub = np.rint(a.astype(np.float64) * 512.0).astype(np.int64).astype(np.uint8)
    out = np.where(a >= np.float32(0.015625), sign | normal, sub | sign).astype(np.uint8)
    return np.where(np.isnan(x32), np.uint8(0x7F), out)


def e4m3_to_f32(b):
    b = np.asarray(b, np.uint8).astype(np.int64)
    e = (b >> 3) & 0xF
    m = (b & 7).astype(np.float64)
    v = np.where(e == 0, m * 2.0 ** -9, (m * 0.125 + 1.0) * np.exp2(e - 7.0))
    v = np.where(b & 0x80, -v, v)
    return np.where((~b & 0x7F) == 0, np.nan, v).astype(np.float32)


class Mat:
    """NeuralHub's fp8 matrix: rows, cols, row-major bytes."""

    def __init__(self, rows, cols, data):
        self.rows, self.cols = rows, cols
        self.data = np.asarray(data, np.uint8).reshape(rows, cols)

    @staticmethod
    def from_f16(u16, rows, cols):                      # FUN_180114320
        return Mat(rows, cols, to_e4m3(f16_to_f32(u16[: rows * cols])))

    def row_slice(self, start, n):                      # FUN_1801164b0
        return Mat(n, self.cols, self.data[start:start + n])

    def col_slice(self, start, n):                      # FUN_1801166d0
        return Mat(self.rows, n, self.data[:, start:start + n])

    def vcat(self, other):                              # FUN_1801163a0
        return Mat(self.rows + other.rows, self.cols, np.concatenate([self.data, other.data]))


def frag_offsets(rows, cols, layout=LAYOUT, op="A"):
    """FUN_180113e10: destination byte of element (r, c) in the WMMA fragment order."""
    r = np.arange(rows, dtype=np.int64)[:, None]
    c = np.arange(cols, dtype=np.int64)[None, :]
    if layout == 0:
        base = (cols << 5) * (r >> 5)
        if op == "A":
            base = base + (c >> 5) * 0x200 + (cols << 4) * ((r & 0x10) >> 4)
        else:
            base = base + (c >> 5) * 0x400 + ((r & 0x10) << 5)
        u3 = c >> 1
        return base + ((((u3 & 3) | ((r * 4) & 0x1C)) << 4) | (u3 & 4) | ((c >> 3) & 2) | (c & 1) | (r & 8))
    u1 = ((r & 0x7FFFFFE1) | ((r >> 3) & 2)) + (r & 0xE) * 2
    u5 = 2 * c
    u9 = 16 * c
    low = (u5 & 4) | (c & 1) | ((c >> 3) & 2)
    high = ((u5 & 0x10) << 4) + (u9 & 0xFFFFFFE00) + (cols << 4) * (u1 >> 4)
    return (low | high) + ((u1 & 0xF) + (u5 & 8) * 2) * 8


class Blob:
    def __init__(self):
        self.b = bytearray()

    def frag(self, m: Mat, op="A"):                     # FUN_180113e10
        off = frag_offsets(m.rows, m.cols, LAYOUT, op)
        out = np.zeros(m.rows * m.cols, np.uint8)
        out[off.ravel()] = m.data.ravel()
        self.b += out.tobytes()

    def f16(self, u16):                                 # FUN_1801140a0
        u16 = np.asarray(u16, np.uint16)
        n = u16.size
        if LAYOUT == 0 or (2 * n) & 0x3E:
            self.b += u16.tobytes()
            return
        i = np.arange(n, dtype=np.int64)
        byte = (i & 0x1C) + ((8 * i) & 0x10) * 2 + (i & 0x7FFFFFFFFFFFFFE1) * 2
        self.b += u16[byte // 2].tobytes()

    def f16_block(self, u16):                           # FUN_180116540: a 16x16 f16 tile
        src = np.asarray(u16, np.uint16)[:256]
        if LAYOUT != 0:
            self.b += src.tobytes()
            return
        out = np.zeros(256, np.uint16)
        for r in range(16):
            for c in range(16):
                out[(c & 1) | ((r >> 2) & 2) | ((r & 7) << 5) | ((4 * c) & 0x18) | ((c >> 1) & 4)] = src[16 * r + c]
        self.b += out.tobytes()

    def raw(self, b):
        self.b += bytes(b)


def scale_f16(u16, k):
    """FUN_180115ee0 (k = sqrt(C)) and FUN_180116130 (two products): f32 product rounded to f16."""
    x = f16_to_f32(u16)
    for f in k:
        x = (x * np.float32(f)).astype(np.float32)
    return x.astype(np.float16).view(np.uint16)


def pack_layer(S: Slots, layer):
    """FUN_180111850: one encoder / tube / decoder layer. Returns (blob bytes, f32 bound)."""
    name, b, C, g, Cn, mode = layer
    need = {0x40: C, 0x50: C * g * 64, 0x60: C * g * 32, 0x70: C, 0x80: g * 256, 0x90: C,
            0xA0: 4 * C * C, 0xB0: 4 * C, 0xC0: 4 * C * C, 0xD0: C}
    if mode == 0:
        need.update({0xE0: Cn * 4 * C, 0xF0: Cn})
    if mode == 2:
        need.update({0x100: Cn * 4 * C, 0x110: 4 * C})
    for off, n in need.items():
        if S.count(b + off) != n:
            raise SystemExit(f"{name}: slot {b + off:#x} has {S.count(b + off)} elements, expected {n}")
    w210 = Mat.from_f16(S.f16(b + 0x50), g * 64, C)
    w1f0 = Mat.from_f16(S.f16(b + 0x60), C, g * 32)
    w180 = Mat.from_f16(S.f16(b + 0xA0), 4 * C, C)
    w1d0 = Mat.from_f16(S.f16(b + 0xC0), C, 4 * C)
    s1b0 = scale_f16(S.f16(b + 0x40), [math.sqrt(C)])
    s198 = scale_f16(S.f16(b + 0x90), [math.sqrt(C)])
    v270 = S.f16(b + 0x70)
    v258 = scale_f16(S.f16(b + 0x80), [np.float32(5.656854152679443), np.float32(0.5)])
    v228 = S.f16(b + 0xB0)
    v240 = S.f16(b + 0xD0)

    # the fp8 range bound (param_6): max over |gains| and every row's sum |gain| * |w|
    a1b0 = np.abs(f16_to_f32(s1b0)).astype(np.float64)
    a198 = np.abs(f16_to_f32(s198)).astype(np.float64)
    bound = max(a1b0.max(), a198.max(), 0.0)
    bound = max(bound, (np.abs(e4m3_to_f32(w210.data)).astype(np.float64) @ a1b0).max())
    rows = np.abs(f16_to_f32(v228)).astype(np.float64) + np.abs(e4m3_to_f32(w180.data)).astype(np.float64) @ a198
    bound = max(bound, rows.max())
    bound = np.float32(bound * 1.05)

    out = Blob()
    if mode == 2:
        out.frag(Mat.from_f16(S.f16(b + 0x100), 4 * C, Cn))
        out.f16(S.f16(b + 0x110))
    out.f16(s1b0)
    for h in range(g):
        out.frag(w210.row_slice(h * 32, 32).vcat(w210.row_slice(h * 32 + g * 32, 32)), "B")
        out.f16_block(v258[h * 256:(h + 1) * 256])
        out.frag(w1f0.col_slice(h * 32, 32))
    out.f16(v270)
    out.f16(s198)
    for j in range(C // 8):
        out.frag(w180.row_slice(j * 32, 32))
        out.f16(v228[j * 32:j * 32 + 32])
        out.frag(w1d0.col_slice(j * 32, 32))
        if j == 0:
            out.f16(v240)
    if mode == 0:
        out.frag(Mat.from_f16(S.f16(b + 0xE0), Cn, 4 * C))
        out.f16(S.f16(b + 0xF0))
    return bytes(out.b), bound


def build(S: Slots, preset: str, dither: bytes):
    res = {"preset": preset, "stages": []}
    blobs = {}
    for i, layer in enumerate(LAYERS):
        blob, bound = pack_layer(S, layer)
        if len(blob) != STAGE_BYTES[i]:
            raise SystemExit(f"{layer[0]}: unexpected blob size {len(blob):#x} (NeuralHub: {STAGE_BYTES[i]:#x})")
        blobs[f"stage{i}"] = blob
        res["stages"].append(dict(name=layer[0], bound=float(bound), fp8_safe=bool(bound < FP8_MAX)))
    enc0 = Blob()
    enc0.raw(dither)
    if preset == "L":
        for k, n in ((0x120, 0x280), (0x130, 0x20), (0x140, 0x400), (0x150, 0x20)):
            if S.count(k) != n:
                raise SystemExit("L enc0 embedding slots have unexpected sizes")
        e = f16_to_f32(S.f16(0x120)).reshape(32, 20)
        m = np.zeros((32, 32), np.uint8)
        m[:, :20] = to_e4m3(e)
        enc0.frag(Mat(32, 32, m))
        enc0.f16(S.f16(0x130))
        enc0.frag(Mat.from_f16(S.f16(0x140), 32, 32))
        enc0.f16(S.f16(0x150))
        blob, bound = pack_layer(S, ENC0L)
        enc0.raw(blob)
        res["enc0L"] = dict(bound=float(bound), fp8_safe=bool(bound < FP8_MAX))
        blob, bound = pack_layer(S, DEC0L)
        dec0 = Blob()
        dec0.raw(blob)
        res["dec0L"] = dict(bound=float(bound), fp8_safe=bool(bound < FP8_MAX))
    else:
        if S.count(0x200) != 0x1400 or S.count(0x210) != 0x40:
            raise SystemExit("enc0 slots have unexpected sizes")
        w = f16_to_f32(S.f16(0x200, pad=0x10))
        r = np.arange(64)[:, None]
        c = np.arange(128)[None, :]
        enc0.frag(Mat(64, 128, to_e4m3(w[(c & 0x1F) + r * 0x50 + (c >> 5) * 0x14])))
        enc0.f16(S.f16(0x210))
        for k, n in ((0xF40, 0x2000), (0xF50, 0x80), (0xF60, 1), (0xF70, 1), (0xF80, 1)):
            if S.count(k) != n:
                raise SystemExit("dec0 slots have unexpected sizes")
        dec0 = Blob()
        dec0.frag(Mat.from_f16(S.f16(0xF40), 0x80, 0x40))
        dec0.f16(S.f16(0xF50))
    # FUN_180114540: exp() of the slot's first f16
    sc = [float(np.exp(np.float64(f16_to_f32(S.f16(k)[:1])[0]))) for k in (0xF60, 0xF70, 0xF80)]
    res["dec0_int"] = int(min(np.float32(sc[0]), np.float32(8.0)))
    res["dec0_f0"] = float(np.float32(sc[1]))
    res["dec0_f1"] = float(np.float32(sc[2]))
    blobs["enc0"], blobs["dec0"] = bytes(enc0.b), bytes(dec0.b)
    if len(blobs["enc0"]) != ENC0_BYTES[preset] or len(blobs["dec0"]) != DEC0_BYTES[preset]:
        raise SystemExit(f"enc0/dec0: unexpected blob size {len(blobs['enc0']):#x} / {len(blobs['dec0']):#x}")
    return blobs, res


def main():
    if len(sys.argv) != 4 or sys.argv[2] not in ("M", "L"):
        raise SystemExit(__doc__)
    root, preset, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
    dither = (root / "dither.bin").read_bytes()
    blobs, res = build(Slots(root / preset), preset, dither)
    out.mkdir(parents=True, exist_ok=True)
    for k, v in blobs.items():
        (out / f"{k}.bin").write_bytes(v)
    (out / "params.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
