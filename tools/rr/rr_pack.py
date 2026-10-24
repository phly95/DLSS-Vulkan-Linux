#!/usr/bin/env python3
"""Build the DLSS-RR (preset F) GPU weight blobs exactly as NeuralHub v0.8.0 does, from the files
`rr_weights.py --out` writes (docs/rr/HOST-RE.md):

- FUN_18012ba70 turns the F table into 16 per-unit structs W[k] (k = enc0..enc5, dec4_0, dec4_1, ...,
  dec0_1): fp8 matrices, f16 vectors and scalars;
- FUN_180131f80 packs each W[k] into one blob (16-byte aligned items) and records the item offsets.

    rr_pack.py WEIGHTS_DIR OUT_DIR

writes OUT_DIR/blob{0..15}.bin, bicubic.bin and params.json (per-unit header, item offsets, scalars).
Nothing this writes may be committed: the values are NVIDIA's.
"""
import json
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "sr"))
from sr_pack import f16_to_f32, to_e4m3  # noqa: E402  (FUN_18012dda0 is the same conversion)

BLOCK = 0x238
C_OF = [32, 64, 96, 128, 160, 160, 160, 128, 96, 64, 32]        # DAT_1801bacc0, per level block L
G_OF = [2, 2, 4, 4, 8, 8, 8, 4, 4, 2, 2]                        # DAT_1801bacf0
CDOWN = [64, 96, 128, 160, 160]                                 # DAT_1801bb420 (L = 1..4)
CUP = [160, 160, 128, 96, 64]                                   # DAT_1801bb440 ((k - 6) >> 1)
NAMES = ["enc0", "enc1", "enc2", "enc3", "enc4", "enc5", "dec4_0", "dec4_1", "dec3_0", "dec3_1",
         "dec2_0", "dec2_1", "dec1_0", "dec1_1", "dec0_0", "dec0_1"]


class Table:
    def __init__(self, root: Path):
        self.root = root
        self.idx = {int(k, 16): v for k, v in json.loads((root / "index.json").read_text()).items()}
        self.raw = (root / "table.bin").read_bytes()
        self.written = (root / "written.bin").read_bytes()

    def f16(self, k, n):                                # FUN_18012dc40 (count check) + FUN_18012b900
        if self.idx.get(k, {}).get("count") != n:
            raise SystemExit(f"slot {k:#x} has {self.idx.get(k, {}).get('count')} elements, expected {n}")
        return np.fromfile(self.root / f"t_{k:04x}.f16", np.uint16)

    def u16(self, off):
        return struct.unpack_from("<H", self.raw, off)[0]


def mat(u16, rows, cols):                               # FUN_18012dda0: f16 -> e4m3, rows x cols
    return to_e4m3(f16_to_f32(u16[: rows * cols])).reshape(rows, cols)


def sym_scaled(u16, g):
    """FUN_18012dfc0: g 64x64 f16 matrices made symmetric, (m + m^T) * 0.5 * 2*sqrt(2), clamped to
    the f16 range and rounded to f16 (RNE)."""
    m = f16_to_f32(u16).reshape(g, 64, 64)
    s = (m + m.transpose(0, 2, 1)).astype(np.float32) * np.float32(0.5) * np.float32(2.8284270763397217)
    s = np.clip(s.astype(np.float32), np.float32(-65504.0), np.float32(65504.0))
    return [x.astype(np.float16).view(np.uint16).ravel() for x in s]


def perm_cols(m, C):                                    # FUN_18012e4b0: column j <- j / C + (j % C) * 4
    j = np.arange(4 * C)
    return m[:, j // C + (j % C) * 4]


def perm_rows(m, C):                                    # FUN_18012e5e0
    j = np.arange(4 * C)
    return m[j // C + (j % C) * 4]


def perm_vec(v, C):                                     # FUN_18012e6f0
    j = np.arange(4 * C)
    return v[j // C + (j % C) * 4]


def build_w(T: Table, k):
    """FUN_18012ba70 for unit k: returns a dict of the W[k] members NeuralHub's packer reads."""
    L = k if k < 6 else ((k - 6) >> 1) + 6
    odd = (k & 1) if k > 6 else 0
    C, g = C_OF[L], G_OF[L]
    b = L * BLOCK
    o = b + odd * 0xA0
    W = dict(k=k, name=NAMES[k], C=C, g=g, L=L, odd=odd, Cdown=0, Cup=0)
    t50 = mat(T.f16(o + 0x50, C * g * 64), g * 64, C)
    t60 = T.f16(o + 0x60, C * g * 32)
    t70 = T.f16(o + 0x70, C)
    t80 = T.f16(o + 0x80, g * 4096)
    ta0 = T.f16(o + 0xA0, 4 * C * C)
    tb0 = T.f16(o + 0xB0, 4 * C)
    tc0 = T.f16(o + 0xC0, 4 * C * C)
    td0 = T.f16(o + 0xD0, C)
    t1e0 = T.f16(b + 0x1E0, 1)
    W["groups"] = [np.concatenate([t50[i * 32:(i + 1) * 32], t50[(g + i) * 32:(g + i + 1) * 32]])
                   for i in range(g)]                                          # W+0x18: g x (64 x C)
    W["gram"] = sym_scaled(t80, g)                                             # W+0x30: g x 64x64 f16
    W["m48"] = mat(t60, C, g * 32)                                             # W+0x48
    W["v128"] = t70                                                            # W+0x128
    W["m68"] = mat(ta0, 4 * C, C)                                              # W+0x68
    W["v140"] = tb0                                                            # W+0x140
    W["m88"] = mat(tc0, C, 4 * C)                                              # W+0x88
    W["v158"] = td0                                                            # W+0x158
    W["s1d2"] = T.u16(b + odd * 4 + 0x220)
    W["s1d4"] = T.u16(b + odd * 4 + 0x222)
    W["s1d0"] = int(t1e0[0])
    if k < 6:
        if 1 <= k <= 4:
            cd = CDOWN[L]
            W["Cdown"] = cd
            W["mA8"] = perm_cols(mat(T.f16(b + 0x180, 4 * C * cd), cd, 4 * C), C)   # W+0xa8
            W["v170"] = T.f16(b + 0x190, cd)                                        # W+0x170
        if k == 1:
            W["mC8"] = perm_cols(mat(T.f16(0x180, 0x2000), 0x40, 0x80), 0x20)        # W+0xc8
            W["v188"] = T.f16(0x190, 0x40)                                          # W+0x188
        W["m108"] = mat(T.f16(b + 0x1C0, 3 * C), 3, C)                              # W+0x108
        W["v1b8"] = T.f16(b + 0x1D0, 3)                                             # W+0x1b8
    else:
        cu = CUP[(k - 6) >> 1]
        W["Cup"] = cu
        W["s1d6"] = T.u16(b + 0x228)
        if odd == 0:
            W["mE8"] = perm_rows(mat(T.f16(b + 0x1A0, 4 * C * cu), 4 * C, cu), C)   # W+0xe8
            W["v1a0"] = perm_vec(T.f16(b + 0x1B0, 4 * C), C)                        # W+0x1a0
        if L != 10:
            W["m108"] = mat(T.f16(b + 0x1C0, 4 * C), 4, C)
            W["v1b8"] = T.f16(b + 0x1D0, 4)
    if k == 15:
        e = [float(np.exp(f16_to_f32(T.f16(a, 1))[0].astype(np.float64)).astype(np.float32))
             for a in (0x1820, 0x1830, 0x1840)]
        W["dec0_scalars"] = [min(e[0], 8.0), e[1], e[2]]                           # W[0x92..0x94], f32
    if k == 0:
        e0 = mat(T.f16(0, 0xAC0), 0x40, 0x2B)
        pad = np.zeros((0x40, 0x40), np.uint8)
        pad[:, :0x2B] = e0
        W["m1d8"] = pad                                                             # W+0x1d8: 64 x 64
        W["v218"] = T.f16(0x10, 0x40)
        W["m1f8"] = mat(T.f16(0x20, 0x800), 0x20, 0x40)
        W["v230"] = T.f16(0x30, 0x20)
    return W


def frag_rr(m, permute_rows=True):
    """FUN_180133070(blob, mat, 1, 'A', permute_rows): rows padded to a multiple of 16, every element
    placed in the RDNA4 WMMA fragment order."""
    rows, cols = m.shape
    r16 = (rows + 15) // 16 * 16
    if r16 != rows:
        m = np.vstack([m, np.zeros((r16 - rows, cols), np.uint8)])
        rows = r16
    r = np.arange(rows, dtype=np.int64)[:, None]
    c = np.arange(cols, dtype=np.int64)[None, :]
    u = (((r & 0x7FFFFFE1) | ((r >> 3) & 2)) + (r & 0xE) * 2) if permute_rows else r
    hi = (((4 * c) & 0x10) | (u & 0xF)) << 4
    lo = ((c & 8) | ((16 * c) & 0xFFFFFFE00) | ((2 * c) & 4) | (c & 1) | ((c >> 3) & 2)) + cols * 16 * (u >> 4)
    off = (hi + lo).ravel()
    out = np.zeros(rows * cols, np.uint8)
    if len(np.unique(off)) != off.size or off.max() >= off.size:
        raise SystemExit(f"fragment layout is not a permutation for {rows}x{cols}")
    out[off] = m.ravel()
    return out.tobytes()


def f16_rr(v):
    """FUN_1801336e0(blob, vec, 1, 1): 32-element groups permuted when the size allows (as SR's)."""
    v = np.asarray(v, np.uint16)
    n = v.size
    if (2 * n) & 0x3E:
        return v.tobytes()
    i = np.arange(n, dtype=np.int64)
    byte = (i & 0x1C) + ((8 * i) & 0x10) * 2 + (i & 0x7FFFFFFFFFFFFFE1) * 2
    return v[byte // 2].tobytes()


class Blob:
    def __init__(self):
        self.b = bytearray()

    def align(self):
        self.b += bytes(-len(self.b) % 16)

    def put(self, data):
        off = len(self.b)
        self.b += data
        return off


def pack(W, dither):
    """FUN_180131f80: one unit's blob and its 21 item offsets (-1 = absent)."""
    B = Blob()
    meta = [-1] * 21
    for i, gm in enumerate(W["groups"]):
        if i:
            B.align()
        off = B.put(frag_rr(gm))
        if i == 0:
            meta[0] = off
    B.align()
    meta[1] = len(B.b)
    for gr in W["gram"]:
        B.put(gr.tobytes())
    B.align()
    meta[2] = B.put(frag_rr(W["m48"]))
    B.align()
    meta[3] = B.put(f16_rr(W["v128"]))
    B.align()
    meta[4] = B.put(frag_rr(W["m68"]))
    B.align()
    meta[5] = B.put(f16_rr(W["v140"]))
    B.align()
    meta[6] = B.put(frag_rr(W["m88"]))
    B.align()
    meta[7] = B.put(f16_rr(W["v158"]))
    for mi, vi, mk, vk in ((8, 9, "mA8", "v170"), (10, 11, "mC8", "v188"), (12, 13, "mE8", "v1a0")):
        if mk in W:
            B.align()
            meta[mi] = B.put(frag_rr(W[mk]))
            B.align()
            meta[vi] = B.put(f16_rr(W[vk]))
    if "m108" in W:
        B.align()
        meta[14] = B.put(frag_rr(W["m108"], permute_rows=False))
        head = np.zeros(16, np.uint16)
        head[: min(16, W["v1b8"].size)] = W["v1b8"][:16]
        B.align()
        meta[15] = B.put(head.tobytes())
    if W["k"] == 0:
        B.align()
        meta[16] = B.put(frag_rr(W["m1d8"]))
        B.align()
        meta[17] = B.put(f16_rr(W["v218"]))
        B.align()
        meta[18] = B.put(frag_rr(W["m1f8"]))
        B.align()
        meta[19] = B.put(f16_rr(W["v230"]))
        B.align()
        meta[20] = B.put(dither)
    B.align()
    return bytes(B.b), meta


def main():
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    T = Table(src)
    dither = (src / "dither.bin").read_bytes()
    params = dict(units=[])
    for k in range(16):
        W = build_w(T, k)
        blob, meta = pack(W, dither)
        (out / f"blob{k}.bin").write_bytes(blob)
        params["units"].append(dict(
            k=k, name=W["name"], C=W["C"], g=W["g"], L=W["L"], odd=W["odd"], Cdown=W["Cdown"], Cup=W["Cup"],
            bytes=len(blob), meta=meta, s1d0=W["s1d0"], s1d2=W["s1d2"], s1d4=W["s1d4"], s1d6=W.get("s1d6", 0),
            dec0_scalars=W.get("dec0_scalars")))
        print(f"{W['name']:7s} C={W['C']:3d} g={W['g']} {len(blob):#8x} bytes", file=sys.stderr)
    # W+0x2598: the f16 pair at table 0x185a with its halves swapped; W+0x259c: 8 bytes at table 0x230
    params["enc0_s2598"] = struct.unpack("<I", T.raw[0x185A:0x185E])[0]
    params["enc0_s2598"] = ((params["enc0_s2598"] & 0xFFFF) << 16) | (params["enc0_s2598"] >> 16)
    params["enc0_s259c"] = T.raw[0x230:0x238].hex()
    (out / "bicubic.bin").write_bytes((src / "bicubic.bin").read_bytes())
    (out / "params.json").write_text(json.dumps(params, indent=1))


if __name__ == "__main__":
    main()
