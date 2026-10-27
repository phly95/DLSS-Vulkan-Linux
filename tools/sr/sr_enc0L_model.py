#!/usr/bin/env python3
"""tools/sr/sr_enc0L_model.py: the NumPy spec of NeuralHub's k_sr_enc0L_in (preset L, oracle
launch 2 of sr_L). Companion of tools/sr/sr_enc0_model.py (preset M, kernel k_sr_enc0).

    build/venv/bin/python -I tools/sr/sr_enc0L_model.py [--frame F] [--oracle DUMP] [--out DIR]
    build/venv/bin/python -I tools/sr/sr_enc0L_model.py --self-check        # L-1, < 60 s

===================================================================================================
SPEC  (every formula, constant, rounding point and buffer encoding; kernel: enc0L_in.s ->
enc0L_in.ssa in build/agents/enc0L-in/, the ISA listing in execution order 0x11000..0x1497c.
Enc0Args layout: tools/sr/srref/srref.cpp, block "enc0 (FUN_180116e70)" - the SAME 280-byte
struct as preset M plus the l0_in pointer at 0x118 (288 B total, kernel signature `Enc0Args Ph`).
===================================================================================================

STAGE TABLE  (all rows measured vs the sr_L oracle dump, both frames; "probe" = this model's
--self-check bit-exactness against 002.l0_in.bin / the three MV dumps)
| stage                      | content                                                       |
|----------------------------|---------------------------------------------------------------|
| 1  coords (0x11014..0x111d0)| one thread per PADDED pixel: r = idx/(W0+8), c = idx%(W0+8);  |
|                            | yref = mirror(r-2, H), xref = mirror(c-2, W) (mirror =        |
|                            | |v| < n ? |v| : 2n-2-|v|, clamped [0, n-1]; the reflect is      |
|                            | applied when either axis is out of range, one combined mask)  |
| 2  colour 0..2 (0x11250..) | v = e * g_col[yref, xref, 0:3]; tm (as M section 3)           |
| 3  dhist 3..14             | four bilinear dhist samples x R,G,B (as M section 4), base    |
|                            | (xref+0.5, yref+0.5), OOB -> the colour feature, x2.45        |
| 4  motion energy 15        | as M section 5 with k = Args[0xe8] (L: 19.266386032104492 =   |
|                            | 2774.359619140625/H) and SC = Args[0xd0] (L: 0.08225896209478378)|
| 5  hidden 16..19           | the 4 int8 hidden0 channels warped at the sample-0 position   |
|                            | (as M section 6; L reads hidden0 in BOTH frames)              |
| 6  dq                      | M section 2, dither table = enc0L.bin[0:4096) (byte-identical  |
|                            | to M's), index phase = |coord-2| (&15) - i.e. the level-0     |
|                            | coordinate except the top/left 2-pixel border, where the abs   |
|                            | before the reflect is what the kernel uses                    |
| 7  store (0x14960)         | 32 bytes/pixel at l0_in + 32*(r*264 + c); stored byte j =     |
|                            | logical feature P[j] (P of sr_stage_model; logical 20..31 =   |
|                            | 0, never written - verified 100% vs the oracle)               |
| 8  MV outs (0x11c80..)     | M section 7 verbatim: prev_mvec[p^1], mv_ero, mv_nondil =     |
|                            | f16(3.0 * (mvoff + g_mv[3x3-tournament])), reset select,      |
|                            | stored for level-0 y in [0, H) and x in [0, W)                |
| probe l0_in                | 100.0000% frame0, 100.0000% frame1 (content AND the 2-pixel   |
|                            | mirrored border: the whole padded grid is a mirror-extension   |
|                            | of M's 160x256 feature grid); the three MV outputs 100%       |

DIFFERENCES vs PRESET M (all verified field-by-field against both oracle dumps):
  - output: t0 (128x80x64 embed) -> l0_in (264x168x32 = (W0+8)x(H0+8) features, P-stored);
    no GEMM at all in the kernel (no WMMA in the ISA) - the W1/W2 embed layers live in
    k_sr_l0<0,2> (launch 3) and read l0_in.
  - grid: (W0/8, H0/8) x 32 threads -> ((W0+8)*(H0+8)+63)/64 x 64 (one thread per padded pixel).
  - hidden input: hidden[p] -> hidden0 in both frames (args 0x28 = hidden0 on f0 AND f1).
  - feature grid: 160x256 (y mirrored for reads) -> 168x264 with a 2-pixel MIRROR BORDER on
    all four sides (r-2/c-2 reflected about the render edges); the dither phase stays the
    level-0 coordinate.
  - constants 0xcc/0xd0/0xe8 are preset-specific (L: 5.137702941894531 / 0.08225896209478378 /
    19.266386032104492); everything else in Enc0Args is identical to M (dword-compared).
  - wbuf+1577728 = build/re/packed/L/enc0.bin (30464 B); only its dither table [0,4096) is
    read here (W1/b1/W2/b2/ENC0L belong to launch 3).
  - the t0 pointer at 0x48 is carried (srref puts t[0].d there) but NEVER TOUCHED: the
    kernel's complete s_load inventory (enc0L_in.s, every args access is an s_load from the
    kernarg base s[0:1]) covers offsets 0x0..0x47, 0x50..0x113 and 0x118..0x11f - 0x48..0x4f
    is in no load, and the kernel has no t0 store (the dump's wrote list = prev_mvec*, mv_ero,
    mv_nondil, l0_in).  l-rewire's "t0 read" note (progress.md item 3) came from the dump's
    pointer listing, which lists every non-null args slot - presence is not a read.
  - MV outputs, their reset select, the 3x3 depth tournament, the dq, tm, the dhist/hidden
    sample math: identical to M (measured 100% on both frames).
"""
import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sr_stage_model as m  # noqa: E402

f32 = np.float32
f16 = m.f16
KERNEL = "_ZN2sr2io13k_sr_enc0L_inENS0_8Enc0ArgsEPh"
REPO = m.REPO
WEIGHTS = REPO / "build/re/packed/L"
NFEAT = 20           # logical features stored in l0_in (logical 20..31 are zero)
NPAD = 8             # the padded grid is (W0+8) x (H0+8)


# ------------------------------------------------------------------ inputs
def parse_args(b):
    """Enc0Args (288 B: M's 280 B + l0_in at 0x118), every field the kernel reads."""
    a = {}
    a["W"], a["H"] = struct.unpack_from("<2i", b, 0x58)
    a["W0"], a["H0"] = struct.unpack_from("<2i", b, 0x68)
    a["W15"], a["H15"] = struct.unpack_from("<2i", b, 0x70)
    a["hW15"], a["hH15"] = struct.unpack_from("<2i", b, 0x78)
    a["p"] = struct.unpack_from("<I", b, 0x88)[0]
    a["F20x"], a["F20y"] = struct.unpack_from("<2f", b, 0x8c)
    a["F28x"], a["F28y"] = struct.unpack_from("<2f", b, 0x94)
    a["mvoffx"], a["mvoffy"] = struct.unpack_from("<2f", b, 0xbc)   # 0 for unjittered MVs
    a["reset"] = struct.unpack_from("<I", b, 0xd8)[0]
    a["fc"] = struct.unpack_from("<I", b, 0xdc)[0]
    a["SC"] = struct.unpack_from("<f", b, 0xd0)[0]                  # L: 0.08225896209478378
    a["k"] = struct.unpack_from("<f", b, 0xe8)[0]                   # L: 19.266386032104492 = 2774.36/H
    return a


def load(dump, frame, launch=2):
    """The launch's args and its input buffers as they stood before the launch (oracle or GPU dump)."""
    fdir = Path(dump) / f"frame{frame}"
    rows = [json.loads(l) for l in open(fdir / "launches.jsonl")]
    d = next(r for r in rows if r["n"] == launch and r["kernel"] == KERNEL)
    names = {int(k): v.split("+")[0] for k, v in d["pointers"].items()}
    a = parse_args(bytes.fromhex(d["args"]))
    H, W = a["H"], a["W"]
    buf = lambda off: m.buffer_before(fdir, names[off], launch)
    g = {}
    g["g_col"] = np.frombuffer(buf(0), np.uint16)[: W * H * 4].reshape(H, W, 4)
    g["g_dep"] = np.frombuffer(buf(8), np.float32)[: W * H].reshape(H, W)
    g["g_mv"] = np.frombuffer(buf(16), np.float16)[: W * H * 2].reshape(H, W, 2)
    g["prev_mvec_p"] = np.frombuffer(buf(24), np.float16)[: W * H * 2].reshape(H, W, 2)
    g["dhist"] = np.frombuffer(buf(32), np.uint32)[: a["hW15"] * a["hH15"]].reshape(a["hH15"], a["hW15"])
    g["hidden_p"] = np.frombuffer(buf(40), np.int8)[: W * H * 4].reshape(H, W, 4)
    expo = np.frombuffer(buf(128), np.uint16)[:2]
    # the kernel reads expo[p] as f16 and uses 1.0 when that is zero
    e = expo[a["p"]]
    a["e"] = f32(1.0) if e == 0 else f16(e.view(np.float16)).astype(f32)
    a["dither"] = np.frombuffer((WEIGHTS / "enc0.bin").read_bytes()[:4096], np.uint16)
    return a, g


# ------------------------------------------------------------------ padded-grid coordinates
def mirror(v, n):
    """The kernel's reflect of one axis (prologue 0x11150..0x111d0): |v| < n ? |v| : 2n-2-|v|,
    clamped to [0, n-1].  Applied when either axis is out of range (one combined mask); on the
    in-range path the value is |v| itself, which equals the reflection, so the two paths agree."""
    av = np.abs(v)
    r = np.where(av < n, av, 2 * n - 2 - av)
    return np.clip(r, 0, n - 1)


def coords(a):
    """(rows, cols): the reflected row/col source indices for the (H0+8) x (W0+8) padded grid."""
    H, W = a["H"], a["W"]
    return mirror(np.arange(a["H0"] + NPAD) - 2, H), mirror(np.arange(a["W0"] + NPAD) - 2, W)


# ------------------------------------------------------------------ the feature quantizer (dq)
def dq(x, dither, fc, py, px):
    """Dithered E4M3 quantizer of a feature grid x (H0P x W0P, f32) - M's section 2 verbatim,
    with the dither index taken from the phase coordinates py/px = |coord-2| (the level-0
    coordinate, except the top/left 2-pixel border where the abs before the reflect is what
    the kernel uses: (r-2)&15 = 14,15 there vs |r-2|&15 = 2,1)."""
    x16 = f16(np.minimum(np.asarray(x, f32), f32(1024.0))).view(np.uint16)
    idx = ((fc & 7) << 8) | ((np.asarray(py)[:, None] & 15) * 16 + (np.asarray(px)[None, :] & 15))
    d = (dither[idx] >> 9).astype(np.uint16)
    E = x16 & np.uint16(0xFC00)
    K = (E | d).view(np.float16)
    aa = f16(x16.view(np.float16).astype(f32) + K.astype(f32))
    bb = f16(aa.astype(f32) - E.view(np.float16).astype(f32))
    t = (bb.view(np.uint16) & np.uint16(0xFF80)).view(np.float16).astype(f32)
    return q8r_code(np.clip(t, -448, 448))


def q8r_code(x):
    """The E4M3 CODE (u8) q8r maps to its value m.LUT[code]; l0_in stores codes."""
    x = np.asarray(x, np.float32)
    x = np.where(np.isnan(x), np.float32(0), x)
    sign = np.where(x < 0, 0x80, 0).astype(np.uint32)
    a = np.minimum(np.abs(x), np.float32(448.0))
    bits = a.view(np.uint32)
    s = a * np.float32(512.0)                       # exact: subnormal grid = 2^-9
    n = np.floor(s)
    d = s - n
    sub = np.where((d > np.float32(0.5)) | ((d == np.float32(0.5)) & ((n.astype(np.int64) & 1) == 1)),
                   n + np.float32(1), n)
    norm = ((bits + np.uint32(0x407ffff) + ((bits >> 20) & 1)) >> 20) & 0x7f
    return (np.where(a < np.float32(0.015625), sub, norm).astype(np.uint32) | sign).astype(np.uint8)


def q8r(x):
    """f32 -> fp8 VALUE with RNE (ties-to-even), saturating at +-448: m.LUT[q8r_code(x)]."""
    return m.LUT[q8r_code(x)]


# ------------------------------------------------------------------ features 0..2: colour
C_CLAMP = np.frombuffer(bytes.fromhex("0000fe46"), np.float32)[0]   # 0x46ffe000 = 32752
C_LO = np.frombuffer(bytes.fromhex("17b7d138"), np.float32)[0]      # 0x38d1b717 = 1e-4
C_LOG = np.frombuffer(bytes.fromhex("9b209a3e"), np.float32)[0]     # 0x3e9a209b = log10(2)
C_SCALE = np.frombuffer(bytes.fromhex("ec51b83e"), np.float32)[0]  # 0x3eb851ec = 0.36


def tm(v):
    """tm(v) = (log2(max(1 + 100 min(v, 32752), 1e-4)) * log10(2)) * 0.36, f32 steps (fma for
    1 + 100v); NaN -> 0.  Identical to sr_enc0_model.tm."""
    v = np.asarray(v, f32)
    vc = np.where(v > C_CLAMP, C_CLAMP, v).astype(f32)
    t = (np.float64(100) * vc.astype(np.float64) + 1.0).astype(f32)      # fma(100, v, 1)
    t = np.fmax(t, C_LO)
    r = (np.log2(t).astype(f32) * C_LOG).astype(f32)
    r = (r * C_SCALE).astype(f32)
    return np.where(np.isnan(v), f32(0), r).astype(f32)


def colour_features(a, g, rows, cols):
    """Logical features 0, 1, 2 = tm(e * rgb) of the reflected g_col texel, (H0P, W0P, 3)."""
    col = np.ascontiguousarray(g["g_col"][rows][:, cols][..., :3])
    rgb = col.view(np.float16).astype(f32)
    return tm((rgb * a["e"]).astype(f32))


# ------------------------------------------------------------------ features 3..14: four dhist samples
INV1023 = np.frombuffer(bytes.fromhex("0820803a"), np.float32)[0]   # 0x3a802008 = f32 1/1023
C245 = np.frombuffer(bytes.fromhex("cdcc1c40"), np.float32)[0]      # 0x401ccccd = 2.45


def _axis(C, hdim):
    """One axis of a dhist sample position (M's op sequence): (0.5/h)*C -> h*(...) round trip,
    [-2, h+2] clamp (NaN -> -0.5), snap to the 1/256 grid -> (tap, frac)."""
    a_ = (f32(0.5) / f32(hdim)) * C
    b_ = f32(hdim) * a_
    u = b_ - f32(0.5)
    u = np.maximum(u, f32(-2.0))
    u = np.where(np.isnan(u), f32(-0.5), u)
    u = np.minimum(u, f32(hdim + 2))
    s = np.floor(u * f32(256.0) + f32(0.5)) * f32(1.0 / 256.0)
    tap = np.floor(s)
    return tap, s - tap


def dhist_sample(Cx, Cy, chan, dh, hW, hH):
    """One bilinear dhist sample of channel `chan`, *2.45, f32 ops in the kernel's order."""
    tx, fx = _axis(Cx, hW)
    ty, fy = _axis(Cy, hH)
    ix, iy = tx.astype(np.int64), ty.astype(np.int64)
    x0 = np.clip(ix, 0, hW - 1)
    x1 = np.clip(np.maximum(ix, -1) + 1, 0, hW - 1)
    y0 = np.clip(iy, 0, hH - 1)
    y1 = np.clip(np.maximum(iy, -1) + 1, 0, hH - 1)
    C = ((dh >> np.uint32(10 * chan)) & np.uint32(0x3FF)).astype(f32)
    t00, t01 = C[y0, x0] * INV1023, C[y0, x1] * INV1023
    t10, t11 = C[y1, x0] * INV1023, C[y1, x1] * INV1023
    w00 = (f32(1) - fy) * (f32(1) - fx)
    w01 = (f32(1) - fy) * fx
    w10 = fy * (f32(1) - fx)
    w11 = fy * fx
    return (((w00 * t00 + w01 * t01) + w10 * t10) + w11 * t11) * C245


def mv_fields(a, g, rows, cols):
    """The kernel's g_mv reads on the padded grid, as f32 (H0P, W0P, 2) arrays = mvoff + g_mv[pos]:
      'min'/'max'/'center' = the 3x3 depth tournament's min/max/centre reads (t0..t8 = centre,
      (y-1,x-1), (y-1,x), (y-1,x+1), (y,x-1), (y,x+1), (y+1,x-1), (y+1,x), (y+1,x+1), rows/cols
      clamped to the image, ties keep the EARLIER element: max `mx >= t`, min `t < mn`);
      'prev_out'/'ero' = the reset select (reset ? (max,min) : (min,max))."""
    H, W = a["H"], a["W"]
    dep = g["g_dep"]
    mv = g["g_mv"].astype(f32)
    y = np.broadcast_to(rows[:, None].astype(np.int64), (a["H0"] + NPAD, a["W0"] + NPAD))
    x = np.broadcast_to(cols[None, :].astype(np.int64), (a["H0"] + NPAD, a["W0"] + NPAD))

    def idx(dy, dx):
        return np.clip(y + dy, 0, H - 1), np.clip(x + dx, 0, W - 1)

    pos = [idx(0, 0), idx(-1, -1), idx(-1, 0), idx(-1, 1),
           idx(0, -1), idx(0, 1), idx(1, -1), idx(1, 0), idx(1, 1)]
    d = [dep[p] for p in pos]
    t0_, t1 = d[0], d[1]
    ge = t0_ >= t1
    gt = t0_ > t1
    mxv, mnv = np.where(ge, t0_, t1), np.where(gt, t1, t0_)
    mxr, mxc = np.where(ge, pos[0][0], pos[1][0]), np.where(ge, pos[0][1], pos[1][1])
    mnr, mnc = np.where(gt, pos[1][0], pos[0][0]), np.where(gt, pos[1][1], pos[0][1])
    for i in range(2, 9):
        t = d[i]
        keep = mxv >= t
        mxv = np.where(keep, mxv, t)
        mxr, mxc = np.where(keep, mxr, pos[i][0]), np.where(keep, mxc, pos[i][1])
        new = t < mnv
        mnv = np.where(new, t, mnv)
        mnr, mnc = np.where(new, pos[i][0], mnr), np.where(new, pos[i][1], mnc)
    ox, oy = f32(a["mvoffx"]), f32(a["mvoffy"])
    out = {}
    for name, (rr, cc) in (("min", (mnr, mnc)), ("max", (mxr, mxc)), ("center", (y, x))):
        p = mv[rr, cc]
        out[name] = np.stack([ox + p[..., 0], oy + p[..., 1]], -1)
    r = bool(a["reset"])
    out["prev_out"] = out["max"] if r else out["min"]     # -> args[0x30] prev_mvec[p^1]
    out["ero"] = out["min"] if r else out["max"]          # -> args[0x38] mv_ero
    return out


def dhist_positions(a, g, rows, cols):
    """The four samples' kernel position words (Cx, Cy) on the padded grid.  Samples 0..2 share
    the base 3*((p+0.5) - F20); sample 3 has NO -F20 term.  The MV terms are the kernel's own
    three outputs; for sample 3 the centre again."""
    fld = mv_fields(a, g, rows, cols)
    yy = rows.astype(f32)[:, None] + f32(0.5)
    xx = cols.astype(f32)[None, :] + f32(0.5)
    F20x, F20y = f32(a["F20x"]), f32(a["F20y"])
    base_x = f32(3.0) * (xx - F20x)
    base_y = f32(3.0) * (yy - F20y)

    def mv3(name):
        p = fld[name]
        return f32(3.0) * p[..., 0], f32(3.0) * p[..., 1]

    out = []
    for name in ("prev_out", "center", "ero"):                       # s0, s1, s2
        mx_, my_ = mv3(name)
        out.append((base_x + mx_, base_y + my_))
    cx_, cy_ = mv3("center")
    out.append((f32(3.0) * xx + cx_, f32(3.0) * yy + cy_))           # s3: centre MV, no -F20
    return out


def oob_mask(Cx, Cy, a):
    """The kernel's validity guard for one dhist sample: the position word must lie in
    [0, W15] x [0, H15]; out of bounds the sample is not computed and the LDS byte keeps the
    colour feature's value (measured 100% on M and on L's content region)."""
    W15, H15 = f32(a["W15"]), f32(a["H15"])
    return (Cx < 0) | (Cx > W15) | (Cy < 0) | (Cy > H15)


def dhist_features(a, g, rows, cols):
    """Logical features 3..14 = four bilinear dhist samples x R,G,B (raw, before dq)."""
    pos = dhist_positions(a, g, rows, cols)
    col = colour_features(a, g, rows, cols)
    out = np.zeros(col.shape[:2] + (12,), f32)
    for s, (Cx, Cy) in enumerate(pos):
        oob = oob_mask(Cx, Cy, a)
        for ch in range(3):
            v = dhist_sample(Cx, Cy, ch, g["dhist"], a["hW15"], a["hH15"])
            out[..., 3 * s + ch] = np.where(oob, col[..., ch], v)
    return out


# ------------------------------------------------------------------ features 16..19: warped hidden
def _axis_hidden(C, size, inv15, off):
    """The hidden buffer's sample axis (dec0's warp in enc0's op order)."""
    n = f32(size)
    t = (C.astype(np.float64) * np.float64(inv15) + np.float64(off)).astype(f32)   # one fma
    c = n * ((f32(1) / n) * (n * t))
    c = c - f32(0.5)
    c = np.maximum(c, f32(-2.0))
    c = np.where(np.isnan(c), f32(-0.5), c)
    c = np.minimum(c, n + f32(2.0))
    s = np.floor(c * f32(256.0) + f32(0.5)) * f32(1.0 / 256.0)
    tap = np.floor(s)
    return tap, s - tap


def hidden_features(a, g, rows, cols):
    """Logical features 16..19 = the 4 int8 hidden0 channels warped at the sample-0 position
    (the SAME Cx, Cy words as dhist sample 0), value = int8/127 clamped to >= -1.  Read only
    where the sample-0 position is inside [0,W15]x[0,H15], else the byte stays 0."""
    H, W = a["H"], a["W"]
    hin = g["hidden_p"].astype(np.int32)                            # (H, W, 4); L: hidden0 always
    Cx, Cy = dhist_positions(a, g, rows, cols)[0]
    inv15w, inv15h = f32(1.0) / f32(a["W15"]), f32(1.0) / f32(a["H15"])
    tx, fx = _axis_hidden(Cx, W, inv15w, f32(a["F28x"]) / f32(W))
    ty, fy = _axis_hidden(Cy, H, inv15h, f32(a["F28y"]) / f32(H))
    ix, iy = tx.astype(np.int64), ty.astype(np.int64)
    x0 = np.clip(ix, 0, W - 1)
    x1 = np.clip(np.maximum(ix, -1) + 1, 0, W - 1)
    y0 = np.clip(iy, 0, H - 1)
    y1 = np.clip(np.maximum(iy, -1) + 1, 0, H - 1)
    w00 = (f32(1) - fy) * (f32(1) - fx)
    w01 = (f32(1) - fy) * fx
    w10 = fy * (f32(1) - fx)
    w11 = fy * fx
    valid = ~oob_mask(*dhist_positions(a, g, rows, cols)[:1][0], a)
    out = np.zeros(colour_features(a, g, rows, cols).shape[:2] + (4,), f32)
    for k in range(4):
        q = lambda rr, cc: np.maximum(f32(-1.0), hin[rr, cc, k].astype(f32) * f32(1.0 / 127.0))
        t00, t01, t10, t11 = q(y0, x0), q(y0, x1), q(y1, x0), q(y1, x1)
        v = ((w00 * t00 + w01 * t01) + w10 * t10) + w11 * t11
        out[..., k] = np.where(valid, v, f32(0))
    return out


# ------------------------------------------------------------------ feature 15: motion energy
LN2 = np.frombuffer(bytes.fromhex("1872313f"), np.float32)[0]     # 0x3f317218 = ln(2)


def reset_feature(a, g, rows, cols):
    """Logical feature 15 (raw): SC * ln(1 + sqrt(S)), S = (k*dx)^2 + (k*dy)^2 with
    k = Args[0xe8] (L: 2774.359619140625/H) and SC = Args[0xd0] (L: 0.08225896209478378),
    d = prev_out - interp/3, interp = the bilinear of the prev_mvec[p] taps at the render-grid
    position.  Frame 0 (constant oracle MV, zero history): S = k^2*|g_mv|^2 = 3.6060 ->
    v = 0.0875521, which the dithered dq splits 78.125%/21.875% between fp8 codes 27/28 -
    reproduced exactly (the M analogue: 33%/67% at 0.328349)."""
    H, W = a["H"], a["W"]
    yy = rows.astype(f32)[:, None] + f32(0.5)
    xx = cols.astype(f32)[None, :] + f32(0.5)
    fld = mv_fields(a, g, rows, cols)
    cur = fld["prev_out"]
    mvx, mvy = cur[..., 0], cur[..., 1]
    F20x, F20y, F28x, F28y = (f32(a[k]) for k in ("F20x", "F20y", "F28x", "F28y"))

    def axis(u, n):
        t = (u.astype(np.float64) / np.float64(n)).astype(f32)
        t = np.minimum(np.maximum(t, f32(0.0)), f32(1.0))
        c = f32(n) * t - f32(0.5)
        c = np.maximum(c, f32(-2.0))
        c = np.where(np.isnan(c), f32(-0.5), c)
        c = np.minimum(c, f32(n + 2.0))
        s = np.floor(c * f32(256.0) + f32(0.5)) * f32(1.0 / 256.0)
        tap = np.floor(s)
        return tap, s - tap

    ux = F28x + (xx - F20x) + mvx                                  # f32 add order as M
    uy = F28y + (yy - F20y) + mvy
    px, fx = axis(ux, W)
    py, fy = axis(uy, H)
    ix, iy = px.astype(np.int64), py.astype(np.int64)
    x0 = np.clip(ix, 0, W - 1); x1 = np.clip(np.maximum(ix, -1) + 1, 0, W - 1)
    y0 = np.clip(iy, 0, H - 1); y1 = np.clip(np.maximum(iy, -1) + 1, 0, H - 1)
    pm = g["prev_mvec_p"].astype(f32)
    w00 = (f32(1) - fy) * (f32(1) - fx)
    w01 = (f32(1) - fy) * fx
    w10 = fy * (f32(1) - fx)
    w11 = fy * fx
    ip = np.zeros(mvx.shape + (2,), f32)
    for k in range(2):
        t00, t01, t10, t11 = pm[y0, x0, k], pm[y0, x1, k], pm[y1, x0, k], pm[y1, x1, k]
        ip[..., k] = ((w00 * t00 + w01 * t01) + w10 * t10) + w11 * t11
    k = f32(a["k"])
    dx = (mvx - ip[..., 0] * f32(1.0 / 3.0)).astype(f32)
    dy = (mvy - ip[..., 1] * f32(1.0 / 3.0)).astype(f32)
    ax, ay = (k * dx).astype(f32), (k * dy).astype(f32)
    S = (ax * ax + ay * ay).astype(f32)
    # sqrt -> the ISA's nextafter-down selection, then ln(z) = log2(z) * ln2, * SC
    rt = np.sqrt(S, dtype=f32)
    bits = rt.view(np.uint32)
    down = (bits - np.uint32(1)).view(f32)
    up = (bits + np.uint32(1)).view(f32)
    r10 = (S - up * rt).astype(f32)
    r17 = np.where(r10 <= 0, down, rt)
    root = np.where(r10 > 0, up, r17)
    v = (f32(1.0) + root).astype(f32)
    v = (np.log2(v).astype(f32) * LN2).astype(f32) * f32(a["SC"])
    return v


# ------------------------------------------------------------------ the feature grid -> l0_in
def features(a, g):
    """(H0+8, W0+8, NFEAT) logical fp8 CODES (the P-order store happens in l0_in())."""
    rows, cols = coords(a)
    x = np.zeros((a["H0"] + NPAD, a["W0"] + NPAD, NFEAT), f32)
    x[..., 0:3] = colour_features(a, g, rows, cols)
    x[..., 3:15] = dhist_features(a, g, rows, cols)
    x[..., 15] = reset_feature(a, g, rows, cols)
    x[..., 16:20] = hidden_features(a, g, rows, cols)
    # dither phase = |coord - 2| (the ABS before the border reflect): identical to the level-0
    # coordinate everywhere except the top/left 2-pixel border, where (r-2)&15 = 14,15 but the
    # kernel uses |r-2|&15 = 2,1 - measured 100.0000% both frames with abs, 99.91/99.64 with r-2.
    py, px = np.abs(np.arange(a["H0"] + NPAD) - 2), np.abs(np.arange(a["W0"] + NPAD) - 2)
    return np.stack([dq(x[..., L], a["dither"], a["fc"], py, px) for L in range(NFEAT)], -1)


def l0_in(a, g):
    """The l0_in bytes: (H0+8)*(W0+8)*32 u8; stored byte j = logical feature P[j] (logical
    20..31 are never written -> 0).  m.P maps logical -> stored, so channel j carries P[j]."""
    fe = features(a, g)
    out = np.zeros(fe.shape[:2] + (32,), np.uint8)
    for j in range(32):
        L = int(m.P[j])
        out[..., j] = fe[..., L] if L < NFEAT else 0
    return out


# ------------------------------------------------------------------ the MV outputs
def mv_outputs(a, g):
    """prev_mvec[p^1], mv_ero, mv_nondil (f16 (H,W,2)): f16(3.0 * the tournament read), stored
    for level-0 y in [0, H) and x in [0, W) (the padded grid's content columns; the buffers are
    H rows of W).  F34x/mrx rounds to exactly 3.0 in f32 (as M)."""
    rows, cols = coords(a)
    fld = mv_fields(a, g, rows, cols)
    H, W = a["H"], a["W"]
    s = f32(3.0)
    keep = np.zeros(fld["center"].shape[:2], bool)
    keep[2:2 + H, 2:2 + W] = True                    # level-0 y in [0,H), x in [0,W)
    out = {}
    for name, slot in (("prev_out", "prev_mvec"), ("ero", "mv_ero"), ("center", "mv_nondil")):
        v = m.f16(s * fld[name])
        v = np.where(keep[..., None], v, m.f16(0))
        out[slot] = v[2: 2 + H][:, 2: 2 + W]
    return out


# ------------------------------------------------------------------ acceptance L-1
def self_check(dump=None, frames=(0, 1), quiet=False):
    """L-1: model vs the oracle dump, frames 0 and 1 - l0_in bit-exact % (>= 99) + the three
    MV buffers (100%)."""
    dump = Path(dump) if dump else REPO / "build/oracle/oracle-8dbf027a9e-s20261025/sr_L/dump"
    ok = True
    for fr in frames:
        a, g = load(dump, fr)
        mine = l0_in(a, g)
        ref = np.frombuffer((dump / f"frame{fr}/002.l0_in.bin").read_bytes(),
                            np.uint8)[: mine.size].reshape(mine.shape)
        n = int(np.sum(mine == ref))
        pct = 100.0 * n / ref.size
        good = pct >= 99.0
        ok &= good
        if not quiet:
            print(f"frame{fr}: l0_in bit-exact {pct:.4f}%  (target >= 99%)  {'PASS' if good else 'FAIL'}")
            bad = np.nonzero(mine != ref)
            if bad[0].size:
                print(f"          first mismatch: r{bad[0][0]} c{bad[1][0]} stored-ch{bad[2][0]} "
                      f"(logical {int(m.P[bad[2][0]])}) mine {mine[bad[0][0], bad[1][0], bad[2][0]]} "
                      f"oracle {ref[bad[0][0], bad[1][0], bad[2][0]]}")
        for name in ("prev_mvec", "mv_ero", "mv_nondil"):
            fam = (f"002.{name}1.bin", f"002.{name}0.bin",
                   f"start.{name}1.bin", f"start.{name}0.bin") if name == "prev_mvec" else \
                  (f"002.{name}.bin", f"start.{name}.bin")
            refp = next((dump / f"frame{fr}" / n_ for n_ in fam if (dump / f"frame{fr}" / n_).exists()), None)
            refm = np.frombuffer(refp.read_bytes(),
                                 np.uint16)[: a["H"] * a["W"] * 2].reshape(a["H"], a["W"], 2)
            mine_m = mv_outputs(a, g)[name].view(np.uint16)
            nn = int(np.sum(mine_m == refm))
            good = nn == refm.size
            ok &= good
            if not quiet:
                print(f"          {name}: {nn}/{refm.size} bit-exact ({100.0 * nn / refm.size:.4f}%)  "
                      f"{'PASS' if good else 'FAIL'}")
    if not quiet:
        print("L-1:", "PASS" if ok else "FAIL")
    return ok


def oracle_ref(dump, fr, name):
    """The dump's reference file for one MV output (the launch-write file, else the pre-launch
    state the write did not change) - the same fallback self_check() scores against."""
    fam = (f"002.{name}1.bin", f"002.{name}0.bin",
           f"start.{name}1.bin", f"start.{name}0.bin") if name == "prev_mvec" else \
          (f"002.{name}.bin", f"start.{name}.bin")
    return next((dump / f"frame{fr}" / n for n in fam if (dump / f"frame{fr}" / n).exists()), None)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", type=int, default=None)
    ap.add_argument("--oracle", default=None)
    ap.add_argument("--out", default=None, help="write the model's l0_in/mv buffers here "
                    "(skipped when byte-identical to the oracle dump file - no duplicates)")
    ap.add_argument("--self-check", action="store_true")
    args = ap.parse_args(argv)
    if args.self_check:
        return 0 if self_check(args.oracle) else 1
    dump = Path(args.oracle or (REPO / "build/oracle/oracle-8dbf027a9e-s20261025/sr_L/dump"))
    fr = 0 if args.frame is None else args.frame
    a, g = load(dump, fr)
    out = l0_in(a, g)
    if args.out:
        o = Path(args.out)
        o.mkdir(parents=True, exist_ok=True)
        refs = {"l0_in": dump / f"frame{fr}/002.l0_in.bin"}
        refs.update({k: oracle_ref(dump, fr, k) for k in mv_outputs(a, g)})
        bufs = {"l0_in": out.tobytes()}
        bufs.update({k: v.tobytes() for k, v in mv_outputs(a, g).items()})
        for k, b in bufs.items():
            refp = refs[k]
            if refp is not None and refp.read_bytes() == b:
                print(f"frame{fr}.{k}: byte-identical to {refp.name} - not written")
                continue
            (o / f"frame{fr}.{k}.bin").write_bytes(b)
            print(f"frame{fr}.{k}.bin written ({len(b)} B)")
    else:
        print(f"frame{fr}: l0_in {out.shape} u8, nonzero channels: "
              f"{[c for c in range(32) if out[..., c].any()]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
