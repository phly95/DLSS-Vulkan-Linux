#!/usr/bin/env python3
"""Texel encoders for the gather format codes the referee feeds to the oracles and the port.

Codes and their texel layouts come from NeuralHub's own code (docs/sr/HOST-RE.md, docs/rr/HOST-RE.md):

SR (`srunit::k_sr_gather`, `k_sr_gather_mv`, colour map FUN_1800059b0):
    colour  0 RGBA16F, 1 RGBA8 UNORM, 2 RGBA8 UNORM sRGB, 3 RGB10A2 UNORM, 4 R11G11B10F,
            5 RGBA32F, 6 RGB9E5 (the kernel's default branch). 1-3 are LDR (the host's _ldr kernels).
    mv      0 RG16F, 1 RG32F, any other code an 8-byte texel read as two halves (RGBA16F .xy).
    depth   0 R32F, any other code D24 UNORM in the low 24 bits of a 4-byte texel.
RR (`rrunit::k_rr_gather`, DXGI table at 0x1801b58d0): the codes in RR_BPP.

Every file is tightly packed (row pitch = width * bytes per texel). `python3 texfmt.py --self-test`
round-trips each encoder through an independent decoder.
"""
import sys

import numpy as np

SR_COL_BPP = {0: 8, 1: 4, 2: 4, 3: 4, 4: 4, 5: 16, 6: 4}
SR_MV_BPP = {0: 4, 1: 8, 2: 8}
SR_DEP_BPP = {0: 4, 1: 4}
SR_LDR_COL = {1, 2, 3}
RR_BPP = {0: 8, 1: 4, 2: 4, 3: 4, 4: 2, 5: 4, 6: 4, 7: 4, 8: 1, 9: 2, 10: 8, 11: 16, 12: 4, 13: 4,
          14: 8, 15: 8, 16: 4, 17: 4, 18: 4, 19: 4, 20: 2, 21: 2, 22: 2, 23: 1}


def _unorm(x, bits):
    m = (1 << bits) - 1
    return np.clip(np.floor(np.clip(x, 0.0, 1.0) * m + 0.5), 0, m).astype(np.uint32)


def _snorm(x, bits):
    m = (1 << (bits - 1)) - 1
    return (np.clip(np.floor(np.clip(x, -1.0, 1.0) * m + 0.5), -m, m).astype(np.int32) & ((1 << bits) - 1)).astype(np.uint32)


def linear_to_srgb(c):
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1.0 / 2.4) - 0.055)


def _small_float(x, mbits):
    """Unsigned float with a 5-bit exponent (bias 15) and `mbits` mantissa bits, round to nearest even
    (R11G11B10F's channels). Negative and NaN -> 0, overflow -> the largest finite value."""
    x = np.nan_to_num(np.asarray(x, np.float64), nan=0.0, posinf=65000.0)
    x = np.clip(x, 0.0, None)
    h = x.astype(np.float16).view(np.uint16).astype(np.uint32)       # 5-bit exponent, 10-bit mantissa
    shift = 10 - mbits
    lsb = (h >> shift) & 1
    r = (h + ((1 << (shift - 1)) - 1) + lsb) >> shift
    maxv = (0x1e << mbits) | ((1 << mbits) - 1)
    return np.minimum(r, maxv).astype(np.uint32)


def _rgb9e5(rgb):
    """DXGI_FORMAT_R9G9B9E5_SHAREDEXP, the D3D reference encoding."""
    rgb = np.clip(np.nan_to_num(np.asarray(rgb, np.float64)), 0.0, 65408.0)
    maxc = rgb.max(axis=-1)
    e = np.maximum(-16, np.floor(np.log2(np.maximum(maxc, 1e-30)))).astype(np.int64) + 1 + 15
    e = np.clip(e, 0, 31)
    scale = np.power(2.0, (e - 15 - 9).astype(np.float64))
    mmax = np.floor(maxc / scale + 0.5)
    bump = mmax == 512
    e = np.where(bump, e + 1, e)
    scale = np.where(bump, scale * 2, scale)
    m = np.clip(np.floor(rgb / scale[..., None] + 0.5), 0, 511).astype(np.uint32)
    return (m[..., 0] | (m[..., 1] << 9) | (m[..., 2] << 18) | (e.astype(np.uint32) << 27)).astype(np.uint32)


def encode(kind, code, a):
    """kind: 'sr_col', 'sr_mv', 'sr_dep' or 'rr'; a: float array (H, W, C). Returns raw bytes."""
    a = np.asarray(a, np.float64)
    if a.ndim == 2:
        a = a[..., None]
    rgba = np.concatenate([a, np.zeros(a.shape[:2] + (4 - a.shape[2],)) + [0.0, 0.0, 0.0, 1.0][a.shape[2]:]], axis=2) \
        if a.shape[2] < 4 else a
    r, g, b, al = (rgba[..., i] for i in range(4))
    if kind == "sr_col":
        layout = {0: 0, 1: 5, 2: "srgb8", 3: 7, 4: 1, 5: 11, 6: 12}[code]
        if layout == "srgb8":
            s = linear_to_srgb(rgba[..., :3])
            return _pack_u32(_unorm(s[..., 0], 8) | (_unorm(s[..., 1], 8) << 8) | (_unorm(s[..., 2], 8) << 16)
                             | (_unorm(al, 8) << 24))
        return encode("rr", layout, rgba)
    if kind == "sr_mv":
        if code == 0:
            return np.ascontiguousarray(rgba[..., :2], np.float16).tobytes()
        if code == 1:
            return np.ascontiguousarray(rgba[..., :2], np.float32).tobytes()
        return np.ascontiguousarray(np.stack([r, g, np.zeros_like(r), np.zeros_like(r)], -1), np.float16).tobytes()
    if kind == "sr_dep":
        if code == 0:
            return np.ascontiguousarray(r, np.float32).tobytes()
        return _pack_u32(_unorm(r, 24))
    if kind != "rr":
        raise ValueError(kind)
    f16 = lambda x: np.ascontiguousarray(x, np.float16).tobytes()
    f32 = lambda x: np.ascontiguousarray(x, np.float32).tobytes()
    if code == 0:
        return f16(rgba)
    if code == 1:
        return _pack_u32(_small_float(r, 6) | (_small_float(g, 6) << 11) | (_small_float(b, 5) << 22))
    if code == 2:
        return f16(rgba[..., :2])
    if code == 3:
        return f32(r)
    if code == 4:
        return f16(r)
    if code == 5:
        return _pack_u32(_unorm(r, 8) | (_unorm(g, 8) << 8) | (_unorm(b, 8) << 16) | (_unorm(al, 8) << 24))
    if code == 6:
        return _pack_u32(_unorm(b, 8) | (_unorm(g, 8) << 8) | (_unorm(r, 8) << 16) | (_unorm(al, 8) << 24))
    if code == 7:
        return _pack_u32(_unorm(r, 10) | (_unorm(g, 10) << 10) | (_unorm(b, 10) << 20) | (_unorm(al, 2) << 30))
    if code == 8:
        return np.ascontiguousarray(_unorm(r, 8), np.uint8).tobytes()
    if code == 9:
        return np.ascontiguousarray(_unorm(r, 16), np.uint16).tobytes()
    if code == 10:
        return f32(rgba[..., :2])
    if code == 11:
        return f32(rgba)
    if code == 12:
        return _pack_u32(_rgb9e5(rgba[..., :3]))
    if code == 13:
        return _pack_u32(_snorm(r, 8) | (_snorm(g, 8) << 8) | (_snorm(b, 8) << 16) | (_snorm(al, 8) << 24))
    if code == 14:
        return np.ascontiguousarray(np.stack([_unorm(c, 16) for c in (r, g, b, al)], -1), np.uint16).tobytes()
    if code == 15:
        return np.ascontiguousarray(np.stack([_snorm(c, 16) for c in (r, g, b, al)], -1), np.uint16).tobytes()
    if code == 16:
        return _pack_u32(_unorm(r, 24))
    if code == 17:
        return _pack_u32(_unorm(b, 8) | (_unorm(g, 8) << 8) | (_unorm(r, 8) << 16) | (np.uint32(255) << 24))
    if code == 18:
        return _pack_u32(_unorm(r, 16) | (_unorm(g, 16) << 16))
    if code == 19:
        return _pack_u32(_snorm(r, 16) | (_snorm(g, 16) << 16))
    if code == 20:
        return np.ascontiguousarray(np.stack([_unorm(r, 8), _unorm(g, 8)], -1), np.uint8).tobytes()
    if code == 21:
        return np.ascontiguousarray(np.stack([_snorm(r, 8), _snorm(g, 8)], -1), np.uint8).tobytes()
    if code == 22:
        return np.ascontiguousarray(_snorm(r, 16), np.uint16).tobytes()
    if code == 23:
        return np.ascontiguousarray(_snorm(r, 8), np.uint8).tobytes()
    raise ValueError(f"rr code {code}")


def _pack_u32(v):
    return np.ascontiguousarray(v, np.uint32).tobytes()


def bpp(kind, code):
    return {"sr_col": SR_COL_BPP, "sr_mv": SR_MV_BPP, "sr_dep": SR_DEP_BPP, "rr": RR_BPP}[kind][code]


# ---- independent decoders, for the self-test only ----------------------------------------------
def _dec_small(v, mbits):
    e = (v >> mbits) & 31
    m = v & ((1 << mbits) - 1)
    return np.where(e == 0, m / (1 << mbits) * 2.0 ** -14, (1 + m / (1 << mbits)) * 2.0 ** (e.astype(np.float64) - 15))


def self_test():
    rng = np.random.default_rng(7)
    h, w = 5, 7
    x = rng.uniform(0, 1, (h, w, 4))
    hdr = rng.uniform(0, 40, (h, w, 4))
    fails = 0

    def chk(name, got, want, tol):
        nonlocal fails
        err = float(np.max(np.abs(np.asarray(got, np.float64) - want) / np.maximum(1.0, np.abs(want))))
        ok = err <= tol
        fails += not ok
        print(f"  {name:28s} max rel err {err:.3g} (tol {tol:g}) {'ok' if ok else 'FAIL'}")

    u32 = lambda b: np.frombuffer(b, np.uint32).reshape(h, w)
    for kind, codes in (("sr_col", SR_COL_BPP), ("sr_mv", SR_MV_BPP), ("sr_dep", SR_DEP_BPP), ("rr", RR_BPP)):
        for c in codes:
            n = len(encode(kind, c, x))
            if n != h * w * bpp(kind, c):
                print(f"  {kind} {c}: {n} bytes, expected {h * w * bpp(kind, c)} FAIL")
                fails += 1
    v = u32(encode("rr", 1, hdr))
    chk("R11G11B10F r", _dec_small(v & 0x7ff, 6), hdr[..., 0], 2 ** -6)
    chk("R11G11B10F b", _dec_small(v >> 22, 5), hdr[..., 2], 2 ** -5)
    v = u32(encode("rr", 12, hdr))
    e = (v >> 27).astype(np.float64)
    # one exponent for the three channels: the error is relative to the largest channel
    mx = hdr[..., :3].max(-1)
    chk("RGB9E5 g / max channel", ((v >> 9) & 511) * 2.0 ** (e - 24) / mx, hdr[..., 1] / mx, 2 ** -9)
    v = u32(encode("sr_col", 3, x))
    chk("RGB10A2 g", ((v >> 10) & 1023) / 1023.0, x[..., 1], 1 / 2046 + 1e-9)
    v = u32(encode("sr_col", 2, x))
    s = (v & 255) / 255.0
    lin = np.where(s <= 0.04045, s / 12.92, ((s + 0.055) / 1.055) ** 2.4)
    chk("RGBA8 sRGB r", lin, x[..., 0], 0.02)
    v = u32(encode("rr", 6, x))
    chk("BGRA8 r (byte 2)", ((v >> 16) & 255) / 255.0, x[..., 0], 1 / 510 + 1e-9)
    v = u32(encode("sr_dep", 1, x[..., :1]))
    chk("D24 (low 24 bits)", (v & 0xffffff) / 16777215.0, x[..., 0], 1e-7)
    chk("RG32F", np.frombuffer(encode("sr_mv", 1, x), np.float32).reshape(h, w, 2), x[..., :2], 1e-7)
    q = np.frombuffer(encode("sr_mv", 2, x), np.float16).reshape(h, w, 4)
    chk("MV as RGBA16F .xy", q[..., :2], x[..., :2], 1e-3)
    v = np.frombuffer(encode("rr", 15, x * 2 - 1), np.int16).reshape(h, w, 4)
    chk("RGBA16 SNORM", v / 32767.0, x * 2 - 1, 1 / 65534 + 1e-9)
    print("TEXFMT SELF-TEST:", "PASS" if fails == 0 else f"FAIL ({fails})")
    return fails == 0


if __name__ == "__main__":
    if sys.argv[1:] == ["--self-test"]:
        sys.exit(0 if self_test() else 1)
    print(__doc__)
