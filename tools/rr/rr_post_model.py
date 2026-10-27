#!/usr/bin/env python3
"""tools/rr/rr_post_model.py: the NumPy model of NeuralHub's RR kernel rrio::k_post<24>
(_ZN4rrio6k_postILi24EEEvNS_8PostArgsE), the spec that shaders/rr/rr_post.comp is rewritten against.

    build/venv/bin/python -I tools/rr/rr_post_model.py --frame 0 [--out DIR]
    build/venv/bin/python -I tools/rr/rr_post_model.py --self-check

Launch: oracle launch 21 in frame 0 (rst = 1), launch 22 in frame 1 (rst = 0), sequence
build/oracle/oracle-8dbf027a9e-s20261025/rr_F/dump. Inputs are the oracle's own bytes as they stood
before the launch (newest NNN.<buffer>.bin with NNN < launch, else start.<buffer>.bin).

STATUS (2026-10-27, rr-post-model; numbers from `--self-check` on the oracle dump, launch 21 frame 0).
  Frame 0 (reset, no history): A1 PASS. E0cf6 99.9424%, E0cfa 99.9659%, E0c0a 99.9773%, E0c0e 99.9891%.
  Frame 1 (history):           NOT MODELLED (A2 FAIL). The history fetch (E0d02 table + five
                               image_sample_lz) and the blend against it are only partly decoded (SPEC
                               section 6, open). The frame-1 numbers printed by --self-check are the
                               frame-0 law applied to frame-1 inputs with no history term: a baseline, not
                               this model.

A3 VERIFICATION TABLE (stage, what it was checked against, match):
  stage 1  tap geometry dx0/dy0, odd-row shift      verified only through stage 2 (the probe reads q, which uses them)
  stage 2  tap quadratic q, both lanes, tap (-1,-1)  hardware probe rrp_q151 (patchprobe, RR +0x48c, v12):
           even lane 100%, odd lane 100% over all 216 x 768 pairs of frame 0
  stage 3  weights, f16 numerators, Wsum            oracle E0cf6 R/G and E0cfa: 99.96-99.97% through S (frame 0)
  stage 4  normalised average S = N * (1/Wsum)       oracle E0c0a R/G/B 99.97-99.98% (frame 0, f16(decode/tm))
  stage 5  packed E0c0e = 2x2 average of decode(S)   oracle E0c0e 99.99% (frame 0)
  stage 6  history fetch + blend                     NOT VERIFIED (frame 1 open)

SPEC (all constants are immediates of the ISA or the PostArgs; f16 = IEEE half, round to nearest even)

Inputs (PostArgs, 0xb8 bytes, read by the kernel; field names from tools/rr/rrref/rrref.cpp):
  0x00: W, H = 256, 144 (render size, the Tap grid);   0x08: W15, H15 = 768, 432 (output of this pass)
  0x10: rst (1 on frame 0: history not used);           0x14, 0x18: s = (W/W15, H/H15) = (1/3, 1/3), f32
  0x1c, 0x20: jitter (jx, jy), f32                      0x40: E0cf2 Tap buffer (32 B per render pixel)
  0x48: E0c22 exposure (f32 x 4)   0x50: E0c1a (u8 x 4, unit-15 softmax factors, frame 1 only)
  0x58/0x60/0x68: E0bf6|E0bf2, E0bfa, E0bfe (f16 features, the mv record is f16 x 4 = (mvx,mvy,mvx,mvy))
  0x70: history read buffer (E0c06 / E0c0a)  - read through the texture at 0x80 (see open items)
  0x78: E0d02 table (513 x 4 f32: the Catmull-Rom weights per 1/512 fraction, read by the kernel)
  0x80: texture object over the 0x70 history (T#, sampler, image_sample_lz 2D)
  0x88: history WRITE buffer (E0c0a on frame 0, E0c06 on frame 1), f16 x 4 per output pixel, tm domain
  0x90: E0cf6 (f16 x 2 per output pixel: R, G, tm domain)   0x98: E0cfa (f16, B, tm domain)
  0xa0: E0c0e (u32 per 2x2 block, 384 x 216)                0xa8: debug pointer, always NULL in the oracle

1. Tap record (the post_prep Tap, 16 f16 per render pixel, E0cf2): elements
     0..2  colour tm R,G,B (tonemap domain)      3  w0 (gate 0)   4  w1   5  w2
     6  A (x-quadratic coefficient, an0)         7  B (y-quadratic coefficient, an1)
     8  C (cross coefficient, an2)              9..15 zero.
   (The post_prep stores C3*aniso; the packed ISA of k_post is verified on this layout, stage 2.)

2. Geometry, output pixel (y15, x15), centre in render units:
     px = fma(x15 + 0.5, sx, jx)      py = fma(y15 + 0.5, sy, jy)    (one rounding, v_fma_f32)
     X0 = floor(px - 0.5)  Y0 = floor(py - 0.5)
     dx0 = f16((X0 + 0.5) - px)  dy0 = f16((Y0 + 0.5) - py)           (f32 subtract, then f16)
   A thread owns the pair of output rows (2r, 2r+1) and processes them together. Both rows share
   the source rows of the even row: Yb = Y0(2r). The odd row keeps its own dy, shifted by
     shift = Y0(2r+1) - Yb (0 or 1);  dy1 = f16(f16((Y0(2r+1)+0.5) - py(2r+1)) - shift).
   The 4 x 4 footprint: kx in {-1, 0, 1, 2} (dx = dx0 + kx) and ky in {-1, 0, 1, 2} (dy = dy + ky),
   source pixel (clamp(X0 + kx), clamp(Yb + ky)). THIS IS 16 TAPS, not 9 (verified by the ISA: 4 LDS
   columns x 4 LDS rows, 16 exp per pixel; the brief's 9-tap statement is wrong).

3. Tap quadratic, both lanes at once (lane 0 = even row, lane 1 = odd row; verified bit-exact):
     ddx = f16(dx0 + kx);  de = f16(dy0 + ky) (even);  do = f16(dy1 + ky) (odd)
     q = f16( f16( f16(f16(de*de)*B) + f16(f16(ddx*ddx)*A) ) + f16(f16(ddx*de)*C) )   (per lane)
     w = exp2(f32(q))   (v_exp_f32; numpy exp2 here, 1-ULP differences are possible)

4. Weights and numerators. Taps are visited column-major: kx = -1 (ky -1..2), kx = 0, kx = 1,
   kx = 2 (ISA order). Denominator Wsum = sum of w (f32, same order). Numerator for field f:
     n = 0;  for each tap: n = f16(n + f16(w) * field_f(tap))       (f16 fma chain, one rounding per tap)
   fields: 0..2 colour (tm), 3..5 gates.  Normalised value S_f = f32(n_f) * inv, inv = 1/Wsum (f32).

5. Frame 0 outputs (rst = 1: no history term; the blend is the identity, checked by the oracle):
     x = decode(S), decode(h) = 0.01 * (exp2(clamp(h, 1e-4, 3) * (1/0.36) * log2(10)) - 1)  (f32)
     tm = (log2(max(fma(100, x, 1), 1e-4)) * log10(2)) * 0.36                               (f32)
     E0cf6[y15,x15] = (f16(tm_R) | f16(tm_G) << 16)       (lo = R, hi = G)
     E0cfa[y15,x15] = f16(tm_B)
     E0c0a[y15,x15] = f16(tm_R) | f16(tm_G) << 16 , f16(tm_B) | 0 << 16   (history write, alpha 0)
     E0c0e[y/2, x/2] (u32, 384 x 216) = R10 | G10 << 10 | B10 << 20 | 0xc0000000 with, per channel c,
       xbar = ((x(2r,2c) + x(2r,2c+1)) + (x(2r+1,2c) + x(2r+1,2c+1))) * 0.25  (f32, rows first)
       t = ((k * (0.36 * (log2(max(fma(100, xbar, 1), 1e-4)) * log10(2)))) clamped to [0,1]
       Cc = rndne(1023 * t), k = 1/2.45 = f32 0x3ed0fac6.
     (The ISA reads the 2x2 neighbours with ds_bpermute and divides by the lane count: the 4-pixel
      block is complete for every even/even pixel.)

6. Frame 1 (history). OPEN (A2 not met). Verified by hardware probes on the 2-frame directory
   (build/agents/rr-post-model/frames2; an unpatched run reproduces all 55 frame-1 oracle buffers byte for byte):
   a. CR position (output texels, probed at ISA 0x1514/0x151c): pos = (x15 + 0.5 + mvx, y15 + 0.5 + mvy)
      with mv = E0bf6 (-0.2861328125, -0.07501220703125) exactly; no scale. Verified on every even row.
   b. CR table index (ISA 0x0fa0-0x1040): fraction f = pos - (floor(pos - 0.5) + 0.5), clamped to [0,1];
      k = min(512, floor(512*f + 0.5)) (fma, then truncation). Entry k of the E0d02 table (513 x 4 f32):
      (w(-1), a = w0+w1, w2, r), where lane 3 is the bilinear fraction r (w1/a, to 1/512).
   c. Five taps (ISA 0x1738-0x179c, image_sample_lz): main (Ix+0.5+r_x, Iy+0.5+r_y) with weight a_x*a_y;
      pure centres (Ix-0.5, Iy+0.5+r_y) weight w(-1)_x*a_y; (Ix+2.5, ...) weight w2_x*a_y;
      (Ix+0.5+r_x, Iy-0.5) weight a_x*w(-1)_y; (Ix+0.5+r_x, Iy+2.5) weight a_x*w2_y. Coordinates probed:
      t1 u = -1.5/768 (outside the texture), t2 u = 1.5/768 and v = 0.4530/432, t3 v = -1.5/432.
      Sum of the five weights = 0.99642 at every pixel (= 1/inv probe, 1.0035897); h = sum(W*t)/sum(W).
   d. The history texel values are nonzero everywhere, and the frame-1 history input equals the
      frame-0 output 021.E0c0a byte for byte.
   OPEN: (i) the probed CR numerators/h are zero at ISA 0x1890 and later although the t2/t1 sample
   values read nonzero at 0x1880 (the probe cause is not resolved); (ii) the sampler addressing mode
   (clamp vs border for the -1.5 texel) is not decided; (iii) the blend between S, h and the frame-0
   tm history (E0c1a factors, gates, exposure expo0/expo1 with 1/expo1) is not fitted: the linear and
   tm-domain lerps with the modelled h give t outside [0,1] on most pixels.
"""
import argparse
import json
import os
import struct
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
ORACLE = REPO / "build/oracle/oracle-8dbf027a9e-s20261025/rr_F/dump"
W, H, W15, H15 = 256, 144, 768, 432
LAUNCH = {0: 21, 1: 22}
F = np.float32
# column-major tap order (kx outer, ky inner) as the ISA visits the 4 x 4 footprint
KX = [(c, r) for c in range(-1, 3) for r in range(-1, 3)]


def f16(x):
    return np.asarray(x).astype(np.float16)


def bits(u):
    return np.frombuffer(struct.pack("<I", u), np.float32)[0]


C_LOG10_2 = bits(0x3E9A209B)     # log10(2)            (ISA immediate)
C_TM = bits(0x3EB851EC)          # 0.36                (ISA immediate)
C_K = bits(0x3ED0FAC6)           # 1 / 2.45            (ISA immediate)
C_HUNDRED = bits(0x42C80000)     # 100                 (ISA immediate)
C_FLOOR = bits(0x38D1B717)       # 1e-4                (ISA immediate)
C_PT01 = bits(0x3C23D70A)        # 0.01                (ISA immediate)
C_I036 = bits(0x4031C71C)        # 1 / 0.36            (ISA immediate)
C_L210 = bits(0x40549A78)        # log2(10)            (ISA immediate)


def fma(a, b, c):
    """one rounding: fl(a*b + c)."""
    return (np.float64(a) * np.float64(b) + np.float64(c)).astype(F)


# ------------------------------------------------------------------ inputs
def launch_args(frame):
    rows = [json.loads(l) for l in open(ORACLE / f"frame{frame}/launches.jsonl")]
    d = next(r for r in rows if r["n"] == LAUNCH[frame])
    return bytes.fromhex(d["args"]), {int(k): v.split("+")[0] for k, v in d["pointers"].items()}


def buffer_before(dump_frame, name, launch):
    best = None
    for fn in os.listdir(dump_frame):
        if fn.endswith(f".{name}.bin") and fn[:3].isdigit() and int(fn[:3]) < launch:
            if best is None or int(fn[:3]) > int(best[:3]):
                best = fn
    return (Path(dump_frame) / (best if best else f"start.{name}.bin")).read_bytes()


def load_inputs(frame):
    fdir = ORACLE / f"frame{frame}"
    launch = LAUNCH[frame]
    a, ptr = launch_args(frame)
    sx, sy = struct.unpack_from("<ff", a, 0x14)
    jx, jy = struct.unpack_from("<ff", a, 0x1C)
    rst = struct.unpack_from("<i", a, 0x10)[0]
    expo = np.frombuffer(buffer_before(fdir, "E0c22", launch), np.float32)
    tap = np.frombuffer(buffer_before(fdir, "E0cf2", launch), np.float16).reshape(H, W, 16)
    return dict(frame=frame, fdir=fdir, launch=launch, sx=F(sx), sy=F(sy), jx=F(jx), jy=F(jy), rst=rst,
                expo=expo, tap=tap, ptr=ptr)


# ------------------------------------------------------------------ stages 1-2: geometry and tap quadratic
def geometry(inp):
    """X0, Y0 (even row), dx0, dy0 (even), dy1 (odd, shifted by its base) for every output pair (r, x15)."""
    y_e = np.arange(0, H15, 2)[:, None]
    x = np.arange(W15)[None, :]
    px = fma(x.astype(F) + F(0.5), inp["sx"], inp["jx"]) * np.ones((H15 // 2, 1), F)
    py_e = fma(y_e.astype(F) + F(0.5), inp["sy"], inp["jy"])
    py_o = fma((y_e + 1).astype(F) + F(0.5), inp["sy"], inp["jy"])
    X0 = np.floor(px - F(0.5)).astype(np.int64)
    Y0e = np.floor(py_e - F(0.5)).astype(np.int64)
    Y0o = np.floor(py_o - F(0.5)).astype(np.int64)
    dx0 = f16(((X0.astype(F) + F(0.5)) - px).astype(F))
    dy0 = f16(((Y0e.astype(F) + F(0.5)) - py_e).astype(F))
    dy1_own = f16(((Y0o.astype(F) + F(0.5)) - py_o).astype(F))
    shift = (Y0o - Y0e).astype(F)
    dy1 = f16(dy1_own.astype(F) - shift)
    return X0, Y0e, dx0, dy0, dy1


def taps(inp):
    """per (pair, x15): the 16 tap records and the quadratic q of both lanes (even, odd)."""
    X0, Yb, dx0, dy0, dy1 = geometry(inp)
    rec = inp["tap"]
    m = lambda p, q: f16(f16(p).astype(F) * f16(q).astype(F))
    s = lambda p, q: f16(f16(p).astype(F) + f16(q).astype(F))
    out = {}
    for kx, ky in KX:
        ddx = f16(dx0.astype(F) + F(kx))
        de = f16(dy0.astype(F) + F(ky))
        do = f16(dy1.astype(F) + F(ky))
        sxi = np.clip(X0 + kx, 0, W - 1)
        syi = np.clip(Yb + ky, 0, H - 1)
        r = rec[syi, sxi]
        A, B, C = r[..., 6], r[..., 7], r[..., 8]
        tA = m(m(ddx, ddx), A)
        qe = s(s(m(m(de, de), B), tA), m(m(ddx, de), C))
        qo = s(s(m(m(do, do), B), tA), m(m(ddx, do), C))
        out[(kx, ky)] = dict(rec=r, qe=qe, qo=qo)
    return out


# ------------------------------------------------------------------ stages 3-4: weights and the normalised average
def spatial(inp, tp):
    """S[lane][pair, x15, field] for fields 0..5 (colour tm, gates), lane 0 = even row, 1 = odd row."""
    res = {}
    for lane, qk in ((0, "qe"), (1, "qo")):
        ws = {k: np.exp2(tp[k][qk].astype(F)).astype(F) for k in KX}
        wsum = np.zeros(ws[KX[0]].shape, F)
        for k in KX:
            wsum = (wsum + ws[k]).astype(F)
        inv = (F(1.0) / wsum).astype(F)
        S = np.zeros(ws[KX[0]].shape + (6,), F)
        for fi in range(6):
            n = np.zeros(ws[KX[0]].shape, F)
            for k in KX:
                n = f16(n + f16(ws[k]).astype(F) * tp[k]["rec"][..., fi].astype(F)).astype(F)
            S[..., fi] = (n * inv).astype(F)
        res[lane] = S
    return res


def interleave(res):
    """(pair, x, f) per lane -> (H15, W15, f): row 2r from lane 0, row 2r+1 from lane 1."""
    out = np.zeros((H15, W15, res[0].shape[-1]), F)
    out[0::2] = res[0]
    out[1::2] = res[1]
    return out


# ------------------------------------------------------------------ stage 5: outputs
def decode_lin(h):
    hc = np.clip(h, C_FLOOR, F(3.0)).astype(F)
    arg = ((hc * C_I036).astype(F) * C_L210).astype(F)
    return (C_PT01 * (np.exp2(arg).astype(F) - F(1.0)).astype(F)).astype(F)


def tonemap(x):
    """tm(x) = (log2(max(fma(100, x, 1), 1e-4)) * log10(2)) * 0.36 (f32)."""
    y = np.maximum(fma(C_HUNDRED, x, F(1.0)), C_FLOOR).astype(F)
    lg = (np.log2(y).astype(F) * C_LOG10_2).astype(F)
    return (lg * C_TM).astype(F)


def encode10(t):
    t = np.clip((C_K * t).astype(F), F(0.0), F(1.0)).astype(F)
    return np.rint((F(1023.0) * t).astype(np.float64)).astype(np.uint32)


def frame_outputs_frame0(S):
    """outputs of frame 0 from the normalised colour S (interleaved, (H15, W15, >=3))."""
    x = decode_lin(S[..., :3])
    tm = tonemap(x)
    e_cf6 = (f16(tm[..., 0]).view(np.uint16).astype(np.uint32)
             | (f16(tm[..., 1]).view(np.uint16).astype(np.uint32) << 16))
    e_cfa = f16(tm[..., 2])
    hist = np.zeros((H15, W15, 4), np.float16)
    hist[..., :3] = f16(tm)
    xb = x.reshape(H15 // 2, 2, W15 // 2, 2, 3)
    xs = (xb[:, 0] + xb[:, 1]).astype(F)                 # rows: (2r) + (2r+1)
    xs = ((xs[:, :, 0] + xs[:, :, 1]) * F(0.25)).astype(F)   # columns: (2c) + (2c+1), * 0.25
    tmb = tonemap(xs)
    val = encode10(tmb[..., 0]) | (encode10(tmb[..., 1]) << 10) | (encode10(tmb[..., 2]) << 20)
    e_c0e = (val | np.uint32(0xC0000000)).astype(np.uint32)
    return dict(E0cf6=e_cf6, E0cfa=e_cfa, E0c0a=hist, E0c0e=e_c0e)


def model(frame):
    inp = load_inputs(frame)
    if frame != 0:
        raise NotImplementedError("frame 1 (history) is not modelled yet: see SPEC section 6")
    tp = taps(inp)
    S = interleave(spatial(inp, tp))
    return frame_outputs_frame0(S), inp


def model_baseline_frame1():
    """NOT THE MODEL: the frame-0 law (no history term) applied to frame-1 inputs. Used only to show the gap."""
    inp = load_inputs(1)
    tp = taps(inp)
    S = interleave(spatial(inp, tp))
    return frame_outputs_frame0(S), inp


# ------------------------------------------------------------------ scoring
def compare(name, got, want):
    got = np.asarray(got)
    want = np.asarray(want)
    frac = float(np.mean(got == want)) * 100
    return frac


def oracle_outputs(inp):
    n = inp["launch"]
    fd = inp["fdir"]
    rd = lambda nm, dt, sh: np.frombuffer(open(fd / f"{n:03d}.{nm}.bin", "rb").read(), dt).reshape(sh)
    if inp["frame"] == 0:
        hist_name = "E0c0a"
    else:
        hist_name = "E0c06"
    return dict(E0cf6=rd("E0cf6", np.uint32, (H15, W15)), E0cfa=rd("E0cfa", np.float16, (H15, W15)),
                E0c0a=rd(hist_name, np.float16, (H15, W15, 4)), E0c0e=rd("E0c0e", np.uint32, (H15 // 2, W15 // 2)))


def run(frame, out=None, quiet=False, baseline=False):
    m, inp = model_baseline_frame1() if baseline else model(frame)
    o = oracle_outputs(inp)
    res = {}
    for k in ("E0cf6", "E0cfa", "E0c0a", "E0c0e"):
        pct = compare(k, m[k], o[k])
        res[k] = pct
        if not quiet:
            print(f"  {k:6s} bit-exact {pct:.4f}%")
    if out:
        os.makedirs(out, exist_ok=True)
        for k, v in m.items():
            Path(out, f"{k}.bin").write_bytes(np.ascontiguousarray(v).tobytes())
    return res


def self_check():
    thr = {0: 99.0, 1: 97.0}
    ok = True
    print("[frame 0, launch 21: the model]")
    r0 = run(0)
    for k, v in r0.items():
        good = v >= thr[0]
        ok &= good
        print(f"  -> {k} {v:.4f}% >= {thr[0]:.1f}%: {'PASS' if good else 'FAIL'}")
    print("[frame 1, launch 22: NOT MODELLED; baseline = frame-0 law without history]")
    r1 = run(1, baseline=True)
    for k, v in r1.items():
        good = v >= thr[1]
        ok &= good
        print(f"  -> {k} {v:.4f}% >= {thr[1]:.1f}%: {'PASS' if good else 'FAIL'} (baseline, not the model)")
    print("SELF-CHECK:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--self-check", action="store_true")
    a = ap.parse_args()
    if a.self_check:
        return self_check()
    run(a.frame, a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
