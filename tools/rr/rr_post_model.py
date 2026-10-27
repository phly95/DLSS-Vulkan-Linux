#!/usr/bin/env python3
"""tools/rr/rr_post_model.py: the NumPy model of NeuralHub's RR kernel rrio::k_post<24>
(_ZN4rrio6k_postILi24EEEvNS_8PostArgsE), the spec that shaders/rr/rr_post.comp is rewritten against.

    build/venv/bin/python -I tools/rr/rr_post_model.py --frame 0 [--out DIR]
    build/venv/bin/python -I tools/rr/rr_post_model.py --self-check

Launch: oracle launch 21 in frame 0 (rst = 1), launch 22 in frame 1 (rst = 0), sequence
build/oracle/oracle-8dbf027a9e-s20261025/rr_F/dump. Inputs are the oracle's own bytes as they stood
before the launch (newest NNN.<buffer>.bin with NNN < launch, else start.<buffer>.bin).

STATUS (2026-10-27, rr-bil-fix; numbers from `--self-check` on the oracle dump, runtime 2.4 s).
  Frame 0 (reset, no history): A1 PASS. E0cf6 99.9424%, E0cfa 99.9659%, E0c0a 99.9773%, E0c0e 99.9891%.
  Frame 1 (history):           A2 PASS. E0cf6 99.9433%, E0cfa 99.9617%, E0c0a 99.9785%, E0c0e 99.9964%
                               (BEFORE the rr-bil-fix: 99.6293 / 99.6977 / 99.7825 / 99.7372).
                               Frame 1 = the CR history fetch (SPEC 6a-c), H = e0*decode(h)/e1 (6d),
                               TWO gate blend weights t1/t2 (6e-f: hist+E0c0e from t1, E0cf6/E0cfa
                               from t2), encodes identical to frame 0 (6g).

  2026-10-27 rr-bil-fix: the frame-1 border residual was ONE spec error, fixed here - the CR
  bilinear's edge addressing (SPEC 6c, _cr_bilinear). Before the fix, _cr_bilinear clipped the
  lower texel in place and derived the upper texel from the CLIPPED index (the double-clip flagged
  in 972efdc, byte-for-byte SR's bug): at the left/top border the main tap blended texels 0/1 with
  the full sub-texel fraction instead of being pure texel 0. The hardware sampler clamps the
  NORMALIZED coordinate = per-tap PER-TEXEL clamping (RR's own ISA: image_sample_lz x5 at
  0x173c..0x179c, coords from the UNCLIPPED integer position, no software clamp; sampler
  hipAddressModeClamp + bilinear + normalized, rrref.cpp:351-362 - the same class as sr_post, whose
  measurement is build/agents/post-residual/progress.md). No constant touched; the law is the
  sampler's. Evidence (build/agents/rr-bil-fix/): model self-check frame 1 99.94-99.9964% (was
  99.63-99.78); GPU fixture frame 1 GPU==oracle 99.98-99.999 (was 99.85-99.89); the post_prep
  leverage row f1/f2/f3 75.39/70.01/66.53 -> 101.59/100.90/74.59 (min 63.48 -> 70.01, mean 76.26 ->
  85.99); the full-9 no-mask chain is unchanged at report precision (the changed border band is
  34-79 scalars/frame, max |delta| 0.0018 f32 linear - a <0.005 dB effect; the post-mask row is
  117.04-118.83 before and after, unchanged by construction).
  Probe notes: build/agents/rr-post-model-2/progress.md (incl. the patchprobe frame-0 poisoning
  caveat: a probe whose patch fires in frame 0 changes frame 1's env; probes listed in A3 below are
  the ones validated for their env).

A3 VERIFICATION TABLE (stage, what it was checked against, match):
  stage 1  tap geometry px/py/dx0/dy0             hardware probes rrp2_px (+0xc4), rrp2_pye (+0x98),
            rrp2_dx0 (+0x358), rrp2_dy0 (+0x2e4): px, py, f16(dx0), f16(dy0) 100.00% bit-exact on all
            216x768 even rows, frames 0 AND 1 (geometry depends only on the launch args)
  stage 2  tap quadratic q, both lanes, tap (-1,-1)  hardware probe rrp2_q (+0x48c, v12 packed):
            even lane 100%, odd lane 100% over all 216x768 pairs of frame 0
  stage 3  weights, f16 numerators, Wsum            hardware probe rrp2_inv (+0x18d4, v24 = 1/Wsum):
            100.00% f16-exact frame 0 even rows; oracle E0cf6 R/G and E0cfa 99.96-99.97% (frame 0)
  stage 4  normalised average S = N * (1/Wsum)       hardware probe rrp2_S (+0x18d4, v4 = S_R):
            99.97% f16-exact frame 0 even rows; oracle E0c0a R/G/B 99.97-99.98% (frame 0)
  stage 5  packed E0c0e = 2x2 average of decode(S)   oracle E0c0e 99.99% (frame 0)
  stage 6  history endpoint + blend + CR            - rrp2_hsrc (+0x2da0): v5 = decode(h)/e1 (frame 1
              probe env has zero history: 4.571886e-6 == decode(0)/e1) and = decode(S)/e0 on frame 0
              (so the frame-0 blend is an identity via H == cur);
            - rrp2_t (+0x2dac): t1 = fma(conf,1-u1,u1) with conf = clamp01((dec0*Wsum)*(inv*f16(n_w0))),
              u1 = inv*f16(n_w1): 100.00% <=1e-4, 99.99% <=1e-6, 98.06% bit-exact (frame 1 probe env);
            - rrp_xG/rrp_xR/rrp_xB (old, +0x2e9c..): x1 = fma(cur-H,t1,H) per channel: 99.97-99.98%
              <=1e-6, 99.99% <=1e-4 (probe env; the probe NAMES xR/xG are channel-swapped);
            - CR fetch structure: previous-agent probes rrp_posx/posy (pos), rrp_t5tmp (weights,
              oracle env) + the ISA (SPEC 6a-c); the t2/E0cf6-vs-hist output routing was closed
              against the ORACLE itself: tm(x1) gives only 21-27% on frame-1 E0cf6/E0cfa while
              tm(x2) gives 99.63-99.70% (hist keeps tm(x1) at 99.78%).
            - end-to-end vs oracle frame 1 (A2): E0cf6 99.63%, E0cfa 99.70%, E0c0a 99.78%, E0c0e 99.74%
  CAUTION: probe runs whose patch fires in frame 0 poison frame 1's env (frame1 tap md5 e05b4237 !=
  oracle 7eaaa924); progress.md section 3 lists which probes are valid for which env.

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

6. Frame 1 (history): VERIFIED (A2 PASS: E0cf6 99.6293%, E0cfa 99.6977%, E0c0a 99.7825%, E0c0e 99.7372%).
   Every formula below is an immediate of the ISA or a probe-fitted value (A3 lists the probes).

   a. CR position (per OUTPUT pixel, both rows): pos = (x15 + 0.5 + mvx, y15 + 0.5 + mvy), f32.
      mv from the parity-selected record (PostArgs 0x58 = Q(0xbf2 + 4*(parity^1)): frame 0 -> E0bf6,
      frame 1 -> E0bf2); ALL FOUR mv records (E0bf2/E0bf6/E0bfa/E0bfe) are identical and constant in
      the oracle: (-0.2861328125, -0.07501220703125), f16 (probed: rrp_posx/posy at 0x1514).
      On-screen guard: 0 <= pos.x <= W15 and 0 <= pos.y <= H15 (ISA 0x1520/0x1528 tests). On frames
      0/1 the mv is sub-texel so every pixel is on-screen; the off-screen fallback is NOT exercised
      (cr_history raises if it would fire - not modelled, reported per the brief).

   b. CR table index (per axis): f = clamp(pos - (floor(pos - 0.5) + 0.5), 0, 1)  [the `max(a,a) clamp`
      at 0x159c = clamp to [0,1], the same modifier as the conf clamp]; k = min(512, trunc(fma(512, f, 0.5)))
      (0x15b4/0x15e0/0x15f0); entry = E0d02[k] (513 x 4 f32, bicubic.bin) = (wm1, a = w0+w1, w2, r).

   c. The five taps (image_sample_lz x5; sampler = bilinear + clamp-to-edge + NORMALIZED coords,
      rrref.cpp:351-362), Ix = floor(pos.x-0.5), Iy = floor(pos.y-0.5):
        t5 main   (Ix+0.5+rx, Iy+0.5+ry)                 weight W5 = ax*ay
        t1        (Ix-0.5,    main y)  (pure centre x-1)  weight W1 = wm1x*ay
        t2        (Ix+2.5,    main y)  (pure centre x+2)  weight W2 = w2x*ay
        t3        (main x, Iy-0.5)     (pure centre y-1)  weight W3 = ax*wm1y
        t4        (main x, Iy+2.5)     (pure centre y+2)  weight W4 = ax*w2y
      The bilinear sub-texel fraction is the table's r quantised to 8 bits ROUND-TO-NEAREST
      (rq = round(r*256)/256): the same GPU sampler precision measured on sr_post (whose -1/+2 taps
      are likewise pure centres); pure taps pass fraction 0. The +2/-1 taps are texel CENTRES, i.e.
      the taps at (Ix-0.5)/ (Ix+2.5) have no x-blend (only the y blend where stated).
      wsum add order (ISA 0x17ac-0x17c0): (((W3 + W1) + W5) + W4) + W2
      numerator (fma chain, ISA 0x183c-0x18ac): num = t1*W1; num = fma(t3, W3, num);
        num = fma(t5, W5, num); num = fma(t4, W4, num); num = fma(t2, W2, num)
      h = num / wsum (f32).  (Previous-agent probes rrp_t1R..t5, rrp_W3, rrp_inv confirm the taps.)

   d. History endpoint (probe rrp2_hsrc +0x2da0):     v5 = decode(h) / e1      (f32 division)
                                                      H  = e0 * v5              (f32)
      i.e. H = e0 * decode(h) / e1 with e0 = E0c22[0], e1 = E0c22[1]; decode(h) clamps to [1e-4, 3]
      exactly like decode(S) (SPEC 5). E0c22 = (e0, e1, 1/e0, ...): frame 0 = (1.3994141, 1.0, ...),
      frame 1 = (1.390625, 1.3994141, ...). On frame 0 (rst) the CR blocks are skipped and the
      register holds decode(S)/e0 instead, so H = e0*(decode(S)/e0) = cur - the frame-0 blend is an
      identity for ANY t (this is why frame 0 needs no history term).

   e. TWO blend weights (shared by the 3 channels; the sr_post two-weight pattern; conf from gate0,
      u1 from gate1, u2 from gate2 - tap fields 3/4/5, i.e. the post_prep w0/w1/w2):
        conf    = clamp01( f32(f32(dec0 * Wsum) * f32(inv * f16(n_w0))) ), NaN -> 0
                  dec0 = PostArgs 0x34 = 1.126047134399414 (ISA 0x2b58); the clamp is the
                  `v_max_num(a,a) clamp` at 0x2d7c (same modifier as 6b); n_w0 = the f16 numerator of
                  gate field 3 (stage-4 chain); inv = 1/Wsum (hardware division, stage 3).
        u1      = f32(inv * f16(n_w1))         (n_w1 = field 4; ISA 0x2d30/0x2d70)
        t1      = fma(conf, 1 - u1, u1)         (ISA 0x2dac)   [probe rrp2_t: 100.00% <=1e-4]
        u2      = f32(inv * f16(n_w2))         (n_w2 = field 5; ISA 0x2e08/0x2e14)
        t2      = fma(conf, 1 - u2, u2)         (ISA 0x2ecc op2, v24)

   f. The two blended colours (per channel, one fma each; NaN guard: if the 3-channel sum of the
      selected x is unordered the hardware keeps the unblended decode(S)):
        x1 = fma(cur - H, t1, H)     (ISA 0x2dd4/0x2dc8/0x2df4 with t = t1)  -> HISTORY + E0c0e
        x2 = fma(cur - H, t2, H)     (0x2efc/0x2ef8/0x2f00 with t = t2)       -> E0cf6/E0cfa
        cur = decode(S) (SPEC 5).  Routing proved against the oracle: on frame 1 tm(x1) matches the
        history write at 99.78% but E0cf6/E0cfa only at 21-27%, while tm(x2) matches E0cf6/E0cfa at
        99.63/99.70% (both blends share the same H and cur; only the t differs).

   g. Outputs (frame-0 encode chains, unchanged):
        E0cf6 = f16(tm(x2_R)) | f16(tm(x2_G)) << 16        (tm = (log2(max(100x+1,1e-4))*log10(2))*0.36)
        E0cfa = f16(tm(x2_B))
        history write (E0c06 on frame 1, E0c0a on frame 0) = f16(tm(x1_R/G/B)), alpha 0
        E0c0e = frame-0 encode of the 2x2 mean of x1 (the 2x2 gather at 0x31xx reads the x1 packs)
      Frame 0: x1 == x2 == decode(S) by (d), so all four outputs reduce to SPEC 5 (A1 unchanged).

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
    inp = dict(frame=frame, fdir=fdir, launch=launch, sx=F(sx), sy=F(sy), jx=F(jx), jy=F(jy), rst=rst,
               expo=expo, tap=tap, ptr=ptr)
    if frame == 1:
        # frame-1 only inputs: the previous history (frame0's E0c0a output), the CR weight table,
        # the MV (parity selects E0bf2 on frame1; all MV candidates are identical in these frames),
        # and dec0 (PostArgs 0x34, constant).
        inp["hist"] = np.frombuffer(buffer_before(fdir, "E0c0a", launch), np.float16).reshape(H15, W15, 4)
        inp["cr_table"] = np.frombuffer(buffer_before(fdir, "E0d02", launch), np.float32).reshape(513, 4)
        mv = np.frombuffer(buffer_before(fdir, "E0bf2", launch), np.float16)
        inp["mv"] = (F(mv[0]), F(mv[1]))
        inp["dec0"] = F(struct.unpack_from("<f", a, 0x34)[0])
    return inp


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
    """S[lane][pair, x15, field] for fields 0..5 (colour tm, gates), lane 0 = even row, 1 = odd row.
    Also returns res[("w", lane)] = Wsum (f32) and res[("n", lane)] = the f16 numerators (216,768,6),
    the inputs of the frame-1 blend weight t; frame-0 S arithmetic is unchanged."""
    res = {}
    for lane, qk in ((0, "qe"), (1, "qo")):
        ws = {k: np.exp2(tp[k][qk].astype(F)).astype(F) for k in KX}
        wsum = np.zeros(ws[KX[0]].shape, F)
        for k in KX:
            wsum = (wsum + ws[k]).astype(F)
        inv = (F(1.0) / wsum).astype(F)
        S = np.zeros(ws[KX[0]].shape + (6,), F)
        N = np.zeros(ws[KX[0]].shape + (6,), np.float16)
        for fi in range(6):
            n = np.zeros(ws[KX[0]].shape, F)
            for k in KX:
                n = f16(n + f16(ws[k]).astype(F) * tp[k]["rec"][..., fi].astype(F)).astype(F)
            N[..., fi] = n
            S[..., fi] = (n * inv).astype(F)
        res[lane] = S
        res[("w", lane)] = wsum
        res[("n", lane)] = N
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


# ------------------------------------------------------------------ frame 1: history fetch + blend
def _cr_bilinear(hist, xi, yi, rx, ry):
    """Bilinear sample of hist (H15, W15, 3) f32 at texel lower-left (xi, yi) with sub-texel weights
    rx, ry (f32, broadcast over the channel axis).

    ADDRESSING (rr-bil-fix 2026-10-27; the hardware sampler clamps the NORMALIZED coordinate, which
    is per-tap PER-TEXEL clamping - the law measured on sr_post's identical sampler, and RR's own
    ISA confirms the same class: image_sample_lz x5 at 0x173c..0x179c with normalized coords built
    from the UNCLIPPED integer position and no software clamp; sampler = bilinear +
    hipAddressModeClamp + normalized coords, rrref.cpp:351-362): the two texel indices of each
    axis are clamped to [0, W15-1] / [0, H15-1] INDEPENDENTLY, both derived from the UNCLIPPED
    integer position - a tap whose lower texel is -1 blends texel 0 with texel 0 (pure texel 0),
    NOT texel 0 with texel 1. The previous law clipped xi in place and then took xi+1, which at
    the left/top borders blended the wrong pair with the full sub-texel fraction (the SR-side
    double-clip of 972efdc, present here byte-for-byte)."""
    x0 = np.clip(xi, 0, W15 - 1); y0 = np.clip(yi, 0, H15 - 1)
    x1 = np.clip(xi + 1, 0, W15 - 1); y1 = np.clip(yi + 1, 0, H15 - 1)
    one = F(1.0)
    if np.ndim(rx):
        rx = rx[..., None]
    if np.ndim(ry):
        ry = ry[..., None]
    h00 = hist[y0, x0]; h01 = hist[y0, x1]
    h10 = hist[y1, x0]; h11 = hist[y1, x1]
    top = (h00 * (one - rx) + h01 * rx).astype(F)
    bot = (h10 * (one - rx) + h11 * rx).astype(F)
    return (top * (one - ry) + bot * ry).astype(F)


def cr_history(inp):
    """h (H15, W15, 3) f32: the 5-tap bilinear Catmull-Rom fetch of the previous history at
    pos = (x15 + 0.5 + mvx, y15 + 0.5 + mvy) per OUTPUT pixel (SPEC section 6). The ISA:
    f = clamp(pos - (floor(pos-0.5) + 0.5), 0, 1); k = min(512, trunc(fma(512, f, 0.5)));
    entry = E0d02[k] = (wm1, a, w2, r); the bilinear sub-texel fraction = quantised r (8 bits,
    round-to-nearest - the measured sampler precision of SR post, same GPU sampler); the -1/+2
    taps are pure texel fetches at the texel centres (fraction 0)."""
    mvx, mvy = inp["mv"]
    tab = inp["cr_table"]
    hist = inp["hist"][..., :3].astype(F)
    px = (np.arange(W15, dtype=F)[None, :] + F(0.5)) + mvx        # (1, W15)
    py = (np.arange(H15, dtype=F)[:, None] + F(0.5)) + mvy        # (H15, 1)
    if not (bool(np.all(px >= 0)) and bool(np.all(px <= F(W15))) and
            bool(np.all(py >= 0)) and bool(np.all(py <= F(H15)))):
        raise NotImplementedError("off-screen history fetch (never fires on oracle frames 0/1: mv sub-texel)")
    Ix = np.floor(px - F(0.5)).astype(np.int64)
    Iy = np.floor(py - F(0.5)).astype(np.int64)
    fx = (px - (Ix.astype(F) + F(0.5))).astype(F)
    fy = (py - (Iy.astype(F) + F(0.5))).astype(F)
    fcx = np.clip(fx, F(0.0), F(1.0))
    fcy = np.clip(fy, F(0.0), F(1.0))
    kx = np.minimum(fma(F(512.0), fcx, F(0.5)).astype(np.int64), 512)
    ky = np.minimum(fma(F(512.0), fcy, F(0.5)).astype(np.int64), 512)
    ex = tab[kx]                                     # (1, W15, 4): (wm1, a, w2, r) per x
    ey = tab[ky]                                     # (H15, 1, 4): per y
    rx = ex[..., 3]; ry = ey[..., 3]
    rxq = (np.round(rx.astype(np.float64) * 256) / 256.0).astype(F)
    ryq = (np.round(ry.astype(np.float64) * 256) / 256.0).astype(F)
    ix, iy = np.broadcast_arrays(Ix, Iy)
    rxq, ryq = np.broadcast_arrays(rxq, ryq)
    z = F(0.0)
    t1 = _cr_bilinear(hist, ix - 1, iy, z, ryq)      # pure texel x-1, main y
    t2 = _cr_bilinear(hist, ix + 2, iy, z, ryq)      # pure texel x+2, main y
    t3 = _cr_bilinear(hist, ix, iy - 1, rxq, z)      # main x, pure texel y-1
    t4 = _cr_bilinear(hist, ix, iy + 2, rxq, z)      # main x, pure texel y+2
    t5 = _cr_bilinear(hist, ix, iy, rxq, ryq)        # the main bilinear
    W1 = (ex[..., 0] * ey[..., 1])                   # wm1x * ay
    W2 = (ex[..., 2] * ey[..., 1])                   # w2x  * ay
    W3 = (ex[..., 1] * ey[..., 0])                   # ax   * wm1y
    W4 = (ex[..., 1] * ey[..., 2])                   # ax   * w2y
    W5 = (ex[..., 1] * ey[..., 1])                   # ax   * ay
    num = (t1 * W1[..., None]).astype(F)
    num = fma(t3, W3[..., None], num)
    num = fma(t5, W5[..., None], num)
    num = fma(t4, W4[..., None], num)
    num = fma(t2, W2[..., None], num)
    wcr = (W3 + W1).astype(F)                        # ISA add order: (((W3 + W1) + W5) + W4) + W2
    wcr = (wcr + W5).astype(F)
    wcr = (wcr + W4).astype(F)
    wcr = (wcr + W2).astype(F)
    return (num / wcr[..., None]).astype(F)


def blend_weight(inp, sp):
    """(t1, t2)[lane] (216, 768): the TWO blend weights of the pair (the gate1/gate2 pattern of
    sr_post, ISA 0x2b58/0x2d30-0x2ecc):
      conf    = clamp01((dec0 * Wsum) * (inv * f16(n_w0)))   (gate field 3; NaN -> 0)
      u1      = inv * f16(n_w1)      (gate field 4)  -> t1 = fma(conf, 1-u1, u1) -> the HISTORY write
      u2      = inv * f16(n_w2)      (gate field 5)  -> t2 = fma(conf, 1-u2, u2) -> E0cf6/E0cfa
    t1 fitted100.00% <=1e-4 against hardware probe rrp2_t (+0x2dac; progress.md 6); t2 is the
    same formula on the w2 chain (%531/%1516/%1599/@0x2ecc) - E0cf6/E0cfa are tm(x2) (SPEC 6)."""
    dec0 = inp["dec0"]
    out = {}
    for lane in (0, 1):
        wsum = sp[("w", lane)]
        n = sp[("n", lane)]
        inv = (F(1.0) / wsum).astype(F)
        conf_raw = ((dec0 * wsum) * (inv * f16(n[..., 3]).astype(F))).astype(F)
        conf = np.where(np.isfinite(conf_raw), np.clip(conf_raw, F(0.0), F(1.0)), F(0.0)).astype(F)
        one = F(1.0)
        u1 = (inv * f16(n[..., 4]).astype(F)).astype(F)
        u2 = (inv * f16(n[..., 5]).astype(F)).astype(F)
        out[lane] = (fma(conf, one - u1, u1), fma(conf, one - u2, u2))
    return out


def frame_outputs_from_x(x1, x2):
    """The four outputs of frame 1 from the two blended linear colours (SPEC section 6):
    the history write and E0c0e use x1 (t1 blend), E0cf6/E0cfa use tm(x2) (t2 blend) -
    the hardware has two tm pack chains (0x2e9c.. for hist, 0x2f4c.. for E0cf6/E0cfa);
    on frame 0 x1 == x2 == cur so this reduces to the frame-0 law."""
    tm2 = tonemap(x2)
    e_cf6 = (f16(tm2[..., 0]).view(np.uint16).astype(np.uint32)
             | (f16(tm2[..., 1]).view(np.uint16).astype(np.uint32) << 16))
    e_cfa = f16(tm2[..., 2])
    tm = tonemap(x1)
    hist = np.zeros((H15, W15, 4), np.float16)
    hist[..., :3] = f16(tm)
    x = x1
    xb = x.reshape(H15 // 2, 2, W15 // 2, 2, 3)
    xs = (xb[:, 0] + xb[:, 1]).astype(F)
    xs = ((xs[:, :, 0] + xs[:, :, 1]) * F(0.25)).astype(F)
    tmb = tonemap(xs)
    val = encode10(tmb[..., 0]) | (encode10(tmb[..., 1]) << 10) | (encode10(tmb[..., 2]) << 20)
    e_c0e = (val | np.uint32(0xC0000000)).astype(np.uint32)
    return dict(E0cf6=e_cf6, E0cfa=e_cfa, E0c0a=hist, E0c0e=e_c0e)


def model(frame):
    """frame 0 (rst): the frame-0 law. frame 1: h = CR history, H = e0*decode(h)/e1,
    x = fma(cur - H, t, H) per channel with cur = decode(S) and the gate weight t (SPEC 6)."""
    inp = load_inputs(frame)
    tp = taps(inp)
    sp = spatial(inp, tp)
    if frame == 0:
        return frame_outputs_frame0(interleave({k: v for k, v in sp.items() if isinstance(k, int)})), inp
    h = cr_history(inp)
    e0, e1 = F(inp["expo"][0]), F(inp["expo"][1])
    t1t2 = blend_weight(inp, sp)
    x1 = np.zeros((H15, W15, 3), F)
    x2 = np.zeros((H15, W15, 3), F)
    for lane in (0, 1):
        S = sp[lane]
        cur = decode_lin(S[..., :3])
        rows = slice(lane, None, 2)
        v = (decode_lin(h[rows]) / e1).astype(F)      # the hardware computes decode(h)/e1 (probe rrp2_hsrc)
        Hh = (e0 * v).astype(F)                      # H = e0 * decode(h) / e1 (f32, probe-confirmed shape)
        t1, t2 = (tt[..., None] for tt in t1t2[lane])
        d = np.float64(cur) - np.float64(Hh)
        hh = np.float64(Hh)
        x1[rows] = (d * np.float64(t1) + hh).astype(F)
        x2[rows] = (d * np.float64(t2) + hh).astype(F)
    return frame_outputs_from_x(x1, x2), inp


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
    print("[frame 1, launch 22: the model]")
    r1 = run(1)
    for k, v in r1.items():
        good = v >= thr[1]
        ok &= good
        print(f"  -> {k} {v:.4f}% >= {thr[1]:.1f}%: {'PASS' if good else 'FAIL'}")
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
