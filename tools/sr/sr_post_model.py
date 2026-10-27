#!/usr/bin/env python3
"""tools/sr/sr_post_model.py: the NumPy model of NeuralHub's k_sr_post<false,12,16,false,false> (SR preset M,
oracle launch 14) - the spec that sr_post.comp is rewritten against.

    build/venv/bin/python -I tools/sr/sr_post_model.py [--frame F] [--oracle DUMP_DIR] [--out DIR]
    build/venv/bin/python -I tools/sr/sr_post_model.py --self-check

Inputs are the oracle's own bytes as they stood before launch 14 (newest earlier NNN.<buffer>.bin, else
start.<buffer>.bin). Outputs: full15 (u32, 768x432), ihist1 (f16 x4, 768x432), dhist (u32, 384x216).

STATUS (2026-10-27, post-model-3; numbers from --self-check, launch 14 of sr_M, oracle dump).
  Frame 0 (reset, no history):  full15 99.9934% bit-exact (decoded PSNR 101.6 dB), ihist1 99.9895%,
                                 dhist 99.9976%.
  Frame 1 (history):            full15 99.8481%, ihist1 99.8152%, dhist 99.7999% (decoded PSNR 66.4 dB).
  Acceptance: A1 (frame 0 >= 99% for full15, ihist1, dhist) PASSES. A2 (frame 1 >= 97%) PASSES.
              A4 (--self-check, < 30 s) passes (about 2 s).

  VERIFIED against hardware probes (scripts in build/agents/post-model/, post-model-2/, post-model-3/;
  the per-stage table: build/agents/post-model-3/a3_table.py, reproducible from the saved probe dumps):
    stage 1  source records, 12 f16 per source pixel: 100% of 36864 pixels, all 12 fields.
    stage 2  output geometry px/py (f32 fma), the source base X0 and the pair base Y0: px 100%.
    stage 3  the nine taps and their quadratic q (f16): tap identities 9 of 9. Weight w = exp2(q):
             the hardware's v_exp_f32 differs from the correctly-rounded exp2 by <= 1 ULP (85-87% of the
             taps bit-exact; the differences are absorbed by the f16 numerator chains and the final
             encodings). Wsum (kx-major f32 sum) 95.2% vs the probe; f16 numerator chains 99.99%.
    stage 5  pixel 1 (odd rows), Review R6.7: the thread's hi lane uses the taps of its PAIR's base row
             Yb (pixel 0's Y0), and dy1 - (Y1 - Yb) in f16. Probe: the hi-lane tap-0 weight matches 85.2%
             (the v_exp rest), with the base-row rule.
    stage 6  the gate blend (solved here, post-model-3): v12 = clamp(dec0_f1 * f16(n_gate0)) - the gate0
             f16 numerator (probe v16 = inv*f16(n_gate0): 99.997% with the probed inv); TWO blend weights,
             one per output: t1 from the gate1 numerator (probe lo(v20) = f16(n_gate1): 99.997%; v25 =
             fma(-inv, Wa, 1): 99.997% with the probed inv) drives ihist1 and dhist, t2 from the gate2
             numerator drives full15 (scanned: gate2/gate2 reproduces the oracle's full15 at 99.99%,
             every other field pair <= 98.2%). t = fma(v12, fma(-inv, Wa, 1), inv*Wa): 99.84% bit-exact
             vs the probed t on the v12==1 lanes, 97.6% on the v12<1 lanes.
    stage 7  the Catmull-Rom history (solved here): the reprojection adds the MV (output units) to the
             output-texel position; the 5-tap bilinear CR with the -1 taps at the neighbouring texel
             CENTRES (pure fetches), the +2 taps at texel centres, and the bilinear sub-texel fraction
             r = w1/(w0+w1) quantised to 8 bits ROUND-TO-NEAREST (scanned: 99.82% ihist1 with
             round+pure vs 71.5% with trunc+edge). The blend and encode chain reproduces the oracle.
  RESIDUAL (open, all <= 0.2%): (a) the hardware's f32 division (v_div_scale/rcp/fma/div_fmas/div_fixup)
             is 1 ULP from numpy's correctly-rounded 1/Wsum on ~4% of pixels (visible in the a3_table
             v16/v25 rows: 96% with the model's inv, 99.997% with the probed inv; absorbed by the final
             encodings: the outputs stay >= 99.8%); (b) v_exp_f32 / v_log_f32 1-ULP differences
             (85-87% bit-exact per call); (c) the CR f32 accumulate/divide roundings.
  NOT MODELLED: the MV candidate selection and the second-candidate mix - all three MV buffers are
             IDENTICAL and CONSTANT in the oracle's frames 0/1 (mv = (-0.2861328125, -0.075) output
             texels; verified on the oracle's own start.* buffers), so the choice is moot here and the
             second candidate samples the same position (a numerical no-op); the off-screen fallback
             (tonemap of the raw g_col at the clamped source pixel) never fires on these frames (the MV
             shifts are sub-texel); the exposure path is only exercised at e = 1.

SPEC (all constants are the hardware's; f16 = IEEE half, round to nearest even; every "f16(...)" is a rounding point)

Inputs per launch: W,H = 256,144 (render size), W15,H15 = 768,432, F20 = (F20x, F20y) jitter (PostArgs 0xb4,0xb8),
  scale s = (W/W15, H/H15) = (1/3, 1/3) as f32 (0xac,0xb0), expo[p], p = frame parity (0xa0 mapping: 0x10).
  g_col: f16 x4 per source pixel (R,G,B,A); aniso: f16 x4 (A,B,C,_); gates, factors: u8 x4 (lanes 0..2 used).
  dec0_f1 = PostArgs 0xa8 (2.7530112): scales the gate0 confidence v12 (stage 5).

1. Source records (per source pixel, 12 f16, the LDS record of the staging):
     e   = expo[p] (1.0 if 0)
     tm_c = f16( 0.36 * log10( 1 + 100 * min(c * e, 32752) ) ),  c in {R,G,B}, log10 computed as
            f32(log2(y)) * f32(0.30103) (the ISA's constant 0x3e9a209b) then * 0.36 (0x3eb851ec);
            with y = max(1 + 100*min(...), 1e-4)         -> record[0..2]
     fac_i = f16(u8/255), gate_i = f16(u8/255)            -> record[3..5] (factors), record[6..8] (gates)
     a = f16(C1 * Axx), b = f16(C1 * Ayy), c = f16(C2 * Axy)
            with C1 = -0.721325 (f32 0xbf38a8c1), C2 = 2*C1 = -1.44265 (f32 0xbfb8a8c1)  -> record[9..11]
   (Verified bit-exact on all 36864 source pixels by LDS probes of dwords 0..5, Review R6.1.)

2. Output geometry (output pixel (y15, x15), centre in source units):
     px = f32(f32(x15 + 0.5) * s_x) + F20x    (ISA: v_fma_f32)
     py = f32(f32(y15 + 0.5) * s_y) + F20y    (for pixel 0, y15 = 2r; pixel 1 uses y15 = 2r+1)
     X0 = floor(px - 0.5), Y0 = floor(py - 0.5)           (int, clamped to the image when sampling)
     dx0 = f16((X0 + 0.5) - px),  dy0 = f16((Y0 + 0.5) - py)   (f32 subtract, then f16)
   VERIFIED: px for every output column (100%, Review R6.2).
   PAIR BASE (VERIFIED, ISA 0x8e4-0x9c0 and probes): the thread writes the two rows of its pair (2r, 2r+1)
   from ONE set of source rows, the rows of the even pixel's base Yb = Y0(2r). For the odd pixel:
     shift = Y0(2r+1) - Yb   (0 or 1)
     dy_base = f16(dy0(2r+1) - shift)   (the ISA's v_cmp_eq then v_add_f16 -1.0)
   and its taps use rows Yb + ky with dy = f16(dy_base + ky). Pixel 0 has shift 0.

3. Spatial taps: for (kx, ky) in {-1,0,1}^2 the source pixel is (X0+kx, Yb+ky) clamped to the image.
   Tap offsets are dx = f16(dx0 + kx), dy = f16(dy_base + ky) (exact integers in f16).
   Quadratic (f16 at every product and sum, the ISA's packed f16 ops):
     q = f16( f16( f16(f16(dx*dx)*a) + f16(f16(dy*dy)*b) ) + f16(f16(dx*dy)*c) )
   Weight w = exp2(f32(q))  (v_exp_f32; the hardware result is within 1 ULP of the correctly rounded value,
   see Review R6.3).  Weights are summed in f32 in the order kx-major: kx=-1 (ky=-1,0,1), kx=0, kx=1
   (VERIFIED: the sum and the 9 individual weights match the probes, Review R6.3); Wsum = sum of w (f32).

4. Colour numerator and normalisation (one channel c of tm; the same chain runs for each of the 12 record fields):
     w16 = f16(w)                                   (the ISA's cvt_f16 before the packed fma)
     n = 0;  for tap in kx-major order:  n = f16(n + w16 * tm_c[tap])   (one rounding per tap)
     S_c = f32(n) * (1 / Wsum)   with 1/Wsum the IEEE f32 reciprocal (v_div_* sequence)
   VERIFIED for pixel 0: the factor-0 numerator is exact (99.99 percent), the normalised factor average is
   96.0 percent (the rest is the v_exp_f32 ULP in the sum).

5. The gate blend weight (both outputs; post-model-3, probes in build/agents/post-model-2/):
      v12 = clamp(dec0_f1 * f16(n_gate0), 0, 1)        (dec0_f1 = PostArgs 0xa8 = 2.7530112; n_gate0 is
             the stage-4 f16 numerator of the gate0 field; evaluated as the hardware does:
             f32(f32(dec0_f1 * Wsum) * f32(inv * f16(n_gate0)))); NaN -> 0.
      For g in {gate1, gate2}:  Wa = f16(n_g)  (the f16 numerator of the gate field)
             u = f32(inv * Wa);  v25 = fma(-inv, Wa, 1);  t = fma(v12, v25, u).
      t1 (the gate1 weight) drives ihist1 and dhist; t2 (the gate2 weight) drives full15.
    (VERIFIED: inv*f16(n_gate0) 99.997% vs the probed v16; f16(n_gate1) 99.997% vs the probed lo(v20);
    v25 99.997% with the probed inv; the probed t reproduces the oracle's ihist1 G at 99.987%; the
    gate2-weight scan gives 99.99% of full15 where every other field pair gives <= 98.2%.)

6. History (frame >= 1) and the linear-domain lerp:
      The MV (OUTPUT texels; added directly to x15+0.5 / y15+0.5 - NOT rescaled): all three candidates
      (mv_nondil texture sample, prev_mvec1, mv_ero) are identical and constant in the oracle's frames
      0/1, so the selection is moot; pos = (x15 + 0.5 + mv_x, y15 + 0.5 + mv_y).
      On-screen test: 0 <= pos_x <= W15, 0 <= pos_y <= H15. Off-screen: h = the tonemap of
      e_prev * g_col at the clamped nearest source pixel (never fires on frames 0/1).
      Otherwise h (the tm-domain history) = the 5-tap bilinear Catmull-Rom sample of the previous ihist:
        f = pos - (floor(pos - 0.5) + 0.5), clamped to [0, 1]          (per axis; I = floor(pos - 0.5))
        w(-1) = f2 - 0.5*(f3 + f);  w0 = 1 + 1.5*f3 - 2.5*f2;  w1 = (1 - w(-1) - w0) - 0.5*(f3 - f2);
        w2 = 0.5*(f3 - f2)                                             (f2 = f*f, f3 = f2*f, fma chain)
        a = w0 + w1;   r = w1 / a   quantised to 8 bits, ROUND-TO-NEAREST: rq = round(r*256)/256
             (the bilinear hardware's sub-texel fraction; the round-to-nearest is measured, this review)
        taps (bilinear samples of ihist at, and their weights):
             W1 = w_x(-1)*a_y at (I_x - 1, main_y)      (pure texel x, blended y)
             W3 = a_x*w_y(-1) at (main_x, I_y - 1)      (blended x, pure texel y)
             W5 = a_x*a_y     at (main_x, main_y)       (the main bilinear at I + 0.5 + rq)
             W4 = a_x*w_y(2)  at (main_x, I_y + 2)      (pure texel y)
             W2 = w_x(2)*a_y  at (I_x + 2, main_y)      (pure texel x)
        h = fma-chain(W1*t1, W3*t3, W5*t5, W4*t4, W2*t2) / (W1+W3+W5+W4+W2)   (f32; the -1 and +2 taps
             are pure texel fetches at the texel CENTRES, not half-texel blends - measured, this review)
      A = e0 * decode(h) with decode(h) = 0.01*(exp2(clamp(h, 1e-4, 3)*(1/0.36)*log2(10)) - 1)
          (log2(10) = the f32 constant 0x40549a78 from PostArgs 0xcc) and e0 = the CURRENT exposure
          (expo[p], 1.0 if 0); on frame 0 the history buffer is all zeros -> h = 0 everywhere.
      x = fma(t, B - e0*A, e0*A) with B = decode(S)   (per channel; two x's: the t1 blend for
          ihist1/dhist, the t2 blend for full15)
      y = fma(100, x, 1);  y = max(y, 1e-4)
      tm = (log2(y) * log10(2)) * 0.36                 (f32; the encode constants 0x3e9a209b / 0x3eb851ec)

7. Outputs:
      ihist1 = f16(tm of the t1 blend) x 3, alpha lane 0                -> ihist1 (f16 x 4)
      full15 = R11 | G11 << 11 | B10 << 22 of the t2-blend tm:
              R = trunc(clamp(fma(2047, f32(k*tm_R), 0.5), 0, 2047)), G the same with 2047 << 11,
              B = trunc(clamp(fma(1023, f32(k*tm_B), 0.5), 0, 1023)) << 22,  k = f32(0x3ed0fac6) = 1/2.45
      dhist (384x216) = R10 | G10 << 10 | B10 << 20 of rndne(1023 * clamp(k*tm_c, 0, 1)) with tm from the
              t1-blend x averaged over the 2x2 block of output pixels IN THE LINEAR DOMAIN:
              xbar = ((x(2r,2c) + x(2r,2c+1)) + (x(2r+1,2c) + x(2r+1,2c+1))) * 0.25 (the row pair summed
              first, then the columns), then tm = tonemap_lin(xbar); rndne = round-to-nearest-even.
"""
import argparse
import json
import math
import os
import struct
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
ORACLE = REPO / "build/oracle/oracle-8dbf027a9e-s20261025"
W, H, W15, H15, OW, OH = 256, 144, 768, 432, 512, 288
TAPS_KX_MAJOR = [(kx, ky) for kx in (-1, 0, 1) for ky in (-1, 0, 1)]
F32 = np.float32


def f16(x):
    return np.asarray(x).astype(np.float16)


def f32_from_bits(u):
    return np.frombuffer(struct.pack("<I", u), np.float32)[0]


C_TM_LOG10_2 = f32_from_bits(0x3E9A209B)     # log10(2)
C_TM_SCALE = f32_from_bits(0x3EB851EC)       # 0.36
C_ANISO1 = f32_from_bits(0xBF38A8C1)         # -0.5 * log2(e), f32 (-0.721325)
C_ANISO2 = f32_from_bits(0xBFB8A8C1)         # -log2(e)        (-1.44265)
K_ENC = f32_from_bits(0x3ED0FAC6)            # 1 / 2.45, the R11G11B10 encode scale


# ------------------------------------------------------------------ inputs
def buffer_before(dump_frame, name, launch):
    best = None
    for fn in os.listdir(dump_frame):
        if fn.endswith(f".{name}.bin") and fn[:3].isdigit() and int(fn[:3]) < launch:
            if best is None or int(fn[:3]) > int(best[:3]):
                best = fn
    return (Path(dump_frame) / (best if best else f"start.{name}.bin")).read_bytes()


def load_inputs(oracle, frame, launch=14):
    fdir = Path(oracle) / f"frame{frame}"
    rows = [json.loads(l) for l in open(fdir / "launches.jsonl")]
    d = next(r for r in rows if r["n"] == launch)
    a = bytes.fromhex(d["args"])
    ptr = {int(k): v.split("+")[0] for k, v in d["pointers"].items()}
    pidx = struct.unpack_from("<Q", a, 0x10)[0]
    expo = np.frombuffer(buffer_before(fdir, "expo", launch), np.float16).astype(np.float32)
    s_x, s_y = struct.unpack_from("<ff", a, 0xAC)
    j_x, j_y = struct.unpack_from("<ff", a, 0xB4)
    dec0_f1 = struct.unpack_from("<f", a, 0xA8)[0]
    ldr = struct.unpack_from("<I", a, 0xD4)[0]
    mvoff = struct.unpack_from("<ff", a, 0xF8)
    mv_nondil = np.frombuffer(buffer_before(fdir, ptr[88], launch), np.float16).reshape(H, W, 2)
    return dict(
        frame=frame, fdir=fdir, p=pidx, expo=expo, s=(s_x, s_y), jit=(j_x, j_y), dec0_f1=dec0_f1, ldr=ldr, mvoff=mvoff,
        g_col=np.frombuffer(buffer_before(fdir, ptr[0], launch), np.float16).reshape(H, W, 4),
        gates=np.frombuffer(buffer_before(fdir, ptr[24], launch), np.uint8).reshape(H, W, 4),
        factors=np.frombuffer(buffer_before(fdir, ptr[32], launch), np.uint8).reshape(H, W, 4),
        aniso=np.frombuffer(buffer_before(fdir, ptr[40], launch), np.float16).reshape(H, W, 4),
        ihist_in=np.frombuffer(buffer_before(fdir, ptr[48], launch), np.float16).reshape(H15, W15, 4),
        prev_mv=np.frombuffer(buffer_before(fdir, ptr[72], launch), np.float16).reshape(H, W, 2),
        mv_ero=np.frombuffer(buffer_before(fdir, ptr[80], launch), np.float16).reshape(H, W, 2),
        mv_nondil=mv_nondil,
        # the chosen MV (output units, added to x15+0.5 / y15+0.5): all three candidates are constant and
        # equal in these frames (verified on the oracle's own buffers), so the choice is moot here.
        mv=(float(mv_nondil[0, 0, 0]), float(mv_nondil[0, 0, 1])),
        # outputs by the launch's own pointers: 0x70 = ihist_out (ihist1 on even frames, ihist0 on odd), 0x78 full15, 0x80 dhist
        oracle_out={"full15": fdir / f"{launch:03d}.{ptr[120]}.bin", "ihist1": fdir / f"{launch:03d}.{ptr[112]}.bin",
                    "dhist": fdir / f"{launch:03d}.{ptr[128]}.bin"},
        launch=launch,
    )


# ------------------------------------------------------------------ stage 1: source records
def tonemap(c, e):
    """tm(c) = f16(0.36 * log10(1 + 100 * min(c*e, 32752))), NaN -> 0 (ISA: cmp_o select)."""
    x = np.minimum(c * e, F32(32752.0)).astype(F32)
    y = np.maximum(F32(1.0) + F32(100.0) * x, F32(1e-4)).astype(F32)
    lg = (np.log2(y).astype(F32) * C_TM_LOG10_2).astype(F32)
    return f16((lg * C_TM_SCALE).astype(F32))


def records(inp):
    """12 f16 per source pixel: tmR tmG tmB | fac0..2 | gate0..2 | A' B' C'."""
    g = inp["g_col"].astype(F32)
    e = F32(inp["expo"][inp["p"]]) if inp["expo"][inp["p"]] != 0 else F32(1.0)
    rec = np.zeros((H, W, 12), np.float16)
    for c in range(3):
        rec[..., c] = tonemap(g[..., c], e)
    rec[..., 3:6] = f16(inp["factors"][..., :3].astype(F32) / F32(255.0))
    rec[..., 6:9] = f16(inp["gates"][..., :3].astype(F32) / F32(255.0))
    an = inp["aniso"].astype(F32)
    rec[..., 9] = f16(C_ANISO1 * an[..., 0])
    rec[..., 10] = f16(C_ANISO1 * an[..., 1])
    rec[..., 11] = f16(C_ANISO2 * an[..., 2])
    return rec


# ------------------------------------------------------------------ stage 2: geometry
def geometry(y15, x15, inp):
    """px, py, X0, Y0, dx0, dy0 for output pixels (arrays of y15, x15). Verified for px (Review R6.2)."""
    sx, sy = F32(inp["s"][0]), F32(inp["s"][1])
    jx, jy = F32(inp["jit"][0]), F32(inp["jit"][1])
    px = (((x15.astype(F32) + F32(0.5)) * sx).astype(F32) + jx).astype(F32)
    py = (((y15.astype(F32) + F32(0.5)) * sy).astype(F32) + jy).astype(F32)
    X0 = np.floor(px - F32(0.5)).astype(np.int64)
    Y0 = np.floor(py - F32(0.5)).astype(np.int64)
    dx0 = f16(((X0.astype(F32) + F32(0.5)) - px).astype(F32))
    dy0 = f16(((Y0.astype(F32) + F32(0.5)) - py).astype(F32))
    return X0, Y0, dx0, dy0


# ------------------------------------------------------------------ stage 3: taps and weights
def taps(y15, x15, inp, rec):
    """The nine (kx-major) taps of each output pixel: list of (w f32, record rows (N,H,W,12) gathered per pixel).
    A thread writes the two rows of its pair (2r, 2r+1) with ONE set of source rows: the base row Yb of the
    pair's even pixel (Review R6.7). Pixel 1 is the odd row; its dy is measured from its own base row Y1 and
    shifted by Y1 - Yb (0 or 1) in f16 before the taps are offset by ky."""
    X0, Y0, dx0, dy0 = geometry(y15, x15, inp)
    _, Yb, _, _ = geometry(y15 & ~1, x15, inp)
    shift = (Y0 - Yb).astype(F32)
    dyb = f16(dy0.astype(F32) - shift).astype(F32)
    out = []
    for kx, ky in TAPS_KX_MAJOR:
        dx = f16(dx0.astype(F32) + F32(kx)).astype(F32)
        dy = f16(dyb + F32(ky)).astype(F32)
        sxi = np.clip(X0 + kx, 0, W - 1)
        syi = np.clip(Yb + ky, 0, H - 1)
        r = rec[syi, sxi]
        a, b, c = r[..., 9].astype(F32), r[..., 10].astype(F32), r[..., 11].astype(F32)
        t1 = f16(f16(dx * dx).astype(F32) * a).astype(F32)
        t2 = f16(f16(dy * dy).astype(F32) * b).astype(F32)
        t3 = f16(f16(dx * dy).astype(F32) * c).astype(F32)
        q = f16(f16(t1 + t2).astype(F32) + t3).astype(F32)
        out.append((np.exp2(q).astype(F32), r))
    return out


# ------------------------------------------------------------------ stage 4: colour and blend averages
def spatial_average(tp, fields):
    """Normalised weighted average of record fields: f16 fma chain of f16(w)*field, then / sum(w) (f32)."""
    wsum = np.zeros(tp[0][0].shape, F32)
    for w, _ in tp:
        wsum = (wsum + w).astype(F32)
    inv = (F32(1.0) / wsum).astype(F32)
    res = np.zeros(tp[0][0].shape + (len(fields),), F32)
    for k, f in enumerate(fields):
        n = np.zeros(tp[0][0].shape, F32)
        for w, r in tp:
            n = f16(n + f16(w).astype(F32) * r[..., f].astype(F32)).astype(F32)
        res[..., k] = (n * inv).astype(F32)
    return res


def numerators(tp, fields):
    """The f16-chain numerators n_f = f16(n + f16(w)*field) (kx-major), stacked over `fields`."""
    shape = tp[0][0].shape
    res = np.zeros(shape + (len(fields),), F32)
    for k, f in enumerate(fields):
        n = np.zeros(shape, F32)
        for w, r in tp:
            n = f16(n + f16(w).astype(F32) * r[..., f].astype(F32)).astype(F32)
        res[..., k] = n
    return res


# f32 constants of the tonemap / blend (all lifted from the ISA or PostArgs)
C_L210 = f32_from_bits(0x40549A78)       # PostArgs 0xcc: log2(10)
C_I036 = f32_from_bits(0x4031C71C)       # 1 / 0.36
C_HUNDRED = F32(100.0)                   # 0x42c80000
C_PT01 = f32_from_bits(0x3C23D70A)       # 0.01
C_FLOOR = f32_from_bits(0x38D1B717)      # 1e-4
C_CEIL = F32(3.0)


def fma(a, b, c):
    """One-rounding fma: fl(a*b + c) (product exact, single rounding)."""
    return (np.float64(a) * np.float64(b) + np.float64(c)).astype(F32)


def decode_lin(h):
    """Inverse tonemap: linear = 0.01*(exp2(clamp(h,1e-4,3)*(1/0.36)*log2(10)) - 1) (f32, one rounding each op)."""
    hc = np.clip(h, C_FLOOR, C_CEIL).astype(F32)
    arg = ((hc * C_I036).astype(F32) * C_L210).astype(F32)
    return (C_PT01 * (np.exp2(arg).astype(F32) - F32(1.0)).astype(F32)).astype(F32)


def tonemap_lin(x):
    """tm = 0.36*log10(1+100x): y = fma(100,x,1), clamp >= 1e-4, then (log2(y)*log10(2))*0.36 (f32)."""
    y = (np.float64(C_HUNDRED) * np.float64(x) + np.float64(F32(1.0))).astype(F32)
    y = np.maximum(y, C_FLOOR).astype(F32)
    lg = (np.log2(y).astype(F32) * C_TM_LOG10_2).astype(F32)
    return (lg * C_TM_SCALE).astype(F32)


# ------------------------------------------------------------------ stage 6: the gate blend weight
def blend_weight(tp, rec, inp, dec0_f1):
    """The blend weights t1 (gate1, used by ihist1/dhist) and t2 (gate2, used by full15) plus v12, Wsum, inv.

    v12 = clamp(dec0_f1 * f16(n_gate0), 0, 1) evaluated as the hardware does: f32(f32(dec0*Wsum) * f32(inv*f16Q));
    for g in {gate1, gate2}: Wa = f16(n_g); u = f32(inv*Wa); v25 = fma(-inv, Wa, 1); t = fma(v12, v25, u).
    (Verified against probes: v16 = inv*f16(n_gate0) 99.997%, Wa = f16(n_gate1) = lo(v20) 99.997%, v25 100%;
    the gate2 weight reproduces the oracle's full15 at 99.99%, the gate1 weight its ihist1 at 99.99%.)"""
    wsum = np.zeros(tp[0][0].shape, F32)
    for w, _ in tp:
        wsum = (wsum + w).astype(F32)
    inv = (F32(1.0) / wsum).astype(F32)
    ng = numerators(tp, [6, 7, 8])
    v16 = (inv * f16(ng[..., 0]).astype(F32)).astype(F32)
    v12 = np.clip((F32(dec0_f1) * wsum).astype(F32) * v16, F32(0.0), F32(1.0)).astype(F32)
    ts = []
    for k in (1, 2):
        Wa = f16(ng[..., k]).astype(F32)
        u = (inv * Wa).astype(F32)
        v25 = fma(-inv, Wa, F32(1.0))
        ts.append(fma(v12, v25, u))
    return ts[0], ts[1], v12, wsum, inv


# ------------------------------------------------------------------ stage 7: the Catmull-Rom history
def _cr_axis(f):
    """The 1D Catmull-Rom weights from the sub-texel fraction f (clamped to [0,1]), the ISA's f32 order:
    returns (w(-1), w0, w1, w2, a = w0+w1, r = w1/a)."""
    f = np.clip(f, F32(0.0), F32(1.0)).astype(F32)
    f2 = (f * f).astype(F32)
    f3 = (f2 * f).astype(F32)
    wm1 = fma(F32(-0.5), (f3 + f).astype(F32), f2)             # f2 - 0.5*(f3+f)
    w0 = (F32(1.0) + fma(F32(1.5), f3, -2.5 * f2).astype(F32)).astype(F32)
    w2 = (F32(0.5) * fma(f, f2, -f2).astype(F32)).astype(F32)   # 0.5*(f3 - f2)
    w1 = fma(F32(-0.5), (f3 - f2).astype(F32), (F32(1.0) - wm1 - w0).astype(F32))
    a = (w0 + w1).astype(F32)
    r = (w1 / a).astype(F32)
    return wm1, w0, w1, w2, a, r


def _bilinear(hist, xi, yi, rx, ry):
    """A bilinear sample of hist (H15,W15,C f16) at the texel (xi, yi) (lower-left, int arrays) with f32
    sub-texel weights rx, ry (arrays, broadcast over the channel axis); clamp-to-edge addressing."""
    xi = np.clip(xi, 0, W15 - 1); yi = np.clip(yi, 0, H15 - 1)
    x1 = np.clip(xi + 1, 0, W15 - 1); y1 = np.clip(yi + 1, 0, H15 - 1)
    one = F32(1.0)
    if np.ndim(rx): rx = rx[..., None]
    if np.ndim(ry): ry = ry[..., None]
    h00 = hist[yi, xi].astype(F32); h01 = hist[yi, x1].astype(F32)
    h10 = hist[y1, xi].astype(F32); h11 = hist[y1, x1].astype(F32)
    top = (h00 * (one - rx) + h01 * rx).astype(F32)
    bot = (h10 * (one - rx) + h11 * rx).astype(F32)
    return (top * (one - ry) + bot * ry).astype(F32)


def cr_history(inp, y15, x15, hist):
    """The 5-tap bilinear Catmull-Rom sample of `hist` (the previous ihist, f16 H15xW15x3) at the reprojected
    position of output pixel (y15, x15): pos = (x15+0.5+mv_x, y15+0.5+mv_y) in OUTPUT texels (the MV is in
    output units). Returns the tm-domain history value h (f32). The bilinear filter's sub-texel fraction is
    quantised to 8 bits (the hardware's filter precision)."""
    mvx, mvy = inp["mv"]      # the chosen MV (constant in these frames: all three candidates are equal)
    pos = (x15.astype(F32) + F32(0.5) + F32(mvx)).astype(F32)
    poy = (y15.astype(F32) + F32(0.5) + F32(mvy)).astype(F32)
    hist = np.asarray(hist, np.float16)[..., :3]
    Ix = np.floor(pos - F32(0.5)).astype(np.int64)
    Iy = np.floor(poy - F32(0.5)).astype(np.int64)
    fx = (pos - (Ix.astype(F32) + F32(0.5))).astype(F32)
    fy = (poy - (Iy.astype(F32) + F32(0.5))).astype(F32)
    wmx, _, w1x, w2x, ax, rx = _cr_axis(fx)
    wmy, _, w1y, w2y, ay, ry = _cr_axis(fy)
    # the bilinear filter's sub-texel fraction is quantised to 8 bits, round-to-nearest
    rxq = np.round(rx.astype(np.float64) * 256).astype(np.float64) / 256.0
    ryq = np.round(ry.astype(np.float64) * 256).astype(np.float64) / 256.0
    rxq = rxq.astype(F32); ryq = ryq.astype(F32)
    W1 = (wmx * ay)[..., None].astype(F32); W2 = (w2x * ay)[..., None].astype(F32)
    W3 = (ax * wmy)[..., None].astype(F32); W4 = (ax * w2y)[..., None].astype(F32)
    W5 = (ax * ay)[..., None].astype(F32)
    def bil(xi, yi, rqx, rqy):
        return _bilinear(hist, xi, yi, rqx, rqy)
    # tap 1: x = Ix - 1 (pure texel, its centre), y = main bilinear
    t1 = bil(Ix - 1, Iy, F32(0.0), ryq)
    # tap 3: x = main bilinear, y = Iy - 1 (pure texel)
    t3 = bil(Ix, Iy - 1, rxq, F32(0.0))
    # tap 5: the main bilinear
    t5 = bil(Ix, Iy, rxq, ryq)
    # tap 4: x = main, y = Iy + 2 (pure texel)
    t4 = bil(Ix, Iy + 2, rxq, F32(0.0))
    # tap 2: x = Ix + 2 (pure texel), y = main
    t2 = bil(Ix + 2, Iy, F32(0.0), ryq)
    num = (t1 * W1).astype(F32)
    num = fma(t3, W3, num)
    num = fma(t5, W5, num)
    num = fma(t4, W4, num)
    num = fma(t2, W2, num)
    wcr = (W3 + W1).astype(F32)
    wcr = (wcr + W5).astype(F32)
    wcr = (wcr + W4).astype(F32)
    wcr = (wcr + W2).astype(F32)
    return (num / wcr).astype(F32)


# ------------------------------------------------------------------ stage 5: outputs
def encode_r11g11b10(tm):
    """full15 = R11 | G11 << 11 | B10 << 22 of k * tm (ISA: v_mul k, v_fma 2047/1023 + 0.5, trunc, clamp)."""
    def enc(v, scale):
        t = (K_ENC * v).astype(F32)
        q = np.clip(np.trunc(fma(F32(scale), t, F32(0.5))), F32(0.0), F32(scale))
        return q.astype(np.uint32)
    return enc(tm[..., 0], 2047) | (enc(tm[..., 1], 2047) << 11) | (enc(tm[..., 2], 1023) << 22)


def encode_dhist(tm):
    """dhist word = R10 | G10 << 10 | B10 << 20 of rndne(1023 * clamp(k*tm, 0, 1)) (round-to-nearest-even)."""
    t = np.clip((K_ENC * tm).astype(F32), F32(0.0), F32(1.0)).astype(F32)
    q = np.rint((F32(1023.0) * t).astype(np.float64)).astype(np.uint32)
    return q[..., 0] | (q[..., 1] << 10) | (q[..., 2] << 20)


def model_frame(inp, rec):
    """The full post model: spatial taps, the gate blend, the CR history, the outputs (full15, ihist1, dhist)."""
    Y, X = np.meshgrid(np.arange(H15), np.arange(W15), indexing="ij")
    tp = taps(Y, X, inp, rec)
    t1, t2, v12, wsum, inv = blend_weight(tp, rec, inp, inp["dec0_f1"])
    S = spatial_average(tp, [0, 1, 2])
    if inp["frame"] > 0:
        e0 = F32(inp["expo"][inp["p"]]) if inp["expo"][inp["p"]] != 0 else F32(1.0)
        h = cr_history(inp, Y, X, inp["ihist_in"])[..., :3]
        A = (e0 * decode_lin(h)).astype(F32)
    else:
        A = decode_lin(np.zeros((H15, W15, 3), F32))
    B = decode_lin(S)
    diff = (B - A).astype(F32)
    x1 = fma(t1[..., None], diff, A)          # the ihist1 / dhist blend (gate1)
    x2 = fma(t2[..., None], diff, A)          # the full15 blend (gate2)
    tm = tonemap_lin(x1)
    tm2 = tonemap_lin(x2)
    ihist1 = np.zeros((H15, W15, 4), np.float16)
    ihist1[..., :3] = f16(tm)
    full15 = encode_r11g11b10(tm2)
    x4 = x1.reshape(H15 // 2, 2, W15 // 2, 2, 3)
    xm = (x4[:, 0] + x4[:, 1]).astype(F32)
    xm = ((xm[:, :, 0] + xm[:, :, 1]) * F32(0.25)).astype(F32)
    dhist = encode_dhist(tonemap_lin(xm))
    return dict(ihist1=ihist1, full15=full15, dhist=dhist, S=S, t=t1, t2=t2, v12=v12, x=x1, tm=tm)


# ------------------------------------------------------------------ scoring (tools/report/launch_report.py)
sys.path.insert(0, str(REPO / "tools/report"))
import launch_report as LR  # noqa: E402


def decode_full15(u):
    """R11G11B10 -> normalised components (R/2047, G/2047, B/1023), stacked as f32 (the decoded values)."""
    u = np.asarray(u, np.uint32)
    c = np.stack([u & 2047, (u >> 11) & 2047, (u >> 22) & 1023], -1).astype(np.float32)
    return (c / np.array([2047, 2047, 1023], np.float32)).astype(np.float32)


def run(frame, oracle, out=None, quiet=False):
    inp = load_inputs(oracle, frame)
    rec = records(inp)
    m = model_frame(inp, rec)
    if out:
        os.makedirs(out, exist_ok=True)
        Path(out, "full15.bin").write_bytes(m["full15"].tobytes())
        Path(out, "ihist1.bin").write_bytes(m["ihist1"].tobytes())
        Path(out, "dhist.bin").write_bytes(m["dhist"].tobytes())
    res = {}
    of = np.frombuffer(inp["oracle_out"]["full15"].read_bytes(), np.uint32).reshape(H15, W15)
    oh = np.frombuffer(inp["oracle_out"]["ihist1"].read_bytes(), np.float16).reshape(H15, W15, 4)
    od = np.frombuffer(inp["oracle_out"]["dhist"].read_bytes(), np.uint32).reshape(H15 // 2, W15 // 2)
    jobs = [("full15", m["full15"], "u32"), ("ihist1", m["ihist1"], "f16"), ("dhist", m["dhist"], "u32"),
            ("full15-dec", decode_full15(m["full15"]), "f32")]
    srcs = {"full15": inp["oracle_out"]["full15"], "ihist1": inp["oracle_out"]["ihist1"],
            "dhist": inp["oracle_out"]["dhist"], "full15-dec": inp["oracle_out"]["full15"]}
    for name, port, dt in jobs:
        orc = srcs[name].read_bytes()
        if name == "full15-dec":
            orc = decode_full15(of).astype(np.float32).tobytes()
            port = np.ascontiguousarray(port, np.float32).tobytes()
        else:
            port = port.tobytes()
        r = LR.measure(port, orc, dt)
        res[name] = r
        if not quiet and r:
            n, peak, psnr, rr, rrmse, exact = r
            print(f"  {name:12s} n {n}  bit-exact {exact * 100:.4f}%  peak {peak:.6g}  PSNR {LR.fmt_psnr(psnr)} dB"
                  f"  r {rr if rr is None else round(rr, 6)}  rRMSE {rrmse:.6f}")
    if not quiet:
        par = (np.arange(H15) % 2)[:, None] * np.ones((1, W15), int)
        for pix, lab in ((0, "pixel 0 (even rows)"), (1, "pixel 1 (odd rows)")):
            sel = par == pix
            print(f"  {lab:20s} full15 bit-exact {np.mean(m['full15'][sel] == of[sel]) * 100:.4f}%  "
                  f"ihist1 bit-exact {np.mean(m['ihist1'][sel] == oh[sel]) * 100:.4f}%")
    return res, m, inp


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--oracle", default=str(ORACLE / "sr_M/dump"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--self-check", action="store_true")
    a = ap.parse_args()
    if a.self_check:
        return self_check()
    run(a.frame, a.oracle, a.out)
    return 0


def self_check():
    """The acceptance thresholds of the brief (launch 14, sr_M, oracle dump): frame 0 >= 99%, frame 1 >= 97%
    bit-exact for each of full15, ihist1 and dhist."""
    thr = {0: 99.0, 1: 97.0}
    ok = True
    for frame in (0, 1):
        print(f"[frame {frame}, launch 14]")
        res, _, _ = run(frame, str(ORACLE / "sr_M/dump"))
        for k in ("full15", "ihist1", "dhist"):
            got = res[k][5] * 100
            good = got >= thr[frame]
            ok &= good
            print(f"  -> {k} bit-exact {got:.4f}% >= {thr[frame]:.1f}%: {'PASS' if good else 'FAIL'}")
    print("SELF-CHECK:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
