#!/usr/bin/env python3
"""tools/sr/sr_post_model.py: the NumPy model of NeuralHub's k_sr_post<false,12,16,false,false> (SR preset M,
oracle launch 14) - the spec that sr_post.comp is rewritten against.

    build/venv/bin/python -I tools/sr/sr_post_model.py [--frame F] [--oracle DUMP_DIR] [--out DIR]
    build/venv/bin/python -I tools/sr/sr_post_model.py --self-check

Inputs are the oracle's own bytes as they stood before launch 14 (newest earlier NNN.<buffer>.bin, else
start.<buffer>.bin). Outputs: full15 (u32, 768x432), ihist1 (f16 x4, 768x432), dhist (u32, 384x216).

STATUS (2026-10-26, Review R6; numbers from --self-check, frame 0 and 1, launch 14 of sr_M).
  VERIFIED against hardware probes (build/agents/post-model/verify_stages.py, Review R6):
    stage 1  source records, 12 f16 per source pixel: 100% of 36864 pixels, all 12 fields.
    stage 2  output geometry px/py (f32 fma) and the source base X0/Y0: px 100%.
    stage 3  the nine taps and their quadratic q (f16): tap identities 9 of 9; the weight sum 95.2%,
             one weight 86.2% bit-exact (the rest are 1-ULP differences of the hardware v_exp_f32).
    stage 4  f16 numerator chain for factor 0: 99.99%; normalised factor-0 average (f32): 96.0%.
  MEASURED MISMATCH (open):
    stage 5  colour average -> ihist1, pixel-0 rows: 93.3% per channel; the 6.7% that differ are
             one f16 ULP, oracle below model, and they correlate with gate0 (mean gate0 0.51 on the
             mismatches vs 0.62 on the matches): the gate blend that follows the average is not
             yet modelled (for frame 0 it reduces to the identity up to these ULPs).
    stage 6  pixel 1 (odd rows), the hi lane of the packed quadratic: matches the pixel-1 geometry
             in 85.7% of pixels where pixel 1's base row equals pixel 0's, and in 0.0% where it differs
             (about two thirds of all pixels). None of 216 row/column/dy shift combinations, nor mixed
             record sources, reproduces it (Review R6.4). Pixel 1 is the largest error source.
  NOT MODELLED: dhist (the 2x2 combine; a 10-bit encode of 2x2 tm averages matched 62.6% of words,
  not claimed), the MV choice, the history reprojection and the gate/factor blend. Frame 1 needs all
  of them, so the numbers for frame 1 are the spatial path alone and are expected to be low.

SPEC (all constants are the hardware's; f16 = IEEE half, round to nearest even; every "f16(...)" is a rounding point)

Inputs per launch: W,H = 256,144 (render size), W15,H15 = 768,432, F20 = (F20x, F20y) jitter (PostArgs 0xb4,0xb8),
  scale s = (W/W15, H/H15) = (1/3, 1/3) as f32 (0xac,0xb0), expo[p], p = frame parity (0xa0 mapping: 0x10).
  g_col: f16 x4 per source pixel (R,G,B,A); aniso: f16 x4 (A,B,C,_); gates, factors: u8 x4 (lanes 0..2 used).

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

3. Spatial taps: for (kx, ky) in {-1,0,1}^2 the source pixel is (X0+kx, Y0+ky) clamped to the image.
   Tap offsets are dx = f16(dx0 + kx), dy = f16(dy0 + ky) (exact integers in f16).
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
   96.0 percent (the rest is the v_exp_f32 ULP in the sum), and ihist1 = f16(S_c) for 93.3 percent.  The remaining 6.7 percent of ihist1 differ by one f16 ULP and correlate with gate0 (open).

5. Frame-0 outputs (history is zero on a reset frame; the spatial path is what is written here):
     ihist1_c = f16(S_c)    (alpha lane 0; 93.3% per channel on pixel-0 rows, Review R6.5)                                  -> ihist1 (f16 x 4)
     full15: R = trunc(clamp(fma(2047, f32(k*S_R), 0.5), 0, 2047)),
             G = (same with 2047) << 11,  B = (trunc(clamp(fma(1023, f32(k*S_B), 0.5), 0, 1023))) << 22,
             k = f32(0x3ed0fac6) = 1/2.45 (f32), full15 = R | G | B   (R11G11B10)
     The k*S encoding is measured on the oracle's own ihist1 (R: 93.5 percent of pixel-0 rows) (Review R6.6).
6. dhist (384x216): OPEN.  A 2x2 box of the tonemapped values encoded with 10 bits (1023 scale, k as above)
   matches 62.6 percent of the oracle's words when the box is the 4 output pixels (it depends on the pixel-1
   stage that is open).  Not claimed as a spec.

7. History, MV and blend (frame 1 onward): OPEN.  The ISA order (Review R5, docs/notes/k_sr_post.md):
   MV candidates (mv_nondil texture sample, prev_mvec1, mv_ero) scored by the spatial pass; on-screen test
   against W15/H15; Catmull-Rom history over ihist (the 5 bilinear taps); second candidate when its share
   exceeds 1/255; gate/factor blend with dec0_f1 (PostArgs 0xa8 = 2.7530112); inverse tonemap
   clamp [1e-4, 3] -> 10^(x/0.36) -> (y-1)*0.01 with the exposure ratio expo[p^1]/expo[p].
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
    return dict(
        fdir=fdir, p=pidx, expo=expo, s=(s_x, s_y), jit=(j_x, j_y), dec0_f1=dec0_f1, ldr=ldr, mvoff=mvoff,
        g_col=np.frombuffer(buffer_before(fdir, ptr[0], launch), np.float16).reshape(H, W, 4),
        gates=np.frombuffer(buffer_before(fdir, ptr[24], launch), np.uint8).reshape(H, W, 4),
        factors=np.frombuffer(buffer_before(fdir, ptr[32], launch), np.uint8).reshape(H, W, 4),
        aniso=np.frombuffer(buffer_before(fdir, ptr[40], launch), np.float16).reshape(H, W, 4),
        ihist_in=np.frombuffer(buffer_before(fdir, ptr[48], launch), np.float16).reshape(H15, W15, 4),
        prev_mv=np.frombuffer(buffer_before(fdir, ptr[72], launch), np.float16).reshape(H, W, 2),
        mv_ero=np.frombuffer(buffer_before(fdir, ptr[80], launch), np.float16).reshape(H, W, 2),
        mv_nondil=np.frombuffer(buffer_before(fdir, ptr[88], launch), np.float16).reshape(H, W, 2),
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
    """The nine (kx-major) taps: list of (w f32, record rows (N,H,W,12) gathered per pixel)."""
    X0, Y0, dx0, dy0 = geometry(y15, x15, inp)
    out = []
    for kx, ky in TAPS_KX_MAJOR:
        dx = f16(dx0.astype(F32) + F32(kx)).astype(F32)
        dy = f16(dy0.astype(F32) + F32(ky)).astype(F32)
        sxi = np.clip(X0 + kx, 0, W - 1)
        syi = np.clip(Y0 + ky, 0, H - 1)
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


# ------------------------------------------------------------------ stage 5: outputs (frame-0 spatial path)
def encode_r11g11b10(tm):
    """full15 = R11 | G11 << 11 | B10 << 22 of k * tm (ISA: v_mul k, v_fma 2047/1023 + 0.5, trunc, clamp)."""
    def enc(v, scale):
        t = (K_ENC * v).astype(F32)
        q = np.clip(np.trunc((F32(scale) * t + F32(0.5)).astype(F32)), 0, scale)
        return q.astype(np.uint32)
    return enc(tm[..., 0], 2047) | (enc(tm[..., 1], 2047) << 11) | (enc(tm[..., 2], 1023) << 22)


def model_frame(inp, rec):
    Y, X = np.meshgrid(np.arange(H15), np.arange(W15), indexing="ij")
    tp = taps(Y, X, inp, rec)
    S = spatial_average(tp, [0, 1, 2])
    ihist1 = np.zeros((H15, W15, 4), np.float16)
    ihist1[..., :3] = f16(S)
    full15 = encode_r11g11b10(S)
    return dict(ihist1=ihist1, full15=full15, S=S)


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
    res = {}
    of = np.frombuffer(inp["oracle_out"]["full15"].read_bytes(), np.uint32).reshape(H15, W15)
    jobs = [("full15", m["full15"], "u32"), ("ihist1", m["ihist1"], "f16"),
            ("full15-dec", decode_full15(m["full15"]), "f32"), ("full15-dec-ref", None, None)]
    for name, port, dt in jobs[:3]:
        orc = (inp["oracle_out"]["full15"].read_bytes() if name.startswith("full15") else
               inp["oracle_out"]["ihist1"].read_bytes())
        if name == "full15-dec":
            orc = decode_full15(of).astype(np.float32).tobytes()
            port = np.ascontiguousarray(port, np.float32)
            port = port.tobytes()
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
                  f"ihist1 bit-exact {np.mean(m['ihist1'][sel] == np.frombuffer(inp['oracle_out']['ihist1'].read_bytes(), np.float16).reshape(H15, W15, 4)[sel]) * 100:.4f}%")
    if not quiet:
        print("  dhist    NOT MODELLED (open: needs the pixel-1 stage and the 2x2 combine; Review R6.6)")
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
    """Reproduce the measured numbers of Review R6 (launch 14, sr_M, frames 0 and 1). Exact to 0.001%."""
    want = {0: {"full15": 85.6738, "ihist1": 84.9322}, 1: {"full15": 20.3411, "ihist1": 39.6724}}
    ok = True
    for frame in (0, 1):
        print(f"[frame {frame}, launch 14]")
        res, _, _ = run(frame, str(ORACLE / "sr_M/dump"))
        for k in ("full15", "ihist1"):
            got = res[k][5] * 100
            good = abs(got - want[frame][k]) < 0.001
            ok &= good
            print(f"  -> {k} bit-exact {got:.4f}% expected {want[frame][k]:.4f}%: {'PASS' if good else 'FAIL'}")
    print("SELF-CHECK:", "PASS" if ok else "FAIL")
    return 0 if ok else 1




if __name__ == "__main__":
    sys.exit(main())
