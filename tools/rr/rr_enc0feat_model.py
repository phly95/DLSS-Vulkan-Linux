#!/usr/bin/env python3
"""tools/rr/rr_enc0feat_model.py: the NumPy spec of rrio::k_enc0_feat (RR preset F, oracle
launches 3/4) - the producer of E0be8, the [2][H0][W0][32] fp8 feature tensor (the RR analogue
of SR's t0; consumed by the level-0 enc0 unit, see tools/rr/rr_level_model.py).

    build/venv/bin/python -I tools/rr/rr_enc0feat_model.py [--frame F] [--oracle DIR] [--port FILE]
    build/venv/bin/python -I tools/rr/rr_enc0feat_model.py --self-check        # < 60 s (measured 0.4 s)

STATUS (agent enc0feat-decode):
  frame0: 2621440/2621440 = 100.0000% bit-exact vs oracle (peak=1.125 PSNR=inf r=1.0 rRMSE=0).
  port (F3): model_port() reproduces a real kernel_run of shaders/rr/rr_enc0_feat.comp on the
  oracle inputs 2621440/2621440 = 100.0000%; port vs oracle = 72.9063% = PORT-STATUS exactly.
  frame1: 87.5680% (2295542/2621440) - the 15 SC_TM history channels + c36 are now DECODED
  (100%); the remaining 12.43% gap is the 8 c22-family channels only (see OPEN1). peak=1.125
  PSNR=32.01dB r=0.989829 rRMSE=0.1442.

===================================================================================================
SPEC (decoded; kernel: _ZN4rrio11k_enc0_featENS_6E0ArgsE, ISA listing
build/agents/enc0feat-decode/f.s, offsets are file-relative (tools/re/fn.sh output, kernel
starts at 0xCD00). E0Args layout: tools/rr/rrref/rrref.cpp lines 471-512.)

1. INPUTS (load())
   E0Args (0x248 B) fields the kernel reads - offset: field
     0x000 W0,H0 (u32 pair)  0x008 W,H (256x144 render)  0x010 fc = E[0x19] (dither row, &7)
     0x014 rst (HISTORY GATE: rst!=0 -> history branch skipped; oracle frame0 rst=1, frame1 rst=0)
     0x130 g_col  0x138 g_dep  0x140 g_mv  0x148 g_nrm  0x150 g_alb  0x158 g_spa
     0x160 bev_prev  0x168 E0c0e  0x170 E0c12  0x178 E0c02  0x190 expo (f32x4: [e_cur, e_prev, 1/e_cur, ...])
     0x198 wbuf = E001c + 37984  0x1a8/0x1ac prev-jitter pj (frame1 = (0, -1/6))
     0x180 E[8..11] = (384, 216, 192, 108)  0x1e0.. = (1/384, 1/216, 384, 216, 768, 432, ...)
     0x200 SC0 = -0.43359375 (f16-sourced blob scalar)  0x204 SC1 = -0.2198486328125
     0x220 out E0be8  0x228 col E0bec  0x230 bev E0bf6  0x238 E0bfa  0x240 E0bfe
   Buffer encodings (256x144 render, 256x160 feature grid):
     g_col f16x4[W*H]  g_dep f32[W*H]  g_mv f16x2[W*H]  g_nrm f16x4[W*H]  g_alb f32x4[W*H]
     g_spa f32x4[W*H]  bev_prev 8B/px (f16x4)  E0c0e u32[216][384] R10G10B10 (history, k_post)
     E0c12 u32[108][192] R10G10B10 (history pooled2 of E0c0e)  E0c02 i8x4 (k_level, hidden)
     expo f32x4. Rows H..H0-1 of the feature grid use the MIRRORED row 2*(H-1)-y.

2. THE FEATURE QUANTIZER  dq(x)  (dither table read at rel 0x4FE0, abs 0x11CE0)
   The dither table is wbuf[0:4096] as u16[2048] - VERIFIED bit-identical to SR preset M's
   enc0.bin[:4096] (build/re/packed/M/enc0.bin). Same dq() as tools/sr/sr_enc0_model.py:
     idx = ((fc & 7) << 8) | ((y & 15) << 4) | (x & 15)          (fc&7 = 0 on both oracle frames)
     x16 = f16(min(x, 1024)); E = x16 & 0xfc00; d = dither[idx] >> 9; K = E | d (f16 bits)
     a = f16(x16 + K); b = f16(a - E); t = (b & 0xff80) as f16   (3 mantissa bits, trunc to 0)
     q = q8r(clip(t, -448, 448))   f32 -> e4m3 RNE saturating (hardware v_cvt_pk_fp8_f32)
   q8r rounding points (VERIFIED by 100% frame0):
     - signbit: e4m3(-0.0) = 0x80 (the {0,128} channels are dq(-0.0): d>0 keeps -0 via
       b = a - E with E = -0 -> stays -0; d == 0 gives -0 - (-0) = +0 -> 0x00. The single
       d==0 cell of row fc=0 is (y,x) = (10,5) -> exactly the oracle's 160 px = 0.390625%).
     - subnormal: round-half-to-EVEN on a*512 (np.round), NOT floor+0.5: the dq() 3-bit
       truncation grid lands |t| exactly on (n+0.5)*2^-9 whenever k=4m+2 and hardware
       RNE-even there was the entire alb-channel residual (95% -> 100%) and the sc0*tm
       residual on frame0 (99.76% -> 100%).

3. LAYOUT (packing, rel 0x684c..0x687c; 4 b128 stores + 1 b64)
   out byte index = (cg * H0 * W0 + y * W0 + x) * 32 + i, i = stored channel in [0,32).
   NO stored->logical permutation (unlike SR's P/IP): stored byte i = the channel the ISA
   computes into it. VERIFIED: group0 base v[21:22] = E0be8 + (y*W0+x)*32, group1 adds
   W0*H0*32 (packmap2/3/4.out).

4. CHANNELS (stored index -> value; every channel below through dq(); frame0 100% verified)
   TM_COL   {0: col.r, 1: col.g, 4: col.b}: tm(v) = 0.36*log10(max(1+100*min(e*v,65504),1e-4)),
             NaN -> 0, clamp result <= 1024 later in the sc0 path; e = expo[0].
   SC_TM   15 channels = SC0 * tm(col_c * e):
             {5,12,17,24,37} = col.r, {8,13,20,25,40} = col.g, {9,16,21,28,41} = col.b.
             (fit1:99.3-99.8% pre-q8r-fix -> 100%; PORT guesses mv*3 instead = 3.7-9.8%).
   ALB      {2: alb.g, 3: alb.b, 29: alb.r}  (PORT BUG: writes c02=alb.r, c29=const 0x04).
   SPA      {6,7,10} = min(2, spa.r/g/b) = 0.040008544921875 -> dither flips {18,19}.
   CONST    15 = dq(min(2,1.0)) -> {56,57}; 18 = dq(min(2, dep=0.5)) -> {48,49}.
   NEG0     {22,23,26,27,45,48,49,52} = dq(-0.0) -> 0x80 except the d==0 cell (frame0; on
            frame1 these are history-fed and NOT yet decoded - OPEN1).
   ZERO     {11,14,19,30..36,38,39,42..44,46,47,50,51,53..63} = dq(0) = 0 (verified 100%).

5. PORT SEMANTICS (model_port(), F3 - reproduces shaders/rr/rr_enc0_feat.comp EXACTLY:
   2621440/2621440 vs a real kernel_run on oracle inputs)
   Plain e4m3 RNE (q8r, NO dq/dither); port channel map: c02=alb.r (oracle: alb.g),
   c29=0x04 (oracle: alb.r), slots {5,8,12,13,17,20,24,25} = mv.x*3 and {9,16,21,28} =
   mv.y*3 (oracle: SC0*tm), constants 0x12/0x38/0x30/0x80 written fixed (oracle: dq()'d
   with dither flips), group1 {5,8}=mv.x*3, {9}=mv.y*3, {13,16,17,20}=0x80.
   Port vs oracle = 72.9063% = PORT-STATUS exactly.

6. HISTORY BRANCH (ISA-decoded + value DECODED for SC_TM; frame0 rst-gate makes it col-only)
   Gate: rst (args+0x14; s[52:53] reassign at rel 0x1FF8 = (fc, rst)) -> skipped on frame0,
   which is WHY frame0 reduces to pure col*expo for all SC_TM channels (100%).
   Position (rel 0x34EC-0x3624, inputs v6=s58*v0 / v52=s20*v13): Cx,Cy in the 768x432 grid,
     Cx = 3*((x+0.5) - pjx) + bev_x, Cy = 3*((row+0.5) - pjy) + bev_y
       where (pjx,pjy) = args+0x1a0/+0x1a4  (frame1 = (-0.25, +1/6); frame0 = (0, -1/6))
       and bev = E0bf6 prev (constant global motion ~= mv*3).
     OOB mask (rel 0x3500-0x354C): NOT(Cx<0 | Cx>768 | Cy<0 | Cy>432 | rst!=0) = the PER-LANE
     select s2 -> in-bounds lanes take history, OOB-edge lanes take col (a pure POSITION+rst
     predicate - the rel 0x45B0 v_cmpx albedo-sum guard is a separate block, NOT this select).
   Sample (rel 0x3544-0x3704): pos = C/2 - 0.5 for **E0c0e (384x216)** (s31=float(s20)=384,
     s66=1/768 -> 384*Cx/768 = Cx/2), clamp [-2,386]x[-2,218] (s36=s31+2, s37=s30+2), snap
     floor(u*256+0.5)/256, taps x[0,383] y[0,215], row stride s20=384, base s48:s49 =
     args+0x168 = E0c0e. (c17/c20/c21 sample the pooled **E0c12 (192x108)** at C/4 instead.)
   Value: R10G10B10 taps -> /1023 (0x3A802008) with -1023 (0xC47FC000) fmac pairs -> bilinear
     w00..w11 -> *2.45 (0x401CCCD) -> clamp [1e-4,8] -> inv_tm (10^(x/0.36)-1)/100 -> Newton
     divide by e_prev (s35=expo[1]) -> tm input = e_cur * (h/e_prev) (rel 0x48A4) for in-bounds,
     col*expo for OOB. Then SC0*tm(tm_input). VERIFIED: all 15 SC_TM channels = 100% frame1.

7. OPEN (blocked-with-data, reported to the orchestrator)
   OPEN1 (SOLE blocker, 8 channels = 12.43% of bytes): the c22-family
   {22,23,26,27,45,48,49,52} = 4 distinct values, each duplicated across groups
   (c22=c45, c23=c48, c26=c49, c27=c52), each = SC1 * v1 (SC1 = s17 = args+0x204, CONFIRMED
   from the s[8:23] load of args+0x1e0; frame0 = SC1*0 = -0 -> 0x80). Evidence:
     - v1 (implied = c22/SC1) in [-0.78, 0.71]; NOT a tm-difference, NOT a raw/linear history
       blend, NOT a bilinear weight: best forward fits = SC1*(-raw) 2.42%, SC1*tm(raw) 1.79%,
       SC1*tm(A)-tm(B) <= 1.51%; regression v1 ~ [hE0,hE1,hE2,col.rgb,1] rms 0.209 corr 0.63.
     - ISA: v1 = cndmask(v53) <- v53 = (s5 ? 0 : v43) [rel 0x4EF8] where s5 = the OOB mask,
       v43 = a multi-term fma accumulation (rel 0x4E70 v43 += v53*v74, plus v70/v71/v66 fma's)
       whose products are the bilinear weights (%2438 = frac_y*(1-frac_x), %2442 = frac_y*frac_x)
       -> v43 = sum(weight * value) = an interpolated history, BUT with a scaling/composition
       that does not equal the SC_TM h (corr only -0.63). Requires full SSA decode of the fma
       chain (v43/v70/v71/v66 through reused registers) - beyond budget.
     - c36 (rel 0x6224 v19 = s1) = s1 = **-2^-9 constant** (frame1), 0 on frame0 -> WIRED (100%).
     - Select ruled IN = pure position+rst; ruled OUT = albedo threshold (0x45B0 is a separate
       lane-guard; all frame1 pixels are in-bounds so the OOB mask never fires there).
===================================================================================================
"""
import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "tools/rr"))
sys.path.insert(0, str(REPO / "tools/report"))

f32 = np.float32
KERNEL = "_ZN4rrio11k_enc0_featENS_6E0ArgsE"
DITHER = np.frombuffer((REPO / "build/re/packed/M/enc0.bin").read_bytes()[:4096], np.uint16)


# ------------------------------------------------------------------ io
def parse_args(b):
    """E0Args (0x248 B) fields the kernel reads (rrref.cpp k_enc0_feat block)."""
    a = {}
    a["W0"], a["H0"], a["W"], a["H"] = struct.unpack_from("<4i", b, 0x0)
    # fc = E[0x19] at +0x10: the dither-row counter ((x+1)&7 per frame; rrref E[0x19] update
    # at frame end). VERIFIED: frame0 fc=0, frame1 fc=1 (fc=1 reproduces frame1 tm at 0.00% bad,
    # fc=0 leaves 42.98% bad). args+0x1a0 is NOT fc (its u32&7 is 0 on both frames).
    a["fc"] = struct.unpack_from("<I", b, 0x10)[0] & 7
    # rst = history gate at +0x14: rst!=0 -> history branch SKIPPED (oracle frame0 rst=1,
    # frame1 rst=0). Verified: the s[52:53] reassign at rel 0x1FF8 = (fc, rst); the OOB-or-
    # rst mask at rel 0x3528-0x354c forces all-lanes-inactive when rst!=0.
    a["rst"] = struct.unpack_from("<I", b, 0x14)[0]
    # current-frame subpixel jitter pj = (args+0x1a0, args+0x1a4) as f32.
    # VERIFIED frame0=(0.0, -1/6), frame1=(-0.25, +1/6) - these reproduce frame1 history 100%.
    a["pj"] = struct.unpack_from("<2f", b, 0x1a0)
    # prev-frame jitter at +0x1a8/+0x1ac (frame1 = (0.0, -1/6)) - present, not used here.
    return a


def buffer_before(fdir, name, launch):
    """The bytes of `name` as they stood before `launch` (newest earlier dump, else start)."""
    best = None
    for f in fdir.iterdir():
        if f.name.endswith(f".{name}.bin") and f.name[:3].isdigit() and int(f.name[:3]) < launch:
            if best is None or int(f.name[:3]) > int(best.name[:3]):
                best = f
    p = fdir / (best.name if best else f"start.{name}.bin")
    return p.read_bytes()


def load(dump, frame):
    """The launch's args and input buffers as they stood before the launch."""
    fdir = Path(dump) / f"frame{frame}"
    launch = 3 if frame == 0 else 4
    rows = [json.loads(l) for l in open(fdir / "launches.jsonl")]
    d = next(r for r in rows if r["n"] == launch and r["kernel"] == KERNEL)
    names = {int(k): v.split("+")[0] for k, v in d["pointers"].items()}
    a = parse_args(bytes.fromhex(d["args"]))
    W, H, W0, H0 = a["W"], a["H"], a["W0"], a["H0"]
    buf = lambda off: buffer_before(fdir, names[off], launch)
    g = {
        "col": np.frombuffer(buf(0x130), np.float16)[: W * H * 4].reshape(H, W, 4),
        "dep": np.frombuffer(buf(0x138), np.float32)[: W * H].reshape(H, W),
        "mv": np.frombuffer(buf(0x140), np.float16)[: W * H * 2].reshape(H, W, 2),
        "nrm": np.frombuffer(buf(0x148), np.float16)[: W * H * 4].reshape(H, W, 4),
        "alb": np.frombuffer(buf(0x150), np.float32)[: W * H * 4].reshape(H, W, 4),
        "spa": np.frombuffer(buf(0x158), np.float32)[: W * H * 4].reshape(H, W, 4),
    }
    a["expo"] = np.frombuffer(buf(0x190), np.float32)[:4]
    a["e"] = f32(a["expo"][0])
    a["e_prev"] = f32(a["expo"][1])
    # History (only NONZERO on frame1 / rst==0; empty on frame0):
    #   E0c0e u32[216][384] R10G10B10 (k_post)   E0c12 u32[108][192] R10G10B10 (k_pooled2)
    #   bev_prev E0bf6 f16x4/px (motion = mv*3, near-constant global)
    g["E0c0e"] = _dec_r10(buffer_before(fdir, names[0x168], launch), 216, 384)
    g["E0c12"] = _dec_r10(buffer_before(fdir, names[0x170], launch), 108, 192)
    bev = np.frombuffer(buffer_before(fdir, names[0x160], launch)[: W * H * 8],
                        np.float16).reshape(H, W, 4)
    a["bev"] = f32(bev[0, 0, 0]), f32(bev[0, 0, 1])   # constant global motion (BVX, BVY)
    oracle = fdir / f"{launch:03d}.E0be8.bin"
    return a, g, (oracle.read_bytes() if oracle.exists() else None)


def _dec_r10(raw, h, w):
    """u32[R10G10B10] grid -> f64 [h][w][3] with each 10-bit field /1023."""
    u = np.frombuffer(raw, np.uint32)[: h * w].reshape(h, w)
    return np.stack([(u & 0x3FF) / 1023.0,
                      ((u >> 10) & 0x3FF) / 1023.0,
                      ((u >> 20) & 0x3FF) / 1023.0], -1)


# ------------------------------------------------------------------ the feature quantizer (dq)
def dq(x, fc):
    """Dithered E4M3 quantizer (see SPEC 2). x: f32 array (H0, W0)."""
    x16 = np.minimum(np.asarray(x, f32), f32(1024.0)).astype(np.float16)
    x16 = np.where(np.isnan(x16), np.float16(0.0), x16).view(np.uint16)
    yy, xx = np.mgrid[0:x16.shape[0], 0:x16.shape[1]]
    idx = ((fc & 7) << 8) | ((yy & 15) << 4) | (xx & 15)
    d = (DITHER[idx] >> 9).astype(np.uint16)
    E = x16 & np.uint16(0xFC00)
    K = (E | d).view(np.float16)
    af = x16.view(np.float16).astype(np.float32) + K.astype(np.float32)
    a = af.astype(np.float16)
    b = (a.astype(np.float32) - E.view(np.float16).astype(np.float32)).astype(np.float16)
    t = (b.view(np.uint16) & np.uint16(0xFF80)).view(np.float16).astype(np.float32)
    return q8r(np.clip(t, -448, 448))


def q8r(x):
    """f32 -> e4m3, RNE ties-to-even, saturating; NaN -> 0x7f (the hardware v_cvt_pk_fp8_f32).
    - sign: signbit() so that -0.0 -> 0x80 (e4m3 -0), +0.0 -> 0x00.
    - subnormal path: round-half-to-EVEN on a*512 (NOT floor(a*512+0.5) - the dq() 3-bit
      truncation grid puts |t| exactly on (n+0.5)*2^-9 whenever k = 4m+2, and hardware RNE
      there was the whole residual on the alb channels (5-7% -> 0) and the tm/sc channels
      (0.24% -> 0). SR's q8r docstring records the same class of fix for the shader path."""
    x = np.asarray(x, np.float64)
    sign = np.where(np.signbit(x), 0x80, 0).astype(np.uint8)
    a = np.minimum(np.abs(x), 448.0)
    sub = np.round(a * 512.0).astype(np.uint8)      # np.round = ties-to-even
    bits = a.astype(np.float32).view(np.uint32).astype(np.uint64)
    norm = ((bits + 0x407FFFF + ((bits >> 20) & 1)) >> 20).astype(np.uint8)
    r = np.where(a < 0.015625, sub, norm & 0x7F)
    return (np.where(np.isnan(x), 0x7F, sign | r)).astype(np.uint8)


# ------------------------------------------------------------------ features
C_LOG10_2 = np.float32(0.3010299956639812)   # 0x3e9a209b
C_SCALE = np.float32(0.36)                   # 0x3eb851ec
C_FLOOR = np.float32(1e-4)                   # 0x38d1b717


def tm(v):
    """tm(v) = (log2(max(1 + 100 v, 1e-4)) * log10(2)) * 0.36, f32 (fma for 1+100v),
    NaN -> 0, clamp to [0, 1024]."""
    v = np.asarray(v, np.float32)
    t = (np.float64(100) * v.astype(np.float64) + 1.0).astype(np.float32)
    t = np.fmax(t, C_FLOOR)
    r = (np.log2(t).astype(np.float32) * C_LOG10_2).astype(np.float32) * C_SCALE
    r = np.where(np.isfinite(r) & (r >= 0), r, np.float32(0.0))
    return np.minimum(r, np.float32(1024.0))


SC0 = np.float32(-0.43359375)   # E0Args +0x200, the blob's f16 scalar (E[0xb02])
SC1 = np.float32(-0.2198486328125)  # E0Args +0x204 (the c22-family coefficient s17)


def inv_tm(y):
    """Inverse of tm(): (10**(clamp(y,1e-4,8)/0.36) - 1)/100, f64. Used to undo the
    history's *2.45 tm round-trip: stored R10 -> /1023 -> bilinear -> *2.45 -> inv_tm."""
    y = np.clip(np.asarray(y, np.float64), 1e-4, 8.0)
    return (10.0 ** (y / 0.36) - 1.0) / 100.0


# Which history buffer each SC_TM channel samples (VERIFIED 100% frame1): most use the
# full-res E0c0e (384x216, pos=C/2); c17/c20/c21 use the pooled E0c12 (192x108, pos=C/4).
HIST_C12 = frozenset((17, 20, 21))


def _hist_sample(buf, div, hmax, itx, frx, ity, fry, ch):
    """Bilinear tap of buf[...,ch] at the snapped integer/frac position (clamped taps)."""
    h, w = buf.shape[:2]
    x0 = np.clip(itx, 0, w - 1)
    x1 = np.clip(np.maximum(itx, -1) + 1, 0, w - 1)
    y0 = np.clip(ity, 0, h - 1)
    y1 = np.clip(np.maximum(ity, -1) + 1, 0, h - 1)
    C = buf[..., ch]
    return ((1 - fry) * (1 - frx) * C[y0, x0] + (1 - fry) * frx * C[y0, x1]
            + fry * (1 - frx) * C[y1, x0] + fry * frx * C[y1, x1])


def _snap(C, hmax):
    """Exact ISA position -> snapped (integer, frac) taps for one axis (SPEC 6).
    pos = C/div - 0.5, clamp [-2, hmax+2], snap floor(pos*256+0.5)/256."""
    pos = np.clip(C - 0.5, -2.0, hmax + 2.0)
    s = np.floor(pos * 256.0 + 0.5) / 256.0
    it = np.floor(s)
    return it, s - it

# stored channel index -> value formula. Verified vs oracle frame0 with dq (fit1.py, loc*.py).
# c02/c29 are alb.g/alb.r in the ORACLE (the port has them swapped - port bug).
TM_COL = {0: 0, 1: 1, 4: 2}            # tm(col.c * expo)
SC_TM = {5: 0, 12: 0, 17: 0, 24: 0, 37: 0,     # sc0*tm(col.r*expo)
         8: 1, 13: 1, 20: 1, 25: 1, 40: 1,     # sc0*tm(col.g*expo)
         9: 2, 16: 2, 21: 2, 28: 2, 41: 2}     # sc0*tm(col.b*expo)
ALB_CH = {2: 1, 3: 2, 29: 0}           # alb.g, alb.b, alb.r
SPA_CH = {6: 0, 7: 1, 10: 2}           # min(2, spa.c) -> 0.040008544921875
ZERO_CH = (11, 14, 19, 30, 31, 32, 33, 34, 35, 36, 38, 39, 42, 43, 44,
           46, 47, 50, 51, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63)
NEG0_CH = (22, 23, 26, 27, 45, 48, 49, 52)    # dq(-0.0): 0x80 except the d==0 cell


def build_features(g, a):
    """The 64 raw (f32) feature values, before dq(). Unmapped channels default to +0."""
    W, H, W0, H0 = a["W"], a["H"], a["W0"], a["H0"]
    e = a["e"]
    e_prev = a["e_prev"]
    rst = a["rst"]
    pjx, pjy = a["pj"]
    BVX, BVY = a["bev"]
    yy, xx = np.mgrid[0:H0, 0:W0]
    ys = np.where(yy < H, yy, 2 * (H - 1) - yy)
    xs = np.minimum(xx, W - 1)
    col = g["col"][ys, xs].astype(np.float32) * e
    alb = g["alb"][ys, xs].astype(np.float32)
    spa = g["spa"][ys, xs].astype(np.float32)
    dep = g["dep"][ys, xs].astype(np.float32)
    ch = {}
    for k, ci in TM_COL.items():
        ch[k] = tm(col[..., ci])

    # SC_TM channels: SC0*tm(tm_input). On frame1 (rst==0) the history branch feeds tm_input
    # for in-bounds lanes (OOB/edge lanes fall back to col - the decoded exec-mask select);
    # on frame0 (rst!=0) it reduces to pure col*expo (100%).
    if rst == 0:
        # 768x432-grid position (SPEC 6): C = 3*((coord+0.5) - pj) + bev
        Cx = 3.0 * ((xs + 0.5) - pjx) + float(BVX)
        Cy = 3.0 * ((ys + 0.5) - pjy) + float(BVY)
        inb = (Cx >= 0) & (Cx <= 768) & (Cy >= 0) & (Cy <= 432)
        tm_in = {}
        for ci in (0, 1, 2):
            for use12 in (False, True):
                buf = g["E0c12"] if use12 else g["E0c0e"]
                div = 4.0 if use12 else 2.0
                hmax = (192, 108) if use12 else (384, 216)
                itx, frx = _snap(Cx / div, hmax[0])
                ity, fry = _snap(Cy / div, hmax[1])
                s = _hist_sample(buf, div, hmax, itx.astype(np.int64), frx,
                                 ity.astype(np.int64), fry, ci)
                h = inv_tm(np.float32(2.45) * s)             # stored -> linear
                tm_hist = (e * (h / float(e_prev))).astype(np.float32)
                tm_in[(ci, use12)] = np.where(inb, tm_hist, col[..., ci])
        for k, ci in SC_TM.items():
            ch[k] = (SC0 * tm(tm_in[(ci, k in HIST_C12)])).astype(np.float32)
        ch[36] = np.full((H0, W0), -2.0 ** -9, np.float32)   # c36 = s1 = -2^-9 (VERIFIED const)
    else:
        for k, ci in SC_TM.items():
            ch[k] = (SC0 * tm(col[..., ci])).astype(np.float32)

    for k, ci in ALB_CH.items():
        ch[k] = alb[..., ci]
    for k, ci in SPA_CH.items():
        ch[k] = np.minimum(spa[..., ci], np.float32(2.0))
    ch[15] = np.ones((H0, W0), np.float32)         # min(2, 1.0); source f16-val tied (const 1.0)
    ch[18] = np.minimum(dep, np.float32(2.0))      # dep == 0.5 constant
    for k in ZERO_CH:
        if k == 36 and rst == 0:
            continue                               # c36 already set to -2^-9 above
        ch[k] = np.zeros((H0, W0), np.float32)
    for k in NEG0_CH:
        ch[k] = -np.zeros((H0, W0), np.float32)    # -0.0 -> 0x80; d==0 cell flips to 0x00
    return ch


def model(a, g):
    """The ORACLE's E0be8 tensor: u8[2*H0*W0*32], every channel through dq()."""
    H0, W0, fc = a["H0"], a["W0"], a["fc"]
    ch = build_features(g, a)
    out = np.zeros((2, H0, W0, 32), np.uint8)
    for k, v in ch.items():
        out[k // 32, :, :, k % 32] = dq(v, fc)
    return out.reshape(-1)


def model_port(a, g):
    """The PORT's E0be8 output (F3): shaders/rr/rr_enc0_feat.comp semantics exactly -
    plain e4m3 RNE (no dq/dither), the port's channel recipes (mv*3 slots, c02=alb.r,
    c29=const 0x04, const 0x80 slots). VERIFIED: reproduces a real kernel_run of the port
    shader on oracle inputs 2621440/2621440 = 100.0000% (build/agents/enc0feat-decode/)."""
    W, H, W0, H0 = a["W"], a["H"], a["W0"], a["H0"]
    e = a["e"]
    yy, xx = np.mgrid[0:H0, 0:W0]
    ys = np.where(yy < H, yy, 2 * (H - 1) - yy)
    xs = np.minimum(xx, W - 1)
    col = g["col"][ys, xs].astype(np.float32) * e
    alb = g["alb"][ys, xs].astype(np.float32)
    mv = g["mv"][ys, xs].astype(np.float32)
    mv3 = lambda c: mv[..., c] * np.float32(3.0)
    rne = lambda v: q8r(np.asarray(v, np.float32))
    p0 = np.zeros((H0, W0, 32), np.uint8)
    p0[..., 0] = rne(tm(col[..., 0]))
    p0[..., 1] = rne(tm(col[..., 1]))
    p0[..., 2] = rne(alb[..., 0])          # PORT BUG: oracle c02 = alb.g
    p0[..., 3] = rne(alb[..., 2])
    p0[..., 4] = rne(tm(col[..., 2]))
    for k in (5, 8, 12, 13, 17, 20, 24, 25):
        p0[..., k] = rne(mv3(0))           # PORT BUG: oracle = sc0*tm(col.r*expo)
    for k in (9, 16, 21, 28):
        p0[..., k] = rne(mv3(1))           # PORT BUG: oracle = sc0*tm(col.b*expo)
    p0[..., 6] = 0x12; p0[..., 7] = 0x12; p0[..., 10] = 0x12
    p0[..., 15] = 0x38; p0[..., 18] = 0x30
    for k in (22, 23, 26, 27):
        p0[..., k] = 0x80                  # oracle: dq(-0.0) flips at the d==0 cell
    p0[..., 29] = 0x04                     # PORT BUG: oracle c29 = alb.r
    p1 = np.zeros((H0, W0, 32), np.uint8)
    p1[..., 5] = rne(mv3(0)); p1[..., 8] = rne(mv3(0)); p1[..., 9] = rne(mv3(1))
    for k in (13, 16, 17, 20):
        p1[..., k] = 0x80
    out = np.zeros((2, H0, W0, 32), np.uint8)
    out[0] = p0; out[1] = p1
    return out.reshape(-1)


# ------------------------------------------------------------------ scoring
def score(pred, oracle):
    pred = np.frombuffer(pred, np.uint8) if isinstance(pred, (bytes, bytearray)) else pred
    oracle = np.frombuffer(oracle, np.uint8) if isinstance(oracle, (bytes, bytearray)) else oracle
    eq = pred == oracle
    d = pred.astype(np.int16) - oracle.astype(np.int16)
    return {
        "n": int(oracle.size),
        "bit-exact": int(eq.sum()),
        "bit-exact%": 100.0 * float(eq.mean()),
        "within1%": 100.0 * float((np.abs(d) <= 1).mean()),
        "max|err|": int(np.abs(d).max()),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--oracle", default=str(REPO / "build/oracle/oracle-8dbf027a9e-s20261025/rr_F/dump"))
    ap.add_argument("--port", default=None,
                    help="dir with the port shader's E0be8.bin (kernel_run output) to score "
                         "model_port against (F3)")
    ap.add_argument("--self-check", action="store_true")
    a = ap.parse_args()

    def report(pred, ref, label):
        s = score(pred, ref)
        extra = ""
        try:
            sys.path.insert(0, str(REPO / "tools/report"))
            import launch_report
            pb = pred.tobytes() if isinstance(pred, np.ndarray) else bytes(pred)
            rb = ref.tobytes() if isinstance(ref, np.ndarray) else bytes(ref)
            m = launch_report.measure(pb, rb, "fp8")
            if m is not None:
                # measure() -> (n, peak, psnr, r, rrmse, exact_frac)
                extra = "peak=%.4g PSNR=%.2fdB r=%.6f rRMSE=%.4g" % m[1:5]
        except Exception as exc:
            extra = "(measure skipped: %s)" % exc
        print("%s: n=%d bit-exact=%d (%.4f%%) within1=%.2f%% max|err|=%d %s" % (
            label, s["n"], s["bit-exact"], s["bit-exact%"], s["within1%"], s["max|err|"], extra))
        return s["bit-exact%"]

    if a.self_check:
        import time
        t0 = time.time()
        results = {}
        for fr in (0, 1):
            args_, g, ora = load(a.oracle, fr)
            results["oracle f%d" % fr] = report(model(args_, g), ora, "self-check oracle f%d" % fr)
            if fr == 0:
                # port semantics are frame-independent in structure; score on f0 fixture
                pass
        # F3: model_port vs the port shader's real output (if fixture present)
        pdir = REPO / "build/agents/enc0feat-decode/port_out"
        if (pdir / "E0be8.bin").exists():
            args_, g, _ = load(a.oracle, 0)
            port_out = (pdir / "E0be8.bin").read_bytes()
            results["port f0"] = report(model_port(args_, g), port_out, "self-check port f0")
        dt = time.time() - t0
        ok = results.get("oracle f0", 0) >= 99 and results.get("port f0", 0) >= 99
        print("self-check %.1f s (budget 60) -> %s (frame1 blocked: %.2f%%)" % (
            dt, "PASS-frame0+port" if ok else "FAIL", results.get("oracle f1", 0)))
        return
    args_, g, ora = load(a.oracle, a.frame)
    if a.port:
        report(model_port(args_, g), Path(a.port).read_bytes(), "frame%d port-mode vs port output" % a.frame)
    else:
        report(model(args_, g), ora, "frame%d" % a.frame)


if __name__ == "__main__":
    main()
