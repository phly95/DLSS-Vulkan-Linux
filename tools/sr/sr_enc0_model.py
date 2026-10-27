#!/usr/bin/env python3
"""tools/sr/sr_enc0_model.py: the NumPy spec of NeuralHub's k_sr_enc0 (preset M, oracle launch 2).

    build/venv/bin/python -I tools/sr/sr_enc0_model.py [--frame F] [--oracle DUMP_DIR] [--gpu DIR] [--out DIR]
    build/venv/bin/python -I tools/sr/sr_enc0_model.py --self-check        # A1, < 60 s (measured 0.2 s)

====================================================================================================
SPEC  (every formula, constant, rounding point and buffer encoding; kernel: enc0_loop.ssa in
build/work/review/enc0/ - the ISA listing in execution order: region1 0x13bc..0x39fc then region2
0x264..0x13b8, then the GEMM from 0x3a00.  Enc0Args layout: tools/sr/srref/srref.cpp, block
"enc0 (FUN_180116e70, preset M)".)
====================================================================================================

1. INPUTS (load(), parse_args())
   Enc0Args (280 B) fields the kernel reads - offset: field
     0x00 g_col  0x08 g_dep  0x10 g_mv  0x18 prev_mvec[p]  0x20 dhist  0x28 hidden[p]
     0x30 prev_mvec[p^1] (WRITE)  0x38 mv_ero (W)  0x40 mv_nondil (W)  0x48 t0 (W)  0x50 enc0_w
     0x58 W,H   0x60 W,H   0x68 W0,H0   0x70 W15,H15   0x78 hW15,hH15
     0x80 expo   0x88 p    0x8c F20x,F20y (jitter)   0x94 F28x,F28y (jitter_prev)
     0x9c fW/f15w,fH/f15h (=1/3)   0xa4 f15w/fW (=3)   0xa8 f15h/fH (=3)   0xac 1/fW   0xb0 1/fH
     0xb4 F34x/mrx, 0xb8 F34y/mry (mv_scale relative to the output size -> exactly 3.0 in f32:
         F34x = 1, mrx = fW/f15w = f32(1/3), 1/f32(1/3) rounds back to 3.0)
     0xbc mvoffx, 0xc0 mvoffy (=0 for unjittered MVs)
     0xc4 0.5/hW15, 0xc8 0.5/hH15   0xcc 9.045065879821777, 0xd0 0.22356846928596497 (preset M)
     0xd4 0x40549a78 = log2(10)   0xd8 reset   0xdc fc   0xe0 depinv   0xe4 mvpitch
     0xe8 4884.33544921875/fH (=33.918995), 0xec 1/f15w, 0xf0 1/f15h, 0xf4 F28x/fW, 0xf8 F28y/fH
     0xfc ldr  0x100 mvhi  0x104 mvhi?oW:W  0x108 mvhi?oH:H  0x10c oW/fW (=2)  0x110 oH/fH (=2)
   Buffer encodings (sizes for 256x144 render, 768x432 W15, 384x216 hW15):
     g_col  u16[4]  rgba f16 bits, W*H*4;  g_dep  f32[W*H];  g_mv  f16x2[W*H] (147456 B);
     prev_mvec*  f16x2[W*H] (= 3 * the render MV);  dhist  u32[hH15*hW15] = R10G10B10 (10 bits
     each, bits 0/10/20) written by the history kernel;  hidden*  i8[W*H*4];
     expo  u16[f16] (0 -> 1.0);
     wbuf = build/re/packed/M/enc0.bin: [0,4096) dither table (2048 u16), [4096, 4096+frag(64,128))
     the 64x128 weight tile, [12288, 12416) 64 f16 bias.
   The feature grid is (H0=160, W0=256): rows 0..143 are render rows, rows 144..159 use the
   MIRRORED row 2*H-2-y (the kernel reflects y for every read: `mirror_rows`), x is not mirrored
   (W0 == W).

2. THE FEATURE QUANTIZER  dq(x)  (0x2310.. for the dhist, 0xb48.. for feature 15, same form)
   dither = enc0.bin[0:4096] read as u16;   idx = ((fc & 7) << 8) | ((y & 15) << 4) | (x & 15)
   x16 = f16(min(x, 1024));  E = x16 & 0xfc00 (f16 bits);  d = dither[idx] >> 9;
   K = (E | d) as f16 bits;  a = f16(x16 + K);  b = f16(a - E);
   t = (b & 0xff80) as f16            <- truncation to three mantissa bits, toward zero
   q = q8r(clip(t, -448, 448))        <- f32 -> E4M3, RNE (ties to even), saturating
   q8r = sr_stage_model.to_e4m3 with a ties-to-even SUBNORMAL path: m.q8's subnormal branch is
   round-half-UP (floor(a*512+0.5)) which differs from the shader's f32_to_e4m3_rne exactly at
   |t| = (n+0.5)*2^-9 with n even - that difference was the whole residual on features 16..19
   (99.28/99.83/99.99 -> 100.0000%).  NaN in -> 0 (the kernel's cndmask).

3. FEATURES 0..2 - colour tone map  (0x1618..0x16cc, 0x1844.. ; `colour_features`)
   v = f32(e) * g_col[reflect(y), x, 0:3]      e = the expo f16 at args+0x80 (1.0 when it is 0)
   tm(v) = (log2(max(1 + 100*v, 0x38d1b717 = 1e-4)) * 0x3e9a209b) * 0x3eb851ec
       with 1+100*v as an fma, v clamped to 0x46ffe000 first (32752), NaN -> 0
       = 0.36 * log10(1 + 100 v)       (v_log_f32 measured ~exact: build/work/logprobe)
   then dq.

4. FEATURES 3..14 - FOUR BILINEAR dhist SAMPLES x {R,G,B}  (`dhist_positions`, `dhist_sample`)
   Position word per sample, per axis (the kernel's exact op order; h = hW15 for x, hH15 for y):
       Cx = 3*((x+0.5) - F20x) + 3*(mvoffx + mv_x)     [an f32 add of two f32 terms]
       a = (0.5/h) * C ;  b = h * a          <- the round trip really happens (double rounding)
       U = b - 0.5 ;  U = max(U, -2) ; NaN -> -0.5 ;  U = min(U, h+2)
       snap = floor(U*256 + 0.5) * (1/256)  ;  tap = floor(snap) ;  frac = snap - tap
   Taps: t0 = (clamp(tap_y,0,hH15-1), clamp(tap_x,0,hW15-1)); t1 = same row, col+1;
   t2 = row+1, same col; t3 = both +1 - where "col+1" is min(hW15-1, max(tap_x,-1)+1), which
   also collapses both taps to 0 when tap < 0 (clamp-to-edge at the low side).
   mv terms (each = mvoff + one of four g_mv reads; all four coincide on the oracle):
     s0 (L3,4,5)   : prev_mvec[p^1] (args 0x30) - the MIN-depth 3x3 read B (0x1c98)
     s1 (L6,7,8)   : mv_nondil      (args 0x40) - the CENTRE read D (0x1cb0)
     s2 (L9,10,11) : mv_ero         (args 0x38) - the MAX-depth 3x3 read C (0x1ca4)
     s3 (L12,13,14): mv_nondil again but with NO -F20 term: Cx = 3*(x+0.5) + 3*(mvoffx + mv_x)
       (the 0x1bac "A" read fills v52:v53 at 0x1be0, but mv_nondil overwrites those registers at
        0x1d14 / 0x1d20 before sample3's fma at 0x3394: measured 100.0000% for the centre term
        vs 32.8% for the 2x-scaled reading; the oracle's constant MV cannot separate them.)
   Value (0x2264..0x23c4 for s0; identical for the other three):
       t = f32(bits & 0x3ff) * 0x3a802008        (0x3a802008 = f32(1/1023), per tap)
       w00 = (1-frac_y)*(1-frac_x), w01 = (1-frac_y)*frac_x, w10 = frac_y*(1-frac_x), w11 = frac_y*frac_x
       v = ((w00*t00 + w01*t01) + w10*t10) + w11*t11     <- this exact f32 accumulation order
       v = v * 0x401ccccd (=2.45)
       (behind an `s_cbranch` the kernel divides by the exposure and applies an inverse tone map;
        that branch is skipped whenever e > 0, which is true for every run modelled here, and the
        division by e = 1.0 would be exact anyway.)
   OUT OF BOUNDS: if Cx or Cy is outside [0, W15] x [0, H15] (0x2080.., 0x2d70.., 0x33ac..) the
   sample is not computed and the LDS byte keeps the value the register already held - measured
   == the colour feature c: L{3+3s+c} == L{c} at 100% for all four samples, both frames, the
   oracle and the synthetic runs (`oob_mask`).  Then dq.

5. FEATURE 15 - motion energy  (`reset_feature`, 0xad0..0xc14 + 0x1d00..0x1e38)
   prev_out = the args[0x30] write (reset-dependent, see 7);  interp = bilinear of the four
   prev_mvec[p] taps (0x2040) at the RENDER-grid position
       ux = F28x + (x+0.5) - F20x + (mvoffx + prev_out.x)
       u = W * clamp(ux / W, 0, 1) - 0.5            [two f32 muls through 1/W]
       then max(-2), NaN -> -0.5, min(W+2), snap/tap/frac exactly as in 4 (clamped taps as in 4)
   dx = (mvoffx + prev_out.x) - interp_x * (1/3),  dy likewise
   S = (k*dx)^2 + (k*dy)^2,  k = 4884.33544921875/fH  (args 0xe8, 0x4598a2af as f32)
   r = sqrt(S);  the ISA's nextafter-down selection: r_down = bits(r)-1, r_up = bits(r)+1,
       r10 = fma(-r_up, r, S),  root = (r10 > 0) ? r_up : ((r10 <= 0) ? r_down : r)
       (<= 1 f16 ulp, invisible after dq)
   v = 0.22356846928596497 (0x3e64ef22, args 0xd0) * ln(1 + root),  ln(z) = log2(z) * 0x3f317218
   NO position guard: masking it with the sample-3 bounds costs exactly the OOB pixels
   (97.83% -> 100.0000% without it); S = 0 already gives ln(1) = 0 -> dq(0) = 0, so the feature
   is nonzero exactly where this frame's MV differs from the interpolated previous MV - on the
   oracle only frame 0.  The kernel does not read Args+0xd8 in this formula: reset enters only
   through the mask in 7.  Then dq.
   Frame 0: S = k^2*|g_mv|^2 = 11.1766 -> v = 0.328349, the exact fp8 midpoint of 0.3125 and
   0.34375, so the dither alone picks between the two observed codes (33%/67% split).

6. FEATURES 16..19 - the warped hidden state  (`hidden_features`, 0x26c..0x670, 0xc5c..0xf70)
   Position: the SAME words as sample 0 (%492/%493), then dec0's `axis` in enc0's op order:
       t  = fma(C, 1/W15, F28x/W)      [ONE fused multiply-add; args 0xec/0xf4 and 0xf0/0xf8]
       c  = n*((1/n)*(n*t)) - 0.5       [n = W or H, 1/n = args 0xac/0xb0 - three f32 muls]
       c  = max(c, -2); NaN -> -0.5; c = min(c, n+2)
       snap = floor(c*256 + 0.5)*(1/256); tap = floor(snap); frac = snap - tap
   Taps of hidden (i8x4, args 0x28): the clamp-to-edge rule of 4 with n = W / H.
   Value per channel k (0x65c.., weights %1490.0/%1513.0/%1489/%1488):
       t = max(-1, i8 / 127)        (0x3c010204 = f32(1/127); i8 is a signed byte extract)
       w = {(1-fy)(1-fx), (1-fy)fx, fy(1-fx), fy*fx} for the taps (r0c0, r0c1, r1c0, r1c1)
       v = ((w00*t00 + w01*t01) + w10*t10) + w11*t11
   GUARD: the taps are only read when the sample-0 position lies inside [0,W15] x [0,H15]
   (0x39c0..0x39f8), otherwise the byte stays 0 (2.01% of a random-MV run's pixels).  Then dq.

7. THE THREE MV OUTPUTS  (`mv_outputs`, 0x1738..0x1e28)
   The 3x3 depth tournament over g_dep (rows/cols clamped to the image, y = MIRRORED row):
     t0 = centre (y,x), t1 = (y-1,x-1), t2 = (y-1,x), t3 = (y-1,x+1), t4 = (y,x-1),
     t5 = (y,x+1), t6 = (y+1,x-1), t7 = (y+1,x), t8 = (y+1,x+1)      <- note the t3/t4/t5 order
     max = (mx >= t) ? mx : t ;  min = (t < mn) ? t : mn     (ties keep the EARLIER element; the
     initial pair uses (t0 >= t1) for max and (t0 > t1) for min)  -> P_min, P_max, centre.
   Then f16(3.0 * (mvoff + g_mv[P]))  (F34x/mrx is exactly 3.0 in f32), stored only for raw y < H
   (0x1d38 guards the stores with cmp_gt(H, y), so the buffers are H rows, not H0):
       args[0x30] prev_mvec[p^1] = reset ? P_max : P_min
       args[0x38] mv_ero         = reset ? P_min : P_max
       args[0x40] mv_nondil      = the centre
   The reset select is the branch at 0x1aec on (exec & ~reset-mask), where the mask comes from
   s54 = args[0xd8]: %38 = s_cmp_lg_u32(reset, 0), %40 = (scc ? -1 : 0).  Measured 100% on
   frames_A frame0 (reset=1) / frame1 (reset=0) and on frames_C frame0 - a copy of frames_A with
   ONLY params_0.reset=false, which flips the assignment: reset is the single variable.
   On the oracle every g_mv read coincides (constant MV), so all three outputs equal 3*g_mv and A1
   cannot distinguish the select.

8. t0 - THE EMBED  (`t0()`, the GEMM from 0x3a00; layout from build/work/review/enc0/feat.py)
   The kernel stages 32 bytes per pixel in LDS, stored byte j = LOGICAL feature P[j] (P/IP are in
   sr_stage_model).  The weight tile's column f of frag(64,128) is the LOGICAL feature index, so
   a probe's stored-order grid must be re-ordered with IP before the GEMM - without that reorder
   the GEMM gives 4.96% instead of 99.96%.
     Wm = LUT[enc0.bin[4096 + frag_offsets(64,128)]]                      (64, 128)
     b  = enc0.bin[12288:12416] as f16, P-permuted: b[c] = raw[(c//32)*32 + IP[c%32]]
     f  = LUT[feats]      (the 20 logical features; 12 zero channels are appended - logical
          20..31 are never written, verified 0.00% nonzero in every probe)
     acc = f16(b);  for q,(dx,dy) in ((0,0),(1,0),(0,1),(1,1)):
              acc = f16(acc + f[dy::2, dx::2] @ Wm[:, q*32:q*32+32].T)
     t0 = q8(clip(acc, -6.28125, 6.28125))     (m.q8 = RNE; the values are already on the grid)
   decode_act(raw, 64, 80, 128) turns the oracle's raw [C/32][H0/2][W0/2][32] bytes back into the
   (80,128,64) logical grid for comparison.  Every intermediate add is f16 (one rounding each).

9. CONSTANTS (f32 bit patterns)
   1/1023 = 0x3a802008    2.45 = 0x401ccccd    log10(2) = 0x3e9a209b    0.36 = 0x3eb851ec
   log2(10) = 0x40549a78   ln(2) = 0x3f317218   1/0.36 = 0x4031c71c     -0.36 = 0xbeb851ec
   -100 = 0xc2c80000       1e-4 = 0x38d1b717    3.0 = 0x40400000        1/127 = 0x3c010204
   4884.33544921875 = 0x4598a2af    0.22356846928596497 = 0x3e64ef22    1024 = 0x44800000

10. KNOWN CAVEATS / OPEN
   - The branch offsets printed in enc0_loop.ssa are not byte-reliable (a SOPP is 8 B while the
     listing sometimes advances only 4), so the two select paths in 7 were pinned by register
     contents plus measurements rather than by the printed offsets.
   - args 0x80 is read twice by the kernel: expo[p] (the e used above) and expo[p^1] (via
     %48 = expo + (p^1)*2, 0x104+).  Both frames of every run modelled here have equal exposures,
     so this is not exercised; load() reads expo[p].
   - The two enc0feat probe runs of one frames dir diverge in the frame-1 history
     (start.dhist / start.hidden1 differ between *_enc0feat_0 and *_enc0feat_1024), so each probe
     half (rows y%8 < 4 / >= 4) may only be compared against the inputs of the run that produced
     it - build/agents/enc0-model-3/check_all.py stitches this.
====================================================================================================
"""
import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sr_stage_model as m  # noqa: E402
from sr_pack import frag_offsets  # noqa: E402  (NeuralHub's packer: read-only)

f32 = np.float32
f64 = np.float64
f16 = m.f16
KERNEL = "_ZN2sr2io9k_sr_enc0ENS0_8Enc0ArgsE"
REPO = m.REPO
NFEAT = 20           # logical features used by the GEMM (the rest of the 32 are zero)
H0, W0 = 160, 256    # feature grid (render rows padded to 2 x 80 cells)


# ------------------------------------------------------------------ inputs
def parse_args(b):
    """Enc0Args (280 B, srref.cpp `enc0 (FUN_180116e70, preset M)`), every field the kernel reads."""
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
    a["depinv"] = struct.unpack_from("<I", b, 0xe0)[0]
    a["mvpitch"] = struct.unpack_from("<I", b, 0xe4)[0]
    a["ldr"], a["mvhi"] = struct.unpack_from("<2I", b, 0xfc)
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
    a["dither"] = np.frombuffer((REPO / "build/re/packed/M/enc0.bin").read_bytes()[:4096], np.uint16)
    return a, g


# ------------------------------------------------------------------ the feature quantizer (dq)
def dq(x, dither, fc):
    """Dithered E4M3 quantizer of a feature grid x (H0 x W0, f32). x16 = f16(min(x, 1024)); the
    dither word d = dither[((fc&7)<<8) | ((y&15)<<4) | (x&15)] >> 9 is OR-ed into the exponent
    E = x16 & 0xfc00 (K = E | d, an f16 bit pattern); q = f16(f16(x16 + K) - E) & 0xff80 (three
    mantissa bits kept, truncated toward zero), then E4M3 (RNE onto the grid)."""
    x16 = f16(np.minimum(np.asarray(x, f32), f32(1024.0))).view(np.uint16)
    yy, xx = np.mgrid[0:x16.shape[0], 0:x16.shape[1]]
    idx = ((fc & 7) << 8) | ((yy & 15) << 4) | (xx & 15)
    d = (dither[idx] >> 9).astype(np.uint16)
    E = x16 & np.uint16(0xFC00)
    K = (E | d).view(np.float16)
    a = f16(x16.view(np.float16).astype(f32) + K.astype(f32))
    b = f16(a.astype(f32) - E.view(np.float16).astype(f32))
    t = (b.view(np.uint16) & np.uint16(0xFF80)).view(np.float16).astype(f32)
    return q8r(np.clip(t, -448, 448))


# ------------------------------------------------------------------ features 0..2: colour
C_CLAMP = np.frombuffer(bytes.fromhex("0000fe46"), np.float32)[0]   # 0x46ffe000 = 32752 (min before the log)
C_LO = np.frombuffer(bytes.fromhex("17b7d138"), np.float32)[0]      # 0x38d1b717 = 1e-4 (floor of 1 + 100 v)
C_LOG = np.frombuffer(bytes.fromhex("9b209a3e"), np.float32)[0]     # 0x3e9a209b = log10(2)
C_SCALE = np.frombuffer(bytes.fromhex("ec51b83e"), np.float32)[0]  # 0x3eb851ec = 0.36


def tm(v):
    """tm(v) = (log2(max(1 + 100 min(v, 32752), 1e-4)) * log10(2)) * 0.36, f32 steps (fma for 1 + 100v);
    NaN -> 0 (the kernel selects log2(1) there)."""
    v = np.asarray(v, f32)
    vc = np.where(v > C_CLAMP, C_CLAMP, v).astype(f32)
    t = (np.float64(100) * vc.astype(np.float64) + 1.0).astype(f32)      # fma(100, v, 1)
    t = np.fmax(t, C_LO)
    r = (np.log2(t).astype(f32) * C_LOG).astype(f32)
    r = (r * C_SCALE).astype(f32)
    return np.where(np.isnan(v), f32(0), r).astype(f32)


def mirror_rows(y, H):
    """Row index of the g_col texel the kernel reads: |y| if |y| < H, else 2H-2-|y|, clamped to [0, H-1]."""
    ay = np.abs(y)
    r = np.where(ay < H, ay, 2 * H - 2 - ay)
    return np.clip(r, 0, H - 1)


def colour_features(a, g):
    """Logical features 0, 1, 2 = tm(e * rgb) of the reflected g_col texel (rows 0..159), before dq."""
    H = a["H"]
    rows = mirror_rows(np.arange(H0), H)
    col = np.ascontiguousarray(g["g_col"][rows][..., :3])       # (H0, W, 3) f16 bits
    rgb = col.view(np.float16).astype(f32)
    v = (rgb * a["e"]).astype(f32)
    return tm(v)                                                # (H0, W, 3)


# ------------------------------------------------------------------ features 3..14: four dhist samples
INV1023 = np.frombuffer(bytes.fromhex("0820803a"), np.float32)[0]   # 0x3a802008 = f32 1/1023
C245 = np.frombuffer(bytes.fromhex("cdcc1c40"), np.float32)[0]      # 0x401ccccd = 2.45


def _axis(C, hdim):
    """One axis of a dhist sample position: the kernel's position word C (hW15 for x, hH15 for y)
    -> (tap, frac) on the dhist texel grid. Reproduces the kernel's op sequence, including the
    (0.5/h)*C -> h*(...) round trip and the [-2, h+2] clamp (NaN -> -0.5)."""
    a = (f32(0.5) / f32(hdim)) * C
    b = f32(hdim) * a
    u = b - f32(0.5)
    u = np.maximum(u, f32(-2.0))
    u = np.where(np.isnan(u), f32(-0.5), u)
    u = np.minimum(u, f32(hdim + 2))
    s = np.floor(u * f32(256.0) + f32(0.5)) * f32(1.0 / 256.0)
    tap = np.floor(s)
    return tap, s - tap


def dhist_sample(Cx, Cy, chan, dh, hW, hH):
    """One bilinear dhist sample of channel `chan` (0=R,1=G,2=B), *2.45 (0x401ccccd), f32 ops in
    the kernel's order: t = bits/1023 per tap, v = ((w00*t00+w01*t01)+w10*t10)+w11*t11."""
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


def dhist_positions(a, g):
    """The four samples' kernel position words (Cx, Cy), (H0, W0) f32 each.
    Samples 0..2 share the base 3*((p+0.5) - F20); sample 3 has NO -F20 term.
    The MV terms are the kernel's own three outputs (prev_mvec[p^1] = the min-depth g_mv read,
    mv_nondil = the centre read, mv_ero = the max-depth read) and, for sample 3, the centre again."""
    fld = mv_fields(a, g)
    rows = mirror_rows(np.arange(H0), a["H"]).astype(f32)          # the kernel reflects y
    yy = rows[:, None] + f32(0.5)
    xx = np.arange(W0, dtype=f32)[None, :] + f32(0.5)
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
    """The kernel's validity guard for one dhist sample: its position word must lie in
    [0, W15] x [0, H15] (0x2080.. for sample0, 0x2d70.. for sample2, 0x33ac.. for sample3).
    Out of bounds -> the sample is not computed and the LDS byte keeps the value the register
    already held, which measured == the colour feature: L{3+3s+c} == L{c}, 100% everywhere."""
    W15, H15 = f32(a["W15"]), f32(a["H15"])
    return (Cx < 0) | (Cx > W15) | (Cy < 0) | (Cy > H15)


def dhist_features(a, g):
    """Logical features 3..14 = four bilinear dhist samples x R,G,B (raw, before dq).
    Where a sample's position is out of [0,W15]x[0,H15] the kernel does not compute it and the
    LDS byte keeps the colour feature's value (measured == L{c} 100% on oracle and synthetic)."""
    pos = dhist_positions(a, g)
    col = colour_features(a, g)
    out = np.zeros((H0, W0, 12), f32)
    for s, (Cx, Cy) in enumerate(pos):
        oob = oob_mask(Cx, Cy, a)
        for ch in range(3):
            v = dhist_sample(Cx, Cy, ch, g["dhist"], a["hW15"], a["hH15"])
            out[..., 3 * s + ch] = np.where(oob, col[..., ch], v)
    return out


# ------------------------------------------------------------------ features 16..19: warped hidden
def _axis_hidden(C, size, inv15, off):
    """The hidden buffer's sample axis (the same shape as dec0's warp, enc0's exact op order):
    t = fma(C, inv15, off); c = n*((1/n)*(n*t)) - 0.5; clamp [-2, n+2] (NaN -> -0.5); snap 1/256."""
    n = f32(size)
    t = (C.astype(f64) * f64(inv15) + f64(off)).astype(f32)          # one fused multiply-add
    c = n * ((f32(1) / n) * (n * t))
    c = c - f32(0.5)
    c = np.maximum(c, f32(-2.0))
    c = np.where(np.isnan(c), f32(-0.5), c)
    c = np.minimum(c, n + f32(2.0))
    s = np.floor(c * f32(256.0) + f32(0.5)) * f32(1.0 / 256.0)
    tap = np.floor(s)
    return tap, s - tap


def hidden_features(a, g):
    """Logical features 16..19 = the 4 int8 hidden channels warped at the sample-0 position
    (the SAME Cx, Cy words as dhist sample 0, plus F28/size), value = int8/127 clamped to >= -1.
    Read only where the sample-0 position is inside [0,W15]x[0,H15], else the byte stays 0."""
    H, W = a["H"], a["W"]
    hin = g["hidden_p"].astype(np.int32)                            # (H, W, 4)
    Cx, Cy = dhist_positions(a, g)[0]
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
    # the hidden taps are only read when the warp position (the sample-0 words, %492/%493) is inside
    # [0, W15] x [0, H15] (0x39c0..0x39f8); outside them the kernel leaves the byte at 0.
    valid = ~oob_mask(*dhist_positions(a, g)[:1][0], a)
    out = np.zeros((H0, W0, 4), f32)
    for k in range(4):
        q = lambda rr, cc: np.maximum(f32(-1.0), hin[rr, cc, k].astype(f32) * f32(1.0 / 127.0))
        t00, t01, t10, t11 = q(y0, x0), q(y0, x1), q(y1, x0), q(y1, x1)
        v = ((w00 * t00 + w01 * t01) + w10 * t10) + w11 * t11
        out[..., k] = np.where(valid, v, f32(0))
    return out


# ------------------------------------------------------------------ feature 15: motion energy
def reset_feature(a, g):
    """Logical feature 15 (raw, before dq): the kernel's `0.22356846928596497 * ln(1 + sqrt(S))`
    with S = (k*dx)^2 + (k*dy)^2, k = 4884.33544921875/H (args 0xe8), d = mv_current - interp/3
    (the MV difference between this frame's prev_mvec[p^1] write and the bilinear interpolation of
    the prev_mvec[p] taps).  NOTE: the kernel never reads Args+0xd8 (reset) here - reset enters
    through the mask of mv_outputs only; the feature is nonzero where the MV difference is."""
    H, W = a["H"], a["W"]
    rows = mirror_rows(np.arange(H0), H).astype(f32)
    yy = rows[:, None] + f32(0.5)
    xx = np.arange(W0, dtype=f32)[None, :] + f32(0.5)
    fld = mv_fields(a, g)
    cur = fld["prev_out"]                          # prev_mvec[p^1]/3 = the B read (%382.0/%385)
    mvx, mvy = cur[..., 0], cur[..., 1]
    F20x, F20y, F28x, F28y = (f32(a[k]) for k in ("F20x", "F20y", "F28x", "F28y"))
    # ---- bilinear interpolation of the prev_mvec[p] taps (render grid, clamped [0,1] norm)
    def axis(u, n):
        t = (u.astype(f64) / f64(n)).astype(f32)                    # u * (1/n) as one f32 mul
        t = np.minimum(np.maximum(t, f32(0.0)), f32(1.0))
        c = f32(n) * t - f32(0.5)
        c = np.maximum(c, f32(-2.0))
        c = np.where(np.isnan(c), f32(-0.5), c)
        c = np.minimum(c, f32(n + 2.0))
        s = np.floor(c * f32(256.0) + f32(0.5)) * f32(1.0 / 256.0)
        tap = np.floor(s)
        return tap, s - tap

    ux = F28x + (xx - F20x) + mvx                                  # f32 add order: F28x + ((p+0.5)-F20) + mv
    uy = F28y + (yy - F20y) + mvy
    px, fx = axis(ux, W)
    py, fy = axis(uy, H)
    ix, iy = px.astype(np.int64), py.astype(np.int64)
    x0 = np.clip(ix, 0, W - 1); x1 = np.clip(np.maximum(ix, -1) + 1, 0, W - 1)
    y0 = np.clip(iy, 0, H - 1); y1 = np.clip(np.maximum(iy, -1) + 1, 0, H - 1)
    pm = g["prev_mvec_p"].astype(f32)                             # (H, W, 2) f16 bits -> f32
    w00 = (f32(1) - fy) * (f32(1) - fx)
    w01 = (f32(1) - fy) * fx
    w10 = fy * (f32(1) - fx)
    w11 = fy * fx
    ip = np.zeros((H0, W0, 2), f32)
    for k in range(2):
        t00, t01, t10, t11 = pm[y0, x0, k], pm[y0, x1, k], pm[y1, x0, k], pm[y1, x1, k]
        ip[..., k] = ((w00 * t00 + w01 * t01) + w10 * t10) + w11 * t11
    # ---- the MV difference and the energy S
    kk = f32(np.frombuffer(bytes.fromhex("afa29845"), np.float32)[0])   # 0x4598a2af = 4884.33544921875
    k = kk / f32(H)
    dx = (mvx - ip[..., 0] * f32(1.0 / 3.0)).astype(f32)
    dy = (mvy - ip[..., 1] * f32(1.0 / 3.0)).astype(f32)
    ax, ay = (k * dx).astype(f32), (k * dy).astype(f32)
    S = (ax * ax + ay * ay).astype(f32)
    # sqrt -> nextafter-down selection (the ISA's v_fma pair), then 1 + sqrt, ln, * 0.22356846928596497
    rt = np.sqrt(S, dtype=f32)
    bits = rt.view(np.uint32)
    down = (bits - np.uint32(1)).view(f32)
    up = (bits + np.uint32(1)).view(f32)
    r10 = (S - up * rt).astype(f32)
    r17 = np.where(r10 <= 0, down, rt)
    root = np.where(r10 > 0, up, r17)
    ln2 = np.frombuffer(bytes.fromhex("1872313f"), np.float32)[0]     # 0x3f317218 = ln(2)
    SC = np.frombuffer(bytes.fromhex("22ef643e"), np.float32)[0]      # 0x3e64ef22 = 0.22356846928596497
    v = (f32(1.0) + root).astype(f32)
    v = (np.log2(v).astype(f32) * ln2).astype(f32) * SC
    # NOTE: no position guard here - the kernel's vcc at 0x39b8 (the one in front of this store)
    # does NOT zero the feature where the sample-3 position is out of bounds (masking it costs
    # exactly the 1.96% OOB pixels, measured 97.83% -> 100% without it); S = 0 already gives 0.
    return v


def q8r(x):
    """f32 -> fp8 value with RNE (ties-to-even), saturating at +-448: what the hardware
    `v_cvt_pk_fp8` does after the kernel's `& 0xff80` truncation. m.q8's *subnormal* path
    (to_e4m3: floor(a*512+0.5)) rounds half-up, which differs from RNE at the fp8 midpoints
    |t| = (n+0.5)*2^-9 with n even - measured to be exactly the pixels where the model missed
    the probe (the shader is f32_to_e4m3_rne, see sr_stage_model.to_e4m3's docstring)."""
    x = np.asarray(x, np.float32)
    x = np.where(np.isnan(x), np.float32(0), x)     # the kernel's cndmask(0, ., not-NaN) in dq
    sign = np.where(x < 0, 0x80, 0).astype(np.uint32)
    a = np.minimum(np.abs(x), np.float32(448.0))
    bits = a.view(np.uint32)
    s = a * np.float32(512.0)                       # exact: subnormal grid = 2^-9
    n = np.floor(s)
    d = s - n
    sub = np.where((d > np.float32(0.5)) | ((d == np.float32(0.5)) & ((n.astype(np.int64) & 1) == 1)),
                   n + np.float32(1), n)
    norm = ((bits + np.uint32(0x407ffff) + ((bits >> 20) & 1)) >> 20) & 0x7f
    code = (np.where(a < np.float32(0.015625), sub, norm).astype(np.uint32) | sign).astype(np.uint8)
    return m.LUT[code]


def features(a, g):
    """(H0, W0, NFEAT) logical features (fp8 values): the GEMM's input, one channel per logical index."""
    x = np.zeros((H0, W0, NFEAT), f32)
    x[..., 0:3] = colour_features(a, g)
    x[..., 3:15] = dhist_features(a, g)
    x[..., 15] = reset_feature(a, g)
    x[..., 16:20] = hidden_features(a, g)
    return np.stack([dq(x[..., L], a["dither"], a["fc"]) for L in range(NFEAT)], -1)


# ------------------------------------------------------------------ the GEMM (t0)
def wmat():
    """(Wm, bias): the 64x128 weight tile (frag(64,128), column c = q*32 + logical feature f) and
    the 64 f16 bias, P-permuted, both from enc0.bin."""
    blob = np.frombuffer((REPO / "build/re/packed/M/enc0.bin").read_bytes(), np.uint8)
    Wm = m.LUT[blob[4096 + frag_offsets(64, 128)]].astype(f32)              # (64, 128)
    raw = np.frombuffer(blob[12288:12288 + 128].tobytes(), np.float16).astype(f32)
    c = np.arange(64)
    return Wm, raw[(c // 32) * 32 + m.IP[c % 32]]


def t0(feats):
    """The embed: t0 = fp8(clamp(f16(f16(f16(f16(b + S_00) + S_10) + S_01) + S_11), +-6.28125)),
    S_q = W[:, 32q:32q+32] . feat(sub-pixel q), q = dx + 2*dy.  `feats` is either the logical
    (H0, W0, 20) fp8 VALUES from features(), or a probe's (H0, W0, 32) stored-order fp8 CODES
    (stored byte j = logical P[j], so they are re-ordered to logical before the GEMM)."""
    Wm, b = wmat()
    if feats.dtype == np.uint8:
        f = m.LUT[feats[..., m.IP]].astype(f32)
    else:
        f = np.asarray(feats, f32)
        if f.shape[-1] < 32:                       # logical features 20..31 are never written (all 0)
            f = np.pad(f, [(0, 0), (0, 0), (0, 32 - f.shape[-1])])
    acc = np.broadcast_to(m.f16(b), (H0 // 2, W0 // 2, 64)).astype(f32)
    for q, (dx, dy) in enumerate(((0, 0), (1, 0), (0, 1), (1, 1))):
        acc = m.f16(acc + f[dy::2, dx::2] @ Wm[:, q * 32:(q + 1) * 32].T)
    return m.q8(np.clip(acc, -f32(6.28125), f32(6.28125)))


# ------------------------------------------------------------------ the MV outputs
def mv_fields(a, g):
    """The kernel's g_mv reads, as f32 (H0, W0, 2) arrays = mvoff + g_mv[pos]:
      'min'     = B (0x1c98): g_mv at the MIN-depth neighbour of the 3x3 -> mv_ero on reset frames
      'max'     = C (0x1ca4): g_mv at the MAX-depth neighbour           -> prev_mvec[p^1] then
      'center'  = D (0x1cb0): g_mv at (clamp(y), clamp(x))              -> mv_nondil / sample 3
      'prev_out'/'ero' = the two outputs of mv_outputs' reset select.
    The 3x3 tournament (0x1738..0x1b20) walks t0..t8 = centre, (y-1,x-1), (y-1,x), (y-1,x+1),
    (y,x-1), (y,x+1), (y+1,x-1), (y+1,x), (y+1,x+1) with rows/cols clamped to the image, keeping
    the EARLIER element on ties (max: `mx >= t`, min: `t < mn`)."""
    H, W = a["H"], a["W"]
    dep = g["g_dep"]
    mv = g["g_mv"].astype(f32)
    rows = np.clip(mirror_rows(np.arange(H0), H), 0, H - 1)
    y = np.broadcast_to(rows[:, None], (H0, W0))
    x = np.broadcast_to(np.clip(np.arange(W0), 0, W - 1)[None, :], (H0, W0))

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
    # 0x1aec: `s_cbranch_vccnz` on (exec & ~reset-mask).  reset=0 -> both movs: B(load)=min,
    # C(load)=max; reset=1 -> the fall-through sets v38=%319=(min pair) while v37/v40 still hold
    # %301/%317 (the max pair, written by the cndmasks at 0x1a40/0x1ad4): B=max, C=min.
    # Measured 100% both ways on synthetic frames (frames_A f0 reset=1, f1 reset=0, frames_C f0
    # reset=false -> the assignment flips with reset alone).
    r = bool(a["reset"])
    out["prev_out"] = out["max"] if r else out["min"]     # load B -> args[0x30] = prev_mvec[p^1]
    out["ero"] = out["min"] if r else out["max"]          # load C -> args[0x38] = mv_ero
    return out


def mv_outputs(a, g):
    """prev_mvec[p^1], mv_ero, mv_nondil (f16 (H,W,2)): the kernel's three `v_cvt_f16(3.0*(mvoff + g_mv
    read>))` stores (0x1d74..0x1dc4): prev_mvec[p^1] = the min-depth read (B), mv_ero = the max-depth
    read (C), mv_nondil = the centre read (D).  F34x/mrx rounds to exactly 3.0 in f32."""
    fld = mv_fields(a, g)
    H = a["H"]
    s = f32(3.0)
    # the kernel stores the MVs only for raw y < H (0x1d38: cmp_gt(H, y)), so the buffers are H rows.
    # The key is the args[0x30] slot = prev_mvec[p^1] (file 002.prev_mvec1.bin for p=0, ...0 for p=1).
    return {"prev_mvec1": m.f16(s * fld["prev_out"])[:H], "mv_ero": m.f16(s * fld["ero"])[:H],
            "mv_nondil": m.f16(s * fld["center"])[:H]}


# ------------------------------------------------------------------ acceptance A1
def self_check(dump=None, frames=(0, 1), quiet=False):
    """A1: model vs the oracle dump, frames 0 and 1 - t0 bit-exact % (>= 99) + the three MV buffers."""
    dump = Path(dump) if dump else REPO / "build/oracle/oracle-8dbf027a9e-s20261025/sr_M/dump"
    ok = True
    for fr in frames:
        a, g = load(dump, fr)
        fe = features(a, g)
        t = t0(fe)
        orc = m.decode_act(bytes((dump / f"frame{fr}/002.t0.bin").read_bytes()), 64, 80, 128)
        pct = 100.0 * float(np.mean(t == orc))
        good = pct >= 99.0
        ok &= good
        if not quiet:
            print(f"frame{fr}: t0 bit-exact {pct:.4f}%  (target >= 99%)  {'PASS' if good else 'FAIL'}")
        for name in ("prev_mvec1", "mv_ero", "mv_nondil"):
            # the dump writes `002.<buf>` only when the launch changed the buffer: frame1 keeps
            # mv_ero/mv_nondil (constant MV -> identical bytes) and names the prev_mvec write by
            # the p^1 slot, so fall back through the candidates.
            fam = ("002.prev_mvec1.bin", "002.prev_mvec0.bin",
                   "start.prev_mvec1.bin", "start.prev_mvec0.bin") if name == "prev_mvec1" else \
                  (f"002.{name}.bin", f"start.{name}.bin")
            refp = next((dump / f"frame{fr}" / n for n in fam if (dump / f"frame{fr}" / n).exists()), None)
            ref = np.frombuffer(refp.read_bytes(),
                                np.uint16)[: a["H"] * a["W"] * 2].reshape(a["H"], a["W"], 2)
            mine = mv_outputs(a, g)[name].view(np.uint16)
            n = int(np.sum(mine == ref))
            ntot = ref.size
            good = n == ntot
            ok &= good
            if not quiet:
                print(f"          {name}: {n}/{ntot} bit-exact ({100.0 * n / ntot:.4f}%)  "
                      f"{'PASS' if good else 'FAIL'}")
    if not quiet:
        print("A1:", "PASS" if ok else "FAIL")
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", type=int, default=None)
    ap.add_argument("--oracle", default=None)
    ap.add_argument("--gpu", default=None, help="a GPU probe dump dir (same layout) instead of the oracle")
    ap.add_argument("--out", default=None, help="write the model's t0/mv buffers here")
    ap.add_argument("--self-check", action="store_true")
    args = ap.parse_args(argv)
    if args.self_check:
        return 0 if self_check(args.oracle) else 1
    dump = Path(args.gpu or args.oracle or (REPO / "build/oracle/oracle-8dbf027a9e-s20261025/sr_M/dump"))
    fr = 0 if args.frame is None else args.frame
    a, g = load(dump, fr)
    fe = features(a, g)
    t = t0(fe)
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        m.encode_act(t).tofile(out / f"frame{fr}.t0.bin")
        for k, v in mv_outputs(a, g).items():
            v.tofile(out / f"frame{fr}.{k}.bin")
    else:
        print(f"frame{fr}: features {fe.shape} fp8, t0 {t.shape}, "
              f"t0 range [{t.min()}, {t.max()}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
