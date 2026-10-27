#!/usr/bin/env python3
"""tools/rr/rr_level_model.py: the NumPy instrument for one rrk::k_level launch of DLSS-RR's U-Net
(preset F). The RR analogue of tools/sr/sr_stage_model.py: plumbing, scoring and a switchable first
hypothesis. The RR block maths is NOT decoded; everything marked GUESS or BORROWED is a hypothesis.

    build/venv/bin/python -I tools/rr/rr_level_model.py --launch N [--frame F] [--set k=v ...] [--out DIR]
        [--attn-only FILE] [--inject-postattn FILE]
    build/venv/bin/python -I tools/rr/rr_level_model.py --all [--frame F] [--set k=v ...] [--out DIR]
    build/venv/bin/python -I tools/rr/rr_level_model.py --self-check [--frame F]

--all runs every k_level launch of the frame (frame 1: launches 5..20, frame 0: 4..19).

One launch of the oracle's dump is modelled. The Args (W, H, sx, sy, the 20 item offsets, the
scalars) and the pointer roles come from launches.jsonl; the weight blob is the oracle's own
buffer (buffer_before on its blob name). Every activation input is the oracle's own bytes as they
stood before the launch (buffer_before: the newest earlier dump of that buffer in the frame, else
the frame's start.<buf>.bin). Every output the launch writes is scored against its dump
NNN.<buf>.bin with tools/report/launch_report.py's measure() (peak, PSNR, r, rRMSE, bit-exact).

ESTABLISHED BY THIS INSTRUMENT (self-check, see the report):
  * activation layout: [C/32][H][W][32] fp8, logical channel cg*32 + P[i] at stored index i (the same
    P as SR's stage activations). Evidence: the enc0 feature buffer E0be8
    (64 channels) has its 21 zero-padding channels (logical 43..63) exactly all-zero at the
    positions this layout gives; the mode-1 and mode-0 down-projections reproduce the stored
    downsampled outputs (r = 0.99998 for unit 1). See the self-check output.
  * weight blob: the byte map of tools/rr/rr_layout.py. Every blob is decoded to logical
    matrices and re-packed through rr_pack.pack (the packer's own encoder); the re-packed blob and
    item offsets equal the oracle's bytes (self-check A2).
  * the 2x2 patch-merge down-projection (modes 0/1): out_down = mA8 @ [x(0,0) x(1,0) x(0,1) x(1,1)]
    + v170, the concatenation in index q*C + c with q = dx + 2*dy (perm_cols stores mA8 in this order).
  * the 8 B/px extra outputs (O98) are four f16 per pixel, [N][4]; lane 3 is exactly 0 in every one
    the oracle writes in frame 1 (encoders, decoder _1 units, unit 15). The 2 B/px output (OA0) is one
    f16 per pixel (its values are not decoded).
  * the render-resolution outputs of unit 15 (OC0/OC8/OD0/OD8) ARE floats/bytes of the dec0
    heads: OC0 = i8 x 4 (signed tanh bytes), OC8/OD0 = u8 x 4 (OC8 byte 3 = 0), OD8 = f16 x 4
    (lane 3 = 0), all at Wr x Hr = Args +0xe0 x +0xe4 (256 x 144); decoded in the UNIT 15 SPEC below.

FIRST HYPOTHESIS (every part switchable with --set; defaults are the best-stated choice, not tuned):
  * RMSNorm (no gain item exists in the RR blob)                          GUESS (norm=rms|sum)
  * windowed attention, 4x4 windows, shifts (sx, sy), tile 8             BORROWED from SR
  * logits: separate Q.K^T (qk=sep) or the gram form z^T G z (qk=gram), z = [Q; K] = group rows
  * V = K (qk_order, v=k|q); there is no separate V matrix in the blob   GUESS (SR's V = K)
  * per-head Wo = m48 columns 32h..32h+31; v128 (C) added after attention  GUESS (SR's v270 role)
  * MLP: fc1 (4C x C) + v140, act (gelu|silu|erfgelu|nr), fp8 hidden, fc2 (C x 4C) + v158
  * softmax-like exp: SR's ISA-derived bit-built exp (exp=nr)            BORROWED from SR
  * mode 0 / 1 down-projection: the verified relation above, with v170 and the clamp 6.28125 (BORROWED)
  * mode 1 entry merge (level 0 -> 1, mC8 @ merge2x2 + v188): linear                    GUESS
  * mode 2 (up): mE8 @ low + v1a0 (rows [q][c]), pixel-shuffled (up_pix), + skip before the block
  * mode 3 (dec*_1) and encoder heads: O98 lanes 0..2 = m108[:3] @ fp8(out) + v1b8, lane 3 = 0  GUESS
  * mode 5 (enc0): X = m1f8 @ act(m1d8 @ feat + v218) + v230, act = gelu                  GUESS
  * mode 4 (dec0_1, last unit, unit 15): fully decoded, see the UNIT 15 SPEC below;
    OC0/OC8/OD0/OD8 element types: OC0 = i8 x 4 (signed tanh bytes, scored as i8),
    OC8/OD0 = u8 x 4 (byte 3 of OC8 is 0), OD8 = f16 x 4
    (lane 3 is 0), all at render resolution Wr x Hr = Args +0xe0 x +0xe4 (256 x 144).
  * OA0 (2 B/px, one f16) is UNMODELLED (zeros) in every unit.
  * the extra INPUTS (+0x88, +0x90) are consumed only by the mode-3 and mode-4 heads.

NOT DECODED (docs/notes/k_rr_level.md; open questions in the report): the per-unit residuals of
the shared block maths (the block_v2 decode itself, Review R7b), the meaning of OA0, the role of
s1d2/s1d4/s1d6 (Args +0xac..+0xb4) beyond the LayerScale/skip uses stated above, the mode-2 head
(present in the blob, no output).

===============================================================================================
UNIT 15 SPEC (mode 4, k_level<32,2,4,0,0,4>, blob15 = dec0_1; Review u15, 2026-10-26).
Every formula, constant, rounding point and buffer encoding, read from the ISA listing
build/work/review/rr/u15.s (stores 0x6c78 E0bd4, 0x7064 E0c16, 0x7070 E0c1a, 0x775c E0c02,
0x790c E0c1e; the second pass, token y+4, mirrors them at 0x8078/0x8468/0x8474/0x8b3c/0x8cd4)
and probe-fitted against the oracle dumps (channels U15_CHANS; see build/agents/rr-u15/).

Block. The shared block_v2 runs on in = E0b80, but the heads read the block output as the f16
  v_pk_add_f16 result - unit 15 stores no +0x78 buffer, so there is no +-448 clip and no fp8
  roundtrip on this path (block_v2(..., fp8_out=False)).

Inputs and scalars.
  x88 = E0ba8 (Args +0x88): unit 0's o98, [H][W][4] f16 at level 0 (lane 3 is 0).
  x90 = E0bd8 (Args +0x90): unit 13's o98, [H/2][W/2][4] f16 at level 1.
  s1d0 = Args +0xa8 read as f16 (frame dumps: 2.30859375).
  dec0_s0 = Args +0xe8 f32 (8.0 = min(exp(f16 table 0x1820), 8), rr_pack.build_w);
  dec0_f0 = Args +0xec f32 (0.307675 = exp(f16 table 0x1830)); the packer's third dec0 scalar
  (exp(f16 0x1840) = 1.126047) is not passed in Args and this kernel never reads it.
  Render grid Wr x Hr = Args +0xe0 x +0xe4 (256 x 144); level-0 grid W x H = 256 x 160.
  dy-half select: the kernel's vcc = (16 > tid) splits each 8-row window pass into rows
  {0,1}/{2,3} (pass A) and {4,5}/{6,7} (pass B), i.e. dy0 <=> (y + sy) mod 4 < 2. Every head
  slot reads the SAME block channel in both halves (verified: refits on each half agree).

Shared non-linearities (rounding points explicit; f16() = RNE to f16, f32 ops as written).
  sg(h) = f16( v_rcp_f32(1 + v_exp_f32(F32_NEG_L2E * f32(f16(h)))) ) with
    F32_NEG_L2E = -log2e as f32 (0xbfb8aa3b) and v_exp_f32 = 2**y (correctly rounded via f64
    in the model), rcp correctly rounded.
  tanh(x) (x f32, sign preserved by bit merge):
    |x| <= 0.625 (0x3f200000): mag = fma(y, v, |x|) with y = x*x, v = |x| * P(y),
      P = ((((c0*y + c1)*y + c2)*y + c3)*y + c4) - five single-rounding fmaak steps,
      c0..c4 = f32 0xbbbac73d, 0x3ca908c9, 0xbd5c1c4e, 0x3e088382, 0xbeaaaa99 (the tanh/x series);
    else: a = 2|x|, t = f32(a * F32_LOG2E) (0x3fb8aa3b), k = rint(t),
      eps = fma(F32_LOG2E, a, -t), tail = fma(a, F32_L2E_GAP, eps) (gap = 0x32a5705f =
      log2e - f32(log2e)), frac = f32((t - k) + tail), e = ldexp(2**frac, k)
      (2** underflow -> 0, > f32 max -> +inf, as the cndmask/ldexp chain does),
      mag = fma(1/(1 + e), -2, 1);  result = signbit(x) * mag.
  clamp65504(v) (the O98 store guard): NaN/+inf -> 65504 (0x7bff), else min(v, 65504) in f16
    (-inf passes through, exactly like v_min_num + v_cndmask ordered/gt).

E0bd4 (o98, Args +0x98, [H][W][4] f16, store 0x6c78 / 0x8078).
  h_c = block output channel U15_CHANS[bd4_c] (channels 0..3), s_c = sg(h_c), s_3 = sg(h_3).
  up_c = u15_up(x90[c]) - the same 2x bilinear as head_dec (Review R7c, up2_bilinear):
    base = (p-1)>>1 (clamped; p <= 0 uses base 0 with weight 0), weight of base+1 =
    0.75 (p even, f16 0x3a00) / 0.25 (p odd, 0x3400), forced 0 at borders where base+1 is
    out of range; taps in the order (by,bx), (by,nx), (ny,bx), (ny,nx) of x90 at level-1
    coordinates bx = (x-1)>>1, by = (y-1)>>1 (0 when the coordinate is 0), nx/ny clamped to
    dim/2-1; f16 order: r = f16(f16((1-b)*f16((1-a)*t1) + (1-b)*f16(a*t2))
    + f16(b*f16((1-a)*t3)) + f16(b*f16(a*t4))).
  lanes c = 0..2: t = f16(f16(f16(s1d0 * s_c) * up_c) * s_3);  lane 3: t = 0 (the kernel's
    literal 0 operand, no x90 read).
  E0bd4[c] = clamp65504(f16(t + f16(f16(1 - s_3) * x88[c])))   (1 - s_3 in f16).
  Lane 3 is identically 0 (x88[3] = 0). Encoding: [H*W][4] f16, little-endian, row-major.

E0c16 (oc8, Args +0xc8, [Hr][Wr][4] u8, byte 3 = 0; store 0x7064 / 0x8468).
  byte i = rndne(255 * f32(sg(block channel 4+i))) for i = 0, 1, 2, byte i = 0 if the f16
  source is not ordered (never with finite block output); byte 3 = 0 (no store).
  Encoding: u8 x 4 per pixel, row-major (element type u8).

E0c1a (od0, Args +0xd0, [Hr][Wr][4] u8; store 0x7070 / 0x8474). Four-way softmax over block
  channels 10..13, entirely in f16:
    p = block channels, m = pairwise maximum of (p0, p1, p2, p3) with NaN propagation;
    d_i = f16(p_i - m);  prod_i = f16(F16_LOG2E * d_i) (F16_LOG2E = f16 0x3dc5);
    e_i = f16(2**f32(prod_i)), with the kernel's fix-up: if prod_i == f16 0x11c5 then e_i =
      f16 0x3c01 (1.0009765625);
    tot = f16(f16(e0 + e2) + f16(e1 + e3));  r = f16 reciprocal of tot;
    byte_i = rndne(255 * f32(f16(e_i * r))), 0 if the f16 ratio is unordered (i = 0..3).
  The four bytes sum to 253..257 in the oracle dumps. Encoding: u8 x 4, row-major.

E0c02 (oc0, Args +0xc0, [Hr][Wr][4] signed i8; store 0x775c / 0x8b3c).
  byte_i = rndne(127 * clamp(f32(f16(tanh(block channel 14+i))), -1, 1)) & 0xff (two's
  complement), 0 if the f16 is unordered. The tanh is the shared one above; its f16 rounding
  happens BEFORE the +-1 med3 (a no-op for finite tanh). Encoding: i8 x 4, row-major
  (element type i8; byte-identical to a u8 comparison, decoded signed for PSNR).

E0c1e (od8, Args +0xd8, [Hr][Wr][4] f16, lane 3 = 0; store 0x790c / 0x8cd4).
  A = f32(dec0_s0) * sg(clamp(block channel 7, -20, 20)),  B = dec0_s0 * sg(clamp(ch 8, +-20))
    (clamp bounds f16 0x4d00/0xcd00 = +-20; the x dec0_s0 product is plain f32);
  lane0 = f16(A * A), lane1 = f16(B * B);
  x = f32(dec0_f0) * f32(block channel 9); t = tanh(x) (f32);
  bracket: b = bits(t) as u32; lo = f16(A * B * f32(b - 3)), hi = f16(A * B * f32(b + 3))
    (u32 arithmetic = an +-3 f32-ulp reliability bracket);
  lane2 = lo if lo == hi else f16(A * B * f32(tanh_f64(x))) - the kernel's f64 helper
    (two v_cvt_f64_f32 blocks, taken when the bracket straddles an f16 rounding boundary,
    ~0.1% of pixels; modelled with np.tanh in f64);
  lane3 = 0. Encoding: [Hr*Wr][4] f16 little-endian, lane order A*A, B*B, A*B*tanh, 0.

Channels (U15_CHANS): the heads read block channels 0..3 (E0bd4 h0..h3), 4..6 (E0c16 bytes
  0..2), 7/8/9 (E0c1e A/B/p16), 10..13 (E0c1a bytes 0..3), 14..17 (E0c02 bytes 0..3) -
  fitted channel-exact on frame 1's dy<2 half, reproduced on dy>=2 and on frame 0 (the same
  channels; the per-pair half swap the cndmask/permlane chain suggests does NOT occur in the
  oracle data - refits on each half always select the same channel).

Residual (why the buffers stop short of 99%): the only uncertain input of every formula above
  is the shared block_v2 output img. A patchprobe of the hardware (tools/re/patchprobe.py
  --net rr, kernel+0x672c, storing the head's four sigma inputs v[28:29] into E0bd4, run
  build/work/review/patch/runs/u15h) settled it on frame 0, pass A (50% of the pixels):
    * channel identity: the stored h slots correlate 0.9999 with img channels 0..3 (runner-up
      0.980..0.986) - the map U15_CHANS is hardware-exact;
    * block error: probe h - img has mean ~0 (unbiased), median |d| 0.0024..0.0039, p90
      0.021..0.030, max 0.17..0.26 - i.e. block_v2's output is off by ~1-3 f16 codes;
    * head formula: recomputing E0bd4 from the PROBED h (+ the exact x88/x90/weights) matches
      the oracle at 99.76% element-exact (lanes 0..2: 99.66..99.72%, lane 3: 100%), PSNR
      112..121 dB - the ~0.24% residue is v_exp/v_rcp rounding vs the model's exp2/rcp.
  Feeding the model's own img instead lands the buffers at 19.7..68.2 dB / 51..75% element-
  exact: the whole gap is block_v2's output error (the same block that scores 34..45 dB on the
  other 15 units), not the head formulas. Inverting E0c16's oracle gates to h-buckets gives the
  same verdict (img inside the oracle bucket on 62..70% of pixels = the observed byte rates).
    * u15g (kernel+0x672c, pairs v[22:23] for channels 4..7): sigma -> rndne(255*x)
      reproduces E0c16 bytes 0..2 at 100.0000% and E0c1e's f16(A*A) at 100.0000% on the
      dy>=2 rows; the slots identify as channels 4,5,6,7 at corr 0.997..0.9997.
    * u15t (pairs for channels 14..17): the i8 tanh formula gives 100.0000% on E0c02 bytes
      0..1; u15f (channels 10..13): slots 2,3 = channels 12,13 at corr 0.9945/0.9979.
  Per-slot channel scans cross-validate on both dy halves and on frame 0 (a wrong channel fits
  ~0% - e.g. the E0c1a half-swap scores 5.5% vs 73.4% for the fitted tuple - the fitted ones
  fit 35..73% with consecutive channels 0..17). The render heads read channels through the
  kernel's cndmask/permlane gathers: the raw WMMA pair registers hold lane-distributed
  fragments (pre-gather probe reads therefore confirm only some slots), while the fitted net
  map is what the oracle fits, the cross-half refits and the cross-frame check all validate.
===============================================================================================
"""
import argparse
import json
import math
import os
import re
import struct
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "tools/sr"))
sys.path.insert(0, str(REPO / "tools/report"))
import rr_pack  # noqa: E402  (NeuralHub's packer: read-only; pack() is the encoder we re-pack through)
import rr_layout  # noqa: E402  (frag_closed, P, u_of: the byte map, checked by rr_layout.check_rules)
from sr_pack import to_e4m3  # noqa: E402  (FUN_180114320: the packer's f32 -> e4m3, RNE)
from launch_report import fp8_to_f32, measure  # noqa: E402  (the canonical per-launch scoring)

ORACLE = REPO / "build/oracle/oracle-8dbf027a9e-s20261025"
DUMP = ORACLE / "rr_F/dump"
PACKED = REPO / "build/rr/packed"
LUT = fp8_to_f32(bytes(range(256))).astype(np.float32)
P = np.array([rr_layout.P(i) for i in range(32)])      # stored index i holds logical channel P[i] of its 32
ROLE = {0x10: "in", 0x18: "skip", 0x20: "blob", 0x78: "out", 0x80: "down", 0x88: "x88", 0x90: "x90",
        0x98: "o98", 0xa0: "oa0", 0xc0: "oc0", 0xc8: "oc8", 0xd0: "od0", 0xd8: "od8"}
OUT_ROLES = ("out", "down", "o98", "oa0", "oc0", "oc8", "od0", "od8")
DEFAULTS = dict(
    norm="none",           # rms: x / sqrt(mean x^2 + 2^-14) (unit gain); sum: SR's x / sqrt(sum x^2 + 2^-14)
    snorm8=1,              # RMSNorm output to fp8 (SR default)
    qk="sep",              # sep: logits Q K^T;  gram: logits z^T G z with z = [Q; K] and G = gram[h]
    qk_order="qk",         # qk: rows 0..31 of the group are Q, 32..63 are K;  kq: the other way
    v="k",                 # V source: k (SR's V = K) or q
    qk8=1, ctx8=1,         # Q,K (or z) and the attention context to fp8 (SR defaults)
    prob8=0, v8=0,         # probabilities / V to fp8 (SR: off)
    exp="nr",              # nr: SR's ISA-derived exp; exact: np.exp
    act="nhgelu",          # gelu (tanh form) | silu | erfgelu | nr    (the MLP activation)
    v128="after_attn",     # after_attn: v128 added to the post-attention residual; off: not added
    entry_act="none",      # mode 1 entry merge (mC8): none | gelu     GUESS
    enc0_act="gelu",       # mode 5: act between m1d8 and m1f8        GUESS
    head_in="q8",          # mode 3 / encoder heads read fp8(out) (q8) or the f16 block output (f16)  GUESS
    proj_clamp=6.28125,    # the down-projection's clamp (0 = none); borrowed from SR's measured clamp
    down_bias=1,           # v170 added to the down-projection
    up_pix="xy",           # sub-pixel q -> (dx, dy): xy: q = dx + 2 dy (the merge order); yx: q = dy + 2 dx
    skip_at="before",      # mode 2: skip added before the block (before) | after | both | none
    mode4_guess=1,         # 1: mode 4's O98 lanes 0..2 = block output channels 0..2; 0: zeros   GUESS
    blk="v2",              # v1: the first hypothesis (4x4 windows, no LayerScale); v2: block_v2 (Review R7)
    relpos=1, den16=1, ls=1, acc16=1, chain=1, smax="isa", ctx16=2, chain_io=1, up8=1, upref="low", lscale=1.0, qq=1, pv8=1, skip_mix="s6skip", head="v2",   # block_v2: relpos bias, f16 softmax den + rcp, LayerScale, f16 K-chunk sums
)


# ------------------------------------------------------------------ activation layout
def decode_grouped(raw, C, H, W, perm=True):
    """[C/32][H][W][32] fp8 -> logical (H, W, C) f32: stored index i of group cg is channel cg*32 + P[i]."""
    raw = np.frombuffer(bytes(raw), np.uint8)
    if raw.size != C * H * W:
        raise SystemExit(f"activation of {raw.size} bytes, expected C={C} H={H} W={W} ({C*H*W})")
    a = LUT[raw.reshape(C // 32, H, W, 32)]
    out = np.zeros((H, W, C), np.float32)
    for cg in range(C // 32):
        out[:, :, cg * 32 + (P if perm else np.arange(32))] = a[cg]
    return out


def codes_grouped(x, perm=True):
    """(H, W, C) f32 -> stored fp8 codes [C/32][H][W][32] (RNE, saturating; the packer's to_e4m3)."""
    H, W, C = x.shape
    codes = to_e4m3(np.asarray(x, np.float32))
    out = np.zeros((C // 32, H, W, 32), np.uint8)
    for cg in range(C // 32):
        out[cg] = codes[:, :, cg * 32 + (P if perm else np.arange(32))]
    return out


def encode_grouped(x, perm=True):
    return codes_grouped(x, perm).tobytes()


def q8(x):
    return LUT[to_e4m3(np.asarray(x, np.float32))]


# ------------------------------------------------------------------ oracle plumbing
def load_rows(frame, dump=DUMP):
    return [json.loads(l) for l in open(Path(dump) / f"frame{frame}" / "launches.jsonl")]


def k_level_launches(rows):
    return sorted([r["n"] for r in rows if "k_level" in r["kernel"]])


def parse_symbol(sym):
    """rrk::k_level<C, g, mode, Cdown, Cup, n> from the mangled name."""
    ints = [int(v) for v in re.findall(r"Li(\d+)E", sym.split("k_level", 1)[1])[:6]]
    if len(ints) != 6:
        raise SystemExit(f"cannot parse {sym}")
    return dict(C=ints[0], g=ints[1], mode=ints[2], Cdown=ints[3], Cup=ints[4], n=ints[5])


def buffer_before(dump_frame, name, launch):
    """The bytes of `name` as they stood before `launch` (the newest earlier dump, else start)."""
    best = None
    for f in os.listdir(dump_frame):
        if f.endswith(f".{name}.bin") and f[:3].isdigit() and int(f[:3]) < launch:
            if best is None or int(f[:3]) > int(best[:3]):
                best = f
    p = Path(dump_frame) / (best if best else f"start.{name}.bin")
    return p.read_bytes()


def load_launch(frame, launch, dump=DUMP):
    rows = load_rows(frame, dump)
    d = next(r for r in rows if r["n"] == launch)
    a = bytes.fromhex(d["args"])
    W, H, sx, sy = struct.unpack_from("<4i", a, 0)
    items = list(struct.unpack_from("<20i", a, 0x28))
    blob_size = struct.unpack_from("<i", a, 0xb8)[0]
    scal = list(struct.unpack_from("<4I", a, 0xa8))
    # unit 15's render grid and dec0 scalars live at +0xe0 (Wr, Hr) / +0xe8 (s0, f0)
    Wr, Hr = struct.unpack_from("<2i", a, 0xe0) if len(a) >= 0xe8 else (W, H)
    dec0 = list(struct.unpack_from("<2f", a, 0xe8)) if len(a) >= 0xf0 else [0.0, 0.0]
    tmpl = parse_symbol(d["kernel"])
    ptr = {}
    for k, v in d["pointers"].items():
        off = int(k)
        if off in ROLE:
            ptr[ROLE[off]] = v.split("+")[0]
    unit = k_level_launches(rows).index(launch)
    return dict(d=d, W=W, H=H, sx=sx, sy=sy, items=items, blob_size=blob_size, scal=scal, dec0=dec0,
                Wr=Wr, Hr=Hr, tmpl=tmpl, ptr=ptr, unit=unit, frame_dir=Path(dump) / f"frame{frame}")


# ------------------------------------------------------------------ weights
def frag_codes(blob, off, rows, cols, permute=True):
    """fp8 fragment (rr_pack.frag_rr's order) -> the codes of the matrix as the packer was given it."""
    r16 = (rows + 15) // 16 * 16
    fo = rr_layout.frag_closed(r16, cols, permute)[:rows]
    return blob[off + fo]


def f16_bits(blob, off, n):
    """f16_rr's inverse: the vector as f16_rr was given it (uint16 bits). P-permuted in 32-groups."""
    raw = np.frombuffer(bytes(blob[off:off + 2 * n]), np.uint16)
    if n % 32:
        return raw.copy()
    # rr_pack.f16_rr(arange) gives, at stored index i, the logical index it holds (its own gather map;
    # P(i) alone is only the map below 256 elements, so the packer's expression is used)
    idx = np.frombuffer(rr_pack.f16_rr(np.arange(n, dtype=np.uint16)), np.uint16).astype(np.int64)
    out = np.empty(n, np.uint16)
    out[idx] = raw
    return out


def unit_weights(blob, items, k, C, g, Cdown, Cup, mode, dither_off=None):
    """Every item of unit k as the packer's W dict (codes / uint16 bits), decoded from the blob."""
    m = items
    W = dict(k=k, C=C, g=g)
    W["groups"] = [frag_codes(blob, m[0] + i * 64 * C, 64, C) for i in range(g)]
    W["gram"] = [np.frombuffer(bytes(blob[m[1] + j * 8192:m[1] + (j + 1) * 8192]), np.uint16).copy()
                 for j in range(g)]
    W["m48"] = frag_codes(blob, m[2], C, 32 * g)
    W["v128"] = f16_bits(blob, m[3], C)
    W["m68"] = frag_codes(blob, m[4], 4 * C, C)
    W["v140"] = f16_bits(blob, m[5], 4 * C)
    W["m88"] = frag_codes(blob, m[6], C, 4 * C)
    W["v158"] = f16_bits(blob, m[7], C)
    if m[8] != -1:
        W["mA8"] = frag_codes(blob, m[8], Cdown, 4 * C)           # stored (perm_cols) order
        W["v170"] = f16_bits(blob, m[9], Cdown)
    if m[10] != -1:
        W["mC8"] = frag_codes(blob, m[10], 64, 128)                # level-0 -> level-1 merge (unit 1)
        W["v188"] = f16_bits(blob, m[11], 64)
    if m[12] != -1:
        W["mE8"] = frag_codes(blob, m[12], 4 * C, Cup)             # stored (perm_rows) order
        W["v1a0"] = f16_bits(blob, m[13], 4 * C)                   # stored (perm_vec) order
    if m[14] != -1:
        nh = 3 if k < 6 else 4
        W["m108"] = frag_codes(blob, m[14], nh, C, permute=False)
        W["v1b8"] = f16_bits(blob, m[15], 16)[:nh]
    if k == 0:
        W["m1d8"] = frag_codes(blob, m[16], 64, 64)
        W["v218"] = f16_bits(blob, m[17], 64)
        W["m1f8"] = frag_codes(blob, m[18], 32, 64)
        W["v230"] = f16_bits(blob, m[19], 32)
        W["dither"] = bytes(blob[dither_off:dither_off + 4096])
    return W


def repack(W):
    """The packer's own encoder (rr_pack.pack) on the decoded W: (blob, meta)."""
    W = dict(W)
    dither = W.pop("dither", b"")
    return rr_pack.pack(W, dither)


# ------------------------------------------------------------------ the first-hypothesis block
def nr_swin_exp(x):
    """BORROWED from tools/sr/sr_stage_model.py: the ISA-derived bit-built exp of SR's k_sr_stage."""
    hx = np.asarray(x).astype(np.float16)
    y = np.clip((hx * np.float16(0.017181396484375)).astype(np.float16),
                np.float16(-0.55615234375), np.float16(0.55615234375))
    yf = y.astype(np.float32)
    base = (-yf * yf + np.float32(0.92724609375)).astype(np.float16)
    p = (yf * base.astype(np.float32) + np.float32(1.375)).astype(np.float16)
    bits = ((p.view(np.uint16).astype(np.uint32) << 5) + 0x8000) & 0xffff
    return bits.astype(np.uint16).view(np.float16).astype(np.float32)


def act_fn(z, kind):
    z = np.asarray(z, np.float32)
    if kind == "gelu":                                    # tanh form (as SR's act_fn)
        return np.float32(0.5) * z * (np.float32(1.0) + np.tanh(np.float32(0.7978845608) * (z + np.float32(0.044715) * z ** 3)))
    if kind == "silu":
        return z / (np.float32(1.0) + np.exp(-z))
    if kind == "erfgelu":
        return (0.5 * z * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))).astype(np.float32)
    if kind == "nr":                                      # BORROWED: SR's MpCubicSilu (refuted for SR)
        t = np.clip(z, -4.0, 4.0)
        return z * (t * (np.float32(-0.055908203125) * np.abs(t) + np.float32(0.447265625)) + np.float32(0.89453125))
    if kind == "nhgelu":                                  # NeuralHub's f16 GELU spline (SR Review R4)
        zh = z.astype(np.float16)
        t = np.clip(zh, np.float16(-2.0), np.float16(2.0))
        u = (np.abs(t) * np.float16(-0.0810546875)).astype(np.float16) + np.float16(0.412109375)
        gt = ((t * u).astype(np.float16) + np.float16(0.5)).astype(np.float16)
        return (zh * gt).astype(np.float16).astype(np.float32)
    if kind == "none":
        return z
    raise SystemExit(f"unknown act {kind}")


def reflect(p, n):
    p = np.where(p < 0, -p, p)
    p = np.where(p >= n, 2 * (n - 1) - p, p)
    return np.clip(p, 0, n - 1)


def merge2x2(x):
    """(2H, 2W, C) -> (H, W, 4C): the sub-pixel blocks in the order q = dx + 2 dy, index q*C + c."""
    parts = [x[dy::2, dx::2, :] for (dx, dy) in ((0, 0), (1, 0), (0, 1), (1, 1))]
    return np.concatenate(parts, -1)


def chain_mm(a, Wm, b):
    """Review R9b: bias first, then K=32 WMMA chains, each f32 partial added through f16."""
    z = np.broadcast_to(f16a(b), a.shape[:-1] + (Wm.shape[0],)).astype(np.float32)
    for k0 in range(0, a.shape[-1], 32):
        z = f16a(z + a[..., k0:k0 + 32] @ Wm[:, k0:k0 + 32].T)
    return z


def project_down(img, Wt, cfg):
    """Mode 0 / 1 down-projection (verified against the oracle, see the self-check)."""
    cat = merge2x2(q8(img))
    if cfg["chain_io"]:
        y = chain_mm(cat, Wt["mA8"], Wt["v170"] if cfg["down_bias"] else np.zeros(Wt["mA8"].shape[0], np.float32))
    else:
        y = cat @ Wt["mA8"].T + (Wt["v170"] if cfg["down_bias"] else 0.0)
    y = np.asarray(y, np.float32)
    lim = float(cfg["proj_clamp"])
    y = np.clip(y, -lim, lim) if lim > 0 else np.clip(y, -448, 448)
    return LUT[to_e4m3(y.astype(np.float32))].astype(np.float32)


def window_tokens(H, W, sx, sy, C_tile=8):
    """Window origins and token coordinates exactly as the k_level grid covers them (SR's partition)."""
    gx = (W + sx + C_tile - 1) // C_tile
    gy = (H + sy + C_tile - 1) // C_tile
    ox = np.arange(gx * C_tile // 4) * 4 - sx
    oy = np.arange(gy * C_tile // 4) * 4 - sy
    tx = ox[None, :, None, None] + np.arange(4)[None, None, None, :]
    ty = oy[:, None, None, None] + np.arange(4)[None, None, :, None]
    TX = np.broadcast_to(tx, (len(oy), len(ox), 4, 4)).reshape(-1, 16)
    TY = np.broadcast_to(ty, (len(oy), len(ox), 4, 4)).reshape(-1, 16)
    return TX, TY


def attention(s, Wt, cfg, C, g):
    """s: (N, 16, C) normalised tokens. Returns the summed per-head projected contexts (N, 16, C)."""
    attn = np.zeros(s.shape, np.float32)
    for h in range(g):
        Wh = LUT[Wt["groups"][h]]                                # (64, C): rows 0..31 and 32..63
        z = s @ Wh.T                                             # (N, 16, 64)
        if cfg["qk8"]:
            z = q8(z)
        zA, zB = z[..., :32], z[..., 32:]
        Qs, Ks = (zA, zB) if cfg["qk_order"] == "qk" else (zB, zA)
        V = Ks if cfg["v"] == "k" else Qs
        if cfg["v8"]:
            V = q8(V)
        if cfg["qk"] == "sep":
            logits = np.einsum("nqc,nkc->nqk", Qs, Ks)
        else:                                                    # gram: z^T G z, G symmetric 64 x 64
            G = np.frombuffer(Wt["gram"][h].tobytes(), np.float16).astype(np.float32).reshape(64, 64)
            zg = z @ G
            logits = np.einsum("nqc,nkc->nqk", zg, z)
        pe = nr_swin_exp(logits) if cfg["exp"] == "nr" else np.exp(logits).astype(np.float32)
        tot = pe.sum(-1, keepdims=True)
        prob = pe / np.maximum(tot, np.float32(1e-6))
        if cfg["prob8"]:
            prob = q8(prob)
        ctx = np.einsum("nqk,nkc->nqc", prob, V)
        if cfg["ctx8"]:
            ctx = q8(ctx)
        Wo = LUT[Wt["m48"][:, 32 * h:32 * h + 32]]                # (C, 32)
        attn += ctx @ Wo.T
    return attn


def rmsnorm(x, cfg):
    x = np.asarray(x, np.float32)
    ss = (x * x).sum(-1, keepdims=True)
    if cfg["norm"] == "rms":
        s = x / np.sqrt(ss / x.shape[-1] + np.float32(2.0 ** -14))
    else:
        s = x / np.sqrt(ss + np.float32(2.0 ** -14))
    s = s.astype(np.float32)
    return q8(s) if cfg["snorm8"] else s


def block(X, Wt, cfg, H, W, sx, sy, C, g, attn_only=False, postattn=None, fp8_out=True):
    """Windowed attention + MLP on the (H, W, C) tensor X. Returns (block output or None, post-attn).
    fp8_out=True keeps the historic f32 sum clipped to +-448 (what the fp8 store path sees);
    fp8_out=False returns the kernel's raw f16-rounded sum (unit 15's heads read it, no fp8 store)."""
    TX, TY = window_tokens(H, W, sx, sy)
    PX, PY = reflect(TX, W), reflect(TY, H)
    s_act = X[PY, PX].astype(np.float32)                          # (N, 16, C) residual tokens
    s_norm = rmsnorm(s_act, cfg)
    attn = attention(s_norm, Wt, cfg, C, g)
    X1 = np.asarray(s_act + attn, np.float32)
    if cfg["v128"] == "after_attn":
        X1 = X1 + Wt["v128_f"]                                    # v128 (C): post-attention bias
    X1 = X1.astype(np.float16).astype(np.float32)
    if postattn is not None:                                       # injection: replace the post-attn residual
        X1 = postattn[PY, PX].astype(np.float32)
    if attn_only:
        return scatter(X1, TX, TY, H, W, C), None
    s2 = rmsnorm(X1, cfg)
    h1 = np.asarray(s2 @ Wt["W1"].T + Wt["b1"], np.float32)
    h1 = q8(act_fn(h1, cfg["act"]))
    m = h1 @ Wt["W2"].T + Wt["b2"]
    out = f16a(X1 + m) if not fp8_out else np.clip(X1 + m, -448, 448).astype(np.float32)
    return scatter(out, TX, TY, H, W, C), None


def window8_tokens(H, W, sx, sy):
    """k_level's partition: one 8 x 8 window of 64 tokens per workgroup (grid (W+sx+7)/8), token
    = dy*8 + dx. The 64 x 64 relpos table per head ("gram") matches 64-token windows."""
    gx, gy = (W + sx + 7) // 8, (H + sy + 7) // 8
    ox, oy = np.arange(gx) * 8 - sx, np.arange(gy) * 8 - sy
    tx = ox[None, :, None, None] + np.arange(8)[None, None, None, :]
    ty = oy[:, None, None, None] + np.arange(8)[None, None, :, None]
    TX = np.broadcast_to(tx, (len(oy), len(ox), 8, 8)).reshape(-1, 64)
    TY = np.broadcast_to(ty, (len(oy), len(ox), 8, 8)).reshape(-1, 64)
    return TX, TY


def f16a(x):
    return np.asarray(x, np.float32).astype(np.float16).astype(np.float32)


def pair_sum16(e):
    while e.shape[-1] > 1:
        e = f16a(e[..., 0::2] + e[..., 1::2])
    return e[..., 0]


def block_v2(X, Wt, cfg, H, W, sx, sy, C, g, scal, attn_only=False, postattn=None, fp8_out=True, up_entry=None):
    """Second hypothesis (Review R7): SR's decoded block adapted to the RR packer + ISA facts.
    8x8 windows, relpos bias = the packer's symmetrised 64x64 'gram' (x 2*sqrt(2), as SR's v258),
    V = K (f16), no RMSNorm gain (RR packs none), LayerScale: x += s1d2 * f16(attn + v128),
    x += s1d4 * f16(mlp + v158) (Args +0xac / +0xb0, read as f16 bits; ISA v_pk_mul_f16 %44.1/.2),
    MLP activation = NeuralHub's f16 GELU spline (same 0xad30/0x3698 constants in the RR ISA)."""
    TX, TY = window8_tokens(H, W, sx, sy)
    PX, PY = reflect(TX, W), reflect(TY, H)
    s_act = f16a(X[PY, PX])
    if up_entry is not None:                 # Review R9b (as SR's R9): mode-2 border tokens reflect in low-res space
        U, skip, s6, (h2, w2) = up_entry
        LX, LY = reflect(TX >> 1, w2), reflect(TY >> 1, h2)
        UX, UY = 2 * LX + (TX & 1), 2 * LY + (TY & 1)
        up_t = U[np.clip(UY, 0, H - 1), np.clip(UX, 0, W - 1)]
        s_act = f16a(up_t + f16a(np.float32(s6) * skip[PY, PX]))
    def norm(x):
        xf = x.astype(np.float32); ss = (xf * xf).sum(-1, keepdims=True)
        if cfg["norm"] == "none":
            y = f16a(xf)
        else:
            d = ss / xf.shape[-1] if cfg["norm"] == "rms" else ss
            y = f16a(xf / np.sqrt(d + np.float32(2.0 ** -14)))
        return q8(y) if cfg["snorm8"] else y
    ls1, ls2 = scal[1], scal[2]
    def mm(a, Wm):
        """a @ Wm.T; acc16: K=32 chunks (2 fp8 WMMAs) in f32, each chunk added through f16
        (v_fma_mix*_f16 chains, RR ISA unit 1 at 0x10b0 / 0x11f8)."""
        if not cfg["acc16"]:
            return a @ Wm.T
        acc = None
        for k0 in range(0, a.shape[-1], 32):
            part = a[..., k0:k0 + 32] @ Wm[:, k0:k0 + 32].T
            acc = f16a(part) if acc is None else f16a(acc + part)
        return acc
    sn = norm(s_act)
    attn = np.zeros(s_act.shape, np.float32)
    for h in range(g):
        Wh = LUT[Wt["groups"][h]]
        Q = q8(np.clip(mm(sn, Wh[:32]), -448, 448)); Kf = f16a(np.clip(mm(sn, Wh[32:]), -448, 448)); K8 = q8(Kf)
        if cfg["qq"]:
            # Review R7b (patchprobe on rrref, unit 2): BOTH logit WMMA operands hold rows 0..31's
            # projection - the logits are Q.Q^T + relpos (symmetric; the packer symmetrises relpos
            # to match) and rows 32..63 are the VALUE projection.
            K8 = Q
        rel = np.frombuffer(Wt["gram"][h].tobytes(), np.float16).astype(np.float32).reshape(64, 64)
        logits = np.float32(cfg["lscale"]) * np.einsum("nqc,nkc->nqk", Q, K8) + (rel[None] if cfg["relpos"] else 0.0)
        pe = nr_swin_exp(logits)
        if cfg["smax"] == "isa":             # Review R9b: the kernel's 64-key f16 tree (ISA 0x153c..0x164c)
            e = f16a(pe)
            def key(kb, hh, ip, par): return 16 * kb + 8 * hh + 2 * ip + par
            def V(par):
                U = []
                for ip in range(4):
                    T = [f16a(f16a(e[..., key(0, hh, ip, par)] + e[..., key(1, hh, ip, par)])
                              + f16a(e[..., key(2, hh, ip, par)] + e[..., key(3, hh, ip, par)])) for hh in (0, 1)]
                    U.append(f16a(T[0] + T[1]))
                return f16a(f16a(U[0] + U[1]) + f16a(U[2] + U[3]))
            tot = f16a(V(0) + V(1))
            inv = f16a(np.float32(1.0) / tot)
            prob = f16a(e * inv[..., None])
        elif cfg["den16"]:
            inv = f16a(1.0 / np.maximum(pair_sum16(f16a(pe)), np.float32(1e-4)))
        else:
            inv = np.float32(1.0) / np.maximum(pe.sum(-1), np.float32(1e-4))
        if cfg["smax"] != "isa":
            prob = f16a(f16a(pe) * inv[..., None]) if cfg["den16"] else pe * inv[..., None]
        if cfg["pv8"]:                       # RR's P @ V is an fp8 WMMA (the RR kernels have no f16 WMMA)
            prob, Vv = q8(prob), K8 if not cfg["qq"] else q8(Kf)
        else:
            Vv = Kf
        if cfg["ctx16"] == 2:                # Review R9b: P @ V as two K=32 key chains through f16 (ISA 0x18c4/0x1934)
            ctx = f16a(f16a(np.einsum("nqk,nkc->nqc", prob[..., :32], Vv[:, :32]))
                       + np.einsum("nqk,nkc->nqc", prob[..., 32:], Vv[:, 32:]))
            ctx = q8(ctx)
        else:
            ctx = np.einsum("nqk,nkc->nqc", prob, Vv)
            ctx = q8(f16a(ctx)) if cfg["ctx16"] else q8(ctx)
        if cfg["chain"]:                     # Review R9b: the head loop's f16 accumulator, seeded with v128
            if h == 0:
                attn = np.broadcast_to(f16a(Wt["v128_f"]), s_act.shape).astype(np.float32)
            attn = f16a(attn + ctx @ LUT[Wt["m48"][:, 32 * h:32 * h + 32]].T)
        else:
            attn += mm(ctx, LUT[Wt["m48"][:, 32 * h:32 * h + 32]])
    a = attn if cfg["chain"] else f16a(attn + Wt["v128_f"])
    X1 = f16a(s_act + (f16a(np.float32(ls1) * a) if cfg["ls"] else a))
    if postattn is not None:
        X1 = f16a(postattn[PY, PX])
    if attn_only:
        return scatter(X1, TX, TY, H, W, C), None
    s2 = norm(X1)
    if cfg["chain"]:                         # Review R9b: per 32-hidden block, the MLP loop's f16 accumulator
        W1, b1, W2 = Wt["W1"], Wt["b1"], Wt["W2"]
        mo = np.broadcast_to(f16a(Wt["b2"]), X1.shape).astype(np.float32)
        for j0 in range(0, W1.shape[0], 32):
            z = np.broadcast_to(f16a(b1[j0:j0 + 32]), s2.shape[:-1] + (32,)).astype(np.float32)
            for k0 in range(0, s2.shape[-1], 32):
                z = f16a(z + s2[..., k0:k0 + 32] @ W1[j0:j0 + 32, k0:k0 + 32].T)
            hb = q8(act_fn(z, cfg["act"]))
            mo = f16a(mo + hb @ W2[:, j0:j0 + 32].T)
    else:
        h1 = q8(act_fn(np.asarray(mm(s2, Wt["W1"]) + Wt["b1"], np.float32), cfg["act"]))
        mo = f16a(mm(h1, Wt["W2"]) + Wt["b2"])
    add = X1 + (f16a(np.float32(ls2) * mo) if cfg["ls"] else mo)
    # fp8_out=True: the historic clip the fp8 store path sees (identical bytes: to_e4m3 saturates
    # at 448 anyway). fp8_out=False: the kernel's v_pk_add_f16 result, which unit 15's heads read.
    out = np.clip(add, -448, 448).astype(np.float32) if fp8_out else f16a(add)
    return scatter(out, TX, TY, H, W, C), None


def scatter(out, TX, TY, H, W, C):
    img = np.zeros((H, W, C), np.float32)
    ok = (TX >= 0) & (TX < W) & (TY >= 0) & (TY < H)
    img[TY[ok], TX[ok]] = out[ok]
    return img


def upsample(low, Wt, cfg, H, W, C):
    """Mode 2: u = mE8 @ low + v1a0 (rows [q][c]), pixel-shuffled with q = dx + 2 dy."""
    u = chain_mm(low, Wt["mE8f"], Wt["v1a0_f"]) if cfg["chain_io"] else \
        np.asarray(low @ Wt["mE8f"].T + Wt["v1a0_f"], np.float32)
    u = u.reshape(u.shape[0], u.shape[1], 4, C)
    order = ((0, 0), (1, 0), (0, 1), (1, 1)) if cfg["up_pix"] == "xy" else ((0, 0), (0, 1), (1, 0), (1, 1))
    out = np.zeros((H, W, C), np.float32)
    for q, (dx, dy) in enumerate(order):
        out[dy::2, dx::2, :] = u[:, :, q, :]
    if cfg["up8"]:                           # Review R9b: the up-projection is staged in LDS as fp8 (as SR's)
        return q8(np.clip(f16a(out), -448, 448))
    return out.astype(np.float16).astype(np.float32)


def head(out, Wt, cfg, nh):
    """O98 = m108 @ fp8(out) + v1b8 in lanes 0..2. Lane 3 is 0: it is zero in every 8 B/px output
    the oracle writes (encoders, decoder _1 units, unit 15), so the model writes 0 there."""
    src = q8(out) if cfg["head_in"] == "q8" else out.astype(np.float32)
    y = src @ Wt["m108f"][:3].T + Wt["v1b8_f"][:3]
    o = np.zeros(out.shape[:2] + (4,), np.float32)
    o[..., :3] = y
    return o


def head_v2(out, Wt, nh, s1d0, x88):
    """Encoder head (Review R7c, k_level<96,4,0,128,0,1> ISA 0x3908..0x3c70):
    h_c = f16-chain(bias_c + K=32 chunks of m108[c] . fp8(out)),  g_c = f16(sigmoid(h_c)) (f32 exp/rcp),
    o98_c = min(f16(f16(s1d0 * g_c) * x88_c), 65504) for c < 3, lane 3 = 0 (0 * x88_3)."""
    src = q8(out)
    Wm = Wt["m108f"][:nh]
    C = src.shape[-1]
    acc = np.broadcast_to(f16a(Wt["v1b8_f"][:nh]), src.shape[:2] + (nh,)).astype(np.float32)
    for k0 in range(0, C, 32):
        acc = f16a(acc + src[..., k0:k0 + 32] @ Wm[:, k0:k0 + 32].T)
    g = f16a(np.float32(1.0) / (np.float32(1.0) + np.exp(-acc.astype(np.float32))))
    o = np.zeros(out.shape[:2] + (4,), np.float32)
    o[..., :nh] = np.minimum(f16a(f16a(np.float32(s1d0) * g) * x88[..., :nh]), np.float32(65504.0))
    return o


def up2_bilinear(c):
    """2x bilinear upsample with the decoder head's f16 weights: fine p uses coarse base = (p-1)>>1
    and base+1, weight of base+1 = 0.75 (p even) / 0.25 (p odd); at the borders the out-of-range
    tap's weight is 0 and the base is clamped. Order (ISA 0x5264..0x533c):
    f16(f16(f16(f16((1-b) f16((1-a) t1)) + f16((1-b) f16(a t2))) + f16(b f16((1-a) t3))) + f16(b f16(a t4)))"""
    h, w = c.shape[:2]
    def axis(n_fine, n_coarse):
        pidx = np.arange(n_fine)
        base = (pidx - 1) >> 1
        wt = np.where(pidx % 2 == 0, np.float32(0.75), np.float32(0.25))
        lo_bad = base < 0; hi_bad = base + 1 >= n_coarse
        wt = np.where(lo_bad | hi_bad, np.float32(0.0), wt)
        base = np.clip(base, 0, n_coarse - 1); nxt = np.clip(base + 1, 0, n_coarse - 1)
        return base, nxt, wt
    by, ny, b = axis(2 * h, h); bx, nx, a = axis(2 * w, w)
    a = a[None, :, None]; b = b[:, None, None]
    t1 = c[by][:, bx]; t2 = c[by][:, nx]; t3 = c[ny][:, bx]; t4 = c[ny][:, nx]
    one = np.float32(1.0)
    r = f16a(f16a((one - b) * f16a((one - a) * t1)) + f16a((one - b) * f16a(a * t2)))
    r = f16a(r + f16a(b * f16a((one - a) * t3)))
    return f16a(r + f16a(b * f16a(a * t4)))


def head_dec(out, Wt, s1d0, x88, x90):
    """Decoder (mode 3) head, 4 rows (Review R7c, k_level<160,8,3,0,0,1> ISA 0x515c..0x5394):
    h_r = f16-chain(bias_r + K=32 chunks of m108[r] . fp8(out)), s_r = f16(sigmoid(h_r)),
    o98_c = f16( f16( f16( f16(s1d0 * s_c) * up_c ) * s_3 ) + f16( f16(1 - s_3) * x88_c ) ), c < 3,
    up = up2_bilinear(x90) (the coarser decoder's o98), gate s_3; lane 3 = 0."""
    src = q8(out); C = src.shape[-1]; Wm = Wt["m108f"][:4]
    acc = np.broadcast_to(f16a(Wt["v1b8_f"][:4]), src.shape[:2] + (4,)).astype(np.float32)
    for k0 in range(0, C, 32):
        acc = f16a(acc + src[..., k0:k0 + 32] @ Wm[:, k0:k0 + 32].T)
    sg = f16a(np.float32(1.0) / (np.float32(1.0) + np.exp(-acc.astype(np.float32))))
    up = up2_bilinear(x90)
    gate = sg[..., 3:4]; inv = f16a(np.float32(1.0) - gate)
    o = np.zeros(out.shape[:2] + (4,), np.float32)
    v = f16a(f16a(f16a(f16a(np.float32(s1d0) * sg[..., :3]) * up[..., :3]) * gate) + f16a(inv * x88[..., :3]))
    o[..., :3] = np.minimum(v, np.float32(65504.0))
    return o


def head_down2(o98):
    """oa0 (Review R7c, ISA 0x3dcc..0x3ee8): the 2x2 average of the clamped o98, each tap scaled by
    0.25 in f16 first, summed as (p00 + p10) + (p01 + p11) in f16, clamped to 65504."""
    q = lambda v: f16a(np.float32(0.25) * v)
    a, b = q(o98[0::2, 0::2]), q(o98[0::2, 1::2])
    c, d = q(o98[1::2, 0::2]), q(o98[1::2, 1::2])
    return np.minimum(f16a(f16a(a + b) + f16a(c + d)), np.float32(65504.0))


# ------------------------------------------------------------------ unit 15 (mode 4) dec0 outputs
# Read from the k_level<32,2,4,0,0,n> ISA (build/work/review/rr/u15.s): stores 0x6c78 (E0bd4),
# 0x7064 (E0c16), 0x7070 (E0c1a), 0x775c (E0c02), 0x790c (E0c1e); pass B (token y+4) mirrors
# 0x8078/0x8468/0x8474/0x8b3c/0x8cd4.  Every formula is spelled out in the module docstring.
def _f32_of_bits(b):
    return np.frombuffer(np.uint32([b & 0xFFFFFFFF]).tobytes(), np.float32)[0]


def _f16_of_bits(b):
    return np.frombuffer(np.uint16([b & 0xFFFF]).tobytes(), np.float16)[0]


F32_LOG2E = _f32_of_bits(0x3FB8AA3B)        # v_exp scaling constant (f32 truncation of log2e)
F32_NEG_L2E = _f32_of_bits(0xBFB8AA3B)      # -log2e in f32: sigmoid's exp argument multiplier
F32_L2E_GAP = _f32_of_bits(0x32A5705F)      # log2e - F32_LOG2E: the tanh exp-branch tail constant
F16_LOG2E = _f16_of_bits(0x3DC5)            # f16 multiplier of (p - max) in the softmax exp
F16_CLAMP20 = np.float16(20.0)              # aniso pre-sigmoid clamp (+-0x4d00 / 0xcd00)
TANH_C = tuple(_f32_of_bits(b) for b in (0xBBBAC73D, 0x3CA908C9, 0xBD5C1C4E, 0x3E088382, 0xBEAAAA99))
TANH_POLY_MAX = _f32_of_bits(0x3F200000)    # 0.625: |x| <= -> polynomial, else exp branch
F16_EXP_FIX_IN = 0x11C5                     # f16(log2e * d) == this -> the exp is forced to 0x3c01
F16_EXP_FIX_OUT = _f16_of_bits(0x3C01)


def _fma32(a, b, c):
    """Single-rounding f32 fma(a*b + c): exact in f64, one rounding to f32 (the kernel's v_fma*)."""
    return (np.float64(a) * np.float64(b) + np.float64(c)).astype(np.float32)


def _exp2_32(y):
    """v_exp_f32 = 2**y: correctly rounded through an f64 intermediate (negligible residual)."""
    return np.exp2(np.asarray(y, np.float64)).astype(np.float32)


def u15_sigmoid(h):
    """The kernel's f16 sigmoid of an f32 f16-valued h: f16(v_rcp_f32(1 + 2**(F32_NEG_L2E * h)))."""
    e = _exp2_32((np.float32(F32_NEG_L2E) * np.asarray(h, np.float32)).astype(np.float32))
    s = (np.float32(1.0) + e).astype(np.float32)
    r = (np.float64(1.0) / s.astype(np.float64)).astype(np.float32)      # correctly-rounded rcp
    return r.astype(np.float16).astype(np.float32)


def u15_tanh(x):
    """NeuralHub's piecewise f32 tanh (sign preserved):
    |x| <= 0.625: sign(x) * fma(y, fma(y, fma(y, fma(y, fma(c0*y + c1)... + c4))) * |x|, |x|)
        with y = x*x and every step a single-rounding fma (the v_fmaak chain);
    else:        sign(x) * fma(rcp, -2, 1) with rcp = 1/(1 + 2**(2|x|*log2e + the
        (log2e - f32 log2e) tail), both roundings fused as in the ISA."""
    x = np.asarray(x, np.float32)
    ax = np.abs(x)
    y = (ax * ax).astype(np.float32)
    c0, c1, c2, c3, c4 = TANH_C
    p = _fma32(c0, y, c1)
    p = _fma32(y, p, c2)
    p = _fma32(y, p, c3)
    p = _fma32(y, p, c4)
    poly = _fma32(y, (ax * p).astype(np.float32), ax)

    a = (ax + ax).astype(np.float32)
    t = (np.float32(F32_LOG2E) * a).astype(np.float32)
    k = np.rint(t)
    eps = _fma32(np.float32(F32_LOG2E), a, -t)
    tail = _fma32(a, np.float32(F32_L2E_GAP), eps)
    frac = ((t - k) + tail).astype(np.float32)
    with np.errstate(over="ignore"):
        e2 = np.ldexp(_exp2_32(frac), k.astype(np.int32))
    s = (np.float32(1.0) + e2).astype(np.float32)
    rcp = (np.float64(1.0) / s.astype(np.float64)).astype(np.float32)
    expb = _fma32(rcp, np.float32(-2.0), np.float32(1.0))           # v_fma rcp, -2, 1
    mag = np.where(ax <= TANH_POLY_MAX, poly, expb)
    return np.copysign(mag, x).astype(np.float32)


def u15_clamp65504(v):
    """The kernel's store clamp: NaN/overflowing -> 65504, else min(v, 65504) in f16 (values are
    f16-representable; -inf passes through, +inf/NaN become 65504, exactly as v_cndmask/v_min_num)."""
    v = np.asarray(v, np.float32)
    m = np.float32(65504.0)
    return np.where(np.isnan(v) | ~(m > v), m, np.minimum(v, m)).astype(np.float32)


def u15_up(ch, x, y):
    """The dec0 bilinear 2x read of x90 at level-1 pixel (x, y): identical to up2_bilinear (Review
    R7c) - base = (p-1)>>1, weight of base+1 = 0.75 (p even) / 0.25 (p odd), 0 at borders with the
    base clamped (ISA constants 0x3a00 = 0.75, 0x3400 = 0.25); taps (by,bx),(by,nx),(ny,bx),(ny,nx),
    f16 order r = f16(f16((1-b)*t1 + (1-b)*a*t2) + b*(1-a)*t3 + b*a*t4) as in head_dec."""
    h2, w2 = ch.shape[:2]

    def axis(p, n2):
        base = (p - 1) >> 1
        w = np.where(p % 2 == 0, np.float32(0.75), np.float32(0.25))
        bad = (base < 0) | (base + 1 >= n2)
        w = np.where(bad, np.float32(0.0), w)
        b0 = np.clip(base, 0, n2 - 1)
        return b0.astype(np.int64), np.clip(b0 + 1, 0, n2 - 1).astype(np.int64), w.astype(np.float32)

    bx, nx, a = axis(x, w2)
    by, ny, b = axis(y, h2)
    t1 = ch[by, bx]
    t2 = ch[by, nx]
    t3 = ch[ny, bx]
    t4 = ch[ny, nx]
    one = np.float32(1.0)
    r = f16a(f16a((one - b) * f16a((one - a) * t1)) + f16a((one - b) * f16a(a * t2)))
    r = f16a(r + f16a(b * f16a((one - a) * t3)))
    return f16a(r + f16a(b * f16a(a * t4)))


def _u8byte(v255, valid):
    """rndne(255 * v) to a byte, 0 where !valid (the kernel's cmp_o guard on the f16 source)."""
    b = np.rint(np.asarray(v255, np.float64))
    b = np.where(valid, b, 0.0)
    return np.clip(b, -128, 255).astype(np.int64) & 0xFF


def _softmax4(p):
    """The E0c1a factors: p (H,W,4) f32 of f16 -> (H,W,4) uint8, all f16 roundings as in the ISA."""
    m = np.maximum(np.maximum(p[..., 0], p[..., 1]), np.maximum(p[..., 2], p[..., 3]))
    d = f16a(p - m[..., None])
    prod = f16a(np.float32(F16_LOG2E) * d)
    pb = prod.astype(np.float16).view(np.uint16)
    with np.errstate(over="ignore"):
        e = _exp2_32(prod).astype(np.float16)
    e = np.where(pb == F16_EXP_FIX_IN, np.float16(F16_EXP_FIX_OUT), e)
    ef = e.astype(np.float32)
    tot = f16a(f16a(ef[..., 0] + ef[..., 2]) + f16a(ef[..., 1] + ef[..., 3]))
    rcp = (np.float64(1.0) / tot.astype(np.float64)).astype(np.float16).astype(np.float32)
    ratio = f16a(ef * rcp[..., None])
    valid = ~np.isnan(ratio.astype(np.float16))
    return _u8byte(np.float32(255.0) * ratio, valid).astype(np.uint8)


# The logical block-output channels (0..31 of img) each head slot reads, as (dy<2, dy>=2) pairs.
# dy<2 = (y + sy) mod 4 < 2 selects tid < 16's halves; the kernel swaps each pair's halves for the
# other half-window (v_cndmask vcc = 16 > tid), which the fitted values must (and do) reproduce.
# Fitted channel-exact against the oracle on frame 1 (dy<2), cross-validated on dy>=2 and on frame 0
# (see the report); the channel identity is a property of the blob's weights, i.e. frame-invariant.
U15_CHANS = {
    # E0bd4 o98 blend gates: sigma(h_0..h_3) of block channels 0..3
    "bd4_0": (0, 0),
    "bd4_1": (1, 1),
    "bd4_2": (2, 2),
    "bd4_3": (3, 3),
    # E0c16 gates bytes 0..2: block channels 4, 5, 6 (byte 3 = 0)
    "g0": (4, 4),
    "g1": (5, 5),
    "g2": (6, 6),
    # E0c1a softmax bytes 0..3: block channels 10..13
    "f0": (10, 10),
    "f1": (11, 11),
    "f2": (12, 12),
    "f3": (13, 13),
    # E0c02 i8 tanh bytes 0..3: block channels 14..17
    "t0": (14, 14),
    "t1": (15, 15),
    "t2": (16, 16),
    "t3": (17, 17),
    # E0c1e aniso: sigma(clamp(p14)), sigma(clamp(p15)) = ch 7, 8; tanh(f0*p16) = ch 9
    "aA": (7, 7),
    "aB": (8, 8),
    "p16": (9, 9),
}


def unit15_outputs(img, x88, x90, chans, Hr, sy, s1d0, f0, s0):
    """img (H, W, 32) f16-valued block output; x88 (H, W, 4) f16; x90 (H/2, W/2, 4) f16;
    chans: slot -> ((ch_dy0, ch_dy1), ...) with the two dy-half channels (ISA pair-swap);
    Hr: render height (Args +0xe4); returns (o98 f16, oc0 u8, oc8 u8, od0 u8, od8 f16) images."""
    H, W, C = img.shape
    dy0 = ((np.arange(H)[:, None] + sy) % 4 < 2)                   # tid < 16 (window row mod 4 < 2)

    def pick(slot):
        lt, ge = chans[slot][0], chans[slot][1]
        sel = np.where(dy0, np.int64(lt), np.int64(ge))[..., None]
        return np.take_along_axis(img, sel, axis=-1)[..., 0]

    # ---- E0bd4: o98, the dec0 decoder head (store 0x6c78 / pass B 0x8078), full level-0 grid
    h = [pick(f"bd4_{i}") for i in range(4)]
    sg = [u15_sigmoid(v) for v in h]
    gate = sg[3]
    inv = f16a(np.float32(1.0) - gate)
    yy, xx = np.mgrid[0:H, 0:W]
    o98 = np.zeros((H, W, 4), np.float32)
    for c in range(4):
        kk = f16a(np.float32(s1d0) * sg[c]) if c < 3 else np.float32(0.0)
        # lane 3's coefficient is the literal 0 (v_mul_f16 0, gate): no x90 read at all
        prod = np.float32(0.0) if c == 3 else f16a(kk * u15_up(x90[..., c], xx, yy))
        t = f16a(prod * gate)
        o98[..., c] = u15_clamp65504(f16a(t + f16a(inv * x88[..., c])))

    # ---- render-resolution outputs (rows 0..Hr-1): guards y < Hr, x < Wr (all model pixels pass)
    r = slice(0, Hr)
    oc0 = np.zeros((Hr, W, 4), np.uint8)                            # E0c02: 4 x i8 tanh (store 0x775c)
    for c in range(4):
        th = u15_tanh(pick(f"t{c}")[r]).astype(np.float16)
        v = np.clip(th.astype(np.float32), np.float32(-1.0), np.float32(1.0))
        oc0[..., c] = _u8byte(np.float32(127.0) * v, ~np.isnan(th)).astype(np.uint8)

    sg16 = [u15_sigmoid(pick(f"g{i}")[r]) for i in range(3)]        # E0c16: gates (store 0x7064)
    oc8 = np.zeros((Hr, W, 4), np.uint8)
    for c in range(3):
        oc8[..., c] = _u8byte(np.float32(255.0) * sg16[c], ~np.isnan(sg16[c].astype(np.float16)))
    oc8[..., 3] = 0

    od0 = _softmax4(np.stack([pick(f"f{i}")[r] for i in range(4)], -1))   # E0c1a (store 0x7070)

    # ---- E0c1e: aniso (store 0x790c), 4 x f16 with lane 3 = 0
    pA = np.clip(pick("aA")[r], -20.0, 20.0)
    pB = np.clip(pick("aB")[r], -20.0, 20.0)
    pT = pick("p16")[r]
    A = (np.float32(s0) * u15_sigmoid(pA)).astype(np.float32)
    B = (np.float32(s0) * u15_sigmoid(pB)).astype(np.float32)
    l0 = f16a(A * A)
    l1 = f16a(B * B)
    x = (np.float32(f0) * pT).astype(np.float32)
    t = u15_tanh(x)
    ab = (A * B).astype(np.float32)
    tb = t.view(np.uint32)
    vlo = f16a(ab * (tb - np.uint32(3)).view(np.float32))
    vhi = f16a(ab * (tb + np.uint32(3)).view(np.float32))
    l2 = np.where(vlo == vhi, vlo,
                  f16a(ab * np.tanh(x.astype(np.float64)).astype(np.float32))).astype(np.float32)
    od8 = np.zeros((Hr, W, 4), np.float32)
    od8[..., 0] = l0
    od8[..., 1] = l1
    od8[..., 2] = l2
    od8[..., 3] = 0.0
    return o98, oc0, oc8, od0, od8.astype(np.float16).astype(np.float32)


def model_unit(L, blobbytes, cfg, inputs, Wt):
    """Returns (outs: role -> (raw bytes, dtype)), (hyp: notes)."""
    tm = L["tmpl"]
    C, g, mode, Cdown, Cup = tm["C"], tm["g"], tm["mode"], tm["Cdown"], tm["Cup"]
    H, W, sx, sy = L["H"], L["W"], L["sx"], L["sy"]
    k = L["unit"]
    outs, notes = {}, []
    up_entry = None
    if mode == 5:                                                  # enc0: features -> C channels
        F = inputs["in"]                                           # (H, W, 64) fp8 logical
        if cfg["chain_io"]:
            h = q8(act_fn(chain_mm(F, Wt["m1d8f"], Wt["v218_f"]), cfg["enc0_act"]))
            X = chain_mm(h, Wt["m1f8f"], Wt["v230_f"])
        else:
            h = np.asarray(F @ Wt["m1d8f"].T + Wt["v218_f"], np.float32)
            h = q8(act_fn(h, cfg["enc0_act"]))
            X = (h @ Wt["m1f8f"].T + Wt["v230_f"]).astype(np.float16).astype(np.float32)
        notes.append("mode 5 entry (GUESS): m1f8 @ act(m1d8 @ feat + v218) + v230")
    elif mode == 1:                                                # level 0 -> level 1 merge, then the block
        X = chain_mm(merge2x2(inputs["in"]), Wt["mC8f"], Wt["v188_f"]) if cfg["chain_io"] else \
            np.asarray(merge2x2(inputs["in"]) @ Wt["mC8f"].T + Wt["v188_f"], np.float32)
        X = (act_fn(X, cfg["entry_act"])).astype(np.float16).astype(np.float32)
        notes.append("mode 1 entry merge (GUESS act): mC8 @ merge2x2(in) + v188")
    elif mode == 2:                                                # decoder: up from the level below, + skip
        X = upsample(inputs["in"], Wt, cfg, H, W, C)
        s6 = float(np.array([L["scal"][3] & 0xFFFF], np.uint16).view(np.float16)[0])
        if cfg["upref"] == "low" and cfg["skip_mix"] == "s6skip":
            up_entry = (X, inputs["skip"], s6, inputs["in"].shape[:2])
        if cfg["skip_mix"] == "s6skip":          # x = up + s1d6 * skip
            X = f16a(X + f16a(np.float32(s6) * inputs["skip"]))
        elif cfg["skip_mix"] == "s6up":          # x = s1d6 * up + skip
            X = f16a(f16a(np.float32(s6) * X) + inputs["skip"])
        elif cfg["skip_at"] in ("before", "both"):
            X = (X + inputs["skip"]).astype(np.float16).astype(np.float32)
    else:                                                          # 0, 3, 4
        X = inputs["in"]
    postattn = inputs.get("postattn")
    attn_only = bool(inputs.get("attn_only"))
    if cfg["blk"] == "v2":
        scal = [float(np.array([v & 0xFFFF], np.uint16).view(np.float16)[0]) for v in L["scal"]]
        img, _ = block_v2(X, Wt, cfg, H, W, sx, sy, C, g, scal, attn_only=attn_only,
                          postattn=postattn, fp8_out=(mode != 4), up_entry=up_entry)
    else:
        img, _ = block(X, Wt, cfg, H, W, sx, sy, C, g, attn_only=attn_only, postattn=postattn,
                       fp8_out=(mode != 4))
    if mode == 2 and cfg["skip_at"] in ("after", "both"):
        img = np.clip(img + inputs["skip"], -448, 448)
    if attn_only:
        outs["out"] = (encode_grouped(img), "fp8")
        return outs, notes
    if mode in (0, 1, 2, 3, 5):
        outs["out"] = (encode_grouped(img), "fp8")
    if mode in (0, 1) and Cdown > 0:
        down = project_down(img, Wt, cfg)
        outs["down"] = (encode_grouped(down), "fp8")
    if mode in (0, 1, 3, 5):
        nh = 3 if k < 6 else 4
        if "m108f" in Wt and cfg["head"] == "v2" and mode == 3 and "x88" in inputs and "x90" in inputs:
            s1d0 = float(np.array([L["scal"][0] & 0xFFFF], np.uint16).view(np.float16)[0])
            o98 = head_dec(img, Wt, s1d0, inputs["x88"], inputs["x90"])
            outs["o98"] = (o98.reshape(-1, 4).astype(np.float16).tobytes(), "f16")
        elif "m108f" in Wt and cfg["head"] == "v2" and "x88" in inputs and k < 6:
            s1d0 = float(np.array([L["scal"][0] & 0xFFFF], np.uint16).view(np.float16)[0])
            o98 = head_v2(img, Wt, nh, s1d0, inputs["x88"])
            outs["o98"] = (o98.reshape(-1, 4).astype(np.float16).tobytes(), "f16")
            if "oa0" in L["ptr"]:
                oa = head_down2(o98)
                outs["oa0"] = (oa.reshape(-1, 4).astype(np.float16).tobytes(), "f16")
            notes.append("head v2 (decoded for encoder units): s1d0 * sigmoid(m108 . fp8(out) + v1b8) * x88")
        elif "m108f" in Wt:
            o98 = head(img, Wt, cfg, nh)
            outs["o98"] = (o98.reshape(-1, 4).astype(np.float16).tobytes(), "f16")
            notes.append(f"head ({nh} rows) -> O98 (GUESS)")
    if mode == 4:
        # The dec0 unit's five outputs (Review u15, spec in the module docstring). The block
        # output is the f16 pk_add result (fp8_out=False above); the heads read it directly.
        s1d0 = float(np.array([L["scal"][0] & 0xFFFF], np.uint16).view(np.float16)[0])
        s0, f0 = float(L["dec0"][0]), float(L["dec0"][1])            # Args +0xe8 / +0xec
        o98, oc0, oc8, od0, od8 = unit15_outputs(img, inputs["x88"], inputs["x90"], U15_CHANS,
                                                 L["Hr"], L["sy"], s1d0, f0, s0)
        outs["o98"] = (o98.reshape(-1, 4).astype(np.float16).tobytes(), "f16")
        outs["oc0"] = (oc0.tobytes(), "i8")                          # E0c02: 4 x i8 tanh (signed)
        outs["oc8"] = (oc8.tobytes(), "u8")                          # E0c16: 3 gates + 0
        outs["od0"] = (od0.tobytes(), "u8")                          # E0c1a: 4-way softmax
        outs["od8"] = (od8.reshape(-1, 4).astype(np.float16).tobytes(), "f16")   # E0c1e aniso + 0
        notes.append("mode 4 dec0 heads (Review u15): O98 = s1d0*sg*s3 blend with x88 + x90 bilinear; "
                     "OC0/OC8/OD0/OD8 = tanh-i8 / gates / softmax / aniso from block channels "
                     "U15_CHANS")
    if mode == 5:
        notes.append("mode 5 OA0 unmodelled")
    return outs, notes


# ------------------------------------------------------------------ unit weights as float views
def prepare(Wd, C, g, Cdown, Cup, mode, k):
    """Adds the f32 / LUT views the block uses. Codes -> LUT values, bits -> f32."""
    Wt = dict(Wd)
    Wt["groups"] = Wd["groups"]
    Wt["m1d8f"] = LUT[Wd["m1d8"]] if "m1d8" in Wd else None
    Wt["m1f8f"] = LUT[Wd["m1f8"]] if "m1f8" in Wd else None
    Wt["v218_f"] = f32(Wd["v218"]) if "v218" in Wd else None
    Wt["v230_f"] = f32(Wd["v230"]) if "v230" in Wd else None
    Wt["v128_f"] = f32(Wd["v128"])
    Wt["W1"] = LUT[Wd["m68"]]
    Wt["b1"] = f32(Wd["v140"])
    Wt["W2"] = LUT[Wd["m88"]]
    Wt["b2"] = f32(Wd["v158"])
    if "mA8" in Wd:
        Wt["mA8"] = LUT[Wd["mA8"]]
        Wt["v170"] = f32(Wd["v170"])
    if "mC8" in Wd:
        Wt["mC8f"] = LUT[Wd["mC8"]]
        Wt["v188_f"] = f32(Wd["v188"])
    if "mE8" in Wd:
        Wt["mE8f"] = LUT[Wd["mE8"]]
        Wt["v1a0_f"] = f32(Wd["v1a0"])
    if "m108" in Wd:
        Wt["m108f"] = LUT[Wd["m108"]]
        Wt["v1b8_f"] = f32(Wd["v1b8"])
    Wt["gram"] = Wd["gram"]
    Wt["m48"] = Wd["m48"]
    return Wt


def f32(bits):
    return np.frombuffer(np.asarray(bits, np.uint16).tobytes(), np.float16).astype(np.float32)


# ------------------------------------------------------------------ one launch
def load_unit(frame, launch, dump=DUMP):
    L = load_launch(frame, launch, dump)
    tm = L["tmpl"]
    blob_name = L["ptr"]["blob"]
    blob = np.frombuffer(buffer_before(str(L["frame_dir"]), blob_name, launch), np.uint8)
    if blob.size != L["blob_size"]:
        raise SystemExit(f"blob {blob_name} is {blob.size} bytes, Args say {L['blob_size']}")
    pj = json.loads((PACKED / "params.json").read_text())
    dither_off = pj["units"][L["unit"]]["meta"][20] if pj["units"][L["unit"]]["meta"][20] != -1 else None
    if L["unit"] == 0 and dither_off is None:
        raise SystemExit("unit 0 has no dither item")
    Wd = unit_weights(blob, L["items"], L["unit"], tm["C"], tm["g"], tm["Cdown"], tm["Cup"], tm["mode"],
                      dither_off)
    L.update(blob=blob, Wd=Wd, dither_off=dither_off, blob_name=blob_name, pj=pj)
    return L


def read_inputs(L, cfg, extra=None):
    tm = L["tmpl"]
    C, mode, H, W = tm["C"], tm["mode"], L["H"], L["W"]
    fd, n = str(L["frame_dir"]), L["d"]["n"]
    inputs = {}
    if mode == 5:
        inputs["in"] = decode_grouped(buffer_before(fd, L["ptr"]["in"], n), 64, H, W)
    elif mode == 1:
        C_in = L["Wd"]["mC8"].shape[1] // 4
        inputs["in"] = decode_grouped(buffer_before(fd, L["ptr"]["in"], n), C_in, 2 * H, 2 * W)
    elif mode == 2:
        Cup = tm["Cup"]
        inputs["in"] = decode_grouped(buffer_before(fd, L["ptr"]["in"], n), Cup, H // 2, W // 2)
        inputs["skip"] = decode_grouped(buffer_before(fd, L["ptr"]["skip"], n), C, H, W)
    else:
        inputs["in"] = decode_grouped(buffer_before(fd, L["ptr"]["in"], n), C, H, W)
    if "x88" in L["ptr"]:
        raw = buffer_before(fd, L["ptr"]["x88"], n)
        npx = H * W
        a = np.frombuffer(raw[:npx * 8], np.float16).astype(np.float32)
        if a.size == npx * 4:
            inputs["x88"] = a.reshape(H, W, 4)
    if "x90" in L["ptr"]:
        raw = buffer_before(fd, L["ptr"]["x90"], n)
        npx = (H // 2) * (W // 2)
        a = np.frombuffer(raw[:npx * 8], np.float16).astype(np.float32)
        if a.size == npx * 4:
            inputs["x90"] = a.reshape(H // 2, W // 2, 4)
    if extra:
        inputs.update(extra)
    return inputs


def model_launch(frame, launch, cfg, out_dir=None, attn_only=None, inject=None, dump=DUMP, quiet=False):
    L = load_unit(frame, launch, dump)
    tm = L["tmpl"]
    Wt = prepare(L["Wd"], tm["C"], tm["g"], tm["Cdown"], tm["Cup"], tm["mode"], L["unit"])
    extra = {}
    if attn_only:
        extra["attn_only"] = True
    if inject:
        extra["postattn"] = decode_grouped(Path(inject).read_bytes(), tm["C"], L["H"], L["W"])
    inputs = read_inputs(L, cfg, extra)
    mode, n = tm["mode"], L["d"]["n"]
    if not quiet:
        ptr = L["ptr"]
        print(f"launch {n} frame {frame}: unit {L['unit']} ({L['d']['kernel'].split('ILi')[0][-18:]}) "
              f"k_level<{tm['C']},{tm['g']},{mode},{tm['Cdown']},{tm['Cup']},{tm['n']}> dims {L['W']}x{L['H']} "
              f"shift {L['sx']},{L['sy']}")
        print("  pointers: " + ", ".join(f"{r}={b}" for r, b in ptr.items() if r != "blob"))
        s1 = [float(np.array([v & 0xFFFF], np.uint16).view(np.float16)[0]) for v in L["scal"]]   # u16 bits as f16
        print(f"  Args scalars s1d0,s1d2,s1d4,s1d6 (u16 as f16, NOT applied): {s1[0]:.6g} {s1[1]:.6g} {s1[2]:.6g} {s1[3]:.6g}")
    outs, notes = model_unit(L, None, cfg, inputs, Wt)
    if not quiet:
        for nt in notes:
            print(f"  [{nt}]")
    results = {}
    fd = Path(L["frame_dir"])
    if attn_only:
        if out_dir:
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            (Path(out_dir) / f"{L['ptr']['out']}.bin").write_bytes(outs["out"][0])
        results["attn"] = score_one(outs["out"][0], "fp8", Path(attn_only).read_bytes(), "attn-only", quiet)
        return results, outs
    for role in OUT_ROLES:
        if role not in L["ptr"]:
            continue
        name = L["ptr"][role]
        oracle_path = fd / f"{n:03d}.{name}.bin"
        if not oracle_path.exists():
            raise SystemExit(f"oracle dump {oracle_path} missing (launch {n} does not write {name}?)")
        oracle_raw = oracle_path.read_bytes()
        if role not in outs:
            # unmodelled output: zeros of the oracle's size (scored, flagged)
            dt = {"out": "fp8", "down": "fp8", "o98": "f16", "oa0": "f16"}.get(role, "u32")
            outs[role] = (bytes(len(oracle_raw)), dt, "UNMODELLED (zeros)")
        raw, dt = outs[role][0], outs[role][1]
        tag = outs[role][2] if len(outs[role]) > 2 else ""
        if role in ("oc0", "oc8", "od0", "od8") and dt == "u32":
            tag = (tag + " " if tag else "") + "layout undecoded: scored as u32 words"
        if out_dir:
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            (Path(out_dir) / f"{name}.bin").write_bytes(raw)
        results[role] = score_one(raw, dt, oracle_raw, f"{role}:{name}", quiet, tag)
    return results, outs


def score_one(model_raw, dtype, oracle_raw, label, quiet, tag=""):
    if len(model_raw) != len(oracle_raw):
        raise SystemExit(f"{label}: model {len(model_raw)} bytes, oracle {len(oracle_raw)}")
    res = measure(bytes(model_raw), bytes(oracle_raw), dtype, None)
    if res is None:
        print(f"  {label}: not scorable")
        return None
    n, peak, psnr, r, rrmse, _ = res
    # launch_report.measure() compares whole bytes objects (one bool), so its bit-exact fraction is
    # 0 or 1; the fraction is taken element-wise here.
    exact = float(np.mean(np.frombuffer(bytes(model_raw), np.uint8) == np.frombuffer(bytes(oracle_raw), np.uint8)))
    r_s = "n/a" if r is None else f"{r:.6f}"
    if not quiet:
        print(f"  {label:22s} {dtype:3s} peak {peak:9.4f}  PSNR {psnr:9.4f} dB  r {r_s}  rRMSE {rrmse:.6f}  "
              f"bit-exact {100.0 * exact:8.4f}%  {tag}")
    return dict(psnr=psnr, r=r, rrmse=rrmse, exact=exact, peak=peak, tag=tag)


def parse_sets(items):
    cfg = dict(DEFAULTS)
    for kv in items:
        k, v = kv.split("=", 1)
        if k not in cfg:
            raise SystemExit(f"unknown switch {k}; known: {','.join(cfg)}")
        if isinstance(cfg[k], str):
            cfg[k] = v
        elif isinstance(cfg[k], float):
            cfg[k] = float(v)
        else:
            cfg[k] = int(v)
    return cfg


# ------------------------------------------------------------------ self-check (A2 and the structure checks)
def self_check(frame, dump=DUMP):
    rows = load_rows(frame, dump)
    units = k_level_launches(rows)
    ok_all = True
    print(f"== A2 plumbing self-check, frame {frame} ({len(units)} k_level launches: {units[0]}..{units[-1]})")
    # (a) the packer round trip of every unit's weights
    pj = json.loads((PACKED / "params.json").read_text())
    for n in units:
        L = load_unit(frame, n, dump)
        tm = L["tmpl"]
        blob2, meta2 = repack(dict(L["Wd"]))
        same = blob2 == L["blob"].tobytes() and list(meta2[:20]) == list(L["items"])
        packed_same = (PACKED / f"blob{L['unit']}.bin").read_bytes() == L["blob"].tobytes()
        ok_all &= same and packed_same
        print(f"  weights unit {L['unit']:2d} (launch {n}): {len(blob2):7d} bytes, re-packed through rr_pack.pack "
              f"-> {'byte-identical' if same else 'DIFFERS'}; packed/blob{L['unit']}.bin equal: {packed_same}")
    # (b) every input activation re-encodes to the dump bytes (the layout decode)
    for n in units:
        L = load_unit(frame, n, dump)
        tm = L["tmpl"]; fd = str(L["frame_dir"]); mode = tm["mode"]
        checks = []
        if mode == 5:
            checks.append(("in", 64, L["H"], L["W"]))
        elif mode == 1:
            checks.append(("in", L["Wd"]["mC8"].shape[1] // 4, 2 * L["H"], 2 * L["W"]))
        elif mode == 2:
            checks.append(("in", tm["Cup"], L["H"] // 2, L["W"] // 2)); checks.append(("skip", tm["C"], L["H"], L["W"]))
        else:
            checks.append(("in", tm["C"], L["H"], L["W"]))
        good = 0
        for role, C, h, w in checks:
            raw = buffer_before(fd, L["ptr"][role], n)
            x = decode_grouped(raw, C, h, w)
            ok = encode_grouped(x) == raw
            good += ok
            if not ok:
                print(f"    launch {n} input {role} ({L['ptr'][role]}) does NOT re-encode")
        ok_all &= good == len(checks)
        names = ", ".join(f"{r}={L['ptr'][r]}" for r, *_ in checks)
        print(f"  inputs launch {n} (mode {mode}): {good}/{len(checks)} activation inputs re-encode byte-identically ({names})")
    # (c) structure: the mode 0/1 down-projection of the stored primary output vs the stored down output
    print("== structure check (not fidelity): stored down output vs the 2x2 merge of the stored primary output")
    for n in units:
        L = load_unit(frame, n, dump)
        tm = L["tmpl"]
        if tm["mode"] not in (0, 1) or "down" not in L["ptr"]:
            continue
        fd = str(L["frame_dir"])
        Wt = prepare(L["Wd"], tm["C"], tm["g"], tm["Cdown"], tm["Cup"], tm["mode"], L["unit"])
        prim = decode_grouped((L["frame_dir"] / f"{n:03d}.{L['ptr']['out']}.bin").read_bytes(), tm["C"], L["H"], L["W"])
        dn = decode_grouped((L["frame_dir"] / f"{n:03d}.{L['ptr']['down']}.bin").read_bytes(), tm["Cdown"],
                            L["H"] // 2, L["W"] // 2)
        pred = project_down(prim, Wt, dict(DEFAULTS))
        mm = np.asarray(pred, np.float64).ravel(); oo = np.asarray(dn, np.float64).ravel()
        r = np.corrcoef(mm, oo)[0, 1]
        exact = float(np.mean(codes_grouped(pred).ravel() == codes_grouped(dn).ravel()))
        ok_all &= r > 0.999
        print(f"  unit {L['unit']:2d} (launch {n}, mode {tm['mode']}, Cdown {tm['Cdown']}): r {r:.6f}, "
              f"codes equal after RNE {100*exact:.2f}%  [{'PASS' if r > 0.999 else 'FAIL'}]")
    print("SELF-CHECK:", "PASS" if ok_all else "FAIL")
    return 0 if ok_all else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--launch", type=int)
    ap.add_argument("--frame", type=int, default=1)
    ap.add_argument("--all", action="store_true", help="every k_level launch of the frame")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--out", default=None)
    ap.add_argument("--attn-only", default=None, metavar="ORACLE_OUT",
                    help="stop after attention; score the post-attention residual against this stored file")
    ap.add_argument("--inject-postattn", default=None, metavar="FILE",
                    help="replace the post-attention residual by this stored-layout fp8 tensor (MLP probe)")
    ap.add_argument("--self-check", action="store_true")
    a = ap.parse_args()
    if a.self_check:
        return self_check(a.frame)
    cfg = parse_sets(a.set)
    print("config:", ",".join(f"{k}={v}" for k, v in cfg.items()))
    if a.all:
        rows = load_rows(a.frame)
        summary = []
        for n in k_level_launches(rows):
            out = os.path.join(a.out, f"L{n:02d}") if a.out else None
            res, _ = model_launch(a.frame, n, cfg, out_dir=out)
            summary.append((n, res))
        print("== summary (out78 / out80 PSNR dB, r, rRMSE, bit-exact %)")
        for n, res in summary:
            parts = []
            for role in ("out", "down", "o98", "oa0"):
                if role in res and res[role] is not None:
                    x = res[role]
                    parts.append(f"{role} {x['psnr']:.2f}dB r {x['r'] if x['r'] is None else round(x['r'], 4)} "
                                 f"rRMSE {x['rrmse']:.4f} exact {100*x['exact']:.3f}%")
            print(f"  launch {n:2d}: " + " | ".join(parts))
        return 0
    if a.launch is None:
        ap.error("--launch is required (or --all / --self-check)")
    model_launch(a.frame, a.launch, cfg, out_dir=a.out,
                 attn_only=a.attn_only, inject=a.inject_postattn)
    return 0


if __name__ == "__main__":
    sys.exit(main())
