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
  * the render-resolution outputs of unit 15 (OC0/OC8/OD0/OD8) are not floats: their f32 views contain
    values near 1e30 and denormals. Their layout is undecoded and they are scored as u32 words.

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
  * mode 4 (dec0_1, last unit): no head weights exist. O98 lanes 0..2 = block output channels 0..2,
    lane 3 = 0; OC0/OC8/OD0/OD8 (render resolution) are UNMODELLED (zeros).              GUESS
  * OA0 (2 B/px, one f16) is UNMODELLED (zeros) in every unit.
  * the extra INPUTS (+0x88, +0x90) are not consumed by the first hypothesis.

NOT DECODED (docs/notes/k_rr_level.md; open questions in the report): the block maths, whether the
RMSNorm scale exists, the gram/Q-K assignment, the head's nonlinearity (the encoder head outputs are
positive, see self-check/probe output), the meaning of OA0 and of the 8 B/px extras, the role of
s1d0/s1d2/s1d4/s1d6 (Args +0xa8..+0xb4), the mode-2 head (present in the blob, no output), the
dec0 scalars (Args +0xe8..+0xf7), and the render-resolution outputs of unit 15.
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
    relpos=1, den16=1, ls=1, acc16=0, lscale=1.0, qq=1, pv8=1, skip_mix="s6skip",   # block_v2: relpos bias, f16 softmax den + rcp, LayerScale, f16 K-chunk sums
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
    dec0 = list(struct.unpack_from("<2f", a, 0xe8))
    tmpl = parse_symbol(d["kernel"])
    ptr = {}
    for k, v in d["pointers"].items():
        off = int(k)
        if off in ROLE:
            ptr[ROLE[off]] = v.split("+")[0]
    unit = k_level_launches(rows).index(launch)
    return dict(d=d, W=W, H=H, sx=sx, sy=sy, items=items, blob_size=blob_size, scal=scal, dec0=dec0,
                tmpl=tmpl, ptr=ptr, unit=unit, frame_dir=Path(dump) / f"frame{frame}")


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


def project_down(img, Wt, cfg):
    """Mode 0 / 1 down-projection (verified against the oracle, see the self-check)."""
    cat = merge2x2(q8(img))
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


def block(X, Wt, cfg, H, W, sx, sy, C, g, attn_only=False, postattn=None):
    """Windowed attention + MLP on the (H, W, C) tensor X. Returns (block output or None, post-attn)."""
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
    out = np.clip(X1 + m, -448, 448).astype(np.float32)
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


def block_v2(X, Wt, cfg, H, W, sx, sy, C, g, scal, attn_only=False, postattn=None):
    """Second hypothesis (Review R7): SR's decoded block adapted to the RR packer + ISA facts.
    8x8 windows, relpos bias = the packer's symmetrised 64x64 'gram' (x 2*sqrt(2), as SR's v258),
    V = K (f16), no RMSNorm gain (RR packs none), LayerScale: x += s1d2 * f16(attn + v128),
    x += s1d4 * f16(mlp + v158) (Args +0xac / +0xb0, read as f16 bits; ISA v_pk_mul_f16 %44.1/.2),
    MLP activation = NeuralHub's f16 GELU spline (same 0xad30/0x3698 constants in the RR ISA)."""
    TX, TY = window8_tokens(H, W, sx, sy)
    PX, PY = reflect(TX, W), reflect(TY, H)
    s_act = f16a(X[PY, PX])
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
        if cfg["den16"]:
            inv = f16a(1.0 / np.maximum(pair_sum16(f16a(pe)), np.float32(1e-4)))
        else:
            inv = np.float32(1.0) / np.maximum(pe.sum(-1), np.float32(1e-4))
        prob = f16a(f16a(pe) * inv[..., None]) if cfg["den16"] else pe * inv[..., None]
        if cfg["pv8"]:                       # RR's P @ V is an fp8 WMMA (the RR kernels have no f16 WMMA)
            prob, Vv = q8(prob), K8 if not cfg["qq"] else q8(Kf)
        else:
            Vv = Kf
        ctx = q8(np.einsum("nqk,nkc->nqc", prob, Vv))
        attn += mm(ctx, LUT[Wt["m48"][:, 32 * h:32 * h + 32]])
    a = f16a(attn + Wt["v128_f"])
    X1 = f16a(s_act + (f16a(np.float32(ls1) * a) if cfg["ls"] else a))
    if postattn is not None:
        X1 = f16a(postattn[PY, PX])
    if attn_only:
        return scatter(X1, TX, TY, H, W, C), None
    s2 = norm(X1)
    h1 = q8(act_fn(np.asarray(mm(s2, Wt["W1"]) + Wt["b1"], np.float32), cfg["act"]))
    mo = f16a(mm(h1, Wt["W2"]) + Wt["b2"])
    out = np.clip(X1 + (f16a(np.float32(ls2) * mo) if cfg["ls"] else mo), -448, 448).astype(np.float32)
    return scatter(out, TX, TY, H, W, C), None


def scatter(out, TX, TY, H, W, C):
    img = np.zeros((H, W, C), np.float32)
    ok = (TX >= 0) & (TX < W) & (TY >= 0) & (TY < H)
    img[TY[ok], TX[ok]] = out[ok]
    return img


def upsample(low, Wt, cfg, H, W, C):
    """Mode 2: u = mE8 @ low + v1a0 (rows [q][c]), pixel-shuffled with q = dx + 2 dy."""
    u = np.asarray(low @ Wt["mE8f"].T + Wt["v1a0_f"], np.float32)
    u = u.reshape(u.shape[0], u.shape[1], 4, C)
    order = ((0, 0), (1, 0), (0, 1), (1, 1)) if cfg["up_pix"] == "xy" else ((0, 0), (0, 1), (1, 0), (1, 1))
    out = np.zeros((H, W, C), np.float32)
    for q, (dx, dy) in enumerate(order):
        out[dy::2, dx::2, :] = u[:, :, q, :]
    return out.astype(np.float16).astype(np.float32)


def head(out, Wt, cfg, nh):
    """O98 = m108 @ fp8(out) + v1b8 in lanes 0..2. Lane 3 is 0: it is zero in every 8 B/px output
    the oracle writes (encoders, decoder _1 units, unit 15), so the model writes 0 there."""
    src = q8(out) if cfg["head_in"] == "q8" else out.astype(np.float32)
    y = src @ Wt["m108f"][:3].T + Wt["v1b8_f"][:3]
    o = np.zeros(out.shape[:2] + (4,), np.float32)
    o[..., :3] = y
    return o


def model_unit(L, blobbytes, cfg, inputs, Wt):
    """Returns (outs: role -> (raw bytes, dtype)), (hyp: notes)."""
    tm = L["tmpl"]
    C, g, mode, Cdown, Cup = tm["C"], tm["g"], tm["mode"], tm["Cdown"], tm["Cup"]
    H, W, sx, sy = L["H"], L["W"], L["sx"], L["sy"]
    k = L["unit"]
    outs, notes = {}, []
    if mode == 5:                                                  # enc0: features -> C channels
        F = inputs["in"]                                           # (H, W, 64) fp8 logical
        h = np.asarray(F @ Wt["m1d8f"].T + Wt["v218_f"], np.float32)
        h = q8(act_fn(h, cfg["enc0_act"]))
        X = (h @ Wt["m1f8f"].T + Wt["v230_f"]).astype(np.float16).astype(np.float32)
        notes.append("mode 5 entry (GUESS): m1f8 @ act(m1d8 @ feat + v218) + v230")
    elif mode == 1:                                                # level 0 -> level 1 merge, then the block
        X = np.asarray(merge2x2(inputs["in"]) @ Wt["mC8f"].T + Wt["v188_f"], np.float32)
        X = (act_fn(X, cfg["entry_act"])).astype(np.float16).astype(np.float32)
        notes.append("mode 1 entry merge (GUESS act): mC8 @ merge2x2(in) + v188")
    elif mode == 2:                                                # decoder: up from the level below, + skip
        X = upsample(inputs["in"], Wt, cfg, H, W, C)
        s6 = float(np.array([L["scal"][3] & 0xFFFF], np.uint16).view(np.float16)[0])
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
        img, _ = block_v2(X, Wt, cfg, H, W, sx, sy, C, g, scal, attn_only=attn_only, postattn=postattn)
    else:
        img, _ = block(X, Wt, cfg, H, W, sx, sy, C, g, attn_only=attn_only, postattn=postattn)
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
        if "m108f" in Wt:
            o98 = head(img, Wt, cfg, nh)
            outs["o98"] = (o98.reshape(-1, 4).astype(np.float16).tobytes(), "f16")
            notes.append(f"head ({nh} rows) -> O98 (GUESS)")
    if mode == 4:
        o98 = np.zeros((H, W, 4), np.float32)
        if cfg["mode4_guess"]:
            o98[..., :3] = img[..., :3]
        outs["o98"] = (o98.reshape(-1, 4).astype(np.float16).tobytes(), "f16")
        notes.append("mode 4 (GUESS): O98 lanes 0..2 = block output channels 0..2 (no head weights exist), "
                     "lane 3 = 0; OC/OD unmodelled")
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
        if role in ("oc0", "oc8", "od0", "od8"):
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
