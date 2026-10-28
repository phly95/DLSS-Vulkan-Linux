#!/usr/bin/env python3
"""tools/sr/sr_stage_model.py: the NumPy model of NeuralHub's k_sr_stage / k_sr_tube_split - the spec.

    build/venv/bin/python -I tools/sr/sr_stage_model.py --launch N [--frame F] [--preset M|L]
        [--oracle DUMP_DIR] [--weights PACKED_DIR] [--set key=value ...] [--out DIR]
        [--attn-only ORACLE_T1] [--inject-postattn FILE] [--proj-from-oracle]
    build/venv/bin/python -I tools/sr/sr_stage_model.py --all [--frame F] [--out DIR]   (launches 3..12)
    build/venv/bin/python -I tools/sr/sr_stage_model.py --self-check

One launch of the oracle's dump is modelled. Geometry, input/skip buffers and the weight blob are read
from the dump's launches.jsonl, and the inputs are the oracle's own bytes as they stood before that
launch. Every output (t1-like and, in mode 0, t2-like) is scored with the launch_report.py formulas
(peak = max|oracle|, PSNR, Pearson r, relative RMSE, bit-exact). --out writes the outputs in the
oracle's stored layout, for bit-exact comparison against a GPU dump.

WHAT IS DECODED (docs/notes/k_sr_stage.md "Review R1"; byte map docs/sr/STAGE-LAYOUT.md):
  * the per-channel f16 vectors are P-permuted in the blob (read through invP);
  * RMSNorm: s = x / sqrt(sum x^2 + 2^-14), times the sqrt(C)-scaled gain AFTER the norm, to f16;
  * attention per 4x4 window and head h: logits = Q_h K_h^T + relpos_h, the port's ISA-derived exp,
    an f16 pairwise denominator, then ctx_h = prob @ K_h (V = K), out = sum_h ctx_h @ Wo_h^T with Wo_h
    the C x 32 fragment in slot 0x60, added to every channel;
  * MLP: C/8 blocks of fc1 (32 x C) + b1, GELU (tanh form), fp8 hidden, fc2 (C x 32); fc2_bias once;
  * mode 0: the 2x2 patch-merge projection (Cn x 4C), concat order (0,0) (x+1,0) (0,y+1) (x+1,y+1).
  * mode-0 output 2 is rounded to f16 and clamped to +-6.28125 (f16 0x4648) before fp8;
  * mode 2 (decoder): up = up_W (4C x Cn) @ low + up_b, channel index q*C + c, pixel-shuffled with
    sub-pixel q = dx + 2*dy, to f16, plus the encoder skip BEFORE the block; nothing added after.
NOT DECODED: the exact rounding points (Review R1.8), and the attention residuals of stage 1
(C=96, g=4) and the last tube (stage 7) on real inputs (docs/notes/k_sr_stage.md, Review R2).

The --set switches (defaults = the best measured configuration):
  deperm=1       invP on every f16 vector          snorm8=1   RMSNorm outputs to fp8
  qk8=1          Q and K to fp8 before Q.K         ctx8=1     attention context to fp8 before Wo
  v8=0           V (= K) to fp8 before prob @ V (0: f16, as the ISA's 8 f16 WMMAs suggest)
  prob8=0        probabilities to fp8              proj_in=f32|fp8  what the mode-0 projection reads
  proj_clamp=6.28125   the projection's output clamp (0 = none)
  ss16=0 rs16=0 acc16=0   rounding-closure candidates (Review R1.8): f16 sum-of-squares tree,
                 f16 rsqrt, GEMM K=32 chunks summed through f16
  act=nhgelu|gelu|erfgelu|nr  (nhgelu = NeuralHub's f16 spline)   up_order=qc|cq   mode-2 sub-pixel order
  up_pix=xy|yx   mode-2 sub-pixel order          skip_at=before|after|both|none   where mode 2 adds the skip
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sr_pack import LAYERS, STAGE_BYTES, frag_offsets  # noqa: E402  (NeuralHub's packer: read-only)

REPO = Path(__file__).resolve().parents[2]
ORACLE = REPO / "build/oracle/oracle-8dbf027a9e-s20261025"
P = np.array([0, 1, 16, 17, 2, 3, 18, 19, 4, 5, 20, 21, 6, 7, 22, 23,
              8, 9, 24, 25, 10, 11, 26, 27, 12, 13, 28, 29, 14, 15, 30, 31])
IP = np.argsort(P)
DEFAULTS = dict(deperm=1, snorm8=1, qk8=1, ctx8=1, prob8=0, v8=0, proj_in="fp8", proj_clamp="6.28125", maxsub=0, ss16=0, rs16=0, acc16=0, relT=0, relrot=0, qq=1, norm="isa", qkv="chain", fc1="chain", smax="isa", resid="isa", mlpacc="isa", upacc="isa", up8=1, upref="low", projacc="isa", projq="0123", act="nhgelu",
                up_order="qc", up_pix="xy", skip_at="before")


# ------------------------------------------------------------------ fp8 / f16 helpers
def _lut():
    b = np.arange(256, dtype=np.int32)
    s = np.where(b & 0x80, -1.0, 1.0)
    e = (b >> 3) & 15
    m = b & 7
    v = np.where(e == 0, m / 8.0 * 2.0 ** -6, (1 + m / 8.0) * 2.0 ** (e - 7.0))
    return (s * v).astype(np.float32)          # e=15,m=7 is 480 (not NaN), as the shaders' e4m3_lut


LUT = _lut()


def to_e4m3(x):
    """f32 -> E4M3 code, RNE, saturating at 448 (the shaders' f32_to_e4m3_rne)."""
    x = np.asarray(x, np.float32)
    sign = np.where(x < 0, 0x80, 0).astype(np.uint32)
    a = np.minimum(np.abs(x), np.float32(448.0))
    bits = a.view(np.uint32)
    sub = np.floor(a * np.float32(512.0) + np.float32(0.5)).astype(np.uint32)
    norm = ((bits + np.uint32(0x407ffff) + ((bits >> 20) & 1)) >> 20) & 0x7f
    return (np.where(a < np.float32(0.015625), sub, norm) | sign).astype(np.uint8)


def q8(x):
    return LUT[to_e4m3(x)]


def f16(x):
    return np.asarray(x).astype(np.float16)


def nr_swin_exp(x):
    """The ISA-derived bit-built exp of k_sr_stage (attempt 2): clamp, cubic, mantissa shift."""
    hx = np.float16(x) if np.isscalar(x) else np.asarray(x).astype(np.float16)
    y = np.clip(f16(hx * np.float16(0.017181396484375)), np.float16(-0.55615234375), np.float16(0.55615234375))
    yf = y.astype(np.float32)
    base = f16(-yf * yf + np.float32(0.92724609375))
    p = f16(yf * base.astype(np.float32) + np.float32(1.375))
    bits = ((p.view(np.uint16).astype(np.uint32) << 5) + 0x8000) & 0xffff
    return bits.astype(np.uint16).view(np.float16).astype(np.float32)


def act_fn(z, kind):
    if kind == "gelu":
        return np.float32(0.5) * z * (np.float32(1.0) + np.tanh(np.float32(0.7978845608) * (z + np.float32(0.044715) * z ** 3)))
    if kind == "erfgelu":
        return (0.5 * z * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))).astype(np.float32)
    if kind == "nhgelu":
        # NeuralHub's f16 quadratic-spline GELU (k_sr_stage ISA, the v_pk_*_f16 block with constants
        # 0xad30 / 0x3698): z = f16(acc + b1), t = clamp(z, -2, 2),
        # gate = 0.5 + t * (0.412109375 - 0.0810546875 * |t|), act = z * gate - every step in f16.
        z = f16(z)
        t = np.clip(z, np.float16(-2.0), np.float16(2.0))
        u = f16(f16(np.abs(t) * np.float16(-0.0810546875)) + np.float16(0.412109375))
        g = f16(f16(t * u) + np.float16(0.5))
        return f16(z * g).astype(np.float32)
    if kind == "nr":                         # the NR network's MpCubicSilu - refuted for SR (Review R1.5)
        t = np.clip(z, -4.0, 4.0)
        return z * (t * (np.float32(-0.055908203125) * np.abs(t) + np.float32(0.447265625)) + np.float32(0.89453125))
    raise SystemExit(f"unknown act {kind}")


def ss_isa(xs):
    """The kernel's f16 sum of squares (Review R9, read off k_sr_stage<64|96|128> and k_sr_tube_split):
    xs (..., C) in STORED channel order (group*32 + byte). Lane half h holds bytes [8h, 8h+8) and
    [16+8h, 16+8h+8) of every 32-channel group; per 4-byte dword the packed f16 pair (b0^2 + b2^2,
    b1^2 + b3^2); the groups are reduced as a balanced tree ((g0+g1)+(g2+g3), ((g0+g1)+g2)), then
    low load + high load, then dword 0 + dword 1, then the partner lane (h ^ 1), then lo + hi."""
    f32 = np.float32
    sq = f16(xs * xs).astype(f32)
    G = xs.shape[-1] // 32

    def pairs(gi, off):
        b = sq[..., gi * 32 + off: gi * 32 + off + 4]
        return f16(np.stack([b[..., 0] + b[..., 2], b[..., 1] + b[..., 3]], -1)).astype(f32)

    def tree(t):
        while len(t) > 1:
            t = [f16(t[i] + t[i + 1]).astype(f32) for i in range(0, len(t) - 1, 2)] + ([t[-1]] if len(t) % 2 else [])
        return t[0]

    def lane(h):
        dw = []
        for k in (0, 4):
            lo = tree([pairs(gi, 8 * h + k) for gi in range(G)])
            hi = tree([pairs(gi, 16 + 8 * h + k) for gi in range(G)])
            dw.append(f16(lo + hi).astype(f32))
        return f16(dw[0] + dw[1]).astype(f32)

    t = f16(lane(0) + lane(1)).astype(f32)
    return f16(t[..., 0] + t[..., 1]).astype(f32)[..., None]


def rsqrt_lut(v):
    """g_sr_lut's first half (srref.cpp): f16(f32(1 / sqrt(f64(h)))) of the f16 value h."""
    return f16((1.0 / np.sqrt(np.asarray(v, np.float64))).astype(np.float32)).astype(np.float32)


def f16_pair_sum(e):
    """Sum the last axis (16) by the shader's balanced f16 tree: (0+1),(2+3)... then pairs again."""
    while e.shape[-1] > 1:
        e = f16(e[..., 0::2].astype(np.float32) + e[..., 1::2].astype(np.float32))
    return e[..., 0]


# ------------------------------------------------------------------ weights
class StageWeights:
    """One stage blob, walked in pack_layer's order (FUN_180111850). The walk is asserted to
    end exactly at NeuralHub's blob size, and the stage-0 offsets equal docs/sr/STAGE-LAYOUT.md."""

    def __init__(self, blob, C, g, Cn, mode, deperm):
        self.b, self.C, self.g, self.Cn, self.mode, self.deperm = blob, C, g, Cn, mode, deperm
        cur = 0
        if mode == 2:
            self.up_W = self.frag(cur, 4 * C, Cn); cur += 4 * C * Cn
            self.up_b = self.vec(cur, 4 * C); cur += 8 * C
        self.gain1 = self.vec(cur, C); cur += 2 * C
        self.heads = []
        for _ in range(g):
            Wq = self.frag(cur, 32, C)
            Wk = self.frag(cur + 32 * C, 32, C)
            cur += 64 * C
            rel = np.frombuffer(blob[cur:cur + 512].tobytes(), np.float16).astype(np.float32).reshape(16, 16)
            cur += 512
            Wo = self.frag(cur, C, 32); cur += 32 * C
            self.heads.append((Wq, Wk, rel, Wo))
        self.v270 = self.vec(cur, C); cur += 2 * C
        self.gain2 = self.vec(cur, C); cur += 2 * C
        self.mlp = []
        self.fc2_bias = None
        for j in range(C // 8):
            W1 = self.frag(cur, 32, C); cur += 32 * C
            b1 = self.vec(cur, 32); cur += 64
            W2 = self.frag(cur, C, 32); cur += 32 * C
            self.mlp.append((W1, b1, W2))
            if j == 0:
                self.fc2_bias = self.vec(cur, C); cur += 2 * C
        if mode == 0:
            self.dn_W = self.frag(cur, Cn, 4 * C); cur += 4 * C * Cn
            self.dn_b = self.vec(cur, Cn); cur += 2 * Cn
        if cur != len(blob):
            raise SystemExit(f"weight walk ended at {cur}, blob is {len(blob)} bytes")

    def frag(self, base, rows, cols):
        off = frag_offsets(rows, cols)
        return LUT[self.b[base + off]]

    def vec(self, base, n):
        raw = np.frombuffer(self.b[base:base + 2 * n].tobytes(), np.float16).astype(np.float32)
        if self.deperm and n % 32 == 0:
            idx = (np.arange(n) // 32) * 32 + IP[np.arange(n) % 32]
            raw = raw[idx]
        return raw


# ------------------------------------------------------------------ tensors
def decode_act(raw, C, H, W):
    """[C/32][H][W][32] fp8, stored index i = logical channel cg*32 + P[i]  ->  (H, W, C) f32."""
    a = LUT[np.frombuffer(raw, np.uint8).reshape(C // 32, H, W, 32)]
    out = np.zeros((H, W, C), np.float32)
    for cg in range(C // 32):
        out[:, :, cg * 32 + P] = a[cg]
    return out


def encode_act(x):
    """(H, W, C) f32 -> stored fp8 bytes (inverse of decode_act, with RNE)."""
    H, W, C = x.shape
    out = np.zeros((C // 32, H, W, 32), np.uint8)
    for cg in range(C // 32):
        out[cg] = to_e4m3(x[:, :, cg * 32 + P])
    return out.tobytes()


def reflect(p, n):
    """The kernel's border rule (k_sr_stage ISA 0x78..0xdc; k_level's is the same): a = |p|,
    b = 2(n-1) - a, r = min(a, b, n-1), and r = n-1 when b < 0 (beyond a double fold)."""
    a = np.abs(p)
    b = 2 * (n - 1) - a
    r = np.minimum(np.minimum(a, b), n - 1)
    return np.where(b < 0, n - 1, r)


# ------------------------------------------------------------------ the stage
def run_stage(X, Wt, W, H, sx, sy, tile, cfg, skip=None, low=None, attn_only=False, postattn=None):
    """X: (H, W, C) f32 input (mode 0/1).  Mode 2: `low` (H/2, W/2, Cn) and `skip` (H, W, C).
    Returns (out f32 (H, W, C), t2 f32 (H/2, W/2, Cn) or None)."""
    C, g = Wt.C, Wt.g
    if Wt.mode == 2:
        X = upsample(low, Wt, cfg, H, W)
        if cfg["skip_at"] in ("before", "both"):
            X = f16(X + skip).astype(np.float32)
    gx = (W + sx + tile - 1) // tile
    gy = (H + sy + tile - 1) // tile
    ox = np.arange(gx * tile // 4) * 4 - sx                    # window origins
    oy = np.arange(gy * tile // 4) * 4 - sy
    tx = (ox[None, :, None, None] + np.arange(4)[None, None, None, :])  # (1, nx, 1, 4)
    ty = (oy[:, None, None, None] + np.arange(4)[None, None, :, None])  # (ny, 1, 4, 1)
    TX = np.broadcast_to(tx, (len(oy), len(ox), 4, 4)).reshape(-1, 16)  # token = dy*4 + dx
    TY = np.broadcast_to(ty, (len(oy), len(ox), 4, 4)).reshape(-1, 16)
    PX, PY = reflect(TX, W), reflect(TY, H)
    s_act = f16(X[PY, PX])                                      # (N, 16, C)
    if Wt.mode == 2 and cfg["upref"] == "low":   # Review R9: border tokens reflect in LOW-res coordinates
        U = upsample(low, Wt, cfg, H, W)                        # (H, W, C) on the in-range grid
        h2, w2 = low.shape[:2]
        LX, LY = reflect(TX >> 1, w2), reflect(TY >> 1, h2)     # the low-res pixel each token reads
        UX, UY = 2 * LX + (TX & 1), 2 * LY + (TY & 1)           # its up value at the token's own parity
        up_t = U[np.clip(UY, 0, H - 1), np.clip(UX, 0, W - 1)]
        s_act = f16(up_t + skip[PY, PX]) if cfg["skip_at"] in ("before", "both") else f16(up_t)

    def mmc(a, Wm):
        """Review R9: the kernel's K=32 WMMA chains, acc = f16(group 0) then acc = f16(acc + group k)
        with each group's f32 partial sum fed unrounded (v_fma_mixlo_f16)."""
        acc = f16(a[..., :32] @ Wm[:, :32].T).astype(np.float32)
        for k0 in range(32, a.shape[-1], 32):
            acc = f16(acc + a[..., k0:k0 + 32] @ Wm[:, k0:k0 + 32].T).astype(np.float32)
        return acc

    def mm(a, Wm):
        """a @ Wm.T; acc16: K=32 chunks in f32, each rounded to f16, the chunks added in f16."""
        if not cfg["acc16"] or a.shape[-1] <= 32:
            return a @ Wm.T
        acc = None
        for k0 in range(0, a.shape[-1], 32):
            part = f16(a[..., k0:k0 + 32] @ Wm[:, k0:k0 + 32].T)
            acc = part if acc is None else f16(acc.astype(np.float32) + part.astype(np.float32))
        return acc.astype(np.float32)

    logical = (np.arange(C) // 32) * 32 + P[np.arange(C) % 32]

    def rmsnorm(x, gain):
        xf = x.astype(np.float32)
        if cfg["norm"] == "isa":                  # Review R9: the kernel's tree, LUT rsqrt, f16(x * f16(rs * g))
            ss = ss_isa(xf[..., logical])
            rs = rsqrt_lut(f16(ss + np.float32(2.0 ** -13)).astype(np.float32))
            s = f16(xf * f16(rs * gain).astype(np.float32))
            return f16(q8(s.astype(np.float32))) if cfg["snorm8"] else s
        if cfg["ss16"]:
            ss = f16_pair_sum(f16(xf * xf)).astype(np.float32)[..., None]
        else:
            ss = (xf * xf).sum(-1, keepdims=True)
        rs = np.float32(1.0) / np.sqrt(ss + np.float32(2.0 ** -14))
        if cfg["rs16"]:
            rs = f16(rs).astype(np.float32)
        s = f16(f16(xf * rs).astype(np.float32) * gain)
        return f16(q8(s.astype(np.float32))) if cfg["snorm8"] else s

    s_norm = rmsnorm(s_act, Wt.gain1).astype(np.float32)
    if cfg.get("_tap") is not None:                       # diagnostic hook (default off): block input
        cfg["_tap"]["s_act"] = s_act.astype(np.float16)
        cfg["_tap"]["XY"] = (TX.copy(), TY.copy(),
                             ((TX >= 0) & (TX < W) & (TY >= 0) & (TY < H)).copy())
    attn = np.zeros(s_act.shape, np.float32)
    heads = list(Wt.heads)
    if cfg.get("relT"):
        heads = [(a, b, r.T.copy(), o) for (a, b, r, o) in heads]
    if cfg.get("relrot"):                    # pair head h's QK/Wo with head (h+k)'s relpos table
        k = int(cfg["relrot"]); rels = [h[2] for h in heads]
        heads = [(a, b, rels[(i + k) % len(rels)], o) for i, (a, b, r, o) in enumerate(heads)]
    attn_acc = None
    for (Wq, Wk, rel, Wo) in heads:
        mq = mmc if cfg["qkv"] == "chain" else mm
        Q = mq(s_norm, Wq)
        K = f16(mq(s_norm, Wk)).astype(np.float32)
        V = K                                     # V = K (Review R1.4), taken before the fp8 step
        if cfg["qk8"]:
            Q, K = q8(Q), q8(K)
        if cfg["v8"]:
            V = q8(V)
        if cfg["qq"]:                             # RR's form (k_rr_level Review R7b): logits Q.Q^T
            K = Q
        logits = np.einsum("nqc,nkc->nqk", Q, K) + rel[None]
        if cfg.get("_stats") is not None:
            cfg["_stats"].append((float(logits.max()), float((logits > 32.4).mean()), float((logits < -32.4).mean())))
        if cfg["maxsub"]:
            logits = logits - logits.max(-1, keepdims=True)
        pe = nr_swin_exp(logits)
        if cfg["smax"] == "isa":                  # Review R9: the kernel's key order and the 1/h table
            e = f16(pe).astype(np.float32)
            a = lambda i, j: f16(e[..., i] + e[..., j]).astype(np.float32)
            b = lambda u, v: f16(u + v).astype(np.float32)
            ev = b(b(a(0, 8), a(2, 10)), b(a(4, 12), a(6, 14)))
            od = b(b(a(1, 9), a(3, 11)), b(a(5, 13), a(7, 15)))
            tot = b(ev, od)
            inv = f16(np.float32(1.0) / tot).astype(np.float32)
            prob = f16(e * inv[..., None]).astype(np.float32)
        else:
            tot = f16_pair_sum(f16(pe)).astype(np.float32)
            prob = pe * (np.float32(1.0) / np.maximum(tot, np.float32(1e-4)))[..., None]
        if cfg["prob8"]:
            prob = q8(prob)
        ctx = np.einsum("nqk,nkc->nqc", prob, V)
        if cfg["resid"] == "isa":                 # Review R9: ctx -> f16 -> fp8; heads chained through f16
            ctx = q8(f16(ctx).astype(np.float32))
            if cfg.get("_tap") is not None and "ctx" not in cfg["_tap"]:   # diagnostic: head-0 ctx
                cfg["_tap"]["ctx"] = ctx
            if attn_acc is None:
                v270 = np.zeros(C, np.float32) if attn_only else f16(Wt.v270).astype(np.float32)
                attn_acc = f16(s_act.astype(np.float32) + v270).astype(np.float32)
            attn_acc = f16(attn_acc + ctx @ Wo.T).astype(np.float32)
            continue
        if cfg["ctx8"]:
            ctx = q8(ctx)
        attn += ctx @ Wo.T
    if attn_acc is not None:
        s_act = f16(attn_acc)
    else:
        s_act = f16(s_act.astype(np.float32) + attn)
    if postattn is not None:                                   # MLP sub-block check
        s_act = f16(postattn[PY, PX])
    if attn_only:
        out = s_act.astype(np.float32)
    else:
        if cfg["resid"] != "isa":                 # isa: v270 was added before the heads
            s_act = f16(s_act.astype(np.float32) + f16(Wt.v270).astype(np.float32))
        if cfg.get("_tap") is not None:           # diagnostic hook: X1 (post-attention residual)
            cfg["_tap"]["X1"] = s_act.astype(np.float16)
        s_n2 = rmsnorm(s_act, Wt.gain2).astype(np.float32)
        if cfg.get("_tap") is not None:           # diagnostic hook: norm2 output (fc1 input)
            cfg["_tap"]["s_n2"] = s_n2.astype(np.float16)
        s_mlp = np.zeros(s_act.shape, np.float16)
        mlp_acc = f16(s_act.astype(np.float32) + f16(Wt.fc2_bias).astype(np.float32)).astype(np.float32)
        for (W1, b1, W2) in Wt.mlp:
            if cfg["fc1"] == "chain":             # Review R9: bias first, then the K=32 chains through f16
                acc = np.broadcast_to(f16(b1).astype(np.float32), s_n2.shape[:-1] + (W1.shape[0],))
                for k0 in range(0, s_n2.shape[-1], 32):
                    acc = f16(acc + s_n2[..., k0:k0 + 32] @ W1[:, k0:k0 + 32].T).astype(np.float32)
                h1 = q8(act_fn(acc, cfg["act"]))
            elif cfg["fc1"] == "chain_nb":
                h1 = q8(act_fn(f16(mmc(s_n2, W1) + b1).astype(np.float32), cfg["act"]))
            else:
                h1 = q8(act_fn((mm(s_n2, W1) + b1).astype(np.float32), cfg["act"]))
            if cfg.get("_tap") is not None:                # diagnostic hook: fc1 hidden (fp8)
                cfg["_tap"].setdefault("h1", []).append(h1)
            if cfg["mlpacc"] == "isa":            # Review R9: acc = f16(acc + block's f32 fc2 partial)
                mlp_acc = f16(mlp_acc + h1 @ W2.T).astype(np.float32)
            else:
                s_mlp = f16(s_mlp.astype(np.float32) + f16(h1 @ W2.T).astype(np.float32))
        if cfg["mlpacc"] == "isa":
            out = np.clip(mlp_acc, -448, 448)
        else:
            out = np.clip(s_act.astype(np.float32) + s_mlp.astype(np.float32) + Wt.fc2_bias, -448, 448)
    # scatter in-bounds tokens back to the image (each pixel belongs to exactly one window)
    img = np.zeros((H, W, C), np.float32)
    ok = (TX >= 0) & (TX < W) & (TY >= 0) & (TY < H)
    img[TY[ok], TX[ok]] = out[ok]
    if Wt.mode == 2 and cfg["skip_at"] in ("after", "both"):
        img = np.clip(img + skip, -448, 448)
    t2 = project(img, Wt, cfg) if Wt.mode == 0 and not attn_only else None
    return img, t2


def project(img, Wt, cfg):
    """Mode-0 2x2 patch merge. The result is rounded to f16 and clamped to +-6.28125 (f16 0x4648,
    `v_pk_min/max_num_f16` just before the store, k_sr_stage ISA line ~3245) before fp8."""
    src = q8(img) if cfg["proj_in"] == "fp8" else img
    cat = np.concatenate([src[dy::2, dx::2, :] for (dx, dy) in ((0, 0), (1, 0), (0, 1), (1, 1))], -1)
    if cfg["projacc"] == "isa":               # bias first, then the K=32 chains through f16
        y = np.broadcast_to(f16(Wt.dn_b).astype(np.float32), cat.shape[:-1] + (Wt.dn_W.shape[0],))
        C = Wt.C
        qorder = [int(c) for c in str(cfg["projq"])]
        for q in qorder:
            for g0 in range(0, C, 32):
                k0 = q * C + g0
                y = f16(y + cat[..., k0:k0 + 32] @ Wt.dn_W[:, k0:k0 + 32].T).astype(np.float32)
    else:
        y = f16(cat @ Wt.dn_W.T + Wt.dn_b).astype(np.float32)
    lim = float(cfg["proj_clamp"])
    return np.clip(y, -lim, lim) if lim > 0 else np.clip(y, -448, 448)


def upsample(low, Wt, cfg, H, W):
    """Mode-2 hypothesis: up = up_W (4C x Cn) @ low + up_b, pixel-shuffled to (H, W, C)."""
    C = Wt.C
    if cfg["upacc"] == "isa":                 # Review R9: bias first, then the K=32 chains through f16
        u = np.broadcast_to(f16(Wt.up_b).astype(np.float32), low.shape[:-1] + (4 * C,))
        for k0 in range(0, low.shape[-1], 32):
            u = f16(u + low[..., k0:k0 + 32] @ Wt.up_W[:, k0:k0 + 32].T).astype(np.float32)
    else:
        u = low @ Wt.up_W.T + Wt.up_b                               # (H/2, W/2, 4C)
    u = u.reshape(u.shape[0], u.shape[1], 4, C) if cfg["up_order"] == "qc" else \
        u.reshape(u.shape[0], u.shape[1], C, 4).transpose(0, 1, 3, 2)
    out = np.zeros((H, W, C), np.float32)
    order = ((0, 0), (1, 0), (0, 1), (1, 1)) if cfg["up_pix"] == "xy" else ((0, 0), (0, 1), (1, 0), (1, 1))
    for q, (dx, dy) in enumerate(order):
        out[dy::2, dx::2, :] = u[:, :, q, :]
    if cfg["up8"]:                            # Review R9: the up-projection is staged in LDS as fp8
        return q8(np.clip(f16(out).astype(np.float32), -448, 448))
    return f16(out).astype(np.float32)


# ------------------------------------------------------------------ oracle plumbing
def parse_kernel(sym):
    """C, g, mode, Cn, tile from the mangled symbol."""
    if "k_sr_tube_split" in sym:
        return 128, 4, 1, 0, 4
    t = sym.split("k_sr_stageI")[1]
    ints = [int(v) for v in __import__("re").findall(r"Li(\d+)E", t)[:4]]
    return ints[0], ints[1], ints[2], ints[3], 8


def buffer_before(dump_frame, name, launch):
    """The bytes of `name` as they stood before `launch` (the newest earlier dump, else start)."""
    best = None
    for f in os.listdir(dump_frame):
        if f.endswith(f".{name}.bin") and f[:3].isdigit() and int(f[:3]) < launch:
            if best is None or int(f[:3]) > int(best[:3]):
                best = f
    p = Path(dump_frame) / (best if best else f"start.{name}.bin")
    return p.read_bytes()


def load_launch(dump_dir, frame, launch):
    rows = [json.loads(l) for l in open(Path(dump_dir) / f"frame{frame}" / "launches.jsonl")]
    d = next(r for r in rows if r["n"] == launch)
    a = bytes.fromhex(d["args"])
    W, H, sx, sy = np.frombuffer(a[40:56], np.int32).tolist()
    ptr = {int(k): v.split("+") for k, v in d["pointers"].items()}
    return d, W, H, sx, sy, ptr


def stage_index(woff, wdir):
    tot = 0
    for i in range(10):
        if tot == woff:
            return i
        tot += (os.path.getsize(Path(wdir) / f"stage{i}.bin") + 255) & ~255
    raise SystemExit(f"weight offset {woff} is not a stage blob")


def score(port, orc, name):
    a, b = np.asarray(port, np.float64).ravel(), np.asarray(orc, np.float64).ravel()
    peak = float(np.abs(b).max())
    mse = float(np.mean((a - b) ** 2))
    r = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan")
    rr = math.sqrt(mse) / max(math.sqrt(float(np.mean(b * b))), 1e-30)
    psnr = 10 * math.log10(peak * peak / max(mse, 1e-30))
    print(f"  {name:10s} peak {peak:8.4f}  PSNR {psnr:8.4f} dB  r {r:.6f}  rRMSE {rr:.6f}")
    return psnr, r, rr


def model_launch(args, cfg):
    dump = Path(args.oracle)
    fdir = dump / f"frame{args.frame}"
    d, W, H, sx, sy, ptr = load_launch(dump, args.frame, args.launch)
    C, g, mode, Cn, tile = parse_kernel(d["kernel"])
    si = stage_index(int(ptr[0][1]), args.weights)
    blob = np.frombuffer((Path(args.weights) / f"stage{si}.bin").read_bytes(), np.uint8)
    if len(blob) != STAGE_BYTES[si]:
        raise SystemExit(f"stage{si}.bin size {len(blob)} != {STAGE_BYTES[si]}")
    lay = LAYERS[si]
    assert (lay[2], lay[3], lay[5]) == (C, g, mode), (lay, C, g, mode)
    Wt = StageWeights(blob, C, g, lay[4], mode, cfg["deperm"])
    in_name, out_name = ptr[8][0], ptr[24][0]
    print(f"launch {args.launch} frame {args.frame}: stage{si} {lay[0]} C={C} g={g} Cn={lay[4]} mode={mode} "
          f"tile={tile} dims {W}x{H} shift {sx},{sy}  {in_name} -> {out_name}"
          + (f" + {ptr[32][0]}" if 32 in ptr else "") + (f"  skip {ptr[16][0]}" if 16 in ptr else ""))
    kw = {}
    if mode == 2:
        kw["low"] = decode_act(buffer_before(fdir, in_name, args.launch), lay[4], H // 2, W // 2)
        kw["skip"] = decode_act(buffer_before(fdir, ptr[16][0], args.launch), C, H, W)
        X = None
    else:
        X = decode_act(buffer_before(fdir, in_name, args.launch), C, H, W)
    if args.inject_postattn:
        kw["postattn"] = decode_act(Path(args.inject_postattn).read_bytes(), C, H, W)
    img, t2 = run_stage(X, Wt, W, H, sx, sy, tile, cfg, attn_only=bool(args.attn_only), **kw)
    res = {}
    t1_ref = Path(args.attn_only) if args.attn_only else fdir / f"{args.launch:03d}.{out_name}.bin"
    q = encode_act(img)
    orc = np.frombuffer(t1_ref.read_bytes(), np.uint8)
    print(f"  {out_name:10s} bit-exact {100 * np.mean(np.frombuffer(q, np.uint8) == orc):.4f}%")
    res[out_name] = score(LUT[np.frombuffer(q, np.uint8)], LUT[orc], out_name)
    if t2 is not None:
        o2n = ptr[32][0]
        o2 = np.frombuffer((fdir / f"{args.launch:03d}.{o2n}.bin").read_bytes(), np.uint8)
        q2 = encode_act(t2)
        print(f"  {o2n:10s} bit-exact {100 * np.mean(np.frombuffer(q2, np.uint8) == o2):.4f}%")
        res[o2n] = score(LUT[np.frombuffer(q2, np.uint8)], LUT[o2], o2n)
        if args.proj_from_oracle:                               # projection sub-block check
            t1o = decode_act((fdir / f"{args.launch:03d}.{out_name}.bin").read_bytes(), C, H, W)
            q3 = encode_act(project(t1o, Wt, dict(cfg, proj_in="f32")))
            res["proj|oracle"] = score(LUT[np.frombuffer(q3, np.uint8)], LUT[o2], "proj|orc")
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        Path(args.out, f"{out_name}.bin").write_bytes(q)
        if t2 is not None:
            Path(args.out, f"{ptr[32][0]}.bin").write_bytes(encode_act(t2))
    return res


def self_check():
    """Reproduce Review R1's numbers (the loop replica build/work/review/stage_model_rev.py) on SR M
    frame 0 launch 3. Tolerance 0.05 dB: the vectorized sums run in a different f32 order."""
    base = argparse.Namespace(launch=3, frame=0, oracle=str(ORACLE / "sr_M/dump"),
                              weights=str(REPO / "build/re/packed/M"), attn_only=None,
                              inject_postattn=None, proj_from_oracle=False, out=None)
    ra = str(REPO / "build/work/probe/real_attn_dump/frame0/003.t1.bin")
    # The rounding closure of Review R9 is pinned off for every historical case.
    PRE9 = dict(norm="old", qkv="f32", fc1="f32", smax="old", resid="old", mlpacc="old", upacc="f32",
                up8=0, upref="full", projacc="f32")
    cases = [
        # Review R1's numbers (tanh GELU; its replica rounded V exactly when it rounded K: v8 = qk8).
        # qq=0 is pinned here: these document Review R1, which predates the Q.Q^T finding (R8).
        ("R1 decoded, no rounding switches", dict(**PRE9, snorm8=0, qk8=0, ctx8=0, prob8=1, v8=0, act="gelu", qq=0), {}, "t1", 34.0704),
        ("R1 decoded, best switches", dict(**PRE9, v8=1, act="gelu", qq=0), {}, "t1", 34.6343),
        ("R1 attention only vs --isolate", dict(**PRE9, snorm8=0, qk8=0, ctx8=0, prob8=1, v8=0, qq=0), dict(attn_only=ra), "t1", 36.6338),
        ("R1 attention only, probs unquantized", dict(**PRE9, snorm8=0, qk8=0, ctx8=0, v8=0, qq=0), dict(attn_only=ra), "t1", 36.8595),
        ("R1 MLP given oracle post-attention", dict(**PRE9, snorm8=0, qk8=0, ctx8=0, prob8=1, v8=0, act="gelu", qq=0), dict(inject_postattn=ra), "t1", 34.4195),
        # Review R2 / R4 as they stood with Q.K^T; kept at qq=0 for the record.
        ("R2 defaults (tanh GELU), t1", dict(**PRE9, act="gelu", qq=0), {}, "t1", 35.0790),
        ("R2 defaults (tanh GELU), t2", dict(**PRE9, act="gelu", qq=0), {}, "t2", 31.9087),
        ("R4 defaults (NeuralHub f16 GELU), t1", dict(**PRE9, qq=0), {}, "t1", 35.2614),
        # The current default: logits Q.Q^T (Review R8).  Expected values re-derived from the
        # oracle with --set qq=1 and cross-checked against R8's own table (37.74 / 34.01 dB).
        ("R8 default (Q.Q^T), t1", dict(**PRE9), {}, "t1", 37.7384),
        ("R8 default (Q.Q^T), t2", dict(**PRE9), {}, "t2", 34.0088),
        # Review R9: the kernel's rounding closure (ISA norm tree + LUTs, K=32 chains, softmax order,
        # residual and MLP chains). Re-derived from the oracle with the current defaults.
        ("R9 default (rounding closure), t1", {}, {}, "t1", 68.8920),
        ("R9 default (rounding closure), t2", dict(projacc="f32"), {}, "t2", 54.5715),
        # R9 addendum: the mode-0 projection as bias-first K=32 chains (q-major, as the LDS chunks).
        ("R9 projection chain, t2", {}, {}, "t2", 61.9141),
    ]
    ok = True
    for label, sw, extra, key, want in cases:
        print(f"[{label}]")
        res = model_launch(argparse.Namespace(**{**vars(base), **extra}), {**DEFAULTS, **sw})
        got = res[key][0]
        good = abs(got - want) < 0.05
        ok &= good
        print(f"  -> {got:.4f} dB, expected {want:.4f} +- 0.05: {'PASS' if good else 'FAIL'}")
    print("SELF-CHECK:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--launch", type=int)
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--preset", default="M", choices=["M", "L"])
    ap.add_argument("--oracle", default=None, help="dump dir (default: the oracle's sr_<preset>/dump)")
    ap.add_argument("--weights", default=None, help="packed dir (default: build/re/packed/<preset>)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--out", default=None)
    ap.add_argument("--attn-only", default=None, metavar="ORACLE_T1",
                    help="drop v270 + MLP and score against this oracle dump (the --isolate probe)")
    ap.add_argument("--inject-postattn", default=None, metavar="FILE",
                    help="replace the post-attention residual by this stored-layout fp8 tensor")
    ap.add_argument("--proj-from-oracle", action="store_true",
                    help="also score the mode-0 projection fed with the oracle's own t1")
    ap.add_argument("--self-check", action="store_true")
    ap.add_argument("--all", action="store_true", help="every stage/tube launch of the frame (3..12)")
    a = ap.parse_args()
    if a.self_check:
        return self_check()
    if a.all:
        a.oracle = a.oracle or str(ORACLE / f"sr_{a.preset}/dump")
        a.weights = a.weights or str(REPO / f"build/re/packed/{a.preset}")
        cfg = dict(DEFAULTS)
        for kv in a.set:
            k, v = kv.split("=", 1)
            cfg[k] = type(cfg[k])(v) if not isinstance(cfg[k], str) else v
        print("config:", ",".join(f"{k}={v}" for k, v in cfg.items()))
        out_root = a.out
        for n in range(3, 13):
            a.launch = n
            a.out = os.path.join(out_root, f"L{n:02d}") if out_root else None
            model_launch(a, cfg)
        return 0
    if a.launch is None:
        ap.error("--launch is required")
    a.oracle = a.oracle or str(ORACLE / f"sr_{a.preset}/dump")
    a.weights = a.weights or str(REPO / f"build/re/packed/{a.preset}")
    cfg = dict(DEFAULTS)
    for kv in a.set:
        k, v = kv.split("=", 1)
        if k not in cfg:
            raise SystemExit(f"unknown switch {k}")
        cfg[k] = type(cfg[k])(v) if not isinstance(cfg[k], str) else v
    print("config:", ",".join(f"{k}={v}" for k, v in cfg.items()))
    model_launch(a, cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
