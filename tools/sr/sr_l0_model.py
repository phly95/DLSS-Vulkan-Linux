#!/usr/bin/env python3
"""tools/sr/sr_l0_model.py: the NumPy spec of NeuralHub's k_sr_l0 (preset L), BOTH template
variants - the spec.

    build/venv/bin/python -I tools/sr/sr_l0_model.py [--frame F] [--oracle DUMP_DIR]
        [--weights PACKED_DIR] [--gpu GPU_DUMP_FRAME_DIR] [--out DIR]
    build/venv/bin/python -I tools/sr/sr_l0_model.py --self-check

Both launches of the sr_L dump are modelled from the oracle's own inputs (the bytes as they stood
before the launch, buffer_before) and the oracle's own weight blobs; every output is scored with
tools/report/launch_report.py's measure() (peak, PSNR, r, rRMSE, bit-exact).

  k_sr_l0<0,2,b>  sr_L dump launch 3  (enc0L layer, C=32 g=2 Cn=64 mode 0):
      l0_in (enc0L_in's features, 264x168x32 fp8 = the 256x160 level-0 grid with a 2-pixel
      window-shift border at buffer offset (2,2)) -> the enc0L embedding (two 32x32 fp8 layers,
      wbuf+1577728 blob) -> the transformer block (tile 8, shift 2,2) -> the mode-0 down
      projection -> t0 (128x80x64 fp8, stage0's input).
      The launch also writes l0_skip; it is NOT modelled here (L2 takes it from the oracle's
      own 003.l0_skip.bin, as every model input is the oracle's bytes).

  k_sr_l0<2,2,b>  sr_L dump launch 14 (dec0L layer, C=32 g=2 Cn=64 mode 2):
      t8 (stage9's output, 128x80x64) -> the mode-2 up projection (wbuf+1608192 blob) ->
      + l0_skip -> the transformer block (tile 8, shift 0,0) -> the dec0 head families
      (tools/sr/sr_dec0_model.py Review R3b) at the render grid 256x144 ->
      gates / factors / aniso / hidden0.
      L0Args has NO prev_mvec / hin pointers (0x18, 0x20 are NULL), so the hidden head has no
      warp term (see SPEC).

SPEC (established from the ISA listings build/agents/l0-decode/l0_{0,2}.s; every formula is
marked [ISA] when read off the listing and [dec0] when ported from sr_dec0_model):

  Blob layouts (byte-identical to the oracle's wbuf slices, verified):
    enc0L = [dither 4096][W1 frag(32,32) @4096][b1 f16x32 P-permuted @5120]
            [W2 frag(32,32) @5184][b2 f16x32 @6208][ENC0L pack_layer @6272 = 24192 B]
    dec0L = pack_layer(DEC0L) = [up_W frag(128,64)][up_b f16x128][gain1][2 heads
            (QK frag(64,32) + relpos f16 16x16 + Wo frag(32,32))][v270][gain2]
            [4 MLP blocks (fc1 + b1 + fc2), fc2_bias after block 0] = 24320 B.

  Embedding (launch 3, ISA 0x214..0x7f8) [ISA]:
    e1 = fp8(clamp(f16(b1 + W1 . x), 0, 448))     <- pk_max(v, 0) then pk_min(448), then fp8
    e2 = f16(b2 + W2 . e1)                        <- straight into RMSNorm (no fp8 roundtrip)
    over EVERY pixel of the padded 264x168 buffer (border pixels included); the level-0 crop
    (buffer rows/cols 2..161 / 2..257) is what reaches the block output and the down projection.

  Block: the k_sr_stage block (docs/notes/k_sr_stage.md Review R9 closure) at C=32, g=2, tile 8:
    launch 3  shift (2,2) on the 256x160 grid, window tokens read from the PADDED buffer at
              (token + 2) - the border bytes are enc0L_in's (left/top border verified equal to
              the stage's reflect rule, the far border is the buffer's own content);
    launch 14 shift (0,0) - no border at all (the grid is exactly 256x160).
    Rounding closure (sm.run_stage defaults): ISA norm tree + LUT rsqrt, snorm8, QKV K=32
    chains through f16, Q.Q^T logits, f16 softmax key order + 1/h LUT, ctx -> f16 -> fp8,
    v270 first in the residual, nhgelu, f16 fc1/mlp accumulators, mode-0 projection as the
    bias-first K=32 chain clamped to +-6.28125 (0x4648 present in l0_0.s), fp8 projection
    input (+-448 staging); mode 2's up projection staged as fp8 (up8) + skip before the block.

  Heads (launch 14) [dec0 = the formulas, oracle = L's channel indices]: the head input is the
    block output at the render crop (144,256), read from LDS as f16 (ds_load_2addr 0x4534-0x454c,
    base tid*64 = one 32xf16 block per pixel):
      gates   byte i = rndne(255 * clamp01(f16(sg(p_c)))) for c = 0, 1, 6, byte 3 = 0
      factors byte i = rndne(255 * clamp01(e_c/(e_7+e_8+e_9))), e_c = exp(p_c - max) in f32,
              c = 7, 8, 9; byte 3 = 0                    (f32 divide: v_div_scale chain)
      aniso   A = s0 * f16(sg(clamp(p10, +-20))), B = s0 * f16(sg(clamp(p11, +-20))),
              stored f16(A*A), f16(B*B), f16((A*B) * tanh(f0 * p12)), element 3 = 0;
              s0 = Args +0x60 (8.0), f0 = Args +0x64 (0.09785986691713333 = params.json dec0_f0)
      hidden0_k = rndne(127 * clamp(tanh(p_(2+k)))), k = 0..3 (channels 2,3,4,5) - the RR u15
              tanh-byte family: L0Args 0x18/0x20 are NULL, so unlike dec0 there is no hin/mv
              pointer, no warp term and no z gate.

MEASURED (--self-check, tools/report/launch_report.py's measure): see the run at the bottom of
build/agents/l0-decode/progress.md. L2 passes on both frames (all four heads 99.2-99.9%, bar
97%); L1 frame0 passes (t0 99.6599%, bar 99%); L1 frame1 is at 95.94% - OPEN, see "Open" below.

Per-stage verification table (L3) - every row is model vs the oracle dump (frames 0/1) or an
ISA listing; numbers are bit-exact % of the named buffer:
| stage | verification | f0 | f1 |
|---|---|---|---|
| l0_in layout | 264x168x32, level-0 content at (2,2); near border byte-identical to the stage's reflect rule (col0==col4, col1==col3, row0==row4, row1==row3) | 100 | 100 |
| embed clamp e1 in [0,448] | A/B vs oracle l0_skip: [0,448] 99.90/98.24 vs [-448,448] 1.18/4.99 | 99.90 | 98.24 |
| embed accumulation order | cur/chunk/chunkr/f64/seq give byte-identical scores -> the f32 K=32 sums are exact | = | = |
| token reads: buffer vs reflect | buffer reads win (reflect re-computes the far border): 99.897/98.238 vs 98.674/97.255 | 99.897 | 98.238 |
| block output (= oracle 003.l0_skip) | model vs oracle, fp8 | 99.8972 | 98.2384 |
| down projection | project(decode(oracle l0_skip)) vs oracle t0 (input exact) - 4 accumulation orders identical | 99.9925 | 99.9922 |
| t0 (L1, full pipeline) | model vs oracle 003.t0.bin | **99.6599** | **95.9425** |
| norm ss tree | traced register-by-register in l0_0.s 0x730-0x9c8 = ss_isa's grouping (pairs -> low+high -> dword sum -> lane^16 partner -> lo+hi), +2^-13 as f16 add 0x800, rsqrt LUT load 0x984 = the stage's construction | match | match |
| swin exp | 0x2466 scale, 0xb873/0x3873 clamps (x16 = stage x16), 0x3b6b = f16(0.92724609), 0x3d80 = f16(1.375), mantissa <<5 + 0x8000 = nr_swin_exp bit for bit | match | match |
| softmax sum tree | R9 pairing: alternative trees scored 92.69/91.85 (linear) and 95.36/94.21 (adjacent-pair) - worse, so the (i,i+8),(i,i+2) tree is right; reciprocal via f64 identical (no double-rounding case) | 99.897 | 98.238 |
| every config switch | qq/act/relT/resid/mlpacc/norm/smax/qkv/maxsub/prob8/qk8/relrot/snorm8/proj_in/v8/fc1/ss16/rs16/acc16/ctx8/projacc/projq swept vs l0_skip: all alternatives strictly worse or byte-identical no-ops (progress.md table) | baseline wins | baseline wins |
| heads: gates c=(0,1,6) | per-byte exact vs oracle (dec0's channels 0,1,10 score ~1%) | 99.89/99.88/99.94 | >=99.6 all |
| heads: factors c=(7,8,9) | log-ratio corr with p_c 0.999/0.946; full-head exact | 99.884 | 99.666 |
| heads: aniso A=10,B=11,T=tanh(f0*12) | corr 0.988/0.9997/1.0 (B==1.0 also proves l0<2>'s block output) | 99.679 | 99.218 |
| heads: hidden0 = tanh(2..5) | per-channel exact; the RR u15 tanh family (no warp: L0Args 0x18/0x20 = NULL) | 99.888 | 99.644 |

Open (L1 frame1): 95.94% vs the 99% bar. Bisected with a hand-rolled srref --L patchprobe
(build/agents/l0-decode/probenorm{,2}.py; all probe runs < 1 s; evidence table in
build/agents/l0-decode/progress.md):
  post-norm (norm1, exact e2 input):     99.6962% f0 / 99.9080% f1  - input path clean;
  post-attention (norm2, MLP input):     95.81% f0 / 95.36% f1      - per-wg bimodal, set
      (first window per wave) confirmed at 100.0000% on matching workgroups;
  post-MLP (oracle l0_skip):             99.8972% f0 / 98.2384% f1;
  the block residue correlates only weakly with the norm2 mismatches (f1: 1.39% block-bad at
  norm2-clean workgroups vs 1.93% at norm2-bad) because norm2 = fp8(norm(X1)) compresses X1
  differences while X1 itself feeds mlp_acc directly through the residual - i.e. the first
  stage that carries the divergence to the output is the ATTENTION (between the clean post-norm
  probe and the post-attention probe). Frame-1's data amplifies it (block 1.76% -> t0 95.94%).
  Ruled out by measurement: input accumulation order, projection (99.99% from oracle input),
  every run_stage config switch, ss-tree/rsqrt/+2^-13 variants (<=0.02 pp), fc1/fc2 K=32
  chunk order (byte-identical), ctx GEMM order (byte-identical), softmax sum tree (alternatives
  strictly worse). The earliest non-zero measured divergence is the norm formula's own
  0.09-0.30% (input exact, unexplained, frame-INVERSE-correlated) - too small to explain
  frame1 alone; the attention expression trees (QKV f16 boundaries, softmax e/prob rounding,
  ctx->fp8, residual order) against an X1 probe remain untraced - not completed within the
  follow-up budget; the model is submitted with L1 frame1 at 95.94% and this evidence.
"""
import argparse
import json
import struct
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[0] / "report"))
import sr_stage_model as sm  # noqa: E402  (the stage block spec, Review R9 - read-only)
from sr_dec0_model import sg  # noqa: E402  (Review R3b's sigmoid, shared with the dec0 heads)
from sr_pack import frag_offsets  # noqa: E402  (NeuralHub's packer: read-only)
from launch_report import measure  # noqa: E402  (the canonical per-launch scoring)

f16 = sm.f16
q8 = sm.q8
LUT = sm.LUT
f32 = np.float32
REPO = sm.REPO
ORACLE = REPO / "build/oracle/oracle-8dbf027a9e-s20261025/sr_L/dump"
WEIGHTS = REPO / "build/re/packed/L"
KERNEL0 = "_ZN2sr2io7k_sr_l0ILi0ELi2ELb0EEEvNS0_6L0ArgsE"
KERNEL2 = "_ZN2sr2io7k_sr_l0ILi2ELi2ELb0EEEvNS0_6L0ArgsE"
DTYPE = {"t0": "fp8", "gates": "u8", "factors": "u8", "aniso": "f16", "hidden0": "i8"}


# ------------------------------------------------------------------ plumbing
def load_launch(fdir, kernel):
    rows = [json.loads(l) for l in open(Path(fdir) / "launches.jsonl")]
    d = next(r for r in rows if r["kernel"] == kernel)
    ptr = {int(k): v.split("+")[0] for k, v in d["pointers"].items()}
    return d, bytes.fromhex(d["args"]), ptr


def dims(a):
    """L0Args +0x48: (W0, H0) level-0 grid, (sx, sy) window shift, (Wr, Hr) render grid."""
    return struct.unpack_from("<6i", a, 0x48)


def vec(blob, base, n):
    """A blob f16 vector as the packer wrote it (P-permuted in 32-groups)."""
    raw = np.frombuffer(bytes(blob[base:base + 2 * n]), np.float16).astype(np.float32)
    if n % 32 == 0:
        idx = (np.arange(n) // 32) * 32 + sm.IP[np.arange(n) % 32]
        raw = raw[idx]
    return raw


class Enc0L:
    """The launch-3 weight blob: two embedding layers + the ENC0L layer block (walk asserted)."""

    def __init__(self, blob, deperm=True):
        b = np.frombuffer(blob, np.uint8)
        if len(b) != 30464:
            raise SystemExit(f"enc0 blob {len(b)} != 30464")
        self.w1 = LUT[b[4096:5120][frag_offsets(32, 32)]]        # (32 out, 32 in), cols 20..31 = 0
        self.b1 = vec(b, 5120, 32)
        self.w2 = LUT[b[5184:6208][frag_offsets(32, 32)]]
        self.b2 = vec(b, 6208, 32)
        self.layer = sm.StageWeights(b[6272:], 32, 2, 64, 0, deperm)


class Dec0L:
    """The launch-14 weight blob = pack_layer(DEC0L): the mode-2 walk starts with up_W/up_b."""

    def __init__(self, blob, deperm=True):
        b = np.frombuffer(blob, np.uint8)
        if len(b) != 24320:
            raise SystemExit(f"dec0 blob {len(b)} != 24320")
        self.layer = sm.StageWeights(b, 32, 2, 64, 2, deperm)


# ------------------------------------------------------------------ launch 3: l0_in -> t0
def embed(X, w):
    """The enc0L embedding over the padded buffer (ISA 0x214..0x7f8; SPEC above)."""
    a = X @ w.w1.T
    e1 = q8(np.clip(f16(a + w.b1), np.float16(0.0), np.float16(448.0)))
    return f16(e1 @ w.w2.T + w.b2)


def run_block_padded(E, Wt, W, H, sx, sy, tile, cfg, offx=2, offy=2):
    """sm.run_stage's block with the border read from the padded buffer E instead of reflect():
    token (tx, ty) reads E[ty + offy, tx + offx]. The offset is the kernel's constant +2
    (l0_0.s: the load address immediate 64 = 2 px on x, `v_add_nc_u32 2` on y; the level-0
    content sits at buffer (2,2) for EVERY frame - frame 1's shift is (0,2), so the offset is
    NOT the shift). Mode 0 only (launch 3). The in-range gating and every rounding point are
    sm.run_stage's (the Review R9 closure), unchanged."""
    if Wt.mode != 0:
        raise SystemExit("run_block_padded: mode 0 only")
    orig = sm.reflect

    def padded(p, n):
        return np.asarray(p) + (offx if n == W else offy)

    sm.reflect = padded
    try:
        img, t2 = sm.run_stage(E, Wt, W, H, sx, sy, tile, cfg)
    finally:
        sm.reflect = orig
    return img, t2


def check_window(W, H, sx, sy, tile, offx, offy, Ebw, Ebh):
    """Every window token of the launch must land inside the padded buffer."""
    gx = (W + sx + tile - 1) // tile
    gy = (H + sy + tile - 1) // tile
    ox = np.arange(gx * tile // 4) * 4 - sx
    oy = np.arange(gy * tile // 4) * 4 - sy
    tx = (ox[None, :, None, None] + np.arange(4)[None, None, None, :])
    ty = (oy[:, None, None, None] + np.arange(4)[None, None, :, None])
    TX = np.broadcast_to(tx, (len(oy), len(ox), 4, 4))
    TY = np.broadcast_to(ty, (len(oy), len(ox), 4, 4))
    px, py = TX + offx, TY + offy
    if px.min() < 0 or py.min() < 0 or px.max() >= Ebw or py.max() >= Ebh:
        raise SystemExit(f"window tokens out of buffer: x [{px.min()},{px.max()}] "
                         f"y [{py.min()},{py.max()}] vs {Ebw}x{Ebh}")


def model_l0_0(dump, frame, wdir):
    """enc0L: l0_in -> embedding -> block -> down projection. Returns (launch, {name: bytes})."""
    fdir = Path(dump) / f"frame{frame}"
    d, a, ptr = load_launch(fdir, KERNEL0)
    n = d["n"]
    W0, H0, sx, sy, Wr, Hr = dims(a)
    w = Enc0L((Path(wdir) / "enc0.bin").read_bytes())
    raw = sm.buffer_before(fdir, ptr[8], n)                     # l0_in, the oracle's own bytes
    if len(raw) != (W0 + 8) * (H0 + 8) * 32:
        raise SystemExit(f"l0_in {len(raw)} != {(W0 + 8)}x{(H0 + 8)}x32")
    X = sm.decode_act(raw, 32, H0 + 8, W0 + 8)                  # (168, 264, 32) f32
    E = embed(X, w)                                             # f16-valued (168, 264, 32)
    check_window(W0, H0, sx, sy, 8, 2, 2, W0 + 8, H0 + 8)
    img, t2 = run_block_padded(E, w.layer, W0, H0, sx, sy, 8, dict(sm.DEFAULTS))
    if img.shape != (H0, W0, 32) or t2 is None or t2.shape != (H0 // 2, W0 // 2, 64):
        raise SystemExit(f"unexpected block shapes {img.shape} {t2.shape}")
    return n, {"t0": sm.encode_act(t2)}


# ------------------------------------------------------------------ launch 14: t8 -> heads
def heads_l0(p, s0, f0):
    """The four dec0-family heads with L's channel indices (found against the oracle, frame 0;
    the formulas are dec0's Review R3b, unchanged where the ISA agrees):
      gates   c = 0, 1, 6   (dec0: 0,1,10)  per-byte exact 99.89/99.88/99.94%
      factors c = 7, 8, 9   (dec0: 11,12,13)  log-ratio corr 0.999/0.946
      aniso   A = ch10, B = ch11, tanh(f0 * ch12)  (dec0: 14,15,16)  corr 0.988/0.9997/1.0
    Returns (gates u8x4, factors u8x4, aniso f16x4), byte/lane 3 = 0."""
    sh = p.shape[:2] + (4,)
    q255 = lambda s: np.rint(np.float32(255) * np.clip(s, 0, 1))
    gates = np.zeros(sh, np.float32)
    for i, c in enumerate((0, 1, 6)):
        gates[..., i] = q255(f16(sg(p[..., c])))
    x = p[..., 7:10]
    e = np.exp(x - x.max(-1, keepdims=True)).astype(np.float32)
    factors = np.zeros(sh, np.float32)
    factors[..., :3] = q255(e / e.sum(-1, keepdims=True, dtype=np.float32))
    A = np.float32(s0) * f16(sg(np.clip(p[..., 10], -20, 20)))
    B = np.float32(s0) * f16(sg(np.clip(p[..., 11], -20, 20)))
    T = np.tanh((np.float32(f0) * p[..., 12]).astype(np.float64)).astype(np.float32)
    aniso = np.zeros(sh, np.float32)
    aniso[..., 0] = f16(A * A)
    aniso[..., 1] = f16(B * B)
    aniso[..., 2] = f16(((A * B).astype(np.float32).astype(np.float64) * T).astype(np.float32))
    return gates.astype(np.uint8), factors.astype(np.uint8), aniso.astype(np.float16)


def hidden_l0(p):
    """hidden0_k = rndne(127 * clamp(tanh(p_(2+k))), k = 0..3 (channels 2,3,4,5) - the RR u15
    tanh-byte family, NOT dec0's GRU: L0Args carries no hin/mv pointers, so there is no warp
    term and no z gate. Per-channel exact vs the oracle: 99.89/99.91/99.86/99.89%."""
    out = np.zeros(p.shape[:2] + (4,), np.int8)
    for k in range(4):
        out[..., k] = np.rint(np.float32(127) * np.clip(np.tanh(p[..., 2 + k]), -1, 1)).astype(np.int8)
    return out


def model_l0_2(dump, frame, wdir):
    """dec0L: t8 -> up projection + skip -> block -> dec0 heads. Returns (launch, {name: bytes})."""
    fdir = Path(dump) / f"frame{frame}"
    d, a, ptr = load_launch(fdir, KERNEL2)
    n = d["n"]
    W0, H0, sx, sy, Wr, Hr = dims(a)
    s0, f0 = struct.unpack_from("<2f", a, 0x60)
    w = Dec0L((Path(wdir) / "dec0.bin").read_bytes()).layer
    t8 = sm.decode_act(sm.buffer_before(fdir, ptr[8], n), 64, H0 // 2, W0 // 2)
    skip = sm.decode_act(sm.buffer_before(fdir, ptr[16], n), 32, H0, W0)
    img, _ = sm.run_stage(None, w, W0, H0, sx, sy, 8, dict(sm.DEFAULTS), skip=skip, low=t8)
    p = img[:Hr, :Wr]                                           # (144, 256, 32) heads' input
    gates, factors, aniso = heads_l0(p, s0, f0)
    hid = hidden_l0(p)
    outs = {ptr[40]: gates.tobytes(), ptr[48]: factors.tobytes(),
            ptr[56]: aniso.tobytes(), ptr[64]: hid.tobytes()}
    return n, outs


# ------------------------------------------------------------------ scoring
def score(fdir, n, outs):
    res = {}
    for name, mb in outs.items():
        dt = DTYPE[name]
        orc = (Path(fdir) / f"{n:03d}.{name}.bin").read_bytes()
        r = measure(bytes(mb), orc, dt, None)
        if r is None:
            print(f"  {name:9s} {dt:3s} not scorable")
            continue
        cnt, peak, psnr, rr, rrmse, exact = r
        rs = "n/a" if rr is None else f"{rr:.6f}"
        print(f"  {name:9s} {dt:3s} peak {peak:9.4f}  PSNR {psnr:9.4f} dB  r {rs:>9s}  "
              f"rRMSE {rrmse:.6f}  bit-exact {100 * exact:8.4f}%")
        res[name] = dict(peak=peak, psnr=psnr, r=rr, rrmse=rrmse, exact=exact)
    return res


def run_one(args, launch_fn, label, frame):
    n, outs = launch_fn(args.oracle, frame, args.weights)
    print(f"{label} frame {frame}: launch {n}")
    if args.out:
        od = Path(args.out) / f"frame{frame}"
        od.mkdir(parents=True, exist_ok=True)
        for name, mb in outs.items():
            (od / f"{n:03d}.{name}.bin").write_bytes(mb)
    return score(Path(args.oracle) / f"frame{frame}", n, outs)


def self_check(args):
    t0 = time.time()
    ok = True
    for frame in (0, 1):
        res = run_one(args, model_l0_0, "l0<0,2>", frame)
        good = res.get("t0", {}).get("exact", 0) >= 0.99
        ok &= good
        print(f"  L1 t0 >= 99%: {'PASS' if good else 'FAIL'}")
    for frame in (0, 1):
        res = run_one(args, model_l0_2, "l0<2,2>", frame)
        good = all(res.get(k, {}).get("exact", 0) >= 0.97
                   for k in ("gates", "factors", "aniso", "hidden0"))
        ok &= good
        print(f"  L2 heads >= 97% each: {'PASS' if good else 'FAIL'}")
    dt = time.time() - t0
    print(f"self-check: {'PASS' if ok else 'FAIL'}  ({dt:.1f} s)")
    return 0 if ok and dt < 60 else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--oracle", default=str(ORACLE))
    ap.add_argument("--weights", default=str(WEIGHTS))
    ap.add_argument("--gpu", help="a GPU dump's frame directory, files NNN.<name>.bin")
    ap.add_argument("--out")
    ap.add_argument("--launch", type=int, choices=[3, 14], help="model one launch")
    ap.add_argument("--self-check", action="store_true")
    a = ap.parse_args()
    if a.self_check:
        return self_check(a)
    if a.launch == 3:
        run_one(a, model_l0_0, "l0<0,2>", a.frame)
    elif a.launch == 14:
        run_one(a, model_l0_2, "l0<2,2>", a.frame)
    else:
        run_one(a, model_l0_0, "l0<0,2>", a.frame)
        run_one(a, model_l0_2, "l0<2,2>", a.frame)
    return 0


if __name__ == "__main__":
    sys.exit(main())
