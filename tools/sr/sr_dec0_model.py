#!/usr/bin/env python3
"""tools/sr/sr_dec0_model.py: the NumPy model of NeuralHub's k_sr_dec0 (preset M) - the spec.

    build/venv/bin/python -I tools/sr/sr_dec0_model.py [--frame F] [--oracle DUMP_DIR]
        [--weights PACKED_DIR] [--gpu GPU_DUMP_FRAME_DIR] [--out DIR]

One frame's dec0 launch of the oracle's dump is modelled from the oracle's own inputs (t8, the
previous hidden state and prev_mvec as they stood before the launch, and the launch's args). The
outputs are scored bit-exact against the oracle; --gpu also scores a GPU dump's files with the same
names; --out writes the model's outputs in the stored layout.

SPEC (Review R3 + R3b, docs/notes/k_sr_dec0.md; f16(.) = round to f16, fp8(.) = round to E4M3):

Projection. dec0.bin = frag(128 x 64) + 128 P-permuted f16 biases (sr_pack.build). For every t8
  pixel, u = f16(f16(b + W[:, 0:32] . t8[0:32]) + W[:, 32:64] . t8[32:64]) (each half one f32 WMMA
  chain), then p = fp8(clamp(u, +-448)). Channel q*32 + c goes to sub-pixel q = dx + 2*dy of the
  2x upsampled grid (W x H, the render resolution). Only channels 0..17 are read back.
Heads, per pixel (sg(x) = 1 / (1 + exp(-x)) in f32 with an IEEE divide):
  gates   = rndne(255 * clamp01(f16(sg(p_c)))) for c = 0, 1, 10; byte 3 = 0
  factors = rndne(255 * clamp01(e_c / (e_11 + e_12 + e_13))), e_c = exp(p_c - max) in f32,
            c = 11, 12, 13; byte 3 = 0
  aniso   = A = s0 * f16(sg(clamp(p14, +-20))), B = s0 * f16(sg(clamp(p15, +-20))) (f32 products,
            s0 = args +0x60), stored f16(A*A), f16(B*B), f16((A*B) * tanh(f0 * p16)) with tanh the
            f32 tanhf of the f32 product (f0 = args +0x64); element 3 = 0
Hidden state (4 x int8, value / 127), k = 0..3:
  th(a) = copysign(f16((1 - e) / (1 + e)), a), e = exp(-2 |a|)
  z = f16(0.5 + f16(0.5 * th(f16(0.5 * p_(6+k)))))         (= sigmoid(p_(6+k)))
  h = f16(f16(z * th(p_(2+k))) + f16(f16(1 - z) * warp_k))
  out_k = rndne(127 * clamp(h, -1, 1))
  warp = the previous hidden state (args +0x10) bilinearly resampled at the reprojected position:
    px = fma((x + 0.5) - F20x, r15w, mv.x), py likewise (mv = f16 prev_mvec at the clamped pixel,
    args +0x08; F20 = +0x68, r15 = +0x78), all f32;
    valid = 0 <= px <= W15 and 0 <= py <= H15 and args +0x88 == 0 (else warp = 0);
    u = fma(px, 1/W15, F28x/W) (+0x8c, +0x94; y: +0x90, +0x98), xs = ((u * W) * (1/W)) * W - 0.5
    (f32, each step rounded), clamped to [-2, W + 2], snapped to floor(256 xs + 0.5) / 256;
    x0 = floor(xs), fx = xs - x0; taps x0 and min(W-1, max(-1, x0) + 1), x0 clamped to [0, W-1]
    (same for y); taps dequantized as max(-1, i8 * f32(1/127));
    warp = f16(((w00 t00 + w01 t01) + w10 t10) + w11 t11) in f32, w00 = (1-fx)(1-fy),
    w01 = fx(1-fy), w10 = (1-fx)fy, w11 = fx fy.
  The kernel reads hidden[p] and writes hidden[p ^ 1] (frame parity p).

Measured (oracle frames 0 and 1): gates, factors and hidden 99.997-100% bit-exact; aniso[0..1]
99.99-100%, aniso[2] 99.97% (the residue is the kernel's tanhf branch for |x| < ~2e-3 and near 1).
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
f16 = m.f16
L2E = f32(1.4426950408889634)
KERNEL = "_ZN2sr2io9k_sr_dec0ENS0_8Dec0ArgsE"


def project(t8, blob):
    """p (H0*2 x W0*2 x 32): the fp8 pre-activations, pixel-shuffled to the render grid."""
    Wd = m.LUT[blob[frag_offsets(128, 64)]]
    raw = np.frombuffer(blob[8192:8192 + 256].tobytes(), np.float16).astype(f32)
    bd = raw[(np.arange(128) // 32) * 32 + m.IP[np.arange(128) % 32]]
    a0 = t8[..., :32] @ Wd[:, :32].T
    a1 = t8[..., 32:] @ Wd[:, 32:].T
    u = m.q8(np.clip(f16(f16(bd + a0) + a1), -448, 448))
    h, w = t8.shape[:2]
    p = np.zeros((2 * h, 2 * w, 32), f32)
    for q, (dx, dy) in enumerate(((0, 0), (1, 0), (0, 1), (1, 1))):
        p[dy::2, dx::2, :] = u[:, :, q * 32:(q + 1) * 32]
    return p


def sg(x):
    return f32(1) / (f32(1) + np.exp(-x.astype(f32)).astype(f32))


def th(a):
    e = np.exp2(f32(-2) * L2E * np.abs(a)).astype(f32)
    return np.copysign(f16((f32(1) - e) * (f32(1) / (f32(1) + e))), a)


def heads(p, s0, f0):
    """(gates u8x4, factors u8x4, aniso f16x4) for the render grid."""
    sh = p.shape[:2] + (4,)
    q255 = lambda s: np.rint(f32(255) * np.clip(s, 0, 1))
    gates = np.zeros(sh, f32)
    for i, c in enumerate((0, 1, 10)):
        gates[..., i] = q255(f16(sg(p[..., c])))
    x = p[..., 11:14]
    e = np.exp(x - x.max(-1, keepdims=True)).astype(f32)
    factors = np.zeros(sh, f32)
    factors[..., :3] = q255(e / e.sum(-1, keepdims=True, dtype=f32))
    A = f32(s0) * f16(sg(np.clip(p[..., 14], -20, 20)))
    B = f32(s0) * f16(sg(np.clip(p[..., 15], -20, 20)))
    T = np.tanh((f32(f0) * p[..., 16]).astype(np.float64)).astype(f32)
    aniso = np.zeros(sh, f32)
    aniso[..., 0] = f16(A * A)
    aniso[..., 1] = f16(B * B)
    aniso[..., 2] = f16(((A * B).astype(f32).astype(np.float64) * T).astype(f32))
    return gates.astype(np.uint8), factors.astype(np.uint8), aniso.astype(np.float16)


def warp(hin, mv, a):
    """The previous hidden state (H x W x 4 int8) resampled at the reprojected positions (f32)."""
    H, W = hin.shape[:2]
    yy, xx = np.mgrid[0:H, 0:W].astype(f32)
    px = (((xx + f32(0.5)) - a["F20x"]).astype(np.float64) * a["r15w"] + mv[..., 0]).astype(f32)
    py = (((yy + f32(0.5)) - a["F20y"]).astype(np.float64) * a["r15h"] + mv[..., 1]).astype(f32)
    valid = (px >= 0) & (px <= f32(a["W15"])) & (py >= 0) & (py <= f32(a["H15"])) & (a["flag88"] == 0)

    def axis(pos, inv15, off, n, inv_n):
        c = (pos.astype(np.float64) * inv15 + off).astype(f32)
        c = ((c * f32(n)).astype(f32) * inv_n).astype(f32)
        c = (c * f32(n)).astype(f32) - f32(0.5)
        c = np.minimum(np.maximum(c, f32(-2)), f32(n + 2))
        c = (np.floor(f32(256) * c + f32(0.5)) * f32(1 / 256)).astype(f32)
        c0 = np.floor(c)
        fr = (c - c0).astype(f32)
        c0 = c0.astype(np.int64)
        return np.clip(c0, 0, n - 1), np.minimum(n - 1, np.maximum(-1, c0) + 1), fr

    xa, xb, fx = axis(px, a["i15w"], a["F28xW"], W, a["iW"])
    ya, yb, fy = axis(py, a["i15h"], a["F28yH"], H, a["iH"])
    dq = lambda t: np.maximum(f32(-1), t.astype(f32) * f32(1 / 127)).astype(f32)
    t00, t01, t10, t11 = dq(hin[ya, xa]), dq(hin[ya, xb]), dq(hin[yb, xa]), dq(hin[yb, xb])
    gx, gy = fx[..., None], fy[..., None]
    acc = ((f32(1) - gx) * (f32(1) - gy) * t00).astype(f32)
    acc = (acc + gx * (f32(1) - gy) * t01).astype(f32)
    acc = (acc + (f32(1) - gx) * gy * t10).astype(f32)
    w = f16(acc + gx * gy * t11)
    return np.where(valid[..., None], w, f32(0))


def hidden(p, hin, mv, a):
    wp = warp(hin, mv, a)
    out = np.zeros(p.shape[:2] + (4,), np.int8)
    for k in range(4):
        z = f16(f32(0.5) + f16(f32(0.5) * th(f16(f32(0.5) * p[..., 6 + k]))))
        h = f16(f16(z * th(p[..., 2 + k])) + f16(f16(f32(1) - z) * wp[..., k]))
        out[..., k] = np.rint(f32(127) * np.clip(h, -1, 1)).astype(np.int8)
    return out


def parse_args(b):
    W, H, W2, H2, W0, H0, W15, H15 = struct.unpack_from("<8i", b, 0x40)
    names = ("s0", "f0", "F20x", "F20y", "F28x", "F28y", "r15w", "r15h", "iW", "iH")
    a = dict(zip(names, (f32(v) for v in struct.unpack_from("<10f", b, 0x60))))
    a.update(dict(zip(("i15w", "i15h", "F28xW", "F28yH"), (f32(v) for v in struct.unpack_from("<4f", b, 0x8c)))))
    a.update(W=W, H=H, W0=W0, H0=H0, W15=W15, H15=H15, flag88=struct.unpack_from("<I", b, 0x88)[0])
    if (W2, H2) != (W, H):
        raise SystemExit(f"args +0x48 ({W2}, {H2}) != +0x40 ({W}, {H}): not modelled")
    return a


def model_frame(dump, frame, wdir):
    fdir = Path(dump) / f"frame{frame}"
    rows = [json.loads(l) for l in open(fdir / "launches.jsonl")]
    d = next(r for r in rows if r["kernel"] == KERNEL)
    n, ptr = d["n"], {int(k): v.split("+")[0] for k, v in d["pointers"].items()}
    a = parse_args(bytes.fromhex(d["args"]))
    H0, W0, H, W = a["H0"], a["W0"], a["H"], a["W"]
    t8 = m.decode_act(m.buffer_before(fdir, ptr[0], n), 64, H0 // 2, W0 // 2)
    blob = np.frombuffer((Path(wdir) / "dec0.bin").read_bytes(), np.uint8)
    p = project(t8, blob)[:H, :W]
    hin = np.frombuffer(m.buffer_before(fdir, ptr[16], n), np.int8)[:H * W * 4].reshape(H, W, 4)
    mv = np.frombuffer(m.buffer_before(fdir, ptr[8], n), np.float16)[:H * W * 2].reshape(H, W, 2).astype(f32)
    g, fac, an = heads(p, a["s0"], a["f0"])
    hid = hidden(p, hin, mv, a)
    outs = {ptr[24]: g.tobytes(), ptr[32]: fac.tobytes(), ptr[40]: an.tobytes(), ptr[48]: hid.tobytes()}
    return n, outs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--oracle", default=str(m.ORACLE / "sr_M/dump"))
    ap.add_argument("--weights", default=str(m.REPO / "build/re/packed/M"))
    ap.add_argument("--gpu", help="a GPU dump's frame directory, files NNN.<name>.bin")
    ap.add_argument("--out")
    a = ap.parse_args()
    n, outs = model_frame(a.oracle, a.frame, a.weights)
    elem = {"aniso": np.float16}
    fdir = Path(a.oracle) / f"frame{a.frame}"
    for name, mb in outs.items():
        dt = elem.get(name, np.uint8)
        mod = np.frombuffer(mb, dt)
        orc = np.frombuffer((fdir / f"{n:03d}.{name}.bin").read_bytes(), dt)[:mod.size]
        line = f"  {name:8s} model vs oracle bit-exact {100 * np.mean(mod.view(np.uint8 if dt == np.uint8 else np.uint16) == orc.view(np.uint8 if dt == np.uint8 else np.uint16)):8.4f}%"
        if a.gpu:
            g = np.frombuffer((Path(a.gpu) / f"{n:03d}.{name}.bin").read_bytes(), dt)[:mod.size]
            iv = np.uint8 if dt == np.uint8 else np.uint16
            line += f"   gpu vs model {100 * np.mean(g.view(iv) == mod.view(iv)):8.4f}%   gpu vs oracle {100 * np.mean(g.view(iv) == orc.view(iv)):8.4f}%"
        print(line)
        if a.out:
            Path(a.out).mkdir(parents=True, exist_ok=True)
            (Path(a.out) / f"{n:03d}.{name}.bin").write_bytes(mb)


if __name__ == "__main__":
    main()
