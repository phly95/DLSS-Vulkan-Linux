#!/usr/bin/env python3
"""Make a static test sequence for srref / the Vulkan SR: render-resolution frames sampled from a
high-resolution picture at Halton(2,3) jitter offsets (bilinear), zero motion, constant depth.

    gen_frames.py picture.png OUT_DIR --render 1920 1080 --output 3840 2160 --frames 8

Colour is linear (the PNG's sRGB decoded), RGBA16F. Jitter is NGX's `Jitter.Offset` in render
pixels: render pixel (x, y) shows the picture at ((x + 0.5 - jx) * sx, (y + 0.5 - jy) * sy).
NeuralHub's SR accumulates with this sign (39 dB after 8 frames on the Tomb Raider test picture)
and not with the opposite one (31 dB).
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def srgb_to_linear(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def halton(i, b):
    f, r = 1.0, 0.0
    while i > 0:
        f /= b
        r += f * (i % b)
        i //= b
    return r


def bilinear(img, x, y):
    h, w = img.shape[:2]
    x = np.clip(x, 0, w - 1)
    y = np.clip(y, 0, h - 1)
    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    fx = (x - x0)[..., None]
    fy = (y - y0)[..., None]
    return ((img[y0, x0] * (1 - fx) + img[y0, x1] * fx) * (1 - fy)
            + (img[y1, x0] * (1 - fx) + img[y1, x1] * fx) * fy)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("picture", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--render", type=int, nargs=2, default=[1920, 1080])
    ap.add_argument("--output", type=int, nargs=2, default=[3840, 2160])
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--depth", type=float, default=0.5)
    ap.add_argument("--depth-inverted", action="store_true")
    ap.add_argument("--report-jitter", type=float, default=1.0,
                    help="scale the jitter written to params (convention tests; 1 = correct)")
    ap.add_argument("--seed", type=int, default=None, help="seed for jitter/path generation")
    ap.add_argument("--path", action="store_true", help="simulate camera pan with motion vectors")
    ap.add_argument("--reset-frame", type=int, default=None, help="frame index with reset=1")
    a = ap.parse_args()
    src = srgb_to_linear(np.asarray(Image.open(a.picture).convert("RGB"), np.float64) / 255.0)
    H, W = src.shape[:2]
    rw, rh = a.render
    ow, oh = a.output
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "frame.json").write_text(json.dumps(dict(render=a.render, output=a.output, frames=a.frames)))
    sx, sy = W / rw, H / rh
    prev = (0.0, 0.0)

    rng = np.random.default_rng(a.seed) if a.seed is not None else None
    reset_frame = a.reset_frame
    if reset_frame is None and rng is not None:
        reset_frame = int(rng.integers(4, min(8, max(4, a.frames - 1))))

    pan_vx, pan_vy = (0.0, 0.0)
    if a.path:
        pan_vx = float(rng.uniform(-2.0, 2.0)) if rng is not None else 1.5
        pan_vy = float(rng.uniform(-1.0, 1.0)) if rng is not None else 0.5

    for i in range(a.frames):
        jx, jy = halton(i + 1, 2) - 0.5, halton(i + 1, 3) - 0.5
        ox = pan_vx * i
        oy = pan_vy * i
        xs = (np.arange(rw) + 0.5 - jx) * sx - 0.5 + ox
        ys = (np.arange(rh) + 0.5 - jy) * sy - 0.5 + oy
        X, Y = np.meshgrid(xs, ys)
        c = bilinear(src, X, Y)
        rgba = np.concatenate([c, np.ones((rh, rw, 1))], axis=2).astype(np.float16)
        rgba.tofile(a.out / f"col_{i}.f16")
        # R8G8B8A8_UNORM format (fmt_col = 5)
        np.clip(rgba * 255.0 + 0.5, 0, 255).astype(np.uint8).tofile(a.out / f"col_{i}.u8")

        dep = np.full((rh, rw), a.depth, np.float32)
        dep.tofile(a.out / f"dep_{i}.f32")
        # D24 format (fmt_dep = 16)
        (np.clip(dep * 16777215.0 + 0.5, 0, 16777215)).astype(np.uint32).tofile(a.out / f"dep_{i}.u32")
        # D16 format (fmt_dep = 9)
        (np.clip(dep * 65535.0 + 0.5, 0, 65535)).astype(np.uint16).tofile(a.out / f"dep_{i}.u16")

        # NGX: pixel vector pointing to previous frame position
        mv = np.full((rh, rw, 2), [-pan_vx / sx, -pan_vy / sy], np.float16)
        mv.tofile(a.out / f"mv_{i}.f16")

        # High-res MV at output resolution (for k_sr_gather_mv tests)
        mvhi = np.full((oh, ow, 2), [-pan_vx / (W / ow), -pan_vy / (H / oh)], np.float16)
        mvhi.tofile(a.out / f"mvhi_{i}.f16")

        is_reset = bool(i == 0 or (reset_frame is not None and i == reset_frame))
        (a.out / f"params_{i}.json").write_text(json.dumps(dict(
            jitter=[jx * a.report_jitter, jy * a.report_jitter], jitter_prev=[v * a.report_jitter for v in prev],
            mv_scale=[1.0, 1.0], exposure=1.0, pre_exposure=1.0,
            reset=is_reset, depth_inverted=a.depth_inverted)))
        prev = (jx, jy)


if __name__ == "__main__":
    main()
