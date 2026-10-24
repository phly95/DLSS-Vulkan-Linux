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
    a = ap.parse_args()
    src = srgb_to_linear(np.asarray(Image.open(a.picture).convert("RGB"), np.float64) / 255.0)
    H, W = src.shape[:2]
    rw, rh = a.render
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "frame.json").write_text(json.dumps(dict(render=a.render, output=a.output, frames=a.frames)))
    sx, sy = W / rw, H / rh
    prev = (0.0, 0.0)
    for i in range(a.frames):
        jx, jy = halton(i + 1, 2) - 0.5, halton(i + 1, 3) - 0.5
        xs = (np.arange(rw) + 0.5 - jx) * sx - 0.5
        ys = (np.arange(rh) + 0.5 - jy) * sy - 0.5
        X, Y = np.meshgrid(xs, ys)
        c = bilinear(src, X, Y)
        rgba = np.concatenate([c, np.ones((rh, rw, 1))], axis=2).astype(np.float16)
        rgba.tofile(a.out / f"col_{i}.f16")
        np.full((rh, rw), a.depth, np.float32).tofile(a.out / f"dep_{i}.f32")
        np.zeros((rh, rw, 2), np.float16).tofile(a.out / f"mv_{i}.f16")
        (a.out / f"params_{i}.json").write_text(json.dumps(dict(
            jitter=[jx * a.report_jitter, jy * a.report_jitter], jitter_prev=[v * a.report_jitter for v in prev], mv_scale=[1.0, 1.0], exposure=1.0, pre_exposure=1.0,
            reset=i == 0, depth_inverted=a.depth_inverted)))
        prev = (jx, jy)


if __name__ == "__main__":
    main()
