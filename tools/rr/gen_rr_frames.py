#!/usr/bin/env python3
"""Make a static test sequence for rrref: render-resolution frames sampled from a high-resolution
picture at Halton(2,3) jitter (NGX convention, as tools/sr/gen_frames.py), with the guides DLSS-RR
reads: noisy colour, clean diffuse albedo, constant specular albedo / normals / roughness / depth,
zero motion.

    gen_rr_frames.py picture.png OUT_DIR --render 1920 1080 --output 3840 2160 --frames 8 --noise 0.5

Colour is the picture (sRGB decoded to linear) times per-pixel noise of mean 1 (gamma distributed,
relative deviation --noise, a fresh draw each frame: a stand-in for a few-samples path tracer);
diffuse albedo is the clean picture, so the lighting the network has to recover is 1 everywhere and
the clean picture is the answer. Files per frame i: col_i.f16, alb_i.f16, spa_i.f16, nrm_i.f16
(RGBA16F), mv_i.f16 (RG16F), dep_i.f32, rgh_i.f32, params_i.json.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

import importlib.util

_spec = importlib.util.spec_from_file_location("gen_frames", Path(__file__).resolve().parent.parent / "sr" / "gen_frames.py")
_gf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_gf)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("picture", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--render", type=int, nargs=2, default=[1920, 1080])
    ap.add_argument("--output", type=int, nargs=2, default=[3840, 2160])
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--noise", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--path", action="store_true", help="simulate camera pan with motion vectors")
    ap.add_argument("--reset-frame", type=int, default=None, help="frame index with reset=1")
    a = ap.parse_args()
    src = _gf.srgb_to_linear(np.asarray(Image.open(a.picture).convert("RGB"), np.float64) / 255.0)
    H, W = src.shape[:2]
    rw, rh = a.render
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "frame.json").write_text(json.dumps(dict(render=a.render, output=a.output, frames=a.frames)))
    rng = np.random.default_rng(a.seed)
    sx, sy = W / rw, H / rh
    one = np.ones((rh, rw, 1))

    reset_frame = a.reset_frame
    if reset_frame is None and a.seed is not None:
        reset_frame = int(rng.integers(4, min(8, max(4, a.frames - 1))))

    pan_vx, pan_vy = (0.0, 0.0)
    if a.path:
        pan_vx = float(rng.uniform(-2.0, 2.0))
        pan_vy = float(rng.uniform(-1.0, 1.0))

    for i in range(a.frames):
        jx, jy = _gf.halton(i + 1, 2) - 0.5, _gf.halton(i + 1, 3) - 0.5
        ox = pan_vx * i
        oy = pan_vy * i
        xs = (np.arange(rw) + 0.5 - jx) * sx - 0.5 + ox
        ys = (np.arange(rh) + 0.5 - jy) * sy - 0.5 + oy
        X, Y = np.meshgrid(xs, ys)
        clean = _gf.bilinear(src, X, Y)
        if a.noise > 0:
            k = 1.0 / a.noise ** 2
            n = rng.gamma(k, 1.0 / k, size=(rh, rw, 1))
        else:
            n = one
        def f16(x, name):
            np.asarray(x, np.float16).tofile(a.out / f"{name}_{i}.f16")
        def u8(x, name):
            np.asarray(x, np.uint8).tofile(a.out / f"{name}_{i}.u8")
        def u32(x, name):
            np.asarray(x, np.uint32).tofile(a.out / f"{name}_{i}.u32")
        def u16(x, name):
            np.asarray(x, np.uint16).tofile(a.out / f"{name}_{i}.u16")

        noisy_col = np.concatenate([clean * n, one], axis=2)
        f16(noisy_col, "col")
        u8(np.clip(noisy_col * 255.0 + 0.5, 0, 255), "col")

        f16(np.concatenate([clean, one], axis=2), "alb")
        u8(np.clip(np.concatenate([clean, one], axis=2) * 255.0 + 0.5, 0, 255), "alb")

        f16(np.concatenate([np.full((rh, rw, 3), 0.04), one], axis=2), "spa")

        nrm = np.concatenate([np.zeros((rh, rw, 2)), one, np.zeros((rh, rw, 1))], axis=2)
        f16(nrm, "nrm")
        # RGBA16_UNORM normals (fmt_nrm = 14)
        u16(np.clip((nrm * 0.5 + 0.5) * 65535.0 + 0.5, 0, 65535), "nrm")

        mv = np.full((rh, rw, 2), [-pan_vx / sx, -pan_vy / sy], np.float16)
        f16(mv, "mv")

        dep = np.full((rh, rw), 0.5, np.float32)
        dep.tofile(a.out / f"dep_{i}.f32")
        u32(np.clip(dep * 16777215.0 + 0.5, 0, 16777215), "dep")
        u16(np.clip(dep * 65535.0 + 0.5, 0, 65535), "dep")

        np.full((rh, rw), 0.5, np.float32).tofile(a.out / f"rgh_{i}.f32")

        is_reset = bool(i == 0 or (reset_frame is not None and i == reset_frame))
        (a.out / f"params_{i}.json").write_text(json.dumps(dict(
            jitter=[jx, jy], mv_scale=[1.0, 1.0], reset=is_reset, depth_inverted=True, hw_depth=True)))


if __name__ == "__main__":
    main()
