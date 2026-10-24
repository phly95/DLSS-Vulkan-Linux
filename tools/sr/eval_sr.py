#!/usr/bin/env python3
"""Score SR outputs (RGBA32F linear) against the high-resolution picture they were rendered from,
next to a bicubic upscale of the same frame. PSNR on 8-bit sRGB."""
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def to_srgb8(lin):
    lin = np.clip(lin, 0, 1)
    s = np.where(lin <= 0.0031308, lin * 12.92, 1.055 * lin ** (1 / 2.4) - 0.055)
    return np.clip(np.rint(s * 255), 0, 255).astype(np.uint8)


def psnr(a, b):
    m = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 10 * np.log10(255 ** 2 / m) if m else float("inf")


import json
frames, outs = Path(sys.argv[2]), Path(sys.argv[3])
fj = json.loads((frames / "frame.json").read_text())
rw, rh = fj["render"]
W, H = fj["output"]
gtdir = Path(sys.argv[1]) if Path(sys.argv[1]).is_dir() else None   # per-frame gt_i.f32 (tools/scenes)
if gtdir is None:
    ref = np.asarray(Image.open(sys.argv[1]).convert("RGB"))
for i in range(fj["frames"]):
    if gtdir is not None:
        ref = to_srgb8(np.fromfile(gtdir / f"gt_{i}.f32", np.float32).reshape(H, W, 4)[..., :3].astype(np.float64))
    o = outs / f"out_{i}.f32"
    if not o.exists():
        continue
    px = np.fromfile(o, np.float32).reshape(H, W, 4)
    got = to_srgb8(px[..., :3].astype(np.float64))
    col = np.fromfile(frames / f"col_{i}.f16", np.float16).reshape(rh, rw, 4)[..., :3].astype(np.float64)
    bic = np.asarray(Image.fromarray(to_srgb8(col)).resize((W, H), Image.BICUBIC))
    stats = f"nan {np.isnan(px).mean():.3f} min {np.nanmin(px):.3g} max {np.nanmax(px):.3g}"
    print(f"frame {i}: SR {psnr(got, ref):.2f} dB, bicubic {psnr(bic, ref):.2f} dB  ({stats})")
    Image.fromarray(got).save(outs / f"out_{i}.png")
