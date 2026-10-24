#!/usr/bin/env python3
"""Score rrref outputs (RGBA32F linear) against the picture the test frames were made from, next to
a bicubic upscale of the noisy colour and of the clean albedo (the best an upscaler that ignores the
noise could do). PSNR on 8-bit sRGB; writes out_i.png.

    eval_rr.py picture.png FRAMES_DIR OUT_DIR
"""
import json
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


ref = np.asarray(Image.open(sys.argv[1]).convert("RGB"))
frames, outs = Path(sys.argv[2]), Path(sys.argv[3])
H, W = ref.shape[:2]
fj = json.loads((frames / "frame.json").read_text())
rw, rh = fj["render"]
for i in range(fj["frames"]):
    o = outs / f"out_{i}.f32"
    if not o.exists():
        continue
    px = np.fromfile(o, np.float32).reshape(H, W, 4)
    got = to_srgb8(px[..., :3].astype(np.float64))
    def bic(name):
        c = np.fromfile(frames / f"{name}_{i}.f16", np.float16).reshape(rh, rw, 4)[..., :3].astype(np.float64)
        return np.asarray(Image.fromarray(to_srgb8(c)).resize((W, H), Image.BICUBIC))
    print(f"frame {i}: RR {psnr(got, ref):.2f} dB; bicubic of the noisy colour {psnr(bic('col'), ref):.2f} dB, "
          f"of the clean albedo {psnr(bic('alb'), ref):.2f} dB  (min {np.nanmin(px):.3g} max {np.nanmax(px):.3g})")
    Image.fromarray(got).save(outs / f"out_{i}.png")
