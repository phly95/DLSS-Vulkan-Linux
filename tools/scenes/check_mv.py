#!/usr/bin/env python3
"""Check a converted sequence's motion vectors: warp the previous clean frame onto the current one with
them (NGX: the previous position of current pixel x is x + mv, and the jitter offsets of both frames
are undone) and report the error for each sign convention, without and with the jitter term. The
convention convert_seq.py writes should win clearly (row "+x +y, jitter").

    check_mv.py SEQ_DIR/sr
"""
import json
import sys
from pathlib import Path

import numpy as np


def bilinear(img, x, y):
    h, w = img.shape[:2]
    x = np.clip(x, 0, w - 1.001)
    y = np.clip(y, 0, h - 1.001)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = (x - x0)[..., None], (y - y0)[..., None]
    return ((img[y0, x0] * (1 - fx) + img[y0, x0 + 1] * fx) * (1 - fy)
            + (img[y0 + 1, x0] * (1 - fx) + img[y0 + 1, x0 + 1] * fx) * fy)


d = Path(sys.argv[1])
fj = json.loads((d / "frame.json").read_text())
w, h = fj["render"]
X, Y = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
for i in range(1, fj["frames"]):
    cur = np.fromfile(d / f"col_{i}.f16", np.float16).reshape(h, w, 4)[..., :3].astype(np.float64)
    prv = np.fromfile(d / f"col_{i - 1}.f16", np.float16).reshape(h, w, 4)[..., :3].astype(np.float64)
    mv = np.fromfile(d / f"mv_{i}.f16", np.float16).reshape(h, w, 2).astype(np.float64)
    jc = json.loads((d / f"params_{i}.json").read_text())["jitter"]
    jp = json.loads((d / f"params_{i - 1}.json").read_text())["jitter"]
    rows = []
    for sx in (1, -1):
        for sy in (1, -1):
            for jit in (0, 1):
                xs = X + sx * mv[..., 0] + jit * (jp[0] - jc[0])
                ys = Y + sy * mv[..., 1] + jit * (jp[1] - jc[1])
                err = np.mean(np.abs(bilinear(prv, xs, ys) - cur)[8:-8, 8:-8])
                rows.append((err, f"{'+' if sx > 0 else '-'}x {'+' if sy > 0 else '-'}y{', jitter' if jit else ''}"))
    zero = np.mean(np.abs(prv - cur)[8:-8, 8:-8])
    rows.sort()
    print(f"frame {i}: no warp {zero:.4f}; " + "; ".join(f"{n} {e:.4f}" for e, n in rows[:4]))
