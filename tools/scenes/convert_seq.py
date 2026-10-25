# Turn render_seq.py's EXRs into the reference harnesses' inputs (runs in Blender's Python, which has
# OpenImageIO):
#
#   blender -b -P tools/scenes/convert_seq.py -- SEQ_DIR [--mv-sign 1 1]
#
# writes SEQ_DIR/rr (rrref frames: noisy colour + guides), SEQ_DIR/sr (srref frames: the clean
# aliased colour, depth, motion) and SEQ_DIR/gt/gt_i.f32 (RGBA32F at the output size, linear).
#
# Conventions written: motion vectors in render pixels pointing from the current pixel to where it was
# in the previous frame (NGX), y down; depth hardware reverse-Z (near / view depth, inverted); normals
# world space; roughness from the materials' Principled BSDF.
import json
import os
import sys

import numpy as np
import OpenImageIO as oiio


def read(path):
    i = oiio.ImageInput.open(path)
    if i is None:
        raise SystemExit(f"cannot read {path}: {oiio.geterror()}")
    names = list(i.spec().channelnames)
    px = i.read_image(format="float")
    i.close()
    def ch(layer, comps):
        return np.stack([px[..., names.index(f"ViewLayer.{layer}.{c}")] for c in comps], axis=-1)
    return ch


def main():
    argv = sys.argv[sys.argv.index("--") + 1:]
    seq = argv[0]
    sign = (1.0, 1.0)
    if "--mv-sign" in argv:
        k = argv.index("--mv-sign")
        sign = (float(argv[k + 1]), float(argv[k + 2]))
    meta = json.load(open(os.path.join(seq, "sequence.json")))
    n, near = meta["frames"], meta["near"]
    reset_frame = meta.get("reset_frame", -1)
    for d in ("rr", "sr", "gt"):
        os.makedirs(os.path.join(seq, d), exist_ok=True)
    fr = dict(render=meta["render"], output=meta["output"], frames=n)
    for d in ("rr", "sr", "gt"):
        json.dump(fr, open(os.path.join(seq, d, "frame.json"), "w"))
    prev = (0.0, 0.0)
    for i in range(n):
        noisy = read(os.path.join(seq, "exr", f"noisy_{i}.exr"))
        clean = read(os.path.join(seq, "exr", f"clean_{i}.exr"))
        gt_path = os.path.join(seq, "exr", f"gt_{i}.exr")
        gt = read(gt_path) if os.path.exists(gt_path) else None
        h, w = noisy("Combined", "R").shape[:2]
        one = np.ones((h, w, 1), np.float32)
        f16 = lambda x, d, name: np.asarray(x, np.float16).tofile(os.path.join(seq, d, f"{name}_{i}.f16"))
        f32 = lambda x, d, name: np.asarray(x, np.float32).tofile(os.path.join(seq, d, f"{name}_{i}.f32"))
        # Cycles' vector pass: pixels to the previous frame position in X, Y (y up)
        v = noisy("Vector", "XY")
        mv = np.stack([sign[0] * v[..., 0], -sign[1] * v[..., 1]], axis=-1)
        z = noisy("Depth", "Z")[..., 0]
        dep = np.where(z > 0, near / np.maximum(z, near), 0.0)
        f16(np.concatenate([noisy("Combined", "RGB"), one], axis=2), "rr", "col")
        f16(np.concatenate([clean("Combined", "RGB"), one], axis=2), "sr", "col")
        f16(np.concatenate([noisy("DiffCol", "RGB"), one], axis=2), "rr", "alb")
        f16(np.concatenate([noisy("GlossCol", "RGB"), one], axis=2), "rr", "spa")
        f16(np.concatenate([noisy("Normal", "XYZ"), np.zeros((h, w, 1))], axis=2), "rr", "nrm")
        for d in ("rr", "sr"):
            f16(mv, d, "mv")
            f32(dep, d, "dep")
        f32(noisy("rough", "X")[..., 0], "rr", "rgh")
        jx, jy = meta["jitter"][i]
        is_reset = bool(i == 0 or i == reset_frame)
        json.dump(dict(jitter=[jx, jy], mv_scale=[1.0, 1.0], reset=is_reset, depth_inverted=True, hw_depth=True),
                  open(os.path.join(seq, "rr", f"params_{i}.json"), "w"))
        json.dump(dict(jitter=[jx, jy], jitter_prev=list(prev), mv_scale=[1.0, 1.0], exposure=1.0, pre_exposure=1.0,
                       reset=is_reset, depth_inverted=True), open(os.path.join(seq, "sr", f"params_{i}.json"), "w"))
        prev = (jx, jy)
        if gt is not None:
            g = gt("Combined", "RGBA")
            g[..., 3] = 1.0
            f32(g, "gt", "gt")
        print(f"frame {i} converted (reset={is_reset})", flush=True)


main()
