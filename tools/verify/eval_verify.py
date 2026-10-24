#!/usr/bin/env python3
"""tools/verify/eval_verify.py: Accuracy and sensitivity evaluator for referee V4 and V5.

Calculates PSNR on x/(1+x) tone-mapped RGB with peak 1:
    tone_map(x) = max(x, 0) / (1.0 + max(x, 0))
    mse = mean((tone_map(port) - tone_map(ref)) ** 2)
    psnr = -10 * log10(mse)
"""
import argparse
import glob
import json
import math
import os
import sys
import numpy as np


def tone_map(rgb):
    x = np.maximum(rgb, 0.0)
    return x / (1.0 + x)


def compute_psnr(a, b):
    # Expects (H, W, 3) or (H, W, 4) float
    rgb_a = a[..., :3]
    rgb_b = b[..., :3]
    if not np.all(np.isfinite(rgb_a)) or not np.all(np.isfinite(rgb_b)):
        return float("-inf"), True
    ta = tone_map(rgb_a)
    tb = tone_map(rgb_b)
    diff = ta - tb
    mse = float(np.mean(diff ** 2))
    if mse <= 1e-12:
        return 999.0, False
    return -10.0 * math.log10(mse), False


def load_frame(path, w, h):
    raw = np.fromfile(path, dtype=np.float32)
    if raw.size == w * h * 4:
        return raw.reshape((h, w, 4))
    elif raw.size == w * h * 3:
        return raw.reshape((h, w, 3))
    else:
        raise ValueError(f"Size mismatch in {path}: got {raw.size} floats, expected {w}x{h}x4 or 3")


def read_frame_dims(dir_path):
    fj = os.path.join(dir_path, "frame.json")
    if os.path.exists(fj):
        data = json.load(open(fj))
        ow, oh = data["output"]
        return ow, oh, data["frames"]
    # Fallback: scan for out_*.f32
    files = sorted(glob.glob(os.path.join(dir_path, "out_*.f32")) or glob.glob(os.path.join(dir_path, "out*.f32")))
    if not files:
        raise ValueError(f"No output frames in {dir_path}")
    return None, None, len(files)


def check_cell(args):
    port_dir = args.port
    ref_dir = args.ref
    gt_dir = args.gt

    ow, oh, nframes = read_frame_dims(ref_dir)
    if ow is None and args.dims:
        ow, oh = args.dims

    print(f"--- Cell Evaluation: {args.name or port_dir} ({nframes} frames, {ow}x{oh}) ---")
    psnrs = []
    has_non_finite = False

    for i in range(nframes):
        p_path = os.path.join(port_dir, f"out_{i}.f32")
        if not os.path.exists(p_path):
            p_path = os.path.join(port_dir, f"out{i}.f32")
        r_path = os.path.join(ref_dir, f"out_{i}.f32")
        if not os.path.exists(r_path):
            r_path = os.path.join(ref_dir, f"out{i}.f32")

        if not os.path.exists(p_path) or not os.path.exists(r_path):
            print(f"Frame {i:2d}: MISSING (port={os.path.exists(p_path)}, ref={os.path.exists(r_path)})")
            return 1

        img_p = load_frame(p_path, ow, oh)
        img_r = load_frame(r_path, ow, oh)

        p_val, non_fin = compute_psnr(img_p, img_r)
        if non_fin:
            has_non_finite = True
            print(f"Frame {i:2d}: NON-FINITE (NaN or Inf detected)")
            psnrs.append(float("-inf"))
            continue

        gt_str = ""
        if gt_dir:
            gt_path = os.path.join(gt_dir, f"gt_{i}.f32")
            if os.path.exists(gt_path):
                img_gt = load_frame(gt_path, ow, oh)
                p_gt, _ = compute_psnr(img_p, img_gt)
                r_gt, _ = compute_psnr(img_r, img_gt)
                delta = abs(p_gt - r_gt)
                gt_str = f" | vs GT: port={p_gt:.2f} dB, ref={r_gt:.2f} dB (diff {delta:.3f} dB)"
                if delta > 0.1:
                    print(f"Frame {i:2d}: {p_val:6.2f} dB{gt_str} -> GT DIFF EXCEEDS 0.1 dB FAIL")
                    return 1

        status = "PASS" if p_val >= 50.0 else "FAIL (< 50 dB)"
        print(f"Frame {i:2d}: {p_val:6.2f} dB [{status}]{gt_str}")
        psnrs.append(p_val)

    if has_non_finite:
        print("CELL RESULT: FAIL (Non-finite values present)")
        return 1

    min_psnr = min(psnrs)
    mean_psnr = sum(psnrs) / len(psnrs)
    print(f"Summary: Min={min_psnr:.2f} dB (target >= 50 dB), Mean={mean_psnr:.2f} dB (target >= 55 dB)")

    if min_psnr < 50.0 or mean_psnr < 55.0:
        print("CELL RESULT: FAIL (Accuracy threshold not met)")
        return 1

    print("CELL RESULT: PASS")
    return 0


def check_sensitivity(args):
    port_dir = args.port
    ref_dir = args.ref
    bad_dir = args.bad

    ow, oh, nframes = read_frame_dims(ref_dir)
    if ow is None and args.dims:
        ow, oh = args.dims

    print(f"--- Sensitivity Check: {args.name} ---")
    p_scores = []
    b_scores = []

    for i in range(nframes):
        p_path = os.path.join(port_dir, f"out_{i}.f32")
        if not os.path.exists(p_path):
            p_path = os.path.join(port_dir, f"out{i}.f32")
        r_path = os.path.join(ref_dir, f"out_{i}.f32")
        if not os.path.exists(r_path):
            r_path = os.path.join(ref_dir, f"out{i}.f32")
        b_path = os.path.join(bad_dir, f"out_{i}.f32")
        if not os.path.exists(b_path):
            b_path = os.path.join(bad_dir, f"out{i}.f32")

        if not os.path.exists(p_path) or not os.path.exists(r_path) or not os.path.exists(b_path):
            print(f"Frame {i:2d}: Missing frame file")
            return 1

        img_p = load_frame(p_path, ow, oh)
        img_r = load_frame(r_path, ow, oh)
        img_b = load_frame(b_path, ow, oh)

        p_val, _ = compute_psnr(img_p, img_r)
        b_val, b_non_fin = compute_psnr(img_b, img_r)

        if b_non_fin:
            drop = 999.0
            b_val = 0.0
        else:
            drop = p_val - b_val

        p_scores.append(p_val)
        b_scores.append(b_val)
        print(f"Frame {i:2d}: Normal={p_val:6.2f} dB, Degraded={b_val:6.2f} dB (Drop={drop:6.2f} dB)")

    mean_p = sum(p_scores) / len(p_scores)
    mean_b = sum(b_scores) / len(b_scores)
    mean_drop = mean_p - mean_b
    print(f"Mean Normal: {mean_p:.2f} dB, Mean Degraded: {mean_b:.2f} dB, Mean Drop: {mean_drop:.2f} dB")

    # V5: Must fail V4 by at least 10 dB
    if mean_drop < 10.0 and mean_b >= 45.0:
        print("SENSITIVITY RESULT: FAIL (Did not fail by at least 10 dB)")
        return 1

    print("SENSITIVITY RESULT: PASS (Proven vulnerable to corruption/degradation)")
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    c1 = sub.add_parser("check-cell")
    c1.add_argument("port")
    c1.add_argument("ref")
    c1.add_argument("--gt", default=None)
    c1.add_argument("--dims", type=int, nargs=2, default=None)
    c1.add_argument("--name", default="")

    c2 = sub.add_parser("check-sensitivity")
    c2.add_argument("port")
    c2.add_argument("ref")
    c2.add_argument("bad")
    c2.add_argument("--dims", type=int, nargs=2, default=None)
    c2.add_argument("--name", default="")

    args = ap.parse_args()
    if args.cmd == "check-cell":
        sys.exit(check_cell(args))
    elif args.cmd == "check-sensitivity":
        sys.exit(check_sensitivity(args))


if __name__ == "__main__":
    main()
