#!/usr/bin/env python3
"""The NR gate on this card: run the reference's offline runner on the single-frame inputs that
reference-nr-vulkan's NGX verification used, and compare the result with NVIDIA's own output for the
same input (docs/ngx-verification/310.8.0/single-frame-outputs/*_nvidia.png, made by the unmodified
nvngx_dlssnr.dll 310.8.0 on an RTX 5090) and with the reference's published AMD output.

Needs the real model (`extract_model.sh` on the user's nvngx_dlssnr.dll 310.8.0; the run fails on
synthetic weights by design) and numpy + Pillow (build/venv). Writes the RGBA32F blobs and PNGs to
--work.

    build/venv/bin/python tools/hwtest/nr_gate.py --runner build/hwtest/nr_graph \
        --plan-tool build/hwtest/nr_plan --model build/model/dlssnr.bin --work build/hwtest/gate
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
REF = ROOT / "reference-nr-vulkan"
VER = REF / "docs/ngx-verification/310.8.0"
SIZES = ("1920x1080", "2560x1440", "3840x2160")


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runner", required=True, type=Path)
    ap.add_argument("--plan-tool", required=True, type=Path)
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--work", required=True, type=Path)
    ap.add_argument("--sizes", nargs="*", default=list(SIZES))
    ap.add_argument("--extra", action="append", default=[], help="extra runner argument (repeat)")
    a = ap.parse_args()
    a.work.mkdir(parents=True, exist_ok=True)
    results = {}
    for size in a.sizes:
        w, h = map(int, size.split("x"))
        src = np.asarray(Image.open(VER / f"single-frame-inputs/{size}.png").convert("RGB"))
        rgba = np.ones((h, w, 4), np.float32)
        rgba[..., :3] = src.astype(np.float32) / 255.0
        blob = a.work / f"{size}.in.f32"
        rgba.tofile(blob)
        plan = a.work / f"plan-{size}.txt"
        plan.write_text(subprocess.run([str(a.plan_tool), str(w), str(h)], check=True,
                                       capture_output=True, text=True).stdout)
        out = a.work / f"{size}.out.f32"
        cmd = [str(a.runner.resolve()), "--spv-dir", "build/linux/rdna4/network",
               "--model-pack", str(a.model.resolve()), "--plan", str(plan.resolve()),
               "--source-width", str(w), "--source-height", str(h),
               "--in-image", str(blob.resolve()), "--out-image", str(out.resolve()),
               "--host-boundary", "--no-reuse", *a.extra]
        r = subprocess.run(cmd, cwd=REF, capture_output=True, text=True)
        (a.work / f"{size}.log").write_text(r.stdout + r.stderr)
        if r.returncode != 0:
            print(f"{size}: runner failed (exit {r.returncode}), see {a.work / f'{size}.log'}")
            print("\n".join((r.stdout + r.stderr).splitlines()[-10:]))
            return 1
        px = np.fromfile(out, np.float32)
        oh = px.size // (4 * w)
        got = np.clip(np.rint(px.reshape(oh, w, 4)[:h, :, :3] * 255.0), 0, 255).astype(np.uint8)
        Image.fromarray(got).save(a.work / f"{size}_port.png")
        nv = np.asarray(Image.open(VER / f"single-frame-outputs/{size}_nvidia.png").convert("RGB"))
        amd = np.asarray(Image.open(VER / f"single-frame-outputs/{size}_dlssnr-amd.png").convert("RGB"))
        results[size] = dict(vs_nvidia=psnr(got, nv), vs_reference_amd=psnr(got, amd),
                             input_vs_nvidia=psnr(src, nv), reference_amd_vs_nvidia=psnr(amd, nv),
                             identical_to_reference_amd=bool(np.array_equal(got, amd)))
        print(size, json.dumps(results[size]), flush=True)
    (a.work / "results.json").write_text(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
