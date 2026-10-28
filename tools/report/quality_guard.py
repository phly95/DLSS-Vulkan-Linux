"""tools/report/quality_guard.py: the optimisation phase's quality gate - judged against GROUND TRUTH.

    build/venv/bin/python -I tools/report/quality_guard.py [--presets M,L] [--cells seq,static]
        [--runner build/vk/vk_runner] [--work build/work/qguard] [--lock /home/deck/dlss5-wt/gpu.lock]

Runs vk_runner (DLSS5_AMD_SHADER_DIR is honoured as usual) on the referee's ground-truth cells and
compares the port with NeuralHub's own output (srref) on the same inputs:

  seq     Blender Cycles moving sequence, 960x540 -> 1920x1080, 12 frames (referee cell sr_<P>_seq)
  static  one picture under jitter/motion, 1920x1080 -> 3840x2160, 12 frames (sr_<P>_static)

Bars, per cell (the port may be BETTER than NeuralHub against ground truth; it may not be worse):
  G1  mean over frames of PSNR(port, GT) - PSNR(ref, GT)  >= -0.20 dB
  G2  every frame        PSNR(port, GT) - PSNR(ref, GT)  >= -0.50 dB
  T1  temporal: mean over t>=1 of tPSNR(port) - tPSNR(ref) >= -0.20 dB, where tPSNR(x) compares the
      frame-to-frame change of x, tm(x_t) - tm(x_t-1), with ground truth's (flicker / ghosting /
      shimmer that per-frame PSNR does not see)
  B1  backstop: every frame PSNR(port, ref) >= 38.0 dB (the look stays NeuralHub's; HEAD sits at 41.7
      on seq, where the port already deviates - see the commit that added this file)
PSNR is tools/verify/eval_verify.py's (tone-mapped x/(1+x), RGB), imported, not copied.
Data: build/qguard/ (symlinks into the referee run build/verify/run_20261028_003547_1291489, the
inputs, ground truth and srref outputs of that run). Exit 0 = every bar of every cell passes.
"""
import argparse
import json
import math
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "tools", "verify"))
import eval_verify as ev  # noqa: E402

DATA = os.path.join(ROOT, "build", "qguard")
CELLS = {  # name -> (input dir, gt dir, srref output dir template)
    "seq": ("seq/sr", "seq/gt", "out_ref/sr_{P}_seq"),
    "static": ("base/sr", "base/sr_gt", "out_ref/sr_{P}_static"),
}
G1, G2, T1, B1 = -0.20, -0.50, -0.20, 38.0


def tpsnr(d, dg):
    mse = float(np.mean((d - dg) ** 2))
    return 120.0 if mse <= 1e-12 else -10.0 * math.log10(mse)


def score(port, ref, gt):
    ow, oh, n = ev.frame_info(ref)
    rows, prev = [], None
    for i in range(n):
        p = ev.load(os.path.join(port, f"out_{i}.f32"), ow, oh)
        r = ev.load(os.path.join(ref, f"out_{i}.f32"), ow, oh)
        g = ev.load(os.path.join(gt, f"gt_{i}.f32"), ow, oh)
        if p is None or r is None or g is None:
            raise SystemExit(f"FAIL: frame {i} missing (port {p is not None}, ref {r is not None}, gt {g is not None})")
        pr, nf = ev.psnr(p, r)
        if nf:
            raise SystemExit(f"FAIL: frame {i}: non-finite port output")
        pg, _ = ev.psnr(p, g)
        rg, _ = ev.psnr(r, g)
        tm = [ev.tone_map(x[..., :3].astype(np.float64)) for x in (p, r, g)]
        tp = tr = None
        if prev is not None:
            dg = tm[2] - prev[2]
            tp, tr = tpsnr(tm[0] - prev[0], dg), tpsnr(tm[1] - prev[1], dg)
        rows.append((i, pr, pg, rg, tp, tr))
        prev = tm
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--presets", default="M,L")
    ap.add_argument("--cells", default="seq,static")
    ap.add_argument("--runner", default=os.path.join(ROOT, "build", "vk", "vk_runner"))
    ap.add_argument("--work", default=os.path.join(ROOT, "build", "work", "qguard"))
    ap.add_argument("--lock", default="/home/deck/dlss5-wt/gpu.lock")
    ap.add_argument("--no-run", action="store_true", help="score existing outputs in --work")
    a = ap.parse_args()
    ok_all = True
    for P in a.presets.split(","):
        for c in a.cells.split(","):
            inp, gt, ref = (os.path.join(DATA, x.format(P=P)) for x in CELLS[c])
            out = os.path.join(a.work, f"sr_{P}_{c}")
            os.makedirs(out, exist_ok=True)
            if not a.no_run:
                cmd = [a.runner, "--mode", "sr", "--preset", P, "--weights",
                       os.path.join(ROOT, "build", "ngx-weights", P), "--in", inp, "--out", out]
                if a.lock and os.path.exists(a.lock):
                    cmd = ["flock", a.lock] + cmd
                with open(out + ".log", "w") as lf:
                    if subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT).returncode:
                        print(f"sr_{P}_{c}: vk_runner FAILED, see {out}.log")
                        ok_all = False
                        continue
            rows = score(out, ref, gt)
            d = [pg - rg for _, _, pg, rg, _, _ in rows]
            td = [tp - tr for *_, tp, tr in rows if tp is not None]
            mn_pr = min(r[1] for r in rows)
            bars = {"G1": sum(d) / len(d) >= G1, "G2": min(d) >= G2,
                    "T1": (sum(td) / len(td) >= T1) if td else True, "B1": mn_pr >= B1}
            ok = all(bars.values())
            ok_all &= ok
            print(f"=== sr_{P}_{c}  {'PASS' if ok else 'FAIL'}  " + " ".join(f"{k}:{'ok' if v else 'FAIL'}" for k, v in bars.items()))
            print("  frame  vsNH dB   port/GT   NH/GT    dGT    tPSNR port/NH   dT")
            for i, pr, pg, rg, tp, tr in rows:
                t = f"{tp:7.2f} {tr:7.2f} {tp - tr:+6.2f}" if tp is not None else ""
                print(f"  {i:5d} {pr:8.2f} {pg:9.3f} {rg:8.3f} {pg - rg:+7.3f}   {t}")
            print(f"  G1 mean dGT {sum(d) / len(d):+.3f} (>= {G1})  G2 min dGT {min(d):+.3f} (>= {G2})  "
                  f"T1 mean dT {(sum(td) / len(td) if td else 0):+.3f} (>= {T1})  B1 min vsNH {mn_pr:.2f} (>= {B1})")
    print(f"QUALITY GUARD: {'PASS' if ok_all else 'FAIL'}")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
