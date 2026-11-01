"""tools/report/quality_guard.py: the optimisation phase's quality gate - judged against GROUND TRUTH.

    build/venv/bin/python -I tools/report/quality_guard.py [--presets M,L[,F]] [--cells seq,static]
        [--runner build/vk/vk_runner] [--work build/work/qguard] [--lock /home/deck/dlss5-wt/gpu.lock]

Runs vk_runner (DLSS5_AMD_SHADER_DIR is honoured as usual) on the referee's ground-truth cells and
compares the port with NeuralHub's own output (srref) on the same inputs:

  seq     Blender Cycles moving sequence, 960x540 -> 1920x1080, 12 frames (referee cell sr_<P>_seq)
  static  one picture under jitter/motion, 1920x1080 -> 3840x2160, 12 frames (sr_<P>_static)
Preset F = Ray Reconstruction (vk_runner --mode rr, oracle rrref): the referee's rr_F_seq / rr_F_static.

Bars, per cell (the port may be BETTER than NeuralHub against ground truth; it may not be worse):
  G1  mean over frames of PSNR(port, GT) - PSNR(ref, GT)  >= -0.20 dB
  G2  every frame        PSNR(port, GT) - PSNR(ref, GT)  >= -0.50 dB
  T1  temporal: mean over t>=1 of tPSNR(port) - tPSNR(ref) >= -0.20 dB, where tPSNR(x) compares the
      frame-to-frame change of x, tm(x_t) - tm(x_t-1), with ground truth's (flicker / ghosting /
      shimmer that per-frame PSNR does not see)
  B1  backstop: every frame PSNR(port, ref) >= 41.0 dB (see the anchor below)

WHICH METRIC IS WHICH - do not conflate this script's bars with the referee's V4 bars:
  * B1 HERE measures PSNR(port, NeuralHub-oracle OUTPUT) - how far the port's frame drifted from
    what NeuralHub's own srref rendered on the identical inputs.  It is a REGRESSION FLOOR for the
    optimisation phase ("the look still matches the oracle frame-for-frame"), deliberately loose.
    It has NO ground-truth term and it is NOT the V4 bar.  It does share V4's FIRST term (per-frame
    PSNR against the oracle output) but at a deliberately LOOSER threshold - 41.0 < 50.0 - and
    without V4's mean or GT-delta terms, so a green B1 is never a V4 pass; that is intentional and
    is exactly the gap the doc block below exists to name.
  * V4 is the frozen referee's quality bar and the ONLY authority for 50/55/0.1:
    tools/verify/eval_verify.py:30 (READ ONLY - imported here, never modified) = per frame
    PSNR(port, oracle output) >= 50.0 dB AND mean >= 55.0 dB AND, with ground truth,
    |PSNR(port, GT) - PSNR(oracle, GT)| <= 0.1 dB on EVERY frame, no non-finite value.  Those numbers
    gate tools/verify's own cells (run_all.sh); this script does not restate them as its own bars and
    must never be read as if it enforced them.  Coverage-wise the 1.5x/2.0x/3.0x + UV-MV matrix gate
    is tools/report/psnr3x_gate.py, which applies V4's exact numbers to the cells tools/verify never
    built.
  * G1/G2 HERE compare the SIGNED GT delta against -0.20/-0.50 dB - the optimisation phase's
    "no further loss vs the phase-start baseline" rule (KNOWN_GAP below), NOT V4's 0.1 dB absolute
    delta.

B1 anchor (measured on HEAD BEFORE this bar was raised, 2026-11-01:
`build/venv/bin/python -I tools/report/quality_guard.py`):
    sr_M_seq 41.85 | sr_M_static 57.13 | sr_L_static 55.42 | sr_L_seq 33.12
  sr_L_seq already failed G1/G2/T1 in that measurement (an accuracy item, not an optimisation
  regression), so it did not set the bar.  Measured minimum of the cells that pass every other bar
  = 41.85 dB; chosen bar = 41.0 dB, i.e. margin 0.85 dB BELOW that measured minimum (every healthy
  cell keeps passing, and a >0.85 dB loss on the worst cell trips B1).  The old 38.0 dB bar sat
  3.85 dB under that measured minimum, so the old guard could lose 3.85 dB on its worst healthy
  cell - a large regression - and still print `B1:ok`.
AFTER-RAISE run, MEASURED FRESH 2026-11-01 15:13-15:28 (same command, whole run to completion,
tree state = HEAD incl. 80aa2f7, shader spv 15:11-15:12 / vk_runner 15:12, i.e. the run read one
consistent build):
    sr_M_seq 41.91 | sr_M_static 57.13 | sr_L_seq 42.12 | sr_L_static 55.42
  -> every cell PASS (G1/G2/T1/B1 all ok), QUALITY GUARD: PASS, worst cell 41.91 vs bar 41.0
  (margin 0.91 dB).  Per-cell detail of that run:
    sr_M_seq    G1 +2.841  G2 +0.403  T1 +2.935  B1 41.91
    sr_M_static G1 -0.042  G2 -0.106  T1 -0.036  B1 57.13
    sr_L_seq    G1 +2.557  G2 +0.306  T1 +2.722  B1 42.12
    sr_L_static G1 +0.014  G2 -0.082  T1 +0.010  B1 55.42
  CORRECTION to the numbers an earlier draft of this docstring carried (41.85 / 57.13 / 42.11 /
  55.42): those were a run made before 80aa2f7 reached the shared tree; this run is the measured
  reality and is what the bar is reported against.  Two cells reproduce to the digit (57.13,
  55.42), the two seq cells move by +0.06 and +0.01 dB.  The bar was NOT moved for any of it: it
  stays 41.0, anchored to the pre-raise measurement above (41.85), and the fresh worst healthy
  cell (41.91) sits 0.91 dB over it.  sr_L_seq also recovered 33.12 -> 42.12 across the same span
  (G1 -1.983 -> +2.557) - that is another session's accuracy fix landing in the tree, not this
  bar; it never moved the anchor either.
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
CELLS = {  # name -> (input dir, gt dir, oracle output dir template); preset F (RR) uses the rr dirs
    "seq": ("seq/sr", "seq/gt", "out_ref/sr_{P}_seq"),
    "static": ("base/sr", "base/sr_gt", "out_ref/sr_{P}_static"),
}
RR_CELLS = {  # preset F = Ray Reconstruction, oracle = rrref (referee cells rr_F_seq / rr_F_static)
    "seq": ("seq/rr", "seq/gt", "out_ref/rr_F_seq"),
    "static": ("base/rr", "base/rr_gt", "out_ref/rr_F_static"),
}
# B1 = 41.0: anchored 0.85 dB under the measured HEAD minimum 41.85 (sr_M_seq, the worst of the
# cells that pass every other bar - the docstring's anchor block lists both measurements, the
# margins and the fresh 2026-11-01 confirmation run, whose worst healthy cell is 41.91, 0.91 dB
# over this bar).  38.0 (the bar this commit replaced) sat 3.85 dB under that anchor, so it could
# lose 3.85 dB on the worst healthy cell and still print B1:ok.
G1, G2, T1, B1 = -0.20, -0.50, -0.20, 41.0
# Cells that already miss the bars at the start of the optimisation phase: there the bars are relative
# to that baseline (no further loss), and the gap itself is an open accuracy item, not an optimisation
# regression. rr_F_seq at main fb78207 (2026-10-28): G1 mean -1.545, G2 min -2.712, T1 mean -1.517;
# tightened after the RR accuracy fix (gather formats, per-pixel bev, native mv scale) to the new baseline.
KNOWN_GAP = {"rr_F_seq": (-0.394, -0.613, -0.403)}


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


CACHE_DIR = "/home/deck/dlss5-wt/guardcache"


def _hash_files(h, paths):
    for fp in paths:
        h.update(fp.encode() + b"\0")
        with open(fp, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)


def cache_key(runner, preset, cell, inp, gt, ref):
    """Content key of one guard cell: the exact runner binary and shader binaries it loads, the weights,
    the scoring script, the cell's data identity and the DLSS5_* environment. Identical key = identical
    (deterministic) outputs, so the scored rows can be reused. Any miss just means a normal run."""
    import hashlib
    h = hashlib.sha256()
    runner = os.path.realpath(runner)
    shd = os.environ.get("DLSS5_AMD_SHADER_DIR") or os.path.join(os.path.dirname(os.path.dirname(runner)), "shaders")
    files = [runner, os.path.realpath(__file__)]
    if os.path.isdir(shd):
        files += sorted(os.path.join(shd, f) for f in os.listdir(shd))
    wdir = os.path.join(ROOT, "build", "ngx-weights", "rr" if preset == "F" else preset)
    if os.path.isdir(wdir):
        for f in sorted(os.listdir(wdir)):
            st = os.stat(os.path.join(wdir, f))
            h.update(f"{f}:{st.st_size}:{int(st.st_mtime)}".encode())
    _hash_files(h, files)
    h.update(f"{preset}|{cell}|{inp}|{gt}|{ref}".encode())
    for k in sorted(os.environ):
        if k.startswith("DLSS5_") and k != "DLSS5_AMD_SHADER_DIR":   # the dir is hashed by content above
            h.update(f"{k}={os.environ[k]}".encode())
    return h.hexdigest()[:32]


def cache_get(key):
    import json
    try:
        with open(os.path.join(CACHE_DIR, key + ".json")) as f:
            return [tuple(r) for r in json.load(f)["rows"]]
    except Exception:
        return None


def cache_put(key, rows):
    import json, time
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        tmp = os.path.join(CACHE_DIR, f".{key}.{os.getpid()}.tmp")
        with open(tmp, "w") as f:
            json.dump({"rows": rows, "time": time.strftime("%F %T")}, f)
        os.replace(tmp, os.path.join(CACHE_DIR, key + ".json"))
    except Exception:
        pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--presets", default="M,L")
    ap.add_argument("--cells", default="seq,static")
    ap.add_argument("--runner", default=os.path.join(ROOT, "build", "vk", "vk_runner"))
    ap.add_argument("--work", default=os.path.join(ROOT, "build", "work", "qguard"))
    ap.add_argument("--lock", default="/home/deck/dlss5-wt/gpu.lock")
    ap.add_argument("--no-run", action="store_true", help="score existing outputs in --work")
    ap.add_argument("--no-cache", action="store_true",
                    help="ignore and do not write the result cache (default: a cell whose runner+shaders+weights+scorer are "
                         "byte-identical to an earlier run reuses that run's scored rows from " + CACHE_DIR + ")")
    ap.add_argument("--keep", action="store_true",
                    help="keep the out_<i>.f32 frames after scoring (default: delete them - a run is ~3 GB)")
    a = ap.parse_args()
    ok_all = True
    for P in a.presets.split(","):
        for c in a.cells.split(","):
            rr = P == "F"
            inp, gt, ref = (os.path.join(DATA, x.format(P=P)) for x in (RR_CELLS if rr else CELLS)[c])
            out = os.path.join(a.work, f"{'rr' if rr else 'sr'}_{P}_{c}")
            os.makedirs(out, exist_ok=True)
            key = None if (a.no_cache or a.no_run) else cache_key(a.runner, P, c, inp, gt, ref)
            rows = cache_get(key) if key else None
            if rows is not None:
                print(f"{os.path.basename(out)}: cached result (key {key}, identical runner+shaders+weights+scorer)")
            if rows is None and not a.no_run:
                cmd = ([a.runner, "--mode", "rr", "--weights", os.path.join(ROOT, "build", "ngx-weights", "rr")] if rr else
                       [a.runner, "--mode", "sr", "--preset", P, "--weights", os.path.join(ROOT, "build", "ngx-weights", P)]) + \
                      ["--in", inp, "--out", out]
                # correctness runs share the GPU through gpushare.sh (up to 3 at once; timing runs keep
                # it exclusive); without the helper, fall back to the exclusive lock
                share = os.path.join(os.path.dirname(a.lock or ""), "gpushare.sh")
                if a.lock and os.path.exists(share):
                    cmd = [share] + cmd
                elif a.lock and os.path.exists(a.lock):
                    cmd = ["flock", a.lock] + cmd
                with open(out + ".log", "w") as lf:
                    if subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT).returncode:
                        print(f"{os.path.basename(out)}: vk_runner FAILED, see {out}.log")
                        ok_all = False
                        continue
            if rows is None:
                rows = score(out, ref, gt)
                if key:
                    cache_put(key, rows)
            if not a.keep and not a.no_run:
                for f in os.listdir(out):
                    if f.startswith("out_") and f.endswith(".f32"):
                        os.remove(os.path.join(out, f))
            d = [pg - rg for _, _, pg, rg, _, _ in rows]
            td = [tp - tr for *_, tp, tr in rows if tp is not None]
            mn_pr = min(r[1] for r in rows)
            g1, g2, t1 = G1, G2, T1
            gap = KNOWN_GAP.get(os.path.basename(out))
            if gap:
                g1, g2, t1 = gap[0] + G1, gap[1] + G2, gap[2] + T1
            bars = {"G1": sum(d) / len(d) >= g1, "G2": min(d) >= g2,
                    "T1": (sum(td) / len(td) >= t1) if td else True, "B1": mn_pr >= B1}
            ok = all(bars.values())
            ok_all &= ok
            print(f"=== {os.path.basename(out)}  {'PASS' if ok else 'FAIL'}  " + " ".join(f"{k}:{'ok' if v else 'FAIL'}" for k, v in bars.items()))
            print("  frame  vsNH dB   port/GT   NH/GT    dGT    tPSNR port/NH   dT")
            for i, pr, pg, rg, tp, tr in rows:
                t = f"{tp:7.2f} {tr:7.2f} {tp - tr:+6.2f}" if tp is not None else ""
                print(f"  {i:5d} {pr:8.2f} {pg:9.3f} {rg:8.3f} {pg - rg:+7.3f}   {t}")
            print(f"  G1 mean dGT {sum(d) / len(d):+.3f} (>= {g1:.3f})  G2 min dGT {min(d):+.3f} (>= {g2:.3f})  "
                  f"T1 mean dT {(sum(td) / len(td) if td else 0):+.3f} (>= {t1:.3f})  B1 min vsNH {mn_pr:.2f} (>= {B1})"
                  + ("   [known accuracy gap: bars relative to the phase-start baseline]" if gap else ""))
    print(f"QUALITY GUARD: {'PASS' if ok_all else 'FAIL'}")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
