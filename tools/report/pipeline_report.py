#!/usr/bin/env python3
"""tools/report/pipeline_report.py: the canonical whole-pipeline PSNR / GPU-time report.

    build/venv/bin/python -I tools/report/pipeline_report.py PORT_DIR ORACLE_DIR \
        [--frames DIR] [--log LOG]... [--name CELL]
    build/venv/bin/python -I tools/report/pipeline_report.py --self-test

PORT_DIR and ORACLE_DIR are the referee's cell layout (tools/verify/README.md): out_<i>.f32
RGBA32F per frame; the sizes come from frame.json, which the referee writes into the oracle
directory (run_all.sh out_ref copy), falls back to PORT_DIR, or is given with --frames.
--log may be repeated: the referee's port_<cell>.log / ref_<cell>.log or vk_runner's own output;
each is parsed for "GPU_FRAME_MS <i> <ms>" lines and reported as-is, "n/a" when absent.

Every headline whole-pipeline PSNR / GPU-time claim in docs/ once came from tooling that was never
committed -- grepping the tree's Python/C/shell for the headline strings found no producer, which is
how one sequence was recorded at 108.02, then 54.16, then 42.40, then 27.79 dB in a single day
(docs/OPEN.md "Whole-pipeline mismatch"). This script is the first committed place those numbers
come from; pipeline PSNR/timing rows are only trustworthy when produced by it
(docs/MEASUREMENT.md).

PSNR is the referee's, imported so it cannot drift: tone_map(x) = max(x,0)/(1+max(x,0)),
psnr = -10 log10(mean((tm(a) - tm(b))^2)) (tools/verify/eval_verify.py, lines 33-46, frozen by
rule R4 and never edited here -- importing runs only its definitions, main() is under
`if __name__ == "__main__"` at line 263).
"""
import argparse
import contextlib
import io
import json
import math
import os
import shlex
import statistics
import sys
import tempfile

try:
    import numpy as np
except ModuleNotFoundError:  # the repo's interpreter is build/venv/bin/python (HANDOFF.md setup)
    sys.stderr.write("pipeline_report: numpy is required -- run it with "
                     "build/venv/bin/python -I tools/report/pipeline_report.py ...\n")
    raise SystemExit(2)

# The referee is frozen (R4): read it, import it, never modify it.
_VERIFY = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "verify"))
if _VERIFY not in sys.path:
    sys.path.insert(0, _VERIFY)
import eval_verify  # noqa: E402  -- import only; eval_verify.main() is __main__-guarded

V4_MIN_DB, V4_MEAN_DB = eval_verify.V4_MIN_DB, eval_verify.V4_MEAN_DB


def find_frames_dir(port, oracle, frames):
    """The directory whose frame.json describes this cell (the referee puts it in the oracle dir)."""
    for d in (oracle, port, frames):
        if d and os.path.isfile(os.path.join(d, "frame.json")):
            return d
    return None


def measure(port, oracle, frames):
    """(w, h, n, [per-frame PSNR]) or None after saying why; PSNR exactly as eval_verify.psnr."""
    ow, oh, n = eval_verify.frame_info(frames)
    vals = []
    for i in range(n):
        p = eval_verify.load(os.path.join(port, f"out_{i}.f32"), ow, oh)
        r = eval_verify.load(os.path.join(oracle, f"out_{i}.f32"), ow, oh)
        if p is None or r is None:
            print(f"FAIL: frame {i}: port out_{i}.f32 present {p is not None}, "
                  f"oracle present {r is not None}")
            return None
        v, nonfinite = eval_verify.psnr(p, r)
        if nonfinite:
            print(f"FAIL: frame {i}: non-finite values in an input")
            return None
        vals.append(v)
    return ow, oh, n, vals


def read_log(path):
    """absolute path and {frame: ms} from a log's GPU_FRAME_MS lines; an empty dict when the file
    is missing or holds no such lines -- never invent a timing."""
    if not path:
        return None, {}
    ap = os.path.abspath(path)
    if not os.path.isfile(ap):
        return ap, {}
    return ap, eval_verify.frame_times(ap)


def print_log(path):
    ap, times = read_log(path)
    if ap is None:
        return
    print(f"  log    : {ap}")
    if not times:
        print("            n/a (no GPU_FRAME_MS lines)")
        return
    frames = sorted(times)
    contiguous = frames == list(range(len(frames)))
    # V8's convention (eval_verify.timing): exclude the warm-up frame 0 when the log is complete.
    window = frames[1:] if contiguous and len(frames) > 1 else (frames if not contiguous else [])
    print(f"            GPU_FRAME_MS frames {','.join(map(str, frames))} ({len(frames)})")
    if window:
        label = f"frames {window[0]}..{window[-1]}" if contiguous else "all found frames"
        print(f"            median ({label}): {statistics.median(times[i] for i in window):.4f} ms")
        print(f"            min    ({label}): {min(times[i] for i in window):.4f} ms")
    else:
        print("            median/min: n/a (only frame 0 found)")


def report(port, oracle, frames_arg, logs, name=""):
    port, oracle = os.path.abspath(port), os.path.abspath(oracle)
    frames = find_frames_dir(port, oracle, frames_arg and os.path.abspath(frames_arg))
    if frames is None:
        print(f"FAIL: no frame.json in {oracle} or {port}"
              + (f" or {frames_arg}" if frames_arg else "")
              + " (the referee copies it into out_ref/<cell>)")
        return 1
    got = measure(port, oracle, frames)
    if got is None:
        return 1
    ow, oh, n, vals = got

    print(f"--- pipeline report: {name or os.path.basename(port)} ---")
    print("inputs (absolute paths -- exactly what was measured):")
    iso = " -I" if getattr(sys.flags, "isolated", 0) else ""
    print(f"  command: {shlex.quote(sys.executable)}{iso} "
          + " ".join(shlex.quote(a) for a in sys.argv))
    print(f"  port   : {port}")
    print(f"  oracle : {oracle}")
    print(f"  frame  : {os.path.join(os.path.abspath(frames), 'frame.json')}")
    print(f"  shape  : {n} frames, {ow}x{oh} RGBA32F, out_<i>.f32")
    print("  psnr   : tools/verify/eval_verify.py tone-mapped PSNR (imported, not copied)")
    print("  frame   PSNR dB")
    for i, v in enumerate(vals):
        print(f"  {i:5d} {v:10.2f}")
    mn, mean, mx = min(vals), sum(vals) / len(vals), max(vals)
    print(f"  min {mn:.2f}  mean {mean:.2f}  max {mx:.2f}  ({len(vals)} frames)")
    ok = mn >= V4_MIN_DB and mean >= V4_MEAN_DB
    print(f"  V4 bar (eval_verify.py: min >= {V4_MIN_DB}, mean >= {V4_MEAN_DB}): "
          f"{'PASS' if ok else 'FAIL'}")
    if logs:
        print("timings (as found in the logs; n/a = not present, never estimated):")
        for lg in logs:
            print_log(lg)
    return 0


def _reference_psnr(a, b):
    """Independent transcription of the documented formula (tools/verify/eval_verify.py lines 15-17)
    so the self-test checks the imported implementation instead of reusing it."""
    a = a[..., :3].astype(np.float64)
    b = b[..., :3].astype(np.float64)
    ta, tb = np.clip(a, 0.0, None), np.clip(b, 0.0, None)
    mse = float(np.mean(np.square(ta / (1.0 + ta) - tb / (1.0 + tb))))
    return -10.0 * math.log10(mse)


def self_test():
    """Validate on synthetic data only; no GPU, no oracle. Returns False on any failure."""
    cases = []

    def check(name, cond):
        cases.append((name, bool(cond)))

    with tempfile.TemporaryDirectory() as t:
        def seq(d, imgs):
            os.makedirs(d)
            json.dump({"render": [4, 4], "output": [8, 6], "frames": len(imgs)},
                      open(os.path.join(d, "frame.json"), "w"))
            for i, im in enumerate(imgs):
                im.astype(np.float32).tofile(os.path.join(d, f"out_{i}.f32"))

        rng = np.random.default_rng(7)
        ref = [rng.uniform(0, 4, (6, 8, 4)) for _ in range(3)]
        seq(f"{t}/ref", ref)
        seq(f"{t}/same", ref)                                    # one frame removed below
        seq(f"{t}/full", ref)                                    # complete copy of the oracle
        seq(f"{t}/noisy", [r + 0.25 for r in ref])               # known offset noise
        seq(f"{t}/nan", [np.where(rng.uniform(size=r.shape) < 0.05, np.nan, r) for r in ref])
        os.remove(f"{t}/same/out_1.f32")                         # a missing frame must fail
        open(f"{t}/timed.log", "w").write("".join(
            f"GPU_FRAME_MS {i} {2.0 + i}\n" for i in range(3)))
        open(f"{t}/quiet.log", "w").write("VK_DEVICE: synthetic\nPIPELINES_CREATED: 1\n")

        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            # identical inputs: eval_verify caps mse <= 1e-12 at 120.0 dB
            got = measure(f"{t}/full", f"{t}/ref", f"{t}/ref")
            check("identical inputs -> PSNR >= 100 dB",
                  got is not None and min(got[3]) >= 100.0)
            # known noise: the imported formula must match an independent transcription
            got = measure(f"{t}/noisy", f"{t}/ref", f"{t}/ref")
            if got is None:
                check("known-noise case measures", False)
            else:
                diffs = [abs(v - _reference_psnr(a, b))
                         for v, a, b in zip(got[3], [r + 0.25 for r in ref], ref)]
                check("known-noise PSNR matches formula (tol 1e-3 dB)", max(diffs) < 1e-3)
                check("known-noise PSNR in a sane range (20..80 dB)",
                      all(20.0 < v < 80.0 for v in got[3]))
            check("missing out_<i>.f32 fails",
                  measure(f"{t}/same", f"{t}/ref", f"{t}/ref") is None)
            check("non-finite input fails", measure(f"{t}/nan", f"{t}/ref", f"{t}/ref") is None)
            check("complete directories -> exit 0",
                  report(f"{t}/full", f"{t}/ref", None, [f"{t}/timed.log"]) == 0)
            check("missing out_<i>.f32 -> exit 1",
                  report(f"{t}/same", f"{t}/ref", None, []) == 1)
            check("missing frame.json -> exit 1",
                  report(f"{t}/ref", f"{t}/noref", None, []) == 1)
            # timings: parsed exactly as eval_verify parses them; absent means n/a
            check("GPU_FRAME_MS parsed", read_log(f"{t}/timed.log")[1] == {0: 2.0, 1: 3.0, 2: 4.0})
            check("log without timings -> empty (n/a)", read_log(f"{t}/quiet.log")[1] == {})
            check("missing log -> empty (n/a)", read_log(f"{t}/absent.log")[1] == {})
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                print_log(f"{t}/timed.log")
            check("timed log reports median (frames 1..2): 3.5 ms",
                  "median (frames 1..2): 3.5000 ms" in buf.getvalue())
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                print_log(f"{t}/quiet.log")
                print_log(f"{t}/absent.log")
            check("empty/absent logs print n/a", buf.getvalue().count("n/a") == 2)

    for n, r in cases:
        print(f"  {n:44s} {'ok' if r else 'WRONG'}")
    print("PIPELINE_REPORT SELF-TEST:", "PASS" if all(r for _, r in cases) else "FAIL")
    return all(r for _, r in cases)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("port", nargs="?", help="port output directory (out_<i>.f32)")
    ap.add_argument("oracle", nargs="?", help="oracle output directory (out_<i>.f32, frame.json)")
    ap.add_argument("--frames", help="directory holding frame.json when neither side has one")
    ap.add_argument("--log", action="append", default=[], metavar="LOG",
                    help="a runner/referee log to read GPU_FRAME_MS from (repeatable)")
    ap.add_argument("--name", default="", help="cell label for the heading")
    ap.add_argument("--self-test", action="store_true",
                    help="validate the tool on synthetic data; exit non-zero on failure")
    a = ap.parse_args()
    if a.self_test:
        sys.exit(0 if self_test() else 1)
    if not a.port or not a.oracle:
        ap.error("PORT_DIR and ORACLE_DIR are required (or use --self-test)")
    sys.exit(report(a.port, a.oracle, a.frames, a.log, a.name))


if __name__ == "__main__":
    main()
