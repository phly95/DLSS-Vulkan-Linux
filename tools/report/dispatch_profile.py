"""tools/report/dispatch_profile.py: per-dispatch GPU time from a DLSS5_AMD_PROFILE=1 vk_runner log.

    build/venv/bin/python -I tools/report/dispatch_profile.py <runner.log> [--skip 1]

Medians over frames (frame 0 skipped by default: pipeline warm-up), summed per dispatch name in
issue order, then the frame median. Reads the DISPATCH_MS / GPU_FRAME_MS lines vk_runner prints.
"""
import argparse
import re
import statistics
import sys
from collections import OrderedDict, defaultdict


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--skip", type=int, default=1, help="frames to skip at the start (default 1)")
    a = ap.parse_args()
    per = defaultdict(lambda: defaultdict(float))   # frame -> name -> ms (summed over repeats)
    count = defaultdict(lambda: defaultdict(int))
    order: "OrderedDict[str, None]" = OrderedDict()
    frame_ms = {}
    for line in open(a.log, errors="replace"):
        m = re.match(r"DISPATCH_MS (\d+) \d+ (\S+) ([0-9.]+)", line)
        if m:
            f, n, v = int(m[1]), m[2], float(m[3])
            per[f][n] += v
            count[f][n] += 1
            order.setdefault(n, None)
            continue
        m = re.match(r"GPU_FRAME_MS (\d+) ([0-9.]+)", line)
        if m:
            frame_ms[int(m[1])] = float(m[2])
    frames = sorted(f for f in frame_ms if f >= a.skip)
    if not frames:
        print("no profiled frames (was DLSS5_AMD_PROFILE=1 set?)", file=sys.stderr)
        return 1
    print(f"{'dispatch':<18}{'n/frame':>8}{'median ms':>12}")
    for n in order:
        vals = [per[f][n] for f in frames if n in per[f]]
        if vals:
            print(f"{n:<18}{count[frames[0]][n]:>8}{statistics.median(vals):>12.3f}")
    print(f"{'FRAME':<18}{'':>8}{statistics.median(frame_ms[f] for f in frames):>12.3f}"
          f"   (frames {frames[0]}..{frames[-1]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
