#!/usr/bin/env python3
"""tools/optiharness/content_gate.py: the g3 CONTENT sub-gate of run_harness.sh (NOT-FROZEN /
NOT-SCRAMBLED), computed from the out_<i>.f32 read-back frames the harness already has - no oracle,
no reference image, nothing frozen is touched.

    build/venv/bin/python -I tools/optiharness/content_gate.py OUTDIR --frames N \
        [--expect-fail] [--ceilings-only] [--report]

Metric (frames 1..N-1, RGB channels of the raw f32 read-back, alpha excluded - it is a constant 1):
    mean_i  = mean |out_i - out_{i-1}|                 average per-frame change
    frac_i  = fraction of RGB samples with |out_i - out_{i-1}| > 0.05   how MUCH of the frame moved
The synthetic input pans every frame (opti_client.cpp:362 the colour gradient pans 6 px/frame,
:374 the MotionVectors pan +2 px / +0.25 px), so a HEALTHY output changes every frame by a small
but measurable amount, while a scrambled one jumps by the content's full dynamic range.

Calibration - measured 2026-11-01 with this very script, three populations, one real run dir
(build/optiharness/out, run_harness.sh, 1920x1080 -> 3840x2160, 12 frames = 11 pairs) and two
fixtures built FROM those frames.  tools/optiharness/README.md section "g3 content" repeats the
numbers, the rationale and the fixture recipe:
    healthy   (the real run.  Two further healthy samples: build/optiharness/sl-baseline_out holds
               BYTE-identical frames - md5 of out_0 and out_5 equal in both dirs - and reproduces
               these statistics to the digit; a fresh run_harness.sh pass on the post-80aa2f7
               shader build (2026-11-01 15:43, build/optiharness/out_g3b) gives
               mean 0.00778715..0.15700708 / frac 0.03947752..0.33365138, i.e. within 1e-5..1e-3
               of the same bars with the same margins)
                 mean_i : min 0.00779327 (pan-only pairs) .. max 0.15699814 (inversion pairs)
                 frac_i : min 0.03951296 .. max 0.33365218
    frozen    (out_1..out_11 are symlinks to out_0: the same frame replayed 12x)
                 mean 0.0 .. 0.0            frac 0.0 .. 0.0
    scrambled-A (each frame's own RGB samples independently permuted, numpy default_rng seed
                 20261101 - the frame's value distribution, hence its nz, is preserved EXACTLY;
                 only the sample ORDER is destroyed, so consecutive frames become independent
                 draws from their own distributions)
                 mean 0.19201234 .. 0.19498892   frac 0.73412584 .. 0.74969425
    scrambled-B (every RGB sample drawn IID with replacement from the POOLED value distribution
                 of all 12 frames)
                 mean 0.21419674 .. 0.21428631   frac 0.82184992 .. 0.82211018
Bars, all four with an order-of-magnitude-or-better margin on the healthy side:
    (a) NOT-FROZEN    min mean_i > 0.001   (healthy 0.0078 = 7.8x; frozen exactly 0)
                      min frac_i > 0.01    (healthy 0.0395 = 4.0x; frozen exactly 0)
    (b) NOT-SCRAMBLED max mean_i < 0.40    (healthy 0.157 = 2.5x headroom)
                      max frac_i < 0.55    (healthy 0.334 = 1.65x; BOTH scrambles are ABOVE it)
    The frac ceiling is the scramble discriminator: NEITHER mean ceiling catches either scramble
    (0.195 and 0.214 both sit under 0.40), while frac separates 0.334 (healthy) from 0.734 (the
    weaker scramble) by 2.2x.  The bar 0.55 sits between them - 1.65x over the healthy maximum
    and 1.34x under the weaker scramble; the midpoint of those two is 0.534, so the bar keeps one
    step of extra room on the healthy side.  Both statistics are needed; that is why the sub-gate
    carries two.
    The thresholds were NOT moved to fit this measurement: 0.001 / 0.40 / 0.01 / 0.55 are exactly
    the bars every row above was scored against, and every margin still holds.

Modes
    --expect-fail    INVERTS the sub-gate (same evidence standard as g3's nz rule): the archived
                     pre-fix DLL refuses every frame, the read-back stays all-zero and identical,
                     so the run passes only when max mean_i <= 0.001 AND max frac_i <= 0.01.
    --ceilings-only  drops the NOT-FROZEN floors (keeps both scramble ceilings).  Used by
                     run_harness.sh for --corpus runs, whose frame-to-frame motion belongs to the
                     corpus files, not to the harness's synthetic pan.

Exit codes: 0 = PASS, 1 = FAIL, 2 = could not be evaluated (missing/short/non-finite frames, no
numpy) - run_harness.sh maps 2 to g3 FAIL too (fail closed, never silently green).  One parseable
verdict line is always printed:
    CONTENT_GATE <PASS|FAIL|ERROR> n=<pairs> mean_min=.. mean_max=.. frac_min=.. frac_max=..
                 low_mean=.. high_mean=.. low_frac=.. high_frac=.. <reason>
"""
import argparse
import os
import sys

# calibrated bars (see the module docstring; run_harness.sh documents them in README "g3 content")
LOW_MEAN, HIGH_MEAN = 0.001, 0.40
LOW_FRAC, HIGH_FRAC = 0.01, 0.55
FRAC_EPS = 0.05       # per-sample threshold of the "the frame actually moved here" test


def err(reason, low_mean, high_mean, low_frac, high_frac):
    print(f"CONTENT_GATE ERROR n=0 mean_min=nan mean_max=nan frac_min=nan frac_max=nan "
          f"low_mean={low_mean:g} high_mean={high_mean:g} low_frac={low_frac:g} "
          f"high_frac={high_frac:g} reason={reason}")
    return 2


def stats(outdir, n):
    """(mean_i list, frac_i list); SystemExit(2) with a CONTENT_GATE ERROR line on bad input."""
    try:
        import numpy as np
    except ImportError:
        raise SystemExit(err("numpy-unavailable(use-build/venv/bin/python)",
                             LOW_MEAN, HIGH_MEAN, LOW_FRAC, HIGH_FRAC))
    means, fracs, prev = [], [], None
    for i in range(n):
        p = os.path.join(outdir, f"out_{i}.f32")
        if not os.path.isfile(p):
            raise SystemExit(err(f"missing:{p}", LOW_MEAN, HIGH_MEAN, LOW_FRAC, HIGH_FRAC))
        cur = np.fromfile(p, dtype=np.float32)
        if cur.size % 4:
            raise SystemExit(err(f"bad-size:{p}:{cur.size}", LOW_MEAN, HIGH_MEAN, LOW_FRAC, HIGH_FRAC))
        rgb = cur.reshape(-1, 4)[:, :3]
        if not (rgb.size and np.all(np.isfinite(rgb))):
            raise SystemExit(err(f"non-finite:{p}", LOW_MEAN, HIGH_MEAN, LOW_FRAC, HIGH_FRAC))
        if prev is not None:
            d = np.abs(rgb - prev)
            means.append(float(d.mean()))
            fracs.append(float((d > FRAC_EPS).mean()))
        prev = rgb
    return means, fracs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("--frames", type=int, required=True)
    ap.add_argument("--expect-fail", action="store_true",
                    help="invert: the refused run must produce an all-zero, identical-every-frame output")
    ap.add_argument("--ceilings-only", action="store_true",
                    help="--corpus runs: drop the NOT-FROZEN floors, keep the scramble ceilings")
    ap.add_argument("--report", action="store_true", help="print every pair (calibration mode)")
    a = ap.parse_args()
    bars = (f"low_mean={LOW_MEAN:g} high_mean={HIGH_MEAN:g} low_frac={LOW_FRAC:g} "
            f"high_frac={HIGH_FRAC:g}")
    if a.frames < 2:
        print(f"CONTENT_GATE SKIP n=0 mean_min=nan mean_max=nan frac_min=nan frac_max=nan {bars} "
              "reason=needs>=2-frames (no temporal claim to gate)")
        return 0
    means, fracs = stats(a.outdir, a.frames)
    if not means:
        return err("no-pair", LOW_MEAN, HIGH_MEAN, LOW_FRAC, HIGH_FRAC)
    m_min, m_max = min(means), max(means)
    f_min, f_max = min(fracs), max(fracs)
    if a.report:
        for i, (m, f) in enumerate(zip(means, fracs), start=1):
            print(f"CONTENT_PAIR {i - 1}->{i} mean={m:.8f} frac={f:.8f}")
        print(f"CONTENT_DIST n={len(means)} mean_min={m_min:.8f} mean_max={m_max:.8f} "
              f"frac_min={f_min:.8f} frac_max={f_max:.8f} {bars}")

    bad = []
    if a.expect_fail:
        # inverted: nothing was ever written, so consecutive frames must be indistinguishable
        if m_max > LOW_MEAN:
            bad.append(f"max mean {m_max:.6g} > {LOW_MEAN:g} (output changed although the run must "
                       "write nothing)")
        if f_max > LOW_FRAC:
            bad.append(f"max frac {f_max:.6g} > {LOW_FRAC:g} (pixels moved in an all-zero run)")
    else:
        if not a.ceilings_only:
            if m_min <= LOW_MEAN:
                bad.append(f"min mean {m_min:.6g} <= {LOW_MEAN:g} (FROZEN: consecutive frames "
                           "identical although the input pans every frame)")
            if f_min <= LOW_FRAC:
                bad.append(f"min frac {f_min:.6g} <= {LOW_FRAC:g} (FROZEN: frame did not change)")
        if m_max >= HIGH_MEAN:
            bad.append(f"max mean {m_max:.6g} >= {HIGH_MEAN:g} (SCRAMBLED: frame-to-frame jump too "
                       "large)")
        if f_max >= HIGH_FRAC:
            bad.append(f"max frac {f_max:.6g} >= {HIGH_FRAC:g} (SCRAMBLED: too much of the frame "
                       "moved at once)")
    ok = not bad
    if bad:
        why = "; ".join(bad)
    elif a.expect_fail:
        why = (f"inverted ok: max mean {m_max:.6g} <= {LOW_MEAN:g} and max frac {f_max:.6g} <= "
               f"{LOW_FRAC:g} (all-zero output identical every frame - nothing was written)")
    else:
        why = (f"min mean {m_min:.6g} > {LOW_MEAN:g} (not frozen); max frac {f_max:.6g} < {HIGH_FRAC:g} "
               f"(not scrambled)" + ("; floors skipped (--ceilings-only)" if a.ceilings_only else ""))
    print(f"CONTENT_GATE {'PASS' if ok else 'FAIL'} n={len(means)} mean_min={m_min:.8f} "
          f"mean_max={m_max:.8f} frac_min={f_min:.8f} frac_max={f_max:.8f} {bars} {why}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())