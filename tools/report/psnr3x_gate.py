#!/usr/bin/env python3
"""tools/report/psnr3x_gate.py: NON-fail-fast quality gate over the 1.5x / 2.0x / 3.0x + UV-MV matrix.

    build/venv/bin/python -I tools/report/psnr3x_gate.py [--base build/psnr3x] [--suffix 2]

--suffix re-points the PORT side of every cell at out_port/<cell><suffix> (e.g. --suffix 2 scores
the 2026-11-01 re-run A2..E2 produced by the level-15-ratio fix) and leaves out_port/A..E - the
pre-fix "before" evidence - untouched, as are the oracle/gt dirs (they never change).

WHAT THIS COVERS (and tools/verify never did)
    The frozen referee (tools/verify/run_all.sh:151,160,162) only ever built 960->1920 (1.5x),
    1920->3840 (2.0x) and 1280->1920 (1.5x) cells - its PSNR matrix stops at 2.0x, and
    tools/scenario/REPORT.md:901 records that 3.0x "has NEVER been PSNR-validated".  The repo
    already ran a 3.0x-capable matrix OUTSIDE tools/verify (build/psnr3x_matrix.sh) into
    build/psnr3x/; this script is the gate for those outputs:

        A = 1.5x preset M, pixel-unit MVs   (corpus a15f, GT a15_gt) - rig baseline
        B = 2.0x preset M, pixel-unit MVs   (corpus b20f, GT b20_gt) - the Oct-31 in-game cell
        C = 3.0x preset M, pixel-unit MVs   (corpus c30f, GT c30_gt) - the never-validated ratio
        D = 3.0x preset L, pixel-unit MVs   (corpus c30f, GT c30_gt) - the CURRENT in-game combo
        E = 3.0x preset L, in-game UV MVs   (corpus c30u, GT c30_gt, MV.Scale=(1280,-720))
                                          - the S6 decode path end to end

WHY IT IS NOT eval_verify.py check-cell
    tools/verify/eval_verify.py's check-cell is FAIL-FAST: cell() returns at the FIRST frame that
    is missing, non-finite, or over the 0.1 dB ground-truth delta (eval_verify.py:76-93), so one
    bad frame hides every remaining frame of that cell - and the whole rest of the matrix, because
    the caller stops there.  This gate walks EVERY frame of EVERY cell, prints the full table, and
    only then decides.  The math itself is NOT copied: this script does
    `sys.path.insert(tools/verify); import eval_verify` and calls the frozen module's own
    ev.psnr / ev.load / ev.frame_info / ev.tone_map and reads its frozen bars
    (V4_MIN_DB, V4_MEAN_DB, GT_DELTA_DB).  tools/verify/ is READ-ONLY here - never modified.

WHAT EACH COLUMN IS (do not confuse the two bar sets)
    PSNR(port, ref)  - V4's own bar: per frame >= 50.0 dB, mean >= 55.0 dB (eval_verify.py:30).
    PSNR(port, GT)   - port against the render's ground truth.
    PSNR(ref, GT)    - oracle (srref) against the same ground truth.
    |delta|          - |PSNR(port,GT) - PSNR(ref,GT)| <= 0.1 dB per frame (eval_verify.py:30,
                       GT_DELTA_DB), i.e. "the port is as close to GT as the oracle is".
    These are the V4 bars of tools/verify - the ONLY authority for 50/55/0.1.  quality_guard.py's
    B1 (tools/report/quality_guard.py) scores the FIRST column against its own much looser
    regression floor (41.0 dB, an optimisation-phase backstop); that floor is never used here.

VERDICT
    Per cell: PASS iff every frame has finite numbers, min PSNR(port,ref) >= 50.0, mean >= 55.0,
    and max |delta| <= 0.1.  Missing cell dirs print `SKIP/MISSING` (they are listed in the overall
    line, never crash); a PRESENT cell with missing frames/non-finite values FAILs.  One overall
    PASS/FAIL line; exit 0 = PASS, 1 = FAIL (also 1 when no cell at all could be evaluated).
    Skipped cells are coverage gaps, not quality passes - read the overall line's coverage note.
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "tools", "verify"))
import eval_verify as ev  # noqa: E402  (FROZEN referee module: imported, never copied, never edited)

# name -> (label, port dir, ref dir, gt dir) relative to --base
CELLS = {
    "A": ("1.5x preset M pixel-MV", "out_port/A", "out_ref/A", "a15_gt"),
    "B": ("2.0x preset M pixel-MV", "out_port/B", "out_ref/B", "b20_gt"),
    "C": ("3.0x preset M pixel-MV", "out_port/C", "out_ref/C", "c30_gt"),
    "D": ("3.0x preset L pixel-MV", "out_port/D", "out_ref/D", "c30_gt"),
    "E": ("3.0x preset L UV-MV (MV.Scale=(1280,-720))", "out_port/E", "out_ref/E", "c30_gt"),
}


def missing_cell(tag, why):
    print(f"=== {tag}: SKIP/MISSING ({why})")
    return "MISSING"


def eval_cell(tag, label, port, ref, gt):
    """Print the full per-frame table; return 'PASS' | 'FAIL' | 'MISSING'."""
    if not os.path.isdir(port):
        return missing_cell(f"{tag} {label}", f"no port output at {port}")
    fj = os.path.join(ref, "frame.json")
    if not os.path.isfile(fj):
        return missing_cell(f"{tag} {label}", f"no oracle frame.json at {fj}")
    if not os.path.isdir(gt):
        return missing_cell(f"{tag} {label}", f"no ground truth at {gt}")
    port_frames = sorted(f for f in os.listdir(port) if f.startswith("out_") and f.endswith(".f32"))
    if not port_frames:
        return missing_cell(f"{tag} {label}", f"{port} holds no out_<i>.f32")
    ow, oh, n = ev.frame_info(ref)
    print(f"=== {tag} {label}: {n} frames, {ow}x{oh}   port={port}  ref={ref}  gt={gt}")
    print("    f   port/ref dB   port/GT dB    ref/GT dB    |delta| dB")
    prs, deltas, bad = [], [], []
    for i in range(n):
        p = ev.load(os.path.join(port, f"out_{i}.f32"), ow, oh)
        r = ev.load(os.path.join(ref, f"out_{i}.f32"), ow, oh)
        g = ev.load(os.path.join(gt, f"gt_{i}.f32"), ow, oh)
        if p is None or r is None or g is None:
            print(f"  {i:4d}   MISSING (port={p is not None} ref={r is not None} gt={g is not None}) -> FAIL")
            bad.append(f"frame {i} missing")
            continue
        v_pr, nf_pr = ev.psnr(p, r)
        v_pg, nf_pg = ev.psnr(p, g)
        v_rg, nf_rg = ev.psnr(r, g)
        if nf_pr or nf_pg or nf_rg:
            print(f"  {i:4d}   NON-FINITE (port/ref={nf_pr} port/GT={nf_pg} ref/GT={nf_rg}) -> FAIL")
            bad.append(f"frame {i} non-finite")
            continue
        d = abs(v_pg - v_rg)
        print(f"  {i:4d}   {v_pr:10.2f}   {v_pg:11.3f}   {v_rg:11.3f}   {d:11.3f}")
        prs.append(v_pr)
        deltas.append(d)
    if not prs:
        print("    cell summary: no scorable frame -> FAIL")
        return "FAIL"
    mn, mean = min(prs), sum(prs) / len(prs)
    mx = max(deltas)
    ok_min, ok_mean, ok_d = mn >= ev.V4_MIN_DB, mean >= ev.V4_MEAN_DB, mx <= ev.GT_DELTA_DB
    verdict = "PASS" if (ok_min and ok_mean and ok_d and not bad) else "FAIL"
    print(f"    cell summary: min {mn:6.2f} dB (>= {ev.V4_MIN_DB:.1f} {'ok' if ok_min else 'FAIL'}) | "
          f"mean {mean:6.2f} dB (>= {ev.V4_MEAN_DB:.1f} {'ok' if ok_mean else 'FAIL'}) | "
          f"max |delta| {mx:6.3f} dB (<= {ev.GT_DELTA_DB:.1f} {'ok' if ok_d else 'FAIL'}) | "
          f"scorable {len(prs)}/{n} -> {verdict}")
    for b in bad:
        print(f"    cell summary: {b}")
    return verdict


def main() -> int:
    ap = argparse.ArgumentParser(description="non-fail-fast V4 gate over the 1.5x/2.0x/3.0x + UV-MV matrix")
    ap.add_argument("--base", default=os.path.join(ROOT, "build", "psnr3x"),
                    help="matrix root written by build/psnr3x_matrix.sh (default: build/psnr3x)")
    ap.add_argument("--suffix", default="",
                    help="string appended to every PORT dir name, so the same gate can be pointed "
                         "at a re-run of the matrix without touching the original evidence: "
                         "--suffix 2 scores out_port/A2..out_port/E2 (the 2026-11-01 level-15-ratio "
                         "fix) while out_port/A..E stay the pre-fix 'before' numbers. "
                         "ref/gt dirs are NOT suffixed - the oracle outputs do not change. "
                         "Default: '' (the original out_port/A..E)")
    a = ap.parse_args()
    if a.suffix and not a.suffix.isalnum():
        ap.error(f"--suffix must be alphanumeric, got {a.suffix!r}")
    print("psnr3x_gate: non-fail-fast V4 evaluation (tools/verify/eval_verify.py imported, not copied)")
    if a.suffix:
        print(f"port dirs: out_port/<cell>{a.suffix} (re-run matrix); ref/gt dirs unchanged")
    print("bars (FROZEN authority tools/verify/eval_verify.py:30): "
          f"PSNR(port,ref) per frame >= {ev.V4_MIN_DB}, mean >= {ev.V4_MEAN_DB}, "
          f"|PSNR(port,GT)-PSNR(ref,GT)| <= {ev.GT_DELTA_DB} per frame")
    results = {}
    for tag, (label, port, ref, gt) in CELLS.items():
        results[tag] = eval_cell(tag, label, os.path.join(a.base, port) + a.suffix,
                                 os.path.join(a.base, ref), os.path.join(a.base, gt))
    print()
    print("MATRIX: " + "  ".join(f"{t}={results[t]}" for t in CELLS))
    missing = [t for t, r in results.items() if r == "MISSING"]
    failed = [t for t, r in results.items() if r == "FAIL"]
    passed = [t for t, r in results.items() if r == "PASS"]
    if missing:
        print(f"coverage: {len(passed) + len(failed)}/{len(CELLS)} cells evaluated, "
              f"{len(missing)} SKIP/MISSING ({','.join(missing)}) - a skipped cell is a coverage gap, "
              "not a quality pass")
    ok = bool(passed) and not failed
    print(f"OVERALL: {'PASS' if ok else 'FAIL'}  ({len(passed)} pass, {len(failed)} fail, {len(missing)} missing)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())