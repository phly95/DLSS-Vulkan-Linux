#!/usr/bin/env python3
"""tools/report/launch_report.py: the canonical *per-launch* (per-kernel) fidelity report.

    build/venv/bin/python -I tools/report/launch_report.py PORT ORACLE \
        [--name LABEL] [--dtype fp8|f16|f32|u32|u8] [--peak X] [--self-test]

PORT and ORACLE are either two raw buffer files of identical length, or two directories,
in which case every file present in both is paired by basename and reported separately.

WHY THIS EXISTS (operator, 2026-10-26).  The kernel success metric changed: progress is
judged by a kernel's fidelity against *oracle inputs*, not by bit-exact percentage and not
by the end-to-end pipeline number.  Bit-exact percentage stays meaningful only for the
shallow kernels where it falls out for free; it is still printed here, but it is no longer
the primary judgement.  Whole-pipeline numbers remain the exclusive output of
tools/report/pipeline_report.py - this script never states one.

THE THREE NUMBERS, always all three, always from this script:

  peak    max|x| of the ORACLE tensor, stated in the output (or --peak to fix it).
          A per-launch output is an fp8 activation spanning up to +-448, so PSNR means
          nothing without an explicit peak: with an undefined peak the 40 dB milestone
          would be gameable and incomparable between sessions.
  PSNR    10 * log10(peak^2 / mean((port - oracle)^2)), on the decoded values.
  r       Pearson correlation between the flattened port and oracle tensors.
  rRMSE   RMS(port - oracle) / RMS(oracle)  - scale-free, so it stays comparable across
          dumps whose peak differs.

DTYPE.  The network's per-launch tensors are E4M3 fp8, so fp8 is the default and the
decoder reproduces exactly the table `shaders/sr/sr_stage.comp` uses (e4m3_lut), which was
verified bit-identical for all 256 codes: e=15,m=7 is 480, not NaN, and 0x80 is -0.  Both
sides are decoded with the same table, so PSNR compares values, not encodings.

EXIT STATUS.  0 on success (including the degenerate "oracle has zero variance" case, which
prints r as n/a - that is a property of the data, not a tool failure).  1 when a file is
missing, the two buffers differ in length, or a decoded value is non-finite; the non-finite
count is printed first so the failure is never silent.
"""
import argparse
import math
import os
import struct
import sys

import numpy as np


# ---------------------------------------------------------------- fp8 (E4M3) decode
def fp8_to_f32(raw):
    """E4M3 -> float32, bit-identical to sr_stage.comp's e4m3_lut for all 256 codes."""
    b = np.frombuffer(raw, np.uint8).astype(np.uint32)
    sign = (b & 0x80) << 24
    e = (b >> 3) & 0xF
    m = b & 7
    out = np.empty(b.shape, np.uint32)

    hi = e >= 1
    out[hi] = sign[hi] | ((e[hi] + 120) << 23) | (m[hi] << 20)

    lo = ~hi
    if lo.any():
        out[lo] = sign[lo]                       # +/- 0 first
        nz = lo & (m != 0)                       # subnormal e4m3, m in 1..7
        if nz.any():
            kk = np.where(m[nz] >= 4, 2, np.where(m[nz] >= 2, 1, 0)).astype(np.uint32)
            exp = kk + 118
            mant = (m[nz].astype(np.uint32) - (np.uint32(1) << kk)) << (23 - kk)
            out[nz] = sign[nz] | (exp << 23) | mant
    return out.view(np.float32)


def decode(raw, dtype):
    if dtype == "fp8":
        return fp8_to_f32(raw)
    if dtype == "f16":
        return np.frombuffer(raw, np.uint16).view(np.float16).astype(np.float32)
    if dtype == "f32":
        return np.frombuffer(raw, np.float32)
    if dtype == "u32":
        return np.frombuffer(raw, np.uint32).astype(np.float32)
    if dtype == "u8":
        return raw.astype(np.float32)
    raise ValueError(dtype)


# ---------------------------------------------------------------- measurement
def measure(port_raw, oracle_raw, dtype, peak=None):
    """-> (n, peak, psnr, r, rrmse, exact_frac) or None (caller has already explained why)."""
    if len(port_raw) != len(oracle_raw) or not port_raw:
        return None
    a = decode(port_raw, dtype)
    b = decode(oracle_raw, dtype)
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        bad = int((~np.isfinite(a)).sum() + (~np.isfinite(b)).sum())
        print(f"  non-finite values: {bad} (port {int((~np.isfinite(a)).sum())}, "
              f"oracle {int((~np.isfinite(b)).sum())})")
        return None

    err = a.astype(np.float64) - b.astype(np.float64)
    mse = float(np.mean(err * err))
    rms_o = float(np.sqrt(np.mean(b.astype(np.float64) ** 2)))

    if peak is None:
        peak = float(np.max(np.abs(b))) if b.size else 0.0
        if peak == 0.0:
            peak = 1.0                      # all-zero oracle: PSNR would be meaningless
    psnr = math.inf if mse == 0.0 else 10.0 * math.log10((peak * peak) / mse)

    sa = a.astype(np.float64)
    sb = b.astype(np.float64)
    da = sa - sa.mean()
    db = sb - sb.mean()
    na = math.sqrt(float(np.sum(da * da)))
    nb = math.sqrt(float(np.sum(db * db)))
    r = None if (na == 0.0 or nb == 0.0) else float(np.sum(da * db)) / (na * nb)

    rrmse = float("nan") if rms_o == 0.0 else math.sqrt(mse) / rms_o
    exact = float(np.mean(port_raw == oracle_raw))
    return len(a), peak, psnr, r, rrmse, exact


def fmt_psnr(v):
    return "inf" if math.isinf(v) else f"{v:.4f}"


def print_row(name, res):
    n, peak, psnr, r, rrmse, exact = res
    r_s = "n/a" if r is None else f"{r:.6f}"
    rr_s = "n/a" if math.isnan(rrmse) else f"{rrmse:.6f}"
    print(f"  {name}")
    print(f"    peak(|oracle|) = {peak:.6f}   (stated: PSNR uses exactly this peak)")
    print(f"    PSNR           = {fmt_psnr(psnr)} dB")
    print(f"    pearson r      = {r_s}")
    print(f"    rel RMSE       = {rr_s}")
    print(f"    bit-exact      = {100.0 * exact:.4f}%  ({int(round(exact * n))}/{n})")
    print(f"    elements       = {n}")


def report(port, oracle, dtype, peak, name):
    label = name or os.path.basename(os.path.normpath(port))
    if os.path.isdir(port) and os.path.isdir(oracle):
        pf = {f: os.path.join(port, f) for f in os.listdir(port)}
        of = {f: os.path.join(oracle, f) for f in os.listdir(oracle)}
        both = sorted(set(pf) & set(of))
        only_p = sorted(set(pf) - set(of))
        only_o = sorted(set(of) - set(pf))
        if not both:
            print(f"launch_report: no file present in both {port} and {oracle}", file=sys.stderr)
            return 1
        print(f"LAUNCH REPORT {label} (dtype={dtype}, {len(both)} paired file(s))")
        bad = 0
        for f in both:
            res = measure(open(pf[f], "rb").read(), open(of[f], "rb").read(), dtype, peak)
            if res is None:
                print(f"  {f}: FAILED (see above)")
                bad += 1
                continue
            print_row(f, res)
        if only_p:
            print(f"  note: only in port:   {', '.join(only_p)}")
        if only_o:
            print(f"  note: only in oracle: {', '.join(only_o)}")
        return 1 if bad else 0

    for p in (port, oracle):
        if not os.path.isfile(p):
            print(f"launch_report: not a file or directory: {p}", file=sys.stderr)
            return 1
    res = measure(open(port, "rb").read(), open(oracle, "rb").read(), dtype, peak)
    if res is None:
        print("launch_report: measurement failed", file=sys.stderr)
        return 1
    print(f"LAUNCH REPORT {label} (dtype={dtype})")
    print_row("", res)
    return 0


# ---------------------------------------------------------------- self-test
def self_test():
    cases = []

    def check(n, ok):
        cases.append((n, bool(ok)))

    # the fp8 decoder must reproduce sr_stage.comp's e4m3_lut (all 256 codes verified;
    # spot-check the boundaries and the values a naive E4M3 would get wrong)
    tbl = {0x00: 0.0, 0x80: -0.0, 0x38: 1.0, 0x7F: 480.0, 0xFE: -448.0, 0xFF: -480.0,
           0x01: 0.001953125, 0x07: 0.013671875, 0x3F: 1.875, 0x40: 2.0, 0xC0: -2.0}
    got = fp8_to_f32(bytes(tbl.keys()))
    check("fp8 decode matches e4m3_lut spot values",
          all((g == v) or (g == 0.0 and v == 0.0) for g, v in zip(got, tbl.values())))
    check("fp8 0x7f -> 480 (not NaN), 0x80 -> -0",
          fp8_to_f32(bytes([0x7F]))[0] == 480.0 and
          math.copysign(1.0, float(fp8_to_f32(bytes([0x80]))[0])) == -1.0)
    check("fp8 all 256 codes decode finite", np.isfinite(fp8_to_f32(bytes(range(256)))).all())

    # Tie the decoder to the shader it must agree with: every code against the literal
    # e4m3_lut in shaders/sr/sr_stage.comp, when that file is present.
    lut_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "shaders", "sr", "sr_stage.comp")
    if os.path.isfile(lut_path):
        import re
        body = re.search(r'const float16_t e4m3_lut\[256\] = float16_t\[256\]\((.*?)\);',
                         open(lut_path).read(), re.S)
        codes = re.findall(r'float16_t\(([^)]*)\)', body.group(1)) if body else []
        if len(codes) == 256:
            def _f(s):
                s = s.strip()
                return np.float16(-0.0 if s == "-0" else (0.0 if s == "0" else float(s)))
            ref = np.array([_f(s) for s in codes], dtype=np.float16).astype(np.float32)
            mine = fp8_to_f32(bytes(range(256))).astype(np.float32)
            check("fp8 decode matches sr_stage.comp e4m3_lut (all 256 codes, bit-exact)",
                  np.array_equal(mine.view(np.uint32), ref.view(np.uint32)))
        else:
            check("fp8 decode matches sr_stage.comp e4m3_lut (all 256 codes)", False)
    else:
        print("  (sr_stage.comp not present: skipped the e4m3_lut cross-check)")

    rng = np.random.default_rng(7)
    ref = rng.normal(0.0, 1.0, 4096).astype(np.float32)
    raw = ref.tobytes()

    check("identical -> PSNR inf, r == 1, rRMSE == 0, 100% exact",
          (lambda z: z and math.isinf(z[2]) and abs(z[3] - 1.0) < 1e-12
           and z[4] == 0.0 and z[5] == 1.0)(measure(raw, raw, "f32")))

    # PSNR with an explicit peak must follow 10*log10(peak^2/mse)
    noisy = (ref + 0.25).astype(np.float32)
    z = measure(noisy.tobytes(), raw, "f32", peak=3.0)
    expect = 10.0 * math.log10(9.0 / 0.0625)
    check("PSNR with --peak follows the formula", z is not None and abs(z[2] - expect) < 1e-6)

    # peak defaults to max|oracle| and is reported
    z = measure(noisy.tobytes(), raw, "f32")
    check("peak defaults to max(|oracle|)", z is not None and abs(z[1] - float(np.abs(ref).max())) < 1e-6)

    # correlation and relative RMSE are scale/offset sensitive in the right way
    scaled = (ref * 2.0).astype(np.float32)
    z = measure(scaled.tobytes(), raw, "f32")
    check("r == 1 for a pure scale change (shape preserved)",
          z is not None and z[3] is not None and abs(z[3] - 1.0) < 1e-9)
    check("rRMSE == 1 for a pure scale change", z is not None and abs(z[4] - 1.0) < 1e-9)

    # length mismatch and missing input must fail
    check("length mismatch fails", measure(raw, raw[:100], "f32") is None)
    check("missing input fails", measure(b"", b"", "f32") is None)
    nan = np.full(8, np.nan, np.float32).tobytes()
    import contextlib, io as _io
    with contextlib.redirect_stdout(_io.StringIO()) as _cap:   # it prints a count, then fails
        _r = measure(nan, nan, "f32")
    check("non-finite input fails", _r is None)
    check("non-finite input reports its count first",
          "non-finite values: 16" in _cap.getvalue())

    # degenerate oracle (zero variance) -> r is n/a, not a crash
    z = measure(np.zeros(64, np.float32).tobytes(), np.zeros(64, np.float32).tobytes(), "f32")
    check("all-zero oracle -> r n/a, still succeeds", z is not None and z[3] is None)

    for n, ok in cases:
        print(f"  {n:56s} {'ok' if ok else 'WRONG'}")
    print("LAUNCH_REPORT SELF-TEST:", "PASS" if all(ok for _, ok in cases) else "FAIL")
    return all(ok for _, ok in cases)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("port", nargs="?", help="port buffer file, or a directory of them")
    ap.add_argument("oracle", nargs="?", help="oracle buffer file (same length), or a directory")
    ap.add_argument("--dtype", default="fp8", choices=["fp8", "f16", "f32", "u32", "u8"],
                    help="how to decode both sides (default fp8 = E4M3, as the network's tensors)")
    ap.add_argument("--peak", type=float, default=None, metavar="X",
                    help="PSNR peak; default is max(|oracle|), which is always printed")
    ap.add_argument("--name", default="", help="label for the heading")
    ap.add_argument("--self-test", action="store_true",
                    help="validate the tool on synthetic data; exit non-zero on failure")
    a = ap.parse_args()
    if a.self_test:
        sys.exit(0 if self_test() else 1)
    if not a.port or not a.oracle:
        ap.error("PORT and ORACLE are required (or use --self-test)")
    sys.exit(report(a.port, a.oracle, a.dtype, a.peak, a.name))


if __name__ == "__main__":
    main()
