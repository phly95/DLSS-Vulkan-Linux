#!/usr/bin/env python3
"""tools/sr/sr_postL_model.py: the NumPy model of NeuralHub's k_sr_post<true,12,16,false,false> (SR preset L,
oracle launch 15) - the spec that sr_post.comp's L path is written against.

    build/venv/bin/python -I tools/sr/sr_postL_model.py [--frame F] [--oracle DUMP_DIR] [--out DIR]
    build/venv/bin/python -I tools/sr/sr_postL_model.py --self-check

Inputs are the oracle's own bytes as they stood before launch 15 (newest earlier NNN.<buffer>.bin, else
start.<buffer>.bin). Outputs: full15 (u32, 768x432), ihist1 (f16 x4, 768x432), dhist (u32, 384x216).

The model REUSES the committed M spec (tools/sr/sr_post_model.py, k_sr_post<false,...>): everything there
transfers to the L instance except the spatial filter's tap grid, which the L ISA widens from 3x3 to 4x4.
Verified from the disassembly (build/agents/l-post/k_sr_post_L.s vs build/k_sr_post.s):

  * IDENTICAL in both instances (diffed, register-renumbering aside): the scalar loads (same PostArgs
    offsets - NEITHER instance reads +0xe0 g_mvhi, +0xf0/f4 mv_scale, +0xd4 ldr or +0xd8 mvhi), the
    descriptor loads, the LDS record staging (12 f16 fields per source pixel: tmR tmG tmB fac0..2
    gate0..2 A' B' C', 24-byte records, row stride 0x120), the MV sampling (2 image_sample_lz of
    mv_nondil, same dmask/samplers), the 5-tap Catmull-Rom history sets (4 groups of 5 image_sample_lz,
    identical layout - first and second candidate, both pixels), the gate blend, the encodes.
  * DIFFERENT: the spatial tap offsets. M: dx, dy in {-1, 0, 1} (v_add_f16/pk_add_f16 literals -1.0,
    0, 1.0; 9 taps; 81 v_pk_fma_f16). L: dx, dy in {-1, 0, 1, 2} (literals -1.0, 0, 1.0, 2.0 at
    k_sr_post_L.s:317-321,362,379-385,564; 16 taps; 144 v_pk_fma_f16 = 81 * 16/9). The LDS record loads
    match: L reads a 4x4 record grid (rows ky=-1..2 at byte strides 0, 288, 576, 864; columns kx=-1..2),
    M a 3x3. The weight formula w = exp2(f16-chain q(dx,dy)) and every record field are the same; the
    f16 numerator chains, Wsum/inv, the gate blend v12/t1/t2, the CR history and the encodes run over
    the 16 taps in the same (kx-major) order.
  * The off-screen fallback tonemap differs in static log count (M 27 log calls beyond staging, L 15)
    but both fallbacks are dead code on these frames (all three MV candidates are the identical constant
    (-0.2861328125, -0.075) output texels - verified on the oracle's own buffers - so every reprojection
    lands sub-texel on-screen); not modelled here, same as in the M model.

STATUS (numbers from --self-check, launch 15 of sr_L, oracle dump): see the run output; the model is
the 4x4-grid instance of the M spec, no constant of the M spec was touched.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
ORACLE = REPO / "build/oracle/oracle-8dbf027a9e-s20261025"

sys.path.insert(0, str(REPO / "tools/sr"))
import sr_post_model as M  # noqa: E402

# The L instance's spatial filter: 16 taps, dx/dy offsets in {-1, 0, 1, 2} (kx-major order), vs M's 9.
TAPS_L = [(kx, ky) for kx in (-1, 0, 1, 2) for ky in (-1, 0, 1, 2)]
M.TAPS_KX_MAJOR = TAPS_L    # taps() reads the module global


def load_inputs(oracle, frame, launch=15):
    """The M model's loader with L's launch number (sr_L dump: launch 15 = k_sr_post<true,...>)."""
    return M.load_inputs(oracle, frame, launch=launch)


def run(frame, oracle, out=None, quiet=False):
    inp = load_inputs(oracle, frame)
    rec = M.records(inp)
    m = M.model_frame(inp, rec)
    if out:
        os.makedirs(out, exist_ok=True)
        Path(out, "full15.bin").write_bytes(m["full15"].tobytes())
        Path(out, "ihist1.bin").write_bytes(m["ihist1"].tobytes())
        Path(out, "dhist.bin").write_bytes(m["dhist"].tobytes())
    res = {}
    of = np.frombuffer(inp["oracle_out"]["full15"].read_bytes(), np.uint32).reshape(M.H15, M.W15)
    oh = np.frombuffer(inp["oracle_out"]["ihist1"].read_bytes(), np.float16).reshape(M.H15, M.W15, 4)
    od = np.frombuffer(inp["oracle_out"]["dhist"].read_bytes(), np.uint32).reshape(M.H15 // 2, M.W15 // 2)
    jobs = [("full15", m["full15"], "u32"), ("ihist1", m["ihist1"], "f16"), ("dhist", m["dhist"], "u32"),
            ("full15-dec", M.decode_full15(m["full15"]), "f32")]
    srcs = {"full15": inp["oracle_out"]["full15"], "ihist1": inp["oracle_out"]["ihist1"],
            "dhist": inp["oracle_out"]["dhist"], "full15-dec": inp["oracle_out"]["full15"]}
    for name, port, dt in jobs:
        orc = srcs[name].read_bytes()
        if name == "full15-dec":
            orc = M.decode_full15(of).astype(np.float32).tobytes()
            port = np.ascontiguousarray(port, np.float32).tobytes()
        else:
            port = port.tobytes()
        r = M.LR.measure(port, orc, dt)
        res[name] = r
        if not quiet and r:
            n, peak, psnr, rr, rrmse, exact = r
            print(f"  {name:12s} n {n}  bit-exact {exact * 100:.4f}%  peak {peak:.6g}  PSNR {M.LR.fmt_psnr(psnr)} dB"
                  f"  r {rr if rr is None else round(rr, 6)}  rRMSE {rrmse:.6f}")
    if not quiet:
        par = (np.arange(M.H15) % 2)[:, None] * np.ones((1, M.W15), int)
        for pix, lab in ((0, "pixel 0 (even rows)"), (1, "pixel 1 (odd rows)")):
            sel = par == pix
            print(f"  {lab:20s} full15 bit-exact {np.mean(m['full15'][sel] == of[sel]) * 100:.4f}%  "
                  f"ihist1 bit-exact {np.mean(m['ihist1'][sel] == oh[sel]) * 100:.4f}%")
    return res, m, inp


def self_check():
    """The acceptance thresholds of the brief (launch 15, sr_L, oracle dump): frame 0 >= 99%, frame 1 >= 97%
    bit-exact for each of full15, ihist1 and dhist."""
    thr = {0: 99.0, 1: 97.0}
    ok = True
    for frame in (0, 1):
        print(f"[frame {frame}, launch 15]")
        res, _, _ = run(frame, str(ORACLE / "sr_L/dump"))
        for k in ("full15", "ihist1", "dhist"):
            got = res[k][5] * 100
            good = got >= thr[frame]
            ok &= good
            print(f"  -> {k} bit-exact {got:.4f}% >= {thr[frame]:.1f}%: {'PASS' if good else 'FAIL'}")
    print("SELF-CHECK:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--oracle", default=str(ORACLE / "sr_L/dump"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--self-check", action="store_true")
    a = ap.parse_args()
    if a.self_check:
        return self_check()
    run(a.frame, a.oracle, a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
