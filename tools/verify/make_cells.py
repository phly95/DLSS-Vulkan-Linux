#!/usr/bin/env python3
"""Build the referee's variant input directories (run_all.sh V3, V5) from a generated base sequence.
The referee makes every variant itself, on copies; the port only ever sees the resulting directory.

    make_cells.py fmt     SRC DST --mode sr|rr [--col C] [--mv C] [--dep C] [--nrm C] [--alb C] [--spa C]
                          [--rgh C] [--ldr] [--normals-unorm]
        re-encode inputs with gather format codes (tools/verify/texfmt.py); the default-format file of
        every re-encoded input is removed, so only the coded texture can be read
    make_cells.py mvhi    SRC DST --mv C      SR: motion vectors at output size (SRC's mvout_i.f16) in
                                              code C, params mvhi=1; the render-size mv_i is removed
    make_cells.py params  SRC DST key=value   set a parameter in every params_i.json (JSON values)
    make_cells.py zero-mv SRC DST             V5: every motion vector set to 0
    make_cells.py corrupt-weights SRC DST FILE   V5: copy a packed weight directory, FILE reversed byte-wise
"""
import argparse
import json
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import texfmt  # noqa: E402

# name -> (default code, default extension, channels, dtype, texfmt kind)
SR_INPUTS = {"col": (0, "f16", 4, np.float16, "sr_col"), "mv": (0, "f16", 2, np.float16, "sr_mv"),
             "dep": (0, "f32", 1, np.float32, "sr_dep")}
RR_INPUTS = {"col": (0, "f16", 4, np.float16, "rr"), "mv": (2, "f16", 2, np.float16, "rr"),
             "dep": (3, "f32", 1, np.float32, "rr"), "nrm": (0, "f16", 4, np.float16, "rr"),
             "alb": (0, "f16", 4, np.float16, "rr"), "spa": (0, "f16", 4, np.float16, "rr"),
             "rgh": (3, "f32", 1, np.float32, "rr")}


def copy(src, dst):
    if os.path.exists(dst):
        raise SystemExit(f"{dst} exists")
    shutil.copytree(src, dst)


def info(d):
    fj = json.load(open(os.path.join(d, "frame.json")))
    return fj["render"], fj["output"], fj["frames"]


def edit_params(d, n, upd):
    for i in range(n):
        p = os.path.join(d, f"params_{i}.json")
        j = json.load(open(p))
        j.update(upd(i) if callable(upd) else upd)
        json.dump(j, open(p, "w"))


def cmd_fmt(a):
    table = SR_INPUTS if a.mode == "sr" else RR_INPUTS
    copy(a.src, a.dst)
    (w, h), _, n = info(a.dst)
    codes = {k: getattr(a, k) for k in table if getattr(a, k, None) is not None}
    if a.mode == "sr" and "col" in codes and (codes["col"] in texfmt.SR_LDR_COL) != a.ldr:
        raise SystemExit(f"colour code {codes['col']} is {'LDR' if codes['col'] in texfmt.SR_LDR_COL else 'HDR'}: "
                         f"--ldr must {'' if codes['col'] in texfmt.SR_LDR_COL else 'not '}be given")
    for name, code in codes.items():
        c0, ext, ch, dt, kind = table[name]
        bpp = texfmt.bpp(kind, code)
        for i in range(n):
            src = os.path.join(a.dst, f"{name}_{i}.{ext}")
            x = np.fromfile(src, dt).astype(np.float64).reshape(h, w, ch)
            if name == "nrm" and a.normals_unorm:
                x = np.concatenate([x[..., :3] * 0.5 + 0.5, x[..., 3:]], axis=2)
            raw = texfmt.encode(kind, code, x)
            assert len(raw) == w * h * bpp
            os.remove(src)
            out = os.path.join(a.dst, f"{name}_{i}." + (ext if code == c0 else f"c{code}"))
            open(out, "wb").write(raw)
    upd = {f"fmt_{k}": v for k, v in codes.items()}
    if a.ldr:
        upd["ldr"] = 1
    if a.normals_unorm:
        upd["normals_unorm"] = 1
    edit_params(a.dst, n, upd)


def cmd_mvhi(a):
    copy(a.src, a.dst)
    _, (ow, oh), n = info(a.dst)
    for i in range(n):
        src = os.path.join(a.dst, f"mvout_{i}.f16")
        if not os.path.exists(src):
            raise SystemExit(f"{src} missing: generate the base sequence with --mv-out")
        x = np.fromfile(src, np.float16).astype(np.float64).reshape(oh, ow, 2)
        os.remove(src)
        os.remove(os.path.join(a.dst, f"mv_{i}.f16"))
        raw = texfmt.encode("sr_mv", a.mv, x)
        open(os.path.join(a.dst, f"mv_{i}." + ("f16" if a.mv == 0 else f"c{a.mv}")), "wb").write(raw)
    edit_params(a.dst, n, {"mvhi": 1, "fmt_mv": a.mv})


def cmd_params(a):
    copy(a.src, a.dst)
    _, _, n = info(a.dst)
    upd = {}
    for kv in a.kv:
        k, v = kv.split("=", 1)
        upd[k] = json.loads(v)
    edit_params(a.dst, n, upd)


def cmd_zero_mv(a):
    copy(a.src, a.dst)
    for f in sorted(os.listdir(a.dst)):
        if f.startswith("mv_") or f.startswith("mvout_"):
            p = os.path.join(a.dst, f)
            if not f.endswith(".f16"):
                raise SystemExit(f"{p}: zero-mv takes default-format motion vectors only")
            np.zeros(os.path.getsize(p) // 2, np.float16).tofile(p)


def cmd_corrupt(a):
    copy(a.src, a.dst)
    p = os.path.join(a.dst, a.file)
    b = open(p, "rb").read()
    if len(b) < 2:
        raise SystemExit(f"{p} is too small to corrupt")
    open(p, "wb").write(b[::-1])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("fmt")
    c.add_argument("src"); c.add_argument("dst"); c.add_argument("--mode", choices=["sr", "rr"], required=True)
    for k in RR_INPUTS:
        c.add_argument(f"--{k}", type=int)
    c.add_argument("--ldr", action="store_true"); c.add_argument("--normals-unorm", action="store_true")
    c = sub.add_parser("mvhi"); c.add_argument("src"); c.add_argument("dst"); c.add_argument("--mv", type=int, required=True)
    c = sub.add_parser("params"); c.add_argument("src"); c.add_argument("dst"); c.add_argument("kv", nargs="+")
    c = sub.add_parser("zero-mv"); c.add_argument("src"); c.add_argument("dst")
    c = sub.add_parser("corrupt-weights"); c.add_argument("src"); c.add_argument("dst"); c.add_argument("file")
    a = ap.parse_args()
    if a.cmd == "fmt" and a.mode == "sr":
        for k in ("nrm", "alb", "spa", "rgh"):
            if getattr(a, k) is not None:
                raise SystemExit(f"--{k} is an RR input")
    {"fmt": cmd_fmt, "mvhi": cmd_mvhi, "params": cmd_params, "zero-mv": cmd_zero_mv,
     "corrupt-weights": cmd_corrupt}[a.cmd](a)


if __name__ == "__main__":
    main()
