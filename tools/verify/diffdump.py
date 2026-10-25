#!/usr/bin/env python3
"""Compare a Vulkan kernel's output buffers with what NeuralHub's kernel wrote in the same launch
(srref / rrref --dump, tools/verify/oracle_dump.hpp), buffer by buffer.

    diffdump.py launches FRAME_DIR
        list the launches of a dumped frame: number, kernel, buffers written
    diffdump.py inputs FRAME_DIR --launch N --out DIR NAME...
        write DIR/NAME.bin: the bytes buffer NAME held just before launch N (start.* and every earlier
        launch's copies), the exact inputs a port kernel is run on
    diffdump.py compare FRAME_DIR --launch N PORT_DIR [--type NAME=T ...] [--emulate-e4m3 NAME ...]
                        [--layout NAME=C,W,H ...] [--json OUT]
        for every buffer launch N wrote: PORT_DIR/NAME.bin against the oracle's copy; prints element
        count, bit-exact count, max |error|, max ulp error, the first mismatch (and its (c, y, x) with
        --layout, which reads the buffer as [C/32][H][W][32]); exit 1 if any buffer is missing or not
        bit-exact. A buffer launch N did not change is not compared (the kernel did not write it).
    diffdump.py --self-test

Types T: u8, fp8 (OCP E4M3), f16, f32, u32 (default u8, raw bytes). --emulate-e4m3 NAME: the port's
NAME.bin holds f16 values (the lavapipe f16 build) and is rounded to E4M3 here, round to nearest even,
not saturating (|x| >= 464 -> NaN 0x7f, as v_cvt_pk_fp8_f32 without clamp); the report labels that
buffer EMULATED, because the rounding was not done by the kernel.
"""
import argparse
import json
import os
import re
import sys
import tempfile

import numpy as np

DT = {"u8": np.uint8, "fp8": np.uint8, "f16": np.float16, "f32": np.float32, "u32": np.uint32}


def e4m3_to_f32(b):
    b = np.asarray(b, np.uint8).astype(np.int32)
    s = np.where(b & 0x80, -1.0, 1.0)
    e = (b >> 3) & 15
    m = b & 7
    v = np.where(e == 0, m / 8.0 * 2.0 ** -6, (1 + m / 8.0) * np.power(2.0, e - 7.0))
    v = np.where((b & 0x7f) == 0x7f, np.nan, v)
    return (s * v).astype(np.float32)


_E4M3_POS = e4m3_to_f32(np.arange(0x7f, dtype=np.uint8)).astype(np.float64)   # 0 .. 448, increasing


def f32_to_e4m3_rne(x):
    """OCP E4M3, round to nearest even, no saturation: |x| >= 464 (half way past 448) and NaN -> 0x7f."""
    x = np.asarray(x, np.float64)
    a = np.abs(x)
    i = np.clip(np.searchsorted(_E4M3_POS, a), 1, 0x7e)
    lo, hi = _E4M3_POS[i - 1], _E4M3_POS[i]
    pick_hi = (a - lo > hi - a) | ((a - lo == hi - a) & ((i & 1) == 0))
    code = np.where(pick_hi, i, i - 1)
    code = np.where(a <= 0, 0, code)
    code = np.where((a >= 464.0) | np.isnan(a), 0x7f, code)
    sign = np.where(np.signbit(x), 0x80, 0)
    return (code | sign).astype(np.uint8)


def ordered(a, t):
    """integers whose difference counts ulps (sign-magnitude folded onto a line)"""
    if t in ("fp8",):
        u = a.astype(np.int64)
        return np.where(u & 0x80, -(u & 0x7f), u & 0x7f)
    if t == "f16":
        u = a.view(np.uint16).astype(np.int64)
        return np.where(u & 0x8000, -(u & 0x7fff), u & 0x7fff)
    if t == "f32":
        u = a.view(np.uint32).astype(np.int64)
        return np.where(u & 0x80000000, -(u & 0x7fffffff), u & 0x7fffffff)
    return a.astype(np.int64)


def values(a, t):
    if t == "fp8":
        return e4m3_to_f32(a).astype(np.float64)
    return a.astype(np.float64)


def read_launches(fd):
    p = os.path.join(fd, "launches.jsonl")
    if not os.path.exists(p):
        raise SystemExit(f"{p} missing (not an oracle dump frame)")
    return [json.loads(l) for l in open(p) if l.strip()]


def state_before(fd, name, n):
    """bytes of buffer `name` before launch n"""
    best = os.path.join(fd, f"start.{name}.bin")
    if not os.path.exists(best):
        raise SystemExit(f"{best} missing: {name} is not a dumped buffer")
    for L in read_launches(fd):
        if L["n"] >= n:
            break
        if name in L["wrote"]:
            best = os.path.join(fd, f"{L['n']:03d}.{name}.bin")
    return open(best, "rb").read()


def compare_one(name, port, ref, t, emulate, layout):
    rep = {"buffer": name, "type": t, "emulated": bool(emulate)}
    if emulate:
        pv = np.frombuffer(port, np.float16)
        port_arr = f32_to_e4m3_rne(pv.astype(np.float32))
        t = "fp8"
        rep["type"] = "fp8 (port f16, E4M3 RNE emulated here)"
    else:
        port_arr = np.frombuffer(port, DT[t])
    ref_arr = np.frombuffer(ref, DT[t])
    rep["elements"] = int(ref_arr.size)
    if port_arr.size != ref_arr.size:
        rep["error"] = f"size: port {port_arr.size} elements, oracle {ref_arr.size}"
        return rep
    pb, rb = port_arr.view(np.uint8 if t in ("u8", "fp8") else {"f16": np.uint16, "f32": np.uint32, "u32": np.uint32}[t]), \
        ref_arr.view(np.uint8 if t in ("u8", "fp8") else {"f16": np.uint16, "f32": np.uint32, "u32": np.uint32}[t])
    same = pb == rb
    rep["bit_exact"] = int(same.sum())
    pv, rv = values(port_arr, t), values(ref_arr, t)
    both_nan = np.isnan(pv) & np.isnan(rv)
    d = np.where(both_nan, 0.0, np.abs(pv - rv))
    d = np.where(np.isnan(d), np.inf, d)
    rep["max_abs_err"] = float(d.max()) if d.size else 0.0
    ul = np.abs(ordered(port_arr, t) - ordered(ref_arr, t))
    rep["max_ulp_err"] = int(ul.max()) if ul.size else 0
    bad = np.flatnonzero(~same)
    if bad.size:
        i = int(bad[0])
        first = {"index": i, "port": float(pv[i]), "oracle": float(rv[i]),
                 "port_bits": int(pb[i]), "oracle_bits": int(rb[i])}
        if layout:
            C, W, H = layout
            g, rem = divmod(i, H * W * 32)
            y, rem = divmod(rem, W * 32)
            x, c = divmod(rem, 32)
            first.update(c=g * 32 + c, y=y, x=x)
        rep["first_mismatch"] = first
    return rep


def cmd_compare(a):
    types = dict(kv.split("=", 1) for kv in a.type)
    layouts = {k: tuple(int(v) for v in s.split(",")) for k, s in (kv.split("=", 1) for kv in a.layout)}
    L = [l for l in read_launches(a.frame) if l["n"] == a.launch]
    if not L:
        raise SystemExit(f"no launch {a.launch} in {a.frame}")
    L = L[0]
    print(f"launch {a.launch}: {L['kernel']} grid {L['grid']} block {L['block']}")
    reports, ok = [], True
    if not L["wrote"]:
        print("  the oracle's launch changed no buffer: nothing to compare")
    for name in L["wrote"]:
        pf = os.path.join(a.port, f"{name}.bin")
        if not os.path.exists(pf):
            print(f"  {name}: MISSING in {a.port}")
            reports.append({"buffer": name, "error": "missing"})
            ok = False
            continue
        r = compare_one(name, open(pf, "rb").read(), open(os.path.join(a.frame, f"{L['n']:03d}.{name}.bin"), "rb").read(),
                        types.get(name, "u8"), name in a.emulate_e4m3, layouts.get(name))
        reports.append(r)
        if "error" in r:
            print(f"  {name}: {r['error']}")
            ok = False
            continue
        tag = " EMULATED" if r["emulated"] else ""
        line = (f"  {name} [{r['type']}]{tag}: {r['elements']} elements, bit-exact {r['bit_exact']} "
                f"({100.0 * r['bit_exact'] / max(1, r['elements']):.4f}%), max |err| {r['max_abs_err']:.6g}, "
                f"max ulp {r['max_ulp_err']}")
        if "first_mismatch" in r:
            line += f", first mismatch {r['first_mismatch']}"
        print(line)
        ok &= r["bit_exact"] == r["elements"]
    if a.json:
        json.dump(dict(frame=a.frame, launch=a.launch, kernel=L["kernel"], buffers=reports), open(a.json, "w"), indent=1)
    print("DIFFDUMP:", "BIT-EXACT" if ok else "DIFFERS")
    return 0 if ok else 1


def cmd_launches(a):
    for L in read_launches(a.frame):
        print(f"{L['n']:3d} {L['kernel']} grid {L['grid']} block {L['block']} wrote {', '.join(L['wrote'])}")
    return 0


def cmd_inputs(a):
    os.makedirs(a.out, exist_ok=True)
    for name in a.names:
        open(os.path.join(a.out, f"{name}.bin"), "wb").write(state_before(a.frame, name, a.launch))
        print(f"{a.out}/{name}.bin: {name} before launch {a.launch}")
    return 0


def self_test():
    ok = True

    def chk(what, cond):
        nonlocal ok
        print(f"  {what:58s} {'ok' if cond else 'WRONG'}")
        ok &= bool(cond)

    # E4M3: every code decodes and re-encodes to itself; ties go to even; overflow is NaN
    codes = np.arange(256, dtype=np.uint8)
    vals = e4m3_to_f32(codes)
    finite = ~np.isnan(vals)
    rt = f32_to_e4m3_rne(vals[finite])
    zero_ok = (codes[finite] == rt) | ((codes[finite] == 0x80) & (rt == 0x00))
    chk("e4m3 round trip of all 254 finite codes", np.all(zero_ok))
    chk("tie 1.0625 (between 1.0 and 1.125) -> 1.0 (even)", f32_to_e4m3_rne(np.array([1.0625]))[0] == 0x38)
    chk("tie 1.1875 -> 1.25 (even)", f32_to_e4m3_rne(np.array([1.1875]))[0] == 0x3a)
    chk("448 stays 448 (0x7e), 460 -> 448, 464 -> NaN (no saturation)",
        list(f32_to_e4m3_rne(np.array([448.0, 460.0, 464.0, -500.0]))) == [0x7e, 0x7e, 0x7f, 0xff])
    chk("smallest subnormal 2^-9 and half of it (tie -> 0)",
        list(f32_to_e4m3_rne(np.array([2.0 ** -9, 2.0 ** -10, 1.5 * 2.0 ** -10]))) == [0x01, 0x00, 0x01])
    with tempfile.TemporaryDirectory() as t:
        fd = os.path.join(t, "frame0")
        os.makedirs(fd)
        rng = np.random.default_rng(1)
        a0 = rng.integers(0, 256, 4096, dtype=np.uint8)
        a1 = rng.integers(0, 120, 4096, dtype=np.uint8)
        f = rng.standard_normal(256).astype(np.float32)
        a0.tofile(os.path.join(fd, "start.t0.bin"))
        np.zeros(4096, np.uint8).tofile(os.path.join(fd, "start.t1.bin"))
        np.zeros(256, np.float32).tofile(os.path.join(fd, "start.g.bin"))
        a1.tofile(os.path.join(fd, "000.t1.bin"))
        f.tofile(os.path.join(fd, "000.g.bin"))
        open(os.path.join(fd, "launches.jsonl"), "w").write(
            json.dumps(dict(n=0, kernel="k", grid=[1, 1, 1], block=[64, 1, 1], args="", pointers={}, wrote=["t1", "g"])) + "\n")
        chk("inputs before launch 0 = start copy", state_before(fd, "t1", 0) == bytes(4096))
        chk("inputs before launch 1 = launch 0's copy", state_before(fd, "t1", 1) == a1.tobytes())
        pd = os.path.join(t, "port")
        os.makedirs(pd)
        a1.tofile(os.path.join(pd, "t1.bin"))
        f.tofile(os.path.join(pd, "g.bin"))
        args = lambda extra: argparse.Namespace(frame=fd, launch=0, port=pd, type=["t1=fp8", "g=f32"], emulate_e4m3=extra,
                                                layout=["t1=32,8,16"], json=None)
        chk("identical buffers are bit-exact", cmd_compare(args([])) == 0)
        e4 = e4m3_to_f32(a1).astype(np.float16)
        e4.tofile(os.path.join(pd, "t1.bin"))
        chk("f16 port holding the same e4m3 values, emulated: bit-exact", cmd_compare(args(["t1"])) == 0)
        a2 = a1.copy()
        a2[1234] ^= 1
        a2.tofile(os.path.join(pd, "t1.bin"))
        r = compare_one("t1", a2.tobytes(), a1.tobytes(), "fp8", False, (32, 8, 16))
        chk("one flipped lsb: 1 mismatch, 1 ulp, located", r["bit_exact"] == 4095 and r["max_ulp_err"] == 1
            and r["first_mismatch"]["index"] == 1234 and (r["first_mismatch"]["x"], r["first_mismatch"]["c"]) == (6, 18))
        chk("a differing buffer fails compare", cmd_compare(args([])) == 1)
        os.remove(os.path.join(pd, "g.bin"))
        chk("a missing buffer fails compare", cmd_compare(args([])) == 1)
    print("DIFFDUMP SELF-TEST:", "PASS" if ok else "FAIL")
    return ok


def main():
    if sys.argv[1:] == ["--self-test"]:
        sys.exit(0 if self_test() else 1)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("launches"); c.add_argument("frame")
    c = sub.add_parser("inputs"); c.add_argument("frame"); c.add_argument("--launch", type=int, required=True)
    c.add_argument("--out", required=True); c.add_argument("names", nargs="+")
    c = sub.add_parser("compare"); c.add_argument("frame"); c.add_argument("--launch", type=int, required=True)
    c.add_argument("port"); c.add_argument("--type", action="append", default=[])
    c.add_argument("--emulate-e4m3", action="append", default=[]); c.add_argument("--layout", action="append", default=[])
    c.add_argument("--json")
    a = ap.parse_args()
    for kv in getattr(a, "type", []):
        if kv.split("=", 1)[-1] not in DT:
            raise SystemExit(f"--type {kv}: types are {', '.join(DT)}")
    sys.exit({"launches": cmd_launches, "inputs": cmd_inputs, "compare": cmd_compare}[a.cmd](a))


if __name__ == "__main__":
    main()
