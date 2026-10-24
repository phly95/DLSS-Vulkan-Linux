#!/usr/bin/env python3
"""Compare two .hdump directories layer by layer.

    compare.py REF TEST [--map map.json] [--image-layer N] [--json out.json]

REF is the Windows side (capture or emulation), TEST the port. For every output tensor of every
layer: max / mean abs error and PSNR; for e4m3 tensors the ulp histogram and the fp8 tolerance gate
(every element within 1 ulp, at least 99% exact). The last layer's first output (or --image-layer)
is also reported as the final image. Exit status 1 when any layer fails its gate.

--map: {"<test layer index>": <ref layer index>, ...} for layers that are not one-to-one.
"""
import argparse
import json
import math
import sys

import numpy as np

import hdump

FP8_EXACT_MIN = 0.99
FP8_MAX_ULPS = 1


def psnr(ref, test, peak):
    mse = float(np.mean((ref - test) ** 2))
    if mse == 0:
        return math.inf
    if peak == 0:
        return -math.inf
    return 10 * math.log10(peak * peak / mse)


def compare_tensor(rt, tt, ref_raw, test_raw):
    if ref_raw.size != test_raw.size:
        return {"error": f"size {ref_raw.size} vs {test_raw.size}", "pass": False}
    ref_raw, test_raw = ref_raw.reshape(-1), test_raw.reshape(-1)
    r = hdump.decode(ref_raw, rt["dtype"])
    t = hdump.decode(test_raw, tt["dtype"])
    nan_r, nan_t = np.isnan(r), np.isnan(t)
    nan_mismatch = int(np.count_nonzero(nan_r != nan_t))
    ok = ~(nan_r | nan_t)
    d = np.abs(r[ok] - t[ok])
    peak = float(np.max(np.abs(r[ok]))) if ok.any() else 0.0
    out = {
        "name": rt["name"], "dtype": rt["dtype"], "count": int(ref_raw.size),
        "max_abs": float(d.max()) if d.size else 0.0,
        "mean_abs": float(d.mean()) if d.size else 0.0,
        "psnr_db": psnr(r[ok], t[ok], peak), "peak": peak,
        "nan_mismatch": nan_mismatch,
    }
    if rt["dtype"] == "e4m3" and tt["dtype"] == "e4m3":
        u = hdump.e4m3_ulps(ref_raw, test_raw)
        out["ulps"] = {"0": int(np.count_nonzero(u == 0)), "1": int(np.count_nonzero(u == 1)),
                       "2": int(np.count_nonzero(u == 2)), ">2": int(np.count_nonzero(u > 2))}
        exact = out["ulps"]["0"] / max(1, u.size)
        out["exact_fraction"] = exact
        out["pass"] = bool(nan_mismatch == 0 and u.max(initial=0) <= FP8_MAX_ULPS
                           and exact >= FP8_EXACT_MIN)
    else:
        # Non-fp8 tensors feed an fp8 quantisation or the image; a layer passes when it stays within
        # half an e4m3 ulp of the reference's own magnitude (2^-4 relative), i.e. it cannot move a
        # downstream e4m3 value by more than one step.
        rel = d / np.maximum(np.abs(r[ok]), 2.0 ** -9)
        out["max_rel"] = float(rel.max()) if rel.size else 0.0
        out["pass"] = bool(nan_mismatch == 0 and out["max_rel"] <= 2.0 ** -4)
    return out


def fmt_db(x):
    return "inf" if x == math.inf else f"{x:6.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref")
    ap.add_argument("test")
    ap.add_argument("--map")
    ap.add_argument("--image-layer", type=int)
    ap.add_argument("--json")
    a = ap.parse_args()
    ref, test = hdump.Dump(a.ref), hdump.Dump(a.test)
    mapping = {int(k): v for k, v in json.load(open(a.map)).items()} if a.map else {}
    report = {"ref": ref.manifest.get("source"), "test": test.manifest.get("source"), "layers": []}
    failed = 0
    print(f"ref: {a.ref} ({report['ref']})   test: {a.test} ({report['test']})")
    print(f"{'layer':>5}  {'kernel':48} {'tensor':14} {'dtype':5} {'max abs':>10} {'PSNR dB':>8}  fp8 exact  result")
    for tl in test.layers:
        ri = mapping.get(tl["index"], tl["index"])
        if ri >= len(ref.layers):
            print(f"{tl['index']:5}  {tl['kernel'][:48]:48} no reference layer {ri}")
            failed += 1
            continue
        rl = ref.layers[ri]
        routs = [t for t in rl["tensors"] if t["role"] == "out"]
        touts = [t for t in tl["tensors"] if t["role"] == "out"]
        for rt, tt in zip(routs, touts):
            res = compare_tensor(rt, tt, ref.tensor(rt), test.tensor(tt))
            res.update({"layer": tl["index"], "ref_layer": ri, "kernel": tl["kernel"]})
            report["layers"].append(res)
            if "error" in res:
                print(f"{tl['index']:5}  {tl['kernel'][:48]:48} {rt['name'][:14]:14} {res['error']}  FAIL")
                failed += 1
                continue
            ex = f"{100 * res['exact_fraction']:7.3f}%" if "exact_fraction" in res else "       -"
            verdict = "pass" if res["pass"] else "FAIL"
            failed += 0 if res["pass"] else 1
            print(f"{tl['index']:5}  {tl['kernel'][:48]:48} {rt['name'][:14]:14} {rt['dtype']:5} "
                  f"{res['max_abs']:10.3g} {fmt_db(res['psnr_db']):>8}  {ex}  {verdict}")
    # Final image.
    il = a.image_layer if a.image_layer is not None else len(test.layers) - 1
    if test.layers:
        tl = test.layers[il]
        rl = ref.layers[mapping.get(il, il)]
        tt = next(t for t in tl["tensors"] if t["role"] == "out")
        rt = next(t for t in rl["tensors"] if t["role"] == "out")
        r = hdump.decode(ref.tensor(rt).reshape(-1), rt["dtype"])
        t = hdump.decode(test.tensor(tt).reshape(-1), tt["dtype"])
        img = {"psnr_db": psnr(np.clip(r, 0, 1), np.clip(t, 0, 1), 1.0),
               "max_abs": float(np.max(np.abs(r - t)))}
        report["final_image"] = img
        print(f"final image (layer {il}): PSNR {fmt_db(img['psnr_db'])} dB (peak 1.0, clipped to [0,1]), "
              f"max abs {img['max_abs']:.4g}")
    print(f"{failed} failing tensor(s)")
    if a.json:
        json.dump(report, open(a.json, "w"), indent=1)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
