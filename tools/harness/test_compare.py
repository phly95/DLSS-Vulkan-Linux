#!/usr/bin/env python3
"""Self-test of compare.py: known injected errors must be reported as injected."""
import math
import os
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import hdump  # noqa: E402
import compare  # noqa: E402


def e4m3_encode(x):
    """Nearest e4m3 byte, ties to even mantissa (brute force over the table)."""
    t = hdump.E4M3.astype(np.float64)
    finite = np.where(np.isnan(t), np.inf, t)
    idx = np.abs(x.reshape(-1, 1) - finite.reshape(1, -1)).argmin(axis=1)
    return idx.astype(np.uint8).reshape(x.shape)


def main():
    rng = np.random.default_rng(1234)
    # e4m3 table facts from the OCP spec.
    assert hdump.E4M3[0x7E] == 448.0 and np.isnan(hdump.E4M3[0x7F]) and hdump.E4M3[0x01] == 2.0 ** -9
    assert hdump.E4M3[0x38] == 1.0 and hdump.E4M3[0xB8] == -1.0
    assert hdump.e4m3_ulps(np.array([0x38], np.uint8), np.array([0x39], np.uint8))[0] == 1
    assert hdump.e4m3_ulps(np.array([0x01], np.uint8), np.array([0x81], np.uint8))[0] == 2

    # PSNR of a known MSE.
    ref = rng.random(10000)
    noise = rng.standard_normal(10000) * 1e-3
    mse = float(np.mean(noise ** 2))
    got = compare.psnr(ref, ref + noise, 1.0)
    want = 10 * math.log10(1.0 / mse)
    assert abs(got - want) < 1e-9, (got, want)

    with tempfile.TemporaryDirectory() as d:
        a, b = os.path.join(d, "a"), os.path.join(d, "b")
        wa, wb = hdump.Writer(a, "emulation"), hdump.Writer(b, "vulkan")
        x8 = e4m3_encode(rng.standard_normal((64, 32)))
        # Layer 0: identical e4m3.
        wa.layer("k0", [("x", "out", "e4m3", x8)])
        wb.layer("k0", [("x", "out", "e4m3", x8.copy())])
        # Layer 1: 0.5% of elements one ulp off: passes.
        y = x8.copy().reshape(-1)
        sel = rng.choice(y.size, size=10, replace=False)
        y[sel] = np.where(y[sel] & 0x7F < 0x7E, y[sel] + 1, y[sel] - 1).astype(np.uint8)
        wa.layer("k1", [("x", "out", "e4m3", x8)])
        wb.layer("k1", [("x", "out", "e4m3", y.reshape(x8.shape))])
        # Layer 2: one element two ulps off: fails.
        z = x8.copy().reshape(-1)
        z[5] = (z[5] + 2) if (z[5] & 0x7F) < 0x7C else (z[5] - 2)
        wa.layer("k2", [("x", "out", "e4m3", x8)])
        wb.layer("k2", [("x", "out", "e4m3", z.reshape(x8.shape))])
        # Layer 3: final image f32 with known noise.
        img = rng.random((16, 16, 4)).astype(np.float32) * 0.9
        wa.layer("post", [("img", "out", "f32", img)])
        wb.layer("post", [("img", "out", "f32", img + np.float32(1e-4))])
        wa.close()
        wb.close()
        out = os.path.join(d, "r.json")
        p = subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), "compare.py"),
                            a, b, "--json", out], capture_output=True, text=True)
        print(p.stdout)
        import json
        r = json.load(open(out))
        res = {x["layer"]: x for x in r["layers"]}
        assert res[0]["pass"] and res[0]["psnr_db"] == math.inf
        assert res[1]["pass"] and res[1]["ulps"]["1"] == 10, res[1]
        assert not res[2]["pass"] and res[2]["ulps"]["2"] == 1, res[2]
        assert res[3]["pass"]
        # 1e-4 everywhere (f32 rounding of the add aside): 80 dB.
        assert abs(r["final_image"]["psnr_db"] - 80.0) < 0.01, r["final_image"]
        assert p.returncode == 1  # layer 2 fails
    print("compare self-test: OK")


if __name__ == "__main__":
    main()
