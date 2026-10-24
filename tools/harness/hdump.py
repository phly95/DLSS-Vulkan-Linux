"""Read and write .hdump directories (format: docs/harness/PLAN.md)."""
import json
import os

import numpy as np

# E4M3 (OCP FP8, the format of v_cvt_pk_fp8_f32 / VK_EXT_shader_float8 FloatE4M3EXT): 1 sign, 4 exponent
# (bias 7), 3 mantissa bits, no infinities, 0x7f / 0xff are NaN, max finite 448.
def _e4m3_table():
    t = np.zeros(256, dtype=np.float32)
    for b in range(256):
        s = -1.0 if b & 0x80 else 1.0
        e = (b >> 3) & 0xF
        m = b & 0x7
        if e == 0xF and m == 0x7:
            t[b] = np.nan
        elif e == 0:
            t[b] = s * m * 2.0 ** -9
        else:
            t[b] = s * (1 + m / 8.0) * 2.0 ** (e - 7)
    return t


E4M3 = _e4m3_table()

DTYPES = {"e4m3": np.uint8, "u8": np.uint8, "f16": np.float16, "f32": np.float32,
          "u16": np.uint16, "u32": np.uint32, "i32": np.int32}


def decode(raw, dtype):
    """Raw element array -> float64 values (e4m3 through the table)."""
    if dtype == "e4m3":
        return E4M3[raw].astype(np.float64)
    return raw.astype(np.float64)


def e4m3_ulps(a, b):
    """Distance in representable e4m3 steps between two byte arrays (sign-magnitude order)."""
    def order(x):
        x = x.astype(np.int32)
        mag = x & 0x7F
        return np.where(x & 0x80, -mag, mag)
    return np.abs(order(a) - order(b))


class Dump:
    def __init__(self, path):
        self.path = path
        with open(os.path.join(path, "manifest.json")) as f:
            self.manifest = json.load(f)
        self.layers = self.manifest["layers"]

    def tensor(self, t):
        raw = np.fromfile(os.path.join(self.path, t["file"]), dtype=DTYPES[t["dtype"]])
        shape = t.get("shape")
        if shape:
            raw = raw.reshape(shape)
        return raw


class Writer:
    def __init__(self, path, source, frame=0):
        os.makedirs(path, exist_ok=True)
        self.path = path
        self.manifest = {"source": source, "frame": frame, "layers": []}

    def layer(self, kernel, tensors, grid=None, block=None):
        """tensors: list of (name, role, dtype, ndarray)."""
        i = len(self.manifest["layers"])
        entry = {"index": i, "kernel": kernel, "grid": grid, "block": block, "tensors": []}
        for n, (name, role, dtype, arr) in enumerate(tensors):
            fn = f"L{i:04d}_{role}{n}.bin"
            np.ascontiguousarray(arr, dtype=DTYPES[dtype]).tofile(os.path.join(self.path, fn))
            entry["tensors"].append({"name": name, "role": role, "dtype": dtype,
                                     "shape": list(arr.shape), "file": fn})
        self.manifest["layers"].append(entry)

    def close(self):
        with open(os.path.join(self.path, "manifest.json"), "w") as f:
            json.dump(self.manifest, f, indent=1)
