#!/usr/bin/env python3
"""tools/rr/rr_layout.py: the byte map of every DLSS-RR weight blob (blob0..blob15), read off
NeuralHub's own packer (tools/rr/rr_pack.py).

    build/venv/bin/python -I tools/rr/rr_layout.py [--markdown] [--packed DIR]

Unit k runs rr_pack.build_w(T, k) and rr_pack.pack(W, dither) on a fake table: T.f16(key, n) returns
n zero halfs, where n is the count the packer asks for, and nothing else is copied from the packer.
Every write into the blob is recorded (offset, length, encoding, the W member it belongs to, the
table key that member was read from, and the permutations applied to it), so the map follows the
packer rather than a reading of it. The labels below are checked against the recording: every table
read, every item offset the packer returns and every byte of the blob must be accounted for, or the
script stops. No weight value is read; nothing NVIDIA's is involved.

--packed DIR also compares the sizes and the 21 item offsets of each unit with DIR/params.json,
rr_pack.py's own output for the real tensors (its counts are the ones checked here).

Encoding rules (each one is checked numerically by check_rules()):
  fp8 fragment  frag_rr(m, permute_rows): rows are padded to a multiple of 16 with zeros, then element
                (r, c) of the padded matrix is at byte FRAG(r, c) = 16 (u & 15) + 256 bit2(c) +
                8 bit3(c) + 4 bit1(c) + 2 bit4(c) + bit0(c) + 512 (c >> 5) + 16 cols (u >> 4), where
                u = u(r) = r without permute_rows, and u(r) = P^-1(r) with it. With permute_rows the
                row count must be a multiple of 32 (frag_rr refuses other sizes).
  f16 P         f16_rr(v): n % 32 == 0 -> blob position i holds logical element P(i) of its group of
                32, P(i) = (i & 1) | ((i & 2) << 3) | ((i & 0x1c) >> 1) | (i & 0xe0). Otherwise the
                halfs are stored plain.
  perm_*(C)     perm_cols / perm_rows / perm_vec: blob position j holds logical element
                j // C + 4 (j % C) of its column, row or entry, so logical element L is stored at
                j = (L % 4) * C + L // 4. The C is the width of the layer whose 4C input it is.
  raw           bytes as given: the symmetric gram matrices (plain f16 64 x 64, row-major), the
                16-half head bias block and the dither table.
  Blob          16-byte aligned items (B.align() before each), offsets in params.json's meta.
"""
import json
import linecache
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "sr"))
import rr_pack  # noqa: E402
import sr_pack  # noqa: E402  (frag_offsets: SR's fragment order, for the comparison in check_rules)

BLOCK = rr_pack.BLOCK
DITHER_BYTES = 4096    # rr_weights.py: the dither table is 4096 bytes (the same table SR uses)


def P(i):
    return (i & 1) | ((i & 2) << 3) | ((i & 0x1C) >> 1) | (i & 0xE0)


def u_of(r, permute=True):
    """frag_rr's row index (the packer's own expression)."""
    return (((r & 0x7FFFFFE1) | ((r >> 3) & 2)) + (r & 0xE) * 2) if permute else r


def frag_closed(rows, cols, permute):
    """FRAG(r, c) above, for a padded rows x cols matrix."""
    r = np.arange(rows, dtype=np.int64)[:, None]
    c = np.arange(cols, dtype=np.int64)[None, :]
    u = u_of(r, permute)

    def bit(x, k):
        return (x >> k) & 1

    return (16 * (u & 15) + 256 * bit(c, 2) + 8 * bit(c, 3) + 4 * bit(c, 1) + 2 * bit(c, 4) + bit(c, 0)
            + 512 * (c >> 5) + 16 * cols * (u >> 4))


def probe(fn, rows, cols):
    """Byte offset of every element of fn(matrix), found with unit matrices (no formula used)."""
    off = np.zeros((rows, cols), np.int64)
    for r in range(rows):
        for c in range(cols):
            m = np.zeros((rows, cols), np.uint8)
            m[r, c] = 1
            hit = np.flatnonzero(np.frombuffer(fn(m), np.uint8))
            if hit.size != 1:
                raise SystemExit(f"probe ({r},{c}) lands on {hit.size} bytes")
            off[r, c] = hit[0]
    return off


def check_rules():
    """Each stated rule, against the packer's own functions. Prints one line per rule to stderr."""
    def say(text, ok):
        print(f"  rule {text}: {'PASS' if ok else 'FAIL'}", file=sys.stderr)
        if not ok:
            raise SystemExit(f"rule failed: {text}")

    inv_ok = all(P(u_of(r)) == r for r in range(128))
    say("u(r) = P^-1(r) on rows 0..127 (fragment rows and f16 vectors share one permutation)", inv_ok)

    ok = True
    for n in (32, 64, 96, 128, 160):
        out = np.frombuffer(rr_pack.f16_rr(np.arange(n, dtype=np.uint16)), np.uint16)
        ok &= np.array_equal(out, np.array([P(i) for i in range(n)], np.uint16))
    for n in (16, 48, 80):
        out = np.frombuffer(rr_pack.f16_rr(np.arange(n, dtype=np.uint16)), np.uint16)
        ok &= np.array_equal(out, np.arange(n, dtype=np.uint16))
    say("f16_rr: blob position i holds element P(i) for n % 32 == 0, plain otherwise", ok)

    ok = True
    for C in (32, 64, 96, 128, 160):
        for name, fn, axis in (("perm_cols", rr_pack.perm_cols, 1), ("perm_rows", rr_pack.perm_rows, 0)):
            m = np.random.default_rng(C).integers(0, 255, (4 * C, 3) if axis == 0 else (3, 4 * C)).astype(np.uint8)
            out = fn(m, C)
            L = np.arange(4 * C)
            src = L // C + 4 * (L % C)                      # blob position j holds logical src[j]
            pos = (L % 4) * C + L // 4                      # logical L sits at blob position pos[L]
            if axis == 0:
                ok &= np.array_equal(out, m[src]) and np.array_equal(out[pos], m)
            else:
                ok &= np.array_equal(out, m[:, src]) and np.array_equal(out[:, pos], m)
        v = np.random.default_rng(C + 1).integers(0, 65535, 4 * C).astype(np.uint16)
        out = rr_pack.perm_vec(v, C)
        ok &= np.array_equal(out, v[src]) and np.array_equal(out[pos], v)
    say("perm_cols / perm_rows / perm_vec: position j holds j//C + 4(j%C), and L is at (L%4)*C + L//4", ok)

    ok = True
    for rows, cols, permute in ((16, 32, False), (3, 32, False), (4, 96, False), (64, 64, False),
                                (32, 32, True), (64, 32, True), (32, 96, True)):
        r16 = (rows + 15) // 16 * 16
        got = probe(lambda m: rr_pack.frag_rr(m, permute), rows, cols)
        ok &= np.array_equal(got, frag_closed(r16, cols, permute)[:rows])
    say("frag_rr: element (r, c) at FRAG(r, c) for 16x32, 3x32 (padded), 4x96 (padded), 64x64, "
        "32x32 / 64x32 / 32x96 with permute_rows", ok)

    refused = False
    try:
        rr_pack.frag_rr(np.zeros((16, 32), np.uint8), True)
    except SystemExit:
        refused = True
    say("frag_rr with permute_rows refuses 16 rows (not a multiple of 32): the packer's own check", refused)

    # the composed placements: build_w's permutation first, then frag_rr / f16_rr's own
    ok = True
    for C in (32, 64):
        pos = (np.arange(4 * C) % 4) * C + np.arange(4 * C) // 4          # logical L is stored at pos[L]
        got = probe(lambda m: rr_pack.frag_rr(rr_pack.perm_cols(m, C), True), 64, 4 * C)
        ok &= np.array_equal(got, frag_closed(64, 4 * C, True)[:, pos])   # (R, L) at FRAG(R, pos[L])
        got = probe(lambda m: rr_pack.frag_rr(rr_pack.perm_rows(m, C), True), 4 * C, 32)
        ok &= np.array_equal(got, frag_closed(4 * C, 32, True)[pos])      # (R, c) at FRAG(pos[R], c)
        out = rr_pack.f16_rr(rr_pack.perm_vec(np.arange(4 * C, dtype=np.uint16), C))
        blob_index = np.argsort(np.frombuffer(out, np.uint16))            # where logical L is stored
        ok &= np.array_equal(blob_index, u_of(pos, True))                 # at u(pos[L]) = P^-1(pos[L])
    say("composed: perm_cols then frag_rr puts (R, L) at FRAG(R, pos L); perm_rows then frag_rr puts "
        "(R, c) at FRAG(pos R, c); perm_vec then f16_rr puts L at P^-1(pos L) (C = 32, 64)", ok)

    same = int(np.sum(frag_closed(64, 32, False) == sr_pack.frag_offsets(64, 32, 1)))
    print(f"  info: RR and SR fragment orders agree on {same} of 2048 positions (64x32, no row permutation)",
          file=sys.stderr)


class _FakeTable:
    """rr_pack.Table for build_w: f16(key, n) is n zero halfs, u16(off) is 0. Logs every read."""

    def __init__(self):
        self.f16_reads, self.u16_reads = [], []

    def f16(self, key, n):
        self.f16_reads.append((key, int(n)))
        return np.zeros(int(n), np.uint16)

    def u16(self, off):
        self.u16_reads.append(int(off))
        return 0


def _geometry(k):
    L = k if k < 6 else ((k - 6) >> 1) + 6
    odd = (k & 1) if k > 6 else 0
    return L, odd, rr_pack.C_OF[L], rr_pack.G_OF[L]


def _sources(k):
    """The table reads build_w makes for unit k: {member: (key, count)} and the scalar reads."""
    L, odd, C, g = _geometry(k)
    b = L * BLOCK
    o = b + odd * 0xA0
    src = {"groups": (o + 0x50, C * g * 64), "gram": (o + 0x80, g * 4096), "m48": (o + 0x60, C * g * 32),
           "v128": (o + 0x70, C), "m68": (o + 0xA0, 4 * C * C), "v140": (o + 0xB0, 4 * C),
           "m88": (o + 0xC0, 4 * C * C), "v158": (o + 0xD0, C)}
    scal16 = [(b + 0x1E0, 1)]                                    # s1d0
    if k < 6:
        if 1 <= k <= 4:
            cd = rr_pack.CDOWN[L]
            src["mA8"] = (b + 0x180, 4 * C * cd)
            src["v170"] = (b + 0x190, cd)
        if k == 1:
            src["mC8"] = (0x180, 0x2000)
            src["v188"] = (0x190, 0x40)
        src["m108"] = (b + 0x1C0, 3 * C)
        src["v1b8"] = (b + 0x1D0, 3)
        if k == 0:
            src["m1d8"] = (0x0, 0xAC0)
            src["v218"] = (0x10, 0x40)
            src["m1f8"] = (0x20, 0x800)
            src["v230"] = (0x30, 0x20)
    else:
        cu = rr_pack.CUP[(k - 6) >> 1]
        if odd == 0:
            src["mE8"] = (b + 0x1A0, 4 * C * cu)
            src["v1a0"] = (b + 0x1B0, 4 * C)
        if L != 10:
            src["m108"] = (b + 0x1C0, 4 * C)
            src["v1b8"] = (b + 0x1D0, 4)
    if k == 15:
        scal16 += [(0x1820, 1), (0x1830, 1), (0x1840, 1)]     # dec0 scalars (params.json, not the blob)
    scal_u16 = [b + odd * 4 + 0x220, b + odd * 4 + 0x222] + ([b + 0x228] if k >= 6 else [])
    return src, scal16, scal_u16


# the W member each write comes from -> its packer label, the item index it is stored at (meta[i]),
# and the packer's own description. The packer's W+0x comments are quoted where it has them.
INFO = {
    "groups": ("fp8 group {i}", "W+0x18: 64 x C; rows {a}..{a1} and {b}..{b1} of the table slot (g = {g})"),
    "gram": ("f16 symmetric {j}", "W+0x30: 64 x 64, (m + m^T) * 0.5 * 2 sqrt(2) (sym_scaled, FUN_18012dfc0); "
                                  "plain row-major, no fragment"),
    "m48": ("fp8 matrix", "W+0x48: C x 32g; semantics not yet decoded"),
    "v128": ("f16 vector", "W+0x128: C; semantics not yet decoded"),
    "m68": ("fp8 matrix", "W+0x68: 4C x C; semantics not yet decoded"),
    "v140": ("f16 vector", "W+0x140: 4C; semantics not yet decoded"),
    "m88": ("fp8 matrix", "W+0x88: C x 4C; semantics not yet decoded"),
    "v158": ("f16 vector", "W+0x158: C; semantics not yet decoded"),
    "mA8": ("fp8 down-projection", "W+0xa8: Cdown x 4C, columns permuted by the unit's C (HOST-RE: encoder "
                                   "down-projection)"),
    "v170": ("f16 vector", "W+0x170: Cdown; semantics not yet decoded"),
    "mC8": ("fp8 level-0 down-projection", "W+0xc8: 64 x 128, columns permuted by C = 32, the level-0 width "
                                           "(HOST-RE: unit 1 holds level 0's down-projection)"),
    "v188": ("f16 vector", "W+0x188: 64, from absolute 0x190 (level 0's block); semantics not yet decoded"),
    "mE8": ("fp8 up-projection", "W+0xe8: 4C x Cup, rows permuted by the unit's C (HOST-RE: decoder up-projection)"),
    "v1a0": ("f16 vector", "W member v1a0: 4C, permuted with perm_vec (the unit's C); semantics not yet decoded"),
    "m108": ("fp8 head", "W+0x108: 3 x C (encoders) or 4 x C (decoders but dec0), rows padded to 16, not "
                         "permuted (HOST-RE: head)"),
    "head": ("f16 head bias", "W+0x1b8: the 3 or 4 values of W member v1b8, then zeros to 16 halfs, plain "
                              "(HOST-RE: 16-value bias block)"),
    "m1d8": ("fp8 enc0 feature matrix", "W+0x1d8: 64 x 43 zero-padded to 64 x 64 (HOST-RE: unit 0 feature "
                                        "matrices)"),
    "v218": ("f16 vector", "W member v218: 64 from absolute 0x10; semantics not yet decoded"),
    "m1f8": ("fp8 feature matrix", "W member m1f8: 32 x 64 from absolute 0x20 (HOST-RE: unit 0 feature "
                                   "matrix); semantics not yet decoded"),
    "v230": ("f16 vector", "W member v230: 32 from absolute 0x30; semantics not yet decoded"),
    "dither": ("dither table", "4096 raw bytes, the table sr_weights.py extracts (packer name: dither)"),
}
# meta[] index each member is stored at (rr_pack.pack); gram[0] is meta[1], groups[0] is meta[0]
ITEM = {"groups[0]": 0, "gram[0]": 1, "m48": 2, "v128": 3, "m68": 4, "v140": 5, "m88": 6, "v158": 7,
        "mA8": 8, "v170": 9, "mC8": 10, "v188": 11, "mE8": 12, "v1a0": 13, "m108": 14, "head": 15,
        "m1d8": 16, "v218": 17, "m1f8": 18, "v230": 19, "dither": 20}
RAW_LINE = {"B.put(gr.tobytes())": "gram", "meta[15] = B.put(head.tobytes())": "head",
            "meta[20] = B.put(dither)": "dither"}
SOURCE_OF = {"groups": "groups", "gram": "gram", "head": "v1b8"}      # member -> W member read for it


class _Recorder:
    """Patches the packer's permutations, fragment / f16 writers and Blob while one unit is packed. A write
    is named by the W member its array came from (frag_rr / f16_rr receive the member itself), a raw
    write by the source line rr_pack.pack() calls Blob.put from."""

    def __init__(self):
        self.name_of, self.prov, self.keep = {}, {}, []
        self.pending, self.events, self.gram_seen = None, [], 0

    def _perm(self, name, fn):
        def wrapped(m, C):
            out = fn(m, C)
            self.prov[id(out)] = (name, C)
            self.keep.append(out)
            return out
        return wrapped

    def name_members(self, W):
        for key, val in W.items():
            if isinstance(val, np.ndarray):
                self.name_of[id(val)] = key
                self.keep.append(val)
            elif isinstance(val, list):
                for i, a in enumerate(val):
                    if isinstance(a, np.ndarray):
                        self.name_of[id(a)] = f"{key}[{i}]"
                        self.keep.append(a)

    def patched(self):
        rec = self

        class _Ctx:
            def __enter__(self_):
                R = rr_pack
                self_.saved = (R.frag_rr, R.f16_rr, R.perm_cols, R.perm_rows, R.perm_vec,
                               R.Blob.put, R.Blob.align)
                frag0, f160, put0, align0 = R.frag_rr, R.f16_rr, R.Blob.put, R.Blob.align

                def frag(m, permute_rows=True):
                    name = rec.name_of.get(id(m))
                    if name is None:
                        raise SystemExit("frag_rr called on an array that is not a W member")
                    out = frag0(m, permute_rows)
                    rec.pending = dict(kind="frag", name=name, arr=m, rows0=m.shape[0], cols=m.shape[1],
                                       permute=permute_rows, n=len(out))
                    return out

                def f16(v):
                    name = rec.name_of.get(id(v))
                    if name is None:
                        raise SystemExit("f16_rr called on an array that is not a W member")
                    out = f160(v)
                    rec.pending = dict(kind="f16", name=name, arr=v, count=int(np.asarray(v).size), n=len(out))
                    return out

                def put(blob, data):
                    frame = sys._getframe(1)
                    line = linecache.getline(frame.f_code.co_filename, frame.f_lineno).strip()
                    off = len(blob.b)
                    res = put0(blob, data)
                    pend, rec.pending = rec.pending, None
                    if pend is not None:
                        if pend["n"] != len(data):
                            raise SystemExit(f"write of {pend['name']} is {len(data)} bytes")
                        ev = dict(pend)
                    else:
                        if line == "B.put(gr.tobytes())":
                            name = f"gram[{rec.gram_seen}]"
                            rec.gram_seen += 1
                        elif line in RAW_LINE:
                            name = RAW_LINE[line]
                        else:
                            raise SystemExit(f"rr_pack.py:{frame.f_lineno}: unlabelled raw write: {line}")
                        ev = dict(kind="raw", name=name, arr=None, n=len(data))
                    ev.update(off=off, line=line, base=name_base(ev["name"]))
                    rec.events.append(ev)
                    return res

                def align(blob):
                    pad = -len(blob.b) % 16
                    if pad:
                        rec.events.append(dict(kind="pad", name=None, arr=None, n=pad, off=len(blob.b), line=""))
                    align0(blob)

                R.frag_rr, R.f16_rr = frag, f16
                R.perm_cols = rec._perm("perm_cols", R.perm_cols)
                R.perm_rows = rec._perm("perm_rows", R.perm_rows)
                R.perm_vec = rec._perm("perm_vec", R.perm_vec)
                R.Blob.put, R.Blob.align = put, align
                return self_

            def __exit__(self_, *exc):
                R = rr_pack
                R.frag_rr, R.f16_rr, R.perm_cols, R.perm_rows, R.perm_vec, R.Blob.put, R.Blob.align = self_.saved
                return False

        return _Ctx()


def name_base(name):
    return None if name is None else name.split("[")[0]


def _unit(k, dither):
    """Packs unit k on the fake table, recording every write; returns the record and the blob."""
    T = _FakeTable()
    rec = _Recorder()
    with rec.patched():
        W = rr_pack.build_w(T, k)
        rec.name_members(W)
        blob, meta = rr_pack.pack(W, dither)
    return W, T, rec, blob, meta


def _check_unit(k, T, rec, blob, meta):
    src, scal16, scal_u16 = _sources(k)
    if sorted(T.f16_reads) != sorted(list(src.values()) + scal16):
        raise SystemExit(f"unit {k}: table reads {sorted(T.f16_reads)} differ from the labels")
    if sorted(T.u16_reads) != sorted(scal_u16):
        raise SystemExit(f"unit {k}: u16 reads {sorted(T.u16_reads)} differ from the labels")
    pos = 0
    for e in rec.events:
        if e["off"] != pos:
            raise SystemExit(f"unit {k}: write at {e['off']} does not follow {pos}")
        pos += e["n"]
    if pos != len(blob) or len(blob) % 16:
        raise SystemExit(f"unit {k}: writes cover {pos} of {len(blob)} bytes")
    for e in rec.events:
        item = ITEM.get(e["name"]) if e["name"] else None
        if item is not None and meta[item] != e["off"]:
            raise SystemExit(f"unit {k}: {e['name']} at {e['off']}, meta[{item}] = {meta[item]}")
    for item, off in enumerate(meta):
        if off != -1 and not any(ITEM.get(e.get("name") or "") == item for e in rec.events):
            raise SystemExit(f"unit {k}: meta[{item}] = {off} has no write")
    if meta[1] != next(e["off"] for e in rec.events if e.get("name") == "gram[0]"):
        raise SystemExit(f"unit {k}: meta[1] is not the first gram")
    return src


def _perm_text(rec, arr):
    p = rec.prov.get(id(arr)) if arr is not None else None
    if p is None:
        return None
    name, C = p
    fn = {"perm_cols": "FUN_18012e4b0", "perm_rows": "FUN_18012e5e0", "perm_vec": "FUN_18012e6f0"}[name]
    return f"{name}(C={C}) in build_w [{fn}]"


def unit_rows(k, dither):
    """The table rows of unit k and its item offsets, from the recording."""
    W, T, rec, blob, meta = _unit(k, dither)
    src = _check_unit(k, T, rec, blob, meta)
    L, odd, C, g = _geometry(k)
    rows = []
    for e in rec.events:
        if e["kind"] == "pad":
            continue
        base = name_base(e["name"])
        label, note = INFO[base]
        idx = re.search(r"\[(\d+)\]", e["name"] or "")
        i = int(idx.group(1)) if idx and base == "groups" else 0
        j = int(idx.group(1)) if idx and base == "gram" else 0
        region = label.format(i=i, j=j)
        note = note.format(a=32 * i, a1=32 * i + 31, b=32 * (g + i), b1=32 * (g + i) + 31, g=g)
        src_name = SOURCE_OF.get(base, base)
        if src_name in src:
            key, n = src[src_name]
            source = f"{key:#x} x {n}"
        else:
            source = "dither.bin"
        if e["kind"] == "frag":
            r16 = (e["rows0"] + 15) // 16 * 16
            enc = f"fp8 frag {r16}x{e['cols']}" + (f" (rows {e['rows0']} padded)" if r16 != e["rows0"] else "")
            perms = [_perm_text(rec, e["arr"])]
            perms.append("rows u(r) [frag_rr FUN_180133070]" if e["permute"] else "none (permute_rows=False)")
        elif e["kind"] == "f16":
            perm_ok = e["count"] % 32 == 0
            enc = f"f16 {'permuted' if perm_ok else 'plain'} {e['count']}"
            perms = [_perm_text(rec, e["arr"]), "P within 32 [f16_rr FUN_1801336e0]" if perm_ok else "none"]
        elif base == "gram":
            enc = "f16 plain 64x64 row-major"
            perms = ["none"]
            source = f"{src['gram'][0]:#x} x {src['gram'][1]}"
        elif base == "head":
            enc = f"f16 plain 16 ({src['v1b8'][1]} used)"
            perms = ["none"]
        else:
            enc = f"raw {e['n']}"
            perms = ["none"]
        rows.append((region, source, enc, "; ".join(p for p in perms if p), e["off"], e["n"], note))
    _, scal16, scal_u16 = _sources(k)
    scalars = [f"W s1d0 = f16 table {scal16[0][0]:#x}[0]"]            # build_w's own member names
    if k == 15:
        scalars.append("W dec0_scalars = [min(exp(f16 0x1820[0]), 8), exp(f16 0x1830[0]), exp(f16 0x1840[0])]")
    scalars.append(f"W s1d2 = u16 {scal_u16[0]:#x}, W s1d4 = u16 {scal_u16[1]:#x}")
    if k >= 6:
        scalars.append(f"W s1d6 = u16 {scal_u16[2]:#x}")
    return dict(k=k, name=NAMES_OF[k], L=L, odd=odd, C=C, g=g,
                cd=rr_pack.CDOWN[L] if 1 <= k <= 4 else 0,
                cu=rr_pack.CUP[(k - 6) >> 1] if k >= 6 else 0,
                size=len(blob), meta=meta, rows=rows, blob=blob, scalars=scalars)


NAMES_OF = rr_pack.NAMES


def check_packed(units, packed_dir):
    params = json.loads((Path(packed_dir) / "params.json").read_text())
    for u in units:
        p = params["units"][u["k"]]
        if p["bytes"] != u["size"] or p["meta"] != u["meta"]:
            raise SystemExit(f"unit {u['k']}: {u['size']} bytes / meta differ from {packed_dir}/params.json "
                             f"({p['bytes']} bytes)")
    print(f"  packed check: sizes and 21 item offsets of all {len(units)} units equal {packed_dir}/params.json",
          file=sys.stderr)


def main():
    args = sys.argv[1:]
    md = "--markdown" in args
    packed = args[args.index("--packed") + 1] if "--packed" in args else None
    check_rules()
    dither = bytes(DITHER_BYTES)
    units = [unit_rows(k, dither) for k in range(16)]
    total = sum(u["size"] for u in units)
    print(f"  {len(units)} units, {total} bytes; every blob is a multiple of 16 bytes and its writes tile it",
          file=sys.stderr)
    if packed:
        check_packed(units, packed)
    for u in units:
        head = (f"{u['name']}: L={u['L']} C={u['C']} g={u['g']} odd={u['odd']}"
                + (f" Cdown={u['cd']}" if u["cd"] else "") + (f" Cup={u['cu']}" if u["cu"] else "")
                + f", {u['size']} bytes")
        items = ", ".join(str(m) for m in u["meta"])
        if md:
            print(f"\n### blob{u['k']}.bin = {head}\n")
            print(f"Item offsets (meta[0..20], -1 absent): {items}\n")
            print(f"Not in the blob (params.json): {'; '.join(u['scalars'])}\n")
            print("| write | source (table key x count) | encoding | permutation | offset | bytes | note |\n"
                  "|---|---|---|---|---:|---:|---|")
        else:
            print(f"\nblob{u['k']}.bin = {head}\n  items: {items}\n  scalars: {'; '.join(u['scalars'])}")
        for region, source, enc, perm, off, n, note in u["rows"]:
            if md:
                print(f"| {region} | {source} | {enc} | {perm} | {off} | {n} | {note} |")
            else:
                print(f"  {off:7d} {n:7d}  {enc:24s} {perm:60s} {region:24s} {note}")


if __name__ == "__main__":
    main()
