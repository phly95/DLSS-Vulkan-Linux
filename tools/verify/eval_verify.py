#!/usr/bin/env python3
"""tools/verify/eval_verify.py: the referee's numeric checks (V4, V5, V7, V8).

    check-cell PORT REF [--gt GT] [--name N]      V4: per frame PSNR(port, ref) >= 50 dB, mean >= 55 dB,
                                                  no non-finite value; with --gt, every frame's
                                                  |PSNR(port, gt) - PSNR(ref, gt)| <= 0.1 dB
    check-degraded BAD REF [--name N]             V5: the degraded run BAD must FAIL check-cell against REF
    check-timing PORT_LOG REF_LOG --frames N      V8: median GPU frame time (frames 1..N-1) of the port
                                                  <= 3x the oracle's, from the "GPU_FRAME_MS i ms" lines
    check-radv PORT_LOG --device NAME             V7: the port ran on NAME through RADV and every pipeline
                                                  it created has 0 spilled SGPRs / VGPRs and 0 scratch,
                                                  from Mesa's own RADV_DEBUG=shaderstats blocks
    --self-test                                   checks these checks on synthetic data

PSNR is on x/(1+x) tone-mapped RGB with peak 1:
    tone_map(x) = max(x, 0) / (1 + max(x, 0)); psnr = -10 log10(mean((tm(a) - tm(b))^2))
Frames are out_<i>.f32 (RGBA32F); sizes come from REF's frame.json written by run_all.sh.
"""
import argparse
import json
import math
import os
import re
import statistics
import sys
import tempfile

import numpy as np

V4_MIN_DB, V4_MEAN_DB, GT_DELTA_DB, V8_RATIO = 50.0, 55.0, 0.1, 3.0


def tone_map(rgb):
    x = np.maximum(rgb, 0.0)
    return x / (1.0 + x)


def psnr(a, b):
    """(psnr, non_finite)"""
    a, b = a[..., :3].astype(np.float64), b[..., :3].astype(np.float64)
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        return float("-inf"), True
    mse = float(np.mean((tone_map(a) - tone_map(b)) ** 2))
    if mse <= 1e-12:
        return 120.0, False
    return -10.0 * math.log10(mse), False


def frame_info(d):
    fj = os.path.join(d, "frame.json")
    if not os.path.exists(fj):
        raise SystemExit(f"FAIL: {fj} missing")
    data = json.load(open(fj))
    return data["output"][0], data["output"][1], data["frames"]


def load(path, w, h):
    if not os.path.exists(path):
        return None
    raw = np.fromfile(path, dtype=np.float32)
    if raw.size != w * h * 4:
        raise SystemExit(f"FAIL: {path} holds {raw.size} floats, expected {w}x{h}x4")
    return raw.reshape(h, w, 4)


def cell(port, ref, gt, name, quiet=False):
    """0 = passes V4"""
    say = (lambda *a: None) if quiet else print
    ow, oh, n = frame_info(ref)
    say(f"--- V4 {name or port}: {n} frames, {ow}x{oh} ---")
    vals = []
    for i in range(n):
        p = load(os.path.join(port, f"out_{i}.f32"), ow, oh)
        r = load(os.path.join(ref, f"out_{i}.f32"), ow, oh)
        if p is None or r is None:
            say(f"frame {i:2d}: MISSING (port {p is not None}, ref {r is not None}) FAIL")
            return 1
        v, nonfinite = psnr(p, r)
        if nonfinite:
            say(f"frame {i:2d}: non-finite values FAIL")
            return 1
        line = f"frame {i:2d}: {v:7.2f} dB"
        if gt:
            g = load(os.path.join(gt, f"gt_{i}.f32"), ow, oh)
            if g is None:
                say(line + f" | ground truth {gt}/gt_{i}.f32 MISSING FAIL")
                return 1
            pg, _ = psnr(p, g)
            rg, _ = psnr(r, g)
            line += f" | vs GT port {pg:.3f} ref {rg:.3f} (|diff| {abs(pg - rg):.3f} dB)"
            if not abs(pg - rg) <= GT_DELTA_DB:
                say(line + f" > {GT_DELTA_DB} dB FAIL")
                return 1
        say(line + (" FAIL" if v < V4_MIN_DB else ""))
        vals.append(v)
    mn, mean = min(vals), sum(vals) / len(vals)
    ok = mn >= V4_MIN_DB and mean >= V4_MEAN_DB
    say(f"min {mn:.2f} dB (>= {V4_MIN_DB}), mean {mean:.2f} dB (>= {V4_MEAN_DB}): {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def degraded(bad, ref, name):
    print(f"--- V5 {name}: the degraded run must fail V4 against the oracle ---")
    r = cell(bad, ref, None, name, quiet=False)
    if r == 0:
        print(f"V5 {name}: FAIL (the degraded output passes V4, so V4 cannot tell it from the oracle)")
        return 1
    print(f"V5 {name}: PASS (V4 rejects it)")
    return 0


def frame_times(log):
    t = {}
    for line in open(log, errors="replace"):
        m = re.match(r"\s*GPU_FRAME_MS\s+(\d+)\s+([0-9.eE+-]+)\s*$", line)
        if m:
            t[int(m.group(1))] = float(m.group(2))
    return t


def timing(port_log, ref_log, n, name):
    tp, tr = frame_times(port_log), frame_times(ref_log)
    for lbl, t in (("port", tp), ("oracle", tr)):
        if sorted(t) != list(range(n)):
            print(f"V8 {name}: FAIL ({lbl} log has GPU_FRAME_MS for frames {sorted(t)}, expected 0..{n - 1})")
            return 1
        if any(not (v > 0 and math.isfinite(v)) for v in t.values()):
            print(f"V8 {name}: FAIL ({lbl} reports a non-positive frame time)")
            return 1
    mp = statistics.median(tp[i] for i in range(1, n))
    mr = statistics.median(tr[i] for i in range(1, n))
    ok = mp <= V8_RATIO * mr
    print(f"V8 {name}: median GPU frame (frames 1..{n - 1}) port {mp:.3f} ms, oracle {mr:.3f} ms, "
          f"ratio {mp / mr:.2f} (<= {V8_RATIO}): {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


STAT_KEYS = ("SGPRs", "VGPRs", "Spilled SGPRs", "Spilled VGPRs", "Scratch size", "Code size", "Subgroups per SIMD")


def shader_stats(text):
    """Mesa's RADV_DEBUG=shaderstats blocks: '*** SHADER STATS ***' ... '********************'."""
    blocks, cur = [], None
    for line in text.splitlines():
        s = line.strip()
        if s == "*** SHADER STATS ***":
            cur = {}
        elif cur is not None and s.startswith("****"):
            blocks.append(cur)
            cur = None
        elif cur is not None:
            m = re.match(r"([A-Za-z][A-Za-z -]*):\s*(-?\d+)$", s)
            if m:
                cur[m.group(1)] = int(m.group(2))
    return blocks


def radv(log, device):
    text = open(log, errors="replace").read()
    bad = []
    if re.search(r"llvmpipe|lavapipe|softpipe|drm.shim|noop_drm", text, re.I):
        bad.append("the log names a software or shimmed driver")
    devs = re.findall(r"^VK_DEVICE: (.*)$", text, re.M)
    if not devs or any(d.strip() != device for d in devs):
        bad.append(f"VK_DEVICE lines {devs!r}, expected exactly {device!r}")
    m = re.findall(r"^PIPELINES_CREATED: (\d+)$", text, re.M)
    created = int(m[-1]) if m else -1
    blocks = shader_stats(text)
    print(f"V7: {len(blocks)} shaderstats blocks, the port reports {created} pipelines created")
    print(f"  {'#':>3} " + " ".join(f"{k:>18}" for k in STAT_KEYS))
    for i, b in enumerate(blocks):
        print(f"  {i:3d} " + " ".join(f"{b.get(k, -1):18d}" for k in STAT_KEYS))
        for k in ("Spilled SGPRs", "Spilled VGPRs", "Scratch size"):
            if k not in b:
                bad.append(f"block {i} has no '{k}'")
            elif b[k] != 0:
                bad.append(f"block {i}: {k} = {b[k]}")
    if created <= 0:
        bad.append("no PIPELINES_CREATED line (or 0)")
    elif len(blocks) < created:
        bad.append(f"{len(blocks)} shaderstats blocks for {created} pipelines (shader cache not disabled?)")
    for b in bad:
        print("  V7 FAIL:", b)
    print("V7:", "PASS" if not bad else "FAIL")
    return 0 if not bad else 1


def self_test():
    rng = np.random.default_rng(3)
    ok = True
    with tempfile.TemporaryDirectory() as t:
        def seq(d, imgs):
            os.makedirs(d, exist_ok=True)
            json.dump(dict(render=[4, 4], output=[8, 6], frames=len(imgs)), open(os.path.join(d, "frame.json"), "w"))
            for i, im in enumerate(imgs):
                im.astype(np.float32).tofile(os.path.join(d, f"out_{i}.f32"))
        base = [rng.uniform(0, 4, (6, 8, 4)) for _ in range(3)]
        seq(t + "/ref", base)
        seq(t + "/same", base)
        seq(t + "/close", [b * (1 + 1e-4) for b in base])
        seq(t + "/bad", [b * 1.3 for b in base])
        seq(t + "/nan", [np.where(rng.uniform(size=b.shape) < 0.01, np.nan, b) for b in base])
        os.makedirs(t + "/gt")
        for i, b in enumerate(base):
            (b * 1.05).astype(np.float32).tofile(t + f"/gt/gt_{i}.f32")
        cases = [("identical passes", cell(t + "/same", t + "/ref", t + "/gt", "", True) == 0),
                 ("1e-4 relative passes", cell(t + "/close", t + "/ref", None, "", True) == 0),
                 ("30% off fails", cell(t + "/bad", t + "/ref", None, "", True) == 1),
                 ("NaN fails", cell(t + "/nan", t + "/ref", None, "", True) == 1),
                 ("missing GT fails", cell(t + "/same", t + "/ref", t + "/nogt", "", True) == 1)]
        os.remove(t + "/gt/gt_2.f32")
        cases.append(("one GT frame missing fails", cell(t + "/same", t + "/ref", t + "/gt", "", True) == 1))
        open(t + "/p.log", "w").write("".join(f"GPU_FRAME_MS {i} {2.0 + i * 0.01}\n" for i in range(3)))
        open(t + "/r.log", "w").write("".join(f"GPU_FRAME_MS {i} 1.0\n" for i in range(3)))
        open(t + "/slow.log", "w").write("".join(f"GPU_FRAME_MS {i} 3.5\n" for i in range(3)))
        open(t + "/short.log", "w").write("GPU_FRAME_MS 0 1.0\n")
        cases += [("2x passes V8", timing(t + "/p.log", t + "/r.log", 3, "t") == 0),
                  ("3.5x fails V8", timing(t + "/slow.log", t + "/r.log", 3, "t") == 1),
                  ("missing frames fail V8", timing(t + "/short.log", t + "/r.log", 3, "t") == 1)]
        blk = lambda sp, sc: ("*** SHADER STATS ***\nSGPRs: 108\nVGPRs: 24\nSpilled SGPRs: %d\nSpilled VGPRs: 0\n"
                              "Code size: 68\nScratch size: %d\nSubgroups per SIMD: 16\n********************\n" % (sp, sc))
        dev = "AMD Radeon RX 9070 XT (RADV GFX1201)"
        open(t + "/ok.log", "w").write(f"VK_DEVICE: {dev}\nPIPELINES_CREATED: 2\n" + blk(0, 0) + blk(0, 0))
        open(t + "/spill.log", "w").write(f"VK_DEVICE: {dev}\nPIPELINES_CREATED: 2\n" + blk(0, 0) + blk(3, 0))
        open(t + "/scr.log", "w").write(f"VK_DEVICE: {dev}\nPIPELINES_CREATED: 1\n" + blk(0, 256))
        open(t + "/few.log", "w").write(f"VK_DEVICE: {dev}\nPIPELINES_CREATED: 3\n" + blk(0, 0))
        open(t + "/lvp.log", "w").write("VK_DEVICE: llvmpipe (LLVM 18.1.3, 256 bits)\nPIPELINES_CREATED: 1\n" + blk(0, 0))
        open(t + "/shim.log", "w").write("VK_DEVICE: AMD Radeon Graphics (RADV GFX1201)\nPIPELINES_CREATED: 1\n" + blk(0, 0))
        cases += [("clean stats pass V7", radv(t + "/ok.log", dev) == 0),
                  ("a spill fails V7", radv(t + "/spill.log", dev) == 1),
                  ("scratch fails V7", radv(t + "/scr.log", dev) == 1),
                  ("fewer blocks than pipelines fail V7", radv(t + "/few.log", dev) == 1),
                  ("lavapipe fails V7", radv(t + "/lvp.log", dev) == 1),
                  ("another device name fails V7", radv(t + "/shim.log", dev) == 1)]
    for n, r in cases:
        print(f"  {n:40s} {'ok' if r else 'WRONG'}")
        ok &= r
    print("EVAL_VERIFY SELF-TEST:", "PASS" if ok else "FAIL")
    return ok


def main():
    if sys.argv[1:] == ["--self-test"]:
        sys.exit(0 if self_test() else 1)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check-cell"); c.add_argument("port"); c.add_argument("ref")
    c.add_argument("--gt"); c.add_argument("--name", default="")
    c = sub.add_parser("check-degraded"); c.add_argument("bad"); c.add_argument("ref"); c.add_argument("--name", default="")
    c = sub.add_parser("check-timing"); c.add_argument("port_log"); c.add_argument("ref_log")
    c.add_argument("--frames", type=int, required=True); c.add_argument("--name", default="")
    c = sub.add_parser("check-radv"); c.add_argument("log"); c.add_argument("--device", required=True)
    a = ap.parse_args()
    if a.cmd == "check-cell":
        sys.exit(cell(a.port, a.ref, a.gt, a.name))
    if a.cmd == "check-degraded":
        sys.exit(degraded(a.bad, a.ref, a.name))
    if a.cmd == "check-timing":
        sys.exit(timing(a.port_log, a.ref_log, a.frames, a.name))
    sys.exit(radv(a.log, a.device))


if __name__ == "__main__":
    main()
