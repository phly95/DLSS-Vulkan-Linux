#!/usr/bin/env python3
"""tools/sr/sr_probe.py: synthetic-weight oracle probes for the SR stage family.

    build/venv/bin/python -I tools/sr/sr_probe.py isolate OUT_DIR [--preset M] [--stages 0,1,...]
    build/venv/bin/python -I tools/sr/sr_probe.py run OUT_DIR [--frames DIR]

`isolate` copies the packed weights (build/re/packed/<preset>) to OUT_DIR, then zeroes v270, the
MLP blocks and fc2_bias of the selected stage blobs (default: all ten). The mode-0 downsample and
mode-2 upsample projections are kept. NeuralHub's own kernel then writes `input + attention` as
the stage output, so its dump is the oracle's attention sub-block. Score it with the model,
reading inputs from the probe's own dump:

    tools/sr/sr_stage_model.py --launch N --oracle OUT_DIR_dump --weights OUT_DIR --attn-only \\
        OUT_DIR_dump/frame0/<NNN>.<out>.bin

and the MLP sub-block against the real oracle with `--inject-postattn` of that file.

`run` executes NeuralHub's code object through srref (ROCm container) on OUT_DIR. The dump goes to
OUT_DIR_dump/ and the outputs to OUT_DIR_out/. The default frames are the 1-frame directory whose
real-weight dump is byte-identical to the oracle's frame 0 (docs/notes/k_sr_stage.md attempts 5,
R1.1). The probes only ever live under build/ (git-ignored): they hold NVIDIA's weights.
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sr_pack import LAYERS, STAGE_BYTES  # noqa: E402

REPO = Path(__file__).resolve().parents[2]


def regions(C, g, Cn, mode):
    """(start, end) byte ranges of v270, and of the MLP blocks + fc2_bias, in pack_layer's order."""
    cur = (4 * C * Cn + 8 * C) if mode == 2 else 0
    cur += 2 * C + g * (64 * C + 512 + 32 * C)
    v270 = (cur, cur + 2 * C)
    mlp0 = cur + 4 * C                                   # after v270 and gain2
    mlp_end = mlp0 + (C // 8) * (32 * C + 64 + 32 * C) + 2 * C
    return v270, (mlp0, mlp_end)


def isolate(out, preset, stages):
    src = REPO / f"build/re/packed/{preset}"
    if out.exists():
        raise SystemExit(f"{out} exists; pick a new directory")
    shutil.copytree(src, out)
    for i in stages:
        name, _, C, g, Cn, mode = LAYERS[i]
        p = out / f"stage{i}.bin"
        b = bytearray(p.read_bytes())
        if len(b) != STAGE_BYTES[i]:
            raise SystemExit(f"stage{i}.bin: {len(b)} bytes, expected {STAGE_BYTES[i]}")
        (a0, a1), (m0, m1) = regions(C, g, Cn, mode)
        tail = len(b) - m1
        want_tail = (Cn * 4 * C + 2 * Cn) if mode == 0 else 0
        if tail != want_tail:
            raise SystemExit(f"stage{i}: MLP walk ends {tail} bytes before EOF, expected {want_tail}")
        b[a0:a1] = bytes(a1 - a0)
        b[m0:m1] = bytes(m1 - m0)
        p.write_bytes(bytes(b))
        print(f"stage{i} {name}: zeroed v270 {a0}..{a1} and MLP+fc2_bias {m0}..{m1}; kept {tail} tail bytes")


def run(out, frames):
    out = out.resolve()
    rel = out.relative_to(REPO)
    cmd = ["podman", "run", "--rm", "--device", "/dev/kfd", "--device", "/dev/dri",
           "--group-add", "keep-groups", "--security-opt", "label=disable", "-v", f"{REPO}:/w",
           "docker.io/rocm/dev-ubuntu-24.04:7.2.4", "bash", "-c",
           f"cd /w && build/work/srref artifacts/neuralhub-v0.8.0/gfx1201/sr.gfx1201.co {rel} {frames} "
           f"{rel}_out --dump {rel}_dump"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    sys.stdout.write(r.stdout[-2000:])
    sys.stderr.write(r.stderr[-2000:])
    if r.returncode or not (out.parent / f"{out.name}_dump" / "frame0" / "launches.jsonl").exists():
        raise SystemExit(f"srref failed (rc {r.returncode})")
    print(f"dump: {rel}_dump")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["isolate", "run"])
    ap.add_argument("out", type=Path)
    ap.add_argument("--preset", default="M")
    ap.add_argument("--stages", default="0,1,2,3,4,5,6,7,8,9")
    ap.add_argument("--frames", default="build/work/probe/frames1")
    a = ap.parse_args()
    if a.cmd == "isolate":
        isolate(a.out, a.preset, [int(s) for s in a.stages.split(",")])
    else:
        run(a.out, a.frames)


if __name__ == "__main__":
    main()
