#!/usr/bin/env python3
"""Build a synthetic weight directory for the reference NR runner, on a machine that has no model.

The runner reads its weights by name from `--unpacked <root>` (with `NR_WEIGHT_TRACE=1` it names
every one) and tells us what it wants in three ways:

  * "…: 1234 bytes, need 4096"   a size-checked read: write exactly 4096 bytes, except where the
                                 loader is f16 and the printed "need" is a count of elements rather
                                 than bytes (`.skip_weight.bin`); there, write twice that.
  * a segfault                   an unchecked read that then indexes the empty vector: the last path
                                 in the weight trace that is not on disk is the one it died on; write
                                 `--pad` bytes (enough for every record read as a blob).
  * a strict check with no size in the message (`.input_lift.bin` must be 1024 bytes, the post block's
                                 two f16 gains 64 each): the EXACT table below.

Neither the names nor the sizes are NVIDIA's: they are properties of the network's structure, which
is what this recovers. The bytes are a fill pattern, so the pictures are meaningless - the run is a
bring-up test (does the whole graph execute on this driver?) and a frame-time measurement, not a
result. See README.md.

    python3 tools/hwtest/synth_model.py --runner build/hwtest/nr_graph \
        --out build/hwtest/synth-weights
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

NEED = re.compile(r"need (?:\>= )?(\d+)")
READ = re.compile(r"^weight-read (.+)$")

# Reads whose size check prints no size. Keyed by file-name suffix.
EXACT = {
    ".input_lift.bin": 1024,   # pre block's native half lift (nr_graph.cpp: "invalid native input lift size")
    ".skip_gain.bin": 64,      # post block, f16[32]
    ".main_gain.bin": 64,      # post block, f16[32]
}

# Size checks whose printed "need" counts f16 *elements*, not bytes. The loader is
# `load_f16`, which reads two bytes per element, so the file must be twice the number
# printed. (Every other `… need N` in nr_graph.cpp is a byte count.)
ELEM_NEED = (".skip_weight.bin",)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runner", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path, help="the weight directory to create; passed as --unpacked")
    ap.add_argument("--spv-dir", default="build/linux/rdna4/network")
    ap.add_argument("--fill", type=lambda s: int(s, 0), default=0x20, help="byte for every created file")
    ap.add_argument("--pad", type=int, default=4 << 20, help="bytes for a record whose size is not checked")
    ap.add_argument("--max-iters", type=int, default=4000)
    ap.add_argument("--extra", action="append", default=[])
    a = ap.parse_args()

    root = a.out.resolve()
    root.mkdir(parents=True, exist_ok=True)
    fill = bytes([a.fill])
    env = dict(os.environ, NR_WEIGHT_TRACE="1")
    made = 0
    tried: dict[str, int] = {}   # file -> size we last wrote for it (loop breaker)
    for _ in range(a.max_iters):
        cmd = [str(a.runner), "--spv-dir", a.spv_dir, "--unpacked", str(root),
               "--host-boundary", "--no-reuse", *a.extra]
        r = subprocess.run(cmd, capture_output=True, text=True, env=env)
        text = r.stdout + r.stderr
        if r.returncode == 0:
            print(f"--- the graph ran (exit 0) after {made} weight files were created ---")
            print("\n".join(text.splitlines()[-25:]))
            return 0
        # Every traced read, in order, and whether it is on disk yet.
        reads = [Path(m.group(1)) for line in text.splitlines() if (m := READ.match(line.strip()))]
        needs = NEED.findall(text)
        need = int(needs[-1]) if needs else None
        todo = []  # (path, size, growable)
        if need is not None and reads:
            # The size-checked read is the last one, and it is what stopped the run.
            # The printed number is a byte count, except for the f16 skip weights
            # (see ELEM_NEED). These are growable: if the same request comes back after
            # we wrote it, the printed number was not enough - double it and go again.
            size = need * 2 if reads[-1].name.endswith(ELEM_NEED) else need
            todo.append((reads[-1], size, True))
        else:
            # A crash: every traced read that is not on disk is empty, and any of them can be the
            # one that indexed out of bounds (the trace names reads, not the crash).
            for p in reads:
                if not p.is_file():
                    exact = next((n for s, n in EXACT.items() if p.name.endswith(s)), None)
                    todo.append((p, exact or a.pad, False))
            if not todo:
                # Everything is on disk: a strict check with no size in its message (or something
                # else entirely). Resize the reads whose size is known to be fixed.
                for p in reads:
                    exact = next((n for s, n in EXACT.items() if p.name.endswith(s)), None)
                    if exact is not None and p.stat().st_size != exact:
                        todo.append((p, exact, False))
        if not todo:
            # No read is missing and no size was printed. The failing line may name a file
            # directly and without a size ("short QKV record <path>", needs 128 + N*K). Grow
            # the last file named under --out; if none is named, grow the last traced read.
            # The loop breaker doubles it on each round until the check passes.
            cand = None
            for line in reversed(text.splitlines()):
                if line.lstrip().startswith("weight-read"):
                    continue
                tok = next((t for t in line.split() if t.startswith(str(root))), None)
                if tok and Path(tok).is_file():
                    cand = Path(tok)
                    break
            if cand is None and reads:
                cand = reads[-1]
            if cand is not None and cand.is_file():
                todo.append((cand, cand.stat().st_size, True))
        if not todo:
            print(f"--- nothing left to create after {made} files (exit {r.returncode}) ---")
            print("\n".join(text.splitlines()[-25:]))
            return 1
        for target, size, growable in todo:
            rel = target
            if rel.is_absolute() and root in rel.parents:
                rel = rel.relative_to(root)
            dest = root / rel
            if dest.resolve() != root and root not in dest.resolve().parents:
                print(f"refusing to write outside --out: {dest}", file=sys.stderr)
                return 1
            # Loop breaker: if a size-checked read comes back for a file we have already
            # written, the size we used was too small. Double the largest size tried so far
            # and go again, so an element/byte mixup or a check that wants more than it
            # printed cannot spin forever.
            prev = tried.get(str(dest))
            if growable and prev is not None and prev >= size:
                size = prev * 2
                print(f"growing {rel}: {prev} -> {size} B (asked again after writing it)", flush=True)
            tried[str(dest)] = size
            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, "wb") as f:
                f.write(fill * size)
            made += 1
            if made % 20 == 0:
                print(f"{made}: {rel} ({size} B)", flush=True)
    print(f"gave up after {a.max_iters} runs", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
