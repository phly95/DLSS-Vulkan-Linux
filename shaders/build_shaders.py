#!/usr/bin/env python3
"""Build Vulkan compute shaders using pinned glslang for RDNA4."""
import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GLSLANG = ROOT / "toolchain/glslang/bin/glslang"


def compile_shader(src: Path, out: Path, defines=None):
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(GLSLANG),
        "-V",
        "--target-env", "vulkan1.3",
        "-I" + str(ROOT / "shaders/include"),
        "-I" + str(src.parent),
    ]
    if defines:
        for d in defines:
            cmd.append("-D" + d)
    cmd.extend([str(src), "-o", str(out)])
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"Compilation failed for {src}:", file=sys.stderr)
        print(res.stdout, file=sys.stderr)
        print(res.stderr, file=sys.stderr)
        sys.exit(1)
    print(f"  Compiled: {src.name} -> {out.name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "build/vk/spv")
    args = ap.parse_args()

    shaders_dir = ROOT / "shaders"
    for comp in shaders_dir.glob("**/*.comp"):
        rel = comp.relative_to(shaders_dir)
        spv_name = comp.stem + ".spv"
        out_path = args.out / rel.parent / spv_name
        compile_shader(comp, out_path)


if __name__ == "__main__":
    main()
