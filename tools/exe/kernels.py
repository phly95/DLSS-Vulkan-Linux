#!/usr/bin/env python3
"""Per-kernel facts from a HIP code object: launch metadata and an instruction profile.

    kernels.py nr.gfx1201.co [...] > kernels.json

For each kernel: demangled name, kernarg layout (offset/size/kind), LDS and scratch bytes, VGPR/SGPR
counts, wave size, and counts of the instruction classes that decide the Vulkan mapping
(FP8/FP16 WMMA, FP8 conversions, transcendentals, LDS, image/buffer/global memory, atomics).
Disassembly needs llvm-objdump with gfx1201 support (LLVM 18 or newer).
"""
import collections
import json
import os
import re
import struct
import subprocess
import sys

import msgpack

NT_AMDGPU_METADATA = 32
LLVM_OBJDUMP = os.environ.get("LLVM_OBJDUMP", "llvm-objdump")

# Instruction classes that matter for the port. Order matters: first match wins.
CLASSES = [
    ("wmma_fp8", r"^v_wmma_f32_16x16x16_fp8_fp8"),
    ("wmma_fp8_bf8", r"^v_wmma_f32_16x16x16_(fp8_bf8|bf8_fp8|bf8_bf8)"),
    ("wmma_f16", r"^v_wmma_f(16|32)_16x16x16_f16"),
    ("wmma_bf16", r"^v_wmma_(f32|bf16)_16x16x16_bf16"),
    ("wmma_iu", r"^v_wmma_i32"),
    ("swmmac", r"^v_swmmac"),
    ("cvt_to_fp8", r"^v_cvt_(pk_)?(fp8|bf8)_f32|^v_cvt_sr_(fp8|bf8)"),
    ("cvt_from_fp8", r"^v_cvt_(pk_)?f32_(fp8|bf8)"),
    ("dot2_f16", r"^v_dot2"),
    ("exp", r"^v_exp_"),
    ("log", r"^v_log_"),
    ("rcp", r"^v_rcp_"),
    ("rsq", r"^v_rsq_"),
    ("sqrt", r"^v_sqrt_"),
    ("sin_cos", r"^v_(sin|cos)_"),
    ("pk_f16", r"^v_pk_(fma|mul|add|max|min)_f16"),
    ("f16_valu", r"^v_[a-z0-9_]+_f16"),
    ("f32_valu", r"^v_[a-z0-9_]+_f32"),
    ("dpp_perm", r"^(v_permlane|ds_bpermute|ds_permute|ds_swizzle|v_mov_b32_dpp)"),
    ("lds_load", r"^ds_(load|read)"),
    ("lds_store", r"^ds_(store|write)"),
    ("lds_atomic", r"^ds_(add|max|min|cmpst|inc|dec)"),
    ("image_sample", r"^image_sample|^image_gather"),
    ("image_load", r"^image_load"),
    ("image_store", r"^image_store"),
    ("buffer_load", r"^buffer_load"),
    ("buffer_store", r"^buffer_store"),
    ("global_load", r"^global_load"),
    ("global_store", r"^global_store"),
    ("global_atomic", r"^(global|buffer|flat)_atomic"),
    ("scalar_load", r"^s_(load|buffer_load)"),
    ("barrier", r"^s_barrier"),
]
CLASS_RE = [(n, re.compile(p)) for n, p in CLASSES]


def elf_notes(data):
    shoff, = struct.unpack_from("<Q", data, 0x28)
    shentsize, shnum = struct.unpack_from("<HH", data, 0x3A)
    for i in range(shnum):
        sh = shoff + i * shentsize
        stype, = struct.unpack_from("<I", data, sh + 4)
        if stype != 7:  # SHT_NOTE
            continue
        off, size = struct.unpack_from("<QQ", data, sh + 0x18)
        p, end = off, off + size
        while p < end:
            namesz, descsz, ntype = struct.unpack_from("<III", data, p)
            p += 12
            name = data[p:p + namesz]
            p += (namesz + 3) & ~3
            desc = data[p:p + descsz]
            p += (descsz + 3) & ~3
            yield name, ntype, desc


def metadata(data):
    for name, ntype, desc in elf_notes(data):
        if name.startswith(b"AMDGPU") and ntype == NT_AMDGPU_METADATA:
            return msgpack.unpackb(desc, raw=False)
    raise ValueError("no AMDGPU metadata note")


def demangle(names):
    out = subprocess.run(["c++filt"], input="\n".join(names), capture_output=True, text=True,
                         check=True).stdout.splitlines()
    return dict(zip(names, out))


def profile(co_path):
    """Instruction class counts per kernel symbol from the disassembly."""
    dis = subprocess.run([LLVM_OBJDUMP, "-d", "--mcpu=gfx1201", "--no-show-raw-insn", co_path],
                         capture_output=True, text=True, check=True).stdout
    counts = collections.defaultdict(collections.Counter)
    kreads = collections.defaultdict(set)
    load_re = re.compile(r"s_load_b(32|64|96|128|256|512)\s+s\[?[0-9:]+\]?,\s+s\[0:1\],\s+(0x[0-9a-f]+)")
    cur = None
    for line in dis.splitlines():
        m = re.match(r"^[0-9a-f]+ <(.+)>:$", line)
        if m:
            cur = m.group(1)
            continue
        if cur is None:
            continue
        s = line.strip()
        if not s or s.startswith(";") or s.endswith(":"):
            continue
        op = s.split()[0]
        counts[cur]["insts"] += 1
        m = load_re.match(s)
        if m:
            # Kernel arguments arrive through s[0:1]; a load from it names the bytes the kernel uses.
            kreads[cur].add((int(m.group(2), 16), int(m.group(1)) // 8))
        for n, r in CLASS_RE:
            if r.match(op):
                counts[cur][n] += 1
                break
    return counts, kreads


def main():
    result = {}
    for path in sys.argv[1:]:
        data = open(path, "rb").read()
        md = metadata(data)
        prof, kreads = profile(path)
        kernels = md["amdhsa.kernels"]
        dm = demangle([k[".name"] for k in kernels])
        out = []
        for k in kernels:
            args = [{"offset": a[".offset"], "size": a[".size"], "kind": a[".value_kind"],
                     **({"name": a[".name"]} if ".name" in a else {}),
                     **({"type": a[".type_name"]} if ".type_name" in a else {}),
                     **({"address_space": a[".address_space"]} if ".address_space" in a else {})}
                    for a in k.get(".args", [])]
            out.append({
                "symbol": k[".name"],
                "name": dm[k[".name"]],
                "kernarg_bytes": k[".kernarg_segment_size"],
                "args": args,
                "lds_bytes": k[".group_segment_fixed_size"],
                "scratch_bytes": k[".private_segment_fixed_size"],
                "vgprs": k[".vgpr_count"],
                "sgprs": k[".sgpr_count"],
                "wave": k[".wavefront_size"],
                "max_flat_wg": k[".max_flat_workgroup_size"],
                "reqd_wg": k.get(".reqd_workgroup_size"),
                "uses_dynamic_stack": k.get(".uses_dynamic_stack", False),
                "profile": dict(prof.get(k[".name"], {})),
                "kernarg_reads": sorted(kreads.get(k[".name"], ())),
            })
        out.sort(key=lambda k: k["name"])
        result[path.split("/")[-1]] = {"target": md.get("amdhsa.target"), "kernels": out}
    json.dump(result, sys.stdout, indent=1)


if __name__ == "__main__":
    main()
