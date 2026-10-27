#!/usr/bin/env python3
import json
import os
import struct
import sys
import numpy as np

def make_test(name, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    w, h = 32, 16
    n_pix = w * h

    if name in ("sr_gather", "sr_gather_mv"):
        col = np.linspace(0.1, 1.0, n_pix * 4, dtype=np.float16).tobytes()
        mv  = np.linspace(0.01, 0.5, n_pix * 2, dtype=np.float16).tobytes()
        dep = np.linspace(0.1, 100.0, n_pix, dtype=np.float32).tobytes()
        open(os.path.join(out_dir, "in_col.bin"), "wb").write(col)
        open(os.path.join(out_dir, "in_mv.bin"), "wb").write(mv)
        open(os.path.join(out_dir, "in_dep.bin"), "wb").write(dep)

        if name == "sr_gather":
            push = bytearray(88)
            struct.pack_into("<3I3i", push, 24, w * 8, w * 4, w * 4, 0, 0, 0)
            struct.pack_into("<4i", push, 72, w, h, 0, 0)
            cfg = {
                "groups": [(w + 15) // 16, (h + 15) // 16, 1],
                "push": push.hex(),
                "bindings": [
                    {"binding": 0, "file": "in_col.bin"},
                    {"binding": 1, "file": "in_mv.bin"},
                    {"binding": 2, "file": "in_dep.bin"},
                    {"binding": 3, "bytes": n_pix * 8, "output": "g_col", "type": "f16"},
                    {"binding": 4, "bytes": n_pix * 4, "output": "g_mv", "type": "f16"},
                    {"binding": 5, "bytes": n_pix * 4, "output": "g_dep", "type": "f32"}
                ]
            }
        else:
            push = bytearray(32)
            struct.pack_into("<2i", push, 8, w * 4, 0)
            struct.pack_into("<2i", push, 24, w, h)
            cfg = {
                "groups": [(w + 15) // 16, (h + 15) // 16, 1],
                "push": push.hex(),
                "bindings": [
                    {"binding": 0, "file": "in_mv.bin"},
                    {"binding": 1, "bytes": n_pix * 4, "output": "g_mvhi", "type": "f16"}
                ]
            }
    elif name == "sr_exposure":
        expo = np.array([1.0, 1.0], dtype=np.float16).tobytes()
        open(os.path.join(out_dir, "expo.bin"), "wb").write(expo)
        push = bytearray(32)
        struct.pack_into("<Q2fi", push, 0, 0, 1.5, 1.2, 0)
        cfg = {
            "groups": [1, 1, 1],
            "push": push.hex(),
            "bindings": [
                {"binding": 0, "file": "expo.bin", "bytes": 4, "output": "expo", "type": "f16"}
            ]
        }
    elif name == "sr_ae_luma":
        col = np.linspace(0.1, 1.0, n_pix * 4, dtype=np.float16).tobytes()
        open(os.path.join(out_dir, "in_col.bin"), "wb").write(col)
        push = bytearray(48)
        struct.pack_into("<3i5f", push, 8, w, w, h, float(w - 1), float(h - 1), float(w - 1), float(h - 1), 1.0 / float(w * h))
        cfg = {
            "groups": [(w + 7) // 8, (h + 7) // 8, 1],
            "push": push.hex(),
            "bindings": [
                {"binding": 0, "file": "in_col.bin"},
                {"binding": 1, "bytes": n_pix * 4, "output": "ae_luma", "type": "f32"}
            ]
        }
    elif name == "sr_ae_reduce":
        data = np.linspace(0.01, 10.0, 1024, dtype=np.float32).tobytes()
        open(os.path.join(out_dir, "in_data.bin"), "wb").write(data)
        push = bytearray(32)
        struct.pack_into("<ii", push, 8, 1024, 0)
        cfg = {
            "groups": [2, 1, 1],
            "push": push.hex(),
            "bindings": [
                {"binding": 0, "file": "in_data.bin"},
                {"binding": 1, "bytes": 8, "output": "out_data", "type": "f32"}
            ]
        }
    elif name == "sr_ae_copy":
        res = np.array([0.18], dtype=np.float32).tobytes()
        expo = np.array([1.0, 1.0], dtype=np.float16).tobytes()
        open(os.path.join(out_dir, "result.bin"), "wb").write(res)
        open(os.path.join(out_dir, "expo.bin"), "wb").write(expo)
        push = bytearray(24)
        struct.pack_into("<i", push, 16, 0)
        cfg = {
            "groups": [1, 1, 1],
            "push": push.hex(),
            "bindings": [
                {"binding": 0, "file": "result.bin"},
                {"binding": 1, "file": "expo.bin", "bytes": 4, "output": "expo", "type": "f16"}
            ]
        }
    elif name == "sr_unorm8_table":
        cfg = {
            "groups": [1, 1, 1],
            "push": "",
            "bindings": [
                {"binding": 0, "bytes": 512, "output": "table", "type": "f16"}
            ]
        }
    elif name == "sr_downsample":
        n_full = 48 * 48
        full_data = np.full(n_full, 0x05c2784f, dtype=np.uint32).tobytes()
        open(os.path.join(out_dir, "full15.bin"), "wb").write(full_data)
        col = np.linspace(0.1, 1.0, 32 * 32 * 4, dtype=np.float16).tobytes()
        open(os.path.join(out_dir, "g_col.bin"), "wb").write(col)
        expo = np.array([1.0, 1.0], dtype=np.float16).tobytes()
        open(os.path.join(out_dir, "expo.bin"), "wb").write(expo)
        push = bytearray(88)
        struct.pack_into("<2i", push, 24, 0, 0)
        struct.pack_into("<6i", push, 40, 32, 32, 48, 48, 32, 32)
        struct.pack_into("<4f", push, 64, 1.0, 1.0, 0.0, 0.0)
        struct.pack_into("<2I", push, 80, 0x40549a78, 1)
        cfg = {
            "groups": [2, 2, 1],
            "push": push.hex(),
            "bindings": [
                {"binding": 0, "file": "full15.bin"},
                {"binding": 1, "file": "g_col.bin"},
                {"binding": 2, "file": "expo.bin"},
                {"binding": 3, "bytes": 32 * 32 * 16, "output": "out", "type": "f32"}
            ]
        }
    elif name == "sr_enc0":
        w0 = ((w + 7) // 8) * 8
        h0 = ((h + 7) // 8) * 8
        b_col = np.linspace(0.1, 1.0, n_pix * 4, dtype=np.float16).tobytes()
        b_dep = np.linspace(0.1, 100.0, n_pix, dtype=np.float32).tobytes()
        b_mv = np.linspace(0.01, 0.5, n_pix * 2, dtype=np.float16).tobytes()
        open(os.path.join(out_dir, "g_col.bin"), "wb").write(b_col)
        open(os.path.join(out_dir, "g_dep.bin"), "wb").write(b_dep)
        open(os.path.join(out_dir, "g_mv.bin"), "wb").write(b_mv)
        open(os.path.join(out_dir, "prev_mv.bin"), "wb").write(b_mv)
        open(os.path.join(out_dir, "dhist.bin"), "wb").write(np.zeros(n_pix, dtype=np.uint32).tobytes())
        open(os.path.join(out_dir, "hidden.bin"), "wb").write(b_mv)
        open(os.path.join(out_dir, "wbuf.bin"), "wb").write(np.zeros(4096, dtype=np.uint8).tobytes())
        push = bytearray(148)
        struct.pack_into("<8i", push, 88, w, h, w0, h0, w, h, w, h)
        struct.pack_into("<i2f", push, 136, 0, 0.0, 0.0)
        cfg = {
            "groups": [w0 // 8, h0 // 8, 1],
            "push": push.hex(),
            "bindings": [
                {"binding": 0, "file": "g_col.bin"},
                {"binding": 1, "file": "g_dep.bin"},
                {"binding": 2, "file": "g_mv.bin"},
                {"binding": 3, "file": "prev_mv.bin"},
                {"binding": 4, "file": "dhist.bin"},
                {"binding": 5, "file": "hidden.bin"},
                {"binding": 6, "bytes": n_pix * 4, "output": "prev_mv_out", "type": "f16"},
                {"binding": 7, "bytes": n_pix * 4, "output": "mv_ero", "type": "f16"},
                {"binding": 8, "bytes": n_pix * 4, "output": "mv_nondil", "type": "f16"},
                {"binding": 9, "bytes": (w0 // 2) * (h0 // 2) * 64, "output": "t0", "type": "u8"},
                {"binding": 10, "file": "wbuf.bin"}
            ]
        }
    elif name == "sr_dec0":
        # k_sr_dec0 is a 2-D tile kernel: grid (W0/8, H0/8), one 8x8 render tile per workgroup,
        # 32 threads x 2 pixels.  W,H are the render size, W0,H0 the padded one (as the host's
        # (W+31)&~31), and W15,H15 the level-15 grid the motion vectors live in.  t8 (binding 0)
        # is [2][H0/2][W0/2][32] fp8, the weight blob (binding 7) is the 128x64 fp8 matrix at
        # sr_pack.frag_offsets(128,64) positions 0..8191 followed by 128 f16 biases at 8192.
        w0 = (w + 31) & ~31
        h0 = (h + 31) & ~31
        w15, h15 = (w * 3 + 1) >> 1, (h * 3 + 1) >> 1
        wh, hh = w0 // 2, h0 // 2
        idx = lambda n: np.arange(n, dtype=np.uint64)
        mix = lambda n: ((idx(n) * 0x9E3779B97F4A7C15) >> 32).astype(np.uint64)
        # t8: full-range fp8 bytes, so every head's input varies per pixel (a constant input
        # makes every output constant, which O7 rejects).
        open(os.path.join(out_dir, "in_t8.bin"), "wb").write((mix(2 * hh * wh * 32) & 0xFF).astype(np.uint8).tobytes())
        # prev_mvec: f16vec2 per render pixel, spread wide enough that the reprojected position
        # leaves [0,W15]x[0,H15] for part of the frame (exercising the warp-disabled path too).
        mvs = ((mix(n_pix * 2) & 0xFF).astype(np.float64) / 32.0 - 4.0).astype(np.float16)
        open(os.path.join(out_dir, "prev_mv.bin"), "wb").write(mvs.tobytes())
        open(os.path.join(out_dir, "hidden.bin"), "wb").write((mix(n_pix * 4) & 0xFF).astype(np.uint8).tobytes())
        # dec0_w: weights 0..8191 in the moderate fp8 range 0x50..0x6f (|v| 0.031..0.234, so the
        # 64-term dot lands near 0 and every head stays in its varying regime), 128 f16 biases.
        wts = (0x50 + (mix(8192) & 0x1F)).astype(np.uint8).tobytes()
        bias = np.linspace(-1.0, 1.0, 128, dtype=np.float16).tobytes()
        open(os.path.join(out_dir, "wbuf.bin"), "wb").write(wts + bias)
        push = bytearray(160)
        struct.pack_into("<8i", push, 64, w, h, w, h, w0, h0, w15, h15)
        struct.pack_into("<f", push, 96, 2.0)                    # dec0_s0
        struct.pack_into("<f", push, 100, 0.75)                  # dec0_f0
        struct.pack_into("<4f", push, 104, 0.5, 0.25, 0.375, 0.125)   # F20x,F20y,F28x,F28y
        struct.pack_into("<2f", push, 120, w15 / w, h15 / h)     # r15w, r15h
        struct.pack_into("<2f", push, 128, 1.0 / w, 1.0 / h)     # 1/W, 1/H
        struct.pack_into("<I", push, 136, 0)                     # +0x88 warp enabled
        struct.pack_into("<2f", push, 140, 1.0 / w15, 1.0 / h15)  # 1/W15, 1/H15
        struct.pack_into("<2f", push, 148, 0.375 / w, 0.125 / h)  # F28x/W, F28y/H
        cfg = {
            "groups": [w0 // 8, h0 // 8, 1],
            "push": push.hex(),
            "bindings": [
                {"binding": 0, "file": "in_t8.bin"},
                {"binding": 1, "file": "prev_mv.bin"},
                {"binding": 2, "file": "hidden.bin"},
                {"binding": 3, "bytes": n_pix * 4, "output": "gates", "type": "u8"},
                {"binding": 4, "bytes": n_pix * 4, "output": "factors", "type": "u8"},
                {"binding": 5, "bytes": n_pix * 8, "output": "aniso", "type": "f16"},
                {"binding": 6, "bytes": n_pix * 4, "output": "hidden_out", "type": "u8"},
                {"binding": 7, "file": "wbuf.bin"}
            ]
        }
    elif name == "sr_post":
        col = np.linspace(0.1, 1.0, n_pix * 4, dtype=np.float16).tobytes()
        mv = np.linspace(0.01, 0.5, n_pix * 2, dtype=np.float16).tobytes()
        open(os.path.join(out_dir, "g_col.bin"), "wb").write(col)
        open(os.path.join(out_dir, "expo.bin"), "wb").write(np.array([1.0, 1.0], dtype=np.float16).tobytes())
        open(os.path.join(out_dir, "gates.bin"), "wb").write(mv)
        open(os.path.join(out_dir, "factors.bin"), "wb").write(mv)
        open(os.path.join(out_dir, "aniso.bin"), "wb").write(col)
        open(os.path.join(out_dir, "ihist.bin"), "wb").write(col)
        open(os.path.join(out_dir, "prev_mv.bin"), "wb").write(mv)
        open(os.path.join(out_dir, "mv_ero.bin"), "wb").write(mv)
        open(os.path.join(out_dir, "mv_nondil.bin"), "wb").write(mv)
        push = bytearray(220)
        struct.pack_into("<I", push, 16, 0)
        struct.pack_into("<2i", push, 136, w, h)
        struct.pack_into("<2i", push, 152, w, h)
        struct.pack_into("<2i", push, 160, w, h)
        struct.pack_into("<f2I", push, 168, 1.0, 1, 0)
        cfg = {
            "groups": [(w + 15) // 16, (h + 15) // 16, 1],
            "push": push.hex(),
            "bindings": [
                {"binding": 0, "file": "g_col.bin"},
                {"binding": 1, "file": "expo.bin"},
                {"binding": 2, "file": "gates.bin"},
                {"binding": 3, "file": "factors.bin"},
                {"binding": 4, "file": "aniso.bin"},
                {"binding": 5, "file": "ihist.bin"},
                {"binding": 6, "file": "prev_mv.bin"},
                {"binding": 7, "file": "mv_ero.bin"},
                {"binding": 8, "file": "mv_nondil.bin"},
                {"binding": 9, "bytes": n_pix * 8, "output": "out_ihist", "type": "f16"},
                {"binding": 10, "bytes": n_pix * 4, "output": "full15", "type": "u32"},
                {"binding": 11, "bytes": n_pix * 4, "output": "dhist", "type": "u32"}
            ]
        }
    elif name == "sr_tube_split":
        in_act = np.linspace(1, 250, 4096, dtype=np.uint8).tobytes()
        open(os.path.join(out_dir, "in_act.bin"), "wb").write(in_act)
        open(os.path.join(out_dir, "wbuf.bin"), "wb").write(np.zeros(4096, dtype=np.uint8).tobytes())
        push = bytearray(56)
        struct.pack_into("<4i", push, 40, 16, 16, 0, 0)
        cfg = {
            "groups": [64, 1, 1],
            "push": push.hex(),
            "bindings": [
                {"binding": 0, "file": "wbuf.bin"},
                {"binding": 1, "file": "in_act.bin"},
                {"binding": 2, "bytes": 4096, "output": "out_act", "type": "u8"}
            ]
        }
    elif name == "sr_stage":
        # Mode 0 (wbuf_lo = 0, st0): C = 64, 2 heads, 8 MLP blocks, 2x2 patch merge, on a 16x16
        # level. The stage reads five buffers: the f16 vector blob (0), the token bytes (1), the
        # mode-2 skip tensor (4; never read in mode 0, but statically used, so the launcher must
        # bind it) and the row-major matrix copy (5). Sizes cover every access the shader makes
        # here: the f16 blob up to ~72 KB, the matrix copy up to ~68 K elements (~135 KB when the
        # operand-probe build reads the same buffer as halfs).
        open(os.path.join(out_dir, "in_act.bin"), "wb").write(
            np.tile(np.linspace(1, 250, 4096, dtype=np.uint8), 4).tobytes())   # [C/32][16][16][32]
        # Non-zero synthetic data so no code path collapses to a constant output: the f16 vector
        # blob is halfs in [0.5, 2], the matrix copy halfs in [0.05, 0.5] - finite both as halfs
        # and as the (low, high) byte pairs the fp8 build reads.
        open(os.path.join(out_dir, "wbuf.bin"), "wb").write(
            np.linspace(0.5, 2.0, 65536, dtype=np.float16).tobytes())          # 128 KiB
        open(os.path.join(out_dir, "wrep.bin"), "wb").write(
            np.tile(np.linspace(0.05, 0.5, 16384, dtype=np.float16).tobytes(), 8))  # 256 KiB
        push = bytearray(56)
        struct.pack_into("<4i", push, 40, 16, 16, 0, 0)
        cfg = {
            "groups": [64, 1, 1],
            "push": push.hex(),
            "bindings": [
                {"binding": 0, "file": "wbuf.bin"},
                {"binding": 1, "file": "in_act.bin"},
                {"binding": 2, "bytes": 16384, "output": "out_act", "type": "u8"},
                {"binding": 3, "bytes": 8192, "output": "out_act2", "type": "u8"},
                {"binding": 4, "file": "in_act.bin"},
                {"binding": 5, "file": "wrep.bin"}
            ]
        }
    elif name == "rr_level":
        # One k_level unit, mode 0 ("the input is the unit's own level"), C = 32 (a 0 template int
        # falls back to 32), g = 1 head so the attention path runs, Cdown = Cup = 0 (no down or up
        # projection): 2 x 2 workgroups of one 8x8 window over the 16x16 level. Every binding the
        # shader declares (0..13) must be bound. Weights (0) and the item table are zero except s1d0 (unit 0,
        # table word 24 = f16 2.0), so every projection is zero: the block output is the input level through
        # the residual path (neither constant nor empty), and the encoder head (flags 3: o98 and oa0) gives
        # o98 = x88 in lanes 0..2 (gate sigmoid(0) = 0.5) and oa0 = the 2 x 2 averages of o98. The push reaches
        # offset 84 (render grid and head flags).
        open(os.path.join(out_dir, "in_act.bin"), "wb").write(
            np.tile(np.linspace(1, 250, 4096, dtype=np.uint8), 2).tobytes())   # [C/32][16][16][32]
        open(os.path.join(out_dir, "wbuf.bin"), "wb").write(np.zeros(4096, dtype=np.uint8).tobytes())
        open(os.path.join(out_dir, "x88.bin"), "wb").write(
            np.linspace(-2.0, 3.0, 16 * 16 * 4, dtype=np.float32).astype(np.float16).tobytes())   # [16][16][4] f16
        tab = bytearray(256)
        struct.pack_into("<I", tab, 24 * 4, 0x4000)                          # unit 0: s1d0 = f16 2.0
        open(os.path.join(out_dir, "tab.bin"), "wb").write(bytes(tab))
        push = bytearray(84)
        struct.pack_into("<10i", push, 0, 16, 16, 0, 0, 0, 32, 1, 0, 0, 0)    # W H sx sy mode C g Cdown Cup n
        struct.pack_into("<4i", push, 40, 16, 16, 0, 0)                       # kernel-test fallback dims
        struct.pack_into("<3i", push, 56, 32, 0, 0)                           # C_in, blob_off, unit
        struct.pack_into("<3i", push, 68, 16, 16, 3)                          # Wr, Hr, flags (o98 | oa0)
        cfg = {
            "groups": [2, 2, 1],
            "push": push.hex(),
            "bindings": [
                {"binding": 0, "file": "wbuf.bin", "bytes": 16384},
                {"binding": 1, "file": "in_act.bin"},
                {"binding": 2, "bytes": 8192, "output": "out_act", "type": "u8"},
                {"binding": 3, "bytes": 256},
                {"binding": 4, "bytes": 256},
                {"binding": 5, "file": "x88.bin", "bytes": 2048},
                {"binding": 6, "bytes": 256},
                {"binding": 7, "bytes": 2048, "output": "o98", "type": "f16"},
                {"binding": 8, "bytes": 512, "output": "oa0", "type": "f16"},
                {"binding": 9, "bytes": 256},
                {"binding": 10, "bytes": 256},
                {"binding": 11, "bytes": 256},
                {"binding": 12, "bytes": 256},
                {"binding": 13, "file": "tab.bin", "bytes": 256}
            ]
        }
    elif name == "sr_enc0L_in":
        w0 = ((w + 7) // 8) * 8
        h0 = ((h + 7) // 8) * 8
        b_col = np.linspace(0.1, 1.0, n_pix * 4, dtype=np.float16).tobytes()
        b_dep = np.linspace(0.1, 100.0, n_pix, dtype=np.float32).tobytes()
        b_mv = np.linspace(0.01, 0.5, n_pix * 2, dtype=np.float16).tobytes()
        open(os.path.join(out_dir, "g_col.bin"), "wb").write(b_col)
        open(os.path.join(out_dir, "g_dep.bin"), "wb").write(b_dep)
        open(os.path.join(out_dir, "g_mv.bin"), "wb").write(b_mv)
        open(os.path.join(out_dir, "prev_mv.bin"), "wb").write(b_mv)
        open(os.path.join(out_dir, "dhist.bin"), "wb").write(np.zeros(n_pix, dtype=np.uint32).tobytes())
        open(os.path.join(out_dir, "hidden.bin"), "wb").write(b_mv)
        open(os.path.join(out_dir, "wbuf.bin"), "wb").write(np.zeros(4096, dtype=np.uint8).tobytes())
        push = bytearray(148)
        struct.pack_into("<8i", push, 88, w, h, w0, h0, w, h, w, h)
        struct.pack_into("<i2f", push, 136, 0, 0.0, 0.0)
        total_l0 = (w0 + 8) * (h0 + 8) * 32
        cfg = {
            "groups": [(w * h + 63) // 64, 1, 1],
            "push": push.hex(),
            "bindings": [
                {"binding": 0, "file": "g_col.bin"},
                {"binding": 1, "file": "g_dep.bin"},
                {"binding": 2, "file": "g_mv.bin"},
                {"binding": 3, "file": "prev_mv.bin"},
                {"binding": 4, "file": "dhist.bin"},
                {"binding": 5, "file": "hidden.bin"},
                {"binding": 6, "bytes": n_pix * 4, "output": "prev_mv_out", "type": "f16"},
                {"binding": 7, "bytes": n_pix * 4, "output": "mv_ero", "type": "f16"},
                {"binding": 8, "bytes": n_pix * 4, "output": "mv_nondil", "type": "f16"},
                {"binding": 9, "bytes": total_l0, "output": "out_act", "type": "u8"},
                {"binding": 10, "file": "wbuf.bin"}
            ]
        }
    elif name == "sr_l0":
        open(os.path.join(out_dir, "wbuf.bin"), "wb").write(np.zeros(4096, dtype=np.uint8).tobytes())
        in_data = np.linspace(1, 250, 2048, dtype=np.uint8).tobytes()
        open(os.path.join(out_dir, "in_act.bin"), "wb").write(in_data)
        bindings = [
            {"binding": 0, "file": "wbuf.bin"},
            {"binding": 1, "file": "in_act.bin"},
            {"binding": 2, "bytes": 2048, "output": "out_act", "type": "u8"}
        ]
        cfg = {
            "groups": [32, 1, 1],
            "push": "",
            "bindings": bindings
        }
    elif name == "rr_gather":
        col = np.linspace(0.1, 1.0, n_pix * 4, dtype=np.float16).tobytes()
        mv = np.linspace(0.01, 0.5, n_pix * 2, dtype=np.float16).tobytes()
        dep = np.linspace(0.1, 100.0, n_pix, dtype=np.float32).tobytes()
        open(os.path.join(out_dir, "in_col.bin"), "wb").write(col)
        open(os.path.join(out_dir, "in_mv.bin"), "wb").write(mv)
        open(os.path.join(out_dir, "in_dep.bin"), "wb").write(dep)
        open(os.path.join(out_dir, "in_norm.bin"), "wb").write(col)
        open(os.path.join(out_dir, "in_alb.bin"), "wb").write(col)
        open(os.path.join(out_dir, "in_spa.bin"), "wb").write(col)
        open(os.path.join(out_dir, "in_rough.bin"), "wb").write(dep)
        push = bytearray(8)
        struct.pack_into("<2i", push, 0, w, h)
        cfg = {
            "groups": [(w + 15) // 16, (h + 15) // 16, 1],
            "push": push.hex(),
            "bindings": [
                {"binding": 0, "file": "in_col.bin"},
                {"binding": 1, "file": "in_mv.bin"},
                {"binding": 2, "file": "in_dep.bin"},
                {"binding": 3, "file": "in_norm.bin"},
                {"binding": 4, "file": "in_alb.bin"},
                {"binding": 5, "file": "in_spa.bin"},
                {"binding": 6, "file": "in_rough.bin"},
                {"binding": 7, "bytes": n_pix * 8, "output": "out_col", "type": "f16"},
                {"binding": 8, "bytes": n_pix * 4, "output": "out_dep", "type": "f32"},
                {"binding": 9, "bytes": n_pix * 4, "output": "out_mv", "type": "f16"},
                {"binding": 10, "bytes": n_pix * 8, "output": "out_norm", "type": "f16"},
                {"binding": 11, "bytes": n_pix * 16, "output": "out_alb", "type": "f32"},
                {"binding": 12, "bytes": n_pix * 16, "output": "out_spa", "type": "f32"},
                {"binding": 13, "bytes": n_pix * 4, "output": "out_rough", "type": "f32"}
            ]
        }
    elif name == "rr_ae_block":
        col = np.linspace(0.1, 1.0, n_pix * 4, dtype=np.float16).tobytes()
        open(os.path.join(out_dir, "in_col.bin"), "wb").write(col)
        push = bytearray(16)
        struct.pack_into("<2i", push, 8, w, h)
        cfg = {
            "groups": [(w + 15) // 16, (h + 15) // 16, 1],
            "push": push.hex(),
            "bindings": [
                {"binding": 0, "file": "in_col.bin"},
                {"binding": 1, "bytes": 8, "output": "ae_accum", "type": "u32"}
            ]
        }
    elif name == "rr_ae_final":
        open(os.path.join(out_dir, "ae_accum.bin"), "wb").write(struct.pack("<Q", 1 << 33))
        open(os.path.join(out_dir, "ae_prev.bin"), "wb").write(np.array([1.0], dtype=np.float16).tobytes())
        open(os.path.join(out_dir, "ae_next.bin"), "wb").write(np.array([1.0], dtype=np.float16).tobytes())
        push = bytearray(56)
        struct.pack_into("<2i", push, 8, w, h)
        struct.pack_into("<f", push, 24, 2.3283064365386963e-6 / float(w * h))
        struct.pack_into("<i", push, 28, 1)
        cfg = {
            "groups": [1, 1, 1],
            "push": push.hex(),
            "bindings": [
                {"binding": 0, "file": "ae_accum.bin"},
                {"binding": 1, "file": "ae_prev.bin"},
                {"binding": 2, "file": "ae_next.bin"},
                {"binding": 3, "bytes": 16, "output": "expo", "type": "f32"}
            ]
        }
    elif name == "rr_enc0_feat":
        w0 = ((w + 15) // 16) * 16
        h0 = ((h + 15) // 16) * 16
        col = np.linspace(0.1, 1.0, n_pix * 4, dtype=np.float16).tobytes()
        mv = np.linspace(0.01, 0.5, n_pix * 2, dtype=np.float16).tobytes()
        dep = np.linspace(0.1, 100.0, n_pix, dtype=np.float32).tobytes()
        norm = np.linspace(0.1, 1.0, n_pix * 4, dtype=np.float16).tobytes()
        alb = np.linspace(0.1, 1.0, n_pix * 4, dtype=np.float32).tobytes()
        spa = np.linspace(0.1, 1.0, n_pix * 4, dtype=np.float32).tobytes()
        rgh = np.linspace(0.1, 1.0, n_pix, dtype=np.float32).tobytes()
        expo = np.array([1.0, 1.0, 1.0, 0.0], dtype=np.float32).tobytes()
        open(os.path.join(out_dir, "in_col.bin"), "wb").write(col)
        open(os.path.join(out_dir, "in_mv.bin"), "wb").write(mv)
        open(os.path.join(out_dir, "in_dep.bin"), "wb").write(dep)
        open(os.path.join(out_dir, "in_nrm.bin"), "wb").write(norm)
        open(os.path.join(out_dir, "in_alb.bin"), "wb").write(alb)
        open(os.path.join(out_dir, "in_spa.bin"), "wb").write(spa)
        open(os.path.join(out_dir, "in_rgh.bin"), "wb").write(rgh)
        open(os.path.join(out_dir, "in_expo.bin"), "wb").write(expo)
        push = bytearray(16)
        struct.pack_into("<4i", push, 0, w0, h0, w, h)
        cfg = {
            "groups": [(w0 + 15) // 16, (h0 + 15) // 16, 1],
            "push": push.hex(),
            "bindings": [
                {"binding": 0, "file": "in_col.bin"},
                {"binding": 1, "file": "in_mv.bin"},
                {"binding": 2, "file": "in_dep.bin"},
                {"binding": 3, "file": "in_nrm.bin"},
                {"binding": 4, "file": "in_alb.bin"},
                {"binding": 5, "file": "in_spa.bin"},
                {"binding": 6, "file": "in_rgh.bin"},
                {"binding": 7, "file": "in_expo.bin"},
                {"binding": 8, "bytes": w0 * h0 * 64, "output": "out_be8", "type": "u8"},
                {"binding": 9, "bytes": w0 * h0 * 8, "output": "out_bec", "type": "f16"},
                {"binding": 10, "bytes": n_pix * 8, "output": "out_bf6", "type": "f16"},
                {"binding": 11, "bytes": n_pix * 4, "output": "out_bfa", "type": "f16"},
                {"binding": 12, "bytes": n_pix * 4, "output": "out_bfe", "type": "f16"}
            ]
        }
    elif name == "rr_pooled2":
        out_w, out_h = w // 2, h // 2
        c0 = np.arange(n_pix, dtype=np.uint32) & 1023
        c1 = (c0 * 3 + 17) & 1023
        c2 = (c0 * 7 + 31) & 1023
        col = (c0 | (c1 << 10) | (c2 << 20) | 0xc0000000).astype(np.uint32).tobytes()
        open(os.path.join(out_dir, "in_img.bin"), "wb").write(col)
        push = bytearray(32)
        struct.pack_into("<2i", push, 8, w, h)
        struct.pack_into("<2i", push, 24, out_w, out_h)
        cfg = {
            "groups": [(out_w + 15) // 16, (out_h + 15) // 16, 1],
            "push": push.hex(),
            "bindings": [
                {"binding": 0, "file": "in_img.bin"},
                {"binding": 1, "bytes": out_w * out_h * 4, "output": "out_img", "type": "u32"}
            ]
        }
    elif name == "rr_post_prep":
        col = np.linspace(0.1, 1.0, n_pix * 4, dtype=np.float16).tobytes()
        weights = np.full(n_pix, 0x80808080, dtype=np.uint32).tobytes()
        aniso = np.zeros(n_pix * 4, dtype=np.float16).tobytes()
        open(os.path.join(out_dir, "in_feat.bin"), "wb").write(col)
        open(os.path.join(out_dir, "in_weights.bin"), "wb").write(weights)
        open(os.path.join(out_dir, "in_aniso.bin"), "wb").write(aniso)
        push = bytearray(40)
        struct.pack_into("<i", push, 8, w)
        struct.pack_into("<2i", push, 32, w, h)
        cfg = {
            "groups": [(w + 15) // 16, (h + 15) // 16, 1],
            "push": push.hex(),
            "bindings": [
                {"binding": 0, "file": "in_feat.bin"},
                {"binding": 1, "file": "in_weights.bin"},
                {"binding": 2, "file": "in_aniso.bin"},
                {"binding": 3, "bytes": n_pix * 32, "output": "out_prep", "type": "f16"}
            ]
        }
    elif name == "rr_post":
        col = np.linspace(0.1, 1.0, n_pix * 4, dtype=np.float16).tobytes()
        expo = np.array([1.0, 1.0, 1.0, 0.0], dtype=np.float32).tobytes()
        open(os.path.join(out_dir, "in_feat.bin"), "wb").write(col)
        open(os.path.join(out_dir, "in_expo.bin"), "wb").write(expo)
        push = bytearray(120)
        struct.pack_into("<4i", push, 0, w, h, w, h)
        cfg = {
            "groups": [(w + 15) // 16, (h + 15) // 16, 1],
            "push": push.hex(),
            "bindings": [
                {"binding": 0, "file": "in_feat.bin"},
                {"binding": 1, "file": "in_expo.bin"},
                {"binding": 2, "bytes": n_pix * 4, "output": "out_rg", "type": "f16"},
                {"binding": 3, "bytes": n_pix * 2, "output": "out_b", "type": "f16"},
                {"binding": 4, "bytes": n_pix * 8, "output": "out_full", "type": "f16"},
                {"binding": 5, "bytes": ((w + 1) // 2) * ((h + 1) // 2) * 4, "output": "out_packed", "type": "u32"}
            ]
        }
    elif name == "rr_downsample":
        rg = np.linspace(0.1, 1.0, n_pix * 2, dtype=np.float16).tobytes()
        b = np.linspace(0.1, 1.0, n_pix, dtype=np.float16).tobytes()
        expo = np.array([1.0, 1.0, 1.0, 0.0], dtype=np.float32).tobytes()
        open(os.path.join(out_dir, "in_rg.bin"), "wb").write(rg)
        open(os.path.join(out_dir, "in_b.bin"), "wb").write(b)
        open(os.path.join(out_dir, "in_expo.bin"), "wb").write(expo)
        push = bytearray(48)
        struct.pack_into("<2i", push, 16, w, h)
        struct.pack_into("<2i", push, 40, w, h)
        cfg = {
            "groups": [(w + 15) // 16, (h + 15) // 16, 1],
            "push": push.hex(),
            "bindings": [
                {"binding": 0, "file": "in_rg.bin"},
                {"binding": 1, "file": "in_b.bin"},
                {"binding": 2, "file": "in_expo.bin"},
                {"binding": 3, "bytes": n_pix * 16, "output": "out_img", "type": "f32"}
            ]
        }
    else:
        # Default 2-buffer pipeline (binding 0: in, binding 1: out)
        data = np.linspace(0.1, 10.0, 1024, dtype=np.float16).tobytes()
        open(os.path.join(out_dir, "in.bin"), "wb").write(data)
        push = bytearray(16)
        struct.pack_into("<2i", push, 0, 32, 32)
        cfg = {
            "groups": [2, 2, 1],
            "push": push.hex(),
            "bindings": [
                {"binding": 0, "file": "in.bin"},
                {"binding": 1, "bytes": 2048, "output": "out", "type": "f16"}
            ]
        }

    with open(os.path.join(out_dir, "launch.json"), "w") as f:
        json.dump(cfg, f, indent=2)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(1)
    make_test(sys.argv[1], sys.argv[2])
