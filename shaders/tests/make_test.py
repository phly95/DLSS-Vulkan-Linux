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
        b_mv = np.linspace(0.01, 0.5, n_pix * 2, dtype=np.float16).tobytes()
        open(os.path.join(out_dir, "in_t8.bin"), "wb").write(np.zeros(n_pix * 16, dtype=np.uint8).tobytes())
        open(os.path.join(out_dir, "prev_mv.bin"), "wb").write(b_mv)
        open(os.path.join(out_dir, "hidden.bin"), "wb").write(b_mv)
        open(os.path.join(out_dir, "wbuf.bin"), "wb").write(np.zeros(4096, dtype=np.uint8).tobytes())
        push = bytearray(96)
        struct.pack_into("<6i", push, 64, w, h, w, h, w, h)
        cfg = {
            "groups": [(w * h + 31) // 32, 1, 1],
            "push": push.hex(),
            "bindings": [
                {"binding": 0, "file": "in_t8.bin"},
                {"binding": 1, "file": "prev_mv.bin"},
                {"binding": 2, "file": "hidden.bin"},
                {"binding": 3, "bytes": n_pix * 4, "output": "gates", "type": "f16"},
                {"binding": 4, "bytes": n_pix * 4, "output": "factors", "type": "f16"},
                {"binding": 5, "bytes": n_pix * 8, "output": "aniso", "type": "f16"},
                {"binding": 6, "bytes": n_pix * 4, "output": "hidden_out", "type": "f16"},
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
    elif name in ("sr_stage", "sr_tube_split", "rr_level"):
        in_act = np.linspace(1, 250, 4096, dtype=np.uint8).tobytes()
        open(os.path.join(out_dir, "in_act.bin"), "wb").write(in_act)
        open(os.path.join(out_dir, "wbuf.bin"), "wb").write(np.zeros(4096, dtype=np.uint8).tobytes())
        push = bytearray(56)
        struct.pack_into("<4i", push, 40, 16, 16, 0, 0)
        bindings = [
            {"binding": 0, "file": "wbuf.bin"},
            {"binding": 1, "file": "in_act.bin"},
            {"binding": 2, "bytes": 4096, "output": "out_act", "type": "u8"}
        ]
        if name == "sr_stage":
            bindings.append({"binding": 3, "bytes": 4096, "output": "out_act2", "type": "u8"})
        cfg = {
            "groups": [64, 1, 1],
            "push": push.hex(),
            "bindings": bindings
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
                {"binding": 1, "bytes": ((w + 15) // 16) * ((h + 15) // 16) * 4, "output": "block_luma", "type": "f32"}
            ]
        }
    elif name == "rr_ae_final":
        luma = np.array([0.25], dtype=np.float32).tobytes()
        open(os.path.join(out_dir, "block_luma.bin"), "wb").write(luma)
        cfg = {
            "groups": [1, 1, 1],
            "push": "",
            "bindings": [
                {"binding": 0, "file": "block_luma.bin"},
                {"binding": 1, "bytes": 4, "output": "expo", "type": "f16"}
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
        col = np.linspace(0.1, 1.0, n_pix * 4, dtype=np.float16).tobytes()
        open(os.path.join(out_dir, "in_img.bin"), "wb").write(col)
        push = bytearray(32)
        struct.pack_into("<2i", push, 8, w, h)
        struct.pack_into("<2i", push, 24, w, h)
        cfg = {
            "groups": [(w + 15) // 16, (h + 15) // 16, 1],
            "push": push.hex(),
            "bindings": [
                {"binding": 0, "file": "in_img.bin"},
                {"binding": 1, "bytes": n_pix * 8, "output": "out_img", "type": "f16"}
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
        open(os.path.join(out_dir, "in_feat.bin"), "wb").write(col)
        push = bytearray(120)
        struct.pack_into("<2i", push, 112, w, h)
        cfg = {
            "groups": [(w + 15) // 16, (h + 15) // 16, 1],
            "push": push.hex(),
            "bindings": [
                {"binding": 0, "file": "in_feat.bin"},
                {"binding": 1, "bytes": n_pix * 8, "output": "out_full", "type": "f16"}
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
