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
                {"binding": 3, "bytes": 32 * 32 * 8, "output": "out", "type": "f16"}
            ]
        }
    elif name == "sr_enc0":
        # k_sr_enc0 (preset M): grid W0/8 x H0/8, 32 threads (one 8x8 tile = one wave32); the
        # host pads W0/H0 to 32. The push is the 256 bytes of
        # Enc0Args this device can push (maxPushConstantsSize = 256) at the oracle's own
        # offsets; binding 11 is expo (+0x80), binding 12 the repacked 64x128 weight tile.
        # W15/H15/hW15/hH15 follow the host's formulas (oW = 2*W), so W15/W = 3.0 like the
        # oracle and the dhist grid is (W15+1)/2 x (H15+1)/2.
        w0 = (w + 31) & ~31
        h0 = (h + 31) & ~31
        w15, h15 = 3 * w, 3 * h
        hw15, hh15 = (w15 + 1) // 2, (h15 + 1) // 2
        b_col = np.linspace(0.1, 1.0, n_pix * 4, dtype=np.float16).tobytes()
        b_dep = np.linspace(0.1, 100.0, n_pix, dtype=np.float32).tobytes()
        b_mv = np.linspace(0.01, 0.5, n_pix * 2, dtype=np.float16).tobytes()
        open(os.path.join(out_dir, "g_col.bin"), "wb").write(b_col)
        open(os.path.join(out_dir, "g_dep.bin"), "wb").write(b_dep)
        open(os.path.join(out_dir, "g_mv.bin"), "wb").write(b_mv)
        open(os.path.join(out_dir, "prev_mv.bin"), "wb").write(b_mv)
        open(os.path.join(out_dir, "dhist.bin"), "wb").write(
            np.zeros(hw15 * hh15, dtype=np.uint32).tobytes())
        open(os.path.join(out_dir, "hidden.bin"), "wb").write(b_mv)
        open(os.path.join(out_dir, "expo.bin"), "wb").write(
            np.array([1.0, 1.0], dtype=np.float16).tobytes())
        # The enc0 weight blob (12416 B, the size of packed enc0.bin): the dither table
        # (2048 u16) at 0, 8192 weight bytes at the frag positions (every byte is a finite
        # e4m3 code; signs alternate so the 32-term sums land around +-6.28 and t0 stays
        # non-constant - O7 rejects a constant output, and an all-zero blob would make every
        # t0 byte 0), 64 f16 biases at +12288.
        idx = np.arange(2048, dtype=np.uint64)
        dither = ((idx * 2654435761 + 1013904223) & 0xFFFF).astype(np.uint16)
        ii = np.arange(8192, dtype=np.uint64)
        mix = ((ii * 2246822519 + 3266489917) >> 16).astype(np.uint64)
        wts = np.where((ii & 1) == 0, 0x30 + (mix & 0x1F),
                       0xB0 + (mix & 0x1F)).astype(np.uint8)
        bias = np.linspace(-1.0, 1.0, 64, dtype=np.float16).tobytes()
        blob = dither.tobytes() + wts.tobytes() + bias
        assert len(blob) == 12416
        open(os.path.join(out_dir, "wbuf.bin"), "wb").write(blob)
        # The kernel's binding 12: the 64x128 tile repacked [N][K] row-major (the host's
        # sr_repack_enc0 walk, frag order -> rows), K contiguous for the matrix load.
        def frag_off(r, c):
            u1 = ((r & 0x7FFFFFE1) | ((r >> 3) & 2)) + (r & 0xE) * 2
            u5 = 2 * c
            low = (u5 & 4) | (c & 1) | ((c >> 3) & 2)
            high = ((u5 & 0x10) << 4) + ((16 * c) & 0xFFFFFE00) + (128 << 4) * (u1 >> 4)
            return (low | high) + ((u1 & 0xF) + (u5 & 8) * 2) * 8
        open(os.path.join(out_dir, "wrep.bin"), "wb").write(
            np.array([wts[frag_off(r, c)] for r in range(64) for c in range(128)],
                     dtype=np.uint8).tobytes())
        push = bytearray(256)
        struct.pack_into("<4i", push, 88, w, h, w, h)          # W, H, +0x60 render size again
        struct.pack_into("<4i", push, 104, w0, h0, w15, h15)   # W0, H0, W15, H15
        struct.pack_into("<2i", push, 120, hw15, hh15)         # hW15, hH15
        struct.pack_into("<i2f", push, 136, 0, 0.0, 0.0)       # parity, F20x, F20y
        struct.pack_into("<2f", push, 148, 0.0, 0.0)           # F28x, F28y
        struct.pack_into("<2f", push, 188, 0.0, 0.0)           # mvoffx, mvoffy
        struct.pack_into("<2I", push, 216, 0, 0)               # reset, fc
        struct.pack_into("<2I", push, 224, 0, (w + 63) & ~63)  # depinv, mvpitch
        struct.pack_into("<I", push, 252, 0)                   # ldr
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
                {"binding": 8, "bytes": n_pix * 4, "output": "mv_nondil", "type": "f16"},
                {"binding": 9, "bytes": (w0 // 2) * (h0 // 2) * 64, "output": "t0", "type": "u8"},
                {"binding": 10, "file": "wbuf.bin"},
                {"binding": 11, "file": "expo.bin", "bytes": 4, "type": "f16"},
                {"binding": 12, "file": "wrep.bin"}
            ]
        }
    elif name in ("sr_l0blk_pad", "sr_l0blk", "sr_l0dn", "sr_l0up", "sr_l0heads"):
        # Preset L's WMMA rewrite (perf mu_l0): specialised block/projection kernels.
        # Fixtures run under the NH_F16 lavapipe build only (O7): outputs are
        # sized for the widened types (f16 out), inputs are raw bytes (fp8 matrices and
        # activation tensors read identically in both builds).
        rng = np.random.default_rng(0x1E0)
        # moderate E4M3 codes (signs alternating): 0x10..0x1f / 0x90..0x9f
        codes = lambda n: np.where(rng.integers(0, 2, n) == 0,
                                   0x10 + rng.integers(0, 16, n),
                                   0x90 + rng.integers(0, 16, n)).astype(np.uint8)
        w0, h0 = 32, 16
        if name == "sr_l0blk_pad":
            iw, ih = w0 + 8, h0 + 8
            open(os.path.join(out_dir, "l0_in.bin"), "wb").write(codes(iw * ih * 32).tobytes())
            blob = bytearray(30464)
            blob[0:4096] = codes(4096).tobytes()
            blob[6272:6272 + 64] = np.linspace(0.5, 1.5, 32, dtype=np.float16).tobytes()
            for h in range(2):
                hb = 6336 + h * 3584
                blob[hb:hb + 1024] = codes(1024).tobytes()
                blob[hb + 1024:hb + 2048] = codes(1024).tobytes()
                blob[hb + 2048:hb + 2048 + 512] = \
                    np.linspace(-1.0, 1.0, 256, dtype=np.float16).tobytes()
                blob[hb + 2560:hb + 2560 + 1024] = codes(1024).tobytes()
            blob[13504:13504 + 64] = np.linspace(0.5, 1.5, 32, dtype=np.float16).tobytes()
            blob[13568:13568 + 64] = np.linspace(0.5, 1.5, 32, dtype=np.float16).tobytes()
            for j in range(4):
                b = 13632 + j * 2112 + (0 if j == 0 else 64)
                blob[b:b + 1024] = codes(1024).tobytes()
                blob[b + 1024:b + 1024 + 64] = \
                    np.linspace(-0.5, 0.5, 32, dtype=np.float16).tobytes()
                blob[b + 1088:b + 1088 + 1024] = codes(1024).tobytes()
                if j == 0:
                    blob[15744:15744 + 64] = \
                        np.linspace(-0.5, 0.5, 32, dtype=np.float16).tobytes()
            blob[4096:4096 + 2048] = codes(2048).tobytes()
            # step (a): the dn bias the fused epilogue reads (dn fixture's own values)
            blob[30336:30336 + 128] = np.linspace(-1, 1, 64, dtype=np.float16).tobytes()
            open(os.path.join(out_dir, "wbuf.bin"), "wb").write(bytes(blob))
            # step (a): the fused epilogue reads the dn matrix from wrep at the chain's
            # dnW offset (22144 = 6272 + 192 + 7168 + 8448 + 64); the head is the emb weights
            wrep = bytearray(30336)
            wrep[0:2048] = codes(2048).tobytes()
            wrep[22144:22144 + 8192] = codes(8192).tobytes()
            open(os.path.join(out_dir, "wrep.bin"), "wb").write(bytes(wrep))
            open(os.path.join(out_dir, "wvec.bin"), "wb").write(
                np.linspace(0.25, 1.75, 4096, dtype=np.float32).tobytes())
            open(os.path.join(out_dir, "wrepT.bin"), "wb").write(codes(131072).tobytes())
            sx = sy = 2
            push = struct.pack("<3I", 6272, 0, 0) + bytes(28) + \
                struct.pack("<6i", iw, ih, sx, sy, w0, h0)
            cfg = {
                "groups": [(w0 + sx + 7) // 8, (h0 + sy + 7) // 8, 1],
                "push": push.hex(),
                "bindings": [
                    {"binding": 0, "file": "wbuf.bin"},
                    {"binding": 1, "file": "l0_in.bin"},
                    {"binding": 2, "bytes": w0 * h0 * 32 * 2, "output": "skip", "type": "f16"},
                    {"binding": 3, "file": "wrep.bin"},
                    {"binding": 4, "file": "wvec.bin"},
                    {"binding": 5, "file": "wrepT.bin"},
                    {"binding": 6, "bytes": (w0 // 2) * (h0 // 2) * 64 * 2,
                     "output": "t0", "type": "f16"},
                ]
            }
        elif name == "sr_l0blk":
            # Step (b1): the mode-2 block now fuses the up projection + skip (binding 1 = the
            # half-res low tensor t8, binding 6 = l0_skip); no U input any more.
            open(os.path.join(out_dir, "low.bin"), "wb").write(codes((w0 // 2) * (h0 // 2) * 64).tobytes())
            open(os.path.join(out_dir, "skip.bin"), "wb").write(codes(w0 * h0 * 32).tobytes())
            blob = bytearray(24320)   # dec0L: up_W [128][64] fp8 at 0, up_b f16 at 8192, gain1 at 8448
            blob[8192:8192 + 256] = np.linspace(-1, 1, 128, dtype=np.float16).tobytes()
            blob[8448:8448 + 64] = np.linspace(0.5, 1.5, 32, dtype=np.float16).tobytes()
            open(os.path.join(out_dir, "wbuf.bin"), "wb").write(bytes(blob))
            open(os.path.join(out_dir, "wrep.bin"), "wb").write(codes(24320).tobytes())
            open(os.path.join(out_dir, "wvec.bin"), "wb").write(
                np.linspace(0.25, 1.75, 4096, dtype=np.float32).tobytes())
            open(os.path.join(out_dir, "wrepT.bin"), "wb").write(codes(131072).tobytes())
            sx = sy = 0
            # push: [0..11] wbuf_lo/hi + vec_off (0: the fixture's wvec holds one table's worth),
            # [12..27] renW/renH/s0/f0 (step b2's heads), [40..63] dimW/dimH/sx/sy/outW/outH
            # (bytes 28..39 stay zero padding)
            push = struct.pack("<3I", 0, 0, 0) + \
                struct.pack("<2i2f", w0, h0, 8.0, 0.09785986691713333) + bytes(12) + \
                struct.pack("<6i", w0, h0, sx, sy, w0, h0)
            cfg = {
                "groups": [(w0 + sx + 7) // 8, (h0 + sy + 7) // 8, 1],
                "push": push.hex(),
                "bindings": [
                    {"binding": 0, "file": "wbuf.bin"},
                    {"binding": 1, "file": "low.bin"},
                    {"binding": 2, "bytes": w0 * h0 * 32 * 2, "output": "B", "type": "f16"},
                    {"binding": 3, "file": "wrep.bin"},
                    {"binding": 4, "file": "wvec.bin"},
                    {"binding": 5, "file": "wrepT.bin"},
                    {"binding": 6, "file": "skip.bin"},
                    {"binding": 7, "bytes": w0 * h0 * 4, "output": "gates", "type": "u8"},
                    {"binding": 8, "bytes": w0 * h0 * 4, "output": "factors", "type": "u8"},
                    {"binding": 9, "bytes": w0 * h0 * 8, "output": "aniso", "type": "f16"},
                    {"binding": 10, "bytes": w0 * h0 * 4, "output": "hidden", "type": "u8"},
                ]
            }
        elif name == "sr_l0heads":
            wr, hr = 16, 16
            open(os.path.join(out_dir, "B.bin"), "wb").write(codes(w0 * h0 * 32).tobytes())
            push = struct.pack("<4I2f", w0, h0, wr, hr, 8.0, 0.09785986691713333)
            cfg = {
                "groups": [(wr * hr + 63) // 64, 1, 1],
                "push": push.hex(),
                "bindings": [
                    {"binding": 0, "file": "B.bin"},
                    {"binding": 1, "bytes": wr * hr * 4, "output": "gates", "type": "u8"},
                    {"binding": 2, "bytes": wr * hr * 4, "output": "factors", "type": "u8"},
                    {"binding": 3, "bytes": wr * hr * 8, "output": "aniso", "type": "f16"},
                    {"binding": 4, "bytes": wr * hr * 4, "output": "hidden0", "type": "u8"},
                ]
            }
        elif name == "sr_l0dn":
            open(os.path.join(out_dir, "skip.bin"), "wb").write(codes(w0 * h0 * 32).tobytes())
            blob = bytearray(30464)
            blob[30336:30336 + 128] = np.linspace(-1, 1, 64, dtype=np.float16).tobytes()
            open(os.path.join(out_dir, "wbuf.bin"), "wb").write(bytes(blob))
            wrep = bytearray(24192)
            wrep[22144:22144 + 8192] = codes(8192).tobytes()
            open(os.path.join(out_dir, "wrep.bin"), "wb").write(bytes(wrep))
            push = struct.pack("<3I", 6272, 0, 0) + bytes(28) + struct.pack("<4i", w0, h0, 0, 0)
            cfg = {
                "groups": [((w0 // 2) + 3) // 4, ((h0 // 2) + 3) // 4, 1],
                "push": push.hex(),
                "bindings": [
                    {"binding": 0, "file": "wbuf.bin"},
                    {"binding": 1, "file": "skip.bin"},
                    {"binding": 2, "bytes": (w0 // 2) * (h0 // 2) * 64 * 2,
                     "output": "t0", "type": "f16"},
                    {"binding": 3, "file": "wrep.bin"},
                ]
            }
        elif name == "sr_l0up":
            open(os.path.join(out_dir, "low.bin"), "wb").write(
                codes((w0 // 2) * (h0 // 2) * 64).tobytes())
            open(os.path.join(out_dir, "skip.bin"), "wb").write(codes(w0 * h0 * 32).tobytes())
            blob = bytearray(24320)
            blob[8192:8192 + 256] = np.linspace(-1, 1, 128, dtype=np.float16).tobytes()
            open(os.path.join(out_dir, "wbuf.bin"), "wb").write(bytes(blob))
            wrep = bytearray(24320)
            wrep[0:8192] = codes(8192).tobytes()
            open(os.path.join(out_dir, "wrep.bin"), "wb").write(bytes(wrep))
            push = struct.pack("<3I", 0, 0, 0) + bytes(28) + struct.pack("<4i", w0, h0, 0, 0)
            cfg = {
                "groups": [(w0 + 7) // 8, (h0 + 7) // 8, 1],
                "push": push.hex(),
                "bindings": [
                    {"binding": 0, "file": "wbuf.bin"},
                    {"binding": 1, "file": "low.bin"},
                    {"binding": 2, "file": "skip.bin"},
                    {"binding": 3, "bytes": w0 * h0 * 32 * 2, "output": "U", "type": "f16"},
                    {"binding": 4, "file": "wrep.bin"},
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
        # dec0_w: binding 7 is sr_dec0's OWN repacked layout (pipeline.cpp sr_repack_dec0), not
        # dec0.bin: [q][n][k] f16 weights (16384 B) then [q][n] f16 biases (256 B). The fixture
        # builds it from the same fp8 blob shape dec0.bin has, so the packing is still exercised.
        wts = (0x50 + (mix(8192) & 0x1F)).astype(np.uint16)     # fp8 matrix, sr_frag_off order
        bias = np.linspace(-1.0, 1.0, 128, dtype=np.float16)
        def _frag_off(r, c, cols=64):
            u1 = ((r & 0x7FFFFFE1) | ((r >> 3) & 2)) + (r & 0xE) * 2
            u5 = 2 * c
            low = (u5 & 4) | (c & 1) | ((c >> 3) & 2)
            high = ((u5 & 0x10) << 4) + ((16 * c) & 0xFFFFFE00) + (cols << 4) * (u1 >> 4)
            return (low | high) + ((u1 & 0xF) + (u5 & 8) * 2) * 8
        _invP = lambda c: (((c >> 1) & 7) << 2) | (c & 1) | (((c >> 4) & 1) << 1)
        src = wts[[_frag_off(j, k) for j in range(128) for k in range(64)]].reshape(128, 64)
        ee, mm = (src >> 3) & 15, src & 7
        vv = np.where(ee == 0, mm / 512.0, (1.0 + mm / 8.0) * np.exp2(ee.astype(np.int32) - 7))
        vv = np.where(src & 0x80, -vv, vv).astype(np.float16)    # e4m3 -> f16 is exact
        rbias = np.array([bias[(j & ~31) | _invP(j & 31)] for j in range(128)], dtype=np.float16)
        open(os.path.join(out_dir, "wbuf.bin"), "wb").write(vv.tobytes() + rbias.tobytes())
        # binding 9: the same fp8 128x64 tile, [N][K] row-major (sr_repack_dec0f8 on the host)
        open(os.path.join(out_dir, "wdec0f8.bin"), "wb").write(src.astype(np.uint8).tobytes())
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
                {"binding": 5, "bytes": n_pix * 8, "output": "aniso", "type": "f16"},
                {"binding": 6, "bytes": n_pix * 4, "output": "hidden_out", "type": "u8"},
                {"binding": 7, "file": "wbuf.bin"},
                # binding 8: the same t8 bytes as 16-byte vectors (the staging loop's view)
                {"binding": 8, "file": "in_t8.bin"},
                {"binding": 9, "file": "wdec0f8.bin"}
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
        # The kernel's tile is sized for a real 2x upscale (the SR chain's W15/W = 2), so the fixture
        # renders w x h and outputs 2w x 2h; the old 1x fixture (W15 = W) asked for a 34-pixel tile
        # and only worked because the tile clamp silently clamped it.
        ow, oh = 2 * w, 2 * h
        hw, hh = ow // 2, oh // 2
        # in_ihist is [H15][W15] f16vec4 = the OUTPUT size, not the render size
        open(os.path.join(out_dir, "ihist.bin"), "wb").write(
            np.linspace(0.1, 1.0, ow * oh * 4, dtype=np.float16).tobytes())
        open(os.path.join(out_dir, "prev_mv.bin"), "wb").write(mv)
        open(os.path.join(out_dir, "mv_ero.bin"), "wb").write(mv)
        open(os.path.join(out_dir, "mv_nondil.bin"), "wb").write(mv)
        push = bytearray(220)
        struct.pack_into("<I", push, 16, 0)              # parity
        struct.pack_into("<2i", push, 136, w, h)         # W, H
        struct.pack_into("<2i", push, 152, ow, oh)       # W15, H15
        struct.pack_into("<2i", push, 160, hw, hh)       # hW15, hH15
        # PostArgs +0xa8..+0xcc as pipeline.cpp now fills them (5981a2e): without dec0_f1, scaleX/Y
        # and log2_10 the rewritten sr_post computes zeros on every output and O7 sees a constant.
        struct.pack_into("<f", push, 168, 2.7530112)     # dec0_f1, the oracle's gate0 confidence scale
        struct.pack_into("<2f", push, 172, 0.5, 0.5)     # scaleX = W/W15, scaleY = H/H15
        struct.pack_into("<2f", push, 180, 0.0, 0.0)     # jitterX, jitterY
        struct.pack_into("<I", push, 204, 0x40549A78)    # log2_10
        cfg = {
            # sr_post's group is 32 output columns x 16 row pairs (a 2x2 output block per thread)
            "groups": [(ow + 31) // 32, (oh + 15) // 16, 1],
            "push": push.hex(),
            "bindings": [
                {"binding": 0, "file": "g_col.bin"},
                {"binding": 1, "file": "expo.bin"},
                {"binding": 2, "file": "gates.bin"},
                {"binding": 3, "file": "factors.bin"},
                {"binding": 4, "file": "aniso.bin"},
                # 5 is the history SAMPLED image and 9 the history STORAGE image (both W15 x H15
                # rgba16f; the sampler is linear/clamp-to-edge, set by kernel_run). The readback of
                # binding 9 is the same raw tightly-packed texel bytes the old f16 buffer dumped.
                {"binding": 5, "image": {"format": "r16g16b16a16_uint", "width": ow,
                                         "height": oh, "file": "ihist.bin", "sampled": True}},
                {"binding": 6, "file": "prev_mv.bin"},
                {"binding": 7, "file": "mv_ero.bin"},
                {"binding": 8, "file": "mv_nondil.bin"},
                {"binding": 9, "image": {"format": "r16g16b16a16_uint", "width": ow,
                                         "height": oh, "sampled": False, "output": "out_ihist"}},
                # 10 is the scalar full15 fallback (odd W15 only); 13 is the packed view the kernel
                # actually writes, so it carries the output name the O7/referee dumps read.
                {"binding": 10, "bytes": ow * oh * 4},
                {"binding": 11, "bytes": hw * hh * 4, "output": "dhist", "type": "u32"},
                {"binding": 13, "bytes": ow * oh * 4, "output": "full15", "type": "u32"}
            ]
        }
    elif name in ("sr_tube_split", "sr_swin64", "sr_swin96", "_shared_tube_swin"):
        # One wave per 4x4 window (16 tokens). The shader walks the whole 184576-byte tube blob
        # through wbuf16 (f16 vectors, binding 0) and wrep (row-major matrices, binding 3), and the
        # push dims are 16x16, so the tensors cover [4][16][16][32] = 32768 elements. in_act/out_act
        # are typed NR_OPERAND, which the O7 NH_F16 build widens to f16 (2 bytes an element) - the
        # fixture must hold 65536 bytes for those runs, while the fp8 build reads the same indices.
        open(os.path.join(out_dir, "in_act.bin"), "wb").write(
            np.tile(np.linspace(1, 250, 4096, dtype=np.uint8), 16).tobytes())       # 65536 B
        open(os.path.join(out_dir, "wbuf.bin"), "wb").write(
            np.tile(np.linspace(0.5, 2.0, 32768, dtype=np.float16).tobytes(), 6))   # 393216 B
        open(os.path.join(out_dir, "wrep.bin"), "wb").write(
            np.tile(np.linspace(0.05, 0.5, 32768, dtype=np.float16).tobytes(), 6))  # 393216 B
        push = bytearray(56)
        struct.pack_into("<4i", push, 40, 16, 16, 0, 0)
        cfg = {
            "groups": [64, 1, 1],
            "push": push.hex(),
            "bindings": [
                {"binding": 0, "file": "wbuf.bin"},
                {"binding": 1, "file": "in_act.bin"},
                {"binding": 2, "bytes": 65536, "output": "out_act", "type": "u8"},
                {"binding": 3, "file": "wrep.bin"}
            ]
        }
    elif name in ("sr_swin64", "sr_swin96"):
        # Same as the tube case plus binding 4 = the sr_swin vector table (brief item b): 16384 B
        # a stage, and push+8 = the table base (0 here).
        open(os.path.join(out_dir, "in_act.bin"), "wb").write(
            np.tile(np.linspace(1, 250, 4096, dtype=np.uint8), 16).tobytes())
        open(os.path.join(out_dir, "wbuf.bin"), "wb").write(
            np.tile(np.linspace(0.5, 2.0, 32768, dtype=np.float16).tobytes(), 6))
        open(os.path.join(out_dir, "wrep.bin"), "wb").write(
            np.tile(np.linspace(0.05, 0.5, 32768, dtype=np.float16).tobytes(), 6))
        open(os.path.join(out_dir, "wvec.bin"), "wb").write(
            np.linspace(0.25, 1.75, 4096, dtype=np.float32).tobytes() * 4)         # 65536 B
        open(os.path.join(out_dir, "wrepT.bin"), "wb").write(
            np.tile(np.linspace(0.05, 0.5, 32768, dtype=np.uint8).tobytes(), 4))   # 131072 B
        push = bytearray(56)
        struct.pack_into("<4i", push, 40, 16, 16, 0, 0)
        cfg = {
            "groups": [64, 1, 1],
            "push": push.hex(),
            "bindings": [
                {"binding": 0, "file": "wbuf.bin"},
                {"binding": 1, "file": "in_act.bin"},
                {"binding": 2, "bytes": 65536, "output": "out_act", "type": "u8"},
                {"binding": 3, "file": "wrep.bin"},
                {"binding": 4, "file": "wvec.bin"},
                {"binding": 5, "file": "wrepT.bin"}
            ]
        }
    elif name in ("sr_swin64m0", "sr_swin96m0", "sr_swin64m2", "sr_swin96m2"):
        # Fused block (Opus proposal 1): the split chain's sr_swin fixture plus binding 6 (the
        # mode wire - m0 writes the low grid there, m2 reads the encoder skip), on the group
        # grid: 16x16 render / ceil(16/8) = 2x2 groups of 8x8 pixels. O7 runs the f16 build,
        # which is the proven scalar body over that grid and never touches binding 6 - so the
        # low grid is a plain buffer here, NOT an output (a constant one fails O7).
        open(os.path.join(out_dir, "in_act.bin"), "wb").write(
            np.tile(np.linspace(1, 250, 4096, dtype=np.uint8), 16).tobytes())       # 65536 B
        open(os.path.join(out_dir, "wbuf.bin"), "wb").write(
            np.tile(np.linspace(0.5, 2.0, 32768, dtype=np.float16).tobytes(), 6))   # 393216 B
        open(os.path.join(out_dir, "wrep.bin"), "wb").write(
            np.tile(np.linspace(0.05, 0.5, 32768, dtype=np.float16).tobytes(), 6))  # 393216 B
        open(os.path.join(out_dir, "wvec.bin"), "wb").write(
            np.linspace(0.25, 1.75, 4096, dtype=np.float32).tobytes() * 4)          # 65536 B
        open(os.path.join(out_dir, "wrepT.bin"), "wb").write(
            np.tile(np.linspace(0.05, 0.5, 32768, dtype=np.uint8).tobytes(), 4))    # 131072 B
        push = bytearray(56)
        struct.pack_into("<4i", push, 40, 16, 16, 0, 0)
        cfg = {
            "groups": [2, 2, 1],
            "push": push.hex(),
            "bindings": [
                {"binding": 0, "file": "wbuf.bin"},
                {"binding": 1, "file": "in_act.bin"},
                {"binding": 2, "bytes": 65536, "output": "out_act", "type": "u8"},
                {"binding": 3, "file": "wrep.bin"},
                {"binding": 4, "file": "wvec.bin"},
                {"binding": 5, "file": "wrepT.bin"},
                {"binding": 6, "bytes": 65536}
            ]
        }
    elif name in ("sr_swin_proj", "sr_swin_proj64", "sr_swin_proj96"):
        # mode-0 patch projection: wbuf16 (dn_b), in_act (the block's output, fp8), out_act2,
        # wrep (dn_W). Push is the same StageArgs; the grid covers the low-pixel tiles.
        open(os.path.join(out_dir, "in_act.bin"), "wb").write(
            np.tile(np.linspace(1, 250, 4096, dtype=np.uint8), 8).tobytes())        # 32768 B
        open(os.path.join(out_dir, "wbuf.bin"), "wb").write(
            np.tile(np.linspace(0.5, 2.0, 32768, dtype=np.float16).tobytes(), 6))   # 393216 B
        open(os.path.join(out_dir, "wrep.bin"), "wb").write(
            np.tile(np.linspace(0.05, 0.5, 32768, dtype=np.float16).tobytes(), 6))  # 393216 B
        push = bytearray(56)
        struct.pack_into("<4i", push, 40, 16, 16, 0, 0)
        cfg = {
            "groups": [2, 2, 1],
            "push": push.hex(),
            "bindings": [
                {"binding": 0, "file": "wbuf.bin"},
                {"binding": 1, "file": "in_act.bin"},
                {"binding": 2, "bytes": 32768, "output": "out_act2", "type": "u8"},
                {"binding": 3, "file": "wrep.bin"}
            ]
        }
    elif name in ("sr_swin_up", "sr_swin_up64", "sr_swin_up96"):
        # mode-2 up + skip: wbuf16 (up_b), in_low, in_skip, out (the block's input), wrep.
        # wbuf_lo = 0 in this push, so the shader reads the C=96 / Cn=128 shape.
        open(os.path.join(out_dir, "wbuf.bin"), "wb").write(
            np.tile(np.linspace(0.5, 2.0, 32768, dtype=np.float16).tobytes(), 6))   # 393216 B
        open(os.path.join(out_dir, "wrep.bin"), "wb").write(
            np.tile(np.linspace(0.05, 0.5, 32768, dtype=np.float16).tobytes(), 6))  # 393216 B
        for f in ("low.bin", "skip.bin"):
            open(os.path.join(out_dir, f), "wb").write(
                np.tile(np.linspace(1, 250, 4096, dtype=np.uint8), 8).tobytes())    # 32768 B
        push = bytearray(56)
        struct.pack_into("<4i", push, 40, 16, 16, 0, 0)
        cfg = {
            "groups": [2, 2, 1],
            "push": push.hex(),
            "bindings": [
                {"binding": 0, "file": "wbuf.bin"},
                {"binding": 1, "file": "low.bin"},
                {"binding": 2, "file": "skip.bin"},
                {"binding": 3, "bytes": 32768, "output": "out", "type": "u8"},
                {"binding": 4, "file": "wrep.bin"}
            ]
        }
    elif name in ("sr_stage", "sr_stage_c64m0", "sr_stage_c96m0", "sr_stage_c96m2", "sr_stage_c64m2"):
        # Mode 0 (wbuf_lo = 0, st0): C = 64, 2 heads, 8 MLP blocks, 2x2 patch merge, on a 16x16
        # level. The stage reads five buffers: the f16 vector blob (0), the token bytes (1), the
        # mode-2 skip tensor (4; never read in mode 0, but statically used, so the launcher must
        # bind it) and the row-major matrix copy (5). Sizes cover every access the shader makes
        # here: the f16 blob up to ~72 KB, the matrix copy up to ~68 K elements (~135 KB when the
        # operand-probe build reads the same buffer as halfs).
        # The _c64/_c96/_m0/_m2 names are the specialized builds of the same source (C and mode
        # compile-time constants, see NR_SR_* in sr_stage.comp); they still take wbuf_lo = 0 for
        # this fixture - the constants replace the push-derived dispatch, the addresses stay low,
        # and the C = 96 variants just need larger buffers (3 channel planes, 12 MLP blocks).
        big = "_c96" in name
        open(os.path.join(out_dir, "in_act.bin"), "wb").write(
            np.tile(np.linspace(1, 250, 8192, dtype=np.uint8), 4).tobytes())   # [C/32][16][16][32]
        # Non-zero synthetic data so no code path collapses to a constant output: the f16 vector
        # blob is halfs in [0.5, 2], the matrix copy halfs in [0.05, 0.5] - finite both as halfs
        # and as the (low, high) byte pairs the fp8 build reads.
        open(os.path.join(out_dir, "wbuf.bin"), "wb").write(
            np.tile(np.linspace(0.5, 2.0, 65536, dtype=np.float16).tobytes(), 4 if big else 1))
        open(os.path.join(out_dir, "wrep.bin"), "wb").write(
            np.tile(np.linspace(0.05, 0.5, 16384, dtype=np.float16).tobytes(), 16 if big else 8))
        push = bytearray(56)
        struct.pack_into("<4i", push, 40, 16, 16, 0, 0)
        cfg = {
            "groups": [64, 1, 1],
            "push": push.hex(),
            "bindings": [
                {"binding": 0, "file": "wbuf.bin"},
                {"binding": 1, "file": "in_act.bin"},
                {"binding": 2, "bytes": 32768 if big else 16384, "output": "out_act", "type": "u8"},
                # mode 2 never writes out_act2 (the patch-merge operand is mode 0 only), so the
                # _m2 fixtures bind it without "output": an unbound buffer the shader declares
                # would crash, but a constant all-zero output file fails O7's const check.
                *([{"binding": 3, "bytes": 16384, "output": "out_act2", "type": "u8"}]
                  if not name.endswith("m2") else
                  [{"binding": 3, "bytes": 16384}]),
                {"binding": 4, "file": "in_act.bin"},
                {"binding": 5, "file": "wrep.bin"}
            ]
        }
    elif name == "rr_level":
        # One k_level unit, mode 0 ("the input is the unit's own level"), C = 32 (a 0 template int
        # falls back to 32), g = 1 head so the attention path runs, Cdown = Cup = 0 (no down or up
        # projection): 2 x 2 workgroups of one 8x8 window over the 16x16 level. Every binding the
        # shader declares (0..14) must be bound. Weights (0) and the item table are zero except s1d0 (unit 0,
        # table word 24 = f16 2.0), so every projection is zero: the block output is the input level through
        # the residual path (neither constant nor empty), and the encoder head (flags 3: o98 and oa0) gives
        # o98 = x88 in lanes 0..2 (gate sigmoid(0) = 0.5) and oa0 = the 2 x 2 averages of o98. The push reaches
        # offset 84 (render grid and head flags).
        # Bytes 1..120 only: offline O7 also runs the NH_F16 build, which reads this buffer as
        # f16 - byte pairs must avoid the f16 Inf/NaN range (u16 & 0x7c00 == 0x7c00 needs a byte
        # >= 124), and 0x7f/0xff would be e4m3 NaN codes.  1..120 is finite under both readings.
        open(os.path.join(out_dir, "in_act.bin"), "wb").write(
            np.tile(np.linspace(1, 120, 4096, dtype=np.uint8), 2).tobytes())   # [C/32][16][16][32]
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
                {"binding": 13, "file": "tab.bin", "bytes": 256},
                {"binding": 14, "file": "wbuf.bin", "bytes": 16384}
            ]
        }
    elif name == "rr_swin_u5":
        # rr_swin unit 5 (C=160, G=8, mode 0 + o98 head): same grid/push shape as
        # rr_level's case, but binding 0 = tiled stored-order weights and binding 13
        # = the f32 table at the unit-5 row (8MB + 5*139264). 2x2 workgroups of 128
        # threads over the 16x16 level; C/32 = 5 channel groups.
        open(os.path.join(out_dir, "in_act.bin"), "wb").write(
            np.tile(np.linspace(1, 250, 4096, dtype=np.uint8), 10).tobytes())
        open(os.path.join(out_dir, "wt.bin"), "wb").write(
            np.tile(np.linspace(1, 250, 8192, dtype=np.uint8), 64).tobytes())
        open(os.path.join(out_dir, "x88.bin"), "wb").write(
            np.linspace(-2.0, 3.0, 16 * 16 * 4, dtype=np.float32).astype(np.float16).tobytes())
        tab = bytearray(8 * 1024 * 1024 + 6 * 139264)
        toff = 8 * 1024 * 1024 + 5 * 139264
        vec = np.linspace(0.5, 2.0, 6 * 160 + 20 + 8 * 4096, dtype=np.float32)
        tab[toff:toff + len(vec.tobytes())] = vec.tobytes()
        open(os.path.join(out_dir, "wtab.bin"), "wb").write(bytes(tab))
        push = bytearray(84)
        struct.pack_into("<10i", push, 0, 16, 16, 0, 0, 0, 160, 8, 0, 0, 1)
        struct.pack_into("<4i", push, 40, 16, 16, 0, 0)
        struct.pack_into("<3i", push, 56, 160, 0, 5)
        struct.pack_into("<3i", push, 68, 16, 16, 1)
        cfg = {
            "groups": [2, 2, 1],
            "push": push.hex(),
            "bindings": [
                {"binding": 0, "file": "wt.bin"},
                {"binding": 1, "file": "in_act.bin"},
                {"binding": 2, "bytes": 40960, "output": "out_act", "type": "u8"},
                {"binding": 3, "bytes": 256},
                {"binding": 4, "bytes": 256},
                {"binding": 5, "file": "x88.bin", "bytes": 2048},
                {"binding": 6, "bytes": 256},
                {"binding": 7, "bytes": 2048, "output": "o98", "type": "f16"},
                {"binding": 8, "bytes": 512},
                {"binding": 9, "bytes": 256},
                {"binding": 10, "bytes": 256},
                {"binding": 11, "bytes": 256},
                {"binding": 12, "bytes": 256},
                {"binding": 13, "file": "wtab.bin"}
            ]
        }
    elif name == "rr_swin_u2t":
        # TEMPORARY block-only probe (C=96 G=8? no C=96 G=4 mode 0, no head): same shape
        # as rr_swin_u5's case but C=96 (NT=6, 3 channel groups), 2x2 workgroups over
        # a 16x16 level. Revert before merge (real u2 pipeline has down+heads).
        open(os.path.join(out_dir, "in_act.bin"), "wb").write(
            np.tile(np.linspace(1, 250, 4096, dtype=np.uint8), 6).tobytes())
        open(os.path.join(out_dir, "wt.bin"), "wb").write(
            np.tile(np.linspace(1, 250, 8192, dtype=np.uint8), 40).tobytes())
        open(os.path.join(out_dir, "x88.bin"), "wb").write(
            np.linspace(-2.0, 3.0, 16 * 16 * 4, dtype=np.float32).astype(np.float16).tobytes())
        tab = bytearray(8 * 1024 * 1024 + 6 * 139264)
        toff = 8 * 1024 * 1024 + 2 * 139264
        vec = np.linspace(0.5, 2.0, 6 * 96 + 20 + 4 * 4096, dtype=np.float32)
        tab[toff:toff + len(vec.tobytes())] = vec.tobytes()
        open(os.path.join(out_dir, "wtab.bin"), "wb").write(bytes(tab))
        push = bytearray(84)
        struct.pack_into("<10i", push, 0, 16, 16, 0, 0, 0, 96, 4, 0, 0, 1)
        struct.pack_into("<4i", push, 40, 16, 16, 0, 0)
        struct.pack_into("<3i", push, 56, 96, 0, 2)
        struct.pack_into("<3i", push, 68, 16, 16, 0)
        cfg = {
            "groups": [2, 2, 1],
            "push": push.hex(),
            "bindings": [
                {"binding": 0, "file": "wt.bin"},
                {"binding": 1, "file": "in_act.bin"},
                {"binding": 2, "bytes": 24576, "output": "out_act", "type": "u8"},
                {"binding": 3, "bytes": 256},
                {"binding": 4, "bytes": 256},
                {"binding": 5, "file": "x88.bin", "bytes": 2048},
                {"binding": 6, "bytes": 256},
                {"binding": 7, "bytes": 2048},
                {"binding": 8, "bytes": 512},
                {"binding": 9, "bytes": 256},
                {"binding": 10, "bytes": 256},
                {"binding": 11, "bytes": 256},
                {"binding": 12, "bytes": 256},
                {"binding": 13, "file": "wtab.bin"}
            ]
        }
    elif name == "rr_swin_u1":
        # rr_swin unit 1 (C=64 G=2 mode-1 merge + enc head + down): 2x2
        # workgroups over the 16x16 level; down out = 4 windows x 16 px x 96.
        # Small-magnitude synthetics (f16-half-safe byte pattern; lavapipe O7
        # needs finite, non-constant output - hot linspace saturates e4m3).
        # small finite byte pattern: e4m3 values in [0, 1.5]; keeps the down
        # GEMM under +-6.28125 (hot linspace clamp-saturates out_b = constant).
        patp = np.tile(np.array([0x00, 0x30, 0x00, 0x38, 0x00, 0x34, 0x00, 0x3C], dtype=np.uint8), 512)
        # weights on the subnormal rung (e4m3 0..0.004): keeps the down GEMM
        # (mA8 x merge x 4C) inside +-6.28125 so out_b is varied, not clamp-flat
        patw = np.tile(np.array([0x00, 0x01, 0x02, 0x03], dtype=np.uint8), 1024)
        open(os.path.join(out_dir, "in_act.bin"), "wb").write(
            np.tile(patp, 4).tobytes())
        open(os.path.join(out_dir, "wt.bin"), "wb").write(
            np.tile(patw, 24).tobytes())
        open(os.path.join(out_dir, "x88.bin"), "wb").write(
            np.linspace(-2.0, 3.0, 16 * 16 * 4, dtype=np.float32).astype(np.float16).tobytes())
        tab = bytearray(8 * 1024 * 1024 + 2 * 139264)
        toff = 8 * 1024 * 1024 + 1 * 139264
        # +962 covers the uniform extras incl. v170 at VH+800 (down bias)
        vec = np.linspace(0.01, 0.08, 6 * 64 + 20 + 2 * 4096 + 962, dtype=np.float32)
        # v170 ramp: per-channel bias variation keeps out_b from collapsing to
        # one e4m3 code (all-zero bias + uniform micro-weights = constant sum)
        vec[6 * 64 + 20 + 2 * 4096 + 800:6 * 64 + 20 + 2 * 4096 + 800 + 96] = \
            np.linspace(-0.4, 0.4, 96, dtype=np.float32)
        tab[toff:toff + len(vec.tobytes())] = vec.tobytes()
        open(os.path.join(out_dir, "wtab.bin"), "wb").write(bytes(tab))
        push = bytearray(84)
        struct.pack_into("<10i", push, 0, 16, 16, 0, 0, 1, 64, 2, 96, 0, 1)
        struct.pack_into("<4i", push, 40, 16, 16, 0, 0)
        struct.pack_into("<3i", push, 56, 32, 0, 1)
        struct.pack_into("<3i", push, 68, 16, 16, 3)
        cfg = {
            "groups": [2, 2, 1],
            "push": push.hex(),
            "bindings": [
                {"binding": 0, "file": "wt.bin"},
                {"binding": 1, "file": "in_act.bin"},
                {"binding": 2, "bytes": 16384, "output": "out_act", "type": "u8"},
                {"binding": 3, "bytes": 256},
                {"binding": 4, "bytes": 6144, "output": "out_b", "type": "u8"},
                {"binding": 5, "file": "x88.bin", "bytes": 2048},
                {"binding": 6, "bytes": 256},
                {"binding": 7, "bytes": 2048, "output": "o98", "type": "f16"},
                {"binding": 8, "bytes": 512, "output": "oa0", "type": "f16"},
                {"binding": 9, "bytes": 256},
                {"binding": 10, "bytes": 256},
                {"binding": 11, "bytes": 256},
                {"binding": 12, "bytes": 256},
                {"binding": 13, "file": "wtab.bin"}
            ]
        }
    elif name == "rr_swin_u2":
        # rr_swin unit 2 (C=96 G=4 mode-0 + enc head + down): down out =
        # 4 windows x 16 px x 128.
        open(os.path.join(out_dir, "in_act.bin"), "wb").write(
            np.tile(np.linspace(1, 250, 4096, dtype=np.uint8), 6).tobytes())
        open(os.path.join(out_dir, "wt.bin"), "wb").write(
            np.tile(np.linspace(1, 250, 8192, dtype=np.uint8), 40).tobytes())
        open(os.path.join(out_dir, "x88.bin"), "wb").write(
            np.linspace(-2.0, 3.0, 16 * 16 * 4, dtype=np.float32).astype(np.float16).tobytes())
        tab = bytearray(8 * 1024 * 1024 + 3 * 139264)
        toff = 8 * 1024 * 1024 + 2 * 139264
        vec = np.linspace(0.5, 2.0, 6 * 96 + 20 + 4 * 4096 + 96, dtype=np.float32)
        tab[toff:toff + len(vec.tobytes())] = vec.tobytes()
        open(os.path.join(out_dir, "wtab.bin"), "wb").write(bytes(tab))
        push = bytearray(84)
        struct.pack_into("<10i", push, 0, 16, 16, 0, 0, 0, 96, 4, 128, 0, 1)
        struct.pack_into("<4i", push, 40, 16, 16, 0, 0)
        struct.pack_into("<3i", push, 56, 96, 0, 2)
        struct.pack_into("<3i", push, 68, 16, 16, 3)
        cfg = {
            "groups": [2, 2, 1],
            "push": push.hex(),
            "bindings": [
                {"binding": 0, "file": "wt.bin"},
                {"binding": 1, "file": "in_act.bin"},
                {"binding": 2, "bytes": 24576, "output": "out_act", "type": "u8"},
                {"binding": 3, "bytes": 256},
                {"binding": 4, "bytes": 8192, "output": "out_b", "type": "u8"},
                {"binding": 5, "file": "x88.bin", "bytes": 2048},
                {"binding": 6, "bytes": 256},
                {"binding": 7, "bytes": 2048, "output": "o98", "type": "f16"},
                {"binding": 8, "bytes": 512, "output": "oa0", "type": "f16"},
                {"binding": 9, "bytes": 256},
                {"binding": 10, "bytes": 256},
                {"binding": 11, "bytes": 256},
                {"binding": 12, "bytes": 256},
                {"binding": 13, "file": "wtab.bin"}
            ]
        }
    elif name == "rr_swin_u3":
        # rr_swin unit 3 (C=128 G=4 mode-0 + enc head + down): down out =
        # 4 windows x 16 px x 160.
        open(os.path.join(out_dir, "in_act.bin"), "wb").write(
            np.tile(np.linspace(1, 250, 4096, dtype=np.uint8), 8).tobytes())
        open(os.path.join(out_dir, "wt.bin"), "wb").write(
            np.tile(np.linspace(1, 250, 8192, dtype=np.uint8), 56).tobytes())
        open(os.path.join(out_dir, "x88.bin"), "wb").write(
            np.linspace(-2.0, 3.0, 16 * 16 * 4, dtype=np.float32).astype(np.float16).tobytes())
        tab = bytearray(8 * 1024 * 1024 + 4 * 139264)
        toff = 8 * 1024 * 1024 + 3 * 139264
        vec = np.linspace(0.5, 2.0, 6 * 128 + 20 + 4 * 4096 + 96, dtype=np.float32)
        tab[toff:toff + len(vec.tobytes())] = vec.tobytes()
        open(os.path.join(out_dir, "wtab.bin"), "wb").write(bytes(tab))
        push = bytearray(84)
        struct.pack_into("<10i", push, 0, 16, 16, 0, 0, 0, 128, 4, 160, 0, 1)
        struct.pack_into("<4i", push, 40, 16, 16, 0, 0)
        struct.pack_into("<3i", push, 56, 128, 0, 3)
        struct.pack_into("<3i", push, 68, 16, 16, 3)
        cfg = {
            "groups": [2, 2, 1],
            "push": push.hex(),
            "bindings": [
                {"binding": 0, "file": "wt.bin"},
                {"binding": 1, "file": "in_act.bin"},
                {"binding": 2, "bytes": 32768, "output": "out_act", "type": "u8"},
                {"binding": 3, "bytes": 256},
                {"binding": 4, "bytes": 10240, "output": "out_b", "type": "u8"},
                {"binding": 5, "file": "x88.bin", "bytes": 2048},
                {"binding": 6, "bytes": 256},
                {"binding": 7, "bytes": 2048, "output": "o98", "type": "f16"},
                {"binding": 8, "bytes": 512, "output": "oa0", "type": "f16"},
                {"binding": 9, "bytes": 256},
                {"binding": 10, "bytes": 256},
                {"binding": 11, "bytes": 256},
                {"binding": 12, "bytes": 256},
                {"binding": 13, "file": "wtab.bin"}
            ]
        }
    elif name == "rr_swin_u4":
        # rr_swin unit 4 (C=160 G=8 mode-0 + enc head + down): down out =
        # 4 windows x 16 px x 160.
        # small pattern; wt must cover ALL tiles incl. mA8 (432640 B for
        # u4: 1690 tiles - np.tile needs >= 106 reps of 4096 B).
        patp = np.tile(np.array([0x00, 0x30, 0x00, 0x38, 0x00, 0x34, 0x00, 0x3C], dtype=np.uint8), 512)
        patw = np.tile(np.array([0x00, 0x01, 0x02, 0x03], dtype=np.uint8), 1024)
        open(os.path.join(out_dir, "in_act.bin"), "wb").write(
            np.tile(patp, 10).tobytes())
        open(os.path.join(out_dir, "wt.bin"), "wb").write(
            np.tile(patw, 112).tobytes())
        open(os.path.join(out_dir, "x88.bin"), "wb").write(
            np.linspace(-2.0, 3.0, 16 * 16 * 4, dtype=np.float32).astype(np.float16).tobytes())
        tab = bytearray(8 * 1024 * 1024 + 5 * 139264)
        toff = 8 * 1024 * 1024 + 4 * 139264
        vec = np.linspace(0.01, 0.08, 6 * 160 + 20 + 8 * 4096 + 962, dtype=np.float32)
        vec[6 * 160 + 20 + 8 * 4096 + 800:6 * 160 + 20 + 8 * 4096 + 800 + 160] = \
            np.linspace(-0.4, 0.4, 160, dtype=np.float32)
        tab[toff:toff + len(vec.tobytes())] = vec.tobytes()
        open(os.path.join(out_dir, "wtab.bin"), "wb").write(bytes(tab))
        push = bytearray(84)
        struct.pack_into("<10i", push, 0, 16, 16, 0, 0, 0, 160, 8, 160, 0, 1)
        struct.pack_into("<4i", push, 40, 16, 16, 0, 0)
        struct.pack_into("<3i", push, 56, 160, 0, 4)
        struct.pack_into("<3i", push, 68, 16, 16, 3)
        cfg = {
            "groups": [2, 2, 1],
            "push": push.hex(),
            "bindings": [
                {"binding": 0, "file": "wt.bin"},
                {"binding": 1, "file": "in_act.bin"},
                {"binding": 2, "bytes": 40960, "output": "out_act", "type": "u8"},
                {"binding": 3, "bytes": 256},
                {"binding": 4, "bytes": 10240, "output": "out_b", "type": "u8"},
                {"binding": 5, "file": "x88.bin", "bytes": 2048},
                {"binding": 6, "bytes": 256},
                {"binding": 7, "bytes": 2048, "output": "o98", "type": "f16"},
                {"binding": 8, "bytes": 512, "output": "oa0", "type": "f16"},
                {"binding": 9, "bytes": 256},
                {"binding": 10, "bytes": 256},
                {"binding": 11, "bytes": 256},
                {"binding": 12, "bytes": 256},
                {"binding": 13, "file": "wtab.bin"}
            ]
        }
    elif name in ("rr_swin_u7", "rr_swin_u9", "rr_swin_u11", "rr_swin_u13"):
        # rr_swin mode-3 decoder units (own-level copy entry + dec o98 head):
        # W=16 H=16, shift (4,0) -> grid 3x2; head reads x88/x90 (small f16
        # patterns keep the o98 f16 chain finite on lavapipe O7).
        cfg_m3 = {
            "rr_swin_u7": (160, 8, 7), "rr_swin_u9": (128, 4, 9),
            "rr_swin_u11": (96, 4, 11), "rr_swin_u13": (64, 2, 13),
        }
        C, G, uid = cfg_m3[name]
        NT = C // 16
        ntiles = G * 4 * NT + G * NT * 2 + (C // 8) * 2 * NT + (C // 8) * NT * 2 + NT
        wt_n = (ntiles * 256 + 4095) // 4096
        patp = np.tile(np.array([0x00, 0x30, 0x00, 0x38, 0x00, 0x34, 0x00, 0x3C], dtype=np.uint8), 512)
        patw = np.tile(np.array([0x00, 0x01, 0x02, 0x03], dtype=np.uint8), 1024)
        open(os.path.join(out_dir, "in_act.bin"), "wb").write(
            np.tile(patp, (16 * 16 * C) // 4096).tobytes())
        open(os.path.join(out_dir, "wt.bin"), "wb").write(
            np.tile(patw, wt_n).tobytes())
        open(os.path.join(out_dir, "x88.bin"), "wb").write(
            np.linspace(-1.0, 1.0, 16 * 16 * 4, dtype=np.float32).astype(np.float16).tobytes())
        open(os.path.join(out_dir, "x90.bin"), "wb").write(
            np.linspace(-1.0, 1.0, 8 * 8 * 4, dtype=np.float32).astype(np.float16).tobytes())
        tab = bytearray(8 * 1024 * 1024 + 16 * 139264)
        toff = 8 * 1024 * 1024 + uid * 139264
        vec = np.linspace(0.01, 0.08, 6 * C + 20 + G * 4096 + 96, dtype=np.float32)
        tab[toff:toff + len(vec.tobytes())] = vec.tobytes()
        open(os.path.join(out_dir, "wtab.bin"), "wb").write(bytes(tab))
        push = bytearray(84)
        struct.pack_into("<10i", push, 0, 16, 16, 4, 0, 3, C, G, 0, 0, 1)
        struct.pack_into("<4i", push, 40, 16, 16, 4, 0)
        struct.pack_into("<3i", push, 56, C, 0, uid)
        struct.pack_into("<3i", push, 68, 16, 16, 1)
        cfg = {
            "groups": [3, 2, 1],
            "push": push.hex(),
            "bindings": [
                {"binding": 0, "file": "wt.bin"},
                {"binding": 1, "file": "in_act.bin"},
                {"binding": 2, "bytes": 16 * 16 * C, "output": "out_act", "type": "u8"},
                {"binding": 3, "bytes": 256},
                {"binding": 4, "bytes": 256},
                {"binding": 5, "file": "x88.bin", "bytes": 2048},
                {"binding": 6, "file": "x90.bin", "bytes": 1024},
                {"binding": 7, "bytes": 2048, "output": "o98", "type": "f16"},
                {"binding": 8, "bytes": 256},
                {"binding": 9, "bytes": 256},
                {"binding": 10, "bytes": 256},
                {"binding": 11, "bytes": 256},
                {"binding": 12, "bytes": 256},
                {"binding": 13, "file": "wtab.bin"}
            ]
        }
    elif name in ("rr_swin_u6", "rr_swin_u8", "rr_swin_u10", "rr_swin_u12", "rr_swin_u14"):
        # rr_swin up units (mode-2 + tube entry, no heads): 2x2 workgroups over a
        # 16x16 level; low input is 8x8xCup, skip is full-res 16x16xC. Small-magnitude
        # synthetic weights/tables (lavapipe O7 needs finite, non-constant output).
        cfg_up = {
            "rr_swin_u6": (160, 8, 160, 6), "rr_swin_u8": (128, 4, 160, 8),
            "rr_swin_u10": (96, 4, 128, 10), "rr_swin_u12": (64, 2, 96, 12),
            "rr_swin_u14": (32, 2, 64, 14),
        }
        C, G, Cup, uid = cfg_up[name]
        NT = C // 16
        ntiles = G * 4 * NT + G * NT * 2 + (C // 8) * 2 * NT + (C // 8) * NT * 2 + NT + (4 * C // 16) * (Cup // 16)
        wt_need = ntiles * 256
        wt_n = (wt_need + 8191) // 8192
        open(os.path.join(out_dir, "in_act.bin"), "wb").write(
            np.tile(np.linspace(1, 40, 4096, dtype=np.uint8), max(1, 8 * 8 * Cup // 4096)).tobytes())
        open(os.path.join(out_dir, "wt.bin"), "wb").write(
            np.tile(np.linspace(1, 40, 8192, dtype=np.uint8), wt_n).tobytes())
        open(os.path.join(out_dir, "skip.bin"), "wb").write(
            np.tile(np.linspace(1, 40, 4096, dtype=np.uint8), 16 * 16 * C // 4096).tobytes())
        tab = bytearray(8 * 1024 * 1024 + 16 * 139264)
        toff = 8 * 1024 * 1024 + uid * 139264
        vec = np.linspace(0.02, 0.6, 6 * C + 20 + G * 4096 + 96, dtype=np.float32)
        tab[toff:toff + len(vec.tobytes())] = vec.tobytes()
        open(os.path.join(out_dir, "wtab.bin"), "wb").write(bytes(tab))
        push = bytearray(84)
        struct.pack_into("<10i", push, 0, 16, 16, 0, 0, 2, C, G, 0, Cup, 1)
        struct.pack_into("<4i", push, 40, 16, 16, 0, 0)
        struct.pack_into("<3i", push, 56, Cup, 0, uid)
        struct.pack_into("<3i", push, 68, 16, 16, 0)
        cfg = {
            "groups": [2, 2, 1],
            "push": push.hex(),
            "bindings": [
                {"binding": 0, "file": "wt.bin"},
                {"binding": 1, "file": "in_act.bin"},
                {"binding": 2, "bytes": 16 * 16 * C, "output": "out_act", "type": "u8"},
                {"binding": 3, "file": "skip.bin"},
                {"binding": 4, "bytes": 256},
                {"binding": 5, "bytes": 256},
                {"binding": 6, "bytes": 256},
                {"binding": 7, "bytes": 256},
                {"binding": 8, "bytes": 256},
                {"binding": 9, "bytes": 256},
                {"binding": 10, "bytes": 256},
                {"binding": 11, "bytes": 256},
                {"binding": 12, "bytes": 256},
                {"binding": 13, "file": "wtab.bin"}
            ]
        }
    elif name == "rr_swin_u0":
        # rr_swin unit 0 (C=32 G=2 mode-5 enc0 entry + enc head): C/32 = 1 channel
        # group; grid 2x2 over the 16x16 level. Table row unit 0 (8MB + 0).
        open(os.path.join(out_dir, "in_act.bin"), "wb").write(
            np.tile(np.linspace(1, 250, 4096, dtype=np.uint8), 8).tobytes())
        open(os.path.join(out_dir, "wt.bin"), "wb").write(
            np.tile(np.linspace(1, 250, 8192, dtype=np.uint8), 16).tobytes())
        open(os.path.join(out_dir, "x88.bin"), "wb").write(
            np.linspace(-2.0, 3.0, 16 * 16 * 4, dtype=np.float32).astype(np.float16).tobytes())
        tab = bytearray(8 * 1024 * 1024 + 1 * 139264)
        toff = 8 * 1024 * 1024 + 0 * 139264
        vec = np.linspace(0.5, 2.0, 6 * 32 + 20 + 2 * 4096 + 96, dtype=np.float32)
        tab[toff:toff + len(vec.tobytes())] = vec.tobytes()
        open(os.path.join(out_dir, "wtab.bin"), "wb").write(bytes(tab))
        push = bytearray(84)
        struct.pack_into("<10i", push, 0, 16, 16, 0, 0, 5, 32, 2, 0, 0, 4)
        struct.pack_into("<4i", push, 40, 16, 16, 0, 0)
        struct.pack_into("<3i", push, 56, 64, 0, 0)
        struct.pack_into("<3i", push, 68, 16, 16, 3)
        cfg = {
            "groups": [2, 2, 1],
            "push": push.hex(),
            "bindings": [
                {"binding": 0, "file": "wt.bin"},
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
                {"binding": 13, "file": "wtab.bin"}
            ]
        }
    elif name == "rr_swin_u15":
        # rr_swin unit 15 (C=32 G=2 mode-4 + unit-15 epilogue): grid 2x2 over the
        # 16x16 level; render grid 16x16. Table row unit 15 (8MB + 15*139264).
        # Small-magnitude synthetics (lavapipe O7 needs finite, non-constant output).
        # f16-safe pattern: low byte cycles 0..255, high byte in {0x30,0xB0} => small finite f16
        lo = np.tile(np.arange(256, dtype=np.uint8), 16)
        hi = np.tile(np.array([0x30, 0xB0], dtype=np.uint8), 2048)
        patf = np.empty(8192, dtype=np.uint8)
        patf[0::2] = lo[:4096]
        patf[1::2] = hi[:4096]
        open(os.path.join(out_dir, "in_act.bin"), "wb").write(
            np.tile(patf, 8).tobytes())
        open(os.path.join(out_dir, "wt.bin"), "wb").write(
            np.tile(patf, 16).tobytes())
        open(os.path.join(out_dir, "x88.bin"), "wb").write(
            np.linspace(-1.0, 1.0, 16 * 16 * 4, dtype=np.float32).astype(np.float16).tobytes())
        open(os.path.join(out_dir, "x90.bin"), "wb").write(
            np.linspace(-1.0, 1.0, 8 * 8 * 4, dtype=np.float32).astype(np.float16).tobytes())
        tab = bytearray(8 * 1024 * 1024 + 16 * 139264)
        toff = 8 * 1024 * 1024 + 15 * 139264
        vec = np.linspace(0.02, 0.6, 6 * 32 + 20 + 2 * 4096, dtype=np.float32)
        tab[toff:toff + len(vec.tobytes())] = vec.tobytes()
        open(os.path.join(out_dir, "wtab.bin"), "wb").write(bytes(tab))
        push = bytearray(84)
        struct.pack_into("<10i", push, 0, 16, 16, 0, 0, 4, 32, 2, 0, 0, 4)
        struct.pack_into("<4i", push, 40, 16, 16, 0, 0)
        struct.pack_into("<3i", push, 56, 32, 0, 15)
        struct.pack_into("<3i", push, 68, 16, 16, 1)
        cfg = {
            "groups": [2, 2, 1],
            "push": push.hex(),
            "bindings": [
                {"binding": 0, "file": "wt.bin"},
                {"binding": 1, "file": "in_act.bin"},
                {"binding": 2, "bytes": 256},
                {"binding": 3, "bytes": 256},
                {"binding": 4, "bytes": 256},
                {"binding": 5, "file": "x88.bin", "bytes": 2048},
                {"binding": 6, "file": "x90.bin", "bytes": 1024},
                {"binding": 7, "bytes": 2048},
                {"binding": 8, "bytes": 256},
                {"binding": 9, "bytes": 1024},
                {"binding": 10, "bytes": 1024, "output": "oc8", "type": "u8"},
                {"binding": 11, "bytes": 1024},
                {"binding": 12, "bytes": 2048},
                {"binding": 13, "file": "wtab.bin"}
            ]
        }
    elif name == "sr_enc0L_in":
        # k_sr_enc0L_in (preset L, oracle launch 2): one 8x8 tile of the PADDED grid a
        # workgroup, grid ((W0+8)+7)/8 x ((H0+8)+7)/8 (tile-exact here: w0/h0 are 8-aligned),
        # block 64; 12 bindings (0..5 inputs, 6..8 the MV outputs,
        # 9 l0_in, 10 the enc0L blob (only its dither table is read here), 11 expo).  The push
        # is 240 B at the oracle's own Enc0Args offsets incl. the preset-L constants the shader
        # reads: +0xd0 SC, +0xe8 k = 2774.359619140625/H (tools/sr/sr_enc0L_model.py).
        w0 = ((w + 7) // 8) * 8
        h0 = ((h + 7) // 8) * 8
        w15, h15 = 3 * w, 3 * h
        hw15, hh15 = (w15 + 1) // 2, (h15 + 1) // 2
        b_col = np.linspace(0.1, 1.0, n_pix * 4, dtype=np.float16).tobytes()
        b_dep = np.linspace(0.1, 100.0, n_pix, dtype=np.float32).tobytes()
        b_mv = np.linspace(0.01, 0.5, n_pix * 2, dtype=np.float16).tobytes()
        open(os.path.join(out_dir, "g_col.bin"), "wb").write(b_col)
        open(os.path.join(out_dir, "g_dep.bin"), "wb").write(b_dep)
        open(os.path.join(out_dir, "g_mv.bin"), "wb").write(b_mv)
        open(os.path.join(out_dir, "prev_mv.bin"), "wb").write(b_mv)
        open(os.path.join(out_dir, "dhist.bin"), "wb").write(
            np.linspace(0, 1023, hw15 * hh15, dtype=np.uint32).astype(np.uint32).tobytes())
        open(os.path.join(out_dir, "hidden.bin"), "wb").write(b_mv)
        open(os.path.join(out_dir, "expo.bin"), "wb").write(
            np.array([1.0, 1.0], dtype=np.float16).tobytes())
        # the enc0L blob's dither table (2048 u16, nonzero so the dither path runs)
        idx = np.arange(2048, dtype=np.uint64)
        dither = ((idx * 2654435761 + 1013904223) & 0xFFFF).astype(np.uint16)
        open(os.path.join(out_dir, "wbuf.bin"), "wb").write(dither.tobytes())
        push = bytearray(240)
        struct.pack_into("<4i", push, 88, w, h, w, h)          # W, H
        struct.pack_into("<4i", push, 104, w0, h0, w15, h15)   # W0, H0, W15, H15
        struct.pack_into("<2i", push, 120, hw15, hh15)         # hW15, hH15
        struct.pack_into("<i2f", push, 136, 0, 0.25, -0.25)    # parity, F20x, F20y
        struct.pack_into("<2f", push, 148, 0.0, 0.125)         # F28x, F28y
        struct.pack_into("<2f", push, 188, 0.0, 0.0)           # mvoffx, mvoffy
        struct.pack_into("<f", push, 208, 0.08225896209478378)  # +0xd0 SC (preset L)
        struct.pack_into("<2I", push, 216, 1, 0)               # reset, fc
        struct.pack_into("<f", push, 232, 2774.359619140625 / h)  # +0xe8 k (preset L)
        total_l0 = (w0 + 8) * (h0 + 8) * 32
        cfg = {
            "groups": [(w0 + 8) // 8, (h0 + 8) // 8, 1],
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
                {"binding": 9, "bytes": total_l0, "output": "l0_in", "type": "u8"},
                {"binding": 10, "file": "wbuf.bin"},
                {"binding": 11, "file": "expo.bin", "bytes": 4, "type": "f16"}
            ]
        }
    elif name == "sr_l0":
        # k_sr_l0 (preset L), mode 0 = k_sr_l0<0,2>: l0_in -> embed (ReLU clamp [0,448]) -> the
        # C=32 block over the padded (+2) border -> mode-0 down projection -> t0, plus l0_skip.
        # 5x5 workgroups of one 8x4 tile (two 4x4 windows) with shift (2,2) like frame 0, so the
        # +2 border reads and the in-range guards all run; W0=32/H0=16 keeps the buffers small.
        # Weights: the real enc0.bin blob shape (30464 B: dither, W1/W2 frag(32,32) + f16 biases,
        # the 24192 B layer) but synthetic: matrix bytes stay in the small-magnitude codes
        # (<= 0x1F: |v| <= 0.117) so the embedding never saturates the [0,448] clamp or the
        # +-6.28125 projection clamp, and the f16 vectors span [-1,1] - every output buffer then
        # varies per pixel (O7 rejects a constant output) and stays finite.
        w0, h0 = 32, 16
        sx = sy = 2
        padded = (w0 + 8) * (h0 + 8) * 32
        rng = np.random.default_rng(0x51A0)
        # moderate E4M3 codes (signs alternating): 0x10..0x1f / 0x90..0x9f, |v| <= 0.117
        codes = np.where(rng.integers(0, 2, 8192 + 2 * 8192 + 4 * 1024) == 0,
                         0x10 + rng.integers(0, 16, 8192 + 2 * 8192 + 4 * 1024),
                         0x90 + rng.integers(0, 16, 8192 + 2 * 8192 + 4 * 1024)).astype(np.uint8)
        dither = ((np.arange(4096, dtype=np.uint64) * 2654435761 + 1013904223) & 0xFF).astype(np.uint8)
        at = 0
        blob = bytearray()
        blob += dither.tobytes()
        blob += codes[at:at + 1024].tobytes(); at += 1024
        blob += np.linspace(-1, 1, 32, dtype=np.float16).tobytes()
        blob += codes[at:at + 1024].tobytes(); at += 1024
        blob += np.linspace(-1, 1, 32, dtype=np.float16).tobytes()
        blob += np.linspace(-1, 1, 32, dtype=np.float16).tobytes()        # gain1
        for h in range(2):
            blob += codes[at:at + 1024].tobytes(); at += 1024            # Wq
            blob += codes[at:at + 1024].tobytes(); at += 1024            # Wk
            blob += np.linspace(-1.0, 1.0, 256, dtype=np.float16).tobytes()  # rel 16x16
            blob += codes[at:at + 1024].tobytes(); at += 1024            # Wo
        blob += np.linspace(-1, 1, 32, dtype=np.float16).tobytes()        # v270
        blob += np.linspace(-1, 1, 32, dtype=np.float16).tobytes()        # gain2
        for j in range(4):
            blob += codes[at:at + 1024].tobytes(); at += 1024            # fc1
            blob += np.linspace(-1, 1, 32, dtype=np.float16).tobytes()    # b1
            blob += codes[at:at + 1024].tobytes(); at += 1024            # fc2
            if j == 0:
                blob += np.linspace(-1, 1, 32, dtype=np.float16).tobytes()  # fc2_bias
        blob += codes[at:at + 8192].tobytes(); at += 8192                # dn_W frag(64,128)
        blob += np.linspace(-1, 1, 64, dtype=np.float16).tobytes()        # dn_b
        assert len(blob) == 30464, len(blob)
        open(os.path.join(out_dir, "wbuf.bin"), "wb").write(bytes(blob))
        # l0_in: moderate codes on a deterministic ramp so every pixel differs
        idx = np.arange(padded, dtype=np.uint64)
        mix = ((idx * 0x9E3779B97F4A7C15) >> 32).astype(np.uint64)
        l0 = np.where((mix & 1) == 0, 0x10 + (mix & 0x0F), 0x90 + (mix & 0x0F)).astype(np.uint8)
        open(os.path.join(out_dir, "in_act.bin"), "wb").write(l0.tobytes())
        push = bytearray(112)
        struct.pack_into("<6i", push, 0x48, w0, h0, sx, sy, w0, h0)   # W0 H0 sx sy Wr Hr
        struct.pack_into("<2f", push, 0x60, 0.0, 0.0)                  # s0 f0 (mode 0: unused)
        struct.pack_into("<i", push, 0x68, 0)                          # mode 0
        cfg = {
            "groups": [(w0 + sx + 7) // 8, (h0 + sy + 3) // 4, 1],
            "push": push.hex(),
            "bindings": [
                {"binding": 0, "file": "wbuf.bin"},
                {"binding": 1, "file": "in_act.bin"},
                {"binding": 2, "file": "in_act.bin"},
                {"binding": 3, "bytes": (w0 // 2) * (h0 // 2) * 64, "output": "t0", "type": "u8"},
                {"binding": 4, "bytes": w0 * h0 * 32, "output": "l0_skip", "type": "u8"},
                {"binding": 5, "file": "in_act.bin", "bytes": 64},
                {"binding": 6, "file": "in_act.bin", "bytes": 64}
            ]
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
        # 68-byte RrGatherArgs: 7 row pitches (W * bpp of the ACTUAL format), 7 format codes,
        # normals_unorm, W, H. The fixture uses the default formats (col/nrm/alb/spa 0 = f16x4,
        # mv 2 = RG16F, dep/rgh 3 = R32F), so the outputs must equal the inputs bit for bit.
        push = bytearray(68)
        struct.pack_into("<7I", push, 0, w * 8, w * 4, w * 4, w * 8, w * 8, w * 8, w * 4)
        struct.pack_into("<7i", push, 28, 0, 2, 3, 0, 0, 0, 3)
        struct.pack_into("<3i", push, 56, 0, w, h)
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
                {"binding": 9, "bytes": n_pix * 4, "output": "out_mv", "type": "f16"}
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
        # opus_impl: bindings 4/5 are the RAW inputs now (format 0 = RGBA16F, pitch w * 8)
        alb = np.linspace(0.1, 1.0, n_pix * 4, dtype=np.float16).tobytes()
        spa = np.linspace(0.1, 1.0, n_pix * 4, dtype=np.float16).tobytes()
        rgh = np.linspace(0.1, 1.0, n_pix, dtype=np.float32).tobytes()
        # e_prev != e_cur so the history branch's e_cur*h/e_prev path is really exercised
        expo = np.array([1.25, 0.8, 0.8, 0.0], dtype=np.float32).tobytes()
        open(os.path.join(out_dir, "in_col.bin"), "wb").write(col)
        open(os.path.join(out_dir, "in_mv.bin"), "wb").write(mv)
        open(os.path.join(out_dir, "in_dep.bin"), "wb").write(dep)
        open(os.path.join(out_dir, "in_nrm.bin"), "wb").write(norm)
        open(os.path.join(out_dir, "in_alb.bin"), "wb").write(alb)
        open(os.path.join(out_dir, "in_spa.bin"), "wb").write(spa)
        open(os.path.join(out_dir, "in_rgh.bin"), "wb").write(rgh)
        open(os.path.join(out_dir, "in_expo.bin"), "wb").write(expo)
        # binding 13: dq()'s dither table (u16[2048]; the real one is blob0's meta[20] region,
        # here the same synthetic LCG table the SR enc0 fixtures use, so the dither path runs)
        dither = ((np.arange(2048, dtype=np.uint64) * 2654435761 + 1013904223) & 0xFFFF).astype(np.uint16)
        open(os.path.join(out_dir, "dither.bin"), "wb").write(dither.tobytes())
        # bindings 14/15: the history grids, u32 R10G10B10 (the chain's c0e / E0c12 sizes:
        # ((W15+1)/2)x((H15+1)/2) and one pooling level below, W15/H15 = 3x render)
        w15, h15 = 3 * w, 3 * h
        we, he = (w15 + 1) // 2, (h15 + 1) // 2
        wp, hp = (we + 1) // 2, (he + 1) // 2
        c0 = np.arange(we * he, dtype=np.uint32) & 1023
        c1 = (c0 * 3 + 17) & 1023
        c2 = (c0 * 7 + 31) & 1023
        c0e = (c0 | (c1 << 10) | (c2 << 20) | 0xc0000000).astype(np.uint32).tobytes()
        p0 = np.arange(wp * hp, dtype=np.uint32) & 1023
        p1 = (p0 * 5 + 11) & 1023
        p2 = (p0 * 13 + 3) & 1023
        c12 = (p0 | (p1 << 10) | (p2 << 20) | 0xc0000000).astype(np.uint32).tobytes()
        open(os.path.join(out_dir, "in_c0e.bin"), "wb").write(c0e)
        open(os.path.join(out_dir, "in_c12.bin"), "wb").write(c12)
        # binding 16: E0c02 (i8x4 render-grid hidden, the c22-family bilinear source)
        c02 = np.clip(np.arange(w * h * 4, dtype=np.int64).reshape(-1, 4) % 200 - 100,
                      -128, 127).astype(np.int8).tobytes()
        open(os.path.join(out_dir, "in_c02.bin"), "wb").write(c02)
        # 68-byte push; rst = 0 so the lavapipe run takes the HISTORY branch (O7 must see it).
        # +0x38 SC1, +0x3c/+0x40 pj_prev (the E0c02 texel offset).
        push = bytearray(84)
        struct.pack_into("<4iIIIf2f2f2if2f", push, 0, w0, h0, w, h, 3, 0, 0, -0.43359375,
                         0.125, -0.2, 0.05, -0.03, w15, h15, -0.2198486328125, 0.1, -0.15)
        struct.pack_into("<2I2i", push, 68, w * 8, w * 8, 0, 0)   # raw alb/spa pitch + format
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
                {"binding": 13, "file": "dither.bin"},
                {"binding": 14, "file": "in_c0e.bin"},
                {"binding": 15, "file": "in_c12.bin"},
                {"binding": 16, "file": "in_c02.bin"},
                # out_be8 is typed NR_OPERAND: 1 byte an element on the RADV build, and the O7
                # NH_F16 build widens it to f16 (the sr_tube_split convention) - size 2x for both
                {"binding": 8, "bytes": w0 * h0 * 128, "output": "out_be8", "type": "u8"},
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
        # NeuralHub's rrio::k_post<24>: PostArgs (0xb8) at the oracle's own field offsets and 12
        # bindings = the PostArgs pointer slots 0x40..0xa0 (tools/rr/rrref/rrref.cpp:541-544).
        # The fixture renders at the output size (W = W15, H = H15 -> scale 1) and takes the
        # HISTORY path (rst = 0), so the mv record, the E0d02 Catmull-Rom table and the 5-tap CR
        # fetch are all exercised on the lavapipe run: a zero/garbage table would make wcr 0/0
        # and O7's "finite" check would fail (the sr_post push fix 9eb91dd is the precedent).
        col = np.linspace(0.1, 1.0, n_pix * 16, dtype=np.float16)  # 16 f16 = 32 B per render pixel
        open(os.path.join(out_dir, "in_tap.bin"), "wb").write(col.tobytes())
        open(os.path.join(out_dir, "in_expo.bin"), "wb").write(
            np.array([1.0, 1.0, 1.0, 0.0], dtype=np.float32).tobytes())
        open(os.path.join(out_dir, "in_factors.bin"), "wb").write(
            np.full(n_pix * 4, 41, dtype=np.uint8).tobytes())
        mv = np.empty(n_pix * 4, dtype=np.float16)                  # (mvx, mvy, mvx, mvy) per pixel
        mv[0::2] = -0.2861328125
        mv[1::2] = -0.07501220703125
        open(os.path.join(out_dir, "in_mv.bin"), "wb").write(mv.tobytes())
        open(os.path.join(out_dir, "in_mv2.bin"), "wb").write(mv[:n_pix * 2].tobytes())
        open(os.path.join(out_dir, "in_hist.bin"), "wb").write(
            np.linspace(0.0, 0.9, n_pix * 4, dtype=np.float16).tobytes())
        # E0d02: 513 x (wm1, a = w0 + w1, w2, r) at f = k/512; row 0 is (0, 1, 0, 0), the
        # oracle's own first entry, and wcr stays well away from 0 for every index.
        f = np.arange(513, dtype=np.float32) / 512.0
        wm1 = -0.5 * f + f * f - 0.5 * f * f * f
        w0 = 1.0 - 2.5 * f * f + 1.5 * f * f * f
        w1 = 0.5 * f + 2.0 * f * f - 1.5 * f * f * f
        w2 = -0.5 * f * f + 0.5 * f * f * f
        aa = w0 + w1
        tab = np.stack([wm1, aa, w2, w1 / aa], 1).astype(np.float32)
        open(os.path.join(out_dir, "in_crtab.bin"), "wb").write(tab.tobytes())
        push = bytearray(184)
        struct.pack_into("<4i", push, 0, w, h, w, h)                # W, H, W15, H15
        struct.pack_into("<i", push, 16, 0)                         # rst: the history path
        struct.pack_into("<2f", push, 20, 1.0, 1.0)                 # scaleX, scaleY
        struct.pack_into("<2f", push, 28, 0.0, 0.0)                 # jitterX, jitterY
        struct.pack_into("<f", push, 52, 1.126047134399414)         # dec0 (+0x34)
        cfg = {
            "groups": [(w + 15) // 16, (((h + 1) // 2) + 15) // 16, 1],
            "push": push.hex(),
            "bindings": [
                {"binding": 0, "file": "in_tap.bin"},
                {"binding": 1, "file": "in_expo.bin"},
                {"binding": 2, "file": "in_factors.bin"},
                {"binding": 3, "file": "in_mv.bin"},
                {"binding": 4, "file": "in_mv2.bin"},
                {"binding": 5, "file": "in_mv2.bin"},
                {"binding": 6, "file": "in_hist.bin"},
                {"binding": 7, "file": "in_crtab.bin"},
                {"binding": 8, "bytes": n_pix * 8, "output": "out_hist", "type": "f16"},
                {"binding": 9, "bytes": n_pix * 4, "output": "out_rg", "type": "u32"},
                {"binding": 10, "bytes": n_pix * 2, "output": "out_b", "type": "f16"},
                {"binding": 11, "bytes": ((w + 1) // 2) * ((h + 1) // 2) * 4,
                 "output": "out_packed", "type": "u32"}
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
                {"binding": 3, "bytes": n_pix * 8, "output": "out_img", "type": "f16"}
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
