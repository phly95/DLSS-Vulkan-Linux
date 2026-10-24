# DLSS-SR in NeuralHub's host: what the decompile shows

Recovered from `neuralhub_host.dll` (NeuralHub v0.8.0) with Ghidra 12.1.4 (`tools/re/`). Addresses are
the DLL's virtual addresses (image base 0x180000000). Source file names come from its own error
messages (`sr_engine.h`, `sr_unit.hip`).

## Kernel handles and launches

Every kernel is registered by `FUN_180153bd0(fatbin, &handle, "mangled", ...)` in one function per
network (SR: `FUN_180107f70`). A launch is `hipConfigureCall`-style: `FUN_180154220(grid, block, 0, stream)`,
then `FUN_180154340` (pop config) and `FUN_1801544c0(&handle, grid, block, args, ...)`
(`hipLaunchKernel`). `tools/re/annotate.py` names the handles in the decompile.

## One SR frame (`FUN_180101f60`, `sr_unit.hip`)

In launch order:

| step | function | kernels |
| --- | --- | --- |
| gather the game's inputs | `FUN_180101f60` itself | `srunit::k_sr_gather`, `srunit::k_sr_gather_mv` |
| exposure | `FUN_180116c90` | `sr::io::k_sr_exposure` |
| level-0 encoder | `FUN_180116e70` | `k_sr_enc0` / `k_sr_enc0L_in` (preset L), then `k_sr_l0<0,2,b>` (`FUN_1801189d0`) |
| 10 stages | `FUN_180117970` -> `FUN_180118ca0(E, st, stream)` for st = 0..9 | `k_sr_stage<...>` / `k_sr_tube_split<128,4,b>` |
| level-0 decoder | `FUN_180117ab0` | `k_sr_l0<2,2,b>` (`FUN_18011a7b0`), `k_sr_dec0` |
| resolve | `FUN_180117ee0` | `k_sr_post<...>` (four launchers `FUN_18011aa80` ... `FUN_18011b3b0`) |
| history for the next frame | `FUN_180118240` | `k_sr_downsample` |

The trailing `bool` template argument is chosen by a global flag (`DAT_181a190c1`); with it set the
stage runs 128-wide workgroups, else 64.

`sr::StageArgs` is 56 bytes (7 qwords); the stage grid is `(args[5] + args[6] + c) >> 3` per axis.

## The network (layer table at 0x1801b9c00)

`{name, slot base, C, g, Cnext, mode}` rows, the order the frame runs them:

| layer | slot base | C | g | Cnext | mode |
| --- | --- | --- | --- | --- | --- |
| enc0L | 0x120 | 32 | 2 | 64 | 0 |
| enc1 | 0x270 | 64 | 2 | 96 | 0 (encoder: ends with a down-projection) |
| enc2 | 0x3c0 | 96 | 4 | 128 | 0 |
| tube0 ... tube5 | 0xf90, 0x10e0, 0x1230, 0x1380, 0x14d0, 0x1620 | 128 | 4 | 0 | 1 |
| dec2 | 0xba0 | 96 | 4 | 128 | 2 (decoder: starts with an up-projection) |
| dec1 | 0xcf0 | 64 | 2 | 96 | 2 |
| dec0L | 0xe40 | 32 | 2 | 64 | 2 |

Per layer (`FUN_180111850`), slots relative to the base, element counts checked:
`+0x40` C, `+0x50` C*g*32*2, `+0x60` C*g*32, `+0x70` C, `+0x80` g*256, `+0x90` C, `+0xa0` 4C*C,
`+0xb0` 4C, `+0xc0` C*4C, `+0xd0` C; mode 0 adds `+0xe0` Cnext*4C and `+0xf0` Cnext; mode 2 adds
`+0x100` Cnext*4C and `+0x110` 4C.

## Weights from nvngx_dlss.dll 310.9.1.0

`tools/sr/sr_weights.py` reads them the way NeuralHub does (only reads the DLL):

- One function in NVIDIA's DLL (RVA 0x89410 in 310.9.1.0) fills the table object:
  `switch (preset) { case 0xc: /* L */ ...; case 0xd: /* M */ ... }`, every slot k getting an element
  count at `[k]` and a data pointer at `[k+8]`. NeuralHub finds it by the same prologue bytes.
- Both presets: 139 slots (NeuralHub requires 0x8b), 1,596,003 (M) / 1,607,395 (L) values (NeuralHub
  requires 0x185a63 / 0x1886e3): **match**. Values are f16, contiguous in `.data`; slots 0xf60/0xf70/0xf80
  are single values 4 bytes apart.
- NeuralHub's checksum of a preset: SHA-256 over every slot in key order (`count` f16 values, slot
  0x200 read with 16 extra for preset M), then a 4096-byte "dither table" found in `.rdata` by an
  FNV-1a-64 (basis 0x14650fb0739d0383) of its first 16 bytes. In 310.9.1.0 the table is at RVA 0x135290
  and its SHA-256 matches NeuralHub's (`4707682a…`). **Preset L matches NeuralHub's checksum
  (`a9b03890…`)**. Preset M does not yet (`0ff032b7…` expected): the composition for M is not understood.

## Weight packing for the GPU (`FUN_180111850` and helpers)

- f16 -> e4m3 (`FUN_180114320`): to f32, |x| clamped to 448, normal range (|x| >= 2^-6) by
  `(bits + 0x407ffff + ((bits >> 20) & 1)) >> 20` on the f32 bits (round to nearest even), subnormals by
  `nearbyint(|x| * 512)`; NaN -> 0x7f.
- Fragment layout (`FUN_180113e10`, operand 'A' or 'B'): the WMMA register order for 16x16x16 fp8.
- f16 vectors are appended as is, or permuted within 32-element groups (`FUN_1801140a0`).
- Scales: `x * sqrt(C)` (`FUN_180115ee0`) and `x * 4*sqrt(2) * 0.5` (`FUN_180116130`), rounded to f16 RNE.
- A per-layer f32 bound (`* 1.05`) from the weights decides a flag per layer.

Transcribed in `tools/sr/sr_pack.py`: every blob of both presets has the size NeuralHub requires
(stages 0x118c0, 0x27f00, 0x2d000 x6, 0x28100, 0x11a00; enc0 0x3080 / 0x7700, dec0 0x2100 / 0x5f00).
The enc0 / dec0 blobs start with the 4096-byte dither table. Next: checked by running NeuralHub's own code objects under
ROCm (`/dev/kfd` works on this machine: ROCm 7.2.4 in podman sees gfx1201).

## The engine `E` (`FUN_18010db80`, engine init; `sr_engine.h`)

Indices are 32-bit ints of `E` unless marked q (64-bit slot = int index / 2). Names in quotes come
from the host's own error and dump strings.

| int idx | field |
| --- | --- |
| 0, 1 | render W, H |
| 2, 3 | W0, H0 = W, H rounded up to 32 |
| 4, 5 / 6, 7 / 8, 9 | W0/2, H0/2 / W0/4 / W0/8 (levels 1-3) |
| 10, 11 | output W, H |
| 12, 13 | "W15", "H15" = (out*3+1)>>1 (history size, 1.5x output) |
| 14 | frame parity `p0` (ping-pong of ihist / prev_mvec) |
| 15 | set per frame (passed to enc0) |
| 0x10 (q8) | "wbuf": all stage blobs, each 256-aligned; offsets at 0x12..0x24 (q9..q18) |
| 0x26 | "w16buf" (gfx11 only: f16 copy of wbuf by `k_sr_w16`; `g_sr_w16_delta` = w16buf - 2*wbuf) |
| 0x28 | "lutbuf": 2 x 65536 f16: `1/sqrt(h)` and `1/h` for every f16 h (symbol `g_sr_lut`) |
| 0x2a | "unorm8buf" 512 B (`k_sr_unorm8_table`; symbol `io::g_unorm8_p`) |
| 0x2c..0x2e+1 (bytes 0xb0..0xb9) | per-stage fp8-safe flags from the packer |
| 0x30, 0x36, ... 0x60 | 9 level tensors `{ptr, C, W, H}` (24 B each): C 64,64,96,96,128,128,128,96,64 at levels 1,1,2,2,3,3,3,2,1 |
| 0x66, 0x67 | W, H (copy) |
| 0x68, 0x69 | (W15+1)/2, (H15+1)/2 |
| 0x6a (q0x35), 0x6c (q0x36) | enc0 / dec0 blob pointers (inside wbuf) |
| 0x6e / 0x6f / 0x70 | dec0 scalars (f32): min(exp(s0), 8), exp(s1), exp(s2) |
| byte 0x261 / 0x262 / 0x263 | preset is L / enc0L fp8-safe / dec0L fp8-safe |
| 0x72 (q0x39), 0x74 (q0x3a) | "prev_mvec[0/1]" 4 B/px |
| 0x76 (q0x3b) | "mv_ero" 4 B/px |
| 0x78 (q0x3c) | "mv_nondil" 4 B x pitch(0x92) x H; texture object at 0x94 ("mvn_tex": RG16F, bilinear, clamp, normalized) |
| 0x7a, 0x7c (q0x3d, 0x3e) | "hidden[0/1]" 4 B/px |
| 0x7e (q0x3f) / 0x80 (q0x40) | "gates" / "factors" 4 B/px |
| 0x82 (q0x41) | "aniso" 8 B/px |
| 0x84 (q0x42) | "dhist" 4 B x (W15+1)/2 x (H15+1)/2 |
| 0x86, 0x88 (q0x43, 0x44) | "ihist[0/1]": RGBA16F W15 x H15, pitched (pitch at 0x8c); textures at 0x8e, 0x90 (bilinear, clamp, normalized) |
| 0x92 | pitch of mv_nondil in pixels (W rounded up to 64) |
| 0x96 (q0x4b) | "expo": f16x2, starts (1.0, 1.0) |
| 0x98 | bool: out/render ratio below a threshold (`DAT_1801b6318`) |
| 0x9a (q0x4d) / 0x9c (q0x4e) | "l0_skip" 32 B x W0 x H0 / "l0_in" 32 B x (W0+8) x (H0+8): preset L only |
| 0x9e, 0x9f | AE grid "ae_gw", "ae_gh" = min(max(render, out) / 2, const) |
| 0xa0 / 0xa2 | "ae_luma" 4 B per cell / "ae_tmp" (cells + 511) / 512 floats |
| 0xa4 | the AE result pointer of the last reduce |
| 0xa6 | the RGBA32F output, 16 B x outW x outH |

Kernel variant choice: the global `DAT_181a190c1` is "the device is gfx11" (arch name check). On gfx12
it is false: stages launch the `n = 2` instantiations with 64-thread workgroups and no f16 weight copy.

## Per-frame parameters `F` (the second argument of every launcher)

Read so far: +0x00 colour (RGBA16F, 8 B/px), +0x08 depth, +0x18 motion vectors (or +0x50 when byte
+0x4d is set: the output-resolution "mvhi" vectors), +0x10 the game's exposure texture, +0x20 / +0x28
two float pairs (jitter / jitter_prev, by the dump job's order), +0x2c, +0x30, +0x34, +0x38 floats,
+0x3c a float pair, +0x44 float, bytes +0x48 auto-exposure, +0x49 "ae_exact_sum", +0x4a, +0x4b,
+0x4c (fixed exposure: skip), +0x4d mvhi.

## Launch details

- exposure (`FUN_180116c90`): if F.+0x4c, nothing; if F.+0x48, auto-exposure (`FUN_1801184a0`):
  `k_sr_ae_luma` on an ae_gw x ae_gh grid (8x8 groups), then `k_sr_ae_reduce` (256 threads,
  512 values per group) repeated until one value is left, then `k_sr_ae_copy` into expo;
  else `k_sr_exposure(F.+0x10, F.+0x30, F.+0x44, expo, E[14])` (one 32-thread group).
- enc0 (`FUN_180116e70`): preset M runs `k_sr_enc0` on (W0/8, H0/8) groups of 32 threads; preset L
  runs `k_sr_enc0L_in` over (W0+8)(H0+8)/64 groups of 64 into l0_in, then `k_sr_l0<0,2,b>`.
  `Enc0Args` (280 B) is assembled from F, E's buffers above, the parity and derived ratios
  (W15/W, 1/W, ..., the constant log2(10) = 0x40549a78).

## The reference harness: NeuralHub's SR on Linux (`tools/sr/srref/`)

ROCm 7.2.4 (podman, `/dev/kfd`) loads `sr.gfx1201.co` as is and runs it on the RX 9070 XT. `srref.cpp`
transcribes the engine init and the frame (exposure, enc0 / enc0L_in + l0<0>, ten stages,
dec0 / l0<2>, post, downsample) with every argument block byte for byte; the game-input gather is
skipped (its colour path is a plain format conversion to RGBA16F; `g_col`, `g_dep`, `g_mv` are
written directly).

    build/venv/bin/python tools/sr/gen_frames.py picture.png frames/ --frames 8
    podman run --rm --device /dev/kfd --device /dev/dri --group-add keep-groups --security-opt label=disable \
        -v $PWD:/w docker.io/rocm/dev-ubuntu-24.04:7.2.4 bash -c 'cd /w && \
        hipcc -O2 -std=c++17 tools/sr/srref/srref.cpp -o build/sr/srref && \
        build/sr/srref build/nh/sr.gfx1201.co packed/M frames/ out/ [--L] [--dump]'
    build/venv/bin/python tools/sr/eval_sr.py picture.png frames/ out/

Static 1920x1080 -> 3840x2160 sequence from the NR verification's 4K Tomb Raider frame, Halton jitter,
PSNR against the 4K original (bicubic of the same frame: 29.0-32.8 dB):

| frame | 0 | 1 | 2 | 4 | 7 |
| --- | --- | --- | --- | --- | --- |
| preset M | 33.11 | 35.57 | 37.44 | 38.38 | 39.23 |
| preset L | 31.32 | 33.10 | 36.69 | 38.27 | 39.32 |

Two facts the decompile alone got wrong, found by running it:

- E[0x6e] (dec0 scalar 0, Dec0Args / L0Args +0x60) is an **f32** `min(exp(s0), 8)`: the decompiler
  prints `(int)` for a raw `movss`. As an int the dec0 kernel multiplied by ~1e-44, `aniso` stayed
  zero and the output was a soft 27 dB.
- Jitter is NGX's `Jitter.Offset` (sample at pixel centre minus jitter); the other sign gives 31 dB.

## `srunit::k_sr_gather` (`srref --gather`)

Copies the game's inputs into `g_col` (RGBA16F), `g_dep` (f32), `g_mv` (RG16F). `GatherArgs` (88 B,
FUN_180101f60): pointers colour (+0x00), motion vectors (+0x08), depth (+0x10); their row pitches in bytes
(+0x18, +0x1c, +0x20); their format codes (+0x24, +0x28, +0x2c; 0 = RGBA16F / RG16F / R32F, a plain copy:
`srref --gather` output is bit-identical to writing the engine buffers directly), then `g_col` (+0x30), `g_mv` (+0x38), `g_dep` (+0x40), render W, H (+0x48), mvhi flag (+0x50).
Colour format codes seen in the ISA (`build/re/fn.sh sr _ZN6srunit11k_sr_gatherENS_10GatherArgsE`):
5 = RGBA32F (to f16), 4 = R11G11B10F, 3 = RGB9E5, others 10:10:10:2 and 8-bit unorm; alpha is 1.0 for
formats without one. No tone or exposure change: a plain format conversion. `k_sr_gather_mv` converts
output-resolution motion vectors into `g_mvhi` when the game supplies them (F+0x4d).

`tools/re/fn.sh <net> <mangled>` prints one kernel's disassembly from `artifacts/neuralhub-v0.8.0/gfx1201/<net>.s.xz`.
