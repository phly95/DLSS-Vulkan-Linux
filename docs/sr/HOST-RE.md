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
