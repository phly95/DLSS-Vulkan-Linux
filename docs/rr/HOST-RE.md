# DLSS-RR (preset F) in NeuralHub's host: what the decompile shows

Recovered from `neuralhub_host.dll` (NeuralHub v0.8.0) with Ghidra (`tools/re/`), the same way as SR
(`docs/sr/HOST-RE.md`). Addresses are the DLL's virtual addresses. Source names come from its error
strings (`rr_engine.h`, `rr_unit.hip`). Everything here is implemented in `tools/rr/` and runs on the
RX 9070 XT under ROCm.

## Running the reference

    build/venv/bin/python -I tools/rr/rr_weights.py "<Crimson Desert>/bin64/nvngx_dlssd.dll" --out build/rr/weights
    build/venv/bin/python -I tools/rr/rr_pack.py build/rr/weights build/rr/packed
    build/venv/bin/python tools/rr/gen_rr_frames.py picture.png build/rr/frames --render 1920 1080 --output 3840 2160 --frames 8
    podman run --rm --device /dev/kfd --device /dev/dri --group-add keep-groups --security-opt label=disable \
        -v $PWD:/w docker.io/rocm/dev-ubuntu-24.04:7.2.4 bash -c 'cd /w && \
        hipcc -O2 -std=c++17 tools/rr/rrref/rrref.cpp -o build/rr/rrref && \
        build/rr/rrref build/nh/rr.gfx1201.co build/rr/packed build/rr/frames build/rr/out [--dump]'
    build/venv/bin/python -I tools/rr/eval_rr.py picture.png build/rr/frames build/rr/out

`rr_weights.py` needs `build/re/weights/dither.bin` from `tools/sr/sr_weights.py` (the same 4096-byte table
is in both DLLs). `RRREF_SYNC=1` synchronises after every launch (finds a faulting kernel).

Results (static test sequence from the NR verification's 4K Tomb Raider frame, Halton jitter, colour =
picture x gamma noise of relative deviation 0.5, diffuse albedo = clean picture; PSNR on 8-bit sRGB
against the clean picture):

| case | frame 0 | 1 | 2 | 4 | 7 | bicubic of the noisy colour | bicubic of the clean albedo |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1920x1080 -> 3840x2160 (`k_post<24>`) | 32.37 | 34.14 | 35.08 | 35.76 | 36.26 | 21.3-21.7 | 29.1-32.8 |
| 960x540 -> 1920x1080 | 30.38 | 32.05 | 33.05 | 33.71 | 34.19 | 20.9-21.5 | 27.1-30.5 |
| 960x540 -> 960x540 (`k_post<26>`) | 33.57 | 36.13 | 37.51 | 38.59 | (5) 38.97 | 20.0 | 29.3-34.9 |

GPU time 1080p -> 4K: about 12 ms per frame (gather included). This proves the pipeline runs and
denoises; it is not a comparison with NVIDIA's or NeuralHub-on-Windows output (none is available).

## Weights (`tools/rr/rr_weights.py`)

- NVIDIA's `nvngx_dlssd.dll` 310.9.1.0 (SHA-256 `4bc7ea5f…`) fills preset F's table in one straight-line
  function (RVA 0x36f60), found by its first 18 bytes `45 33 c0 4c 89 42 48 44 89 42 40 4c 89 82 98 00 00 00`
  (`xor r8d, r8d; mov [rdx+0x48], r8; ...`). NeuralHub replays its stores into a 0x2000-byte image
  (`FUN_18012aeb0`): tensor k = count at [k], pointer at [k+8]; some f16 scalars are immediates.
- 186 tensors, 3,193,136 f16 values (NeuralHub requires exactly these).
- Fingerprint (`a05b941a…`): SHA-256 over every tensor in key order, then the written 2-byte scalars at
  +0x220..+0x236 of each of the 11 level blocks (0x238 bytes each), then the dither table. **Matches.**
- The post pass's bicubic phase table: `kBicubicHwPhaseWeightsGlobal`, 0x2010 bytes, a symbol in the CUDA
  ELF images embedded in the DLL (12 identical copies; `FUN_18012d5e0`). SHA-256 `8f4c7577…` **matches**.

## Per-unit weights (`FUN_18012ba70`) and packing (`FUN_180131f80`), `tools/rr/rr_pack.py`

16 units k (names at 0x1801bac20): enc0..enc5, dec4_0, dec4_1, dec3_0, ..., dec0_1. Level block
L = k for k < 6, else 6 + (k - 6) / 2; the `_1` units use the block's second set (+0xa0).
C = 32, 64, 96, 128, 160, 160, 160, 128, 96, 64, 32 (by L), g = 2, 2, 4, 4, 8, 8, 8, 4, 4, 2, 2.

Per unit: g fp8 "group" matrices (64 x C, rows i*32.. and (g+i)*32.. of tensor +0x50), g symmetric f16
64x64 matrices ((m + m^T) * 0.5 * 2*sqrt(2), tensor +0x80), fp8 C x 32g (+0x60), f16 C (+0x70), fp8 4C x C
(+0xa0), f16 4C (+0xb0), fp8 C x 4C (+0xc0), f16 C (+0xd0); encoder units 1-4 a down-projection (Cdown x 4C,
columns reordered `j -> j / C + (j % C) * 4`), unit 1 also level 0's (64 x 128); decoder `_0` units an
up-projection (4C x Cup, rows reordered the same way); a 3 x C (encoders) or 4 x C (decoders but dec0) head
with a 16-value f16 bias block; unit 0 the enc0 feature matrices (64 x 43 padded to 64 x 64, 32 x 64);
unit 15 the dec0 scalars min(exp(s0), 8), exp(s1), exp(s2). Four u16 scalars per unit sit at W+0x1d0.

f16 -> e4m3 as SR's (clamp 448, RNE). The blob is 16-byte aligned items; their offsets (21 ints, -1 =
absent) go to the engine and on to every kernel (`rrk::Args` +0x28). RR's fp8 fragment order differs from
SR's (`FUN_180133070`, transcribed in `frag_rr`); the f16 32-group permutation is SR's.

## The engine `E` (`FUN_180126da0`, 0x3448 bytes)

`rrref.cpp` keeps E as NeuralHub's int array and allocates every buffer at NeuralHub's index (`alloc(i, n)`
= `FUN_180132f90`: {d, n} at int i). Fields:

| int | what |
| --- | --- |
| 0-3 | render W, H; output oW, oH |
| 4, 5 | W0, H0 = W, H rounded up to 32 |
| 6, 7 | "internal" 1.5x output ((o + (o & 1)) * 3 / 2); 8, 9 and 0xa, 0xb its halves |
| 0xc-0x11 / 0x12-0x17 | level widths / heights W0 >> l, l = 0..5 |
| 0x18 | parity (ping-pong of histories, halo shift (2,6)/(6,2)) |
| 0x19 | frame counter & 7; 0x1a: reset pending (1 at creation) |
| 0x1c + 4u | weight blob u {d, n}; 0x5c + 0x14u: its 20 item offsets; 0xbf0: blob 0's dither offset |
| 0x19c + 0x96u | W[u] (600 bytes; the kernels read only +0x1d0 scalars and unit 15's +0x248 floats) |
| 0xafa | dec0 scalar 2 (f32); 0xb02: enc0 f16 pair; 0xb03-0xb04: 8 bytes of table 0x230 |
| 0xb14 + 4u | f16 weight copies (gfx11 only; null on gfx12) |
| 0xb54-0xbd4 | U-Net activations per level (sizes in `rrref.cpp`) |
| 0xbe8 / 0xbec | enc0 features (64 B/px at W0 x H0) / 8 B/px |
| 0xbf1 | texture row pitch in pixels of the 1.5x history (E[6] rounded to the pitch alignment) |
| 0xc06, 0xc0a | the two 1.5x RGBA16F histories; texture objects at 0xd08, 0xd0a (bilinear, clamp, normalized) |
| 0xc22 | exposure (16 B); 0xc26 / 0xc2a: ae_tex[0/1] (one f16, set to 1.0 on reset); 0xcee: AE accumulator |
| 0xc6e-0xcda | the gather's outputs (21 buffers, order in `rrref.cpp`) |
| 0xcf2 | post taps (32 B/px); 0xcf6 / 0xcfa: 1.5x result (f32 / f16 planes); 0xcfe: the RGBA32F output |
| 0xd02 | bicubic phase table; 0xd06, 0xd07: previous jitter; 0xd0c-0xd0f: debug probe (0, -1, -1); 0xd10: stage-skip flags |

## One frame

`FUN_180120200` (per job): `k_rr_gather` (GatherArgs, 0x1e8 B: 19 input pointers, their row pitches in
bytes, their format codes, then 21 engine buffers, W/H, a "no inputs" flag), then `FUN_180127f10`:

| step | kernel | grid / block |
| --- | --- | --- |
| history pooling (not on reset) | `k_pooled2` | E[10..11] / 16x16 |
| auto exposure | `k_ae_block`, `k_ae_final` (AEArgs 56 B) | W x H / 16x16; 1 / 1 |
| features | `k_enc0_feat` (E0Args 0x248 B) | E[4..5] / 16x16 |
| enc0 | `k_level<32,2,5,0,0,4>` (`FUN_1801346f0`) | ((w + sx + 7) >> 3, (h + sy + 7) >> 3) / 32x1 |
| 15 units | `FUN_180133980(E, p)`, p = 0..14: enc1-5, then dec4_0, dec4_1 ... dec0_1 | as above, 32x4 (n = 1) or 32x1 (n = 4) |
| resolve | `k_post_prep`, `k_post<24>` (`<26>` when W / E[6] or H / E[7] > 0.5; PostArgs 0xb8 B) | W x H; E[6] x E[7]/2 / 16x16 |
| output | `k_downsample<0>` (1.5x -> output, into E 0xcfe) | oW x oH / 16x16 |

With a transparency layer or alpha upscaling the last step is `k_ds_fused` (or `k_ds_t` +
`k_downsample<1>` under a debug flag) instead; `rrref` does not run those. On gfx12 the `n = 4` variants
of enc0 and dec0_0 run (the gfx11 check `DAT_181a1b929` is false), with 32-thread groups.

### Frame parameters F (0xd0 B, built by `FUN_180120200` from the job struct `FUN_18005bc80` fills)

| F | what |
| --- | --- |
| 0x00 | jitter (2 f32, NGX `Jitter.Offset`) |
| 0x08 | reset |
| 0x0c | MV scale (2 f32) |
| 0x14 / 0x15 / 0x16 | virtual-hit MV on / SSS guide on / depth inverted |
| 0x18-0x97 | two 4x4 matrices for the virtual-hit MV (zero when off) |
| 0x98 | normals not RGBA16F; 0x99: hardware (not linear) depth; 0x9a: UNORM normals (xyz = 2v - 1) |
| 0x9b-0xa0 | inputs 8-13 present (roughness, specular MV, DoF guide, colour before transparency, disocclusion, responsivity) |
| 0xa4, 0xa8 | 1.0f, 0 (job +0xb4) |
| 0xac-0xbb | 4 f32 from ViewToClip when given, else (1, 1, 0, 1) |
| 0xbc | MV offset for jittered MVs (create flag), else 0 |
| 0xc4 | MV format is not RG16F; 0xc5: per-pixel jitter present; 0xc6-0xc8: transparency layer / + opacity / + MVs |
| 0xc9-0xcb | alpha upscaling / alpha input / alpha output |

Inputs (slot: name, format codes NeuralHub accepts; codes from the DXGI table at 0x1801b58d0, e.g. 0 =
RGBA16F, 2 = RG16F, 3 = R32F, 4 = R16F, 5 = RGBA8, 7 = RGB10A2, 10 = RG32F, 11 = RGBA32F): 0 colour,
1 motion vectors, 2 depth, 3 normals, 4 diffuse albedo, 5 specular albedo, 6 hit distance, 7 SSS guide,
8 roughness (null when roughness is in normals.w), 9 specular MV, 10 DoF guide, 11 colour before
transparency, 12 disocclusion mask, 13 responsivity mask, 14 transparency layer, 15 transparency opacity,
16 alpha, 17 per-pixel jitter, 18 transparency MVs.

## Not exercised yet

- Virtual-hit specular MV (hit distance + WorldToView / ViewToClip), SSS guide, specular MV, DoF guide,
  transparency inputs (`k_rr_gather_out`, `k_ds_fused`, `k_ds_t`, `k_downsample<1>`), alpha upscaling,
  per-pixel jitter, disocclusion / responsivity masks, UNORM normals, motion (the test is static).
- Two values are taken as raw float bits where the decompiler printed int casts (E0Args +0x1a8/+0x1ac, the
  previous jitter; same pattern as SR's dec0 scalar): consistent with how they are stored.
