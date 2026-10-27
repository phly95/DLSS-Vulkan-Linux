# Notes: k_sr_dec0

## Launch Details
- Launch 13 in `sr_M` frame 0: `_ZN2sr2io9k_sr_dec0ENS0_8Dec0ArgsE`
- Grid: [32, 20, 1], Block: [32, 1, 1]
- Inputs:
  - `t8`: from launch 12 (128 channels per pixel at 128x80)
  - `prev_mvec1`: from launch 2
  - `hidden0`: from frame start
  - `wbuf+1590272`: dec0 weights from `packed/M/dec0.bin` (128x64 FP8 matrix + 128 half biases)
- Outputs:
  - `hidden1`: 147456 bytes (4 channels per pixel)
  - `gates`: 147456 bytes (3 channels per pixel + byte 3 zero)
  - `factors`: 147456 bytes (3 channels per pixel + byte 3 zero)
  - `aniso`: 294912 bytes (3 channels float16 + element 3 zero)

## Attempts and Hypotheses

### Attempt 1
- **Hypothesis**: Baseline level-0 decoder dispatch using decoded dimensions and argument layouts:
  - Disassembled `k_sr_dec0` from GFX1201 ISA (`/tmp/dec0.s`), showing initial 16x16x16 FP8 WMMA matrix multiplications accumulating across 128 input channels to 64 output channels, followed by clamping and subpixel/motion unpacking.
  - Decoded push constant offsets: `W` (64), `H` (68), `W0` (80), `H0` (84), `W15` (88), `H15` (92), `dec0_s0` (96, float 8.0), `dec0_f0` (100, float 0.078633), `jitterY` (108, float -0.166667).
- **Result**:
  - `hidden1 [u8]`: 147456 elements, bit-exact 784 (0.5317%), max |err| 255, max ulp 255, first mismatch `{'index': 0, 'port': 0.0, 'oracle': 75.0, 'port_bits': 0, 'oracle_bits': 75}`
  - `gates [u8]`: 147456 elements, bit-exact 35844 (24.3083%), max |err| 255, max ulp 255, first mismatch `{'index': 0, 'port': 0.0, 'oracle': 204.0, 'port_bits': 0, 'oracle_bits': 204}`
  - `factors [u8]`: 147456 elements, bit-exact 36077 (24.4663%), max |err| 252, max ulp 252, first mismatch `{'index': 0, 'port': 0.0, 'oracle': 4.0, 'port_bits': 0, 'oracle_bits': 4}`
  - `aniso [u8]`: 294912 elements, bit-exact 72873 (24.7101%), max |err| 255, max ulp 255, first mismatch `{'index': 0, 'port': 0.0, 'oracle': 162.0, 'port_bits': 0, 'oracle_bits': 162}`

## Review R3 (2026-10-26, external review): dec0 mostly decoded from the ISA + oracle data

Method:
- The packer (`sr_pack.build`) gives `dec0.bin` = `frag(128 x 64)` + 128 P-permuted halfs. That
  is a 1x1 projection of t8's 64 channels to 4 sub-pixels x 32 channels (`u = W @ t8 + b`), with
  channel `q*32 + c` and pixel shuffle `q = dx + 2*dy`, as in the stage family's mode 2.
- The output formulas were read from the SSA expression trees of the stores
  (`tools/re/ssa.py build/k_sr_dec0.s`, listing in `build/work/review/dec0/dec0.ssa`) and checked
  against the oracle's launch 13 (frame 0, a reset, so `hidden0 = 0`).
- Scripts (git-ignored): `build/work/review/dec0/dec0_rank.py` (rank correlation of every
  output against the 32 channels), `dec0_fit.py`.

Decoded (p = the projected channels, **rounded to fp8**: the kernel stages them in LDS as fp8 and
reads them back with `v_cvt_f32_fp8`; the LDS byte order is the P permutation, so the stored index
i holds logical channel P[i]):

| output | formula | exact vs oracle |
|---|---|---:|
| gates[0] | `rndne(255 * clamp01(f16(sigmoid(p0))))` | 94.87% |
| gates[1] | data fit `255 * (1 - (1-s(p1))(1-s(p5)))`; the ISA shows a single `sigmoid` of an fp8 LDS byte, so p1/p5 are probably combined while staging (open) | 98.44% |
| gates[2] | data fit `255 * (1 - (1-s(p10))(1-s(p5)))`, same caveat | 98.73% |
| factors[0..2] | `255 * softmax(p11, p12, p13)` (max-subtracted, `exp((a-m) log2 e)`) | 99.16 / 99.19 / 98.82% |
| aniso[0] | `f16(A^2)`, `A = s0 * f16(sigmoid(clamp(p14, +-20)))`, s0 = `dec0_s0` (args +0x60, 8.0) | rel rms 0.0049 |
| aniso[1] | `f16(B^2)`, `B = s0 * f16(sigmoid(p15))` | rel rms 0.0030 |
| aniso[2] | `f16(A * B * tanh(f0 * p16))`, f0 = `dec0_f0` (args +0x64, 0.0786) | ISA |
| hidden1[k] | **signed int8** `rndne(127 * clamp(h, -1, 1))`, with `h = f16(z * tanh(p_h)) + f16((1-z) * h_warp)`, `z = 0.5 + 0.5*tanh(0.5*p_z)` (f16), and `h_warp` = hidden0 (int8/127) bilinearly sampled at the `prev_mvec1`-reprojected position, fractions quantized to 1/256, 0 when off-screen. hidden1[0]: z <- p6, h <- p2 (pattern for k: z <- p(6+k), h <- p(2+k), to be confirmed per channel) | ISA |

Byte 3 of gates/factors and element 3 of aniso are zero. Old port: hidden1 0.55%, gates 32.6%,
factors 26.2%, aniso 25.0% bit-exact. Its channel choices (27, 5, 29 / 24, 8, 20) and its u8
hidden encoding are refuted.

Open:
- gates[1..2]'s staging;
- hidden1's per-channel indices and the exact warp, which needs a frame with `hidden0 != 0`
  (frame 1);
- the exact rounding of the projection (WMMA fp8 with f32 accumulation, and whether partial
  sums go through f16).
A synthetic-weight srref probe on `dec0.bin` (one projected channel live at a time) settles the
first two directly.

## Review R3b (2026-10-26): dec0 fully decoded, 99.98-100% bit-exact on both frames

The R3 residue came from the projection's rounding, not the heads. The ISA (0x1f0..0x374) computes
`u = f16(f16(b + A0) + A1)`, where A0 and A1 are the f32 WMMA chains over t8 channels 0..31 and 32..63.
It then clamps u to +-448 and stores fp8 to LDS. With that p:

- gates are **single** sigmoids, `rndne(255 * clamp01(f16(1 / (1 + exp(-p_c)))))` for c = 0, 1, 10.
  The divide is IEEE (`v_div_scale/fmas/fixup`). R3's "OR" data fits are refuted.
- factors = `255 * softmax(p11..p13)` in f32, not rounded to f16;
- aniso[2] = `f16((A * B) * tanh(f0 * p16))`, where tanh is the f32 `tanhf` (a polynomial for small
  |x|, an exp form above it) of the **unrounded** f32 product;
- hidden: z <- p(6+k), h <- p(2+k) for all four k. The warp is decoded exactly, with an f32
  reprojection through prev_mvec, coordinates snapped to 1/256, edge-clamped taps, an int8/127
  dequant, an f32 fma bilinear and one f16 rounding. It is 0 when the reprojected position leaves
  [0, W15] x [0, H15] or args +0x88 != 0.

The spec is `tools/sr/sr_dec0_model.py` (formulas and rounding points in its docstring).

| frame | gates | factors | aniso | hidden out |
|---|---:|---:|---:|---:|
| 0 (reset) | 99.9993% | 100% | 99.9878% | 99.9986% |
| 1 (warp) | 100% | 99.9973% | 99.9803% | 99.9993% |

`build/venv/bin/python -I tools/sr/sr_dec0_model.py --frame F` (oracle inputs).

The aniso residue is the kernel's tanhf branch choice for |x| < ~2e-3, together with its v_exp
approximation.

The current `sr_dec0.comp` and the host's push constants (pipeline.cpp fills only +0x40..+0x67, so
F20, F28 and the scale factors are 0) predate this spec.

Preset L has no dec0. Its equivalent heads come from `k_sr_l0` (launch 14), not yet decoded.
