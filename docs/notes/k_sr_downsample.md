# Notes: k_sr_downsample

## Launch Details
- Launch 15 in `sr_M` frame 0: `_ZN2sr2io15k_sr_downsampleENS0_8DownArgsE`
- Grid: [32, 18, 1], Block: [256, 1, 1] (16x16 local)
- Inputs: `full15` (R11G11B10 unorm), `g_col`, `expo`
- Output: `out` (RGBA32F, 512x288x4 = 589824 elements)

## Attempts and Hypotheses

### Attempt 1
- **Hypothesis**: 4-tap separable polyphase FIR filter based on initial decompile.
- **Result**: `out [f32]: 589824 elements, bit-exact 158805 (26.9241%), max |err| 0.471925, max ulp 24197722`.
- **First Mismatch**: `{'index': 0, 'port': 0.008588007651269436, 'oracle': 0.00830714963376522, 'port_bits': 1007465655, 'oracle_bits': 1007164086}`

### Attempt 2
- **Hypothesis**: Disassembled GFX1201 ISA (`sr.gfx1201.co` at `0x15600`). Found it is actually a 5-tap separable FIR filter:
  - Even phase weights: `[-0.0413513, 0.0872803, 0.6201172, 0.3720703, -0.0382996]` (sum = 1.0)
  - Odd phase weights: `[-0.0382996, 0.3720703, 0.6201172, 0.0872803, -0.0413513]` (sum = 1.0)
  - Samples loaded from LDS: 5 taps `[bx - 2, bx - 1, bx, bx + 1, bx + 2]`.
  - Clamping: clamped to inner 3 samples `[bx - 1, bx, bx + 1]`.
  - Inverse tonemapping on GPU clamps to `[0.0001, 3.0]` and calculates `(exp2(v * (1/0.36) * log2(10)) - 1.0) * (0.01 / expo)`.
- **Result**: `out [f32]: 589824 elements, bit-exact 329882 (55.9289%), max |err| 0.0320093, max ulp 2648755`.
### Attempt 3
- **Hypothesis**: Apply GFX1201 ISA Dekker residual correction for `/ 0.36`, `* 0.01`, and division by `expo`:
  - `t = c * kInv036; rem1 = fma(t, kNeg036, c); t = fma(rem1, kInv036, t);`
  - `p = t * log2_10; e = exp2(p) - 1.0;`
  - `m = e * k001; rem2 = fma(m, kNeg100, e); m = fma(rem2, k001, m);`
  - `q = m * inv_e; rem3 = fma(-q, expo_val, m); res = fma(rem3, inv_e, q);`
- **Result**: `out [f32]: 589824 elements, bit-exact 417865 (70.8457%), max |err| 0.0320091, max ulp 2648755`.
- **First Mismatch**: `{'index': 4, 'port': 0.00971667468547821, 'oracle': 0.009716675616800785, 'port_bits': 1008677552, 'oracle_bits': 1008677553}` (Pixel 0 indices 0..3 are 100% bit-exact; index 4 is 1 ULP off; 87.74% within 1 ULP).

### Attempt 4
- **Hypothesis**: The FIR clamp in ISA clamps against all 5 taps `[s0, s1, s2, s3, s4]` (both horizontally across columns 0..4 and vertically across rows 0..4), not just inner 3 taps. Also use exact bit-exact unpacking constants `c_rg = uintBitsToFloat(0x3a9ce069u)` and `c_b = uintBitsToFloat(0x3b1cf40au)` matching ISA decompilation.
- **Result**: `out [f32]: 589824 elements, bit-exact 442642 (75.0465%), max |err| 0.005625, max ulp 140863`.
  - Within 1 ULP: 551431 / 589824 (93.49%)
  - CLOSE (<= 4 ULP or <= 1e-3 rel): 563983 / 589824 (95.62%)
  - PSNR vs oracle: 75.37 dB (MSE 2.9057e-08)
- **Pipestats**: 108 SGPRs, 96 VGPRs, 0 spilled SGPRs, 0 spilled VGPRs, 0 scratch size.
