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
- **First Mismatch**: `{'index': 0, 'port': 0.008307147771120071, 'oracle': 0.00830714963376522, 'port_bits': 1007164084, 'oracle_bits': 1007164086}` (only 2 ULPs off on index 0!).
