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
