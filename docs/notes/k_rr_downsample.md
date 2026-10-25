# Notes: k_rr_downsample (`rrio::k_downsample<0>`)

## Architecture & Recovery
- Kernel symbol: `_ZN4rrio12k_downsampleILi0EEEvPKtS2_iiPKfP15HIP_vector_typeIfLj4EEii`
- Launch 22 in `rr_F` (512x288 output from 768x432 1.5x resolution)
- 5-tap separable FIR filter decoded from `rr.gfx1201.co` at `0x9f8a`:
  - Even polyphase filter: `[-0.0413513, 0.0872803, 0.6201172, 0.3720703, -0.0382996]` (hex: `0xa94b, 0x2d96, 0x38f6, 0x35f4, 0xa8e7`)
  - Odd polyphase filter: `[-0.0382996, 0.3720703, 0.6201172, 0.0872803, -0.0413513]`
  - Inner 3-tap clamp: `min/max(s1, s2, s3)` on horizontal and vertical passes
  - Tonemap inversion: `(exp2(v * (1.0 / 0.36) * log2(10)) - 1.0) * (0.01 / expo)` with `v` clamped to `[0.0001, 3.0]`
  - Output is RGBA32F (`float4`), with alpha = 1.0

## Attempt Log
### Attempt 1: 5-tap polyphase separable FIR with inner-3 clamp and GPU tonemap inversion
- Implementation: `shaders/rr/rr_downsample.comp`
- Result on Launch 22:
  - Elements: 589824
  - Bit-exact: 300426 (50.9349%)
  - Max |err|: 0.0289028
  - Max ULP: 1492358
  - First mismatch: `index 1, port: 0.007601836696267128, oracle: 0.007601837161928415, port_bits: 1006180564, oracle_bits: 1006180565` (1 ULP difference!)
  - Status: APPROX (50.93% bit-exact, 1 ULP on first mismatch)
