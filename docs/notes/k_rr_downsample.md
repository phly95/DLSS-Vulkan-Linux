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
### Attempt 2: 5-tap clamp and Dekker residual corrections
- **Hypothesis**:
  - Clamp in GFX1201 ISA (`_ZN4rrio12k_downsampleILi0EEEv...` lines 88782-88799) uses all 5 taps `[s0, s1, s2, s3, s4]`, not just inner 3 taps.
  - Apply Dekker residual steps for `/ 0.36`, `* 0.01`, and `/ expo`.
- **Implementation**: `shaders/rr/rr_downsample.comp`
- **Result on Launch 22**:
  - Elements: 589824
  - Bit-exact: 350173 (59.3691%) (up from 50.93%)
  - Max |err|: 0.00438631 (down 7x from 0.0289028)
  - Max ULP: 110199 (down 14x from 1492358)
  - First mismatch: `index 2, port: 0.004225417971611023, oracle: 0.0042254189029335976, port_bits: 998929760, oracle_bits: 998929762` (c=0, c=1, c=3 at pixel 0 are 100% bit-exact; c=2 is off by 2 ULPs; 94.76% elements satisfy CLOSE <= 4 ULP or <= 1e-3 rel)
  - Status: APPROX (59.37% bit-exact, 94.76% CLOSE)
