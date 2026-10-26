# Notes: `k_post` (RR)

## Launch 21 baseline

- **Oracle launch**: `_ZN4rrio6k_postILi24EEEvNS_8PostArgsE`, grid [48, 14, 1], frame 0.
- **Test**: ran the existing `rr_post` SPIR-V fixture output through `tools/verify/diffdump.py compare` against exact prelaunch inputs / NeuralHub dump.
- **Result**: `E0c0a` [u8], 2,654,208 elements, 1,591,274 bit-exact (59.9529%), max absolute/ULP error 255, first mismatch index 0 (119 vs 97). `E0c0e` is missing. `E0cf6` [f16], 663,552 elements, 40,247 bit-exact (6.0654%), max absolute error 0.467529, max ULP 1948, first mismatch index 0 (0.1166381836 vs 0.1152954102). `E0cfa` [f16], 331,776 elements, 18,989 bit-exact (5.7234%), max absolute error 0.439453, max ULP 1969, first mismatch index 0 (0.0720214844 vs 0.0714721680). Overall: `DIFFDUMP: DIFFERS`.
- **Disposition**: APPROX. The launch writes four buffers; current fixture only models three bindings and does not produce `E0c0e`. The parity-selected history output also needs to be validated against its true launch slot before tuning spatial filtering.

## Host argument map

`tools/rr/rrref/rrref.cpp` constructs the 0xb8-byte `PostArgs` used for launch 21. It passes input taps at 0x40, current history at 0x70, previous/output history at 0x88, result planes `E0cf6`/`E0cfa` at 0x90/0x98, `E0c0e` at 0xa0, and the debug probe at 0xa8. `E0c0e` is therefore a real output argument and its absence from the fixture is a binding omission. The parity-selected history pointer also differs from the current fixture's simple `OutFull` write. Do not tune the filter using the existing direct fixture until it models these pointer slots and the correct formats.
