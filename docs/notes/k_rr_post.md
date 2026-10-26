# Notes: `k_post` (RR)

## Launch 21 baseline

- **Oracle launch**: `_ZN4rrio6k_postILi24EEEvNS_8PostArgsE`, grid [48, 14, 1], frame 0.
- **Test**: ran the existing `rr_post` SPIR-V fixture output through `tools/verify/diffdump.py compare` against exact prelaunch inputs / NeuralHub dump. The first comparison incorrectly specified `E0cf6` as f16, then f32. The actual output is 1,327,104 bytes for 165,888 pixels (8 bytes/pixel), so f16 is the appropriate element interpretation; re-ran with corrected launch-21 dimensions and dtypes.
- **Corrected result**: `E0c0a` [f16], 1,327,104 elements, 391,013 bit-exact (29.4636%), max absolute error 0.467529, max ULP 1969, first mismatch index 0 (0.1166381836 vs 0.1152954102). `E0c0e` is missing. `E0cf6` [f16], 663,552 elements, 40,247 bit-exact (6.0654%), max absolute error 0.467529, max ULP 1,948, first mismatch index 0 (0.1166381836 vs 0.1152954102). `E0cfa` [f16], 331,776 elements, 18,989 bit-exact (5.7234%), max absolute error 0.439453, max ULP 1969, first mismatch index 0 (0.0720214844 vs 0.0714721680). Overall: `DIFFDUMP: DIFFERS`.
- **Disposition**: APPROX. The launch writes four buffers; current fixture only models three bindings and does not produce `E0c0e`. All three compared output arrays now use their byte-appropriate half interpretation. The parity-selected history output still needs validation against its true launch slot before tuning spatial filtering.

## Host argument map

`tools/rr/rrref/rrref.cpp` constructs the 0xb8-byte `PostArgs` used for launch 21. It passes input taps at 0x40, current history at 0x70, previous/output history at 0x88, result planes `E0cf6`/`E0cfa` at 0x90/0x98, `E0c0e` at 0xa0, and the debug probe at 0xa8. `E0c0e` is therefore a real output argument and its absence from the fixture is a binding omission. The parity-selected history pointer also differs from the current fixture's simple `OutFull` write. Do not tune the filter using the existing direct fixture until it models these pointer slots and the correct formats.

## Missing `E0c0e` output probe

The frame-0 launch metadata confirms launch 21 writes four outputs: `E0c0a`, `E0c0e`, `E0cf6`, and `E0cfa`. The `E0c0e` pointer is loaded into `s[22:23]` from `PostArgs+0xa0`; the epilogue forms its value at `0xA08F8..0xA0938` and writes a 32-bit word through that pointer. It shifts one converted value by 20 bits, combines another at bit 10, then ORs the low 10-bit value and `0xc0000000`. Thus `E0c0e` is a packed 10:10:10:2 word with the top two bits set, not a half plane. The first oracle words are `0xc1f0c032`, `0xc240e040`, `0xc240dc41`, `0xc1e0b832`; unpacking gives channels `[50,48,31,3]`, `[64,56,36,3]`, `[65,55,36,3]`, `[50,46,30,3]`. The packed channels are built from rounded f64 values after additional transforms; the shader does not yet reproduce those transforms or emit this output. See the epilogue before adding the missing binding; do not substitute filtered RGB directly.

## Attempt 1: packed color synthesized from current filtered RGB

- **Hypothesis**: the packed output stores the shader's filtered RGB after `log2(1 + 100*c) * log10(2) * 0.36 * 0.408`, rounded to 10-bit UNORM, with A2=3; the one output is written for each even/even pixel into a quarter-resolution plane.
- **Implementation**: added binding 5 (`OutPacked`) to `rr_post.comp`, wired a quarter-resolution u32 buffer through `vk_runner`, and added the same output to the O7 `rr_post` fixture.
- **Test**: RX 9070 XT, frame 0 launch 21, oracle inputs and launch args, output compared as u32.
- **Result**: `E0c0e [u32]` 82,944 elements, bit-exact 0 (0%), max absolute error 193,128,634, first mismatch index 0 (port `0xc89290a6`, oracle `0xc1f0c032`; unpacked port `[166,164,137,3]`, oracle `[50,48,31,3]`). `DIFFDUMP: DIFFERS`. This rejects sourcing the packed plane from the existing filtered RGB values. The same run's color/history results were E0c0a [f16] 391,012/1,327,104 exact (29.4636%), max |err| 0.467529, max ULP 1969; E0cf6 [f16] 40,248/663,552 exact (6.0655%), max |err| 0.467529, max ULP 1948; E0cfa [f16] 18,987/331,776 exact (5.7228%), max |err| 0.439453, max ULP 1969. All differ.
- **Status**: APPROX. The output is now structurally present end-to-end, but its channels are derived from other inputs/transforms than current `c_filt`.
