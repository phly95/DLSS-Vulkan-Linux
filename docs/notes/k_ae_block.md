# Notes: `k_ae_block` / `k_ae_final`

## Oracle launch probe

### Attempt 1: map launch 1 to the current partial-luma shader
- **Hypothesis**: NeuralHub launch 1's `E0cee` output corresponds to the shader's one-float-per-16x16-workgroup `block_luma` buffer; use launch dimensions 256x144 and the exact `E0c6e` input captured by `diffdump.py`.
- **Result**: the mapping is wrong. `diffdump.py compare` reports `E0cee` as a size mismatch (port 144 f32 elements, oracle 2 elements) and reports `E0c26` and `E0c2a` missing. No element-wise first mismatch exists for the size-mismatched `E0cee` buffer.
- **Disposition**: status is APPROX. Decode the launch argument/pointer semantics and all three output buffers before another shader attempt; do not treat the prior O7 fixture or historical exact report as launch fidelity.

## Recovered launch ABI (no new shader attempt)

The frame-0 launch table confirms launch 1 writes `E0c26`, `E0c2a`, and `E0cee`; launch 2 (`k_ae_final`, grid 1x1x1) reads/writes the same pointers and additionally writes `E0c22`. The actual block launch passes the source image at pointer offset 0 and the accumulator at offset 16, matching the ISA pointer loads. The output sizes are 2 bytes for each of `E0c26` and `E0c2a`, and 8 bytes for `E0cee`.

The block ISA performs a global 64-bit atomic add into `E0cee`, after scaling the reduced luma sum by an immediate `0x48d1b717` and converting with round-to-nearest-even. Thus the next testable hypothesis is a packed two-lane fixed-point accumulator, not an array of per-workgroup floats. `E0c26`/`E0c2a` are separate single-half outputs and still need their computation decoded. This is characterization only; no parity claim is made for these outputs.
