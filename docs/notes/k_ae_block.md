# Notes: `k_ae_block` / `k_ae_final`

## Oracle launch probe

### Attempt 1: map launch 1 to the current partial-luma shader
- **Hypothesis**: NeuralHub launch 1's `E0cee` output corresponds to the shader's one-float-per-16x16-workgroup `block_luma` buffer; use launch dimensions 256x144 and the exact `E0c6e` input captured by `diffdump.py`.
- **Result**: the mapping is wrong. `diffdump.py compare` reports `E0cee` as a size mismatch (port 144 f32 elements, oracle 2 elements) and reports `E0c26` and `E0c2a` missing. No element-wise first mismatch exists for the size-mismatched `E0cee` buffer.
- **Disposition**: status is APPROX. Decode the launch argument/pointer semantics and all three output buffers before another shader attempt; do not treat the prior O7 fixture or historical exact report as launch fidelity.
