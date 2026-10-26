# Notes: `k_ae_block` / `k_ae_final`

## Oracle launch probe

### Attempt 1: float partial output (superseded)
- **Hypothesis**: launch 1's `E0cee` output corresponds to one f32 partial per 16x16 workgroup.
- **Result**: incorrect. `E0cee` was 144 f32 elements vs oracle 2 u32 elements; `E0c26` and `E0c2a` were not represented. This exposed a mapping failure, not an element-wise mismatch.
- **Disposition**: rejected after recovering the launch ABI and ISA.

### Attempt 2: ISA-derived 32.32 atomic accumulator
- **Hypothesis**: each workgroup's clamped luma sum is multiplied by `0x48d1b717`, rounded RNE, and atomically accumulated as a u64 fixed-point value. Decode luma as `0.25 R + 0.5 G + 0.25 B`; preserve AE state buffers unchanged.
- **Implementation**: `shaders/rr/rr_ae_block.comp`; enable `VK_KHR_shader_atomic_int64` / `shaderBufferInt64Atomics` in `kernel_run` on devices that expose the extension.
- **First run**: `E0cee` matched 2/2 u32 words exactly. `E0c26` and `E0c2a` were both port 0.0 vs oracle 1.0 because the raw prelaunch dump predates the reset host writes.
- **Host-precondition check**: `tools/rr/rrref/rrref.cpp` writes 1.0 to both AE states when reset is true before dispatch. Seeding those two exact host writes in the isolated launch fixture, with all other inputs from `diffdump.py inputs`, produced `DIFFDUMP: BIT-EXACT`: E0c26 1/1 exact, E0c2a 1/1 exact, E0cee 2/2 exact; every buffer max absolute error 0 and max ULP 0.
- **Steady-state check**: on frame 1, launch 2 is `k_ae_block` (launch 1 is `k_pooled2` because reset is false). Using exact prelaunch inputs without modifying either half-state produced `DIFFDUMP: BIT-EXACT`: E0cee 2/2 u32 words, 100%, max absolute error 0, max ULP 0. E0c26/E0c2a were unchanged and therefore not compared by diffdump.
- **Disposition**: `k_ae_block` EXACT for this measured launch/precondition. This does not establish `k_ae_final`; it remains APPROX.

## Recovered launch ABI

The frame-0 launch table confirms launch 1 writes `E0c26`, `E0c2a`, and `E0cee`; launch 2 (`k_ae_final`, grid 1x1x1) reads/writes the same pointers and additionally writes `E0c22`. The actual block launch passes the source image at pointer offset 0 and the accumulator at offset 16, matching the ISA pointer loads. The output sizes are 2 bytes for each of `E0c26` and `E0c2a`, and 8 bytes for `E0cee`.

The block ISA performs a global 64-bit atomic add into `E0cee`, after scaling the reduced luma sum by an immediate `0x48d1b717` and converting with round-to-nearest-even. The shader now models that fixed-point accumulator. `E0c26`/`E0c2a` are separate single-half states and remain unchanged by this kernel; the reset host path seeds them to 1.0 before dispatch.

## Host argument map (from `tools/rr/rrref/rrref.cpp`)

The shared `AEArgs` fields are: `+0x00` source `E0c6e`; `+0x08/+0x0c` W/H; `+0x10` accumulator `E0cee`; `+0x18` scale `2.3283064365386963e-6 / (W*H)`; `+0x1c` reset; `+0x20/+0x28` parity-selected AE half-state pointers; `+0x30` exposure `E0c22`. Launch 1 and launch 2 use this same packed structure.

The final-kernel ISA loads the accumulator as two u32 values and constructs a double-precision sum (`low + high * 2^32`), multiplies by the supplied scale, clamps the mean to at least `0.005`, then forms exposure using the immediate `0x3e711f48` (0.235470891f). It also reads both half AE-state values and conditionally updates them. The current `rr_ae_final.comp` instead treats the accumulator as a float array and sums `W*H` partials, so its launch model must be replaced before numerical tuning. This is ABI/algorithm characterization only; no new shader attempt or diffdump result.
