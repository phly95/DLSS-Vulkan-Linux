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
- **Disposition**: `k_ae_block` EXACT for the measured reset and steady-state launches. `k_ae_final` is measured separately below and remains APPROX until non-reset adaptation is ported.

## Recovered launch ABI

The frame-0 launch table confirms launch 1 writes `E0c26`, `E0c2a`, and `E0cee`; launch 2 (`k_ae_final`, grid 1x1x1) reads/writes the same pointers and additionally writes `E0c22`. The actual block launch passes the source image at pointer offset 0 and the accumulator at offset 16, matching the ISA pointer loads. The output sizes are 2 bytes for each of `E0c26` and `E0c2a`, and 8 bytes for `E0cee`.

The block ISA performs a global 64-bit atomic add into `E0cee`, after scaling the reduced luma sum by an immediate `0x48d1b717` and converting with round-to-nearest-even. The shader now models that fixed-point accumulator. `E0c26`/`E0c2a` are separate single-half states and remain unchanged by this kernel; the reset host path seeds them to 1.0 before dispatch.

## Host argument map (from `tools/rr/rrref/rrref.cpp`)

The shared `AEArgs` fields are: `+0x00` source `E0c6e`; `+0x08/+0x0c` W/H; `+0x10` accumulator `E0cee`; `+0x18` scale `2.3283064365386963e-6 / (W*H)`; `+0x1c` reset; `+0x20/+0x28` parity-selected AE half-state pointers; `+0x30` exposure `E0c22`. Launch 1 and launch 2 use this same packed structure.

The final-kernel ISA loads the accumulator as two u32 values and constructs a double-precision sum (`low + high * 2^32`), multiplies by the supplied scale, clamps the mean to at least `0.005`, then forms exposure using the immediate `0x3e711f48` (0.235470891f). It also reads both half AE-state values and conditionally updates them. The shader now implements the decoded reset path and clears the accumulator; non-reset temporal adaptation remains unresolved.

### `k_ae_final` attempt 1: reset launch
- **Hypothesis**: on reset, output exposure is `0.235470891 / max(float(u64_accum) * scale, 0.005)`, floored to f16; `E0c22` is `[new_half, previous_half, 1/new_half, 0]`, the selected next AE half is written, and the accumulator is cleared.
- **Test**: frame 0 launch 2 with exact prelaunch `E0cee`, `E0c2a`, and launch arguments from `diffdump.py inputs`; bind outputs for `E0c22`, `E0c26`, `E0cee`.
- **Result**: `DIFFDUMP: BIT-EXACT`. E0c22 [f32] 4/4 (100%), E0c26 [f16] 1/1 (100%), E0cee [u32] 2/2 (100%); all max absolute error 0 and max ULP 0.
- **Disposition**: reset branch is exact for this launch. The shader remains APPROX for the family because non-reset adaptation is not implemented or measured.

### `k_ae_final` attempt 2: use target exposure directly on a steady-state frame
- **Hypothesis**: the reset target computation might also be sufficient on a non-reset frame, retaining the previous AE value only as `E0c22.y`.
- **Test**: frame 1 launch 3 with exact prelaunch `E0cee`, previous AE state `E0c26`, and launch arguments. Outputs bound to E0c22, E0c2a, and E0cee.
- **Result**: `DIFFDUMP: DIFFERS`. E0c22 [f32] 3/4 exact (75%), max absolute error 0.00151491, max ULP 986091534; first mismatch index 3, port 0.0 vs oracle -0.0015149132814258337. E0c2a [f16] 0/1 exact, max error 0.000976562, max ULP 1; first mismatch port 1.390625 vs oracle 1.3916015625. E0cee [u32] 2/2 exact, max error 0, max ULP 0.
- **Disposition**: rejected. NeuralHub's non-reset temporal adaptation updates both the next AE half and the exposure delta; direct target exposure is insufficient. The family stays APPROX pending that adaptation decode.

### `k_ae_final` attempt 3: seven-eighths state update
- **Hypothesis**: the steady-state AE state takes a fixed 7/8 step from the previous f16 value toward the floored target exposure.
- **Test**: frame 1 launch 3, same exact prelaunch accumulator and AE state; added the reset flag from push offset 28 and compared `E0c22`, `E0c2a`, and `E0cee`.
- **Result**: the next AE state `E0c2a [f16]` became bit-exact (1/1). The output mapping also confirms `E0c22.x` is the target exposure, not the smoothed state. Latest diffdump: `E0c22 [f32]` 3/4 exact (75%), max |err| 0.00398055, max ULP 15,565,629, first mismatch index 3 (port -0.005495463963598013, oracle -0.0015149132814258337); `E0c2a [f16]` 1/1 exact, max |err| 0, max ULP 0; `E0cee [u32]` 2/2 exact, max |err| 0, max ULP 0. `DIFFDUMP: DIFFERS`.
- **Reset regression check**: frame 0 launch 2 with the reset flag and parity-selected `E0c26` output remained `DIFFDUMP: BIT-EXACT`: E0c22 [f32] 4/4, E0c26 [f16] 1/1, E0cee [u32] 2/2; all max errors and ULPs 0.
- **Disposition**: improved the state update for this one steady-state launch, but the output delta formula remains wrong and this fixed-step hypothesis is not yet validated on other frames. Family remains APPROX.

### `k_ae_final` attempt 4: E0c22[3] delta formula investigation
- **Hypothesis**: The delta in E0c22[3] might use a different smoothing factor or formula than `(updated - prev) / prev` with a 7/8 step.
- **Analysis**: Oracle delta = -0.001514913, port = -0.005495464. The implied step factor from the oracle is ~0.2413 × (target - prev), NOT 0.875 × (target - prev). This is not a clean fraction (not 1/4, 7/29, etc.), suggesting either: (a) the temporal adaptation uses an IIR with frame-dependent alpha, (b) the delta formula is computed differently from what we assumed, or (c) E0c22[3] has a completely different meaning than `(updated - prev) / prev`.
  Key observation: `oracle_delta * prev / (target - prev) = 0.24121`, suggesting the effective adaptation rate on this particular frame is ~0.24, much slower than our assumed 0.875. The ISA's non-reset temporal adaptation may use a multi-step or history-dependent smoothing algorithm.
- **Result**: No shader change. Analysis only. E0c2a remains exact (7/8 step is correct for the next AE half-state), but the exposure delta E0c22[3] remains wrong.
- **Status**: APPROX; needs deeper ISA decode of the non-reset path. The E0c22[0,1,2] and E0c2a are all exact; only E0c22[3] differs.
