# Deliverable 2: the acceptance harness

## Constraint that shapes it

The goal's ground truth is the Windows build. The machine this port is written on has no GPU and no
Windows, and the only Windows material allowed is `NeuralHub_v0.8.0_setup.exe`. So the harness has
two sources of "Windows numbers", and every result says which one it used:

| source | where it runs | what it is |
| --- | --- | --- |
| **capture** (`hiprec`) | Windows + RX 9000, NeuralHub installed | a proxy `amdhip64_7.dll` that records every kernel launch of the real build: kernel, grid, kernarg bytes, every device allocation the kernarg points into, before and after. Ground truth in the strict sense. |
| **emulation** (`rdna4emu`) | anywhere (this machine) | executes NeuralHub's own gfx1201 code objects (the bytes the Windows build runs on an RX 9070 XT) on a CPU model of the RDNA4 instructions they use. Ground truth up to the emulator's correctness, which is itself checked (below). |

The Vulkan side is dumped in the same format by the port's runtime (`NRVK_DUMP=dir`), so one
comparison tool serves both.

## Tensor dump format (`.hdump` directory)

```
manifest.json   {"source": "capture|emulation|vulkan", "frame": n, "layers": [
                   {"index": i, "kernel": "<demangled name or pipeline>", "grid": [x,y,z], "block": [x,y,z],
                    "tensors": [{"name": "...", "role": "in|out", "dtype": "e4m3|f16|f32|u8|u16|u32",
                                 "shape": [...], "file": "L0003_out0.bin"}]}]}
L0003_out0.bin  raw little-endian elements
```

Layers are matched by `index` + `kernel` between two dumps, or through a mapping file when one HIP
kernel became several Vulkan pipelines (or several kernels one pipeline).

## Metrics (tools/harness/compare.py)

Per tensor:

- max abs error, mean abs error, and PSNR = 10·log10(peak² / MSE) with peak = the reference tensor's max
  |value| (stated in the report; images use 1.0 after the network's own encode);
- for e4m3 tensors, the e4m3 ulp distance histogram (0, 1, 2, >2 ulps) and NaN mismatches;
- **fp8 tolerance**: a layer passes when every element is within 1 e4m3 ulp of the reference and at
  least 99% are exact. (The reference port's own measure: its first kernel reproduced NVIDIA's at
  98.62% of bytes, the rest one ulp off.) A layer outside it is a divergence and fails the gate.

Final image: PSNR and max abs error over RGB in [0, 1]; gate 45 dB for SR / RR, 49 dB for NR.

## Expected numbers before any port work

- emulator vs emulator: identical (determinism check);
- compare.py on synthetic data with known injected error: reproduces the injected MSE to 1e-6 rel.;
- NR, reference-nr-vulkan vs NVIDIA (from the reference's own documentation, not re-measured here):
  49.15 / 49.40 / 50.59 dB at 1080p / 1440p / 4K. Re-measuring needs an RX 9070 XT.

## Checking the emulator

There is no hardware here to check it against, so it is checked against facts that do not need a GPU:

1. instruction semantics unit tests (IEEE f32/f16 arithmetic against the host CPU, E4M3 conversions
   against the OCP spec's table, WMMA against a double-precision matmul with the fragment layout from
   AMD's RDNA4 ISA guide);
2. kernels whose function is fixed by their name and arguments (`k_sr_exposure`, `k_sr_ae_copy`,
   `k_w16_decode`, `k_mean`, …) must give the closed-form result;
3. once a Windows capture exists, emulation of the captured launch must reproduce the captured output
   bit for bit. Until then, emulator results are labelled "emulation".
