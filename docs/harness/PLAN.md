# Deliverable 2: the acceptance harness

## Constraint that shapes it

The goal's ground truth is the Windows build. The machine this port is written on has no GPU and no
Windows, and the only Windows material allowed is `NeuralHub_v0.8.0_setup.exe`. Reference numbers
therefore come only from running NeuralHub on Windows + an RX 9000 card: its own dump switches
(`DLSSNR_DUMP_LAYERS`, `DLSSNR_DUMP_NOISE`, `[Debug] DlssSrDumpNextFrame=1`), converted to `.hdump`.
(An emulator that executes NeuralHub's code objects was tried and stopped; see HANDOFF.md.)

The Vulkan side is dumped in the same format by the port's runtime (not written yet), so one
comparison tool serves both.

## Tensor dump format (`.hdump` directory)

```
manifest.json   {"source": "windows|vulkan", "frame": n, "layers": [
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

- compare.py on synthetic data with known injected error: reproduces the injected MSE to 1e-6 rel.;
- NR, reference-nr-vulkan vs NVIDIA (from the reference's own documentation, not re-measured here):
  49.15 / 49.40 / 50.59 dB at 1080p / 1440p / 4K. Re-measuring needs an RX 9070 XT.
