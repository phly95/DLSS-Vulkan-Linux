# Handoff: state of the Linux/Proton port of NeuralHub (DLSS NR / SR / RR on RDNA4)

Read this first. It says what exists, how to reproduce it, what blocks the rest, and what must not
be retried.

## The goal (as given)

Port NeuralHub v0.8.0 (Windows: HIP kernels + D3D12 hooks) to Linux/Proton on SteamOS for the
RX 9070 XT (RADV GFX1201, Mesa 26.2+, FP8 cooperative matrices): NR, SR (DLSS presets M/L) and
RR (preset F) as Vulkan compute shaders, no HIP at run time, numerically matching the Windows build.
Seven deliverables in order: inventory, harness, NR, SR, RR, integration + installer, performance.
Constraint: the only Windows material is `NeuralHub_v0.8.0_setup.exe` (Mesa and other open tools
are allowed). The reference Vulkan NR port is `reference-nr-vulkan/` (MIT).

## Status

| # | deliverable | state |
| --- | --- | --- |
| 1 | kernel + interop inventory | **done**: `docs/inventory/KERNELS.md`, `docs/inventory/kernels-generated.md` |
| 2 | harness | **partial**: dump format + `tools/harness/compare.py` (self-tested). **No Windows reference dumps** (need Windows + RX 9000). No Vulkan-side dumper yet. |
| 3 | NR | **runs on the RX 9070 XT with the real 310.8.0 model**: 45.59 / 47.97 / 49.00 dB vs NVIDIA's output at 1080p / 1440p / 4K (offline runner, `tools/hwtest/nr_gate.py`), the same as the reference's published figures. ≥ 49 dB at every size needs the 310.9.16 model, and the in-game runtime path is not measured yet. See `docs/hardware/RX9070XT-STEAMOS.md`. |
| 4 | SR | **host recovered, reference running**: NeuralHub's own SR code runs on the RX 9070 XT under ROCm (`tools/sr/srref`, presets M and L, 39 dB after 8 frames on the test sequence; `docs/sr/HOST-RE.md`). Weights from the user's `nvngx_dlss.dll` 310.9.1.0 packed exactly as NeuralHub does (`tools/sr/sr_weights.py`, `sr_pack.py`). **Vulkan shaders: not started** (next: `k_sr_stage`, ISA notes below). |
| 5 | RR | not started |
| 6 | integration + installer | not started (the reference's `linux/package/install.sh` is the model to follow) |
| 7 | performance | not started (needs the card) |

## The real machine

The port now also runs on the target: an RX 9070 XT under SteamOS (`docs/hardware/RX9070XT-STEAMOS.md`,
`tools/hwtest/`). Build outputs go to `build/hwtest/` and the model to `build/model/`. Both are ignored by git.
This machine has the user's `nvngx_dlss.dll` / `nvngx_dlssd.dll` 310.9.1.0, which NeuralHub's SR / RR target
(Crimson Desert's `bin64/`).

## The cloud container (first sessions) and what it can do

- No GPU, no Windows. Ubuntu 24.04, 4 cores, 15 GB RAM.
- **Compile for the real card without it**: `bash tools/radv-offline/build.sh` (unpacks the prebuilt
  Mesa in `artifacts/mesa/`; `MESA_FROM_SOURCE=1` rebuilds Mesa main e2ab668 + libdrm b97cbde), then
  `tools/radv-offline/run.sh tools/radv-offline/pipestats [--isa DIR] file.spv ...`. RADV reports
  "AMD Radeon RX 9070 XT (RADV GFX1201)" through Mesa's amdgpu drm-shim; pipelines compile with ACO
  exactly as on the card; nothing executes.
- **CPU Vulkan**: the same Mesa build has lavapipe with `VK_KHR_cooperative_matrix` (f16×f16→f32,
  int8) but **no FP8** (`VK_EXT_shader_float8` is RADV GFX11.7+ only). The reference's FP8 network
  cannot run on it. E4M3 values are exact in f16, so an f16-widened shader variant could run there
  for functional tests; not written yet.
- Reference build: `cd reference-nr-vulkan && bash fetch_deps.sh && python3 linux/build/build_network.py rdna4`
  (pinned glslang 16.5.0, needed for byte-identical SPIR-V).

## Artifacts (private repository; binaries are intentional)

| path | what |
| --- | --- |
| `artifacts/neuralhub-v0.8.0/NeuralHub_v0.8.0_setup.exe` | the only Windows input (sha256 `aae53654…0b834`) |
| `artifacts/neuralhub-v0.8.0/gfx1201/{nr,sr,rr,fg}.gfx1201.co` | the RX 9070 XT code objects out of `neuralhub_host.dll`'s `.hip_fat` |
| `artifacts/neuralhub-v0.8.0/gfx1201/*.s.xz` | their `llvm-objdump --mcpu=gfx1201` disassembly |
| `artifacts/neuralhub-v0.8.0/gfx1201/kernels.json` | per-kernel metadata, kernarg reads, instruction profile (`tools/exe/kernels.py`) |
| `artifacts/neuralhub-v0.8.0/host_strings.txt.xz` | strings of `neuralhub_host.dll` (log messages, ini/env keys, source expressions of its error-check macro) |
| `artifacts/isa/amdgpu_isa_rdna4.xml.xz` | AMD's machine-readable RDNA4 ISA (GPUOpen): encodings + prose, no pseudocode, no WMMA lane map |
| `artifacts/isa/radv_nir_lower_cooperative_matrix.c` | Mesa source with the GFX12 WMMA lane layout comment |
| `artifacts/mesa/*.tar.xz` | prebuilt Mesa install (RADV, lavapipe, drm-shim) used by `build.sh` |
| `artifacts/reference-nr/` | ACO ISA + statistics of all 64 reference NR pipelines on GFX1201 |

`neuralhub.exe` (167 MB) and `neuralhub_host.dll` (98 MB) are not committed (GitHub's 100 MB limit);
`python3 -I tools/exe/extract.py artifacts/neuralhub-v0.8.0/NeuralHub_v0.8.0_setup.exe out/` recreates
them and every code object in seconds (needs `7z`, `pip install pefile msgpack`).

## Key findings (details in docs/inventory/KERNELS.md)

- NeuralHub's D3D12 backend runs the HIP gfx ISA through AMD's `amdxc64.dll` ElfHsa extension. Proton
  has no equivalent, so every kernel must become SPIR-V.
- SR weights: user's `nvngx_dlss.dll` 310.9.1.0, presets M/L only; RR: `nvngx_dlssd.dll` 310.9.1.0
  preset F; NR: `nvngx_dlssnr.dll` 310.8.0.0. None are available here; none may be committed.
- NeuralHub has built-in dumps usable on Windows: env `DLSSNR_DUMP_LAYERS`, `DLSSNR_DUMP_NOISE`, and
  `[Debug] DlssSrDumpNextFrame=1` (writes `sr_dump_<n>/stage_in_color.bin`, `stage_in_depth.bin`,
  `after_hidden0/1.bin`, `stage_output.bin`, `game_output.bin`; HIP backend only). Their file format is
  not yet known. RR has no dump.
- FP8 is E4M3, RNE, non-saturating (`v_cvt_pk_fp8_f32` without clamp); WMMA is E4M3×E4M3→F32.

## Do not retry

- **Do not build a CPU emulator that executes NeuralHub's code objects.** That was attempted in the
  first session and was stopped by a safety check; the partial work was deleted. Do not resume it in
  another form.
- (Resolved 2026-10-24) The user chose to port NeuralHub's SR/RR kernels to Vulkan; that is no longer blocked.

## Open questions for the user

1. (Answered 2026-10-24: port NeuralHub's kernels.)
2. (Answered: this machine is the RX 9070 XT. Windows dumps are no longer needed: NeuralHub's kernels run under ROCm here.)
3. NR at 310.9.16 (the 49 dB figure at 1080p) needs `nvngx_dlssnr.dll` 310.9.16.

## SR: how to continue (2026-10-24)

1. `python3 -I tools/exe/extract.py artifacts/neuralhub-v0.8.0/NeuralHub_v0.8.0_setup.exe build/nh` (code objects, host DLL).
2. Weights: `build/venv/bin/python -I tools/sr/sr_weights.py <Crimson Desert>/bin64/nvngx_dlss.dll --out build/re/weights`,
   then `tools/sr/sr_pack.py build/re/weights {M,L} build/re/packed/{M,L}`.
3. Reference: see `docs/sr/HOST-RE.md` (ROCm 7.2.4 in podman; `srref ... --dump` writes every stage tensor and the
   dec0 outputs). Every Vulkan kernel is to be checked against it, stage by stage.
4. Port order: `k_sr_stage<64,2,0,96,2,false>` (encoder), `k_sr_tube_split<128,4,false>`, the decoder stages,
   `k_sr_enc0`, `k_sr_dec0`, `k_sr_post<false,12,16,false,false>`, `k_sr_downsample`, exposure; then preset L's
   `k_sr_enc0L_in` / `k_sr_l0`. On this card every stage is "fp8-safe", so only the `false` instantiations run.
   What is known of `k_sr_stage` so far: activations are `[C/32][H][W][32]` fp8; a workgroup (64 threads)
   covers an 8x8 pixel tile at grid*8 - shift, mirror-padded at the borders; RMSNorm = f16 sum of squares
   (fixed packed-add tree + a lane^16 exchange) + 2^-14, rsqrt from `g_sr_lut`, times the sqrt(C)-scaled gain,
   to fp8; GEMMs as pairs of K=32 WMMA chains whose f32 results are added through f16 (`v_fma_mixlo_f16`).
   No `exp` anywhere: the "attention" is not softmax.
5. Faster route to games, if wanted: NeuralHub's kernels run under ROCm, so a Vulkan layer could drive them
   through HIP external-memory interop. This breaks the goal's "no HIP at run time" and needs ROCm
   libraries; ask the user before going this way.

## Next steps once unblocked

1. On Windows + RX 9000: run NeuralHub with its dump switches on seeded inputs; work out the dump format;
   convert to `.hdump` (`tools/harness/hdump.py`).
2. On an RX 9070 XT: run the reference NR against NVIDIA's output at 1080p/1440p/4K (gate ≥ 49 dB).
3. SR, then RR, one kernel per commit, each checked by `compare.py` against the dumps and by
   `pipestats` for spills on GFX1201 before it reaches hardware.
