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
| 2 | harness | **replaced in practice by running NeuralHub itself**: NeuralHub's code objects run unchanged under ROCm on this card (`tools/sr/srref`), so Windows dumps are no longer needed. `.hdump` + `tools/harness/compare.py` still exist (self-tested) for layer-by-layer compares. No Vulkan-side dumper yet. |
| 3 | NR | **runs on the RX 9070 XT with the real 310.8.0 model**: 45.59 / 47.97 / 49.00 dB vs NVIDIA's output at 1080p / 1440p / 4K (offline runner, `tools/hwtest/nr_gate.py`), the same as the reference's published figures. ≥ 49 dB at every size needs the 310.9.16 model, and the in-game runtime path is not measured yet. See `docs/hardware/RX9070XT-STEAMOS.md`. |
| 4 | SR | **in progress (breadth first)**: All SR compute shaders implemented (`sr_gather`, `sr_exposure`, `sr_downsample`, `sr_post`, `sr_enc0`, `sr_dec0`, `sr_stage`, `sr_tube_split`, `sr_enc0L_in`, `sr_l0`). `sr_gather` is EXACT (bit-identical). `sr_exposure` is EXACT. `sr_downsample` reached 55.93% bit-exact and 2 ULPs on index 0 against launch 15 via decoded 5-tap separable FIR filter. Pipeline runner `vk_runner` runs SR M and L end-to-end on RX 9070 XT. See `docs/PORT-STATUS.md` and `docs/notes/`. |
| 5 | RR | **in progress (breadth first)**: All RR compute shaders implemented (`rr_gather`, `rr_enc0_feat`, `rr_level`, `rr_post_prep`, `rr_post`, `rr_downsample`, `rr_avg2x2`, `rr_pooled2`, `rr_ds_t`, `rr_ds_fused`, `rr_ae_*`). All 27 shaders pass O4 R1 lint, O5 dual-SPIRV build, O6 ACO 0-spill/0-scratch compilation, and O7 lavapipe functional tests. `vk_runner` runs RR F end-to-end on RX 9070 XT (~0.30 ms/frame). |
| 6 | integration + installer | **in progress**: `build/package/install.sh` implemented with `optiscaler` and `remove` routes (verified byte-for-byte uninstallation). MinGW cross-compiled `_nvngx.dll` PE binary exports all 14 D3D12 and Vulkan NGX functions with parameter cross-conversion and NVIDIA DLL size/presence verification gating. |
| 7 | performance | **measured on RX 9070 XT**: GPU timestamp frame times ~0.36 ms (SR M/L) and ~0.30 ms (RR F) at 256x144 -> 512x288, with 0 spilled SGPRs, 0 spilled VGPRs, and 0 scratch across all pipelines. |

## The real machine (where work happens now)

An RX 9070 XT under SteamOS 3.9 (`docs/hardware/RX9070XT-STEAMOS.md`). SteamOS's root is immutable and has
no compiler: everything is built in podman containers and the binaries run on the host. Everything
generated goes under `build/` (ignored by git); nothing from NVIDIA or NeuralHub may be committed except
the NeuralHub installer and the artifacts already listed below.

### Setting up from a fresh clone

```bash
git submodule update --init                       # reference-nr-vulkan (MIT), pinned at c71b0f4
(cd reference-nr-vulkan && bash fetch_deps.sh && python3 linux/build/build_network.py rdna4)
python3 -m venv build/venv && build/venv/bin/pip install numpy pillow pefile msgpack capstone
build/venv/bin/python -I tools/exe/extract.py artifacts/neuralhub-v0.8.0/NeuralHub_v0.8.0_setup.exe build/nh
#   -> build/nh/{neuralhub_host.dll, sr|rr|nr|fg.<target>.co}
podman pull docker.io/rocm/dev-ubuntu-24.04:7.2.4   # ROCm for the reference harness (/dev/kfd is usable as deck)
tools/hwtest/build_native.sh && tools/hwtest/build_plan.sh && tools/hwtest/build_probe.sh   # NR runner etc. -> build/hwtest/
```

Ghidra (for host-code questions; `tools/re/README.md`): Ghidra 12.1.4 unzipped to `build/ghidra/`, a JDK 21
image `localhost/ghidra:1` (`podman run --name t ubuntu:24.04 bash -c 'apt-get update && apt-get install -y
openjdk-21-jdk-headless' && podman commit t localhost/ghidra:1`), then the commands in `tools/re/README.md`.
A full decompile of `neuralhub_host.dll` takes about a minute; `tools/re/annotate.py` names the kernels in it.

### The private storage repository (DLLs and other inputs not in this repo)

**https://github.com/phly95/dlss5-private-storage** (private; the user's): every NVIDIA DLL below
(310.9.1.0, 310.9.16, the patched 310.8.0 NR, split into <100 MB parts), the Ghidra decompile of
`neuralhub_host.dll` (saves the Ghidra setup), and the rendered moving test sequence `sponza16`. Its README
has the restore commands into `build/` and the SHA-256 of every file. A fresh machine: clone it, run them,
then continue with the setup above. Put any new non-committable input there (split files over 90 MB) and
list it in its README. Never copy its contents into this repository.

### The user's DLLs (never commit them or anything extracted from them)

| what | where on this machine |
| --- | --- |
| `nvngx_dlss.dll` / `nvngx_dlssd.dll` 310.9.1.0 (SR / RR, the versions NeuralHub targets) | `~/.local/share/Steam/steamapps/common/Crimson Desert/bin64/` |
| `nvngx_dlssnr.dll` 310.8.0.0, community-patched (architecture check lifted) | downloaded from https://huggingface.co/Bandukids/DLSS-Runtimes (the user's pointer); weights verified identical to the reference's tested model |
| 310.9.16 `nvngx_dlss/dlssd/dlssg.dll` (no NR inside) | the user's zip; not needed so far |
| all of the above | also in the private storage repository (section above) |

NR model: `tools/hwtest/extract_nr_model.sh <nvngx_dlssnr.dll>` -> `build/model/dlssnr.bin` (the reference's
extractor with only the whole-file hash check relaxed; every weight entry is still checked), then
`build/venv/bin/python tools/hwtest/nr_gate.py --runner build/hwtest/nr_graph --plan-tool build/hwtest/nr_plan
--model build/model/dlssnr.bin --work build/hwtest/gate`.

## The cloud container (first sessions only; still useful for offline compiles)

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
  preset F; NR: `nvngx_dlssnr.dll` 310.8.0.0. All three are on this machine (table above); none may be committed.
- NeuralHub has built-in dumps usable on Windows: env `DLSSNR_DUMP_LAYERS`, `DLSSNR_DUMP_NOISE`, and
  `[Debug] DlssSrDumpNextFrame=1` (writes `sr_dump_<n>/stage_in_color.bin`, `stage_in_depth.bin`,
  `after_hidden0/1.bin`, `stage_output.bin`, `game_output.bin`; HIP backend only). Their file format is
  not yet known. RR has no dump.
- FP8 is E4M3, RNE, non-saturating (`v_cvt_pk_fp8_f32` without clamp); WMMA is E4M3×E4M3→F32.

## Session 2026-10-25 (cloud container, no GPU): where it stopped

- **verify v2 is frozen**: `1fea806` "verify: freeze v2 ..." fixes every defect listed under "Audit"
  below; `e4c9367` "verify: freeze v2 (correction ...)" fixes two bugs found in it before any port work
  (r2scan flagged its own pinned hash table; `rr_weights` needed `--dither`). `run_all.sh` V10 takes the
  newest "verify: freeze v2" commit, i.e. `e4c9367`. `tools/verify` must not change again; a real bug in it
  is a question for the user.
- `tools/verify/offline_check.sh` (GPU-less checks O1-O7): O1-O3 pass here; O4-O7 fail with "no shaders/
  directory" because no kernel exists yet. It does not pass until the first kernel is committed.
- `tools/gpu/collect_oracle.sh`: oracle dumps (srref M/L, rrref F, frames 0-1, seeded 256x144 -> 512x288)
  for the private repository. `--dry-run` checked here; the real run needs the card.
- `tools/re/slice.py` (def-use slices) and `tools/re/ssa.py` (straight-line SSA listing, `--expr` trees):
  reading aids for the ISA; text only, nothing evaluated.
- `k_sr_stage`: partial RE in `docs/sr/VULKAN.md`; no shader written. **Resume there**: the section
  "Seen but not yet decoded" and its softmax lead (unconfirmed), with
  `tools/re/fn.sh sr <symbol> > k.s; tools/re/ssa.py k.s --to 4660` (0x4660 is the first loop's start).
- The work was stopped at the user's request (goal cleared) with everything pushed.

**GPU-pending (run on the RX 9070 XT, from a clean checkout of this branch):**

    git fetch origin ccr-a7b81023-mipeai && git checkout ccr-a7b81023-mipeai
    # inputs: tools/verify/README.md "Restoring the inputs"; build/venv as in "Setting up from a fresh clone"
    python3 -I tools/exe/extract.py artifacts/neuralhub-v0.8.0/NeuralHub_v0.8.0_setup.exe build/nh
    bash tools/gpu/collect_oracle.sh
    # then follow the printed commands: copy build/oracle/<id>.tar.xz.part* and README into
    # phly95/dlss5-private-storage under oracle/, commit, push

With those dumps in the private repository, each kernel can be diffed offline (`tools/verify/diffdump.py`).
`run_all.sh` itself only makes sense once the port runner and package exist.

## Audit of the 2026-10-24 Vulkan attempt (`c229591`..`560ca36`)

A session asked to port SR/RR under a frozen verify script (`tools/verify/run_all.sh`, rules R1-R5,
checks V1-V10) produced `c229591` ("verify: freeze ...") and four port commits. The port commits are
reverted; they stay in history for reference. The freeze commit is left as is: changing anything in it
is the user's call (R4).

**Reverted** (`140482f`, `18bd87e`, `831401d`, `560ca36`): `shaders/`, `src/vk/`, `docs/sr/VULKAN.md`.
None of it matched the oracle and nothing compared it: there is no port-vs-`srref --dump` diff script,
and VULKAN.md's "verified / PASS" rows had no numbers. The GEMMs were scalar fp8->f16 loops (the
`coopmm.glsl` macros, copied from the reference, were included but unused); `sr_post` was an invented
bicubic + bilinear-history blend at a fixed alpha 0.85; `sr_downsample` plain bicubic; auto-exposure the
textbook 0.18 / exp(mean log luma); `sr_enc0`'s input-feature layout a guess; `rr_level.comp` a lightly
edited copy of `sr_stage.comp`. The runner had no barriers between dispatches, never filled `g_sr_lut`,
ignored `--mode` (RR cells ran the SR network), hard-coded the dec0 scales to 1.0 and did not enable
`VK_EXT_shader_float8`.

**Kept, with known defects** (`c229591`; all fixed in `1fea806` + `e4c9367`, "verify: freeze v2", above):
- V9: an absent `build/package/install.sh` is skipped and the run still passes; the NGX Vulkan path of
  `ngx_client` binds no images and reads nothing back; Proton outputs are never checked against the
  oracle; RR (feature 13) never runs; there is no missing / wrong DLL test.
- V5: the corrupted-weights and no-history runs come from the port's own flags; a 10 dB drop passes even
  if the degraded output would still pass V4 (`eval_verify.py:180`); SR M static only.
- V8: wall-clock of the whole process, the oracle's including podman start-up; under 50 ms exempt.
- V7: any log line with "RADV" passes, lavapipe is not excluded, the shaderstats table is not printed,
  and RADV's own "Spilled SGPRs: 0" lines probably trip the spill grep.
- V6: no SteamOS host check, no package scan, no extracted-weight hash scan; the ELF scan skips every
  path containing "nh".
- V1: falls back to `sponza16` (not held out, 16 frames) without Blender; ground truth is always off, so
  the 0.1 dB GT check never runs; the static generators draw the reset frame from 4-7, not 4-8.
- V3: preset L cells are skipped silently without L weights; the format cell sets `fmt_mv` 18 but no
  `mv_*.u32` is generated (srref exits, the cell cannot pass); the generators write 8-bit RGBA under
  colour code 5, which `docs/sr/HOST-RE.md` reads as RGBA32F; codes 16 / 18 are undocumented; srref's
  new auto-exposure arguments are guesses; rrref silently skips missing input files.
- V10: falls back to HEAD when no freeze commit is found; does not print the (empty) diff.

Usable from the freeze commit: the seeded camera / helmet paths and reset frames in the generators,
optional GT in `convert_seq.py`, srref's extra dumps (`g_col`, `g_dep`, `g_mv`, `g_mvhi`, preset L
buffers), the `/proc` maps / fd sampling loop, and the D3D12 path of `ngx_client`.

## Do not retry

- **Do not build a CPU emulator that executes NeuralHub's code objects.** That was attempted in the
  first session and was stopped by a safety check; the partial work was deleted. Do not resume it in
  another form.
- **Do not write approximations of NeuralHub kernels** (a generic TAA blend, bicubic resampling, textbook
  auto-exposure) or mark a kernel verified without a recorded per-buffer max error against `srref --dump`.
  That is what the reverted attempt did.
- (Resolved 2026-10-24) The user chose to port NeuralHub's SR/RR kernels to Vulkan; that is no longer blocked.

## Open questions for the user

1. (Answered 2026-10-24: port NeuralHub's kernels.)
2. (Answered: this machine is the RX 9070 XT. Windows dumps are no longer needed: NeuralHub's kernels run under ROCm here.)
3. NR at 310.9.16 (the 49 dB figure at 1080p) needs `nvngx_dlssnr.dll` 310.9.16.
4. (Answered 2026-10-25: re-freeze as "verify: freeze v2"; done in `1fea806` + `e4c9367`.)
5. Preset M's weight checksum mismatch (Known loose ends) is still unexplained; `collect_oracle.sh`
   requires only L's checksum. Is an M oracle from these weights acceptable as M's reference?
6. `k_sr_stage`'s second normalisation (`docs/sr/VULKAN.md`, "open") must be decoded exactly before a
   shader is written; until then there is no kernel to commit.

## SR: how to continue (2026-10-24)

1. `python3 -I tools/exe/extract.py artifacts/neuralhub-v0.8.0/NeuralHub_v0.8.0_setup.exe build/nh` (code objects, host DLL).
2. Weights: `build/venv/bin/python -I tools/sr/sr_weights.py <Crimson Desert>/bin64/nvngx_dlss.dll --out build/re/weights`,
   then `tools/sr/sr_pack.py build/re/weights {M,L} build/re/packed/{M,L}`.
3. Reference: see `docs/sr/HOST-RE.md` (ROCm 7.2.4 in podman; `srref ... --dump` writes every stage tensor and the
   dec0 outputs). Every Vulkan kernel is to be checked against it, stage by stage. The diff script that does
   this (port dump vs `srref --dump`, per-buffer max error) does not exist yet; write it before the first kernel.
4. Port order: `k_sr_stage<64,2,0,96,2,false>` (encoder), `k_sr_tube_split<128,4,false>`, the decoder stages,
   `k_sr_enc0`, `k_sr_dec0`, `k_sr_post<false,12,16,false,false>`, `k_sr_downsample`, exposure; then preset L's
   `k_sr_enc0L_in` / `k_sr_l0`. On this card every stage is "fp8-safe", so only the `false` instantiations run.
   What is known of `k_sr_stage` so far: activations are `[C/32][H][W][32]` fp8; a workgroup (64 threads)
   covers an 8x8 pixel tile at grid*8 - shift, mirror-padded at the borders; RMSNorm = f16 sum of squares
   (fixed packed-add tree + a lane^16 exchange) + 2^-14, rsqrt from `g_sr_lut`, times the sqrt(C)-scaled gain,
   to fp8; GEMMs as pairs of K=32 WMMA chains whose f32 results are added through f16 (`v_fma_mixlo_f16`).
   No `exp` instruction anywhere, but see `docs/sr/VULKAN.md`: a bit-built exp2 and the `1/h` table suggest a
   softmax after all (unconfirmed).
5. Faster route to games, if wanted: NeuralHub's kernels run under ROCm, so a Vulkan layer could drive them
   through HIP external-memory interop. This breaks the goal's "no HIP at run time" and needs ROCm
   libraries; ask the user before going this way.

## Test content

- Static: the NR verification's 4K Tomb Raider frame (`reference-nr-vulkan/docs/ngx-verification/310.8.0/
  single-frame-inputs/`), jittered by `tools/sr/gen_frames.py` / `tools/rr/gen_rr_frames.py` (RR: + noise).
- Moving, path-traced: `tools/scenes/` (README there): Blender 4.2.3 Cycles renders Khronos's Sponza with the
  DamagedHelmet flying through it, a panning dolly camera, real low-sample noise, exact motion vectors
  (checked by `check_mv.py`), every RR guide, and denoised ground truth. Cycles runs on the 9070 XT through
  HIP inside a ROCm container (~49 s/frame, mostly the 512-spp ground truth). `sponza16` (16 frames,
  960x540 -> 1920x1080) is in the storage repo. Results: RR 33.2 dB (frame 0) -> 36.4 dB (frame 15)
  against ~22 dB for the noisy input; SR 33.0-33.4 dB vs bicubic 30.1-32.0 dB (its "clean" input still has
  256-spp noise and fireflies while the ground truth is denoised, so this caps SR's score).

## Next steps, in order

1. SR in Vulkan (section above): one kernel per commit, each compared with `srref --dump` output for the
   same inputs (bit-exact where the ISA's rounding points are mirrored; otherwise report the difference),
   compiled with zero spills (`RADV_DEBUG=shaderstats`, as in `docs/hardware/RX9070XT-STEAMOS.md`).
   Shaders follow the reference NR port's conventions (`reference-nr-vulkan/CLAUDE.md`: wave32 pinned,
   FP8 only where the original uses it, 16-byte aligned LDS coopmat loads). Then a Vulkan host that runs
   the whole SR frame and matches srref end to end (39 dB test above, and per-frame diff vs srref).
2. RR: the ROCm reference exists (`docs/rr/HOST-RE.md`); port its kernels to Vulkan the same way as SR's,
   checked against `rrref --dump` (it writes every engine buffer as `E<index>.bin`). The `rrk::k_level`
   family (16 units, 18 instantiations) is the bulk; `k_rr_gather` (84 KB of ISA) is a format converter.
3. Integration: the reference's Vulkan layer + OptiScaler route (`reference-nr-vulkan/linux/`), SR/RR
   behind NGX/FSR inputs; installer modelled on `reference-nr-vulkan/linux/package/install.sh`.
4. Performance on this card; NR's in-game runtime path vs NVIDIA (the gate so far used the offline runner).

## Known loose ends

- Preset M's raw weights do not reproduce NeuralHub's M checksum (L does); M still packs to the expected
  sizes and runs well under srref. Cause unknown (`docs/sr/HOST-RE.md`).
- srref runs NeuralHub's `k_sr_gather` with `--gather` (only the RGBA16F / RG16F / R32F inputs, code 0:
  bit-identical to the direct upload); the other formats, `k_sr_gather_mv` (output-resolution MVs) and the
  auto-exposure path are still unexercised. The Vulkan port needs them all.
- Coverage still missing in the references: RR's transparency / alpha / virtual-hit / SSS / mask inputs,
  UNORM normals, linear depth; SR preset L on the moving scene; a mid-sequence reset; longer sequences.
- `sponza16` frame 6 has one firefly (8688 in a pixel): fine for scoring, but clamp
  (`cycles.sample_clamp_direct`) if a future render needs clean warping checks.
- The decompiler shows raw float stores as `(int)` casts; check every scalar's type against the kernel ISA
  (this hid a bug once: dec0's first scalar).
