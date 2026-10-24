# The target machine: RX 9070 XT on SteamOS

Everything in the port that says "on the card" was measured on the machine this file describes, not
on Mesa's amdgpu drm-shim. It answers the device gate of
[docs/inventory/KERNELS.md](../inventory/KERNELS.md) section 3.5 with a real RDNA4 device.

## What it is

| | |
| --- | --- |
| GPU | `AMD Radeon RX 9070 XT (RADV GFX1201)`, vendor `1002` device `7550` (Navi 48) |
| OS | SteamOS 3.9.2 (holo), kernel `7.2.7-valve1-1-neptune-72` |
| Driver | RADV, `Mesa 26.2.0-devel (git-6c05e09fac)`, Vulkan 1.4.350, conformance 1.4.5 (`vulkan-radeon 26.2.0_devel.222142.steamos_26.05.18-1`) |
| Compute units | 64 (4 shader engines × 2 arrays × 8 CU/array, 2 SIMD/CU, wave size 64) |
| LDS | 65536 B per workgroup, 1024 invocations per workgroup |
| Timestamp | `timestampPeriod` 10.0 ns, `timestampComputeAndGraphics` yes |

The CU count is what a persistent kernel's grid is sized from, and it is exposed the way
`KERNELS.md` says: `VkPhysicalDeviceShaderCorePropertiesAMD`.

## The gate

`tools/hwtest/hwprobe` (build: `tools/hwtest/build_probe.sh`, run: `build/hwtest/hwprobe`)
prints this and exits non-zero if the FP8 configuration is missing.

Every feature the port needs is present: `cooperativeMatrix`, `cooperativeMatrixRobustBufferAccess`,
`shaderFloat8`, `shaderFloat8CooperativeMatrix`, `storageBuffer8/16BitAccess`, `shaderFloat16`,
`shaderInt8`, `vulkanMemoryModel`, `subgroupSizeControl`, `synchronization2`,
`workgroupMemoryExplicitLayout` with 8- and 16-bit access.

20 cooperative matrix configurations are reported, all `16x16x16` at `subgroup` scope:

| A × B → C | notes |
| --- | --- |
| **e4m3 × e4m3 → f32** | **the network's, non-saturating** |
| e4m3 × e5m2, e5m2 × e4m3, e5m2 × e5m2 → f32 | unused |
| u8/s8 combinations → u32/s32 | int8; the int4 option uses them |
| f16 × f16 → f16, f16 × f16 → f32 | the epilogue and the pre/post blocks |
| bf16 × bf16 → bf16, bf16 × bf16 → f32 | unused |

There is no FP8 K=32 configuration (that is gfx1250), which is why the port's fragments are all
`16x16x16`.

**The default compute subgroup size on this driver is 64, not 32.** `minSubgroupSize` is 32,
`maxSubgroupSize` is 64 and `requiredSubgroupSizeStages` includes compute, so every network pipeline
has to pin wave32 with `VkPipelineShaderStageRequiredSubgroupSizeCreateInfo` (the reference's
`nrvk.hpp` does, and so does `hwprobe`'s sibling tools). A build that forgets this compiles and runs
with different fragment sizes and LDS traffic.

## Running the network on it

`tools/hwtest/build_native.sh` builds the reference's own offline runner (`nr_graph.cpp`'s `main`)
for this machine: SteamOS has no compiler in its immutable root, so the build happens in a `podman`
Ubuntu 24.04 container and the binary is written to the host, where it runs against the host's Mesa.

The runner needs the network's weights, which are NVIDIA's and reach a machine only through
`nvngx_dlssnr.dll` (the reference's `linux/package/model-tools/extract_model.sh`). Without that DLL,
`tools/hwtest/synth_model.py` recovers the *names and sizes* of every weight file the network reads -
they are properties of the network's structure - and fills them with a constant, which is enough to
execute the whole graph and measure it. See `tools/hwtest/README.md`.

The runner also needs the **per-resolution plan** the runtime derives from the frame size
(`nr::make_native_plan`): `tools/hwtest/nr_plan.cpp` prints it for any width/height. The runner's
default plan (`linux/src/core/nr_frame.txt`) is a single frozen geometry; using it at another
resolution leaves every layer's token count at 4K, so the frame time barely moves with resolution
(15.7-18.7 ms from 64x64 to 4K - the fixed part is ~15.6 ms). With the right plan the cost scales
with pixels.

## Measured on this card

The reference's own offline runner (`tools/hwtest/build_native.sh`) executes the whole network on
this device. The weights are synthetic, so the pictures are meaningless; the numbers are a bring-up
and frame-time measurement, not a match against NVIDIA.

| frame | GPU time / frame (synthetic weights) | clock | reference 310.8 offline, Linux |
| --- | --- | --- | --- |
| 1920x1080 | 4.77 ms | 3061 MHz | 5.60 ms |
| 2560x1440 | 8.30 ms | 3017 MHz | 9.70 ms |
| 3840x2160 | 19.00 ms | 2922 MHz | 21.89 ms |

Three trials at each size agree to +/-0.02 ms (each with `--warmup 5 --repeats 30`). The
comparison is the reference README's 310.8 *offline benchmark* row because this network is the 310.8
`Original` model (`make_native_plan` fingerprint `ce136b3b...`); the 310.9.16 in-game row
(3.27/5.53/12.67 ms) is a different, newer model and is not the like-for-like figure. Synthetic
weights do not change an instruction count (the network's work is data-independent), so the frame
times are the real kernels' times.

Every pipeline the graph compiles does so with **zero VGPR/SGPR spills and zero scratch** on this
driver (`RADV_DEBUG=shaderstats`; clear the shader cache with `MESA_SHADER_CACHE_DISABLE=1` or the
dump is empty on a second run). Peak values: 32 KiB LDS, 256 VGPRs (one wave/SIMD for the heaviest
GEMM). The SPIR-V is `OpCooperativeMatrixMulAddKHR` over `Float8E4M3` 16x16x16 (578 such ops in
`g_attn.spv` alone), so the FP8 WMMA path is the one executing.

## The NR gate with the real model (310.8.0)

`tools/hwtest/nr_gate.py` runs the offline runner with the real 310.8.0 model on the three single-frame
inputs of the reference's NGX verification (`reference-nr-vulkan/docs/ngx-verification/310.8.0/`) and
compares the result with **NVIDIA's own output** for the same input (the unmodified DLL on an RTX 5090,
`Reset` on every frame, constant depth, zero motion), as 8-bit RGB.

The model came from a community-patched `nvngx_dlssnr.dll` 310.8.0.0 (SHA256 `8270b350…44cc206`; the
patch lifts NVIDIA's architecture check). Its whole-file hash is not the 310.8.0 hash
`extract_model.sh` accepts, so the extraction ran with that one check swapped for this hash. The
script's other check still ran and passed: all 599 extracted entries are byte-identical to the
reference's tested model (`model-files.sha256`). The weights are NVIDIA's own. Nothing from the DLL is
committed: `build/model/` is ignored.

| frame | port on this card vs NVIDIA | reference's published AMD vs NVIDIA | input vs NVIDIA | port vs reference's AMD picture |
| --- | --- | --- | --- | --- |
| 1920x1080 | **45.59 dB** | 45.56 dB | 26.44 dB | 49.50 dB |
| 2560x1440 | **47.97 dB** | 47.99 dB | 27.72 dB | 49.51 dB |
| 3840x2160 | **49.00 dB** | 49.06 dB | 28.69 dB | 50.70 dB |

So the port reproduces the reference's own accuracy on this card (within ±0.06 dB). It is **not ≥ 49 dB
at 1080p or 1440p**, and neither is the reference with the 310.8.0 model: the 49 dB figure in the goal
matches the reference's 310.9.16 numbers (49.15 / 49.40 / 50.59 dB), and checking that needs
`nvngx_dlssnr.dll` 310.9.16. The output is not byte-identical to the reference's published AMD
picture, which was made through the in-game runtime (`nr_runtime`, with its auto mask and local
controls), not the offline runner. The runtime path is what the gate should finally be measured on.
