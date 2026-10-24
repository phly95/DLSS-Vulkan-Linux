# Deliverable 1: kernel and interop inventory of NeuralHub v0.8.0 (Windows)

Source: `NeuralHub_v0.8.0_setup.exe` only (sha256 `aae53654…0b834`), nothing else from the Windows
project. Every fact below comes from one of three places in that file, and each statement says which:

- **[md]**: the AMDGPU code-object metadata (`NT_AMDGPU_METADATA` note): kernel names, kernarg sizes,
  LDS, scratch, VGPR/SGPR counts, wave size.
- **[isa]**: the gfx1201 disassembly (`llvm-objdump --mcpu=gfx1201`): instruction classes, which kernarg
  bytes a kernel reads, conversion modifiers.
- **[host]**: strings in `neuralhub_host.dll`: log messages, ini/env keys, and the source expressions its
  error-check macro kept (`hipMalloc(&g_col, 8 * rp)` and the like).

Anything not backed by one of those is marked **inferred** (usually from a kernel or template name).

Reproduce:

```sh
python3 -I tools/exe/extract.py NeuralHub_v0.8.0_setup.exe out/      # needs 7z, pefile
python3 -I tools/exe/kernels.py out/{nr,sr,rr,fg}.gfx1201.co > out/kernels.json   # needs msgpack, llvm-objdump >= 18
python3 -I tools/exe/inventory.py out/kernels.json > docs/inventory/kernels-generated.md
```

The extracted binaries are NeuralHub's code and are never committed; only these tools and the
derived tables are.

## 1. What is in the installer

| file (inside `neuralhub.exe`'s `.rdata`) | what it is |
| --- | --- |
| `neuralhub.exe` | Tauri manager app: imports the user's NVIDIA DLLs, extracts their weights into `dlssnr_on_amd_weights.bin` and caches, installs the mod. **[host]** |
| `neuralhub_host.dll` (101.8 MB) | The in-game mod. A `version.dll` proxy that also exports the full NGX D3D12/D3D11/Vulkan/CUDA entry set, `nvapi_QueryInterface` (GPU spoof), `NeuralHub_HostStart/HostAbi/HostVersion`. Carries a 74 MB `.hip_fat` section with four HIP fat binaries. Loads `amdhip64_7.dll` with `LoadLibrary` (no static HIP import). **[host]** |
| `neuralhub_stub.dll` (94 KB) | Proxy loader stub with the same export set. **[host]** |
| `amd_fidelityfx_upscaler_dx12.dll`, `amd_fidelityfx_framegeneration_dx12.dll` | AMD's stock FSR 4.1.1 / FSR FG DLLs (`ffx*` exports), used as the "FSR provider" fallback. Not part of the port. |
| HIP fat binary 1 (`nr`) | 193 kernels; gfx1030/10.3-generic, gfx1100/1101/1102/11-generic, gfx1200/**gfx1201**, gfx9-generic. |
| HIP fat binary 2 (`sr`) | 52 kernels; gfx11xx and gfx12xx. |
| HIP fat binary 3 (`rr`) | 33 kernels; gfx11xx and gfx12xx. |
| HIP fat binary 4 (`fg`) | 84 kernels; gfx11xx and gfx12xx. |

The models themselves are not in the installer. They come from the user's own NVIDIA DLLs **[host]**:

| network | NVIDIA DLL | version the port targets | presets |
| --- | --- | --- | --- |
| NR | `nvngx_dlssnr.dll` | 310.8.0.0 ("DLSS 5 Neural Rendering, 2026-08") | one model |
| SR | `nvngx_dlss.dll` | 310.9.1.0 | **M** and **L** only; K and the rest are "NOT ported", M or L serves them instead |
| RR | `nvngx_dlssd.dll` | 310.9.1.0 | **F** only ("preset F serves every mode") |
| FG | `nvngx_dlssg.dll` | - | (present; not in scope of this port) |

Consequence for this port: no real weights can be loaded on this machine. The rule in this repository
is the reference's: weights reach users only through an extractor run on their own DLL.

## 2. The networks

All counts are for the gfx1201 code object, the one that runs on an RX 9070 XT. Every kernel is
wave32 **[md]**. The generated per-family and per-kernel tables are in
[kernels-generated.md](kernels-generated.md).

| | kernels | instructions | FP8 WMMA | F16 WMMA | f32→e4m3 cvt | tex samples |
| --- | --- | --- | --- | --- | --- | --- |
| NR | 193 | 455,252 | 12,688 | 134 | 19,536 | 42 |
| SR | 52 | 137,993 | 4,548 | 200 | 4,112 | 256 |
| RR | 33 | 89,146 | 2,188 | 0 | 2,764 | 40 |
| FG | 84 | 142,056 | 0 | 299 | 0 | 0 |

### 2.1 NR (`nr`)

The NR model is the same network the reference implements (`CC_SILVER_AARDWOLD`, 310.8.0): a Swin
U-Net with ViT-1D attention levels. NeuralHub runs it with three kernel generations side by side:

- generic kernels (`k_qkv`, `k_attention*`, `k_ffwd*`, `k_expand*`, `k_conv_res*`, `k_contract2`,
  `k_dec_upsample`, `k_final_head`): the `DLSSNR_NO_REG` / `DLSSNR_SLOW_PREPOST` debug path, RDNA4 only **[host]**;
- "register" kernels (`k_reg_*`, `k_reg1d_*`, `k_v1dl_*`, `k_swin_var`): the default path **[host]**;
- persistent "Phase C" chains (`k_reg_swin_chain<256>`, `k_reg_swin_run<64|128|256>`, `DLSSNR_CHAIN=1`,
  gfx12 only) **[host]**.

Families (role **inferred** from names unless marked):

| family | role | channels / template |
| --- | --- | --- |
| `k_pre_block_1h_32_fp8`, `k_post_block_1h_32_fp8` | image → level-0 tokens (with in-kernel noise on D3D12 **[host]**), level-0 tokens → image | C = 32, one head |
| `k_swin_1h_32_fp8`, `k_swin_var<C,…>`, `k_reg_swin32<n,…>`, `k_reg_swin_mh<C,n,…>` | Swin window attention + FFN blocks per level | C = 32, 64, 128, 256 (heads 1/2/4/8 per the reference) |
| `k_reg_vit_*`, `k_reg1d_*`, `k_v1dl_*`, `k_vit512_*` | ViT-1D bottleneck levels (`VIT1D_*` limits, "level-6 cells") **[host]** | 512 |
| `k_reg_decup`, `k_dec_upsample` | decoder up-sampling | |
| `k_reg_head`, `k_final_head` | output head | |
| `k_noise` | noise field (`DLSSNR_DUMP_NOISE`, `HipNoiseHandoff`) **[host]** | |
| `k_reproject`, `k_import`, `k_export`, `k_mean`, `k_expo_*` | history reprojection, colour encode / decode, exposure | |
| `k_tin_to_frag*`, `k_frag*_to_tin`, `k_repack`, `k_w16_decode` | layout converters (NVIDIA "tin" 4×4 layout ↔ WMMA fragments), weight repack and e4m3→f16 decode | |
| `k_flag_*`, `k_abort`, `k_test_stall`, `k_clock_probe`, `k_align_probe`, `selftest::*` | runtime plumbing, see §3 | |

### 2.2 SR (`sr::`, `sr::io::`, `srunit::`)

A convolutional encoder/decoder (**inferred** from names and template parameters):

| family | role | template parameters |
| --- | --- | --- |
| `srunit::k_sr_gather`, `k_sr_gather_mv` | copy the game's colour / depth / MVs into the engine's buffers: `g_col` 8 B/px (RGBA16F), `g_dep` 4 B/px (f32), `g_mv` 4 B/px (2×f16), `g_mvhi` 4 B per output px **[host]** | |
| `sr::io::k_sr_ae_luma`, `k_sr_ae_reduce`, `k_sr_ae_copy`, `k_sr_exposure` | auto-exposure (luma grid, 512-wide reduce: `ae_tmp` = ⌈n/512⌉ floats **[host]**), exposure → f16 | |
| `sr::io::k_sr_downsample`, `k_sr_enc0`, `k_sr_enc0L_in`, `k_sr_l0<m,2,b>` | input preparation and level-0 encoder: `l0_in` 32 B/px with a `L0_EXT` border, `l0_skip` 32 B/px **[host]** | `k_sr_l0<0|2, 2, bool>` |
| `sr::k_sr_stage<C, g, mode, Cnext, n, bool>` | encoder/decoder stage; FP8 WMMA | C ∈ {64, 96, 128}, Cnext ∈ {96, 128, 0}, mode ∈ {0, 1, 2} |
| `sr::k_sr_tube_split<128, 4, bool>` | bottleneck | C = 128 |
| `sr::io::k_sr_dec0` | level-0 decoder | |
| `sr::io::k_sr_post<bool, a, b, bool, bool>` | output resolve against the history `ihist` (8 B/px at `W15×H15`, read through a HIP texture object) into `out` RGBA32F **[host]** | (a, b) ∈ {(12, 16), (15, 27)}: **inferred** one per preset (M / L) |
| `sr::k_sr_w16`, `sr::io::k_sr_unorm8_table` | e4m3 → f16 weight decode with `g_sr_w16_delta`; unorm8 → f16 table (`g_unorm8_p`, 512 B) **[host]** | |

The trailing `bool` on most SR kernels doubles every instantiation: **inferred** to be the
history-valid / reset variant.

### 2.3 RR (`rrk::`, `rrio::`, `rrunit::`)

A five-level U-Net (C = 32, 64, 96, 128, 160) with recurrent hidden state (**inferred**), preset F:

| family | role | template parameters |
| --- | --- | --- |
| `rrunit::k_rr_gather` (84 KB of code, the largest RR kernel), `k_rr_gather_out` | gather every guide into the engine layout | |
| `rrio::k_ds_t`, `k_ds_fused`, `k_downsample<0|1>`, `k_pooled2`, `rrk::k_avg2x2` | pyramid / down-sampling | |
| `rrio::k_enc0_feat` | level-0 feature encoder | |
| `rrk::k_level<C, g, mode, Cdown, Cup, n>` | U-Net level; FP8 WMMA | C ∈ {32, 64, 96, 128, 160}; n ∈ {1, 4} |
| `rrio::k_post_prep`, `rrio::k_post<24|26>` | output resolve | 24 / 26: **inferred** output-alpha on / off |
| `rrio::k_ae_block`, `k_ae_final` | RR's own auto-exposure ("DLSS-RR always runs its own" **[host]**) | |
| `rrk::k_rr_w16` | e4m3 → f16 weight decode | |

RR state buffers **[host]**: `hidden[0]`, `hidden[1]`, `gates`, `factors`, `aniso` (4 or 8 B per render
px), `mv_ero`, `mv_nondil`, `prev_mvec[0/1]`, `dhist` (`DW×DH`), `ohr` (`W15×H15`), `ih_tex[]` /
`ihist_tex[]` (history textures), `mvn_tex`.

RR guide inputs from the NGX parameter block **[host]**: colour, motion vectors (render resolution;
`MVLowRes` is ignored as NVIDIA's F does), depth (inverted), `GBuffer.Normals` (SNORM handled;
roughness in `normals.w` when `DLSS.Roughness.Mode != 0`), `GBuffer.Roughness`,
`DLSS.Input.DiffuseAlbedo`, `DLSS.Input.SpecularAlbedo`, `DLSSD.SpecularHitDistance` (with the
"virtual hit" path, `[Debug] DlssRrVirtualHit`), the SSS guide, the DoF guide, and
`DLSSD.ResponsivityMask`.

### 2.4 FG (`fg::`)

Out of scope (the goal covers NR, SR, RR). Recorded for completeness: F16 WMMA only, no FP8; an
optical-flow / candidate-blend pipeline (`k_EstimatePrev2CurrScatter`, `k_BlendCandidatesFused`,
`k_OutputPush/Pull*`) plus a small net (`fg::nk::kConv`, `k_DL1Net_Input/Output`, `kUp8`). It runs on
the D3D12 runtime only **[host]**, and needs Streamline's frame pacing.

## 3. HIP runtime and D3D12 ↔ HIP interop calls

The host resolves these from `amdhip64_7.dll` at run time **[host]**. Column 3 is what the port uses
in their place.

| call(s) | what NeuralHub does with it | Vulkan / Linux replacement |
| --- | --- | --- |
| `__hipRegisterFatBinary`, `__hipRegisterFunction`, `__hipRegisterVar`, `__hipUnregisterFatBinary` | register the four fat binaries and their kernels / device globals | `vkCreateShaderModule` + `vkCreateComputePipelines` per pipeline of `pipelines.json` |
| `hipLaunchKernel` | every kernel launch; struct parameters passed by value (kernarg up to 288 B) | `vkCmdDispatch`; struct parameters as push constants (≤ 128 B on RADV guaranteed; RADV gives 256) or a small UBO when larger |
| `hipMalloc`, `hipMallocPitch`, `hipFree`, `hipMemset*`, `hipMemsetD16Async`, `hipMemset2D*` | engine buffers, history, weights | `VkBuffer` arenas with offsets (the reference's convention), `vkCmdFillBuffer` / a fill shader for 16-bit patterns |
| `hipMemcpy*`, `hipMemcpy2D`, `hipMemcpyAsync` | weight upload, exposure readback, dumps | staging buffer + `vkCmdCopyBuffer` |
| `hipMemcpyToSymbol`, `hipMemcpyFromSymbol` | device globals: `g_e4m3_lut` (512 B), `g_sr_lut`, `g_sr_w16_delta`, `io::g_unorm8_p` | specialization constants for scalars, a bound read-only buffer for tables |
| `hipCreateTextureObject`, `hipDestroyTextureObject` | history (`ih_tex`, `ihist_tex`), `mvn_tex`, the game's inputs, sampled by `image_sample` | `VkImageView` + `VkSampler` (combined image sampler) with the same filter / address mode |
| `hipStreamCreate*`, `hipStreamCreateWithPriority`, `hipDeviceGetStreamPriorityRange`, `hipStreamSynchronize`, `hipStreamDestroy` | a second, high-priority HIP queue beside the game's | **none needed**: record into the game's command buffer (the reference's layer does this) |
| `hipEventCreate*`, `hipEventRecord`, `hipEventQuery`, `hipEventSynchronize`, `hipEventElapsedTime` | per-stage GPU timing and job completion | `vkCmdWriteTimestamp2` / timestamp query pool; completion via the game's own submission order |
| `hipImportExternalMemory`, `hipExternalMemoryGetMappedBuffer`, `hipDestroyExternalMemory` + D3D12 `CreateSharedHandle`, `CreateCommittedResource` | **zero-copy D3D12 ↔ HIP shared buffers** for every input and the result | **no equivalent, not needed**: under Proton the game's resources are already `VkImage`/`VkBuffer`s of the same device (vkd3d-proton / DXVK); a layer reads them directly |
| `hipHostMalloc`, `hipHostGetDevicePointer`, `hipHostFree` + `k_flag_wait`, `k_flag_set`, `k_abort`, a D3D12 "flag pipeline" and `SpinDraw` | **the inline wait**: the game's D3D12 queue spins on a flag in host-visible memory until the HIP job is done; graphics-draw spin slices to avoid the watchdog | **no equivalent, not needed**: one queue, ordinary barriers |
| `hipGetDeviceCount`, `hipSetDevice`, `hipGetDevice`, `hipGetDeviceProperties`, `hipOccupancyMaxActiveBlocksPerMultiprocessor` | pick the HIP device matching the D3D12 adapter (LUID), read WGP / CU count to size persistent kernels | `vkGetPhysicalDeviceProperties2`; CU count from `VkPhysicalDeviceShaderCorePropertiesAMD` (RADV exposes it) |
| `hipDriverGetVersion`, `hipRuntimeGetVersion` | **driver-version check**: the inline wait is only known to work on HIP runtime 70260201 / Adrenalin 26.x | **no equivalent**: replaced by checking Mesa version (`VkPhysicalDeviceDriverProperties`, ≥ 26.2) and the features below |
| `AmdExtD3DCreateInterface` (`amdxc64.dll`), `IAmdExtD3DDevice7`, `pipelineCrossCompileElfHsa` | **the D3D12 backend**: loads the very same gfx ISA code objects as D3D12 compute pipelines through AMD's driver extension, in the game's command list | **no equivalent**: neither vkd3d-proton nor RADV accepts raw HSA code objects. This is why every kernel has to be rewritten as SPIR-V. |
| `ID3D12Device::CreateRootSignature`, `CreateComputePipelineState`, descriptor heaps ("SR store", "RR store", "bridge store") | copy the RGBA32F result into the game's output texture | a small Vulkan compute store pass writing the game's output image |
| D3D12 `CreateQueryHeap`, `GetTimestampFrequency`, fences | timings, staging | timestamp queries; nothing for fences |
| NGX exports (`NVSDK_NGX_D3D12_*`, `NVSDK_NGX_Parameter_*`), `nvapi_QueryInterface` | NGX front end: claims to be NVIDIA's NGX, spoofs an RTX 5070 Ti so the game offers DLSS | OptiScaler (the reference's `optiscaler` route) owns the NGX front end on Linux; the port plugs in behind it as the SR/RR backend. GPU spoofing under Proton: `DXVK_NVAPI` / `dxgi.customVendorId` as OptiScaler documents |
| `ffx*` (FSR 3.1 / 4 upscaler and FG contexts), `libxess.dll` hooks | take the game's FSR / XeSS upscale inputs and serve them with DLSS-SR | OptiScaler's FSR / XeSS input hooks (same role) |

### Flags: things with no Vulkan equivalent

1. **ElfHsa cross-compiled pipelines** (`amdxc64.dll`, `IAmdExtD3DDevice7`). NeuralHub's D3D12 backend
   depends on it; nothing in Proton or Mesa replaces it. Every kernel is reimplemented in GLSL/SPIR-V.
2. **Shared D3D12 ↔ HIP handles** (`CreateSharedHandle` + `hipImportExternalMemory`). Not available
   in Proton (vkd3d-proton cannot export to a HIP runtime that is not there), and not needed: the
   port lives in the same Vulkan device as the game.
3. **Inline cross-queue spin wait** (`k_flag_wait` and friends, host-mapped flags, `SpinDraw`). Not
   needed with a single queue.
4. **HIP device enumeration / LUID matching**. Replaced by the Vulkan device the game already uses.
5. **Driver-version gate** (HIP runtime 70260201 / Adrenalin 26.x; `amdhip64_7.dll` presence; HIP SDK
   7.2 on RDNA2). Replaced by Mesa ≥ 26.2 and the feature checks: `VK_KHR_cooperative_matrix` with a
   16×16×16 E4M3×E4M3→F32 subgroup-scope entry, `VK_EXT_shader_float8`, `subgroupSize = 32`
   (`VK_EXT_subgroup_size_control`), `shaderFloat16`, `storageBuffer8/16BitAccess`.
6. **Adrenalin dependency** (`amdxc64.dll`, `amdhip64_7.dll`, `amd_comgr_3.dll`). None at run time on Linux.
7. **FG**: needs Streamline presentation pacing and HAGS spoofing (`IgnoreHagsForFrameGeneration`).
   Out of scope.

## 4. Precision traps found in the ISA

The goal's numerical-match requirement comes down to these instruction-level facts **[isa]**:

1. **FP8 is OCP E4M3, round-to-nearest-even, not saturating.** Every f32→fp8 conversion is
   `v_cvt_pk_fp8_f32` without `clamp` (26,412 sites across NR/SR/RR). The SPIR-V equivalent is
   `OpFConvert` to `FloatE4M3EXT` **without** `SaturatedToLargestFloat8NormalConversionEXT`; ACO emits
   the same instruction for it. A port that clamps changes results.
2. **WMMA is E4M3 × E4M3 → F32 with an F32 accumulator** (`v_wmma_f32_16x16x16_fp8_fp8`; no BF8
   variants, no F16 accumulate). This is the reference's `coopmm.glsl` contract. The order of the K
   reduction inside one WMMA is the hardware's in both builds. Across K tiles, the port has to keep
   NeuralHub's tile order: a different reduction order is a different picture (the reference's rule).
3. **F16 WMMA** (`v_wmma_f32_16x16x16_f16`) appears in NR pre/post blocks, `k_qkv`, `k_dec_upsample`, SR
   and FG: coopmat f16 × f16 → f32.
4. **Mixed-precision FMA** (`v_fma_mix*_f16`): an f32 product rounded **once** to f16. In GLSL,
   `float16_t(a*b)` rounds twice (f32 then f16). Use `float16_t(fma(a, b, c))` with f32 operands,
   which ACO lowers to `v_fma_mix`. Seen already in `sr::io::k_sr_exposure`.
5. **`v_dot2_f32_f16`** (1,232 in NR): the two f16 products are exact in f32, but the hardware sum rounds
   once and two chained `fma`s round twice. Expect ≤ 1 f32 ulp per call before the next fp8
   quantisation. That is within the fp8 tolerance, but must not be called bit-exact.
6. **Transcendentals**: `v_exp_f32` / `v_log_f32` are base-2. `v_rcp_f32` / `v_rsq_f32` are the
   hardware approximations (1 ulp), and RADV produces the same instructions for `1.0/x` and
   `inversesqrt` under default float controls. `v_sin_f32` / `v_cos_f32` take revolutions, not radians
   (only `k_noise` uses them, 18 sites).
7. **Lookup tables live in device globals** (`g_e4m3_lut` 256 × f16 in NR, SR and RR; `g_unorm8_p`,
   `g_sr_lut`). They have to be uploaded byte-for-byte as NeuralHub builds them on the host, not
   recomputed.
8. **Texture sampling** (`image_sample*`, 338 sites, mostly SR's history): the sampler's filter and
   address modes come from `hipCreateTextureObject`'s descriptor, which is in host code, not in the
   ISA. Bilinear weights on RDNA4 are 8-bit fixed point in both D3D12 and Vulkan, so matching the
   sampler state makes this exact.

## 5. Open questions (need host-code disassembly, not done yet)

- Exact field layout of each by-value parameter struct (`sr::StageArgs` 56 B, `sr::io::PostArgs`
  256 B, `rrk::Args`, `VarParams`, …). The kernarg-read offsets per kernel are in `kernels.json`; the
  meaning of each field needs the host's launch code (MSVC x64, no symbols) or the ISA.
- The order of launches per frame (the layer graph) for SR and RR: lives in the host's engine code.
- The on-disk format of `DLSSNR_DUMP_LAYERS` and of the SR dump job's `sr_dump_<n>/*.bin`.
- How `dlssnr_on_amd_weights.bin` / the SR and RR weight caches are laid out (the manager writes them
  from the NVIDIA DLLs).
