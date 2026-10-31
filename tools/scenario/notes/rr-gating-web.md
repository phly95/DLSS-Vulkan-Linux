# How DLSS Ray Reconstruction (RR) is gated — research notes

Scope: what must be true for a game to show/enable RR, aimed at a Linux/Proton (custom `_nvngx.dll`) port of a Streamline-based title. Every bullet has a source URL. Negative findings are stated as negative.

---

## 1. Streamline side: is there a separate RR plugin?

- **Yes, but the file is `sl.dlss_d.dll`, not `sl.dlss_rr.dll`.** The RR plugin's own descriptor is `"name": "sl.dlss_d"`, `"id": 1001`, `"required_plugins": ["sl.common"]`, `"rhi": ["d3d11","d3d12","vk"]`. The Super Resolution plugin is a *different* descriptor: `"name": "sl.dlss"`, `"id": 0`.
  - https://github.com/NVIDIA-RTX/Streamline/blob/main/source/plugins/sl.dlss_d/dlss_d.json
  - https://github.com/NVIDIA-RTX/Streamline/blob/main/source/plugins/sl.dlss/dlss.json
  - DeepWiki indexes the same naming: "DLSS Ray Reconstruction Plugin (sl.dlss_d)" — https://deepwiki.com/NVIDIA-RTX/Streamline/4.5-dlss-ray-reconstruction-plugin-(sl.dlss_d)
  - A GitHub-wide search for `"sl.dlss_rr.dll"` returned **zero** results (GitHub issue search, 2026-10-31). Treat `sl.dlss_d.dll` as the real name.
- **Feature enum: `sl::kFeatureDLSS_RR = 1001`** (SR = 0, `kFeatureNRD_INVALID` = 1 is the *removed* old denoiser). There is **no `kFeatureDLSSRR` and no `kFeatureDLSSRRDenoise`** anywhere in the current SDK headers (`sl_core_types.h`) — Sourcegraph found 0 matches for `NVSDK_NGX_Feature_Denoise` across public repos too.
  - https://github.com/NVIDIA-RTX/Streamline/blob/main/include/sl_core_types.h
- **Docs**: RR has its own guide, §2.0 "CHECK IF DLSS-RR IS SUPPORTED", which is literally `slIsFeatureSupported(sl::kFeatureDLSS_RR, adapterInfo)` and, on failure, maps `eErrorOSOutOfDate` / `eErrorDriverOutOfDate` / `eErrorNoSupportedAdapterFound` → "fall back to the default method". Same pattern as `kFeatureDLSS`.
  - https://github.com/NVIDIA-RTX/Streamline/blob/main/docs/ProgrammingGuideDLSS_RR.md
  - https://github.com/NVIDIA-RTX/Streamline/blob/main/docs/ProgrammingGuide.md (§2 `slIsFeatureSupported` contract: adapter-NULL query returns OS/driver-level compat; after device creation the supported set can only shrink; `slGetFeatureRequirements` returns OS/driver/API requirements per feature)
- **A feature only runs if the game *requests* it**: "SL will only load features specified via the `Preferences::featuresToLoad` (assuming they are supported on user's system). If this parameter is empty, no features will be loaded." Also default prefs carry `eAllowOTA | eLoadDownloadedPlugins` (driver-side OTA plugin download). If the title never lists `kFeatureDLSS_RR`, the RR option cannot appear regardless of DLLs present.
  - https://github.com/NVIDIA-RTX/Streamline/blob/main/docs/ProgrammingGuide.md (§2.2.2 Preferences, §2.2.2.1 Managing Features)
- **Shipped separately in real games?** Yes — a Monster Hunter Stories 3 Steam thread lists that game's Streamline folder as `sl.common.dll, sl.dlss.dll, sl.dlss_d.dll, sl.dlss_g.dll, sl.interposer.dll, sl.pcl.dll, sl.reflex.dll`, i.e. RR's plugin sits next to `sl.dlss.dll` as a distinct file; the poster also notes RR only uses its own model file `nvngx_dlssd.dll`.
  - https://steamcommunity.com/app/2852190/discussions/0/806845425982327499/
- SDK binaries are distributed per-release zips (not in-tree since SL 2.7.32), so game builds bake whatever plugin set the developer integrated:
  - https://github.com/NVIDIA-RTX/Streamline/releases  ·  https://github.com/NVIDIA-RTX/Streamline#readme

**Practical implication:** if your game never ships `sl.dlss_d.dll` *and* never asks for `kFeatureDLSS_RR` in `featuresToLoad`, RR is invisible. Providing `_nvngx.dll` alone does nothing.

---

## 2. NGX side: model DLLs, feature id, capability params

- **Feature id = `NVSDK_NGX_Feature_RayReconstruction = 13`** in the current NGX SDK bundled with Streamline (`NVSDK_NGX_Feature_SuperSampling = 1 … FrameGeneration = 11, DeepDVC = 12, RayReconstruction = 13`). I found **no `NVSDK_NGX_Feature_Denoise` and no value like `0x00200000`** in current headers; Sourcegraph returned 0 matches for `NVSDK_NGX_Feature_Denoise`.
  - https://github.com/NVIDIA-RTX/Streamline/blob/main/external/ngx-sdk/include/nvsdk_ngx_defs.h
- **Model binary = `nvngx_dlssd.dll`** (DLSS-D = DLSS Denoiser = RR). Streamline reads its `FileVersion`: `func("nvngx_dlssd.dll", "FileVersion", ngxVersion)` (used for stats/OTA display).
  - https://github.com/NVIDIA-RTX/Streamline/blob/main/source/plugins/sl.dlss_d/dlss_dEntry.cpp (lines 256–264, 1283)
- **Support gates used by the plugin** (all in `dlss_dEntry.cpp`):
  1. driver ≥ **535.68** (accepted windows: 535.15–535.58, or ≥ 535.68 — older code has an ngx_core crash bug), else `"requires a driver supporting DLSSD"`;
  2. `getFeatureRequirements(NVSDK_NGX_Feature_RayReconstruction, info)` must succeed, else **`"DLSSD feature is not supported. Please check if you have a valid nvngx_dlssd.dll or your driver is supporting DLSSD."`**;
  3. `info.minGPUArchitecture = NV_GPU_ARCHITECTURE_TU100`, `info.minOS = 10.0.0`;
  4. at init: `params->Get(NVSDK_NGX_Parameter_SuperSamplingDenoising_Available, &supported)` → if false: `"NGX indicates DLSSD is not available - DLSSDContext cannot run"`.
  - https://raw.githubusercontent.com/NVIDIA-RTX/Streamline/main/source/plugins/sl.dlss_d/dlss_dEntry.cpp (lines 177–219, 1236–1247)
- **Capability parameter names (RR)** — in `nvsdk_ngx_defs_dlssd.h`:
  - `SuperSamplingDenoising.Available`, `SuperSamplingDenoising.NeedsUpdatedDriver`, `SuperSamplingDenoising.MinDriverVersionMajor/Minor`, `SuperSamplingDenoising.FeatureInitResult`
  - `RayReconstruction.Hint.Render.Preset.{DLAA,Quality,Balanced,Performance,UltraPerformance,UltraQuality}`
  - `RayReconstruction_Hint_Render_Preset`: `0=Default, D=4 (Transformer), E=5 (latest Transformer/DoF), F=6 (RR2 default)`, A–C removed, G–O "Do not use".
  - `DLSS.Denoise.Mode` (`0=Off, 1=DLUnified`), `DLSS.Roughness.Mode` (packed/unpacked)
  - https://github.com/NVIDIA-RTX/Streamline/blob/main/external/ngx-sdk/include/nvsdk_ngx_defs_dlssd.h
- **`NVSDK_NGX_Parameter_IsRayReconstructionSupported` does not appear** in current public headers (absent from `nvsdk_ngx_defs.h`, `nvsdk_ngx_params.h`, `nvsdk_ngx_defs_dlssd.h`). The availability keys really used are `SuperSamplingDenoising.Available` (RR) and `SuperSampling.Available` (SR). Confirmed by non-Streamline code: an NVIDIA Omniverse RTX log line reads `NGX does not support feature SuperSamplingDenoising.Available … Optional DLSS feature is disabled`.
  - https://github.com/NVIDIA-Omniverse/ovrtx/issues/5
- **Query APIs**: `NVSDK_NGX_D3D11/D3D12/CUDA_GetCapabilityParameters` (capability parameter map) and `NVSDK_NGX_*_GetFeatureRequirements(adapter, FeatureDiscoveryInfo, NVSDK_NGX_FeatureRequirement*)`, whose `NVSDK_NGX_Feature_Support_Result` bitfield = `Supported=0, CheckNotPresent=1, DriverVersionUnsupported=2, AdapterUnsupported=4, OSVersionBelowMinimumSupported=8, NotImplementedError=16`, plus `MinHWArchitecture` and `MinOSVersion`.
  - https://github.com/NVIDIA-RTX/Streamline/blob/main/external/ngx-sdk/include/nvsdk_ngx.h
  - https://github.com/NVIDIA-RTX/Streamline/blob/main/external/ngx-sdk/include/nvsdk_ngx_defs.h
- **Failure code you'll see when NGX refuses RR**: `NVIDIA DLSS-RR not available on this hardware/platform: (code: 0xbad00001, info: NVSDK_NGX_Result_FAIL_FeatureNotSupported)`.
  - https://github.com/3zwr1/AMD-NR---OptiScaler/issues/2

---

## 3. GPU architecture / driver gating

- **Not Ada-only — RR has never been RTX 40-only.** Streamline's RR plugin declares `minGPUArchitecture = NV_GPU_ARCHITECTURE_TU100` (Turing = RTX 20 and newer): https://raw.githubusercontent.com/NVIDIA-RTX/Streamline/main/source/plugins/sl.dlss_d/dlss_dEntry.cpp
- NVIDIA's own DLSS 3.5 announcement: "GeForce RTX 40 Series users can combine Super Resolution and Frame Generation with Ray Reconstruction … while **GeForce RTX 20 and 30 Series users can add Ray Reconstruction**"; also "improves ray-traced image quality **for all GeForce RTX GPUs**". https://www.nvidia.com/en-us/geforce/news/nvidia-dlss-3-5-ray-reconstruction/
- TechPowerUp (Gamescom 2023): "available this Fall for **all GeForce RTX GPUs (RTX 20-series and later)**", "relies on the Tensor Cores … Turing and newer". https://www.techpowerup.com/312669/nvidia-announces-dlss-3-5-ray-reconstruction-technology-works-on-geforce-20-and-newer
- Epic's official Alan Wake 2 support answer: "**DLSS Ray Reconstruction is supported on all NVIDIA RTX GPUs**". https://www.epicgames.com/help/alan-wake-2-c-46305258/technical-support-c-31078580/what-hardware-will-the-new-dlss-ray-reconstruction-be-supported-on-for-alan-wake-2-on-pc-a22441422
- 2026 re-check: a guide explicitly writes "Plenty of people assume Ray Reconstruction needs an RTX 40. It does not. The compatibility table on NVIDIA's own DLSS page lists it on RTX 50, 40, 30 and 20. The fortieth series requirement belongs to **Frame Generation**" — and notes each game may add its own condition. https://framesgames.com/dlss-4-5-ray-reconstruction-out-now/
- **Concrete driver numbers:**
  - Streamline RR plugin floor: **535.68** (or 535.15–535.58), vs **512.15** for plain DLSS — raw source links above; SL README says "MUST be 512.15 or newer": https://github.com/NVIDIA-RTX/Streamline#readme
  - RR's launch Game Ready driver: **537.42 WHQL** (21 Sep 2023, "Game Ready for … DLSS 3.5 … Ray Reconstruction in Cyberpunk 2077"): https://www.nvidia.com/en-us/geforce/news/cyberpunk-2077-phantom-liberty-dlss-3-5-ray-reconstruction-game-ready-driver.md
  - Driver-side evidence that RR can be poisoned by an old `_nvngx.dll`: a user on beta driver **550.09** had RR greyed out; quoting a Guru3d find, "_nvngx.dll is missing mentions of DLSSD / 'SuperSamplingDenoising' in this ver, while 537.13 / 537.34 both mention them" — downgrading to **537.42** fixed it. https://forums.cdprojektred.com/index.php?threads/2-0-ray-trace-reconstruction-greyed-out.11118982/
  - Later-generation (RR2/transformer) RR: Game Ready **580.88 WHQL or newer** for the NVIDIA-app override (checked per-game via "DLSS Override → Model Presets"), while the base Game Ready of that date was **616.56**; DLSS 4.5-era driver cited as **591.74 WHQL**. https://framesgames.com/dlss-4-5-ray-reconstruction-out-now/ · https://www.nvidia.com/en-us/geforce/news/dlss-4-5-geforce-game-ready-driver.md
  - No source found demanding "560+" or "566.14+"; treat those numbers as unverified.

---

## 4. Game-side gating (why the option is absent even on capable hardware)

- **The game must have implemented it at all**: PCGamingWiki — "DLSS Ray Reconstruction was added in Update 2.0" (Cyberpunk 2077); RR is an opt-in integration, not automatic. https://www.pcgamingwiki.com/wiki/Cyberpunk_2077
- **Cyberpunk 2077 gate**: RR ("Ray-trace reconstruction") is greyed out unless **Path Tracing / RT: Overdrive** is active **and** DLSS is on; drivers listed as a third cause in the same thread. https://forums.cdprojektred.com/index.php?threads/2-0-ray-trace-reconstruction-greyed-out.11118982/ · same conclusion in "Does RT Reconstruction work with Standard RT?" https://steamcommunity.com/app/1091500/discussions/0/3878218563598347893/
- **General checklist** (widely reproduced): (1) ray tracing enabled (RR only replaces *ray-traced* denoisers); (2) DLSS/DLAA active (RR "runs inside DLSS"); (3) the RR model file present next to `nvngx_dlss.dll` — "An older integration ships only the second one, and then there is no line in the menu at all, greyed out or otherwise. … the gate is written by each game rather than by NVIDIA." https://framesgames.com/dlss-4-5-ray-reconstruction-out-now/
- **Vendor gate example**: Crimson Desert Enhanced — RR greyed out on an RX 7900 XTX ("RR is for NVIDIA", AMD's equivalent is gated to RDNA4 9000-series). https://steamcommunity.com/app/3321460/discussions/0/805722431784337643/
- **Plugin/feature files shipped but option hidden**: Monster Hunter Stories 3 ships the full Streamline RR/FG DLL set yet exposes neither option ("Why does the game come with Framegen and Ray Reconstruction files, when those two aren't an option?"), and removing those files fixed a user's crashes/stutters with DLSS. https://steamcommunity.com/app/2852190/discussions/0/806845425982327499/
- **Modded/Elevated cases**: OptiScaler reports "ALAN WAKE 2 Crash with Ray Reconstruction disabled (only works with RR on)" https://github.com/optiscaler/OptiScaler/issues/109 ; Silent Hill 2 RR feasibility question https://github.com/optiscaler/OptiScaler/issues/75 ; Hogwarts Legacy RR stays greyed out even after full GPU spoofing to an RTX 5070 Ti https://github.com/danielblnc/DLSS-NR-on-AMD/issues/263
- No verified example found of a game restricting RR to Ada/RTX-40 **at the menu level**; gaming press instead says the opposite (§3).

---

## 5. Linux / Proton angle (known projects, issues, workarounds)

- **Working**: dxvk-nvapi issue on Bazzite + NVIDIA **615** drivers + Proton GE 17 (Control Resonant): "**DLSS Ray Reconstruction, Frame Gen, and SR all work** otherwise" under Proton, while an unrelated RT setting stayed greyed. https://github.com/jp7677/dxvk-nvapi/issues/399
- **Broken path**: under plain Wine with Streamline, "Streamline started but NGX did not: **`DLSSD feature is not supported`**"; the project's plan is to call RR through the **NGX SDK directly on Linux** ("the DLSS SDK ships Linux libraries"), bypassing Streamline. https://github.com/unfocused-mods/vspt-issues/issues/30 (context: https://github.com/unfocused-mods/vspt-issues/issues/21)
- **Native Linux NGX evidence**: Omniverse RTX on CachyOS logs `NGX does not support feature SuperSamplingDenoising.Available` for the binary `libnvidia-ngx-dlssd.so.310.7.128`, i.e. RR is refused per-GPU on Linux too, and removing that snippet makes the renderer fall back cleanly. https://github.com/NVIDIA-Omniverse/ovrtx/issues/5
- **Proton tooling**: `pantsoftime/dlss5-linux` — six injection routes (OptiScaler, ReShade feeder, "DLSS5VKLayer" out-of-process Vulkan layer, `addon-dlssnr-linux` forwarder that "bypasses the driver's over-the-air capability gate that makes the closed add-on fail under Proton"); tested RTX 5090, NVIDIA 610.57/615.71, GE-Proton 11. https://github.com/pantsoftime/dlss5-linux
- **RR-related issues/docs**:
  - OptiScaler+REMIX on AMD: `NVIDIA DLSS-RR not available … NVSDK_NGX_Result_FAIL_FeatureNotSupported` even with Vulkan spoofing; maintainer replied the fix shipped in launcher "0.3.5.1". https://github.com/3zwr1/AMD-NR---OptiScaler/issues/2
  - "ray reconstruction works on linux distros ?" (DLSS-NR-on-AMD) https://github.com/danielblnc/DLSS-NR-on-AMD/issues/241
  - OptiScaler Known-Issues wiki (no RR-specific interop entry found; workarounds are mostly `OptiScaler.ini` flags) https://github.com/optiscaler/OptiScaler/wiki/Known-Issues
  - Related Linux stacks: https://github.com/guentra/dlss5-amd-hip-linux (RR/NR model weights on AMD ROCm), https://github.com/bmitch87/DLSS5VKLayer (cited from dlss5-linux README).
- **Workarounds actually used by people**: drop the missing model DLL (`nvngx_dlssd.dll`) or the whole RR plugin set into the game folder; use OptiScaler/GPU-spoof launches as `winmm.dll`/`dxgi.dll`; use a Vulkan layer instead of in-game Streamline; or set per-game NVIDIA-app overrides on Windows. No env var or ini key was found that makes a Steam title expose an RR toggle it never implemented.

---

## 6. `nvngx_dlssd.dll` vs `nvngx_dlss.dll` versioning, and what happens when it's missing/old

- **Missing `nvngx_dlssd.dll` → RR is *hidden*, not crashed**: Streamline first checks driver, then `getFeatureRequirements(NVSDK_NGX_Feature_RayReconstruction)`; on failure it logs *"check if you have a valid nvngx_dlssd.dll or your driver is supporting DLSSD"* and writes `config["external"]["feature"]["supported"] = false`, so `slIsFeatureSupported(kFeatureDLSS_RR)` fails and the game shows no option. Then at init, `SuperSamplingDenoising.Available == 0` → "DLSSDContext cannot run". https://raw.githubusercontent.com/NVIDIA-RTX/Streamline/main/source/plugins/sl.dlss_d/dlss_dEntry.cpp
- **Log line we should be faking/feeding**: `NGX does not support feature SuperSamplingDenoising.Available` / `Optional DLSS feature is disabled` (real driver-side message on Linux). https://github.com/NVIDIA-Omniverse/ovrtx/issues/5
- **Old driver `_nvngx.dll` without DLSSD strings → option greyed out or feature dead** (Cyberpunk 2.0 + driver 550.09 vs 537.42, quoted in §3): https://forums.cdprojektred.com/index.php?threads/2-0-ray-trace-reconstruction-greyed-out.11118982/
- **Independent versioning**: "All these three DLSS features are using separate DLL files with their own versioning: 1. `nvngx_dlss.dll` 2. `nvngx_dlssd.dll` 3. `nvngx_dlssg.dll`" — dlss-swapper's feature request to *replace* RR DLLs (originally unsupported because RR model files were hard to source). https://github.com/beeradmoore/dlss-swapper/issues/162
- **No hard "dlssd version must equal dlss version" check exists in Streamline's source**: the plugin only reads `nvngx_dlssd.dll`'s `FileVersion` for display/OTA bookkeeping (lines 256–264, 1283) and never compares it to `nvngx_dlss.dll` (which the SR plugin reads the same way, `sl.dlss` `dlssEntry.cpp` line 260). https://github.com/NVIDIA-RTX/Streamline/blob/main/source/plugins/sl.dlss/dlssEntry.cpp
  - But mod/community packs deliberately ship matched cohorts — e.g. "DLSS DLLs: 310.9.16 (`nvngx_dlss / nvngx_dlssd / nvngx_dlssg / nvngx_dlssnr`)" https://github.com/danielblnc/DLSS-NR-on-AMD/issues/267 — and behavior genuinely differs per build:
  - `nvngx_dlssd` **310.9.1** introduced a Vulkan WRITE_AFTER_WRITE hazard that **310.7.0** does not (regression report against SL 2.14.1): https://github.com/NVIDIA-RTX/Streamline/issues/126
  - Release-build RR model crash: https://github.com/NVIDIA-RTX/Streamline/issues/116
  - Access violation inside `nvngx_dlssd.dll` at `NVSDK_NGX_SetInfoCallback`: https://github.com/NVIDIAGameWorks/bridge-remix/issues/13
- **Mismatched/partial copies do crash in practice**: users removed the RR/FG DLL quartet to cure crashes with DLSS in a shipping title (MH Stories 3, §4 link); conversely having the RR lib present on unsupported HW made Omniverse RTPT render black until the file was removed (§5 link).
- **Historical caveat**: when RR first shipped (DLSS 3.5), modders concluded it "cannot be retrofitted in the same way as we are doing with DLSS 2 (hot-swapping dll versions)" — the RR model is loaded by the *driver/NGX core*, not by the game-adjacent SR DLL. https://github.com/emoose/DLSSTweaks/issues/90

---

### Shortest actionable gate list for the port (all sourced above)
1. Game requests `kFeatureDLSS_RR` (1001) in `featuresToLoad` **and** ships `sl.dlss_d.dll` (+ `sl.common.dll`, `sl.interposer.dll`). (§1)
2. `slIsFeatureSupported(kFeatureDLSS_RR)` must not be pre-failed — driven by `minDriver ≥ 535.68`, `minGPUArchitecture ≥ TU100`, OS ≥ Win10. (§2)
3. `NVSDK_NGX_D3D12_GetCapabilityParameters` must yield `SuperSamplingDenoising.Available = 1`, which needs `_nvngx.dll`/NGX core that knows the RayReconstruction feature **and** a loadable `nvngx_dlssd.dll` of a build the driver accepts. (§2, §6)
4. Game's own UI conditions: DLSS/DLAA on + ray tracing on (e.g. CP2077 Overdrive) + adapter not filtered. (§4)
5. On Linux/Proton, skip Streamline's plugin probe and satisfy/forge the NGX capability query, nil or timeout-fake `GetFeatureRequirements(RayReconstruction)` as supported, and place a period-correct `nvngx_dlssd.dll` beside the exe. (§5, §6)