# FFX-seam front end for DLSS5-AMD-Linux — design (no implementation yet)

Date: 2026-11-01. Companion study: [wronggpu-re.md](wronggpu-re.md). Goal: put our existing
Vulkan chain (`src/vk/pipeline.cpp`, `shaders/`, packed weights from `tools/sr/sr_pack.py`)
behind an **AMD FFX API surface** so OptiScaler (and any FFX host) drives it via
`ffxCreateContext`/`ffxDispatch`, instead of via the NGX surface in `src/pe/nvngx.cpp` that the
Nov 1 Crimson Desert update no longer agrees with (90–124 s/frame as `nvngx.dll`; genuine NVIDIA
DLL = full speed).

Contract sources (MIT, in `/tmp/opencode/ffx-api/`, to be vendored under `src/ffx/ffx-api/`):
`ffx_api.h`, `ffx_api_types.h`, `ffx_upscale.h`, `ffx_api_dx12.h`. Consumer behaviour verified
against OptiScaler v0.9.4 sources in
`reference-nr-vulkan/artifacts/ref/OptiScaler-NR-src/OptiScaler/`.

---

## 1. Architecture

```
                     (FFX host: OptiScaler's FFXFeatureDx12, or the game directly)
                                        |
   amd_fidelityfx_upscaler_dx12.dll  ← our new PE  (src/ffx/ffx_front.cpp + pipeline.cpp)
   staged next to OptiScaler's own dlls (harness) / a non-colliding name in the game dir (later)
                                        |
   internal: FfxApiResource → VkImage + VkImageView, command list → vkd3d VkCommandBuffer
                                        |
   src/vk pipeline (UNCHANGED):  sr_gather → enc0 → stages → dec0 → sr_post (+ exposure/DSR)
                                        |
   weights: tools/sr/sr_pack.py trees via DLSS5_AMD_WEIGHTS (unchanged)
```

- **Reuse, unchanged**: the frame chain, shader set, weight packing, output-format contract
  (`OUT_F16`), and the whole `src/vk/` layer. The bridge is a new *surface* in front of
  `build_chain_from_*` + per-frame eval, exactly parallel to how `nvngx.cpp` wraps them.
- **New**: `src/ffx/` — `ffx_front.cpp` (the five exports + provider label), a thin
  `ffx_d3d12.cpp` for resource acquisition (vkd3d interop, same `ID3D12DXVKInteropDevice` /
  `ID3D12GraphicsCommandListExt` dance `nvngx.cpp` already does at `src/pe/nvngx.cpp:24-56`,
  `:1202-1214`, `:1719`), and a vendored copy of the four MIT headers.
- **NGX path untouched**: `nvngx.cpp` stays as-is, keeps building, keeps being the default
  package. The FFX DLL is an additional build output, additive only (blast radius §7).

### 1.1 Why a D3D12 FFX front end when our chain is Vulkan

Same answer as the NGX port: under Proton the game is D3D12; vkd3d-proton resolves every
`ID3D12Resource` handed to us into a VkImage and every `ID3D12GraphicsCommandList` into a
VkCommandBuffer via `ID3D12DXVKInteropDevice::GetVulkanResourceInfo` /
`ID3D12GraphicsCommandListExt::GetVulkanHandle` — the exact machinery `nvngx.cpp` already uses and
proves in the harness every run. The FFX surface just delivers the *same three kinds of objects*
(game resources, a command list, a device) in a different envelope:

| FFX hands us | NGX hands us | what we do |
| --- | --- | --- |
| `ID3D12Device*` (in `ffxCreateBackendDX12Desc.device`, chained into the create desc) | device via `NVSDK_NGX_D3D12_Init*` | QueryInterface `ID3D12DXVKInteropDevice` → VkInstance/PhysicalDevice/Device (nvngx.cpp:1719) |
| `void* commandList` in the dispatch desc | command list per evaluate | `GetVulkanHandle` → VkCommandBuffer (nvngx.cpp:25-38) |
| `FfxApiResource.resource` = `ID3D12Resource*` | resource pointers per slot | same fetch as nvngx.cpp:1205-1214: GetVulkanResourceInfo → (VkImage, offset), GetVulkanImageLayout, format from `description.format` |

### 1.2 `FfxApiResourceDescription` construction

For each input/output we build the description from the D3D12 resource's own `GetDesc()` plus the
FFX format mapping — `ffx_api_dx12.h` ships `ffxApiGetResourceDX12()` and
`ffxApiGetSurfaceFormatDX12()` as `static inline` helpers precisely for this; we re-implement them
in our file (they are MIT, keep the header provenance comment) rather than including D3D12-heavy
headers twice. Points that our chain cares about:

- **`format`**: use `ffxApiGetSurfaceFormatDX12(desc.Format)`; then resolve typeless exactly like
  OptiScaler does for FSR4 (`FSR31Feature.h:37-70 ffxResolveTypelessFormat`,
  `FFXFeature_Dx12.cpp:421-428`): `R16G16B16A16_TYPELESS→FLOAT`, `R32_TYPELESS→FLOAT` (depth),
  `R10G10B10A2_TYPELESS→UNORM`, `B8G8R8A8_TYPELESS→UNORM`, etc. Our in-game MV case is
  `DXGI_FORMAT_R32G32_FLOAT` (16) → `FFX_API_SURFACE_FORMAT_R32G32_FLOAT` — one of the formats the
  bridge already decodes (`nvngx.cpp:1147-1153`, slot table `:1304`: RG32F is a supported MV
  alternative).
- **`usage`**: depth resources get `FFX_API_RESOURCE_USAGE_DEPTHTARGET` (the helper derives it
  from `D16/D32/D24S8` formats); colour/MV/output typically `READ_ONLY` + `UAV` where
  `D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS` is set (the output slot needs it — Opti passes
  `FFX_API_RESOURCE_STATE_UNORDERED_ACCESS` for output, `FFXFeature_Dx12.cpp:311-313`).
- **`state`**: pass through what the host declares (`COMPUTE_READ` for inputs,
  `UNORDERED_ACCESS` for output); our bridge barriers on the Vulkan side exactly as the NGX path
  does today (`nvngx.cpp` image_barrier, `:560`).
- **Buffers**: `type = FFX_API_RESOURCE_TYPE_BUFFER`, `size/stride` from Width/Height — only
  relevant if we later expose the exposure scalar as a buffer; not needed for v1 (§4.4).
- **The exposure resource** is optional; when absent we use the chain's own exposure path
  (`DLSS.Exposure.Scale`-equivalent = 1.0 and the auto-exposure luma/reduce passes we already
  run), when present it is a 1×1 R32F (`exposure texture is not R32F` is also WrongGPU's
  validation; same constraint suits our `sr_exposure` decode).

### 1.3 Load path: which OptiScaler slot we occupy

OptiScaler v0.9.4 keeps two module slots per effect family (`proxies/FfxApi_Proxy.h`):

- **main** (`main_dx12`): probes `amd_fidelityfx_loader_dx12.dll` then
  `amd_fidelityfx_dx12.dll` from `optiPath` (= `Config::MainDllPath`, ini
  `[Libraries] OptiDllPath`), or `[Libraries] FfxDx12Path` override, via
  `Util::LoadProxyLibrary` (`Util.cpp:782-842`: GetModuleHandle-if-already-loaded → override
  path → optiPath dir → bare name → dll search path). If nothing of its own loads, it *falls
  back to `main_dx12_hooked`* — the module the game itself loaded (`:250-255`).
- **SR** (`upscaling_dx12`): probes `amd_fidelityfx_upscaler_dx12.dll` the same way
  (`:368-386`); the game-loaded module is captured as `upscaling_dx12_hooked` and — key line —
  `InitFfxDx12_SR(HMODULE)` is invoked with the *hooked* module when Opti captures one, in which
  case **`upscaling_dx12.dll` is NOT set from a file probe** (`:376-386`: the file probe only
  runs `if (upscaling_dx12.dll == nullptr)` … and the hooked branch fills
  `upscaling_dx12_hooked.dll = module` first, so the probe then loads Opti's own copy only if
  `LoadProxyLibrary` still finds one — see below for the collision case).
- Routing (`D3D12_CreateContext`, `:1182-1233`): an upscale-type context goes to
  `upscaling_dx12` when that dll exists, else to `main_dx12`. `IsSRReady()` =
  `(main_dx12 && !isLoader) || upscaling_dx12` (`:126-135`).

Crimson Desert ships the **real** `amd_fidelityfx_upscaler_dx12.dll` (28 761 864 B) in `bin64`
(verified by `ls -la`; also `amd_fidelityfx_dx12.dll` and `amd_fidelityfx_loader_dx12.dll`,
26 376 B each = FidelityFX SDK 2 loader stubs). Consequences:

1. **We must never be named `amd_fidelityfx_upscaler_dx12.dll` inside the game folder** — we
   would shadow AMD's real FSR4 DLL (or collide with it in LoadLibrary order), and every FFX
   host (including the game's own FSR4 path) would suddenly see us. Same for
   `amd_fidelityfx_dx12.dll`/`amd_fidelityfx_loader_dx12.dll`.
2. **WrongGPU's answer** (its DLL *is* named `amd_fidelityfx_dx12.dll`) works because their
   installer renames the game's originals to `*.wronggpu-original.dll` and their front end is a
   *complete* stand-in including AMD-proxy behaviour. We are not building that in v1 — we only
   serve upscale contexts and refuse/delegate everything else.
3. **The harness staging trick**: `Util::LoadProxyLibrary`'s *optiPath* is OptiScaler's own
   directory (`MainDllPath` points at Opti's folder — in the harness that is `$STAGE`, where
   Opti's `dxgi.dll` lives). Opti loads its companion DLLs **from its own directory first**.
   So staging our DLL *next to OptiScaler's dxgi.dll under a name Opti probes* makes Opti load
   *us* instead of the game's real DLL — with zero files added to the game folder:
   - harness: `$STAGE/amd_fidelityfx_upscaler_dx12.dll` = our DLL (today `$STAGE/$name` is a
     *symlink to the game's* real one, `run_harness.sh:157-167`; for the FFX cells we replace
     that one symlink with our DLL, or point `[Libraries] FfxDx12SRPath` at a copy).
   - Because `upscaling_dx12.dll != nullptr` (ours), `D3D12_CreateContext` routes upscale
     contexts to us; `main_dx12` stays the game's loader stub (26 KB, `isLoader=true`,
     harmless, `IsSRReady` already satisfied by the SR slot).
4. **In the game dir (later)**: add an ini override instead of renaming anything —
   `[Libraries] FfxDx12SRPath=<path>\dlss5-amd\dlss5-amd-ffx.dll` — Opti loads our DLL from that
   path (`LoadProxyLibrary` override branch, `Util.cpp:795-812`) and the game's real AMD DLLs are
   untouched. This is cleaner than WrongGPU's rename dance and fully reversible
   (`dlss5-amd-install.txt` records the ini edit; `remove` restores it).
   **HYPOTHESIS**: needs a live test whether Opti's *ini-read* of `FfxDx12SRPath` happens before
   the module probe (it does read config first at startup, `Config.cpp:884-885`, so it should).

Alternative slot we deliberately do *not* take in v1: `main_dx12` (`amd_fidelityfx_dx12.dll`).
Being a non-loader main dll makes Opti hook *us* with `ffx*_Dx12` detours when
`EnableFfxInputs=true` (default, `Config.h:758`) — i.e. our own five exports would be detoured and
`ffxCreateContext_Dx12` (`inputs/FfxApi_Dx12.cpp:193-285`) would translate our create into an NGX
feature creation *on top of our own NGX bridge*, a loop we want nothing of. The SR slot is the
clean one: Opti treats it as "the SR effect library" and calls it directly
(`D3D12_CreateContext` SR branch, `:1197-1205`).

### 1.4 What the host does around us (v0.9.4, SR path = `FFXFeatureDx12`)

Verified call sequence we must survive (`upscalers/ffx/FFXFeature_Dx12.cpp`):

1. ctor: `FfxApiProxy::InitFfxDx12()` (loads our dll), `IsSRReady()`;
2. `InitFFX`: `QueryVersionsDx12(Device)` → `ffxQuery(NULL, GET_VERSIONS)` twice (count, fill);
   `InitFlags()` builds `ffxCreateContextDescUpscale.flags` from NGX init flags
   (`FFXFeature.cpp:42-100`); create desc chain = `UPSCALE → BACKEND_DX12.device →
   [OVERRIDE_VERSION]`; `D3D12_CreateContext(&_context, &hdr, NULL)` (`:642-676`);
3. per frame `EvaluateInternal` (`:86-630`): builds `ffxDispatchDescUpscale` from NGX params
   (jitter, reset, renderSize, resources via `ffxApiGetResourceDX12`, MV scale from
   `MV_Scale_X/Y`, camera fields from `FSR.camera*` keys or config, preExposure), optional
   `ffxConfigureDescUpscaleKeyValue` for FSR3.1 tuning keys, then
   `D3D12_Dispatch(&_context, &hdr)`.
4. destroy: `D3D12_DestroyContext` (also tried on *all* modules, `:1307-1347` — our
   `ffxDestroyContext` must tolerate being called with a context that is not ours and return
   `FFX_API_RETURN_ERROR_PARAMETER`-style failure without crashing).

Also observed and handled: typeless resolution only for `major >= 4` (`:421`) — we must accept
*typeless* formats regardless of our claimed version and resolve them ourselves (our claimed
version is ours to choose; §4.6); the FSR4 autoexposure WAR (`CopyTextureRegion` to a smaller
color, `:131-186`) is host-side, invisible to us.

---

## 2. Per-frame contract: `ffxDispatchDescUpscale` → our chain

Field-by-field (header layout per `ffx_upscale.h`; our consumption per
`src/pe/nvngx.cpp:480-554` and `shaders/sr/sr_gather*.comp`):

| FfxDispatchDescUpscale field | type | our chain input | notes |
| --- | --- | --- | --- |
| `commandList` | `void*` (`ID3D12GraphicsCommandList*`) | VkCommandBuffer via `GetVulkanHandle`; the chain records into it | identical to NGX D3D12 path |
| `color` | `FfxApiResource` (render res) | `Color` slot → `g_col` via `sr_gather` | expect RGBA16F/RGBA32F family; typeless resolved |
| `depth` | `FfxApiResource` | `Depth` slot → `g_dep` | R32F / D32F / D32F_S8X24; `DEPTHTARGET` usage |
| `motionVectors` | `FfxApiResource` | `MotionVectors` slot → `g_mv` | R32G32_FLOAT (in-game) or RG16F; units via `motionVectorScale` |
| `exposure` | `FfxApiResource`, optional 1×1 R32F | exposure scalar pass-through | absent → chain's auto-exposure; present → feed `sr_exposure`/scale |
| `reactive` | `FfxApiResource`, optional | **ignored** (chain has no reactive input today) | log-once per context |
| `transparencyAndComposition` | optional | **ignored** | same |
| `output` | `FfxApiResource` (upscale res) | `Output` slot ← `OUT_F16` RGBA16F result | UAV state |
| `jitterOffset` | `FfxApiFloatCoords2D` | `fp.jx, fp.jy` (this frame's jitter) | same semantics as NGX `Jitter.Offset.X/Y`; sign per §5 |
| `motionVectorScale` | `FfxApiFloatCoords2D` | `fp.mv_scale_x/y` → gather push constants | **the big win: FFX carries the scale explicitly** — see §5 |
| `renderSize` | `FfxApiDimensions2D` | `s->rw/rh` | must equal the color texture's active size; else refuse like the subrect refusal |
| `upscaleSize` | `FfxApiDimensions2D` (optional, = maxUpscaleSize) | `s->ow/oh` | Opti fills it (`FFXFeature_Dx12.cpp:571-586`) |
| `enableSharpening`, `sharpness` | bool, float | **ignored** (no RCAS in chain); log value | could later map onto `sr_post`'s sharpening if we add one |
| `frameTimeDelta` | float ms | **ignored** (chain is stateless w.r.t. time today) | keep in log |
| `preExposure` | float | `fp.pre_exposure` | guard `<=0 / !finite → 1.0` (same as NGX path and WrongGPU) |
| `reset` | bool | `fp.reset` | history reset |
| `cameraNear` / `cameraFar` / `cameraFovAngleVertical` / `viewSpaceToMetersFactor` | floats | **ignored** (chain uses depth directly; no depth→view-space conversion today) | log once; they matter if we ever enable depth-linearisation |
| `flags` | `FFX_UPSCALE_FLAG_DRAW_DEBUG_VIEW / NON_LINEAR_COLOR_SRGB / _PQ` | SRGB/PQ: refuse-or-delegate (§4.6); DEBUG_VIEW: ignore | see below |

Create-time flags (`ffxCreateContextDescUpscale.flags`) and what we do:

| flag | our handling |
| --- | --- |
| `FFX_UPSCALE_ENABLE_HIGH_DYNAMIC_RANGE` | accept, no-op (chain is colour-space agnostic; input is HDR fp16 as delivered) |
| `FFX_UPSCALE_ENABLE_DISPLAY_RESOLUTION_MOTION_VECTORS` | accept; MV resource is then display-res — our gather's `mvhi`/`g_mvhi` path (`sr_gather_mv.comp`) exists for exactly this; select by comparing MV resource size vs renderSize (same detection the NGX bridge does implicitly by resource size) |
| `FFX_UPSCALE_ENABLE_MOTION_VECTORS_JITTER_CANCELLATION` | accept; MVs already contain jitter → chain expects that convention (NGX/SL in-game delivers jittered MVs and our chain consumes them with `mv_scale` — no change needed; log the flag) |
| `FFX_UPSCALE_ENABLE_DEPTH_INVERTED` | accept; our depth decode consumes [0,1] as-is from the resource — *verify against in-game depth convention* (open question §6) |
| `FFX_UPSCALE_ENABLE_DEPTH_INFINITE` | accept, no-op (no far-plane math in chain) |
| `FFX_UPSCALE_ENABLE_AUTO_EXPOSURE` | accept; if the host still passes an exposure texture, use it (WrongGPU rule: `exposure texture given with auto-exposure on: texture used`) |
| `FFX_UPSCALE_ENABLE_DYNAMIC_RESOLUTION` | accept; every resize goes through destroy+recreate (§3) |
| `FFX_UPSCALE_ENABLE_DEBUG_CHECKING` | accept; optional: emit one warnings pass |
| `FFX_UPSCALE_ENABLE_NON_LINEAR_COLORSPACE` | **refuse create** (`FFX_API_RETURN_ERROR_UNSUPPORTED`-ish → `FFX_API_RETURN_ERROR_RUNTIME_ERROR`); our network wants linear, we have no sRGB/PQ converter pass (WrongGPU delegates to AMD; we have no AMD to delegate to) |
| `FFX_UPSCALE_ENABLE_DEBUG_VISUALIZATION` | accept, no-op |

### 2.1 Jitter vs NGX jitter — one subtlety

NGX passes `Jitter.Offset.X/Y` as the *camera* jitter in render-pixel fractions; our chain stores
`fp.jx/jy` and the previous frame's pair (`pipeline.h:57-58`). FFX's `jitterOffset` is the same
quantity in the same unit (subpixel fraction of the render resolution). The only convention risk
is **sign**: Opti's own FSR31 path forwards NGX jitter verbatim into `jitterOffset`
(`FFXFeature_Dx12.cpp:105-107`), and the game's NGX jitter is what our chain already consumes
correctly in-game — so pass-through preserves the working behaviour. WrongGPU keeps
`jitter_sign_x/y = 1` defaults for their test matrix, i.e. they also treat the FFX convention as
identical by default. **Plan: pass through, no sign flip**, and keep the S6-style one-line log of
the first non-zero pair for the first live session.

### 2.2 Dispatch returns

- OK → `FFX_API_RETURN_OK`.
- Chain refusal (format/size/route) → `FFX_API_RETURN_ERROR_RUNTIME_ERROR` (Opti reacts by
  recreating the feature: `FFXFeature_Dx12.cpp:595-602`) — same recovery semantics the NGX path
  already exercises through `changeBackend`.
- Missing color/output → `FFX_API_RETURN_ERROR_PARAMETER` (matches WrongGPU's validation set:
  no color input / no output / render size 0 / output size 0).
- Unknown desc type → `FFX_API_RETURN_ERROR_UNKNOWN_DESCTYPE`.
- Not-taken-this-frame cases (e.g. native AA render==output when we refuse): decide —
  either record nothing and return OK (would need a fallback copy — WrongGPU bilinear-falls-back;
  we could reuse `sr_gather`-free bilinear later) or return
  `FFX_API_RETURN_ERROR_RUNTIME_ERROR`. v1: refuse with RUNTIME_ERROR; the harness cells render
  at DLAA-ish ratios where this never triggers.

---

## 3. Context lifecycle

`ffxCreateContext`:

- walk the `pNext` chain: collect `ffxCreateContextDescUpscale` (flags, maxRenderSize,
  maxUpscaleSize, fpMessage) and `ffxCreateBackendDX12Desc` (device). Missing BACKEND_DX12 →
  `FFX_API_RETURN_ERROR_PARAMETER` (WrongGPU's rule: `no CREATE_BACKEND_DX12 description (D3D12
  device) in the chain`); zero maxRender/maxUpscale → same.
- store an `ffxContext` = our context object (heap ptr). Keep `fpMessage` for warning emission
  (`FFX_API_MESSAGE_TYPE_WARNING`), wire it into the chain's log echo (one line per anomaly, not
  per frame).
- build the Vulkan side lazily on first dispatch or eagerly at create — eager is better: the NGX
  path already builds the chain at CreateFeature and the harness measures that separately. Create
  the chain from `maxRender/maxUpscale` (the chain is built for a size; resize = recreate, below).

`ffxDestroyContext`: release the chain, tolerate foreign/NULL contexts (`*context == nullptr →
OK`, Opti double-destroys through several modules).

**Resize policy**: OptiScaler creates a *new feature* on resolution change (its `IsInited` +
`changeBackend` machinery); a v0.9.4 FFX host has no in-place-resize call — there is no
"reconfigure size" desc in the contract. So: **destroy + recreate**. Park-and-reuse
(WrongGPU's `context_park_ms`) is a v2 nicety, not needed for correctness: recreation cost is the
same chain build the NGX path already pays on resize.

`ffxQuery` (all with `context == NULL` allowed):

- `GET_VERSIONS (4)`: two-call protocol; report exactly **one** version, name
  `"4.0.2 DLSS5-AMD (FFX)"`-style — *must parse* as `major.minor.patch …` because Opti calls
  `parse_version(versionNames[0])` (`FfxApi_Proxy.h:956,1017`). Pick major 4 semantics?
  **Decision**: claim `3.1.x`-compatible *behaviour* but a name whose leading triple is
  `3.1.<n>`; rationale: v0.9.4's host branches on `Version() >= {4,0,2}` for the autoexposure WAR
  and `>= {3,1,x}` for configure keys; claiming 3.1 keeps every 4.x quirk path off us. Do NOT
  claim 4.x until we test against a 4.x-expecting host. (WrongGPU claims `3.1.5 WrongGPU
  (Model M)` — same reasoning, visible in their string `wg_e9_strings.txt:3410`.)
- `GET_PROVIDER_VERSION (6)`: return the same id/name (validates a created context).
- `GETJITTERPHASECOUNT (0x00010004)`: compute the FSR formula `phaseCount = 2*ceil(2*ratio+0.5)`
  family — actually implement the published FSR equation for the given render/display widths
  (documented behaviour of the reference runtime; single small function).
- `GETJITTEROFFSET (0x00010005)`: Halton(2,3)-based offsets for (index, phaseCount), the standard
  sequence — implement from the published formula.
- `GETUPSCALERATIOFROMQUALITYMODE (2)` / `GETRENDERRESOLUTIONFROMQUALITYMODE (3)`: the five
  quality ratios (1.0/1.5/1.7/2.0/3.0) from `ffx_upscale.h`'s enum — trivial table.
- `GET_RESOURCE_REQUIREMENTS (0x0001000a)`: report `INPUT_COLOR|INPUT_DEPTH|INPUT_MV` required,
  `INPUT_EXPOSURE|INPUT_REACTIVEMASK|INPUT_TRANSPARENCYCOMPOSITION` optional.
- `GPU_MEMORY_USAGE (8/9)`: v1 → zeros (legal; host displays n/a), or rough sums from the chain's
  allocations later.

`ffxConfigure`:

- `GLOBALDEBUG1 (1)`: store `fpMessage`+level; emit our log lines through it too.
- `UPSCALE_KEYVALUE (0x00010007)`: accept the five FSR 3.1 tuning keys
  (`FVELOCITYFACTOR`, `FREACTIVENESSSCALE`, `FSHADINGCHANGESCALE`,
  `FACCUMULATIONADDEDPERFRAME`, `FMINDISOCCLUSIONACCUMULATION`) as **no-ops with OK** — the
  values tune FSR3's accumulation constants which we do not have; returning OK keeps Opti's
  per-frame configure churn from erroring (`FFXFeature_Dx12.cpp:489-545` fires these whenever the
  ini values change). Unknown keys → `FFX_API_RETURN_ERROR_UNKNOWN_DESCTYPE`.
- **`WG_RUNTIME_OPTIONS`-style private keys**: out of scope for the FFX surface — our runtime
  options go through the environment (`DLSS5_AMD_*`), not through FFX (WrongGPU smuggles theirs
  through the ini + create-options string; we already have a working env path).

---

## 4. MV-unit reconciliation (S6 finding ports over)

**The finding**: our chain consumes motion vectors in *plane-pixel* units (`prev = cur + mv`,
`tools/scenes/check_mv.py`); the gather decode multiplies the raw texel by host-provided
`mv_scale_x/y` (`sr_gather.comp:39-40,68-69`, commit e30a1ca). NGX carries the scale via
`MV.Scale.X/.Y`; Crimson Desert's SL writes `(1280,-720)` = UV-normalized flipped-Y (S6,
`nvngx.cpp:494-525`). FFX's `ffxDispatchDescUpscale.motionVectorScale` is *exactly* the same
concept, delivered in the same struct as the MVs — FSR3's contract is "multiply the MVs by this
scale to get render-pixel units", matching what we already do.

**Conclusion: the FFX path needs no new decode logic.** The host (Opti) reads
`MV_Scale_X/Y` from its NGX parameter bag and writes them into `motionVectorScale`
(`FFXFeature_Dx12.cpp:394-402`) — i.e. whatever the game's SL layer put in the NGX bag flows into
us unchanged, and our existing `mv *= scale` gather decode applies verbatim. The S6 guard
(non-finite/0 → 1.0) moves into the bridge's frame-param collection.

What FFX does **not** carry that we consume: a motion **bias/offset** (WrongGPU's runtime has
`mv_bias`/`mo` fields; NGX has none and FFX has none). Our chain has no bias either — nothing to
do.

### 4.1 Options ranked

1. **Pass-through `motionVectorScale` → `mv_scale_x/y` (RECOMMENDED, v1)**: zero new code in the
   chain; the bridge maps field→push-constant; the gather decode stays the single source of unit
   truth; harness coverage (`--mv-fmt rg32|rg16` with `mv_scale` fixture pairs,
   `opti_client.cpp:1048-1056`) transfers because the exe already writes the NGX keys that Opti
   forwards.
2. **Bake-in in the bridge (fallback if a host passes scale=1 with non-pixel MVs)**: the bridge
   could inspect the MV resource description and guess units — **rejected**: guessing violates the
   repo's "refuse instead of guessing" rule (subrect refusal precedent, `nvngx.cpp:543-547`), and
   FFX gives us no unit metadata beyond the scale.
3. **Pass-through with a debug escape hatch**: add a `DLSS5_AMD_FFX_MV_FORCE_SCALE` env override
   for diagnostics only; default absent → pure pass-through. Cheap, do it with option 1.

The gather decode differences in the FFX path: **none**. `display-resolution MVs` (flag) selects
which resource feeds `g_mv` vs `g_mvhi` (already implemented for the NGX path's size-based
detection); `motionVectorScale` multiplies after `san_mv` exactly as today
(`sr_gather.comp:68-71`).

---

## 5. Non-linear colour, HDR, depth

- `NON_LINEAR_COLORSPACE` (create) and `NON_LINEAR_COLOR_SRGB/_PQ` (dispatch): refuse create /
  return RUNTIME_ERROR on dispatch. Our chain's input contract is linear fp16 colour (in-game
  SL delivers linear; verified S4/S6 sessions). WrongGPU delegates these to AMD; we have no AMD
  to delegate to, so refusal + one clear log line is the honest behaviour. (If the game ever
  flips this flag, that is a *new converter pass* — out of scope, logged as a risk §6.)
- HDR: no-op accept.
- Depth inverted/infinite: no-op accept; the depth convention question is §6's open question.

---

## 6. Risks / open questions

| # | risk / question | mitigation / answer path |
| --- | --- | --- |
| 1 | vkd3d-proton `ID3D12GraphicsCommandListExt::GetVulkanHandle` inside a *non-NGX* caller: the NGX path gets its list from the NGX evaluate call; the FFX path gets it from `ffxDispatch`. **Same object type, same interop call** — low risk, but must be proven in the harness before any game attempt. | harness FFX cell (§8) — first thing it exercises. |
| 2 | Barrier states: FFX declares resource *states*; we convert state→VkImageLayout like the NGX bridge does. `UNORDERED_ACCESS` output ↔ vkd3d's UAV layout mapping is already exercised by the NGX path. | reuse `nvngx.cpp`'s state→layout code (factor it into a shared TU later; v1 can duplicate). |
| 3 | Opti hooks its *main* slot only; but `EnableFfxInputs` (default true) also detours `upscaling_dx12_hooked` **when the game itself loaded that module** (`FfxApi_Proxy.h:451-491`). In the game dir the game loads its real AMD DLL → Opti detours *AMD's* dll, not ours (ours came via LoadProxyLibrary override, never hooked — `hooked` flag only set on the hooked slot). In the harness, the "game-loaded" module is the *game's real* upscaler dll only if something loads it; our staged DLL is loaded via LoadLibraryExW by Opti itself → lands in the `upscaling_dx12.dll` (non-hooked) slot. | verified by reading both `InitFfxDx12*` bodies; confirm in harness log (no `fsrHooks` line for our module). |
| 4 | `FSR4ModelSelection::Hook` byte-scans our DLL on load (`FfxApi_Proxy.h:389`, pattern `83 F9 05 0F 87`): finds nothing → no-op. Also `isLoader` (file size < 1 MB) — our DLL will be multi-MB → treated as a real SDK dll → `IsSRReady` true via SR slot. No action. | harness log check. |
| 5 | Version-name parsing: our chosen name must survive `feature_version::parse_version`. | unit-check in the skeleton's self-test (log the parsed triple). |
| 6 | Depth convention (reversed [1..0]?) in Crimson Desert via the FFX path: NGX and FFX *both* receive the same game depth; our chain consumes it unchanged today. But Opti's `InitFlags` derives `DepthInverted` from its own game heuristics, not from us — if its heuristic disagrees with the NGX path's, behaviour could differ between dlss and ffx cells. | compare `DepthInverted` flag in harness logs between cells; the chain itself never flips depth, so a mismatch shows up as PSNR loss in the ffx cell, not a crash. |
| 7 | Jitter phase: does the *host* generate jitter, or do we? In Opti the **game/host** supplies jitter (NGX params); `GETJITTERPHASECOUNT/OFFSET` queries are for hosts that want the upscaler to generate it. Crimson Desert via SL generates its own. | implement the queries correctly but never *rely* on them. |
| 8 | Our chain needs the render↔output sizes at create; `upscaleSize` may be 0 at dispatch (= maxUpscaleSize). | derive `ow/oh` = `upscaleSize ?: maxUpscaleSize` at create; refuse dispatch when `upscaleSize != created` (same as NGX's size-locked contexts) until park/recreate lands. |
| 9 | Exposure: NGX carries `DLSS.Exposure.Scale` (float); FFX carries a *texture*. Our `sr_exposure` consumes the scale float. | bridge reads texel 0 of the R32F exposure texture (a tiny copy or a uniform push after a map — prefer: bind it as a 1×1 image and let `sr_exposure`'s first pass load it, **or** copy 4 B via a staging buffer at frame start). Decide by which is less invasive to `sr_exposure`; default plan: staging read at frame begin (the chain already has staging copy machinery). |
| 10 | `frameTimeDelta`/camera fields unused → some hosts check `viewSpaceToMetersFactor != 0`. | log-only. |

---

## 7. Blast radius

- **Additive only**: new `src/ffx/`, new optional build output `build/package/dlss5-amd-ffx.dll`,
  one *guarded* edit in `src/vk/build.sh` (`FFX_SKELETON=1` env gates the new TU; default build
  byte-identical).
- NGX path untouched: `nvngx.cpp` exports unchanged; package/install scripts unchanged until the
  FFX DLL proves itself in the harness.
- Harness: new *optional* staging branch + a new ini cell; existing cells keep running
  byte-identically (the companion-staging loop stays as is — the FFX cell only *replaces* which
  file one symlink points at, and only when `--ffx` is passed).
- bin64: **untouched** (read-only rule). No rename dance (that is WrongGPU's approach; ours is
  ini-override based).

---

## 8. Harness proof plan (tools/optiharness)

New cell `ffx1` (design; to be implemented after the skeleton compiles):

- **Stage**: current companion loop stages the game's real
  `amd_fidelityfx_upscaler_dx12.dll` into `$STAGE` (N5, `run_harness.sh:157-167`). The FFX cell
  replaces that one entry with **our** DLL at the same name — legal in the harness because
  `$STAGE` is OptiScaler's own directory (`optiPath`), where Opti looks *first*; the game folder
  is never modified. Alternative equivalent: stage ours as `dlss5-amd-ffx.dll` and add
  `[Libraries] FfxDx12SRPath=$STAGE/dlss5-amd-ffx.dll` to the cell ini — prefer this variant even
  in the harness, because it exercises the exact mechanism the game dir will use.
- **ini**: the `--ini` cell file gets
  ```
  [Upscalers] Dx12Upscaler=ffx
  [Inputs] EnableFfxInputs=false     ; do NOT let Opti detour other modules; we ARE the SR dll
  [Libraries] FfxDx12SRPath=$STAGE/dlss5-amd-ffx.dll
  ```
  (`Dx12Upscaler=ffx` maps to `Upscaler::FFX`, `OptiTypes.cpp:184-194`; `fsr31` also maps to FFX
  on DX12).
- **Expected log evidence** (what PASS looks like):
  - `OptiScaler.log`: `amd_fidelityfx_dx12.dll methods loaded!` or the SR-slot equivalent
    (`Loaded from ...dlss5-amd-ffx.dll`), `LoadResult: true`, then per-frame
    `FFXFeatureDx12::EvaluateInternal` DEBUG lines and **no** `changing backend to fsr21`
    fallback line (that is the N5 failure signature).
  - our log (`dlss5-amd-ffx.log` next to the exe — the bridge writes its own log file like
    `dlss5-amd-ngx.log`): `ffxCreateContext: UPSCALE maxRender=1280x720 maxUpscale=1920x1080
    flags=…`, one line per dispatch with jitter/mv_scale/renderSize (d5-diag style, 1 Hz),
    `eval #N OK`.
- **Gates** (adapted g1–g4; g5 n/a — no SL):
  - `g1-ffx`: OptiScaler.log contains the FFX-feature init lines *and* zero
    `changing backend to` lines (the provider really served the frame);
  - `g2-ffx`: our log shows create + `eval #i OK` for every frame, `mv_scale=(x,y)` matches the
    fixture's pair, fmt lines match the `--mv-fmt` expectation;
  - `g3-ffx`: OUT_NZ non-zero-fraction check unchanged (the output is ours, same checker);
  - `g4-ffx`: EVAL_MS/GPU_EVAL_MS medians recorded (frame 0 excluded, repo convention) —
    comparison table vs the dlss cell is the deliverable, not a threshold.
- **Run matrix** (first proof): 1 run, `--ini cell_ffx1.ini --mv-fmt rg32`, 12 frames. Then the
  A/B: same corpus through the *dlss* cell for the timing delta, and `--mv-fmt rg16` for the
  unit-variant. Only after all gates PASS twice in a row: consider the game dir.

---

## 9. Skeleton → implementation order (post-doc)

1. `src/ffx/ffx_front.cpp` skeleton (5 exports + label, stubbed OK/error returns, MIT header
   comment) — commit "ffx: skeleton …" (this is deliverable 3, done separately).
2. Vendor the 4 MIT headers into `src/ffx/ffx-api/` (with provenance headers) so the skeleton
   includes the real types; keep `FFX_SKELETON=1` build guard.
3. Resource acquisition: port `nvngx.cpp`'s interop/resource-fetch helpers into
   `src/ffx/ffx_d3d12.cpp` (same code, new surface); create-path builds the chain.
4. Dispatch path: frame params from the desc, gather/exposure wiring, barriers.
5. Harness cell `ffx1` + gates; measure vs dlss cell.
6. Game dir: ini-override install (`FfxDx12SRPath`), measure in Crimson Desert behind OptiScaler.
7. Fork follow-up (stage-2 polish): a first-class OptiScaler entry, tested as a parallel path.

   The ini-override route needs no OptiScaler modification, but OptiScaler v0.9.4 is also
   forkable at trivial cost — source is vendored in
   `reference-nr-vulkan/artifacts/ref/OptiScaler-NR-src/` (GPL-3.0; personal use
   unencumbered, publish the diff if ever distributed). A dedicated "DLSS5-AMD" menu entry is
   a 4-line first-class diff, replacing WrongGPU.asi's entire fragile pattern-patching
   apparatus (ImGui::Selectable confidence voting, runtime byte patches):

   1. `OptiTypes.h` — add `DLSS5AMD` to `enum class Upscaler` (OptiTypes.h:42-66).
   2. `OptiTypes.cpp:37 UpscalerDisplayName()` — `case Upscaler::DLSS5AMD: return "DLSS5-AMD";`.
   3. `menu/menu_common.cpp:546 AddDx12Backends()` — append `Upscaler::DLSS5AMD` to the
      `RenderUpscalerCombo` options list (this is the exact function WrongGPU.asi byte-patches).
   4. Backend routing — map `Upscaler::DLSS5AMD` to the FFX feature class with our library
       path default (or reuse the FFX path: the dll loaded is whatever `[Libraries]` points at).

   Real new work = building OptiScaler itself once (Proton-host build script; the vendored
   tree carries its build system). Tracking cost per OptiScaler release ≈ re-applying the
   4-line diff (mechanisable: `tools/optiscaler-fork/patch.sh`). Plan: land BOTH — the ini
   route first (works with stock Opti, proves the seam), then the fork for the named entry;
   the two are additive and independent, and the harness should gate them as separate cells
   (`ffx1` = ini route over stock Opti; `ffx2` = fork route) so a regression in either is
   attributable.
8. Two-path test plan (user-approved 2026-11-01: prepare both without interaction):
   `ffx1` = ini-override cell over STOCK OptiScaler v0.9.4 (dxgi.dll as shipped in bin64);
   `ffx2` = forked-Opti cell (our patched dxgi.dll carrying the `DLSS5AMD` entry, same ini
   minus the override since the fork's routing targets our dll directly). Both cells share
   the port dll + gates; the fork's build runs in a scratch copy of the vendored tree, never
   in place.

   The ini-override route needs no OptiScaler modification, but OptiScaler v0.9.4 is also
   forkable at trivial cost — source is vendored in
   `reference-nr-vulkan/artifacts/ref/OptiScaler-NR-src/` (GPL-3.0; personal use
   unencumbered, publish the diff if ever distributed). A dedicated "DLSS5-AMD" menu entry is
   a 4-line first-class diff, replacing WrongGPU.asi's entire fragile pattern-patching
   apparatus (ImGui::Selectable confidence voting, runtime byte patches):

   1. `OptiTypes.h` — add `DLSS5AMD` to `enum class Upscaler` (OptiTypes.h:42-66).
   2. `OptiTypes.cpp:37 UpscalerDisplayName()` — `case Upscaler::DLSS5AMD: return "DLSS5-AMD";`.
   3. `menu/menu_common.cpp:546 AddDx12Backends()` — append `Upscaler::DLSS5AMD` to the
      `RenderUpscalerCombo` options list (this is the exact function WrongGPU.asi byte-patches).
   4. Backend routing — map `Upscaler::DLSS5AMD` to the FFX feature class with our library
       path default (or reuse the FFX path: the dll loaded is whatever `[Libraries]` points at).

   Real new work = building OptiScaler itself once (Proton-host build script; the vendored
   tree carries its build system). Tracking cost per OptiScaler release ≈ re-applying the
   4-line diff (mechanisable: `tools/optiscaler-fork/patch.sh`). Plan: land BOTH — the ini
   route first (works with stock Opti, proves the seam), then the fork for the named entry;
   the two are additive and independent, and the harness should gate them as separate cells
   (`ffx1` = ini route over stock Opti; `ffx2` = fork route) so a regression in either is
   attributable.
