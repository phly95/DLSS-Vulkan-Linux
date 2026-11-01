# WrongGPU 0.5.5-ea — FFX-seam reverse-engineering (interface level)

Date: 2026-11-01. Scope: **interface-level study only** — export tables, string/RTTI grammar,
documented contracts, log formats. No decompilation-to-copy; nothing here may be lifted into our
source (see §9 ethics note). Materials: `/tmp/opencode/wg_r3e9.bin`, `wg_r3ea.bin`,
`wg_r3ed.bin` (PE sections of the three DLLs, extracted from `WrongGPU-Setup-0.5.5-ea.exe`),
string dumps `wg_e9_strings.txt` / `wg_ea_strings.txt` / `wg_ed_strings.txt` / `wgbin_strings.txt`,
AMD FFX SDK headers in `/tmp/opencode/ffx-api/`, WrongGPU's own docs in
`/tmp/opencode/WrongGPU/` (RELEASE-NOTES.md, wronggpu-notices.txt). OptiScaler-side claims cite
`reference-nr-vulkan/artifacts/ref/OptiScaler-NR-src/OptiScaler/...` (v0.9.4 sources).

Why this matters to this repo: the Nov 1 Crimson Desert update renders at 90–124 s/frame behind our
NGX surface (`src/pe/nvngx.cpp`) while the genuine NVIDIA DLL is full speed — the updated exe
disagrees with the NGX surface we present. WrongGPU 0.5.5-ea is a shipped existence proof of the
Plan-B architecture: present a *thin FFX API surface* (`ffxCreateContext`/`ffxDispatch`/…) to
OptiScaler and run your own network behind it. This note records what their seam looks like so
`docs/notes/ffx-seam-plan.md` can design ours.

---

## 1. The three DLLs and their roles

Measured with a static PE walk (export directory + section table); all RVAs are file-accurate for
the shipped binaries.

| file (scratch name) | internal name (export dir) | size | role |
| --- | --- | --- | --- |
| `wg_r3e9.bin` | `amd_fidelityfx_dx12.dll` | 2 182 144 B | **FFX front end.** Exports the five FFX API functions + `wgGetActiveProviderLabel`. Loaded by the host (OptiScaler) in the FFX/fsr31 slot; owns provider selection, the AMD-original stand-in, fallback, stand-in upscaler, overlay, and the D3D12 command-list split. |
| `wg_r3ed.bin` | `wronggpu_runtime.dll` | 24 689 152 B | **Network runtime.** 30 `wg_rt_*` / `wg_native_candidate_*` exports + `wgGetUpscalerBackend`; the Model M network executors (HIP, native D3D12, test), model container loader, capture/profiling. |
| `wg_r3ea.bin` | `WrongGPU.dll` (ships as `WrongGPU.asi`) | 910 336 B | **OptiScaler plugin (.asi).** `InitializeASI` + `WrongGpuAsiQuery`; locates OptiScaler's ImGui menu at runtime and patches it so WrongGPU appears in the *main* DX12 upscaler list (not only the FFX sub-box). |

PDB path in `wg_r3ea.bin` (strings, `wg_ea_strings.txt:1617`):
`R:\WrongGPU-clean\build\release\0.5.5-ea\frontend\WrongGPU.pdb` — the .asi is built from the same
tree as the front end ("frontend" directory), consistent with the notices file, which lists exactly
`amd_fidelityfx_dx12.dll, ffx_fsr3_x64.dll, ffx_fsr3upscaler_x64.dll` as their DLLs
(`wronggpu-notices.txt` §1).

### 1.1 `wg_r3e9` export table (front end)

Export directory RVA 0x1fcf40, 6 names (`amd_fidelityfx_dx12.dll`):

| export | RVA |
| --- | --- |
| `ffxConfigure` | 0x23a10 |
| `ffxCreateContext` | 0x23a50 |
| `ffxDestroyContext` | 0x23a90 |
| `ffxDispatch` | 0x23ad0 |
| `ffxQuery` | 0x23b10 |
| `wgGetActiveProviderLabel` | 0x5ffd0 |

The five `ffx*` exports are exactly the set OptiScaler's `FfxApi_Proxy` resolves
(`proxies/FfxApi_Proxy.h:275-295` main slot, `:430-443` SR slot). `wgGetActiveProviderLabel` is
their own addition — nothing in OptiScaler v0.9.4 resolves it (grep of the reference tree: no
hit), so it is for their own overlay/diagnostics and, possibly, for the .asi plugin
(`WrongGpuAsiQuery` suggests a query protocol between the plugin and the front end; the label
strings "WrongGPU (Model M)" / "WrongGPU (stand-in)" / "AMD FSR" / "Upscaling unavailable" live in
the front end, `wg_e9_strings.txt:3011-3015`). **HYPOTHESIS**: the label export is how the .asi
plugin shows the active provider in OptiScaler's menu without duplicating state.

Self-identification on load: `"WrongGPU FSR front end %s (%s) loaded as %s"` +
`"WrongGPU FSR front end unloaded"` (`wg_e9_strings.txt:2770-2771`) — the third `%s` is the module
name it was loaded under (they must tolerate being renamed; the "hosted by another upscaler mod
(loaded under its own name)" string at `:2768` confirms they are designed to be loadable inside
another mod's module list).

### 1.2 `wg_r3ed` export table (network runtime)

Export directory RVA 0x177cf80, 29 names (`wronggpu_runtime.dll`):

| export | RVA | | export | RVA |
| --- | --- | --- | --- | --- |
| `wgGetUpscalerBackend` | 0x76e60 | | `wg_rt_frame_io_timed` | 0x2aac0 |
| `wg_native_candidate_create` | 0x9f240 | | `wg_rt_frame_timed` | 0x2b5f0 |
| `wg_native_candidate_destroy` | 0x9f9c0 | | `wg_rt_info` | 0x2b750 |
| `wg_native_candidate_dispatch` | 0x9f6f0 | | `wg_rt_kernels` | 0x2b9b0 |
| `wg_native_candidate_error` | 0x9f990 | | `wg_rt_model_source` | 0x2b9e0 |
| `wg_native_candidate_sync` | 0x9f890 | | `wg_rt_notes` | 0x2b9c0 |
| `wg_rt_alloc` | 0x2a8d0 | | `wg_rt_recon_size` | 0x2a890 |
| `wg_rt_buffer` | 0x2b7a0 | | `wg_rt_stage_blob` | 0x2ba00 |
| `wg_rt_create` | 0x2a5d0 | | `wg_rt_sync` | 0x2b6f0 |
| `wg_rt_create_external` | 0x2a8b0 | | `wg_rt_timing_labels` | 0x2ace0 |
| `wg_rt_destroy` | 0x2b520 | | `wg_rt_upload` | 0x2a990 |
| `wg_rt_download` | 0x2b940 | | `wg_rt_upload_inputs` | 0x2b580 |
| `wg_rt_error` | 0x2b550 | | `wg_rt_upload_motion_hi` | 0x2b5a0 |
| `wg_rt_frame` | 0x2b5d0 | | `wg_rt_frame_external` | 0x2b450 |
| `wg_rt_frame_io` | 0x2a9f0 | | | |

### 1.3 `wg_r3ea` export table (the .asi plugin)

Export directory RVA 0xd3b70, 2 names (`WrongGPU.dll`): `InitializeASI` @ 0x12ac0,
`WrongGpuAsiQuery` @ 0x12e40.

---

## 2. How the front end presents to the host (OptiScaler)

### 2.1 Which slot it occupies

WrongGPU's DLL is literally named `amd_fidelityfx_dx12.dll` (export dir name, measured above), i.e.
it takes OptiScaler's **main** FFX slot: `FfxApi_Proxy.h:211` probes
`amd_fidelityfx_loader_dx12.dll` first, then `amd_fidelityfx_dx12.dll`
(`reference-nr-vulkan/.../proxies/FfxApi_Proxy.h:205-247`, `Util::LoadProxyLibrary` at
`Util.cpp:782-842`). OptiScaler's `IsLoader` test is `file_size < 1 MiB`
(`FfxApi_Proxy.h:95-101`); WrongGPU's front end is 2.2 MB, so it classifies as a *non-loader* main
DLL → `IsSRReady()` = true (`FfxApi_Proxy.h:126-135`) and `FSR4ModelSelection::Hook` runs against
it — that hook is a byte-pattern scan (`"83 F9 05 0F 87"`,
`fsr4/FSR4ModelSelection.cpp:139-141`) that simply finds nothing in a DLL without FSR4 model
exports, so it is harmless. This is the whole compatibility trick: be a real, fat
`amd_fidelityfx_dx12.dll` and OptiScaler treats you as "the FSR4 SDK dll".

Their RELEASE-NOTES confirm the second entry point too: *"OptiScaler v10 nightlies: the WrongGPU
entry in OptiScaler's main list is not available yet; choose WrongGPU in OptiScaler's FFX Upscaler
list"* (`RELEASE-NOTES.md` "Known issues") — i.e. the same binary also serves the
`amd_fidelityfx_upscaler_dx12.dll` slot (`FfxApi_Proxy.h:368`) when staged under that name. One
binary, either slot, same five exports.

### 2.2 Provider/version enumeration — how "WrongGPU (Model M)" lands in OptiScaler's menu

`ffxQuery(NULL, FFX_API_QUERY_DESC_TYPE_GET_VERSIONS)` (type 4, `ffx_api.h`) is what fills
OptiScaler's `ffxUpscalerVersionIds/Names` (`upscalers/ffx/FFXFeature_Dx12.cpp:646`,
`proxies/FfxApi_Proxy.h:932-976`: two calls — count, then fill; then
`version.parse_version(versionNames[0])`). The front end's answer strings
(`wg_e9_strings.txt:3504-3513`):

- `; AMD's versions: ` — it appends the versions of AMD's *real* DLL (loaded as their
  `OriginalApi`, see §2.3) after its own;
- `3.1.5 WrongGPU (Model M)` (`:3410`) — its own first version entry (name grammar
  `<major>.<minor>.<patch> <display name>`, which `feature_version::parse_version` parses);
- `WrongGPU, then AMD's upscaler versions` (`:3513`) — the ordering rule;
- `the AMD FidelityFX DLL listed no upscaler versions for this device` / `no usable AMD FidelityFX
  DLL (see the "AMD provider" lines above)` (`:3508-3509`) — degraded answers.

`versionId=%s versionName="%s"` (`:3740`) is their log grammar for the table. A version id that
belongs to AMD routes the context to the original DLL; WrongGPU's own id routes to the network:

- ` the host chose ` / `, which WrongGPU did not list; WrongGPU upscales` (`:3447-3448`),
- ` note: the host asked for upscaler version ` (`:3449`),
- `OptiScaler plugin: AMD FSR selected` / `OptiScaler plugin: AMD FSR selected in the host's
  upscaler list` (`:3441,3449`) — when the host's *main-list* choice means "plain AMD FSR",
  the front end forwards instead of serving.

### 2.3 The `OriginalApi` (AMD's real DLL) and the provider probe

RTTI (`wg_e9_strings.txt:4525,4556,4572`): `wgfsr::OriginalApi`,
`wgfsr::ProviderCandidate`, `wgfsr::ProbeResult`, and `wgwin::CompareApi` returning
`OriginalApi*` from a `std::string` (a module *path*). The log grammar around it
(`wg_e9_strings.txt:2756-2768, 3506-3512`) reconstructs the whole selection procedure:

- candidate labels probed: `the game's original AMD DLL (renamed)`,
  `the game's AMD FidelityFX SDK upscaler DLL`, `the game's amd_fidelityfx_dx12.dll`,
  `the game's AMD FidelityFX SDK loader DLL`, `AMD's FSR 4 upscaler DLL next to WrongGPU`,
  `the FSR 4 DLL the installer found on this PC`, `the installer's private copy of AMD's DLL`
  (`compare_dll`), `setting compare_dll` (`wg_e9_strings.txt:3555-3567`);
- each candidate is sanity-checked: `@usable / not there / a WrongGPU DLL / not loaded /
  redirected by another module / hooked by another module / no FidelityFX API exports`
  (`:3545-3552`) — "a WrongGPU DLL" catches the case where the name they want to load actually
  resolves to *their own* file (`ERROR: %s is a copy of WrongGPU's own DLL, not AMD's; it is not
  used`, `:2758`);
- the winner is loaded by **explicit path** (`AMD's original DLL loaded: %s, file version %s, %s`,
  `:2766`), its five exports resolved (`ffxCreateContext … ffxDispatch`, `:2759-2763`,
  `all five FidelityFX exports found` / `ERROR: some FidelityFX exports are missing`,
  `:2764-2765`), and extra exports it has are noted (`note: AMD's DLL has exports WrongGPU does
  not provide:%s`, `:2767`);
- result log: `AMD provider: chose %s` (`:3592`) and
  `AMD provider for the host's upscaler list: %s` (`:3506`) — the string the task brief cites.

`wronggpu-notices.txt` states the contract side of this: AMD's own DLLs *"stay in the game folder
(renamed `*.wronggpu-original.dll`) and are neither modified nor redistributed"* — so their
installer renames the game's originals and the front end finds them under those names. This is the
non-colliding pattern worth copying (see ffx-seam-plan §3).

### 2.4 Dispatch interception and fallback (their "seam" behaviour)

The front end is a full stand-in for AMD's DLL, not a wrapper: every FFX call is logged
(`ffxCreateContext context=…`, `ffxDispatch context=…`, `ffxQuery context=…`,
`wg_e9_strings.txt:3496-3520`) and then routed by state machine:

- `upscaler = network` (default): the context is served by the runtime; on create/frame failure →
  `fallback_to_amd = 1` hands the context to AMD's FSR (`; forwarded to AMD`,
  `upscaler context goes to AMD FSR`, `now runs on AMD FSR for the rest of its life`,
  `:3468-3480`);
- `create_wait_ms` (default 3000): while their network is still starting (model load), AMD's FSR
  upscales meanwhile (`" is still starting after "`, `" started after "`,
  `; it upscales from this frame on`, `:3481-3486`);
- per-frame flips are allowed: `: AMD FSR upscales this frame -> ` and
  `: the stand-in upscales this frame (` (`:3535-3537`) — the strings the task brief cites;
- `WG_ERROR_NOT_READY` frames get the stand-in (`upscaler not ready this frame (WG_ERROR_NOT_READY):
  stand-in for this frame`, `:3526`);
- history-reset frames are returned to WrongGPU after N frames
  (` frame(s) it could not take (history reset)`, `: WrongGPU upscales again after `, `:3487-3488`);
- a compare mode duplicates the dispatch to a second DLL
  (` compares with `, `; the game's AMD FSR compares`, `AMD comparison dispatch failed: `,
  `:3530-3533`) — `compare_key` toggles live in-game between WrongGPU's and AMD's output.

The five `WG_ERROR` codes, verbatim (`:3522-3527`): `WG_ERROR (unspecified failure…)`,
`WG_ERROR_UNSUPPORTED (this input combination is not supported: format, size or flag)`,
`WG_ERROR_DEVICE (the D3D12 or HIP runtime returned an error)`, `WG_ERROR_MEMORY (out of memory)`,
`WG_ERROR_PARAMETER (a required input is missing or invalid)`, `WG_ERROR_NOT_READY (not possible
this frame)`.

Reactive masks: `-> OK (WrongGPU: reactive mask not needed, nothing recorded)` and
`-> ERROR_UNKNOWN_DESCTYPE (WrongGPU)` (`:3521-3522`) — `ffxDispatchDescUpscaleGenerateReactiveMask`
is a no-op OK; unknown dispatch desc types get `FFX_API_RETURN_ERROR_UNKNOWN_DESCTYPE`.

Non-linear colour is delegated, not handled:
`non-linear (perceptual) colour (FFX_UPSCALE_ENABLE_NON_LINEAR_COLORSPACE): AMD FSR handles it
(spec 20, ANSWERS item 5)` (`:3451`) and at runtime
`non-linear (sRGB/PQ) colour flagged by the game: AMD FSR handles it` (`:3517`); the runtime side
says why: `the game reports non-linear color; the network expects linear color (spec 20)`
(`wg_ed_strings.txt:7571`).

### 2.5 Context parking (dynamic resolution without a hitch)

`wgfsr::UpscaleContext`, `FrontEnd::Park`, `PendingCreate` (RTTI
`wg_e9_strings.txt:4546,4561,4614-4618`). `context_park_ms = 3000`: a destroyed context's upscaler
stays warm and a new context with the *same parameters* takes it over
(`; its upscaler is parked for %u ms (context_park_ms) for a context with the same parameters`,
`:3474-3475`; takeover conditions enumerated at `:3477-3481`: same upscaler implementation, same
D3D12 device, same sizes, same flags, same options/model). Purely a UX optimisation; ours can skip
it (v1) and eat the recreate.

---

## 3. The .asi plugin (`wg_r3ea`) — the AddDx12Backends patch

Purpose (all strings `wg_ea_strings.txt:1234-1285`): when WrongGPU is installed as
`WrongGPU.asi` into an ASI loader next to OptiScaler, `InitializeASI` finds OptiScaler's own
module and patches its **main** DX12 upscaler menu so "WrongGPU" is a first-class entry:

1. Identify the host module: candidates are `the plugin itself`, `the caller of InitializeASI`,
   `its version resource names OptiScaler`, `a name OptiScaler is installed under`,
   `the process's executable`, `a system DLL` — with skip reasons (`): skipped, `).
2. Find `ImGui::Selectable` by voting: `could not identify ImGui::Selectable (need >= 3 agreeing
   labels and a unique winner; got count=` — i.e. several signature scans must agree and leave one
   winner: `ImGui::Selectable at RVA 0x`.
3. Find `AddDx12Backends`: `could not uniquely identify AddDx12Backends (candidate functions: ` /
   `AddDx12Backends at RVA 0x` — OptiScaler's function that appends the DX12 upscaler entries to
   the menu.
4. Verify the menu: `too few Selectable call sites in AddDx12Backends: ` /
   ` Selectable call sites in AddDx12Backends`, `ImGui::BeginCombo at RVA 0x`,
   `): OptiScaler's DX12 upscaler menu recognised` vs `): not OptiScaler's menu`.
5. Patch: allocate `a stub page within reach of OptiScaler` (`not hooked: could not allocate a
   stub page within reach of OptiScaler`), rewrite the Selectable *call sites*
   (`not hooked: no Selectable call site could be patched`,
   `hooked: %d of %zu Selectable call sites patched; preview relabel %s. "%s" is now an entry of
   OptiScaler's main DX12 upscaler list; selected: %s`). Labels involved: `FSR 3.X`, `FSR 3.X/4`,
   `FSR 2.1.2`, `FSR 2.2.1`, `XeSS w/Dx12`, `AMD (FSR 3.X)` — the entry relabelled is Opti's FFX
   one, and `BeginCombo not uniquely resolved; the closed-combo preview text will not be
   relabelled` shows they also fix the combo's collapsed label.
6. Persist the user's pick: an ini with `selection=` lines (`; (wronggpu or amd). Deleting this
   file selects WrongGPU.`) — `; WrongGPU OptiScaler plugin: the entry last selected in
   OptiScaler's upscaler list` — which the FFX front end then reads (the
   `OptiScaler plugin: AMD FSR selected` strings in the front end).

Graceful degradation: `not hooked: OptiScaler's DX12 upscaler menu was not recognised in %zu
candidate module(s) (no OptiScaler loaded, or this OptiScaler build differs). WrongGPU stays in
the FFX sub-box; nothing was patched.` — i.e. the plugin is purely cosmetic convenience; the FFX
sub-box route (ini `Dx12Upscaler=ffx` + version pick) works without it. **We do not need any of
this**: our harness selects the backend by ini, not by menu.

---

## 4. `wgGetUpscalerBackend` — the front end ↔ runtime contract

The front end loads `wronggpu_runtime.dll` from its own folder (`WrongGPU runtime %s not found: it
must be in the same folder as WrongGPU's FSR DLLs`, `wg_e9_strings.txt:3061`; DLL-dependency
diagnosis: `(the file exists, so a DLL it depends on is missing, e.g. the AMD HIP runtime
amdhip64_7.dll …)`, `:3063`). It calls the single export `wgGetUpscalerBackend` and validates the
returned table (`:3065-3076`):

- `asked for version %u` — version negotiation;
- `returned null`, `returned a table of interface version %u`, `%llu bytes, too small for
  version %u`, `returned a table of %u`, `returned a table without name, create, dispatch or
  destroy`;
- success log: `WrongGPU runtime upscaler "%s": interface version %u, flags 0x%x%s%s%s` where the
  flag suffixes include `(same-frame split)` (`:3073-3074`).

Runtime-side limits: `network runtime: the front end is older than interface version 2`
(`wg_ed_strings.txt:7515`) and `the front end offers no command-list split (interface version 2)`
(`:7523`) — interface 2 added the split services and grew `WgFrame`
(`WgFrame too small (front end older than interface 2)`, `:7701`).

**WgCreateInfo** (reconstructed from validation strings, `wg_ed_strings.txt:7518-7524`,
`:7560-7566`, `:7669-7690`): `d3d12_device` (checked `d3d12_device is not an ID3D12Device`),
max render / output sizes, `flags`, `options` = the `WG_RUNTIME_OPTIONS` string
(`options from WG_RUNTIME_OPTIONS: `, `WG_RUNTIME_OPTIONS is longer than 4095 characters and is
ignored`), plus (interface 2) the split-service callbacks — RTTI shows the exact callback
signatures (`wg_e9_strings.txt:4524-4528,4581-4589`):
`WgUpscalerBackend* (*)(const Config&, std::string&)` (backend factory),
`OriginalApi* (*)()` (AMD original accessor),
`WgFrontEndServices* (*)(const WgUpscalerBackend*, void*)`, and three split lambdas taking
`(uint64, const WgFrame&)`, `(uint64, uint64, WgResult)`, `(void*, WgFrame*)` — i.e. split
request, order-ticket completion, and frame hand-over.

**WgFrame** (reconstructed, `wg_ed_strings.txt:7701-7734` + the frame log grammar
`wgbin_strings.txt:960-963`):

```
frame=%llu R=%ux%u at (%u,%u) O=%ux%u
 jitter=(%.6g,%.6g) motion=raw*(%.6g,%.6g)+(%.6g,%.6g)
 exposure=%s(%.6g) pre_exposure=%.6g reset=%u
```

with validation errors: `color, depth, motion and output are required`,
`render size %u...` `active region outside the color or depth texture`,
`output size differs from the context's`, `frame not taken`, `invalid exposure or reset`,
`nonfinite jitter or motion transform`. The JSON-ish diagnostics line
(`wg_e9_strings.txt:2681-2696`) shows the same fields as the front end logs them:
`,"pre_exposure":` `,"exposure":` `],"exposure_mode":` `],"motion_offset":[` `],"motion_scale":[`
`],"jitter":[` `,"command_list_type":` `,"command_list_id":` `,"frame":` `{"context":`.

**`wgfsr::WgUpscalerBackend`** is the object the runtime hands back; the front end drives it with
create / dispatch / destroy (RTTI factory signatures above) and the `UpscaleStatus` callback
(`_Func_base@X AEBUUpscaleStatus@wgfsr@@`, `wg_e9_strings.txt:4567`).

### 4.1 The `wg_rt_*` surface (per-call semantics from names + strings)

- **Lifecycle**: `wg_rt_create` / `wg_rt_create_external` (external I/O route:
  `native direct resources require external_io, backend 3 and Model M`,
  `native D3D12 external I/O requires the private direct-resource route`,
  `wg_ed_strings.txt:6938-6941`) / `wg_rt_destroy`.
- **Memory**: `wg_rt_alloc`, `wg_rt_buffer` (`unknown buffer: `,
  `buffer unavailable on direct-resource route: `).
- **Frames**: `wg_rt_frame` (staged, HIP-style), `wg_rt_frame_external`,
  `wg_rt_frame_io` / `wg_rt_frame_io_timed` (the FrameIo struct — see below),
  `wg_rt_frame_timed`; `wg_rt_upload` / `wg_rt_upload_inputs` /
  `wg_rt_upload_motion_hi` (`wg_rt_upload_motion_hi needs the runtime option
  motion_outres=original`, `:7200`); `wg_rt_download`; `wg_rt_sync`.
- **Introspection**: `wg_rt_info` (the ready line),
  `wg_rt_notes` (`from the executor: ` / `not available from this executor`,
  `:7538-7539`), `wg_rt_kernels` (`network kernels: `), `wg_rt_model_source`,
  `wg_rt_stage_blob` (per-stage weight blobs), `wg_rt_recon_size`
  (`WG_RECON_SIZE` / `(option recon=output)`, `:7561-7563`), `wg_rt_error`,
  `wg_rt_timing_labels` (the `[0]..[4]` event labels quoted in §6.3).
- **Native candidates** (the private D3D12 route, `wg_native_candidate_*`): grammar at
  `wgbin_strings.txt:1029-1077`: `private candidate requires backend 3, device, explicit queue,
  build and supported flags`; `Model M native D3D12, supplied device/queue, direct textures` vs
  `Model M native D3D12, supplied device/queue, staged reference`; phase labels
  `native_direct_begin / native_staged_begin / direct_input_states / staged_input_conversion /
  direct_output_states / staged_output_conversion / native_direct_end / native_staged_end`;
  failure handling: `candidate cannot resume after an exception; synchronize and destroy it`,
  `candidate is poisoned by a prior recording/submission failure`,
  `native candidate quarantine: completion unknown: %s`; per-frame validation:
  `same-device input-ready fence and nonzero value required`, `frame extents outside configured
  capacity`, `nonfinite jitter or motion transform`, `unsupported private {color,depth,motion,
  output} format`, `active region exceeds an input texture`, `aliased input/output resources
  unsupported`, `exposure requires same-device R32_FLOAT texture`, `: requires matching single-mip,
  single-sample typed 2D texture`, `: incompatible resource flags`,
  `: unsupported or combined write state for supplied queue`,
  `: stated write state incompatible with resource flags`.

### 4.2 `FrameIo` (the structured per-frame descriptor)

Strings (`wg_ed_strings.txt:7125-7203`): `FrameIo.native_direct does not match the runtime's I/O
mode`, `native direct runtime requires FrameIo.native_direct`, `native direct frame does not
support raw buffer conversion, pitched/packed output or a motion_hi buffer`,
`native direct frame: motion_outres=original needs FrameIo.motion_hi_direct`,
`the post kernels write RGBA16F; the caller's output buffer is 3 x f32 (FrameIo.out_rgba16f)`,
`state.motion_history`. So `FrameIo` carries at least: `native_direct` (bool route selector),
`motion_hi_direct`, `out_rgba16f` (output layout selector), plus the shared frame scalars
(jitter / motion scale+offset / exposure / pre-exposure / reset / active region).

---

## 5. The executors (`wgi` namespace)

RTTI (`wg_ed_strings.txt:49344-49381`):

- `wgi::Executor` (base) with virtuals seen in lambdas: `prepare`, `prepare_device`, `record`,
  `dispatch`, `dispatch_timed`, `sync`;
- `wgi::D3d12Executor` — `prepare_device(ID3D12Device*, std::string*)` and
  `record(const Job&, ID3D12GraphicsCommandList*, bool*, std::string*)`;
- `wgi::HipExecutor` — `prepare(std::string*)`;
- `wgi::TestExecutor` — the `d3d12-test` self-test executor (its own HLSL
  `wronggpu_test_executor.hlsl`, queue/fence smoke test);
- device layer `wgrt`: `Device`, `HipDevice`, `EmuDevice`, `Dx12Device`, `NativeD3D12Device`,
  `NativeD3D12Ext` — i.e. the same network runs on HIP (amdhip64_7.dll) or natively on D3D12
  (ported kernels) or in an emulator (`/wg_emu.so`, `wg_emu__pack12.so`).

The choice: `executor dx12 (default on RDNA 4)` (`:7653`), `executor hip (default)` is the
non-RDNA4 default (`:7526` — the string `; the default on RDNA 4 (executor=hip selects the HIP
executor)`), forced either way with `executor=dx12|hip`. Reason lines for falling back to HIP are
each one log line (`RELEASE-NOTES.md` "When HIP is used instead"): non-RDNA4 GPU, sparkle guard or
capture enabled, frames the D3D12 path cannot take (e.g. native AA where render == output, or a
format the direct route refuses).

### 5.1 The D3D12 executor (the part most relevant to us)

From `wg_ed_strings.txt:7790-8090` (interface-level summary):

- **Borrowed device/queue validation**: `native D3D12 borrowed device requires a single GPU node`,
  `… requires the supported Radeon RX 9070/9060 adapter`, `… requires shader model 6.6`,
  `native D3D12 borrowed queue must be DIRECT or COMPUTE`,
  `native D3D12 borrowed device requires native_driver_probe (co=-, buffer, block/wave=32,
  args=p)`.
- **The driver-channel probe**: before trusting the queue it verifies *bit-exact* FP16/FP8
  behaviour through the driver: `; borrowed device/queue; exact FP16/FP8 driver-channel probe
  passed` (`:7981`), failure `native D3D12 borrowed driver-channel probe mismatch at float %s`
  (`:8093`), forced-failure env `WG_TEST_DRIVER_PROBE_FAIL`. This is their answer to "does the
  AMD driver's D3D12 compute path round exactly like HIP does" — a gate we should copy
  *conceptually* (probe, log one line, fall back) for our Vulkan-on-RDNA4 chain, which has the
  equivalent risk class.
- **Recording**: the network's kernels are recorded into *the game's command list* —
  `executor dx12: the network runs on the game's D3D12 device, recorded into the game's command
  list; its kernels read the game's colour, depth and motion and write the output texture (no
  input or output conversion pass, no HIP, no command-list split, no shared fences)` (`:7527`).
  Direct-resource rules: single-mip/single-sample typed 2D, output needs unordered access, no
  input/output aliasing, active-region bounds, explicit per-format refusals
  (`(color is not RGBA16F`, `depth is not R32F / D32F / D32F_S8X24 (view `,
  `motion is not RGBA16F / RG16F`, `exposure texture is not R32F`,
  `output is not RGBA16F`, `:7580-7590` area).
- **Recording discipline** (the invariant set we must respect too):
  `native D3D12: synchronization from record_pass callback is not allowed`,
  `native D3D12: submission from record_pass callback is not allowed`,
  `native D3D12: device operations from record_pass callback are not allowed`,
  `native D3D12 record_pass: recursive recording`, plus
  `native D3D12: the caller's command list with frame %u did not complete within 5 s (descriptor
  set reuse)` — completion is observed with a *completion marker* written by the caller's list
  (`end_external: the caller's list has no ID3D12GraphicsCommandList2 (completion marker)`),
  `SetEventOnCompletion`/`Signal` on fences, and a 5 s watchdog.
- **Arena**: one committed resource as a sub-allocated heap for the network's intermediates
  (`WG_D3D12_ARENA_MB` 16..4095, `the D3D12 arena is exhausted: `, `arena_offset`,
  `native D3D12 arena exhausted for %s`), `WG_DX12_BATCH` for batching kernels.
- **Manifest-driven kernels**: `/dx12/manifest.json` with per-kernel entries (group, gscale,
  root-signature kind, argument DWORD tables, sha256, struct args 4..228 B) validated at load
  (`native D3D12 manifest: …` strings); `WG_DX12_KERNELS` names subsets; `WG_D3D12_FUSION` /
  `WG_AB_FUSION` pick fused code-object variants (`pm,xe,mb,rv,xo,xd`), `WG_D3D12_NOBAR` /
  `WG_AB_NOBAR` drop barriers, `WG_D3D12_KSUB`/`WG_AB_KSUB` tweak kernel substitution.
- **Profiling**: `WG_D3D12_PROFILE` writes a JSONL capture (`{"type":"meta","version":1,…}`,
  `{"type":"gpu","session":…,"start_tick":…}`) with bottom-of-pipe timestamps; D3D12 timestamps
  in the game's list: `[0] start of the network -> [1] start of the first Swin kernel -> [2] start
  of dec0 -> [3] start of post -> [4] after the last kernel` (`:7547`).
- **Timing labels** (`wg_rt_timing_labels`, HIP flavor): `[0] start of the first kernel after the
  input wait (color_divide …) -> [1] start of the first Swin kernel (enc1) -> [2] start of
  coef_decoder -> [3] start of post_recon -> [4] end of an empty kernel after the last one
  (post_filter or scale_rgb); enc0 = [0]-[1] …, Swin = [1]-[2], dec0 = [2]-[3], post = [3]-[4]`
  (`wgbin_strings.txt:696`). Their internal stage names map 1:1 onto our chain's
  `sr_gather → enc0 → stages → dec0 → sr_post` split (colour-divide/exposure ≙ our gather+exposure
  passes; enc1/Swin ≙ our stages; dec0 ≙ dec0; post ≙ sr_post).

### 5.2 The HIP executor and its synchronization zoo

Strings `wg_ed_strings.txt:6773-6860` (HIP imports) + `:7246-7264` (interop HLSL) + `:7517-7544`:

- shares D3D12 buffers/fences with HIP via `CreateSharedHandle` → `hipImportExternalMemory` /
  `hipImportExternalSemaphore` (arena mode `share_path = arena|underlying|own`), or copies
  (`input_copy`, `output_copy`, `copy paths requested:`);
- `sync = inline | pipelined`; `ready_wait = fence | memory` (fence: HIP waits the shared input
  fence; memory: `the network waits for a flag in the arena` via `hipStreamWaitValue32`);
- `done_mode = fence | spin` (spin: the *game's queue* executes a `spin` shader that polls the
  done flag — `done_mode=spin (experimental): the game's queue polls a flag the network writes,
  not the done fence`; the HLSL is in the interop dump, `wg_ed_strings.txt:7256-7264`);
- `handover = fence | staged` (`handover=staged not available (needs model K, inline sync, the
  arena, no done_mode=spin, and the HIP executor); using the done fence` — staged handover is the
  Model K path, disabled in this build);
- `motion_outres = point | original` (`motion_outres must be original or point`) — how output-
  resolution motion vectors are consumed: `point` = the executor's `motionpt` pass samples the raw
  texel under each render pixel's centre into a render-size RG32F texture
  (`wg_ed_strings.txt:7299-7310`), `original` = the network's direct encoder reads the game's
  motion texture itself (D3D12 executor, `motion_hi_direct`);
- `sparkle_guard` + `sparkle_guard_thr` (`the sparkle guard runs on HIP only`,
  `sparkle guard on: brighten rate `, `sparkle guard not available with this executor`) —
  their name for the bright-pixel stability guard; `capture_dir`/`capture_frames` frame capture
  (HIP only), `watchdog` (stuck dispatches > 3 s), `stream_priority`, `timing_log`,
  `timing_stages`, `test_view`, `test_handles`, `test_keep`, `model_dir`, `kernels_dir`,
  `rocm_dir`, `model_format`, `pre_exposure`, `post_bands`, `post_tf`, `output_filter`,
  `d1d0_p12`, `dec_rdiv`, `dec_bdiv`, `swin_resident`, `post_tf_cpairs`, `post_diet`, `tf_vdiet`,
  `post_idx`, `post_trim`, `post_sel`, `model_k_dir`, `model_k_map`, `fp8_stages`, `k_fast`,
  `k_input`, `k_diag`, `k_head`, `input_tiled` (full list from the ini/option table,
  `wg_e9_strings.txt:3245-3295` and `wg_ed_strings.txt:7568-7580`).

The same frame-scalars log line exists at the executor level
(`run=%d value=%llu slot=%d rw=%d rh=%d jitter=… mv_scale=… mv_bias=… exposure=… pre_exposure=…
reset=%d exposure_mode=%d raw_inputs=%d color_pitch=%d depth_pitch=%d motion_pitch=%d
motion_channels=%d ms=%.9g,%.9g mo=%.9g,%.9g out_half_pitch=%d`,
`wg_ed_strings.txt:7407`) — note **`mv_scale` + `mv_bias`**: the runtime also carries a motion
*offset*, which the FFX dispatch desc does not have (FFX only scales; our NGX path likewise has no
bias key) — in FFX terms the bias is 0.

---

## 6. The D3D12 command-list split (their HIP same-frame trick)

Interface-level description from the front-end strings (`wg_e9_strings.txt:2789-2903`); this is
the machinery that lets the HIP executor run the network *between* the game's own commands on the
same frame:

- vtable hooks installed on the game's `ID3D12Device` (`CreateCommandList`/`CreateCommandList1`),
  `ID3D12GraphicsCommandList` (`Reset`, `Close`, `ResourceBarrier`, …) and `ID3D12CommandQueue`
  (`ExecuteCommandLists`) — `installing WrongGPU's D3D12 vtable hooks (%s) on device %p`,
  `installed (device %d, queue %d, command list %d vtables); split timeout %u ms…`;
- when the FSR dispatch list is submitted, `ExecuteCommandLists` is intercepted: the list is
  **split at the FSR dispatch point** (`split S%llu of list %p (%s): first half on W%u behind
  order ticket %llu, done %p = %llu; list reset with allocator %p, %d bindings restored; second
  half waits for resume %p >= %llu`) — first half runs on a WrongGPU-owned queue behind an order
  ticket, the network runs, then the second half resumes after the done fence
  (`split S%llu released in ExecuteCommandLists #%llu on Q%u, position %u of %u: %s wait for
  resume %p >= %llu, then the second half`);
- edge cases handled and logged: lists recording into their own vtables (`own table:`), another
  hook layer already present (`chained to them`), reset-without-submit (`the game did not submit
  the list`), pipeline-statistics query continuation (`split S%llu: continued %zu
  pipeline-statistics queries with exact result aggregation` — they even re-aggregate PSO queries
  across the split), refusals logged one-in-600;
- `WG_BACKEND_NEEDS_SPLIT` (`:2903`) is the runtime→front-end flag that says "I need the split to
  run on this frame", which only the HIP executor sets (the D3D12 executor explicitly does not:
  `no command-list split` in its capability line, `wg_ed_strings.txt:7527`).

Relevance to us: none for the Vulkan chain (vkd3d-proton is not hooked), but it explains why their
HIP path needs `WG_BACKEND_NEEDS_SPLIT` in the backend flags and why our front end, which runs the
chain on the game's command stream equivalently (via vkd3d interop), will *not* need any split
service — the same conclusion their D3D12 executor reached.

---

## 7. The model container (`model.wgcm`) vs our `tools/sr/sr_pack.py`

### 7.1 Their container (from the runtime's string table)

`wgbin_strings.txt:111-200` (all strings are loader diagnostics, i.e. the format's own vocabulary):

- magic/version: ` is not a WrongGPU model file`, `: unknown container version %u`, `: bad header`;
- named blobs: `enc0 enc1 enc2 dec2 dec1 dec0 post filter output_taps` + `.bin` — per-stage blob
  files inside the container, plus an `output_taps` entry;
- integrity: `digest`, `: bad tensor entry %s`, `%s does not match its stored digest`,
  `: stage %s does not match the map's digest`, `fails its integrity check`;
- shape/type validation: `: tensor %s has the wrong shape or type`, `: truncated index`,
  `%s missing`, ` has no stages`, `enc0 blob size mismatch`, `dec0 blob size mismatch`,
  `%s block size mismatch`, `no packer for stage %s`;
- scalars: `sigma1 sigma2 gamma1 gamma2` (with ` has a value above 127`,
  ` has a non-finite value` — i.e. u8 quantized scalars, sanity-checked);
- converter map: `converter map missing: `, `converter map damaged: `, `model data does not match
  this installation (converted with a different map): `, `model.wgcm needs the converter map
  (map_path)` — the container is produced by a *converter* whose map
  (`preset-M.v1.map.json`, `wgbin_strings.txt:799`) must match the installed runtime;
- output filter: `output_filter=original: the model file has no output filter weights (group
  filter / output_taps; written by the converter of spec 08 draft 2 when its source has them)`.

### 7.2 Ours (`tools/sr/sr_pack.py`)

`sr_pack.py WEIGHTS_DIR {M|L} OUT_DIR` writes `stage{0..9}.bin`, `enc0.bin`, `dec0.bin` and
`params.json` (per-stage fp8 bound flags + decoder scalars), sizes pinned to what NeuralHub's
loader requires (`STAGE_BYTES`, `ENC0_BYTES`, `DEC0_BYTES` in the script); blobs are raw packed
layouts (frag/f16/e4m3 packing, `dec0_s0/dec0_f0/dec0_f1` scalars) with *no* container, *no*
digests, and the mapping from slot → stage hardcoded from the NeuralHub RE
(`docs/sr/HOST-RE.md`).

### 7.3 Comparison

| aspect | WrongGPU `model.wgcm` | our `sr_pack.py` output |
| --- | --- | --- |
| packaging | single container file + external converter map (JSON) | loose per-stage `.bin` files + `params.json` |
| integrity | per-tensor + per-stage digests, map digest binding | none (trust by install path) |
| scalars | u8-quantized, range-checked (sigma/gamma) | f32 in `params.json` (`dec0_s0/f0/f1`) |
| stage naming | `enc0 enc1 enc2 dec0 dec1 dec2 post filter output_taps` | `stage0..9` (= enc1..dec1 layer table), `enc0`, `dec0` |
| variants | code-object variants selected by env (`WG_*_VARIANT`, fused/separate) | single layout per stage |
| provenance | converted from NVIDIA's nvngx_dlss model data by *their* converter (map-versioned) | extracted from nvngx_dlss.dll slots per NeuralHub's packer (RE'd) |

Take-away: nothing format-compatible — a port would re-pack, not re-use. The interesting *idea* is
the converter-map binding ("model data does not match this installation"): a digest that ties a
weight file to the kernel set that expects it. Our harness already effectively does this via
`DLSS5_AMD_WEIGHTS` pointing at a specific packed tree; a digest line in `params.json` would make
it self-checking (additive, low priority).

---

## 8. Recovered configuration surface

### 8.1 `wronggpu.ini` (front end; full default template recovered from
`wg_e9_strings.txt:3353-3412`)

Sections `[wronggpu]` and `[fsr]` ("The FSR front end (spec 20) reads the sections [wronggpu] and
[fsr]; other sections belong to other WrongGPU components"):

| key | default | meaning (verbatim condensed) |
| --- | --- | --- |
| `upscaler` | `network` | `network` (Model M via runtime) \| `amd` (forward to AMD, WrongGPU only logs) \| `standin` (diagnostic upscaler) |
| `runtime_dll` | `wronggpu_runtime.dll` | the runtime DLL next to the FSR DLLs |
| `fallback_to_amd` | `1` | on create/frame failure hand the context to AMD's FSR |
| `create_wait_ms` | `3000` | how long AMD FSR covers while the network starts |
| `context_park_ms` | `3000` | warm-park a destroyed context for reuse (0 = never) |
| `view` / `filter` / `jitter_compensation` / `warp_weight` / `debug_gain` | `upscale` / `bilinear` / `1` / `0.9` / `4` | stand-in controls |
| `jitter_sign_x/y`, `motion_sign_x/y` | `1` | convention switches for test 20-T2 (values 1 or -1) |
| `fsr3_route` | `auto` | `auto \| upscaler \| fsr3 \| off` — where the stand-in takes over FSR 3.0 component DLLs |
| `fsr3_amd_first` | `auto` | `auto \| always \| never` — whether AMD's upscaler runs first on the same frame (FG needs it) |
| `d3d12_hooks` | `auto` | `auto \| on \| off` — the command-list split hooks |
| `split_test` / `split_timeout_ms` | `0` / `2000` | stand-in through the split; submission timeout |
| `runtime_options` | (empty) | `key=value;…` passed to the runtime unchanged (WG_RUNTIME_OPTIONS) |
| `overlay` / `overlay_key` / `overlay_position` / `compare_key` / `compare_dll` | `on` / `insert` / `bottom-left` / `home` / (empty) | in-game overlay + A/B compare |
| `log_dispatches` / `log_max_mb` | `0` / `512` | per-frame dispatch logging / log cap |

Plus `[fsr]` keys seen as bare strings: `upscaler`, `standin`, `runtime_dll`, `fallback_to_amd`,
`jitter_sign_*`, `motion_sign_*`, `fsr3_route`, `fsr3_amd_first`, `d3d12_hooks`,
`create_wait_ms`, `context_park_ms`, `split_timeout_ms`, `runtime_options`, `overlay*`,
`compare_*`, `log_*` (`wg_e9_strings.txt:3195-3240`). `model=` lives in `[wronggpu]`'s runtime
option space and `model=K` is *ignored with a note* in this build (`note: model=K (` … `) is
ignored: Model K is not part of this build; Model M is used`, `:3296-3301`). Per-game quirks are
keyed on exe names: `crimsondesert.exe`, `cyberpunk2077.exe`, `controlresonant.exe` →
`ready_wait=fence` forced (`Model M game compatibility: shared-fence input synchronization
(ready_wait=fence); explicit runtime options retained`, `:3050`).

### 8.2 `WG_RUNTIME_OPTIONS` (runtime) keys

Recovered from the option table (`wg_e9_strings.txt:3245-3295`) and the WG_* env strings
(`wg_ed_strings.txt:6925-7052`): `executor, model_dir, kernels_dir, rocm_dir, model_format,
test_handles, test_keep, share_path, timing_log, timing_stages, watchdog, ready_wait, capture_dir,
capture_frames, sparkle_guard, sparkle_guard_thr, motion_outres, done_mode, spin_polls, handover,
input_copy, output_copy, test_view, stream_priority, post_bands, post_tf, output_filter,
pre_exposure, d1d0_p12, dec_rdiv, dec_bdiv, swin_resident, post_tf_cpairs, post_diet, tf_vdiet,
post_idx, post_trim, post_sel, model_k_dir, model_k_map, fp8_stages, k_fast, k_input, k_diag,
k_head, input_tiled` and env overrides `WG_RECON_SIZE, WG_D3D12_KSUB, WG_AB_KSUB,
WG_D3D12_XD_MAX_PIXELS, WG_D3D12_NOBAR, WG_DEC1_DEC0, WG_SWIN_FUSE_STAGES, WG_D3D12_FUSION,
WG_AB_FUSION, WG_AB_NOBAR, WG_AB_BLOCK, WG_SWIN_VARIANT, WG_ENC0_VARIANT, WG_DEC0_VARIANT,
WG_POST_KERNEL, WG_FILTER_KERNEL, WG_POST_PACK, WG_EXPOSURE_VARIANT, WG_POST_WAVE,
WG_SWIN_STAGE_VARIANTS, WG_POST_BANDS, WG_SWIN_RESIDENT, WG_POST_TF, WG_SWIN_MID, WG_SWIN_BNECK,
WG_DX12_ARENA_MB, WG_DX12_BATCH, WG_DX12_DEBUG, WG_DX12_KERNELS, WG_D3D12_ARENA_MB,
WG_D3D12_BATCH, WG_D3D12_PROFILE, WG_TEST_DRIVER_PROBE_FAIL, WG_RUNTIME_OPTIONS`.

---

## 9. Ethics / licence note

WrongGPU is **closed, free-of-charge software** ("WrongGPU's DLLs … contain no third-party code",
`wronggpu-notices.txt`); its FFX types come from AMD's MIT-licensed FidelityFX SDK headers, which
is exactly the licence basis our front end will use too. What this study used:

- **usable**: the *public* contracts — AMD's FFX API (MIT headers, in `ffx-api/`), OptiScaler's
  GPL-documented probing/proxy behaviour (we read its source, we don't copy it into our MIT/EULA
  tree either — we *call* the same documented seam), and the observable log/config grammar of a
  shipped third-party binary, recorded here as facts for interoperability.
- **not usable**: WrongGPU's implementation — its binaries, their decompilation, their HLSL
  (recovered as strings above for *documentation of behaviour only*), their menu-patching stubs,
  their model container/converter. Nothing from their code may be copied or transliterated into
  `src/ffx/`. Our front end is written from the AMD SDK headers and our own chain; where a
  behaviour is only observable through their logs (e.g. exact fallback timing), we implement our
  own equivalent rather than mimic internals.

In short: contracts yes, code no — same rule the repo already applies to NeuralHub's and NVIDIA's
material (README §1 step 3 licence note).

---

## 10. Quick fact index (for the seam plan)

- FFX surface = 5 exports; desc types exercised by Opti v0.9.4 SR path: create chain
  `UPSCALE(0x00010000) + BACKEND_DX12(2) [+ OVERRIDE_VERSION(5)]`, dispatch `0x00010001`,
  reactive `0x00010006` (no-op OK), configure `GLOBALDEBUG1(1)` + `UPSCALE_KEYVALUE(0x00010007)`,
  query `GET_VERSIONS(4)`, `GET_PROVIDER_VERSION(6)`, upscale ratio/render-res/jitter-phase/
  jitter-offset (`0x00010002..5`), GPU memory (`0x00010008/9`), resource requirements
  (`0x0001000a`).
- Version names must parse as `<major>.<minor>.<patch> …` (`parse_version`, applied to
  `versionNames[0]`).
- Non-linear colour = delegate or refuse; their network needs linear like ours.
- Exposure: they accept the 1x1 R32F texture (`exposure texture given with auto-exposure on:
  texture used`, `wg_e9_strings.txt:3519`).
- Sizes validated against create's maxima (`render size %ux%u exceeds the color texture %ux%u`,
  `render size exceeds maxRenderSize of create`, `:3520` area).
- `preExposure <= 0 or not finite: 1 used` (`:3519`) — same guard we already apply.
