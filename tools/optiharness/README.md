# tools/optiharness: timing + integrity of the DLSS port *with OptiScaler in the loop*

`tools/optiharness` stages a Windows-neutral harness directory under the git-ignored
`build/optiharness/harness/` - OptiScaler's `dxgi.dll` proxy, our `_nvngx.dll`, the shader and
weight paths - and runs its own `opti_client.exe` under Proton against the **game's compatibility
prefix**, so frames travel the same path they take in Crimson Desert:

    game-style D3D12 frames -> OptiScaler v0.9.4 (dxgi.dll middle layer) -> _nvngx.dll (Vulkan port)
                                                               -> RGBA16F output -> read back

without ever launching the game.  It exists to (a) measure clean ms timings with OptiScaler
present, (b) prove the upscaled output is real (not black), (c) reproduce both known in-game
issues from the harness alone, and (d) with `--sl`, boot the game's NVIDIA Streamline stack
headlessly and gate on its exact bootstrap lines (see "Streamline boot").

    bash tools/optiharness/run_harness.sh                                  # default --opti run
    bash tools/optiharness/run_harness.sh --direct                         # A/B: skip OptiScaler
    bash tools/optiharness/run_harness.sh \
        --nvngx "/path/to/_nvngx.dll.bak-mvfix" --expect-fail               # issue-1 reproduction
    bash tools/optiharness/run_harness.sh --sl                             # + boot the game's
                                                                           # Streamline stack (gate g5)
    SL_NVAPI_VER=57602 bash tools/optiharness/run_harness.sh --sl          # env lever (dxvk-nvapi
                                                                           # driver version, g5 data)
    bash tools/optiharness/run_harness.sh --corpus DIR [--frames N]        # V9 file corpus inputs
    bash tools/optiharness/run_harness.sh --mv-fmt rg16 --dep-fmt r32f     # proven-format bisect

Exit code 0 only when every gate passes; each run prints a `HARNESS SUMMARY` (script gates g1–g4,
plus g5 under `--sl`) and the exe writes its own `HARNESS SUMMARY` block plus `out/summary.txt`
and `out/eval_ms.csv`.

## What each number brackets (and why no line mimics the runner's)

| line | brackets | tool |
| --- | --- | --- |
| `EVAL_MS <i> <ms>` | QueryPerformanceCounter around the `NVSDK_NGX_D3D12_EvaluateFeature` call itself - OptiScaler's whole entry path (resource tracking, its hooks, whatever passes it records) down through our port's record | `opti_client.cpp` (QPC just before/after the call) |
| `GPU_EVAL_MS <i> <ms>` | two D3D12 `TIMESTAMP` query heap entries recorded immediately before and after the same evaluate on the harness's own command list, resolved after the fence wait, converted by `GetTimestampFrequency` - true GPU duration of everything the evaluate executes (OptiScaler passes + port copies + the chain).  Degrades to `GPU_EVAL_MS <i> n/a` on an unsupported HRESULT instead of failing | `opti_client.cpp` |
| `OUT_NZ <i> nz=.. min=.. max=..` | the read-back `out_<i>.f32`: fraction of pixels with any non-zero channel, plus global min/max | `opti_client.cpp` |
| port `recorded in N ms` | the port's own host-side stamp inside `record_bridge` - **parsed only**, at exit, into `PORT_RECORD_MS` (it uses `GetTickCount64` ms granularity, nvngx.cpp:241, so most frames honestly read `0 ms`) | `opti_client.cpp` `read_port_log()` |

**No `GPU_FRAME_MS` line is ever emitted from here, on purpose.**  That token belongs to the
port's runner: `runner.cpp:298-300` documents that `GPU_FRAME_MS` (runner.cpp:340) brackets *only*
`chain->record()` - two query timestamps, TOP to BOTTOM - and the frozen consumers parse exactly
that grammar from runner logs (`eval_verify.py:115`).  This harness has no steady-state handle on
the chain's dispatch window, nor on the port side at all (the DLL path's only stamps are
`GetTickCount64` point times, nvngx.cpp:241); a look-alike with a different meaning would poison
V8-style consumers.  Its own grammar (`EVAL_MS` / `GPU_EVAL_MS` / `OUT_NZ`) is net-new and self-
consistent.

**Statistics skip frame 0 everywhere** - the runner/eval_verify convention (eval_verify.py `range(1, n)`),
plus OptiScaler may early-return its first evaluate for change-backend bookkeeping
(NVNGX_DLSS_Dx12.cpp:908-915) and the first `CreateFeature` sleeps 500 ms
(DLSSFeature_Dx12.cpp:38).  `median` and nearest-rank `p95` are computed over frames `1..N`.

## The gates (`HARNESS SUMMARY`, run_harness.sh + exe)

- **g1 OptiScaler**: `Creating new dlss upscaler` + `Loaded from ...\_nvngx.dll` in OptiScaler.log,
  `fsr_fallback=0`, and **no** `_EvaluateFeature result: BAD00000` (normal) / **exactly the
  refusals present** (`--expect-fail`).  Also printed as evidence: `Running on Nvidia` (the
  `isNvidia()` gate, dllmain.cpp:1978 - it passes because the run uses the game's compatdata
  prefix, whose nvapi stack OptiScaler recognises, plus `PROTON_FORCE_NVAPI=1`).
- **g2 port log**: normal = `≥ N` `eval #K OK:` lines, `0` FAIL lines, and the triplet
  `fmt col=0 mv=1 dep=0` once per frame with default formats (`rg32` MVs make the port re-derive
  code 1, nvngx.cpp:1158-1161; `rg16`/corpus expect `mv=0`).  `--expect-fail` = every frame
  refused, zero OK, and issue 1's exact line `Depth: DXGI format 19 is not one this bridge
  decodes` + `eval #1 FAIL: ... slot=Depth` present.
- **g3 non-black**: every `out_<i>.f32` has `nz >= 0.10`.  **Threshold justification**: the
  synthetic colour is a moving gradient + checker with *every channel* strictly non-zero, so any
  healthy evaluate sits at `nz ≈ 1.00` (measured: 1.000000 on all frames); 0.10 is a conservative
  floor that still catches "never written".  Under `--expect-fail` the inequality inverts
  (`nz < 0.10` on every frame = output never written, measured: exactly 0.000000).
- **g4 timing**: `EVAL_MS` n and median > 0 over frames 1.. plus a table with `GPU_EVAL_MS`
  median/p95; `n/a` lines are reported as a *degradation*, not a failure (per the contract above).
- **g5 SL boot** (`--sl` only, never folded into g1–g4): five independent subchecks, each printed
  as `ok`/`bad(...)` in the `HARNESS SUMMARY`, evaluated over `g5_evidence.txt` = client.log
  (`[sl-cb]` callback lines + `SLDLSS` probes) concatenated with OptiScaler's `[streamline]`
  echo (whichever registered first carried the line is fine):
  1. a `getSystemCaps` line is present AND NOT `NVAPI failed to initialize` (or, if a driver
     number is reported, it is shown and must satisfy the required one),
  2. no `Failed to initialize NGX`,
  3. no `requires driver 512.15.0 or newer - detected 0.0.0` (generalized: any such line's
     detected version must be >= the required one, `sort -V`),
  4. no `Missing NGX context`,
  5. `Loaded plugin 'sl.dlss'` present.
  g5 joins the exit verdict ONLY under `--sl`; without `--sl` the run is byte-for-byte the old
  four-gate run.

The exe adds its own four gates (`GATE eval / nonblack / timing / port`, `RESULT=` in
`summary.txt`), inverted wholesale by `--expect-fail`, and its exit code feeds `HARNESS RESULT`.

## Reproducing the two in-game issues

**Issue 1 - DLSS black (Depth refused).**  In-game the port saw the Depth slot as DXGI 19
(R32G8X24_TYPELESS) and refused every frame (63,760/session); the fix in `src/pe/nvngx.cpp`
maps 19/20/21 → `VK_FORMAT_D32_SFLOAT_S8_UINT` and copies the DEPTH aspect.  The harness's
*default* depth input is exactly that resource (typeless 19 + DSV typed 20 + constant-clear
content - a constant clear because D3D12 gives a depth-stencil resource no `COPY_DEST`, so no
per-pixel gradient can be uploaded; valid floats are all the black gate needs), so `run_harness.sh`
is the green run.  The reproduction stages an **archived pre-fix DLL** instead:

    bash tools/optiharness/run_harness.sh \
        --nvngx "$GAME_BIN64/_nvngx.dll.bak-mvfix" --expect-fail

must *pass* its inverted gates: 12/12 refusals with `Depth: DXGI format 19 ...` in the port log
while the refusal comes back through OptiScaler as `BAD00000` (g1 `bad00000=12`), port
`ok=0/12 fail=12`, and every output black - the in-game failure, end to end, without the game.

**Issue 2 - XeSS black (spoofed upscaler).**  The game's `OptiScaler.ini` had been force-set to
`Dx12Upscaler=dlss` during diagnostics, making OptiScaler serve *every* pick (including XeSS)
through the DLSS path; it was restored to `auto`.  This harness deliberately uses that force -
`Dx12Upscaler=dlss` in `OptiScaler.harness.ini` - to pull OUR path under test - but the ini lives
*only* in the staging dir (OptiScaler reads the ini next to its own dxgi.dll, Config.cpp:45), so
the game's ini is never read or written by this tool.

## Streamline boot headless (`--sl`) - reproducing the game's SL doom in ~90 s

The in-game black correlates with NVIDIA Streamline failing **before any upscale**: in the archived
game bootstrap (`bin64/runs/OptiScaler.slayer3-noeval.log-20261030`) the plugin map dies inside
`slOnPluginStartup` and SL unloads its DLSS context.  `--sl` boots the SAME stack in the harness:

    LoadLibraryA("sl.interposer.dll")  (BEFORE any dxgi/D3D12 activity, the doc-pinned ordering)
    -> slInit(prefs, sl::kSDKVersion)  [logMessageCallback = ours -> [sl-cb] lines in client.log]
    -> D3D12CreateDevice -> slSetD3DDevice (IMMEDIATELY after the device, sl_core_api.h doc)
    -> SLDLSS probes: slIsFeatureSupported / slIsFeatureLoaded / slGetFeatureRequirements (kFeatureDLSS)
    -> the unchanged Opti DLSS evaluate flow (g1-g4 keep running)
    -> slShutdown at exit.

Stage: all 11 game `sl.*.dlls` symlinked read-only next to the exe (so `findPlugins` looks in the
exe dir - archive:309 does the game's `.../bin64`).

**Preferences are mirrored from the archive, field by field** (vendored headers, see below):
| field | value | archive evidence |
|---|---|---|
| sdkVersion | `sl::kSDKVersion` = host SDK **v2.14.1** | `setHostSDKVersion Streamline v2.14.1.98614dad6 ... host SDK v2.14.1` (slayer3:277) |
| applicationId | **101640911** | `initializePlugins ... - application ID 101640911` (slayer3:454) |
| featuresToLoad | DLSS(0), Reflex(3), PCL(4), DLSS_G(1000), DLSS_RR(1001) | `Ignoring plugin 'sl.deepdvc'/'sl.directsr'/'sl.nis'/'sl.nvperf' ... not requested` (slayer3:370/377/404/407) vs the six `Loaded plugin` lines (ids 0/3/4/1000/1001 + common `4294967295`) |
| flags | `eDisableCLStateTracking \| eAllowOTA`, NOT `eLoadDownloadedPlugins` | OTA manifest attempted during slInit (`readServerManifest ... nvngx_config.txt`, slayer3:278) + `eLoadDownloadedPlugins flag not passed to preferences` (slayer3:333) |
| pathsToPlugins | the exe dir | `findPlugins Looking for plugins in <exe dir>` (slayer3:309) |
| logLevel/logMessageCallback | eVerbose + ours | our own capture; Opti's `StreamlineHooks` echo appears in OptiScaler.log anyway |
| engine/engineVersion/projectId | left at defaults (`eCustom`, unset) | the archive's `InProjectId: 0F71CA1E-... / InEngineType: 0 / InEngineVersion: 0.9.4` are **OptiScaler's own** (`OPTI_GUID`, SysUtils.h:33; "0.9.4" = Opti's version), not game preferences - applicationId alone satisfies sl_core_types.h |
| renderAPI | `eD3D12` | the game's render path (dxgi CreateDXGIFactoryInternal -> d3d12 createKernel) |

**ABI validation (the brief's stop- condition met before shipping)**: the game runs SL
**v2.14.1** and tag `v2.14.1` exists on NVIDIA-RTX/Streamline (commit `2122257e...`), so
`sl-include/` holds the nine headers from THAT tag verbatim (url/commit/date/sha in
`sl-include/VENDORED.txt`, NVIDIA's MIT-style notice preserved inside every file);
`sl_abi_check.cpp` hand-derives the layouts (BaseStructure 32, Preferences 144 with
renderAPI@136, AdapterInfo 56, kSDKVersion) and is compiled by `build.sh` with host g++ BEFORE
the cross-compile, and the same `static_assert`s live in `opti_client.cpp`.

### Today's measured baseline (`run_harness.sh --sl`, 2026-10-31)

g5 **FAIL** = the game's doom reproduced headlessly (exit 1 by design), with g1–g4 still green
(12/12 evals, non-black, timings) - i.e. the port keeps working while SL refuses to arm DLSS:

    g5 SL boot : FAIL   c1_getSystemCaps_nvapi=ok  c2_no_failed_NGX=bad(x2)  c3_driver=ok
                         c4_no_missing_ctx=bad(x2)  c5_sl.dlss_loaded=ok
    SLDLSS slIsFeatureSupported(kFeatureDLSS)  result=32 (eErrorFeatureNotSupported)
    SLDLSS slIsFeatureLoaded(kFeatureDLSS)      result=31 (eErrorFeatureMissing) loaded=0
    SLDLSS slGetFeatureRequirements(...)        result=0 (eOk) flags=0x00000007 threads=0
             viewports=4 tags=4 driverDetected=999.99.0 driverRequired=512.15.0
             osDetected=10.0.19045 osRequired=10.0.0
    [sl-cb] ... getSystemCaps] NVIDIA driver 999.99
    [sl-cb] ... getNGXFeatureRequirements] ngxResult failed 0xbad00002      (x3, like the game)
    [sl-cb] ... slOnPluginStartup] At least one plugin requires NGX, trying to initialize ...
    [sl-cb] ... slOnPluginStartup] Failed to initialize NGX, any SL feature requiring NGX will be
             unloaded and disabled                       <- c2 bad, game line verbatim
    [sl-cb] ... dlssEntry.cpp:923[slOnPluginStartup] Missing NGX context - DLSSContext cannot run
                                                                  <- c4 bad, game line verbatim

**Where it differs from the game, and why** (reported, not papered over): the game archive's c1/c3
come from `NVAPI failed to initialize` + `detected 0.0.0`, while the harness reads
`NVIDIA driver 999.99` - because in THIS stack OptiScaler's Vulkan spoof answers the driver probe
(`spoofing/Vulkan_Spoofing.cpp:177/235: strcpy(ddp->driverInfo, "999.99")`; the literal has a
single producer in the stack), so 999.99 >= 512.15.0 and both checks pass.  The lever stream
(`SL_NVAPI_VER=<n>` -> `DXVK_NVAPI_DRIVER_VERSION=<n>` in the run env) is a one-line addition as
shipped; measured with `57602`: **g5 unchanged, driver still 999.99** - the spoof layer sits
*downstream* of dxvk-nvapi from SL's point of view, so the env var cannot move the number it
reads.  The two FAILING checks are the decisive doom lines anyway (`Failed to initialize NGX`,
`Missing NGX context`), and they reproduce verbatim.

## Freeze stance

- `tools/optiharness/` is a **new directory outside `tools/verify/`**; nothing frozen is edited.
  The frozen `tools/verify/ngx_client/ngx_abi.h` is included **read-only** (`-I tools/verify/ngx_client`
  in `build.sh`), and `run_harness.sh`'s Proton probe is *copied* from `run_all.sh:282-301`.
- All outputs live under git-ignored `build/optiharness/`: the staging dir, `out/` (frames,
  `summary.txt`, `eval_ms.csv`, `client.log`, `run.stdout.log`), probe prefixes and logs.
- The game folder is read via **symlinks only** (`dxgi.dll` or its `dxgi.dll.optiscaler-off`
  park name, `nvngx_dlss.dll`, the 11 `sl.*.dlls`, and the shader fallback); `build/package/_nvngx.dll`
  is only ever `cp`'d - this script never builds the port, never starts/stops the game, never
  touches Steam (the prefix clone exists precisely so a LIVE game session stays untouched).
- The Streamline headers are vendored *from NVIDIA's own repo* (tag v2.14.1) under
  `tools/optiharness/sl-include/` with the licence preserved - legitimate to commit, and pinned
  to the exact release the game reports.

## Files

| file | what |
| --- | --- |
| `opti_client.cpp` | the exe: host sequence `Init → AllocateParameters → CreateFeature → per-frame EvaluateFeature` (the NVSDK export surface verified by `objdump -p` on the shipped dxgi.dll: 53+ NVSDK entries), synthetic/corpus inputs, timing pairs, read-back + black detector, the `--sl` Streamline boot + `SLDLSS` probes |
| `OptiScaler.harness.ini` | the harness-local ini template (each key and every deliberately-unset key documented inline) |
| `build.sh` | cross-compile to `build/optiharness/opti_client.exe`, FMT16=1 (Output RGBA16F = `DLSS5_AMD_OUT_F16=1` of the package), mirrors `tools/verify/ngx_client/build.sh` incl. the podman fallback; runs `sl_abi_check.cpp` on host g++ first |
| `run_harness.sh` | staging (port DLL copy, OptiScaler + `nvngx_dlss.dll` + `nvngx_dlssd.dll` + 11 sl.* + shaders as symlinks, ini = `--ini` file or the template), prefix-busy detection + clone, Proton 10.0-pinned probe, env incl. `SL_NVAPI_VER`, run, gates g1-g5 |
| `sl-include/` | the nine NVIDIA Streamline v2.14.1 headers, verbatim from tag v2.14.1 (`VENDORED.txt` = url/commit/date/licence) |
| `sl_abi_check.cpp` | hand-derived layout assertions, host-compiled by build.sh before every cross-build |

## Environment the run pins (run_harness.sh, with reasons)

    env -u LD_PRELOAD
    WINEDLLOVERRIDES=winmm,dxgi=n,b      # OptiScaler's dxgi must load natively; winmm too (diagnostics)
    PROTON_FORCE_NVAPI=1                 # makes the dxvk-nvapi stack OptiScaler's isNvidia() probe sees
    DLSS5_AMD_WEIGHTS=build/ngx-weights  # the port reads its packed weights from here (nvngx.cpp:279)
    STEAM_COMPAT_DATA_PATH=.../compatdata/3321460   # THE GAME PREFIX: nvapi64.dll -> isNvidia()=true
    VK_ICD_FILENAMES/VK_DRIVER_FILES=.../radeon_icd.x86_64.json   # as run_all.sh:91/97 does (RADV only)
    MESA_SHADER_CACHE_DISABLE=true
    [SL_NVAPI_VER=<n>]                   # --sl only: adds DXVK_NVAPI_DRIVER_VERSION=<n>

**Two live-environment fallbacks, both read-only on the game folder** (measured 2026-10-31):
1. `STEAM_COMPAT_DATA_PATH` normally is the game prefix, but **another session's GE-Proton11-6
   wineserver can be holding compatdata/3321460 with CrimsonDesert.exe RUNNING** (measured at
   00:14 - pid 3046121); a second Proton on a live prefix dies before `main()` with
   `wine client error: version mismatch 935/856`.  The script detects that wineserver via its
   `/proc/<pid>/environ`, `cp -a`s the 478 MB prefix to `build/optiharness/prefix_3321460.clone`
   and runs against the clone - the running game is never touched or even signalled.
2. the game's `dxgi.dll` may be parked by a parallel diagnostics session as
   `dxgi.dll.optiscaler-off` (also measured 00:14); the script prefers `dxgi.dll` and falls
   back to that name (same 25 379 632-byte v0.9.4 proxy).

cwd = staging dir, because the port writes `dlss5-amd-ngx.log` to the CWD (nvngx.cpp:255) and
OptiScaler writes `OptiScaler.log` next to the exe (Logger.cpp, `Util::ExePath`) - both land
together.  Proton is probed newest-first with a bogus argument (rc must be 1, no Python Traceback)
*starting at* `Proton 10.0`: `docs/MEASUREMENT.md:252-256` documents the Wine 11 builds hanging
before they load anything; `PROTON=<path>` skips the probe.

## Known quirks

- **First CreateFeature blocks ~500 ms** (`DLSSFeature_Dx12.cpp:38`) plus the port's chain build
  (~5.1 s at 1920x1080→3840x2160, 29 pipelines) - the run's first seconds are exactly that.
- **Warmup**: OptiScaler can early-return its first evaluate (change-backend bookkeeping,
  NVNGX_DLSS_Dx12.cpp:908-915) without touching the port; frame 0 is excluded from every stat
  regardless.  (Measured runs so far: no skips - 12/12, 4/4, 12/12 OK; script g2 requires `≥ N`, the exe's
  own port gate allows `≥ N-2` so the two cannot both trip over a single bookkeeping skip.)
- **`OverlayMenu` must stay unset/true** (Config.h:288): `false` draws ImGui *into* the output
  texture after frame 30 and corrupts the pixel checks - hence no `[Menu]` section in the ini.
- **`isNvidia()` gate** (dllmain.cpp:1691): without the game prefix + nvapi, OptiScaler logs
  `Not running on Nvidia, disabling DLSS` and GetFeature falls to FSR 2.1.2 silently - g1 is what
  catches that (its tell-tales: absence of `Creating new dlss upscaler`, port eval count 0).
  Measured also green on the prefix *clone* (isNvidia `Using dxvk-nvapi / Detected: true`).
- **OptiScaler exports only the API-specific** `NVSDK_NGX_D3D12_ReleaseFeature` (not the
  unprefixed alias `ngx_client` resolves) - the exe takes D3D12 first, alias second.
- **PE stdout/stderr do not reach the shell under `proton run`** on this stack (the frozen V9 logs
  `build/verify/run_*/v9_*.log` contain only Proton's own lines) - every harness message is mirrored
  into `out/client.log`, which is what the gates parse; `run.stdout.log` keeps Proton's own noise.
- **D3D12 command-list methods**: mint `EndQuery(heap, Type, Index)` (SDK `d3d12.h`, MIDL) is
  index-based; a first draft that passed a byte offset wrote query 0 twice and left the resolved
  pair as ~`2^63/freq` garbage - timestamps are now sane (measured medians ~8 ms GPU vs ~4.4 ms
  host around the same evaluate).
- **`--expect-fail` outputs are all-zero on purpose** (the refused evaluate never writes the UAV);
  the inversion subjects every gate to the same evidence standard as the green run.
- Proton probe/prefix scratch dirs and re-staging are cheap: the staging dir is rebuilt fresh on
  every invocation, so logs can never leak from a previous run into a gate.
- **`--sl` emits every SL line twice** when Opti's StreamlineHooks echo is also active - g5 counts
  presence/absence over both sources (client.log [sl-cb] + OptiScaler.log echo) so the ×2 counts
  are evidence breadth, not duplicate work; `[sl-cb]`, `SLDLSS`, `SL ...` lines are the harness's
  own grammar.
- **The sl.interposer loads BEFORE the exe's explicit dxgi LoadLibrary**, satisfying the doc
  ordering ("interposer injects dxgi hooks ... before the harness creates its DXGI factory/D3D
  device"); if the loader pulled Opti's dxgi in at process start (static -ldxgi), its LdrLoadDll
  hooks already exist - both orders are the ones the game exhibits, and both ran green here.

## Timing results to expect (RX 9070 XT, Proton 10.0, 2026-10-30)

    default --opti, 1920x1080 -> 3840x2160, 12 frames:
      EVAL_MS      median ~4.4 ms   p95 ~4.6 ms     (whole stack, CPU wall)
      GPU_EVAL_MS  median ~8.0 ms   p95 ~9.9 ms     (GPU span of the evaluate)
      same frames --direct: EVAL_MS median ~4.3 ms   (the A/B says OptiScaler's host overhead
                                                      at this resolution is within noise today)
      port `recorded in` median 0 ms (GetTickCount ms granularity; first frame 13 ms)
    --expect-fail (archived pre-fix DLL): EVAL_MS median ~3.4 ms, GPU_EVAL_MS ~0.002 ms
      (the refusal does no GPU work), 12/12 `slot=Depth`, outputs all zero.

## 2026-10-31 - `--ini <path>` (N1) and staged `nvngx_dlssd.dll` (N2)

    bash tools/optiharness/run_harness.sh --ini tools/scenario/inis/cell_B03.ini  # stage THAT ini
    bash tools/optiharness/run_harness.sh                                         # default template

- **`--ini <path>`** (optional, parsed like `--out`/`--nvngx`): stage THAT file's contents as the
  harness `OptiScaler.ini` instead of `tools/optiharness/OptiScaler.harness.ini`.  Absent = the old
  behaviour byte-for-byte (the default variable *is* the old literal path, same `cp`).  Validated
  before anything is built, staged or run: no such file or no `[section]` header ⇒
  `HARNESS FAIL: --ini: ...` rc 1.  The effective path is printed in the normal status line
  (`staged ... (nvngx=..., ini=<path>, ...)`); the game's ini is still never read or written
  (OptiScaler reads the staged copy, `Config.cpp:45`).
- **`nvngx_dlssd.dll`** is now staged on the same waypoint as `nvngx_dlss.dll`, same ABSOLUTE
  symlink style, source = the game's `bin64/nvngx_dlssd.dll` (the real NVIDIA DLSS-D companion,
  48 343 664 B, `nvngx.cpp:187-189`).  Missing source dies in preflight exactly like
  `nvngx_dlss.dll` does - the sibling is fatal, so this is too, never a dangling link.  This is what
  the D1 RR cells (N2) were blocked on.
- **Consistency note for `tools/scenario` (not changed here)**: nothing under `optiharness/` writes a
  staging manifest (checked `opti_client.cpp`, `build.sh`, this README) - `run_cell.sh:87` does, with
  `stat -c %s`, which reports the *symlink length*, not the target size (`nvngx_dlss.dll` reads 82,
  not 58956912).  `gate_cell.sh` G-STAGE / G-PORT-RR compare target byte sizes, so they need
  `stat -Lc %s` (one character) in `run_cell.sh` before any manifest size gate can go green.
- Both flags parse-validated (bash -n + usage paths); not exercised end-to-end — execution slots release later.

## 2026-10-31 - RR inputs (N4): the `GBuffer.Normals` / `GBuffer.Roughness` slots

**The gap (measured by D1, `D1-DESIGN.md` §6 N4, `REPORT.md` `Tier B` 248-263)**: with
`--ini .../inis/dlssd.ini` (`Dx12Upscaler=dlssd`) the RR *create* worked end to end — `RR ... creating
the frame chain from .../ngx-weights/rr`, `D3D12 CreateFeature: id=1 feature=13 preset=F`, 28
pipelines — but every evaluate refused `slot=GBuffer.Normals why=GetD3D12Resource returned null for
this key`: the client only ever created/bound Color, MotionVectors, Depth.

**What was added (`opti_client.cpp`)**

- **Four** slots are created and bound for an RR run, not two — the port fetches ALL of `extra[]` for
  a RayReconstruction feature (`nvngx.cpp:1154-1161` loops the whole array; `nvngx.cpp:1132-1136`
  lists four), so binding only Normals + Roughness would just move the refusal to
  `slot=DiffuseAlbedo`:

  | key | resource | DXGI format (= port contract) | `UpSlot.kind` |
  | --- | --- | --- | --- |
  | `GBuffer.Normals` | committed TEXTURE2D, DEFAULT heap, COPY_DEST → NON_PIXEL_SHADER_RESOURCE | `DXGI_FORMAT_R16G16B16A16_FLOAT` (10) = `VK_FORMAT_R16G16B16A16_SFLOAT` | 3 |
  | `DLSS.Input.DiffuseAlbedo` | same | `DXGI_FORMAT_R16G16B16A16_FLOAT` (10) | 4 |
  | `DLSS.Input.SpecularAlbedo` | same | `DXGI_FORMAT_R16G16B16A16_FLOAT` (10) | 5 |
  | `GBuffer.Roughness` | same | `DXGI_FORMAT_R32_FLOAT` (41) = `VK_FORMAT_R32_SFLOAT` | 6 |

  Same creation style, upload buffer (`pitch256`-aligned), per-frame `CopyTextureRegion` +
  barrier pattern, and `p->Set(key, ID3D12Resource*)` binding as Color/MV/Depth (the Depth-slot
  precedent, README issue 1).  The formats match the chain's default gather codes
  `fmt_nrm/fmt_alb/fmt_spa = 0` (8 B/px), `fmt_rgh = 3` (4 B/px) (`src/vk/pipeline.h:36-39`,
  `rr_gather.comp:16`), built into `static_assert`s against the DXGI enum numbers.

- **Exact data fed** (synthetic, *frame-independent* like `synth_mv`/`synth_depth` — no `rand`
  anywhere, run-to-run input variance 0; `run_harness.sh` never sets `DLSS5_AMD_SKIP_INPUTS`):
  - `GBuffer.Normals`: unit-length normal sweep `n = normalize(2·x/(W−1)−1, 2·y/(H−1)−1, 1)` stored
    as f16 xyzw with `w = 0` (centre faces +z, corners lean outward; roughness-in-`normals.w` mode is
    off, roughness travels in its own texture).
  - `DLSS.Input.DiffuseAlbedo`: constant `(0.5, 0.5, 0.5, 1.0)` f16 (mid-grey), `a = 1`.
  - `DLSS.Input.SpecularAlbedo`: constant `(0.04, 0.04, 0.04, 1.0)` f16 (dielectric F0), `a = 1`.
  - `GBuffer.Roughness`: constant `0.5` (mid-grey), one R32F float per texel.

  Nothing else changed in the byte stream: Colour (gradient+checker, per-frame), MotionVectors
  (+2 px, +0.25 px pan) and Depth (y-plane / 0.5 clear) generators are untouched.

- **When they are bound — and when not**: an RR flag is derived from the *staged* ini the run will
  actually load — `exe_dir + "OptiScaler.ini"`'s `[Upscalers] Dx12Upscaler=dlssd` (OptiScaler reads
  the ini next to its own dxgi.dll, `Config.cpp:45`; `auto` never picks dlssd, `Config.cpp:1561-1562`,
  and the dlssd→feature-13 rewrite happens inside `DLSSDFeature_Dx12.cpp:52` where the client's
  resolved exports cannot observe it).  Guarded `!o.direct && o.corpus.empty()`:
  - **default dlss / SR runs: not created, not uploaded, not bound, no new log lines** — byte-for-byte
    previous behaviour (validated by the regression run below).
  - `--direct`: the client asks the port for feature 1 itself, ini irrelevant → off.
  - `--corpus`: the V9 corpus contract is col/mv/dep only → off (a corpus × dlssd combo is NOT
    supported and remains untested).

- **Per-eval slot diagnostics** (`slot=` grammar kept self-explaining): in RR runs only, after each
  `EVAL_MS` the client prints one line per evaluate with every bound RR slot — key, resource pointer
  and format — so the port's `eval #N FAIL: ... slot=X why=...` can be read straight against what was
  bound:

  ```
  RR_SLOTS i=0 eval_ok=1 feat=13 slot=GBuffer.Normals res=... fmt=R16G16B16A16_SFLOAT(DXGI 10)
           slot=DLSS.Input.DiffuseAlbedo ... slot=DLSS.Input.SpecularAlbedo ...
           slot=GBuffer.Roughness res=... fmt=R32_SFLOAT(DXGI 41)
  ```

**What changed in `run_harness.sh`** (both changes are inert for SR runs — identical counts/strings):

- g1's create line also accepts `Creating new dlssd upscaler` (`grep -cE "Creating new dlssd? upscaler"`
  — same `FeatureProvider_Dx12.cpp:177` site, `newBackend` = `dlssd`).
- g2's expected fmt triplet is `mv=10` for a dlssd run: feature 13 re-derives the same in-game RG32F
  MotionVectors to code **10**, not 1 (`nvngx.cpp:1177 fp.fmt_mv = rr ? 10 : 1`), so an RR eval-OK
  line can only ever read `fmt col=0 mv=10 dep=0`.  Detection = the same `Dx12Upscaler=dlssd` grep on
  the ini being staged; `--mv-fmt rg16`/`--corpus` keep code 0 as before.
  **Note for `tools/scenario` (not edited here)**: `gate_cell.sh:130` hardcodes `fmt col=0 mv=1 dep=0`
  for *all* backends — a fixed dlssd cell will read `mv=10`, so its `n_fmt` needs the same treatment
  before G-PORT can go green.

**Validation (2026-10-31, RX 9070 XT / Proton 10.0)**

- **RR smoke** `run_harness.sh --sl --frames 12 --w 2560 --h 1440 --ow 3840 --oh 2160 --ini
  tools/scenario/inis/dlssd.ini --out /tmp/opencode/n4_rr_smoke` → **the N4 gap is closed**:
  - port log: `RR ... chain ready in 13304 ms, 28 pipelines` (weights `.../ngx-weights/rr`),
    `D3D12 CreateFeature: id=1 feature=13 preset=F`, **12/12 `eval #N OK: ... feature=13 n=7`** with
    `[3 GBuffer.Normals vk=97 dxgi=10] [4 DiffuseAlbedo vk=97 dxgi=10] [5 SpecularAlbedo vk=97
    dxgi=10] [6 Roughness vk=100 dxgi=41]` on every eval, `GetD3D12Resource returned null` **x0**,
    `FAIL:` x0, `fmt col=0 mv=10 dep=0` x12;
  - client: `RR=1 ...` header + **12 `RR_SLOTS` lines, all `eval_ok=1`**;
  - script gates `g1 PASS / g2 PASS (ok=12/12) / g3 PASS (min nz=1.000000) / g4 PASS
    (EVAL 7.12 ms, GPU 24.12 ms median)`; exe `RESULT=PASS`, `PORT_EVAL ok=12 fail=0`.
  - `g5 FAIL` is the game's Streamline doom reproduced headlessly by design on every `--sl` run
    (README baseline) — it joins the verdict under `--sl` only; all N4-specific criteria are green
    with or without it.
- **Regression** (same command without `--ini`, dlss template, `--out /tmp/opencode/n4_dlss_regress`)
  → `g1-g4 PASS`, `g2 ... 'fmt col=0 mv=1 dep=0'x12` **unchanged**, `ok=12/12 fail=0`,
  `PORT_EVAL ok=12`, exe `RESULT=PASS`; the SR run contains **zero** RR traces (`RR=1` x0,
  `RR_SLOTS` x0, `GBuffer` x0, `feature=13` x0, every eval `n=3`), i.e. the previous behaviour is
  byte-for-byte preserved wherever the old code looked.  Timings sit on the documented baseline
  (EVAL median 4.42 ms vs ~4.4, GPU 10.23 ms vs ~10.2-10.6 sentinel).

**Not tested / out of scope here**: `--corpus` × dlssd, `--direct` × dlssd (both structurally
cannot bind RR), a non-default `--mv-fmt` RR combination, 60-frame D1 B-cells re-runs (D1 re-runs
them after `N4 LANDED`), in-game verification (D3), and `gate_cell.sh`'s `fmt mv=1` expectation
(D1-owned file, flagged above).
