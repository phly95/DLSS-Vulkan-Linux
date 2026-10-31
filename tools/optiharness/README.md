# tools/optiharness: timing + integrity of the DLSS port *with OptiScaler in the loop*

`tools/optiharness` stages a Windows-neutral harness directory under the git-ignored
`build/optiharness/harness/` - OptiScaler's `dxgi.dll` proxy, our `_nvngx.dll`, the shader and
weight paths - and runs its own `opti_client.exe` under Proton against the **game's compatibility
prefix**, so frames travel the same path they take in Crimson Desert:

    game-style D3D12 frames -> OptiScaler v0.9.4 (dxgi.dll middle layer) -> _nvngx.dll (Vulkan port)
                                                               -> RGBA16F output -> read back

without ever launching the game.  It exists to (a) measure clean ms timings with OptiScaler
present, (b) prove the upscaled output is real (not black), and (c) reproduce both known in-game
issues from the harness alone.

    bash tools/optiharness/run_harness.sh                                  # default --opti run
    bash tools/optiharness/run_harness.sh --direct                         # A/B: skip OptiScaler
    bash tools/optiharness/run_harness.sh \
        --nvngx "/path/to/_nvngx.dll.bak-mvfix" --expect-fail               # issue-1 reproduction
    bash tools/optiharness/run_harness.sh --corpus DIR [--frames N]        # V9 file corpus inputs
    bash tools/optiharness/run_harness.sh --mv-fmt rg16 --dep-fmt r32f     # proven-format bisect

Exit code 0 only when every gate passes; each run prints a `HARNESS SUMMARY` (script gates g1-g4)
and the exe writes its own `HARNESS SUMMARY` block plus `out/summary.txt` and `out/eval_ms.csv`.

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

## Freeze stance

- `tools/optiharness/` is a **new directory outside `tools/verify/`**; nothing frozen is edited.
  The frozen `tools/verify/ngx_client/ngx_abi.h` is included **read-only** (`-I tools/verify/ngx_client`
  in `build.sh`), and `run_harness.sh`'s Proton probe is *copied* from `run_all.sh:282-301`.
- All outputs live under git-ignored `build/optiharness/`: the staging dir, `out/` (frames,
  `summary.txt`, `eval_ms.csv`, `client.log`, `run.stdout.log`), probe prefixes and logs.
- The game folder is read via **symlinks only** (`dxgi.dll`, `nvngx_dlss.dll`, and the shader
  fallback); `build/package/_nvngx.dll` is only ever `cp`'d - this script never builds the port.

## Files

| file | what |
| --- | --- |
| `opti_client.cpp` | the exe: host sequence `Init → AllocateParameters → CreateFeature → per-frame EvaluateFeature` (the NVSDK export surface verified by `objdump -p` on the shipped dxgi.dll: 53+ NVSDK entries), synthetic/corpus inputs, timing pairs, read-back + black detector |
| `OptiScaler.harness.ini` | the harness-local ini template (each key and every deliberately-unset key documented inline) |
| `build.sh` | cross-compile to `build/optiharness/opti_client.exe`, FMT16=1 (Output RGBA16F = `DLSS5_AMD_OUT_F16=1` of the package), mirrors `tools/verify/ngx_client/build.sh` incl. the podman fallback |
| `run_harness.sh` | staging (copy port DLL, symlink OptiScaler + `nvngx_dlss.dll` + shaders, generate ini), Proton 10.0-pinned probe, env, run, gates g1-g4 |

## Environment the run pins (run_harness.sh, with reasons)

    env -u LD_PRELOAD
    WINEDLLOVERRIDES=winmm,dxgi=n,b      # OptiScaler's dxgi must load natively; winmm too (diagnostics)
    PROTON_FORCE_NVAPI=1                 # makes the dxvk-nvapi stack OptiScaler's isNvidia() probe sees
    DLSS5_AMD_WEIGHTS=build/ngx-weights  # the port reads its packed weights from here (nvngx.cpp:279)
    STEAM_COMPAT_DATA_PATH=.../compatdata/3321460   # THE GAME PREFIX: nvapi64.dll -> isNvidia()=true
    VK_ICD_FILENAMES/VK_DRIVER_FILES=.../radeon_icd.x86_64.json   # as run_all.sh:91/97 does (RADV only)
    MESA_SHADER_CACHE_DISABLE=true

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

## Timing results to expect (RX 9070 XT, Proton 10.0, 2026-10-30)

    default --opti, 1920x1080 -> 3840x2160, 12 frames:
      EVAL_MS      median ~4.4 ms   p95 ~4.6 ms     (whole stack, CPU wall)
      GPU_EVAL_MS  median ~8.0 ms   p95 ~9.9 ms     (GPU span of the evaluate)
      same frames --direct: EVAL_MS median ~4.3 ms   (the A/B says OptiScaler's host overhead
                                                      at this resolution is within noise today)
      port `recorded in` median 0 ms (GetTickCount ms granularity; first frame 13 ms)
    --expect-fail (archived pre-fix DLL): EVAL_MS median ~3.4 ms, GPU_EVAL_MS ~0.002 ms
      (the refusal does no GPU work), 12/12 `slot=Depth`, outputs all zero.