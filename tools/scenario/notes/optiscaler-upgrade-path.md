# OptiScaler upgrade path — getting the PR #1126-class fix (dxvk-nvapi recursion) into our stack

Scope: Crimson Desert under Proton (game dir `/home/deck/.local/share/Steam/steamapps/common/Crimson Desert`),
shipping OptiScaler **v0.9.4** as `bin64/dxgi.dll`, port plugged in behind Opti via
`[Libraries] NvngxPath=S:\steamapps\common\Crimson Desert\bin64\_nvngx.dll`.
Research date: 2026-10-31. Read-only investigation; nothing was installed into the game dir.
Local source checkouts used (not modified): `/tmp/opencode/OptiScaler` (= tag v0.9.4, HEAD `7534ad0`)
and a fresh shallow clone `/tmp/opencode/OptiScaler-master` (master HEAD `e7e3872d`, 2026-10-29).

---

## 0. TL;DR / recommendation preview

* **There is no stable OptiScaler release after v0.9.4 at all.** The fix only exists on `master`,
  published only as **daily nightlies** in the separate `OptiScaler-nightly` repo.
* **First release containing PR #1126 = `nightly-20260902`** (asset URL below).
* v0.9.4 **does not contain the buggy code path** (`IdentifyGpu.cpp` doesn't exist in the 0.9 line),
  so there is nothing to "backport verbatim" *and* nothing broken that #1126 would fix on 0.9.4 —
  our remaining NVAPI/NGX failures under 0.9.4 have a different root cause (see §6).
* Recommended: **(B) replace `dxgi.dll` with a pinned nightly + a small ini migration**, not (C).
  Details and risks in §6.

---

## 1. Release cadence after v0.9.4

Source: GitHub API `https://api.github.com/repos/optiscaler/OptiScaler/releases` (queried 2026-10-31).

| tag | published | assets |
|---|---|---|
| **v0.9.4** (current, latest stable) | 2026-07-18T21:46:07Z | `Optiscaler_0.9.4-final.20260718._MM.7z` — https://github.com/optiscaler/OptiScaler/releases/download/v0.9.4/Optiscaler_0.9.4-final.20260718._MM.7z (55,016,448 B, verified HTTP 200) |
| v0.9.3 | 2026-06-18 | `Optiscaler_0.9.3-final.20260618.7z` |
| `nightly` (prerelease, main repo) | 2026-07-04T14:06:34Z | **no assets**; body: "# Actual Nightly build downloads can now be found [HERE](https://github.com/optiscaler/OptiScaler-nightly/releases)" |

* **No release exists between v0.9.4 and today** — releases after v0.9.4: **not found (none)**.
* Nightlies live in **https://github.com/optiscaler/OptiScaler-nightly/releases** — 84 releases
  (API `.../OptiScaler-nightly/releases?per_page=100`, page 1 count = 84), oldest
  `nightly-20260807` (2026-08-07T17:23Z), running daily to `nightly-20261030` (2026-10-30T10:47Z).
  Asset naming: `OptiScaler_v10.0.0-pre1_YYYYMMDD.7z`, i.e. the whole post-0.9 line is branded **v10.0.0-pre1**.

### First release containing PR #1126

* PR #1126 "Fix dxvk-nvapi recursion in IdentifyGpu::queryNvapi (fixes #1101)", merged
  **2026-09-01T21:07:26Z**, merge commit `c983a500335134ecff512bfcdadcf912d1286547`, base `master`
  (API `https://api.github.com/repos/optiscaler/OptiScaler/pulls/1126`).
  Diff = 1 file, `OptiScaler/misc/IdentifyGpu.cpp`, +6/0 (PR files API).
* Commits API `since=2026-09-01T09:00Z&until=2026-09-02T09:00Z` shows the order:
  `d7a4ecf` (09-01T16:28Z, "Fix GPU probe recursion with dxvk-nvapi (issue #1101)") →
  `c983a500` merge (09-01T21:07Z) → further commits up to `8030ecca` (09-02T08:18Z).
* `nightly-20260901` was published **2026-09-01T09:15:32Z** → *before* the merge ⇒ does NOT contain it.
* `nightly-20260902` was published **2026-09-02T08:29:39Z** (after `8030ecca`), built from
  `optiscaler/OptiScaler` default branch (`nightly` repo `.github/workflows/build.yml`
  lines 17, 24–31: repo `${{ env.MAIN_REPO }}`, no ref ⇒ master) ⇒ **first nightly with the fix**.

**Download assets (Linux/Proton-usable: package is OS-agnostic PE + ini):**

* First-with-fix (pinned, minimal post-fix delta):
  https://github.com/optiscaler/OptiScaler-nightly/releases/download/nightly-20260902/OptiScaler_v10.0.0-pre1_20260902.7z (56,867,496 B, verified 200)
* Latest at research date (recommended if we upgrade; most subsequent fixes included):
  https://github.com/optiscaler/OptiScaler-nightly/releases/download/nightly-20261030/OptiScaler_v10.0.0-pre1_20261030.7z (72,309,885 B, verified 200)

### Package contents (verified by `7z l` on nightly-20261030)

```
OptiScaler.dll            (26.7 MB — rename → dxgi.dll ourselves, as we already do)
OptiScaler.ini            (template, 60,540 B)
setup_linux.sh, setup_windows.bat, OptiScaler/Streamlined_fetcher_windows.bat
OptiScaler/               (amd_fidelityfx_*_{dx12,vk}.dll, libxess*.dll, libxell.dll,
                           dlssg_to_fsr3_amd_is_better.dll, D3D12_OptiScaler/D3D12Core.dll)
Licenses/*
```
No `nvapi64.dll`, **no `fakenvapi.dll`** (fakenvapi is now compiled into OptiScaler, §3),
no `dxgi.dll` (setup scripts rename `OptiScaler.dll`; `setup_windows.bat` lines 33–48).
Build packaging: `OptiScaler/OptiScaler.vcxproj` PostBuild lines 135–156 (moves `$(TargetFileName)`
into `x64\Release\a\`, copies ini + scripts + `OptiScaler\` subdir); nightly workflow line 126
archives `x64\Release\a\*.*`.

### Why no "0.9.5 with the fix"

* `release/0.9` branch (`0fc1ef63d98e`) `OptiScaler/misc/` contents API ⇒
  `['FrameLimit.cpp','FrameLimit.h','Quirks.h']` — **no `IdentifyGpu.cpp`**, so #1126 cannot apply there.
* Compare API `compare/c983a500...release/0.9` ⇒ `status: diverged` (422 ahead / 882 behind),
  `compare/c983a500...master` ⇒ `status: ahead` (contained in master).
* ⇒ **No backport of #1126 to any release branch exists.**

---

## 2. Breaking-risk audit v0.9.4 → nightly/v10 (for OUR config)

### 2.1 Keys we rely on (game dir `bin64/OptiScaler.ini`, verified 2026-10-31)

| key | our line | in v10 template (packaged `OptiScaler.ini`) | still read by v10 code (`master/OptiScaler/Config.cpp`) | verdict |
|---|---|---|---|---|
| `[Libraries] NvngxPath=S:\steamapps\...\bin64\_nvngx.dll` | 681 | 801 (`NvngxPath=auto`) | 819 `readWString("Libraries","NvngxPath")` | **kept** ✔ |
| `[Upscalers] Dx12Upscaler=auto` | 10 | 12 | 61–67 | **kept** ✔ |
| `[Spoofing] DxgiFactoryWrapping=false` (our P1 lever) | 878 | 1110 | 710 | **kept** ✔ |
| `[Spoofing] StreamlineSpoofing=true` | 869 | 1101 | 717 | **kept** ✔ |
| `[Spoofing] Dxgi=true` | 873 | 1105 | 709 | **kept** ✔ |
| `[NvApi] DisableFlipMetering=auto` | 969 | 1248 | 703 | **kept** ✔ |
| `[NvApi] OverrideNvapiDll=auto` | 964 | **absent** | **no read anywhere** (grep `OverrideNvapi` ⇒ 0 hits) | **REMOVED** ✖ |
| `[Menu] OverlayMenu=auto` | 730 | 856 | 492 | **kept** ✔ |
| `[FrameGen] FGInput=auto` | 32 | 37 | enum read under `[FrameGen]` (Config.cpp 82–145) | **kept** ✔ |
| `[Inputs] EnableDlssInputs=auto` | 387 | 489 | 773 | **kept** ✔ |
| `[Hooks] UseNtdllHooks=auto` | 1006 | 1286 | 532 | **kept** ✔ |
| `[DLSS]` / `[DLSSD]` sections | 609 / 639 | 715 / 745 | 343–392 (`DLSS`,`DLSSD` parse incl. RenderPreset*) | **kept** ✔ |

The absolute `NvngxPath` external-`_nvngx.dll` mechanism is unchanged in v10:
`NVNGX_Proxy.h:599` (`dllNames = {_nvngx.dll, nvngx.dll}`), `:602` `overridePath = Config…NvngxPath`,
`:611` `Util::LoadProxyLibrary(dllNames[i], optiPath, overridePath, …)`,
and `Util.cpp:911–927` loads `overridePath` first if `std::filesystem::exists` (file-or-dir, dir ⇒ `path/name`),
registry path is only the **fallback** (`NVNGX_Proxy.h:622–632`).
⇒ **external `_nvngx.dll` via `NvngxPath` still works**, and takes priority over game-local `nvngx.dll`.

### 2.2 Section-level changes (diff of `grep '^\[…\]' v0.9.4 ini` vs master/v10 template)

```
+ [DLSSG]         (new — DLSSG/DMFG options)
  [Nukems]  →  [NvngxFG]   (RENAMED)
+ [fakenvapi]     (new — replaces [NvApi] fakenvapi controls)
+ [Magnifier]     (new)
```
All other 38 sections incl. `[Upscalers] [FrameGen] [Inputs] [DLSS] [DLSSD] [Libraries] [Menu]
[Spoofing] [NvApi] [Hooks]` are unchanged (same names).

### 2.3 Key-level removals (both files diffed with `diff <(grep -o '^[A-Za-z0-9_]*=' …)`)

Removed in v10 template (=> become **inert** if left in our ini — `Config::readBool` at
`Config.cpp:2315–2323` simply returns `nullopt` for unknown keys via simpleini `GetValue`):

* `[NvApi] OverrideNvapiDll` — **the lever we already falsified** (`REPORT.md:27`)
* `[NvApi] DontUseFakenvapiForXeLLOnNvidia` (our line 973)
* `[FSR] Fsr4Update` (561), `Fsr4ForceEnableInt8` (565), `Fsr4EnableDebugView` (575) → replacement keys
  `Fsr4GpuSupport`, `Fsr4Model`
* `[Sharpness] UseDepthAwareSharpen` (1087), `UseDASDepthAwareSharpen`, `DADepthIsLinear` (1101)
* whole `[Nukems]` section (our line 660) → functionality moved to
  `[FrameGen] FGNvngxReplacement=` (Config.cpp:91–101) + `[NvngxFG]`

Added (all default `auto`, no action required): `[fakenvapi] UseFakenvapi`, `ForceXeLL`, `ForceReflex`,
`ForceLatencyFlex`, `LatencyFlexMode`, `DisableOTA`; `[Spoofing] DxgiBlacklist` (ini:1115, comment:
*"does not work on Linux"*), `Vulkan*`, `DxgiVRAM`; `[FrameGen] FGNvngxReplacement`, `LowLatencyInput/Output`;
`[DLSSG] DynamicMFG/OverrideInterpolationCount/…`; `FpsOverlay*` (≈30 keys); `Magnifier*`.

**Net migration list if we move to v10 (file-level, our game ini):**

1. Keep every key listed as "kept ✔" above verbatim (values unchanged).
2. Delete or ignore the §2.3 removed keys (deletion recommended for clarity; they are harmless in place).
3. If we had used `[Nukems]`, re-express as `[FrameGen] FGNvngxReplacement=…` (we don't — `FGNvngxReplacement`
   absent from our ini).
4. `OverrideNvapiDll` has **no** v10 equivalent *key* — the functionality moved into the binary
   (`[fakenvapi] UseFakenvapi=auto`, default-on: `Config.h:746 SoftDefault{true}`).
5. Extract `OptiScaler/` subfolder **next to** `dxgi.dll` (we launch from `bin64/`, so `bin64/OptiScaler/…`)
   — required for FFX/XeSS/D3D12-core loader paths (§1 package listing).

### 2.4 Other behavioral deltas that could hit us

* **Fakenvapi integrated** into OptiScaler (v10 WIP changelog lines 22–23; source tree
  `OptiScaler/nvapi/fakenvapi/nvapi_calls.cpp` exists in the clone) — bundled standalone
  `fakenvapi.dll` is no longer shipped or needed; if `bin64/fakenvapi.dll` exists it is likely unused
  (whether v10 still *optionally* loads an external one: **not found**).
* **New UI input system** (v10 changelog) — "some games might still act up where the overlay keyboard
  won't work, mouse will; Close button workaround".
* **All bundled helper DLLs moved into `OptiScaler\` subdir** ("no longer overwrites game files",
  v10 changelog MAJOR §1) — changes our staging layout only.
* Crimson-Desert-specific default preserved: `GameConfigs.cpp:553–557`
  (`crimsondesert.exe → DxgiSpoofing = false`, "unsupported GPU" error) in v10, equivalent to
  v0.9.4 `Quirks.h:328–330` (`QUIRK_ENTRY("crimsondesert.exe", GameQuirk::DisableDxgiSpoofing)`).

---

## 3. Does v10 carry *stronger* NVAPI / Streamline / dxvk-nvapi spoofing fixes?

From the v10 WIP changelog (published as the `nightly` release body in the main repo,
API `.../releases/tags/nightly`, 7,439 chars; mirrors https://github.com/optiscaler/OptiScaler/releases):

| line (body) | entry |
|---|---|
| 107 | "Use DXVK interop for GPU info detection (@FakeMichau, @Astyyyyy)" |
| 114 | "**Reworked spoof skipping** in order to avoid some race condition issues (@FakeMichau)" |
| 115–116 | "**Reworked spoof skipping**…" / "Kill native AL2 if SL Reflex is hooked" |
| **118–119** | **"Fixed GPU probe recursion with dxvk-nvapi (@levzzz5154) — *Fixes adapter identification on Linux*"** ← that's PR #1126 |
| 22–23 | "**Integrated Fakenvapi** into OptiScaler … No longer needs `fakenvapi.dll` bundled" |
| 17–18 | "Streamlined Fetcher BAT" (downloads Streamline SDK dlls) |

* Streamline driver spoof still present in v10: `Streamline_Hooks.cpp:890, 910, 1860`
  (`caps->driverVersionMajor = 999`) — matches the 999.99 we already observe
  (`levers/lever1-run.log:15`).
* New Linux-relevant `[Spoofing] DxgiBlacklist` macro-list exists but is documented
  "does not work on Linux" (template line 1115).
* Fixes mentioning **"black"** (black screen) in the v10 changelog: **not found** (grep `-i black` = 0 hits).
* Issue #1101 (the bug #1126 fixes): closed 2026-09-01T21:07:27Z, title
  "DLSS hidden + falls back to FSR2 on Linux/Proton (dxvk-nvapi recursion in getAllGpus)";
  the reporter reproduced it on **OptiScaler v10.0.0-dev (312c7df2, 20260813)** — i.e. the *dev/v10* line,
  not 0.9.x (issue body, `api.github.com/repos/optiscaler/OptiScaler/issues/1101`).

---

## 4. Crimson Desert duality — what upstream says

* **README.md / Changelog.md / Issues.md / Config.md / Spoofing.md in the repo: "Crimson"
  mention → not found** (grep over the master clone docs returned nothing).
* **Documentation lives in the wiki**, Compatibility-List
  (https://github.com/optiscaler/OptiScaler/wiki/Compatibility-List, line 158):
  `Crimson Desert | ✅ | DLSS, FSR3.1/4 | …` → links the per-game page.
  https://github.com/optiscaler/OptiScaler/wiki/Crimson-Desert (fetched):
  * "Last Tested Version **0.9.4**  Filename dxgi.dll, OptiScaler.asi  OS W11 24H2  GPU RX 9070 XT, RTX 3070"
  * "Upscaler Inputs **DLSS**, FSR3.1/4, XeSS — FG Inputs FSR 3.1 FG, OptiFG, DLSSG via Streamline"
  * Known issue: "DLSSG via Streamline inputs don't provide a HUDless resource → HUD ghosting; use
    XeFG/FSR4 FG output … game now has native XeSS and XeFG, recommended to use **XeSS inputs** …
    Disable FG in-game before selecting OptiFG to avoid black screens."
  * **No statement anywhere in upstream docs that "the in-game DLSS option must stay selectable"
    for Crimson Desert → not found.** (Wiki treats DLSS as an *available input*; the requirement that
    the game's own DLSS entry stays selectable is a constraint of *our* port design — the game feeds
    Streamline `sl.dlss_d`, which our `_nvngx.dll` satisfies — evidenced in `REPORT.md` D2 notes:
    `sl.dlss requires driver 512.15.0 … detected 0.0.0` comes from the game's `sl.dlss_d` plugin.)
* Wiki Known-Issues (`wiki/Known-Issues.md`): only Linux-specific entry is
  "**Shader Compilation error on Linux**" (RCAS/Reactive Mask Bias/Output Scaling need `d3dcompiler_47`
  via Winetricks/Protontricks, lines 217–223). No Linux/NVAPI entry, no Crimson entry.

---

## 5. Feasibility of each option

### (A) Stay on v0.9.4 + ini levers
* **Cannot contain #1126** — `IdentifyGpu.cpp` never existed in the 0.9 line (§1). Local checkout
  `/tmp/opencode/OptiScaler/OptiScaler/misc/` = `FrameLimit.cpp/h, Quirks.h` only;
  `grep -r IdentifyGpu` over local tree = 0 hits.
* v0.9.4 has its own dxvk-nvapi guard instead: `dllmain.cpp:1704–1722` detects
  "NVAPI Open Source Interface (DXVK-NVAPI)" and **skips `NvAPI_Initialize`** ("calls CreateDxgiFactory
  which we can't do because we are inside DLL_PROCESS_ATTACH"), using `EnumDisplayDevices` VEN_10DE instead.
  The `ScopedSkipDxgiLoadChecks` primitive exists (`OptiScaler/State.h:394`) but is used in proxies/hooks,
  not around NVAPI init.
* Our current evidence: levers pass g1–g4 and c1/c3/c5, but g5 still fails
  `c2_no_failed_NGX=bad`, `c4_no_missing_ctx=bad` with `driverDetected=999.99`
  (`tools/scenario/levers/lever1-run.log:14-15`, `lever2-run.log:14`) ⇒ the residual failure is
  **NGX-context, not NVAPI enumeration** — i.e., #1126 alone would not have closed our remaining gap.

### (B) Replace `dxgi.dll` with a nightly + ini migration
* Cheapest way to *actually receive* #1126 and the other Linux fixes (§3). Byte-shift: one
  26.7 MB `OptiScaler.dll`→`dxgi.dll` rename + extract `OptiScaler\` subdir + §2.3 ini edits.
* Choose: `nightly-20260902` (smallest delta from the fix, pinned) or `nightly-20261030` (latest,
  324 commits after the merge, includes all Oct GPU-detection/FSR4 Vulkan work).

### (C) Build from source
* **Toolchain: MSVC-only.** `OptiScaler.sln` + `OptiScaler/OptiScaler.vcxproj`;
  `PlatformToolset v143` (vcxproj:28,35,42), `WindowsTargetPlatformVersion 10.0` (:22),
  `LanguageStandard stdcpplatest` (:83). **No `CMakeLists.txt`, no `vcpkg.json`** (`ls` = No such file),
  no cross/mingw files. Submodules required (`.gitmodules`: simpleini, unordered_dense, xess,
  Vulkan-Headers, spdlog, FidelityFX-SDK(-v2), magic_enum).
* **No Linux CI anywhere** — every workflow is `runs-on: windows-latest`:
  `build.yml:35`, `just_build.yml:29`, `just_build_no_signature.yml:29`, `release_debug.yml:29`,
  `test.yml:29`, nightly `build.yml:24`. Only non-build workflow is `clang-format.yml`.
  ⇒ **cannot build on Linux with mingw** on any existing recipe; "build on Linux" = not found.
* Realistic (C): fork `optiscaler/OptiScaler`, `workflow_dispatch` on
  **`just_build_no_signature.yml`** (lines 97–112: `msbuild … Release`, then
  `actions/upload-artifact` of `x64\Release\a\OptiScaler.dll` — no SignPath secret needed; plain
  `just_build.yml` additionally needs `secrets.SIGNPATH_API_TOKEN`, lines 113–121).
  Pinned sha to build: `c983a500335134ecff512bfcdadcf912d1286547` (contains #1126) — or current
  master `e7e3872d410f704479d94d318d68d163b6b58947` if we also want the Oct fixes.
* Backporting #1126 *into* v0.9.4 (option C′): **not sensible** — the patch modifies
  `IdentifyGpu::queryNvapi` which doesn't exist there; the analogous 0.9.4 code
  (`dllmain.cpp:1704–1722`) already special-cases dxvk-nvapi, so there is no verbatim code to drop in.

---

## 6. Recommendation

1. **Primary: (B).** Take **`nightly-20261030`** (or, to minimize churn, a pinned
   `nightly-20260902`), rename `OptiScaler.dll → bin64/dxgi.dll`, extract `bin64/OptiScaler/`,
   and apply the §2.3 ini migration on top of a *fresh* v10 `OptiScaler.ini`
   (do **not** reuse our 0.9.4 ini verbatim — it carries dead keys and a renamed section).
   Pros: only way to get #1126 + "Reworked spoof skipping" + DXVK-interop GPU detection + integrated
   fakenvapi; upstream's own Crimson-Desert path (GameConfigs default) is preserved; our
   `NvngxPath` mechanism is byte-for-byte still supported (`Config.cpp:819`,
   `NVNGX_Proxy.h:602/611`, `Util.cpp:911–927`).
2. **Contingency: (A)** — keep 0.9.4 while the harness gap persists: #1126 would fix something we
   don't have (the v10-only recursion). Continue ini levers for the `c2/c4` NGX-context failures;
   note `OverrideNvapiDll` is not available if we later move to v10.
3. **(C) only if we need a custom patch** — fork + GitHub Actions `just_build_no_signature.yml`
   (windows-latest/msbuild, no mingw/Linux path), sha `c983a500335134ecff512bfcdadcf912d1286547`.
   Highest cost (v143 toolchain + submodules + SignPath-less artifact flow) for a fix that is already
   published as a nightly — **not worth it for #1126 alone**.

**Risk notes for (B):**
* Everything after 0.9.4 is branded `v10.0.0-pre1` — a moving prerelease (daily builds); pin a date.
* Config migration required (§2.3): dropped `[NvApi] OverrideNvapiDll`,
  renamed `[Nukems]→[NvngxFG]`, `[FrameGen] FGNvngxReplacement`, new `[fakenvapi]`.
* Layout change: helper DLLs now under `OptiScaler\` next to `dxgi.dll` (extraction target = `bin64/`).
* `_nvngx.dll` pathing unaffected (same absolute-path override + registry fallback), but verify after
  upgrade with the harness: 0.9.4 ini line 681 value must be re-applied to the new template.
* Menu input system was rewritten (possible overlay keyboard issues) — cosmetic, "Close" button workaround.
* fakenvapi now internal ⇒ our old `bin64/fakenvapi.dll` becomes dead weight; `UseFakenvapi=true` default.
* Re-validate Crimson-Desert quirk still applies (`GameConfigs.cpp:556 DxgiSpoofing=false`).

---

## 7. Explicit "not found" list

* Releases after v0.9.4 in `optiscaler/OptiScaler`: **not found (none)**; v0.9.5/0.10: **not found**.
* Backport of #1126 onto `release/0.9` or any stable tag: **not found** (branch lacks `IdentifyGpu.cpp`).
* "Crimson"/"Crimson Desert" in upstream `README.md`, `Changelog.md`, `Issues.md`, `Config.md`,
  `Spoofing.md`: **not found** (only `misc/Quirks.h:328-330` (0.9.4) / `misc/GameConfigs.cpp:553-557` (v10)
  and the wiki page).
* Upstream statement that the in-game DLSS option must remain selectable (Crimson or general): **not found**.
* "black"/black-screen fixes in the v10 changelog: **not found**.
* Linux/mingw/CMake build recipe for OptiScaler: **not found** (CMake/vcpkg files absent; all CI windows-latest).
* `NvngxPath` in upstream `Config.md`: **not found** (documented only in `OptiScaler.ini` + source).
* Whether v10 still loads an *external bundled* `fakenvapi.dll` when present: **not found** in source/changelog.

## 8. Key URLs

* Repo releases: https://api.github.com/repos/optiscaler/OptiScaler/releases
* Nightlies: https://github.com/optiscaler/OptiScaler-nightly/releases
* PR #1126: https://github.com/optiscaler/OptiScaler/pull/1126 (merge `c983a500335134ecff512bfcdadcf912d1286547`)
* Issue #1101: https://github.com/optiscaler/OptiScaler/issues/1101
* v10 WIP changelog (nightly release body): https://github.com/optiscaler/OptiScaler/releases/tag/nightly
* Wiki: https://github.com/optiscaler/OptiScaler/wiki/Crimson-Desert ·
  https://github.com/optiscaler/OptiScaler/wiki/Compatibility-List ·
  https://github.com/optiscaler/OptiScaler/wiki/Known-Issues
* First-with-fix asset:
  https://github.com/optiscaler/OptiScaler-nightly/releases/download/nightly-20260902/OptiScaler_v10.0.0-pre1_20260902.7z
* Latest asset:
  https://github.com/optiscaler/OptiScaler-nightly/releases/download/nightly-20261030/OptiScaler_v10.0.0-pre1_20261030.7z