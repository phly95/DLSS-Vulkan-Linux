# g5 lever run #1 — 2026-10-31 (headless Streamline harness, `run_harness.sh --sl`)

Session: 2 sequential levers, one config change each, one harness run each, pristine baseline
restored between runs. Repo tracked files: **0 modified** at the end (`git diff --stat` empty).
Only untracked leftovers are the mandated `.bak-g5lever-*` files and this `tools/scenario/levers/`
subdir (which I own).

Harness command (both runs, verbatim):

    bash /home/deck/DLSS-Vulkan-Linux/tools/optiharness/run_harness.sh --sl

Log of record: `tools/scenario/levers/lever1-run.log`, `tools/scenario/levers/lever2-run.log`.

Which ini the harness actually reads: `run_harness.sh:104` —
`cp "$TOOL/OptiScaler.harness.ini" "$STAGE/OptiScaler.ini"` into
`build/optiharness/harness/`, regenerated fresh **every run** (rm -rf $STAGE at line 90).
Therefore the lever-1 file MUST be `tools/optiharness/OptiScaler.harness.ini` (the staged copy is
recreated; the GAME's `bin64/OptiScaler.ini` is never read — OptiScaler loads the ini next to the
loaded dxgi.dll, i.e. the staging dir).

## Baseline g5 reference (committed source)

`tools/optiharness/README.md:150-171`, "Today's measured baseline (`run_harness.sh --sl`, 2026-10-31)":

    g5 SL boot : FAIL   c1_getSystemCaps_nvapi=ok  c2_no_failed_NGX=bad(x2)  c3_driver=ok
                         c4_no_missing_ctx=bad(x2)  c5_sl.dlss_loaded=ok
    SLDLSS slIsFeatureSupported(kFeatureDLSS)  result=32 (eErrorFeatureNotSupported)
    SLDLSS slIsFeatureLoaded(kFeatureDLSS)      result=31 (eErrorFeatureMissing) loaded=0
    SLDLSS slGetFeatureRequirements(...)        driverDetected=999.99.0 driverRequired=512.15.0
    [sl-cb] ... getSystemCaps] NVIDIA driver 999.99
    [sl-cb] ... getNGXFeatureRequirements] ngxResult failed 0xbad00002      (x3, like the game)
    [sl-cb] ... slOnPluginStartup] Failed to initialize NGX, any SL feature requiring NGX will be
             unloaded and disabled                       <- c2 bad, game line verbatim
    [sl-cb] ... dlssEntry.cpp:923[slOnPluginStartup] Missing NGX context - DLSSContext cannot run
                                                                  <- c4 bad, game line verbatim

Live baseline log snapshot (before lever 1 overwrote `build/optiharness/out/`), archived at
`tools/scenario/levers/baseline-20261031/{client.log,g5_evidence.txt,run.stdout.log,summary.txt}`:

    Failed to initialize NGX  x2   Missing NGX context x2   Loaded plugin 'sl.dlss' x2
    NVIDIA driver 999.99      x2   arch=0x130 (Ada) x2      'NVAPI failed to initialize' x0
    'requires driver ...' line x0  (so g5 c3 = ok by absence)

---

## Lever 1 — [Spoofing] DxgiFactoryWrapping=false

### Current value before the change (verbatim)

`tools/optiharness/OptiScaler.harness.ini` has **no `[Spoofing]` section at all** — the key is
**absent**, so it collapses to the compiled default
(`reference-nr-vulkan/.../OptiScaler/Config.h:598`):

    CustomOptional<bool> DxgiFactoryWrapping { false };

Read path `Config.cpp:783 DxgiFactoryWrapping.set_from_config(readBool("Spoofing", "DxgiFactoryWrapping"))`.
So the *effective* value was **false** already. For reference, the GAME's (never-read) ini says at
`bin64/OptiScaler.ini:878` exactly `DxgiFactoryWrapping=false` with the comment
"; true or false - Default (auto) is false".

Per instruction step 3 I still set the key **explicitly** (the key was absent, not literally
present-as-false), so lever 1 is an explicit-value run that doubles as a no-op confirmation.

### Exact change (one change, one file)

    + [Spoofing]
    + DxgiFactoryWrapping=false

appended after `[OutputScaling] Enabled=false` in `tools/optiharness/OptiScaler.harness.ini`
(every other byte identical). Backup:

    tools/optiharness/OptiScaler.harness.ini.bak-g5lever-1-20261031   (md5 df1208a9fa0050281e24381cc4a319a7,
                                                                      orig mtime Oct 30 21:36 preserved by cp -p)

### Run

    command : bash tools/optiharness/run_harness.sh --sl
    exit    : 1          (opti_client rc=0; non-zero solely because g5=FAIL, by design)
    Prefix  : "game prefix free: /home/.../compatdata/3321460 (default game compatdata)"

    g1 OptiScaler : PASS   create=1 loaded_from_nvngx=1 running_on_nvidia=1 fsr_fallback=0 bad00000=0
    g2 port log   : PASS   ok=12/12 fail=0 'fmt col=0 mv=1 dep=0'x12 depth19_refusal=0 slotdepth@eval1=0
    g3 non-black  : PASS   threshold 'every frame nz >= 0.10', viol=0/12, min nz=1.000000
    g4 timing     : PASS   EVAL_MS n=11 median=4.4057 ms p95=4.5791 ms | GPU_EVAL_MS n=11 median=6.9522 ms p95=8.5994 ms (frames 1..)
    g5 SL boot    : FAIL   c1_getSystemCaps_nvapi=ok  c2_no_failed_NGX=bad(x2)  c3_driver=ok  c4_no_missing_ctx=bad(x2)  c5_sl.dlss_loaded=ok
                   nvapi: getSystemCaps] NVIDIA driver 999.99; no 'requires driver' line; sl-requirements: driverDetected=999.99.0 driverRequired=512.15.0; evidence: [sl-cb]=124 SLDLSS=3 slInit=1 ...
    HARNESS RESULT: FAIL  (opti_client rc=0, exe summary RESULT=PASS, g5=FAIL)

### Grep hits (g5_evidence.txt of the lever-1 run)

    getNVDAArchitecture  Found GPU 0, arch=0x130                 (Ada / AD100 family, NOT Pascal/TU100)
    getSystemCaps        Adapter 0 architecture 0x130 impl 0x2 revision 0xffffffff
    getSystemCaps        NVIDIA driver 999.99                    (x2 sources → x1 dedup in detail line)
    SLDLSS slGetFeatureRequirements driverDetected=999.99.0 driverRequired=512.15.0   (999.99 >= 512.15 → c3 ok)
    pluginManager mapPlugins  Loaded plugin 'sl.dlss' - version 2.14.1.98614dad6 PRODUCTION
    commonEntry slOnPluginStartup  Failed to initialize NGX ...                       (x2 → c2 bad)
    dlssEntry.cpp:923 slOnPluginStartup  Missing NGX context - DLSSContext cannot run (x2 → c4 bad)
    getNGXFeatureRequirements ngxResult failed 0xbad00002     (x3)
    OptiScaler: 'OverrideNvapiDll not set, setting it to: false'
    OptiScaler: 'fakenvapi::Init Trying to get fakenvapi-specific functions' / 'fakenvapi::Init Failed to initialize fakenvapi'
    OptiScaler: 'isNvidia Using dxvk-nvapi'
    ota.cpp:743 readServerManifest  Failed to open manifest file at: C:\ProgramData\NVIDIA/NGX/models/nvngx_config.txt
    (no 'NVAPI failed to initialize' line, no 'requires driver' line)

### Verdict + restoration

**g5 still FAIL, byte-for-byte the same subcheck pattern (c2 bad x2, c4 bad x2) → lever 1 is a
confirmed NO-OP** (explicit `false` == default `false`). g1–g4 unchanged green.

Restoration: `cp -p .bak-g5lever-1-20261031 → OptiScaler.harness.ini`, then

    cmp tools/optiharness/OptiScaler.harness.ini tools/optiharness/OptiScaler.harness.ini.bak-g5lever-1-20261031
    -> rc=0 (identical); md5 back to df1208a9fa0050281e24381cc4a319a7; `git diff` for it empty

---

## Lever 2 — fakenvapi deployed as the prefix's `nvapi64.dll`

### Target discovery

    live prefix nvapi64.dll : /home/deck/.local/share/Steam/steamapps/compatdata/3321460/
                              pfx/drive_c/windows/system32/nvapi64.dll   (PRESENT, 1 847 310 B)
    md5 before              : 035510e7e87ab56674a7e2887abd28db
    == md5 of Proton 10.0 runtime copy:
      /home/deck/.local/share/Steam/steamapps/common/Proton 10.0/files/lib/wine/nvapi/x86_64-windows/nvapi64.dll
    source fakenvapi.dll    : <game>/bin64/fakenvapi.dll  (415 232 B, md5 254791686889e4b1a81e99df477fbc90)
    both export exactly one symbol: nvapi_QueryInterface  → drop-in ABI-compatible in name only

### Where I put it and why (read `run_harness.sh` first)

`run_harness.sh:152-169`: `RUN_PREFIX=$PREFIX` (the LIVE game prefix) unless a wineserver owning
`compatdata/3321460` is alive, in which case the clone is made **at run start** from `$PREFIX`
(`cp -a "$PREFIX"/. "$CLONE"/`, line 165) — i.e. *after* my deployment, so the clone would inherit
it too. Both lever-2 log and prior run printed
`game prefix free: /home/.../compatdata/3321460 (default game compatdata)` → no clone, the live
prefix is what the game process loads from (Proton loads `nvapi64.dll` from `system32`; Opti's log
confirms `LibraryLoadHooks::LoadLibraryCheckW nvapi64.dll call!`). Conclusion recorded: **deploy to
the live prefix's `system32/nvapi64.dll` = correct target in both the busy and free cases.**
The prefix is outside the repo, so the repo-modification rule is not implicated.

### Change + backup

    BACKUP : .../pfx/drive_c/windows/system32/nvapi64.dll.bak-g5lever-2-20261031
             (1 847 310 B, mtime Oct 31 01:26 preserved by cp -p, md5 035510e7e87ab56674a7e2887abd28db)
    CHANGE : cp -p <game>/bin64/fakenvapi.dll → .../system32/nvapi64.dll   (md5 25479168… verified)

**fakenvapi.ini observation (not fixed, per instruction):** no `fakenvapi.ini` (nor
`fakenvapi.log`) exists in `system32/` after deployment —

    ls: cannot access '.../system32/fakenvapi.ini': No such file or directory
    ls: cannot access '.../system32/fakenvapi.log': No such file or directory

`fakenvapi.ini` lives only in the game dir (`enable_logs=1` etc.), so a system32-mounted
fakenvapi runs on defaults (logs off). Observation only.

### Run

    command : bash tools/optiharness/run_harness.sh --sl
    exit    : 1          (opti_client rc=0; g5=FAIL only)
    Prefix  : "game prefix free: .../compatdata/3321460 (default game compatdata)"

    g1 OptiScaler : PASS   create=1 loaded_from_nvngx=1 running_on_nvidia=1 fsr_fallback=0 bad00000=0
    g2 port log   : PASS   ok=12/12 fail=0 'fmt col=0 mv=1 dep=0'x12 depth19_refusal=0 slotdepth@eval1=0
    g3 non-black  : PASS   threshold 'every frame nz >= 0.10', viol=0/12, min nz=1.000000
    g4 timing     : PASS   EVAL_MS n=11 median=4.3999 ms p95=7.4035 ms | GPU_EVAL_MS n=11 median=7.0181 ms p95=8.0300 ms (frames 1..)
    g5 SL boot    : FAIL   c1_getSystemCaps_nvapi=ok  c2_no_failed_NGX=bad(x2)  c3_driver=ok  c4_no_missing_ctx=bad(x2)  c5_sl.dlss_loaded=ok
                   nvapi: getSystemCaps] NVIDIA driver 999.99; no 'requires driver' line; sl-requirements: driverDetected=999.99.0 driverRequired=512.15.0; evidence: [sl-cb]=124 SLDLSS=3 slInit=1 ...
    HARNESS RESULT: FAIL  (opti_client rc=0, exe summary RESULT=PASS, g5=FAIL)

### Grep hits (lever 2, `tools/scenario/levers/lever2/g5_evidence.txt`)

    arch / Ada / TU100 / Pascal / 0x130:
      getNVDAArchitecture  Found GPU 0, arch=0x130          x2
      getSystemCaps        Adapter 0 architecture 0x130     x2
      NvApiHooks::hkNvAPI_GPU_GetArchInfo Original arch: 130 impl: 2 rev: FFFFFFFF!   x2
      (16 case-insensitive 'ada' hits, 0 'TU100', 0 'Pascal')
    driver inside SL lines:
      getSystemCaps] NVIDIA driver 999.99                   x2
      SLDLSS slGetFeatureRequirements driverDetected=999.99.0 driverRequired=512.15.0
      (NOT 999.99 from fakenvapi — README's Vulkan_Spoofing.cpp:177 strcpy(ddp->driverInfo,"999.99"))
    doom lines (unchanged vs baseline):
      Failed to initialize NGX      x2      Missing NGX context   x2
      Loaded plugin 'sl.dlss'       x2      ngxResult failed 0xbad00002 x3
      'NVAPI failed to initialize'  x0      'requires driver' line x0
    identity lines:
      fakenvapi::Init Failed to initialize fakenvapi ; OverrideNvapiDll not set, setting it to: false
      isNvidia Using dxvk-nvapi ; LoadLibraryCheckW nvapi64.dll call!

### CRITICAL mechanical finding — the deployment did not survive

Immediately after the run the prefix file was back to Proton's binary:

    md5 035510e7e87ab56674a7e2887abd28db   (= Proton 10.0's nvapi64.dll),
    mtime Oct 31 01:28 → rewritten DURING the run (start ~01:28:2x), 1 847 310 B
    (sibling openvr_api_dxvk.dll also rewritten 01:28 → Proton re-copies its
     dxvk/dxvk-nvapi DLL set into system32 on EVERY `proton run`)

So Proton 10.0's per-launch staging **clobbered the fakenvapi copy before the process loaded
nvapi64.dll**; the run therefore measured Proton's dxvk-nvapi again. Lever 2 = measured no-op, and
its g5 result cannot be attributed to fakenvapi at all. This is an environment/setup fact, not a
gate result — recorded so no one mis-reads "unchanged" as "fakenvapi doesn't help".

### Restoration

    cp -p nvapi64.dll.bak-g5lever-2-20261031 → nvapi64.dll
    cmp <file> <backup> -> rc=0 (identical); md5 035510e7e87ab56674a7e2887abd28db
    (backup file left in place: .../system32/nvapi64.dll.bak-g5lever-2-20261031)

Both lever files restored; `git status --short` shows only
`?? tools/optiharness/OptiScaler.harness.ini.bak-g5lever-1-20261031` and `?? tools/scenario/`
(untracked), `git diff --stat` empty.

---

## Verdicts

| lever | config change | exit | g5 | subchecks | attribution |
| --- | --- | --- | --- | --- | --- |
| 1 | `[Spoofing] DxgiFactoryWrapping=false` (was absent → default false) | 1 | FAIL | c1 ok, c2 **bad x2**, c3 ok, c4 **bad x2**, c5 ok | no-op (key was already effectively false); none moved |
| 2 | `fakenvapi.dll` copied over prefix `system32/nvapi64.dll` | 1 | FAIL | identical to baseline | **clobbered by Proton's per-run DLL re-copy** — lever never took effect |

No subcheck moved on either run: the decisive doom lines (`Failed to initialize NGX` x2,
`Missing NGX context` x2) reproduce verbatim, g1–g4 stay green.

## Ranked recommendation for the next lever

1. **`[Libraries] OverrideNvapiDll=true` (or the exact key Opti's shipped build reports) in
   `tools/optiharness/OptiScaler.harness.ini`** — Opti itself logs `OverrideNvapiDll not set,
   setting it to: false` and then `fakenvapi::Init Failed to initialize fakenvapi`; this lever
   routes nvapi64 through OptiScaler's own override *inside the process*, so it cannot be erased
   by Proton's per-run file copy (the failure mode that killed lever 2). Single key, one run.
2. **Make the fakenvapi deployment survive before re-trying it as a prefix file** — e.g. deploy
   *after* Proton's staging step, or suppress that step (PROTON env / prefix stamp). Only worth
   doing once mechanism (1) is measured, because today's negative data point for fakenvapi is
   confounded (the file never loaded).
3. **`[Spoofing] Registry=true`** — SL's `getNGXFeatureRequirements` fails `0xbad00002` and SL
   reads registry-hosted NGX/CUDA state; opti's key exists (`Config.cpp:798`) and is default off.
4. **NGX discovery path for sl.common** — `readServerManifest Failed to open
   C:\ProgramData\NVIDIA/NGX/models/nvngx_config.txt`, `NGX Updater not available`, and the app-id
   mismatch (SL `application ID 101640911` vs Opti `NVSDK_NGX_D3D12_Init_Ext AppId: 322376503`).
   Next step here is a *file-staging* lever (put a valid `nvngx_config.txt` under the prefix's
   `drive_c/ProgramData/NVIDIA/NGX/models/`, outside the repo) rather than an ini key.
5. **Do not spend a run on `[Spoofing] DxgiFactoryWrapping` again** (measured no-op) and keep
   `SL_NVAPI_VER` untouched (already a validated negative).