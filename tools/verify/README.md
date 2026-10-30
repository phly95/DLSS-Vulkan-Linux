# tools/verify: the referee (frozen)

Everything in this directory was frozen in the commit whose message starts with `verify: freeze v2`. It
does not change afterwards (`run_all.sh` V10 checks it); a real bug in it is a question for the user.

| file | what |
| --- | --- |
| `run_all.sh` | the referee, on the RX 9070 XT under SteamOS: checks V1-V10 (list in its header) |
| `offline_check.sh` | what can be checked without the card (steps O1-O7 in its header); never a verdict on the port |
| `eval_verify.py` | V4 (accuracy), V5 (a degraded run must fail V4), V7 (RADV statistics), V8 (GPU time) |
| `make_cells.py`, `texfmt.py` | the referee's variant inputs: gather format codes, output-size motion vectors, parameters, corrupted copies |
| `r2scan.py` | NVIDIA / NeuralHub content in tracked files, a build, a package or a game folder |
| `diffdump.py` | one kernel's outputs against the oracle's dump of the same launch |
| `oracle_dump.hpp` | the per-launch dumper `srref` / `rrref --dump` use |
| `vkprobe.cpp` | the referee's own view of the Vulkan device (V7) |
| `ngx_client/` | the Windows program that drives SR / RR through NGX under Proton (V9) |
| `hipstub/` | declarations only, so `g++ -fsyntax-only` can check the oracles without ROCm |

## What the port must provide

**The runner** (`build/vk/vk_runner`, or `PORT_RUNNER=`):

    vk_runner --mode sr|rr --preset M|L|F --weights DIR --in FRAMES --out OUT

- `DIR` is a packed weight directory exactly as `tools/sr/sr_pack.py` (SR, `stage0..9.bin`, `enc0.bin`,
  `dec0.bin`, `params.json`) or `tools/rr/rr_pack.py` (RR, `blob0..15.bin`, `bicubic.bin`, `params.json`)
  writes it. The referee builds it from the user's DLL for every run and gives the port corrupted copies
  too (V5): the weights must be read from `DIR`, nowhere else.
- `FRAMES` is a frames directory (below), with every parameter and format code it may carry.
- It writes `OUT/out_<i>.f32` (RGBA32F, output size) for every frame and prints, each on its own line:
  `VK_DEVICE: <VkPhysicalDeviceProperties::deviceName>`, `PIPELINES_CREATED: <n>` (every compute
  pipeline it created) and `GPU_FRAME_MS <i> <ms>` per frame, measured with timestamp queries around
  all of that frame's GPU work. Exit 0.
- It runs with `VK_ICD_FILENAMES` set to the system RADV ICD, implicit layers off,
  `RADV_DEBUG=shaderstats` and `MESA_SHADER_CACHE_DISABLE=true`: Mesa prints one statistics block per
  pipeline, which V7 reads.

**Its build output** is under `build/vk/` (programs) and `build/shaders/` (the SPIR-V it loads); V6
scans both, and the package, for NVIDIA / NeuralHub content.

**The package**: `build/package/install.sh <game folder> optiscaler` installs, `... remove` restores
the folder byte for byte. If the game needs environment variables, the installer writes them as
`KEY=VALUE` lines to `<game folder>/dlss5-amd-launch.env`. The game folder holds the user's
`nvngx_dlss.dll` / `nvngx_dlssd.dll`; CreateFeature must refuse when they are missing or not 310.9.1.0.
Under Proton, `ngx_client.exe` loads `_nvngx.dll`, then `nvngx.dll`, from the game folder.

**Shaders** (for `offline_check.sh`): `shaders/pipelines.json` lists every pipeline,
`{"pipelines": [{"name": N, "source": "<path under shaders/>", "defines": [...]}]}`; every `.comp` under
`shaders/` must be listed. Cooperative matrices and FP8 only through `shaders/include/nh_mm.glsl`, which
includes `reference-nr-vulkan/linux/shaders/rdna4/include/coopmm.glsl` and selects the lavapipe build
when `NH_F16` is defined (offline_check adds `-DNH_F16=1`). `python3 shaders/tests/make_test.py N DIR`
writes `DIR/launch.json` and its input files for pipeline `N`; `bash src/vk/build.sh` builds
`build/vk/kernel_run`, which runs one dispatch:

    kernel_run --spv FILE --launch DIR/launch.json --out OUTDIR

`launch.json`: `{"groups": [x, y, z], "push": "<hex>", "bindings": [{"binding": b, "file": "in.bin"} |
{"binding": b, "bytes": n, "output": "name", "type": "u8|f16|f32|u32"}]}`; inputs are read relative to
`DIR`, each output is written as `OUTDIR/<name>.bin`, and it prints `VK_DEVICE: <name>`.

## Frames directory

`frame.json`: `{"render": [W, H], "output": [oW, oH], "frames": N}`; per frame `i` a `params_i.json` and
the game's textures, tightly packed (row pitch = width x texel size):

| | SR | RR |
| --- | --- | --- |
| parameters | `jitter`, `jitter_prev`, `mv_scale`, `exposure`, `pre_exposure`, `reset`, `depth_inverted`; optional `fmt_col`, `fmt_mv`, `fmt_dep`, `mvhi`, `ldr`, `auto_exposure`, `ae_exact_sum` | `jitter`, `mv_scale`, `reset`, `depth_inverted`, `hw_depth`; optional `fmt_col`, `fmt_mv`, `fmt_dep`, `fmt_nrm`, `fmt_alb`, `fmt_spa`, `fmt_rgh`, `normals_unorm` |
| textures, default format | `col_i.f16` RGBA16F, `mv_i.f16` RG16F, `dep_i.f32` R32F | `col_i.f16`, `nrm_i.f16`, `alb_i.f16`, `spa_i.f16` RGBA16F; `mv_i.f16` RG16F; `dep_i.f32`, `rgh_i.f32` R32F |
| another format code `c` | `<name>_i.c<c>` | `<name>_i.c<c>` |

Codes and texel layouts: `texfmt.py` (from NeuralHub's gather kernels and host tables,
`docs/sr/HOST-RE.md`, `docs/rr/HOST-RE.md`). With `mvhi` the SR motion vectors are at output size.
Motion vectors are NGX's: render pixels, previous position = current + mv. Ground truth (`gt_i.f32`,
RGBA32F, output size) is in a separate directory the port is never given.

## Restoring the inputs on a machine

From `phly95/dlss5-private-storage` (its README), plus the two DLLs only V9's wrong-DLL test uses:

    S=/path/to/dlss5-private-storage
    cp $S/nvidia/310.9.1.0/nvngx_dlss.dll  build/re/nvngx_dlss_310.9.1.dll
    cp $S/nvidia/310.9.1.0/nvngx_dlssd.dll build/re/nvngx_dlssd_310.9.1.dll
    cp $S/nvidia/310.9.16/nvngx_dlss.dll   build/re/nvngx_dlss_310.9.16.dll
    cp $S/nvidia/310.9.16/nvngx_dlssd.dll  build/re/nvngx_dlssd_310.9.16.dll

NeuralHub's own code objects are bring-your-own: its license forbids redistributing it, so this
repository holds neither the installer nor anything extracted from it. Download
`NeuralHub_v0.8.0_setup.exe` from the official release
<https://github.com/danielblnc/DLSS-NR-on-AMD/releases/tag/v0.8.0>, then:

    NH_INSTALLER=/path/to/NeuralHub_v0.8.0_setup.exe tools/exe/fetch_neuralhub.sh build/nh

Without `NH_INSTALLER` the script downloads that release asset itself (`curl -L`). Either way it
checks the SHA-256 it pins before unpacking and stops with expected vs actual on a mismatch. It
writes `build/nh` only (git-ignored; a destination the repository tracks is refused): the `<net>.co`
code objects, `neuralhub_host.dll` and `manifest.json` from `tools/exe/extract.py`, plus the
reference inputs `host_strings.txt.xz` (needs `strings`) and, when an `llvm-objdump` with
`--mcpu=gfx1201` is installed, `kernels.json` and the `*.s.xz` disassembly — those last steps print
the exact command to run later and exit 0 when the tool is missing. Everything that runs NeuralHub's
kernels reads `build/nh` (`DLSS5_AMD_NH=DIR` points them at another directory inside the checkout):
`run_all.sh`, `tools/gpu/collect_oracle.sh`, `tools/sr/sr_probe.py`, `tools/re/patchprobe.py` and
`tools/re/fn.sh`; `run_all.sh` and `offline_check.sh` invoke `fetch_neuralhub.sh` when it is empty.
A copy of any of NeuralHub's twelve files anywhere in the tree fails R2: `r2scan.py` bans their
digests, it does not pin them any more.

`run_all.sh` also needs Blender 4.2.3, the Sponza / DamagedHelmet assets and `localhost/rocm-blender:1`
(`tools/scenes/README.md`), `localhost/nrbuild:2` (`tools/hwtest/build_native.sh`), podman with
`docker.io/rocm/dev-ubuntu-24.04:7.2.4` for the oracles, and a Proton under `~/.local/share/Steam`.
