# DLSS-Vulkan-Linux

Add DLSS Super Resolution (presets M/L) and Ray Reconstruction (preset F) to games on Linux/Proton:
a Vulkan reimplementation of NeuralHub's DLSS SR/RR compute pipeline for AMD RDNA4 (plus the MIT
reference port for NR), shipping as `_nvngx.dll` — it exports `NVSDK_NGX_VULKAN_EvaluateFeature` and
`NVSDK_NGX_D3D12_EvaluateFeature` (the D3D12 one through vkd3d-proton's interop) — together with
SPIR-V shaders and weight symlinks, installed into a game folder by `install.sh`.

**Status:** the whole path works — measured under Proton 10.0 with
`tools/verify/ngx_client --api {vulkan,d3d12}`, where RR output is byte-identical to the native chain
on both APIs and SR matches it within its own run-to-run drift — but quality is not finished (as of
the last full referee run, 2026-10-28): in the V4 gate SR M and SR L pass while RR F sits 0.54 dB
under the mean bar; in the full 25-cell matrix only 5 static, default-format cells pass (the
format-variant cells fail at 4-19 dB). The performance campaign that followed (RR F 86.6 → 10.0 ms
per frame, SR M 6.04 → 4.36, SR L → 6.09) passed the quality guard's no-regression bars on every
landed patch, but did not re-run the V4/V8 gates — those verdicts predate it and the current numbers
are unmeasured. Measured numbers: [docs/PORT-STATUS.md](docs/PORT-STATUS.md); open
items: [docs/OPEN.md](docs/OPEN.md).

## Requirements

- Linux / SteamOS (built and measured on SteamOS 3.9; SteamOS's root is immutable and has no
  compiler, so builds run in podman containers and the binaries run on the host)
- AMD RX 9070 XT (RDNA4, GFX1201)
- Mesa 26.2+ RADV with `VK_KHR_cooperative_matrix` and FP8 (`VK_EXT_shader_float8`), Vulkan 1.3+
- Proton with vkd3d-proton for D3D12 titles; measured on Proton 10.0 (Proton 11 and Proton
  Experimental hang on this machine, see [docs/OPEN.md](docs/OPEN.md))
- Your own model weights, never committed: `nvngx_dlss.dll` 310.9.1.0 (SR presets M/L) and
  `nvngx_dlssd.dll` 310.9.1.0 (RR preset F) from a game you own; `nvngx_dlssnr.dll` for NR
- podman (reference harness and container builds), `7z` (installer extraction), and either `g++` +
  `x86_64-w64-mingw32-g++` on PATH or the podman build images created below

## 1. Build from a fresh clone

```bash
# 1. submodule + its pinned toolchain (glslang, Vulkan headers)
git submodule update --init                                # reference-nr-vulkan (MIT), pinned at c71b0f4
(cd reference-nr-vulkan && bash fetch_deps.sh && python3 linux/build/build_network.py rdna4)

# 2. python deps
python3 -m venv build/venv
build/venv/bin/pip install numpy pillow pefile msgpack capstone

# 3. (needed only for the oracle/referee reference runs — not to build the package or run a game)
#    NeuralHub's installer is bring-your-own: its license forbids redistribution and instructs
#    linking to the official release, so fetch NeuralHub_v0.8.0_setup.exe yourself from
#    https://github.com/danielblnc/DLSS-NR-on-AMD/releases/tag/v0.8.0 (sha256 aae53654…0b834),
#    then verify the hash, extract it, and regenerate the reference inputs into build/nh:
NH_INSTALLER=/path/to/NeuralHub_v0.8.0_setup.exe tools/exe/fetch_neuralhub.sh build/nh
#    (omit NH_INSTALLER to let the script download it; only §5's referee runs read build/nh —
#     building and using the package does not need NeuralHub at all)

# 4. reference harness: NeuralHub's kernels run unchanged under ROCm on this GPU
podman pull docker.io/rocm/dev-ubuntu-24.04:7.2.4
tools/hwtest/build_native.sh     # builds the g++ podman image once (used by step 5 if no host g++)

# 5. shaders + runners + package
bash src/vk/build.sh             # -> build/shaders/*.spv, build/vk/{kernel_run,vk_runner},
                                 #    build/package/{install.sh,_nvngx.dll}
```

Notes:

- `_nvngx.dll` needs `x86_64-w64-mingw32-g++` on PATH; `src/vk/build.sh` skips it silently without it.
- Optional device gate: `tools/hwtest/build_probe.sh && build/hwtest/hwprobe` exits non-zero if FP8 / cooperative-matrix support is missing.
- Offline shader work (no card): `bash tools/radv-offline/build.sh` unpacks the prebuilt Mesa for compiling SPIR-V for a GFX1201.

## 2. Pack the weights (once)

The game's weights are packed from your own DLLs into `build/`; `install.sh` symlinks these as
`DLSS5_AMD_WEIGHTS`. Put your DLLs at `build/re/nvngx_dlss_310.9.1.dll` and
`build/re/nvngx_dlssd_310.9.1.dll` first (see `tools/verify/README.md`, "Restoring the inputs"), then:

```bash
build/venv/bin/python -I tools/sr/sr_weights.py build/re/nvngx_dlss_310.9.1.dll --out build/re/weights
build/venv/bin/python -I tools/sr/sr_pack.py build/re/weights M build/re/packed/M
build/venv/bin/python -I tools/sr/sr_pack.py build/re/weights L build/re/packed/L
build/venv/bin/python -I tools/rr/rr_weights.py build/re/nvngx_dlssd_310.9.1.dll --out build/rr/weights
build/venv/bin/python -I tools/rr/rr_pack.py build/rr/weights build/rr/packed
```

Without them `install.sh` warns, and the package refuses to create a feature at runtime.

## 3. Add it to a game

```bash
bash build/package/install.sh /path/to/game_folder optiscaler   # install
bash build/package/install.sh /path/to/game_folder remove       # uninstall (byte-for-byte restore)
```

What installation does:

- copies `build/package/_nvngx.dll` to `<game>/_nvngx.dll` and `build/shaders/*.spv` to
  `<game>/dlss5-amd/shaders/`
- writes `<game>/dlss5-amd-launch.env` with the launch environment: `WINEDLLOVERRIDES=_nvngx,nvngx=n,b`
  and `DLSS5_AMD_WEIGHTS=<repo>/build/ngx-weights` (symlinks to the packed M/L/rr trees — the weights
  stay outside the game folder)
- records every file in `<game>/dlss5-amd-install.txt`; `remove` deletes exactly those and nothing else
- never installs `nvngx.dll`

For CreateFeature to succeed, the game folder must also hold your own `nvngx_dlss.dll` (SR) or
`nvngx_dlssd.dll` (RR) at the expected version; when they are missing or wrong the package refuses and
logs the reason to `dlss5-amd-ngx.log` (written to the process's working directory).

How the NGX exports are reached: Vulkan titles go through `NVSDK_NGX_VULKAN_EvaluateFeature` (the
frame chain runs directly on the game's images); D3D12 titles go through
`NVSDK_NGX_D3D12_EvaluateFeature`, bridged to Vulkan via vkd3d-proton's `ID3D12DXVKInteropDevice` /
`ID3D12GraphicsCommandListExt`. Under Proton the DLLs load from the game folder, native first.

> `dlss5-amd-launch.env` is plain `KEY=VALUE` lines. Only the test harness is shown applying them for you
> (it reads the file and passes each line to the Proton process); how your own launcher picks them up is
> not documented here — export the same lines in your launch environment (for Steam: launch options).

NR (DLSS Neural Rendering) comes from the MIT reference port in `reference-nr-vulkan/`, with its own
installer and guide: `reference-nr-vulkan/linux/package/README.txt`. Its accuracy on this card (offline
gate, real 310.8.0 model) is 45.59 / 47.97 / 49.00 dB vs NVIDIA at 1080p / 1440p / 4K
([docs/hardware/RX9070XT-STEAMOS.md](docs/hardware/RX9070XT-STEAMOS.md)); the in-game NR runtime path is
not measured in this repository.

## 4. Verify a game / Proton run

```bash
bash tools/verify/ngx_client/build.sh          # -> build/verify/ngx_client.exe (mingw, else podman)
cp build/verify/ngx_client.exe /path/to/game/  # next to _nvngx.dll and your nvngx_dlss*.dll
```

`ngx_client.exe` drives NGX exactly as a game does:

```
ngx_client.exe --api d3d12|vulkan --feature sr|rr --frames DIR --out DIR [--preset M|L]
               [--ngx-dll NAME] [--test-cycles] [--expect-create-fail]
```

- `--test-cycles` also releases the feature, re-creates it at half size, evaluates one frame there and
  releases it (create / resize / release); `--expect-create-fail` passes only if CreateFeature refuses.
  Exit 0 = success; outputs are `out_<i>.f32`.
- Run it with the environment from `dlss5-amd-launch.env` under Proton 10.0, from a working directory of
  your choice (NGX writes `dlss5-amd-ngx.log` there) and with `Z:\`-style paths for `--frames` / `--out`.
  The canonical commands and measured tables are in [docs/MEASUREMENT.md](docs/MEASUREMENT.md).
- `DLSS5_AMD_TRACE=1` makes the chain print its dispatch list (which oracle launch each dispatch is) —
  a quick way to see the frame chain actually running.

## 5. Verifying correctness and quality (developers)

```bash
bash tools/verify/offline_check.sh     # no GPU: pipeline builds, ACO spill/scratch, lavapipe double-runs
bash tools/verify/run_all.sh           # the referee on the card: V1-V10 -> "VERIFY RESULT: PASS <run id>"
build/venv/bin/python -I tools/report/quality_guard.py   # quality bars against ground truth
```

- `run_all.sh` scores the port against NeuralHub's own kernels as the referee, so it needs the
  oracle inputs in `build/nh` first: bring your own `NeuralHub_v0.8.0_setup.exe` (official release,
  see build step 3) and run `tools/exe/fetch_neuralhub.sh build/nh`.
- `run_all.sh` needs a clean working tree and a SteamOS host; `PROTON=<path>` pins the Proton (10.0
  works; 11 / Experimental hang), `PORT_RUNNER=<path>` overrides the runner. It deletes the previous
  runs' oracle outputs, so copy anything you intend to cite right after the run.
- `quality_guard.py` judges the port against ground truth and NeuralHub's own output on the referee's
  cells (bars G1/G2/T1/B1); it needs the referee data linked into `build/qguard/`.
- All published numbers live in [docs/PORT-STATUS.md](docs/PORT-STATUS.md). Whole-pipeline numbers come
  only from `tools/report/pipeline_report.py`, per-launch numbers only from
  `tools/report/launch_report.py` ([docs/MEASUREMENT.md](docs/MEASUREMENT.md)).

## Repository layout

- `docs/inventory/`: every HIP kernel and interop call of NeuralHub v0.8.0, mapped to Vulkan.
- `docs/harness/`, `tools/harness/`: dump format and the per-layer compare tool.
- `tools/exe/`: the bring-your-own entry point (`fetch_neuralhub.sh`) plus installer extraction and
  analysis tooling.
- `tools/radv-offline/`: compile SPIR-V for an RX 9070 XT (RADV GFX1201) without the card.
- `reference-nr-vulkan/`: the reference Vulkan NR port (submodule, MIT) — see its
  `linux/package/README.txt` for installing NR into a game.
- `artifacts/`: only redistributable reference material — AMD's RDNA4 ISA, the prebuilt Mesa for
  offline compiles, and the reference NR pipeline statistics; no NeuralHub content by design (its
  license forbids redistribution — bring your own, build step 3).

## Acknowledgements and attribution

This project reimplements the DLSS SR/RR compute pipeline of NeuralHub v0.8.0 (a Windows mod) as
Vulkan compute shaders for Linux/Proton. Kernel structure, constants and launch sequence were
recovered from NeuralHub's code objects and host DLL (`docs/inventory/`, `docs/*/HOST-RE.md`,
`tools/re/`); the shaders and host are an independent reimplementation in GLSL/C++ whose numerics are
scored per launch against NeuralHub's own kernels running under ROCm on the same GPU. No NeuralHub
code and no NeuralHub-derived data ship in the product: model weights are read at install time from
your own NVIDIA DLSS DLLs. NeuralHub's installer, code objects and disassembly are NOT
redistributed in this repository (its license forbids redistribution and instructs linking to the
official release); bring your own to run the oracle.

NR is provided by `reference-nr-vulkan/`, the reference Vulkan NR port (MIT license, submodule).

## License

This project is licensed **GPL-3.0-or-later** (see [LICENSE](LICENSE)). The license covers this
repository's original code only: the third-party binaries you bring in yourself (NeuralHub's
installer, NVIDIA's DLSS DLLs) remain under their own terms, and the vendored toolchains and
submodules keep their own licenses (MIT/Apache, e.g. `reference-nr-vulkan/`). GPL-3.0-or-later was
chosen to permit bundling GPL components such as OptiScaler later (a combination distributes under
GPLv3).

## Learn more

[HANDOFF.md](HANDOFF.md) — project history, status by deliverable, how numbers are stated, and what
must not be retried.
