# DeepSeek assessment: ROCm-in-game vs the Vulkan port, and how big SR/RR are

Written 2026-10-24 by DeepSeek V4.1 Flash (OpenCode) for the session that picks this up next.
This is a *decision record*, not a restatement of `HANDOFF.md`; `HANDOFF.md` stays authoritative for
project state and for the do-not-retry rules. Read this first, then `HANDOFF.md`, then
`tools/verify/README.md`.

Everything below is either (a) measured on this machine, with the command or file named, or (b) marked
**estimate**. Do not turn an estimate into a fact without measuring it.

---

## 1. The question that was asked

> Could we skip the Vulkan port, make ROCm work on SteamOS with no container, and integrate the ROCm
> build into games?

## 2. Verdict: no — and the reason is not performance

The ROCm path is not a shortcut past the port. It is the port plus one problem that has no existing
solution. The four blockers, in the order they bind:

| # | blocker | status |
| - | --- | --- |
| 1 | SteamOS kernel side | **already works.** `amdgpu` is in-tree and `/dev/kfd` exists: `crw-rw-rw- 1 root render 237, 0`. Only ROCm *userspace* is missing on the host (`build/sr/srref` fails with `libamdhip64.so.7: cannot open shared object file`); the same binary runs inside the ROCm 7.2.4 image. |
| 2 | ROCm userspace without a container | **possible, ~10 GB, root-owned.** Untar to `/opt/rocm` and set `LD_LIBRARY_PATH` — no `pacman` needed, so no `steamos-readonly` / keyring / rolling-repo-vs-frozen-base problem. It is wiped by the next SteamOS A/B update. Fine as a dev convenience, not as a product dependency. |
| 3 | getting HIP into the game process | **no path.** Everything that ships into a game here is Windows PE running under Proton: `reference-nr-vulkan/linux/package/install.sh` installs `dlssnr-amd/*.dll` into the game folder via the OptiScaler or ReShade routes. ROCm's runtime is Linux ELF. Wine has no HIP thunk; a PE→ELF `winehip` (the analogue of `winevulkan`) does not exist anywhere. Every `hipMalloc` / `hipLaunchKernel` / `hipMemcpy` would need marshalling across the PE/unix boundary. |
| 4 | sharing the game's resources | **requires forking vkd3d-proton.** DLSS is not a post-process. It needs colour, depth, motion vectors, jitter, exposure, reactive/transparency masks at render resolution plus persistent history, and returns display resolution inside the game's frame. Those are vkd3d-proton D3D12 resources, i.e. Vulkan memory in the game's device. HIP can only reach them by dma-buf export/import plus timeline semaphores, and vkd3d-proton does not expose dma-buf fds to third-party DLLs. Patching it makes the installer a patched Proton build instead of a DLL in a game folder. |

The reference project already solved the same problem for Vulkan, and the shape of its solution is the
proof: `reference-nr-vulkan/linux/vulkan-loader/` contains `vulkan1_forwarder.c`,
`winevulkan_forwarder.c` and `patch-queue-info.py` — they ship a `vulkan-1.dll` that forwards to Wine's
`winevulkan` **and patch queue-info**, so PE code shares the game's own device and command buffers.
Vulkan is the one API where that bridge exists.

`reference-nr-vulkan/docs/ARCHITECTURE.md` §"Why Vulkan and not HIP/ROCm" states the same four points
independently: code runs inside the game; same device, no copies; one code base for Linux and Windows;
the WMMA instructions are reachable. That section exists because the author faced this exact question.

**The other half of the verdict:** the working HIP implementations in this repo (`tools/sr/srref`,
`tools/rr/rrref`) are *oracles*, not runtimes. Fixed frame sequences, one `.co`, no NGX state machine,
no history, no VRAM budget, no scheduling; they compute and print numbers. Treating them as "the thing
we already have" overstates what they are — and their real value is exactly what the port needs:
per-launch ground truth to diff every kernel against.

## 3. Calibration: how different is the Vulkan NR from the HIP NR?

This is the measurement that answers "how hard will SR/RR be", because NR is the one network that has
been done both ways.

| | HIP (what runs today) | Vulkan (the reference port) |
| --- | --- | --- |
| kernels | 193 | **9 shader bodies** + 14 shared includes |
| instructions | 455,252 | ~14.7k lines GLSL in `linux/shaders/rdna4/` (includes dominate) |
| variants | 257 template families | one body + spec constants and `-D` defines |
| runtime passes | — | ~5.1k lines in `linux/shaders/passes/` (history, taps, upscale, prep, ...) |
| host code | HIP streams, one `.co` per arch | ~29k lines C++ in `linux/src/` (device, layer, menu, routes, interop) |
| fidelity | byte-identical to itself | 45.6 / 48.0 / 49.0 dB vs NVIDIA at 1080p / 1440p / 4K |

Sources: `docs/inventory/kernels-generated.md` §NR; `reference-nr-vulkan/docs/ARCHITECTURE.md`
§"The network"; `find`/`wc` over `linux/shaders` and `linux/src`; `README.md` for the dB figures.

So the port did **not** restructure the network. One shader body covers a whole template family, and the
mapping is per *operation*: `coopmat 16x16x16 E4M3` for FP8 WMMA, `OpFConvert f32<->E4M3` (RNE, no
saturation) for the conversions, wave32 shuffle patterns for reductions, explicit LDS. The inventory's
"Vulkan construct" column already answers this for every family — it is a lookup, not research.

Two claims from `ARCHITECTURE.md` worth keeping:

- "Precision follows the original operation by operation: FP8 where NVIDIA's kernels use FP8, FP16/FP32
  everywhere else. The output is close to NVIDIA's but not bit-identical."
- the network is "one sequential chain of about 150 layers (only one adjacent pair is independent)", so
  async/streams buy nothing; there is nothing to overlap.

The same shaders and host code build for Linux and Windows; only the driver differs.

## 4. Sizing SR and RR

| | kernels | instructions | substantial shader bodies |
| --- | --- | --- | --- |
| NR | 193 | 455,252 | 9 |
| SR (presets M/L) | 52 | 137,993 | ~9 |
| RR (preset F) | 33 | 89,146 | ~6 |
| **SR + RR** | **85** | **227,139 (0.5x NR)** | **~15 (1.6x NR)** |

Families, from `docs/inventory/kernels-generated.md`:

- SR: `ae_copy`, `ae_luma`, `ae_reduce`, `dec0`, `downsample`, `enc0`, `enc0L_in`, `exposure`, `l0`,
  `post` (16 instantiations), `unorm8_table`, `stage` (18), `tube_split`, `w16`, `gather`, `gather_mv`.
- RR: `ae_block`, `ae_final`, `downsample`, `ds_fused`, `ds_t`, `enc0_feat`, `pooled2`, `post`,
  `post_prep`, `avg2x2`, `level` (18), `rr_w16`, `gather`, `gather_out`.

Effort tracks bodies, not instructions, because the instruction count is dominated by WMMA loops that
one body covers. **Estimate: SR+RR is ~1.5-2x the NR network port**, at half the instruction volume.

Two structural advantages NR did not have, which push the real number toward the low end:

1. **The graph is already recovered.** `srref` and `rrref` run NeuralHub's own kernels with per-launch
   dumps (`launches.jsonl`, `buffers.json`), so launch order, buffer layouts and flags are known and
   executable. The NR port's authors had to recover the graph from the DLL.
2. **The hard helpers exist and are MIT.** `linux/shaders/rdna4/include/coopmm.glsl` (FP8/f16 WMMA),
   the e4m3 conversions, the wave32 patterns, plus the 11 `runtime_*` passes (history, taps, upscale,
   prep, depth, downscale) that map close to 1:1 onto what SR's history and post chain need.

## 5. Risk register — where "different" becomes "hard"

Ordered by expected trouble, not by size.

1. **`k_sr_stage`** (18 instantiations, 56 call sites each). The softmax / `exp2` block is still
   undecoded; see `docs/sr/VULKAN.md`. Precedent exists in the NR port (`k_attention`, `k_qkv_attn`,
   `swin_math.glsl`), so this is a decode problem, not an unknown. This is the kernel that decides
   whether the port needs a frontier model.
2. **Texture sampling.** SR has **256 texture samples** vs NR's 42. Exact filter, border mode and
   coordinate rounding are where "close but not bit-identical" is decided, and no spec constant fixes
   them. Expect this to be the largest source of per-kernel max-error iteration.
3. **`k_level` (RR, 18 instantiations).** One body, different LDS/template configs — low risk once the
   body is decoded, high cost if not.
4. **`k_sr_post` (16 instantiations).** Mechanical.
5. **Host-side DLSS state** — jitter, history, masks, exposure, resolution changes. API work, bounded,
   and needed on any route.
6. **The known failure mode** is approximation under pressure. `HANDOFF.md` §Audit records a draft
   `rr_level.comp` that was reverted for exactly this. A kernel with no measured max-abs-error number is
   not done.

## 6. Card budget (measured)

Reference harness VRAM on this card, with `srref` / `rrref` in the ROCm 7.2.4 image, 1080p -> 4K:

- SR, 4 frames: peak 1878 MiB, baseline 816 MiB -> **~1.06 GB net**
- RR, 8 frames: peak 3495 MiB, baseline 816 MiB, 13 ms/frame -> **~2.6 GB net**
- oracle sequence (256x144 -> 512x288): tens of MB

The correctness loop — oracle, harness, every kernel diff at 512x288 — is VRAM-trivial. Only the
whole-frame 1080p -> 4K check and `run_all.sh` (Blender + ground truth) are heavy: run those as
deliberate, measured steps rather than in the background, and never make a GPU timing a Done test while
anything else is using the card. Build containers get an explicit `--memory=8g` so a compile cannot take
the machine down.

## 7. What a later session should do first

1. Read `HANDOFF.md` (authoritative state, audit, do-not-retry) and `tools/verify/README.md` (the port
   contract: runner CLI, `pipelines.json`, frames format, V1-V10).
2. Do **not** re-derive §2. The ROCm-in-game route was evaluated and rejected for structural reasons;
   revisit only if the target changes from Proton games to native Linux games.
3. Check whether the oracle exists (`ls build/oracle`). If not, commit this file and collect it
   (`bash tools/gpu/collect_oracle.sh`; the script refuses a dirty tree) — it is the first unit of work
   and everything after it is checkable against it.
4. Put the strongest available model on `k_sr_stage` and SR's sampling work (§5 items 1-2); the oracle,
   the harness and the mechanical kernels survive a weaker one.
5. If the user's actual goal shifts from "port" to "play now": upstream `DLSSNR-AMD` covers NR today, and
   FSR4 covers SR natively on RDNA4. Neither covers RR. That is a different project, not a shortcut
   through this one.

## 8. Reproducing the measurements

```bash
# SR / RR reference runs in the ROCm image (VRAM sampled from the host in parallel)
podman run --rm --device /dev/kfd --device /dev/dri --group-add keep-groups \
  --security-opt label=disable -v "$PWD":/w docker.io/rocm/dev-ubuntu-24.04:7.2.4 \
  bash -c 'cd /w && build/sr/srref build/nh/sr.gfx1201.co build/re/packed/M build/sr/frames1080 build/sr/out'
# -> build/sr/srref: peak 1878 MiB vs 816 MiB baseline; RR equivalent 3495 MiB
```

Kernel inventory numbers: `docs/inventory/kernels-generated.md` (per-family rows carry instances,
LDS, VGPRs and the Vulkan construct). Reference port shape: `reference-nr-vulkan/docs/ARCHITECTURE.md`,
`linux/shaders/rdna4/pipelines.json`, `linux/build/build_network.py`.
