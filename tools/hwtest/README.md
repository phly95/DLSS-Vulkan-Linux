# tools/hwtest — running the port's own kernels on the real card

This directory holds the tools that only make sense on a machine with an RX 9000 card (RDNA4). The
cloud container the port was written in has none, so everything here is a first: the numbers in
`artifacts/reference-nr/` come from Mesa's amdgpu drm-shim, which compiles but never executes.

Nothing here contains or needs NVIDIA material. What it *can* do without the user's
`nvngx_dlssnr.dll` is check the device gate, compile every pipeline of the network on the real
driver, and (with `synth_model.py`) run the whole graph with synthetic weights so that the runtime
path, the grids, the arena and the frame time are measured on hardware rather than argued about.

## `synth_model.py` — a weight directory for a machine that has no model

The reference's native runner (`nr_graph.cpp`'s `main`) reads every weight file by name from
`--unpacked <root>` and checks each one's size. The names and sizes are a property of the network's
structure, not of NVIDIA's numbers, so they can be recovered without the DLL: run the runner, let it
name the file it wanted and how many bytes it needed, create a file of exactly that many bytes, and
repeat. Two wrinkles are handled rather than guessed: a few checks count f16 *elements* instead of
bytes (the `.skip_weight.bin` files, read through `load_f16`), and a few print no size at all (the
ViT QKV record needs `128 + N*K` bytes). For those the script grows the file and runs the graph again
until the check passes, so a wrong guess cannot spin. A file whose size is simply not checked gets
`--pad` bytes (4 MiB by default). The result is a complete weight set that runs the graph and
produces meaningless pictures — which is the point: it exercises every kernel, every grid and every
barrier on the card, and the FP8 WMMA time it measures is the same as the real model's (the
instructions are data-independent).

    python3 tools/hwtest/synth_model.py --runner /path/to/nr_graph --out build/hwtest/synth-weights

Then run the graph against it:

    /path/to/nr_graph --spv-dir build/linux/rdna4/network --unpacked build/hwtest/synth-weights \
        --host-boundary --no-reuse --plan plan-1920x1080.txt \
        --source-width 1920 --source-height 1080 --repeats 10 --warmup 2

Do not compare the output of a synthetic run with anything: the weights are not the model's.

## `nr_plan.cpp` / `build_plan.sh` — the right plan for a resolution

The runtime does not hand the graph a fixed plan: it calls `nr::make_native_plan(width, height)` and
passes the parsed rows in (`nr_runtime.cpp`). The runner's `--plan` takes a *file*, and its default
(`linux/src/core/nr_frame.txt`) is one frozen geometry — so running the same binary at another
resolution without a matching plan leaves every layer's token count at 4K, and the frame time barely
moves with resolution. `nr_plan` prints the plan text for any size:

    tools/hwtest/build_plan.sh
    build/hwtest/nr_plan 1920 1080 > plan-1920x1080.txt

Build it with the reference sources and no Vulkan at all (`nr_native_plan.cpp` is self-contained), in
the same `podman` container as the runner.

## `build_native.sh` — the runner, built for this machine

`nr_graph.cpp` has a native `main()` (the one the reference's offline tools use) and compiles on
Linux with nothing but a C++17 compiler and Vulkan headers. This machine (SteamOS) has no compiler
in its immutable root, so the script builds inside a `podman` Ubuntu 24.04 container and writes the
binary to the host, where it runs against the host's Mesa (the driver under test).

## `extract_nr_model.sh` / `nr_gate.py` — the real model and the NR gate

The NR weights come from `nvngx_dlssnr.dll` 310.8.0.0. One copy (community-patched, weights unchanged)
is at https://huggingface.co/Bandukids/DLSS-Runtimes:

    curl -L -o nvngx_dlssnr.dll https://huggingface.co/Bandukids/DLSS-Runtimes/resolve/main/nvngx_dlssnr.dll
    tools/hwtest/extract_nr_model.sh nvngx_dlssnr.dll          # -> build/model/dlssnr.bin
    build/venv/bin/python tools/hwtest/nr_gate.py --runner build/hwtest/nr_graph \
        --plan-tool build/hwtest/nr_plan --model build/model/dlssnr.bin --work build/hwtest/gate

`extract_nr_model.sh` relaxes only the reference extractor's whole-file hash check; every weight entry is
still checked against the reference's tested model. Never commit the DLL or the model.
