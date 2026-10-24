# DLSS5-AMD-Linux

Work towards a Linux/Proton (SteamOS, RX 9070 XT, RADV) port of NeuralHub's DLSS NR / SR / RR for AMD.
Start with [HANDOFF.md](HANDOFF.md): status, how to reproduce, what blocks the rest.

- `docs/inventory/`: every HIP kernel and interop call of NeuralHub v0.8.0, mapped to Vulkan.
- `docs/harness/`, `tools/harness/`: dump format and the per-layer compare tool.
- `tools/exe/`: extraction and analysis of the installer.
- `tools/radv-offline/`: compile SPIR-V for an RX 9070 XT (RADV GFX1201) without the card.
- `reference-nr-vulkan/`: the reference Vulkan NR port (submodule, MIT).
- `artifacts/`: binaries for this private repository (see HANDOFF.md).
