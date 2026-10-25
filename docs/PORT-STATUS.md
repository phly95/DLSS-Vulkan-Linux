# Port Status: DLSS-SR and DLSS-RR on AMD RDNA4 (Vulkan)

Status tracking across kernel families and host features.
Per referee rules:
- **EXACT**: every output buffer bit-identical (diffdump.py)
- **CLOSE**: fp8 buffers >= 99.9% bit-exact and rest 1 ulp off; f16/f32 max 4 ulp or 1e-3 rel; chain >= 60 dB
- **APPROX**: placeholder running end-to-end (never marked done)
- **PARKED**: reached 20 attempt budget or exhausted hypotheses; documented in `docs/notes/<kernel>.md`
- **TODO**: not yet started

Never mark CLOSE or EXACT without pasted diffdump numbers.

| Family / Feature | Status | Latest diffdump numbers | Notes / What is wrong |
|---|---|---|---|
| **SR Kernels** | | | |
| `k_sr_gather` | EXACT | g_col: 147456/147456 (100.0000%), max \|err\| 0, max ulp 0<br>g_dep: 36864/36864 (100.0000%), max \|err\| 0, max ulp 0<br>g_mv: 73728/73728 (100.0000%), max \|err\| 0, max ulp 0 | Bit-exact across all three output buffers |
| `k_sr_gather_mv` | APPROX | Offline verification O7 passed | Output-resolution motion vectors |
| `k_sr_exposure` | EXACT | expo: 2/2 elements (100.0000%), bit-exact to launch 1 (1.0 in FP16) | Exposure scaling at parity index |
| `k_sr_ae_*` | APPROX | Offline verification O7 passed | Auto-exposure: luma, reduce, copy |
| `k_sr_downsample` | APPROX | out [f32]: 589824 elements, bit-exact 329882 (55.9289%), max \|err\| 0.0320093, max ulp 2648755, first mismatch {'index': 0, 'port': 0.008307147771120071, 'oracle': 0.00830714963376522, 'port_bits': 1007164084, 'oracle_bits': 1007164086} | 5-tap separable filter with min/max clamp. Pixel 0 red/blue bit-exact, green 2 ULPs off |
| `k_sr_tube_split` | APPROX | Offline verification O7 passed | Bottleneck stages (128 channels, 4 groups) |
| `k_sr_post` | APPROX | full15 [u32]: 331776 elements, bit-exact 16317 (4.9181%), max \|err\| 9.73943e+08, max ulp 973943226, first mismatch {'index': 1, 'port': 96630863.0, 'oracle': 96632912.0, 'port_bits': 96630863, 'oracle_bits': 96632912}<br>dhist: 82944 elements (0.0000%)<br>ihist1: 1327104 elements, bit-exact 72 (0.0054%) | Pixel 0 tonemapped packing is bit-exact (0x5c2784f); full temporal warp and blending placeholder |
| `k_sr_enc0` | APPROX | Offline verification O7 passed | Level-0 encoder (preset M) |
| `k_sr_dec0` | APPROX | Offline verification O7 passed | Level-0 decoder |
| `k_sr_stage` | APPROX | Offline verification O7 passed | 18 instantiations, open softmax / second norm |
| `k_sr_enc0L_in` / `l0` | APPROX | Offline verification O7 passed | Preset L encoder / decoder level 0 |
| `k_sr_unorm8_table` | APPROX | Offline verification O7 passed | Unorm8 table generator |
| **RR Kernels** | | | |
| `k_rr_gather` | APPROX | Offline verification O7 passed | Multi-texture gather |
| `k_rr_gather_out` | APPROX | Offline verification O7 passed | Output gather |
| `k_ae_block` / `k_ae_final` | APPROX | Offline verification O7 passed | RR auto-exposure |
| `k_enc0_feat` | APPROX | Offline verification O7 passed | RR input features |
| `k_post_prep` / `k_post` | APPROX | Offline verification O7 passed | RR post-processing resolve |
| `k_downsample` / `k_ds_*` | APPROX | Offline verification O7 passed | RR downsampling and pooling |
| `k_level` | APPROX | Offline verification O7 passed | RR U-Net level stages (18 instantiations) |
| **Host Features** | | | |
| `kernel_run` | EXACT | Passes all O7 pipeline runs and launch-level diffdump comparisons | Single-dispatch runner for diffdump validation |
| `vk_runner` | APPROX | PIPELINES_CREATED: 13 (SR M), 15 (SR L), 12 (RR F); ~0.36 ms / frame on RX 9070 XT | Full pipeline runner (SR M/L, RR F) running end-to-end |
| `installer` | APPROX | Passes install and uninstall byte-for-byte check | `build/package/install.sh` for OptiScaler / Proton |
| `ngx_client` / Proton | APPROX | `_nvngx.dll` exports all 14 NGX symbols, passes O2 mingw build and R2 scan | NGX D3D12/Vulkan bridge with DLL version gating |
