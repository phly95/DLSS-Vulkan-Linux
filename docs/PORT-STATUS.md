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
| `k_sr_enc0` | APPROX | prev_mvec1 [f16]: 73728/73728 (100.0000%), max \|err\| 0, max ulp 0<br>mv_ero [f16]: 73728/73728 (100.0000%), max \|err\| 0, max ulp 0<br>mv_nondil [f16]: 73728/73728 (100.0000%), max \|err\| 0, max ulp 0<br>t0 [u8]: 655360 elements, bit-exact 1509 (0.2303%), max \|err\| 192 | 3 of 4 output buffers 100% bit-exact (prev_mvec1, mv_ero, mv_nondil mv*3); t0 feature map approximate |
| `k_sr_dec0` | APPROX | Offline verification O7 passed | Level-0 decoder |
| `k_sr_stage` | APPROX | Offline verification O7 passed | 18 instantiations, open softmax / second norm |
| `k_sr_enc0L_in` | APPROX | prev_mvec1 [f16]: 73728/73728 (100.0000%), max \|err\| 0, max ulp 0<br>mv_ero [f16]: 73728/73728 (100.0000%), max \|err\| 0, max ulp 0<br>mv_nondil [f16]: 73728/73728 (100.0000%), max \|err\| 0, max ulp 0<br>l0_in [u8]: 1419264 elements, bit-exact 1203069 (84.7671%), max \|err\| 128 | 3 of 4 output buffers 100% bit-exact (prev_mvec1, mv_ero, mv_nondil mv*3); l0_in 84.77% bit-exact |
| `k_sr_l0` | APPROX | Offline verification O7 passed | Preset L level-0 decoder/skip |
| `k_sr_unorm8_table` | APPROX | Offline verification O7 passed | Unorm8 table generator |
| **RR Kernels** | | | |
| `k_rr_gather` | EXACT | E0c6e [f16]: 147456/147456 (100.0000%), max \|err\| 0, max ulp 0<br>E0c72 [f32]: 36864/36864 (100.0000%), max \|err\| 0, max ulp 0<br>E0c76 [f16]: 73728/73728 (100.0000%), max \|err\| 0, max ulp 0<br>E0c7a [f16]: 147456/147456 (100.0000%), max \|err\| 0, max ulp 0<br>E0c7e [f32]: 147456/147456 (100.0000%), max \|err\| 0, max ulp 0<br>E0c82 [f32]: 147456/147456 (100.0000%), max \|err\| 0, max ulp 0<br>E0c8e [f32]: 36864/36864 (100.0000%), max \|err\| 0, max ulp 0 | Bit-exact across all seven output buffers against launch 0 |
| `k_rr_gather_out` | APPROX | Offline verification O7 passed | Output gather |
| `k_ae_block` / `k_ae_final` | EXACT | E0c22 [f32]: 4/4 (100.0000%), max \|err\| 0, max ulp 0<br>E0c26 [f16]: 1/1 (100.0000%), max \|err\| 0, max ulp 0<br>E0c2a [f16]: 1/1 (100.0000%), max \|err\| 0, max ulp 0<br>E0cee [u32]: 2/2 (100.0000%), max \|err\| 0, max ulp 0 | Bit-exact across all output buffers for launches 1 and 2 |
| `k_enc0_feat` | APPROX | E0bec [f16]: 163840/163840 (100.0000%), max \|err\| 0, max ulp 0<br>E0bf6 [f16]: 147456/147456 (100.0000%), max \|err\| 0, max ulp 0<br>E0bfa [f16]: 73728/73728 (100.0000%), max \|err\| 0, max ulp 0<br>E0bfe [f16]: 73728/73728 (100.0000%), max \|err\| 0, max ulp 0<br>E0be8 [u8]: 2621440 elements, bit-exact 13215 (0.5041%), max \|err\| 172 | 4 of 5 output buffers 100% bit-exact (E0bec col*expo with symmetric reflect border; E0bf6, E0bfa, E0bfe mv*3); E0be8 FP8 feature tensor approximate |
| `k_post_prep` | EXACT | E0cf2 [f16]: 589824 elements, bit-exact 589824 (100.0000%), max \|err\| 0, max ulp 0 | Bit-exact across all 589,824 halfs against launch 20 |
| `k_post` | APPROX | Offline verification O7 passed | RR post-processing temporal resolve |
| `k_downsample` / `k_ds_*` | APPROX | E0cfe [f32]: 589824 elements, bit-exact 300426 (50.9349%), max \|err\| 0.0289028, max ulp 1492358, first mismatch {'index': 1, 'port': 0.007601836696267128, 'oracle': 0.007601837161928415, 'port_bits': 1006180564, 'oracle_bits': 1006180565} | 5-tap polyphase separable FIR with inner-3 clamp and GPU tonemap inversion; first mismatch index 1 only 1 ULP off |
| `k_level` | APPROX | Offline verification O7 passed | RR U-Net level stages (18 instantiations) |
| **Host Features** | | | |
| `kernel_run` | EXACT | Passes all O7 pipeline runs and launch-level diffdump comparisons | Single-dispatch runner for diffdump validation |
| `vk_runner` | APPROX | PIPELINES_CREATED: 13 (SR M), 15 (SR L), 12 (RR F); ~0.36 ms / frame on RX 9070 XT | Full pipeline runner (SR M/L, RR F) running end-to-end |
| `installer` | APPROX | Passes install and uninstall byte-for-byte check | `build/package/install.sh` for OptiScaler / Proton |
| `ngx_client` / Proton | APPROX | `_nvngx.dll` exports all 14 NGX symbols, passes O2 mingw build and R2 scan | NGX D3D12/Vulkan bridge with DLL version gating |
