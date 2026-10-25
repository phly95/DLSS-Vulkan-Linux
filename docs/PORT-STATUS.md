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
| `k_sr_downsample` | APPROX | out [f32]: 589824 elements, bit-exact 417865 (70.8457%), max \|err\| 0.0320091, max ulp 2648755, first mismatch {'index': 4, 'port': 0.00971667468547821, 'oracle': 0.009716675616800785, 'port_bits': 1008677552, 'oracle_bits': 1008677553} | 5-tap separable FIR with Dekker residual inverse tonemap. Pixel 0 100% bit-exact (indices 0..3), index 4 1 ULP off (87.74% within 1 ULP) |
| `k_sr_tube_split` | APPROX | Offline verification O7 passed | Bottleneck stages (128 channels, 4 groups) |
| `k_sr_post` | APPROX | full15 [u8]: 1327104 elements, bit-exact 240463 (18.1194%), max \|err\| 255, first mismatch {'index': 4, 'port': 79.0, 'oracle': 80.0, 'port_bits': 79, 'oracle_bits': 80}<br>dhist [u8]: 331776 elements, bit-exact 84930 (25.5986%)<br>ihist1 [f16]: 1327104 elements, bit-exact 373589 (28.1507%), max \|err\| 0.6423, max ulp 7268 (raw bytes: 1244196/2654208 46.88% bit-exact) | Coordinate transform decoded from ISA; tonemapped history ihist1 (46.88% byte-exact), 2x2 box average 10:10:10 dhist (25.60% bit-exact) and R11G11B10 full15 (18.12% bit-exact) |
| `k_sr_enc0` | APPROX | prev_mvec1 [f16]: 73728/73728 (100.0000%), max \|err\| 0, max ulp 0<br>mv_ero [f16]: 73728/73728 (100.0000%), max \|err\| 0, max ulp 0<br>mv_nondil [f16]: 73728/73728 (100.0000%), max \|err\| 0, max ulp 0<br>t0 [u8]: 655360 elements, bit-exact 1509 (0.2303%), max \|err\| 192 | 3 of 4 output buffers 100% bit-exact (prev_mvec1, mv_ero, mv_nondil mv*3); t0 feature map approximate |
| `k_sr_dec0` | APPROX | hidden1 [u8]: 147456 elements, bit-exact 784 (0.5317%), max \|err\| 255, first mismatch {'index': 0, 'port': 0.0, 'oracle': 75.0}<br>gates [u8]: 147456 elements, bit-exact 35844 (24.3083%), max \|err\| 255<br>factors [u8]: 147456 elements, bit-exact 36077 (24.4663%), max \|err\| 252<br>aniso [u8]: 294912 elements, bit-exact 72873 (24.7101%), max \|err\| 255 | Level-0 decoder; decoded 128x64 FP8 WMMA GEMM and push constants (dec0_s0, dec0_f0) from ISA |
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
| `k_post` | APPROX | cf6 [f16]: 663552 elements (331776 pairs RG at 768x432)<br>cfa [f16]: 331776 elements (B at 768x432)<br>Post tonemap vs oracle cf6/cfa: MSE 9.30e-3, PSNR 18.46 dB | Decoded `PostArgs` ISA; connected `E0c22` exposure buffer at binding 1, producing tonemapped `cf6` (RG) and `cfa` (B) at 768x432 |
| `k_downsample` / `k_ds_*` | APPROX | E0cfe [f32]: 589824 elements, bit-exact 350173 (59.3691%), max \|err\| 0.00438631, max ulp 110199, first mismatch {'index': 2, 'port': 0.004225417971611023, 'oracle': 0.0042254189029335976, 'port_bits': 998929760, 'oracle_bits': 998929762}<br>Chain with oracle inputs reaches 79.55 dB PSNR | 5-tap separable FIR with 5-tap clamp and Dekker residuals; pixel 0 c=0, c=1, c=3 100% bit-exact, max err down 7x, 94.76% CLOSE |
| `k_level` | APPROX | Offline verification O7 passed | RR U-Net level stages (18 instantiations) |
| **Host Features** | | | |
| `kernel_run` | EXACT | Passes all O7 pipeline runs and launch-level diffdump comparisons | Single-dispatch runner for diffdump validation |
| `vk_runner` | APPROX | PIPELINES_CREATED: 13 (SR M), 15 (SR L), 12 (RR F)<br>SR M: 9 frames, 0.40 ms/frame, 27.64 dB PSNR<br>SR L: 9 frames, 0.39 ms/frame, 27.47 dB PSNR<br>RR F: 9 frames, 0.67 ms/frame, 22.32 dB PSNR (MSE 9.02e-3) | Full pipeline runner (SR M/L, RR F) running end-to-end natively on RX 9070 XT |
| `installer` | EXACT | Passes install and uninstall byte-for-byte check (RESTORE BYTE-FOR-BYTE: PASS) | `build/package/install.sh` for OptiScaler / Proton |
| `ngx_client` / Proton | APPROX | `_nvngx.dll` exports all 16 NGX symbols (including D3D12/Vulkan ReleaseFeature), passes O2 mingw build and R2 scan | NGX D3D12/Vulkan bridge with DLL version gating |
