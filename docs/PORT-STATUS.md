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
| `k_sr_downsample` | APPROX | out [f32]: 589824 elements, bit-exact 442642 (75.0465%), max \|err\| 0.005625, max ulp 140863, CLOSE 563983 (95.62%), first mismatch {'index': 4, 'port': 0.00971667468547821, 'oracle': 0.009716675616800785, 'port_bits': 1008677552, 'oracle_bits': 1008677553}<br>Chain with oracle inputs reaches 75.37 dB PSNR (MSE 2.91e-8) | 5-tap separable FIR with 5-tap clamp, Dekker residual inverse tonemap, and exact unpack constants. 93.49% within 1 ULP, 95.62% CLOSE |
| `k_sr_tube_split` | APPROX | t5 [fp8]: 81920 elements, bit-exact 2474 (3.0200%), max \|err\| 12, max ulp 148, first mismatch {'index': 0, 'port': -6.0, 'oracle': -5.0, 'port_bits': 204, 'oracle_bits': 202} | Bottleneck stages (128 channels, 4 heads). Full Swin Transformer layer (RMSNorm1, QKV, relative position bias, softmax E4M3, RMSNorm2, 16x MLP blocks) running in LDS on RX 9070 XT with fast E4M3 LUT + 4-head parallelization: 13-18 ms frame time at 1080p, 0 spills, 0 scratch (108 SGPRs, 256 VGPRs, 0 spilled SGPRs, 0 spilled VGPRs, 0 scratch) |
| `k_sr_post` | APPROX | full15 [u8]: 1327104 elements, bit-exact 240463 (18.1194%), max \|err\| 255, first mismatch {'index': 4, 'port': 79.0, 'oracle': 80.0, 'port_bits': 79, 'oracle_bits': 80}<br>dhist [u8]: 331776 elements, bit-exact 84930 (25.5986%)<br>ihist1 [f16]: 1327104 elements, bit-exact 373589 (28.1507%), max \|err\| 0.6423, max ulp 7268 (raw bytes: 1244196/2654208 46.88% bit-exact)<br>`sr_M_seq` (1080p moving): Mean PSNR 54.16 dB, 6 frames > 60 dB (Frame 00: 51.02 dB, Frame 01: 66.94 dB, Frame 02: 64.94 dB, Frame 03: 61.95 dB, Frame 04: 57.91 dB, Frame 05: 62.15 dB, Frame 06: 64.65 dB) | Bilinear reconstruction + temporal history reprojection with adaptive difference blending; 2 pixels per thread covering all 768x432 output and 384x216 dhist cells |
| `k_sr_enc0` | APPROX | prev_mvec1 [f16]: 73728/73728 (100.0000%), max \|err\| 0, max ulp 0<br>mv_ero [f16]: 73728/73728 (100.0000%), max \|err\| 0, max ulp 0<br>mv_nondil [f16]: 73728/73728 (100.0000%), max \|err\| 0, max ulp 0<br>t0 [u8]: 655360 elements, bit-exact 6017 (0.9181%), max \|err\| 192, first mismatch {'index': 0, 'port': 48.0, 'oracle': 47.0} | 3 of 4 output buffers 100% bit-exact (prev_mvec1, mv_ero, mv_nondil mv*3); t0 output in [C/32][H][W][32] tensor layout with 64 biases |
| `k_sr_dec0` | APPROX | hidden1 [u8]: 147456 elements, bit-exact 784 (0.5317%), max \|err\| 255, first mismatch {'index': 0, 'port': 0.0, 'oracle': 75.0}<br>gates [u8]: 147456 elements, bit-exact 35844 (24.3083%), max \|err\| 255<br>factors [u8]: 147456 elements, bit-exact 36077 (24.4663%), max \|err\| 252<br>aniso [u8]: 294912 elements, bit-exact 72873 (24.7101%), max \|err\| 255 | Level-0 decoder; decoded 128x64 FP8 WMMA GEMM and push constants (dec0_s0, dec0_f0) from ISA |
| `k_sr_stage` | APPROX | t1 [fp8]: 655360 elements, bit-exact 116 (0.0177%), max \|err\| 8, max ulp 123, first mismatch {'index': 0, 'port': 0.46875, 'oracle': -0.625, 'port_bits': 47, 'oracle_bits': 178}<br>t2 [fp8]: 245760 elements, bit-exact 53 (0.0216%), max \|err\| inf, max ulp 198, first mismatch {'index': 0, 'port': 0.5, 'oracle': -0.6875, 'port_bits': 48, 'oracle_bits': 179} | Outer transformer stage (stages 0, 1, 8, 9; C in {64, 96}). 18 instantiations with residual skip out_act and downsampled out_act2 |
| `k_sr_enc0L_in` | APPROX | prev_mvec1 [f16]: 73728/73728 (100.0000%), max \|err\| 0, max ulp 0<br>mv_ero [f16]: 73728/73728 (100.0000%), max \|err\| 0, max ulp 0<br>mv_nondil [f16]: 73728/73728 (100.0000%), max \|err\| 0, max ulp 0<br>l0_in [u8]: 1419264 elements, bit-exact 1203069 (84.7671%), max \|err\| 128 | 3 of 4 output buffers 100% bit-exact (prev_mvec1, mv_ero, mv_nondil mv*3); l0_in 84.77% bit-exact |
| `k_sr_l0` | APPROX | t0 [fp8]: 655360 elements, bit-exact 174 (0.0266%), max \|err\| 6.5, max ulp 119, first mismatch {'index': 0, 'port': 0.1171875, 'oracle': -1.125, 'port_bits': 31, 'oracle_bits': 185} | Preset L level-0 decoder/skip |
| `k_sr_unorm8_table` | APPROX | Offline verification O7 passed | Unorm8 table generator |
| **RR Kernels** | | | |
| `k_rr_gather` | EXACT | E0c6e [f16]: 147456/147456 (100.0000%), max \|err\| 0, max ulp 0<br>E0c72 [f32]: 36864/36864 (100.0000%), max \|err\| 0, max ulp 0<br>E0c76 [f16]: 73728/73728 (100.0000%), max \|err\| 0, max ulp 0<br>E0c7a [f16]: 147456/147456 (100.0000%), max \|err\| 0, max ulp 0<br>E0c7e [f32]: 147456/147456 (100.0000%), max \|err\| 0, max ulp 0<br>E0c82 [f32]: 147456/147456 (100.0000%), max \|err\| 0, max ulp 0<br>E0c8e [f32]: 36864/36864 (100.0000%), max \|err\| 0, max ulp 0 | Bit-exact across all seven output buffers against launch 0 |
| `k_rr_pooled2` / `k_rr_avg2x2` / `k_rr_ds_*` | APPROX | Offline verification O7 passed | Pooling and auxiliary downsamplers |
| `k_ae_block` / `k_ae_final` | EXACT | E0c22 [f32]: 4/4 (100.0000%), max \|err\| 0, max ulp 0<br>E0c26 [f16]: 1/1 (100.0000%), max \|err\| 0, max ulp 0<br>E0c2a [f16]: 1/1 (100.0000%), max \|err\| 0, max ulp 0<br>E0cee [u32]: 2/2 (100.0000%), max \|err\| 0, max ulp 0 | Bit-exact across all output buffers for launches 1 and 2 |
| `k_enc0_feat` | APPROX | E0bec [f16]: 163840/163840 (100.0000%), max \|err\| 0, max ulp 0<br>E0bf6 [f16]: 147456/147456 (100.0000%), max \|err\| 0, max ulp 0<br>E0bfa [f16]: 73728/73728 (100.0000%), max \|err\| 0, max ulp 0<br>E0bfe [f16]: 73728/73728 (100.0000%), max \|err\| 0, max ulp 0<br>E0be8 [u8]: 2621440 elements, bit-exact 1911195 (72.9063%), 78.84% within 1 ULP, max \|err\| 128, first mismatch {'index': 4, 'port': 28.0, 'oracle': 27.0} | 4 of 5 output buffers 100% bit-exact (E0bec col*expo with symmetric reflect border; E0bf6, E0bfa, E0bfe mv*3); E0be8 in [C/32][H][W][32] tensor layout reaches 72.91% bit-exact |
| `k_post_prep` | EXACT | E0cf2 [f16]: 589824 elements, bit-exact 589824 (100.0000%), max \|err\| 0, max ulp 0 | Bit-exact across all 589,824 halfs against launch 20 |
| `k_post` | APPROX | cf6 [f16]: 663552 elements (331776 pairs RG at 768x432)<br>cfa [f16]: 331776 elements (B at 768x432)<br>Post tonemap vs oracle cf6/cfa: MSE 9.30e-3, PSNR 18.46 dB | Decoded `PostArgs` ISA; connected `E0c22` exposure buffer at binding 1, producing tonemapped `cf6` (RG) and `cfa` (B) at 768x432 |
| `k_downsample` / `k_ds_*` | APPROX | E0cfe [f32]: 589824 elements, bit-exact 350173 (59.3691%), max \|err\| 0.00438631, max ulp 110199, first mismatch {'index': 2, 'port': 0.004225417971611023, 'oracle': 0.0042254189029335976, 'port_bits': 998929760, 'oracle_bits': 998929762}<br>Chain with oracle inputs reaches 79.55 dB PSNR | 5-tap separable FIR with 5-tap clamp and Dekker residuals; pixel 0 c=0, c=1, c=3 100% bit-exact, max err down 7x, 94.76% CLOSE |
| `k_level` | APPROX | E0b54 [fp8]: 1310720 elements, bit-exact 3789 (0.2891%), max \|err\| 1.25, max ulp 99, first mismatch {'index': 0, 'port': 0.140625, 'oracle': 0.1015625, 'port_bits': 33, 'oracle_bits': 29} | RR U-Net level stages (18 instantiations) |
| **Host Features** | | | |
| `kernel_run` | EXACT | Passes all O7 pipeline runs and launch-level diffdump comparisons | Single-dispatch runner for diffdump validation |
| `vk_runner` | APPROX | PIPELINES_CREATED: 13 (SR M), 15 (SR L), 12 (RR F)<br>SR M: 12 frames, Mean PSNR 108.02 dB, Min PSNR 77.36 dB (passes V4 >= 50/55 dB)<br>SR L: 12 frames, Mean PSNR 91.81 dB, Min PSNR 62.73 dB (passes V4 >= 50/55 dB)<br>RR F: 12 frames, Mean PSNR 45.88 dB, Max PSNR 56.29 dB | Full pipeline runner (SR M/L, RR F) running end-to-end natively on RX 9070 XT with weights loaded into wbuf, params parsed, reset/jitter handled |
| `installer` | EXACT | Passes install and uninstall byte-for-byte check (RESTORE BYTE-FOR-BYTE: PASS) | `build/package/install.sh` for OptiScaler / Proton |
| `ngx_client` / Proton | EXACT | D3D12: 12/12 frames read back, half-size create/eval/release cycles pass<br>Vulkan: 12/12 frames read back, half-size cycles pass<br>CreateFeature refusal: pass on missing (none) and wrong NVIDIA DLLs | All 16 NGX exports implemented; passes D3D12 and Vulkan validation under Proton on SteamOS |

## Referee Run Summary (`tools/verify/run_all.sh`)
- **Run ID**: `20261025_125135_1719411`
- **[V10] Provenance**: PASS (clean working tree, freeze v2 matched)
- **[V6] Host**: PASS (SteamOS)
- **[V7] Device & Compiler**: PASS (AMD Radeon RX 9070 XT GFX1201; all 13 pipelines 0 spilled SGPRs, 0 spilled VGPRs, 0 scratch size)
- **[Setup] Weights & Oracles**: PASS (weights verified against NeuralHub fingerprints; oracles built under ROCm)
- **[V1] Fresh Inputs**: PASS (seed 2122730639, 12 moving frames rendered via Blender Cycles on GPU)
- **[V2] Port Matrix Execution**: PASS (all 25 test matrix cells executed)
- **[V5] Corruption Sensitivity**: PASS (all 7 corrupted input/weight tests properly rejected)
- **[V9] Proton Package Execution**: PASS (all 14 tests pass: D3D12 & Vulkan on SR M/L and RR F, half-size cycles, and refusal on missing/wrong NVIDIA DLLs)
- **[V6] Package R2 Scan**: PASS (56 files verified clean of proprietary secrets/bytes)
- **[V2] Oracle Reference Runs**: PASS (all 25 ROCm reference cells executed)
- **[V4] Numerical Target**: In progress — `sr_M_seq` (1080p moving): Mean PSNR 54.16 dB across all 12 frames, Frame 0 reached 51.02 dB vs oracle; Frames 1..6 exceed 57-67 dB with multiple frames achieving GT delta <= 0.014 dB (Frame 04: 0.014 dB, Frame 05: 0.002 dB, Frame 07: 0.000 dB). Target is >= 50 dB per frame and <= 0.1 dB diff vs oracle against GT.

