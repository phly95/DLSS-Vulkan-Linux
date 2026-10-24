# DLSS-SR in Vulkan Compute (RDNA4 GFX1201)

Implementation of NeuralHub v0.8.0 DLSS Super Resolution compute shaders in Vulkan for AMD RDNA4 (RX 9070 XT), running natively via RADV with zero spills.

## Kernel Implementation & Verification Progress

| Kernel | Shaders / Pipelines | Precision | Wave Size | Spills (SGPR / VGPR) | Scratch | Max Diff vs Oracle | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `k_sr_stage` (encoder st 0) | `sr_stage.comp` (`sr_stage.spv`) | FP8 / FP16 / FP32 | Wave32 (64 thd WG) | 0 / 0 | 0 bytes | verified | PASS |
| `k_sr_tube_split` (bottleneck st 2..7) | `sr_tube.comp` (`sr_tube.spv`) | FP8 / FP16 | Wave32 (64 thd WG) | 0 / 0 | 0 bytes | verified | PASS |
| `k_sr_enc0` | `sr_enc0.comp` (`sr_enc0.spv`) | FP16 / FP8 | Wave32 (32 thd WG) | 0 / 0 | 0 bytes | verified | PASS |
| `k_sr_dec0` | `sr_dec0.comp` (`sr_dec0.spv`) | FP16 / FP8 | Wave32 (32 thd WG) | 0 / 0 | 0 bytes | verified | PASS |
| `k_sr_post` | `sr_post.comp` (`sr_post.spv`) | FP16 / FP32 | Wave32 (256 thd WG) | 0 / 0 | 0 bytes | verified | PASS |
| `k_sr_downsample` | `sr_downsample.comp` (`sr_downsample.spv`) | FP16 / FP32 | Wave32 (256 thd WG) | 0 / 0 | 0 bytes | verified | PASS |
| `k_sr_gather` | `sr_gather.comp` (`sr_gather.spv`) | FP16 / FP32 | Wave32 (256 thd WG) | 0 / 0 | 0 bytes | verified | PASS |
| `k_sr_exposure` | `sr_exposure.comp` (`sr_exposure.spv`) | FP16 / FP32 | Wave32 (64 thd WG) | 0 / 0 | 0 bytes | verified | PASS |

## `k_sr_stage` Details (st 0: `k_sr_stage<64, 2, 0, 96, 2, false>`)
- Grid: `((L1w + shift_x + 7) >> 3, (L1h + shift_y + 7) >> 3)` workgroups of 64 threads.
- Coverage: 8x8 pixel tile with mirror-padding.
- RMSNorm 1: f16 sum of squares + $2^{-14}$ epsilon, rsqrt scaling by $s1b0$.
- First projection $w210$ ($64 \times 128$) with group spatial/channel mixing via $v258$ polynomial.
- Second projection $w1f0$ ($128 \to 64$) + bias $v270$ + residual addition.
- RMSNorm 2 scaling by $s198$.
- FFWD MLP expansion $w180$ ($64 \to 256$) + bias $v228$ + non-linear polynomial activation.
- FFWD MLP contraction $w1d0$ ($256 \to 64$) + bias $v240$ + residual addition.
- Skip output $t1$ store to `[C/32][H][W][32]` planar-blocked memory.
- Down-projection $w168$ ($64 \to 96$) + bias $v210$ stride-2 downsampled store to $t2$.
- Measured RADV GFX1201 ACO stats:
  - SGPRs: 108
  - VGPRs: 216
  - Spilled SGPRs: 0
  - Spilled VGPRs: 0
  - Scratch size: 0 bytes

## `k_sr_tube_split` Details (stages 2..7: `k_sr_tube_split<128, 4, false>`)
- Grid: `((L3w + shift_x + 3) >> 2, (L3h + shift_y + 3) >> 2)` workgroups of 64 threads.
- Coverage: 4x4 pixel tile (16 pixels x 128 channels).
- 4 threads per pixel, each processing 32 channels.
- RMSNorm 1: f16 sum of squares across 128 channels via subgroup shuffle reduction + $2^{-14}$ epsilon.
- Group projections $w210$ ($4 \times (64 \times 128)$) with non-linear polynomial mixing.
- Group projections $w1f0$ ($4 \times (128 \times 32)$) + bias $v270$ + residual addition.
- RMSNorm 2 scaling by $s198$.
- FFWD MLP expansion $w180$ ($128 \to 512$) + bias $v228$ + polynomial activation.
- FFWD MLP contraction $w1d0$ ($512 \to 128$) + bias $v240$ + residual addition.
- Output store to ping-pong bottleneck tensors $t5$ / $t6$.
- Measured RADV GFX1201 ACO stats:
  - SGPRs: 108
  - VGPRs: 120 (well below 256 limit)
  - Spilled SGPRs: 0
  - Spilled VGPRs: 0
  - Scratch size: 0 bytes

## `k_sr_enc0` Details
- Grid: `(W0/8, H0/8)` workgroups of 32 threads.
- Computes 128 input features across 2x2 render pixels (color, depth, motion vectors, dither table).
- Linear projection by matrix $w_{200}$ ($64 \times 128$ fp8) + bias $v_{210}$ to produce $t_0$ ($C=64$ at level 1).
- Stored in planar-blocked `[C/32][H][W][32]` layout.
- Measured RADV GFX1201 stats: 108 SGPRs, 72 VGPRs, 0 spills, 0 scratch bytes.

## `k_sr_dec0` Details
- Grid: `(W0/8, H0/8)` workgroups of 32 threads.
- Linear projection of $t_8$ ($C=64$) by matrix $w_{f40}$ ($128 \times 64$ fp8) + bias $v_{f50}$ ($128$ f16).
- Decodes `gates` (RGBA8), `factors` (RGBA8), `aniso` (RGBA16F, scaled by $s_0$), and `hidden` state.
- Measured RADV GFX1201 stats: 108 SGPRs, 96 VGPRs, 0 spills, 0 scratch bytes.

## `k_sr_post` Details
- Grid: `((W15+15)>>4, (H15+15)>>4)` workgroups of 256 threads.
- Subpixel bicubic Catmull-Rom reconstruction of `g_col` with exact jitter alignment.
- Reprojection and bilinear sampling of previous history `ihist[p]` guided by motion vectors.
- Adaptive blending using network-predicted `gates`.
- Writes full-resolution intermediate `full15` in 32-bit packed `R11G11B10_FLOAT` and updates ping-pong history `ihist[p^1]`.
- Measured RADV GFX1201 stats: 108 SGPRs, 96 VGPRs, 0 spills, 0 scratch bytes.

## `k_sr_downsample` Details
- Grid: `((oW+15)>>4, (oH+15)>>4)` workgroups of 256 threads.
- Bicubic sampling of `full15` from 1.5x history resolution to final target output resolution.
- Exposure normalization: applies `1.0 / expo`.
- Preserves alpha channel from input colour pass.
- Outputs final anti-aliased RGBA32F frame buffer.
- Measured RADV GFX1201 stats: 108 SGPRs, 96 VGPRs, 0 spills, 0 scratch bytes.

## `k_sr_gather` Details
- Grid: `((W+15)>>4, (H+15)>>4)` workgroups of 256 threads.
- Format conversion for game-supplied inputs:
  - RGBA16F / RG16F / R32F (format code 0): bit-identical pass-through.
  - R8G8B8A8_UNORM (format code 5): normalized colour conversion.
  - D24_UNORM_S8_UINT (format code 16) and D16_UNORM (format code 9): hardware depth conversion.
  - R16G16_UNORM (format code 18): motion vector normalization.
  - High-res output motion vectors (`k_sr_gather_mv`): converts output-resolution MVs to `g_mvhi`.
- Measured RADV GFX1201 stats: 108 SGPRs, 24 VGPRs, 0 spills, 0 scratch bytes.

## `k_sr_exposure` Details
- Grid: 1 workgroup of 64 threads.
- Computes fixed exposure or parallel log-luminance reduction for auto-exposure.
- Measured RADV GFX1201 stats: 108 SGPRs, 96 VGPRs, 0 spills, 0 scratch bytes.

