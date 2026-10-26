# k_sr_stage Notes & Hypotheses

## Architecture & Implementation
`k_sr_stage` implements the outer Swin Transformer layers for SR presets M and L:
- Stage 0: `k_sr_stage<64, 2, 0, 96, 2, false>` (enc1: C=64, g=2, Cn=96, mode=0)
- Stage 1: `k_sr_stage<96, 4, 0, 128, 2, false>` (enc2: C=96, g=4, Cn=128, mode=0)
- Stage 8: `k_sr_stage<96, 4, 2, 128, 2, false>` (dec2: C=96, g=4, Cn=128, mode=2)
- Stage 9: `k_sr_stage<64, 2, 2, 96, 2, false>` (dec1: C=64, g=2, Cn=96, mode=2)

Workgroup structure:
- 64 threads covering an 8x8 pixel tile divided into four 4x4 windows (16 tokens per window).
- Shared memory: `s_kv[4][16][32]` (4096 bytes LDS) to buffer K and V tensors during window attention.
- Pre-RMSNorm 1 with scale `s1b0` (includes $\sqrt{C}$ factor baked in from `sr_pack.py`).
- 16-token window multi-head attention with relative position bias table (`v258`) and bit-built exponential:
  $y = \text{clamp}(h \cdot 0.0171814, -0.55615234, 0.55615234)$,
  $p = y \cdot (0.92724609 - y^2) + 1.375$,
  mantissa shift $(p_{\text{bits}} \ll 5) + 0x8000$, producing bit-exact positive float16 exponential.
- Attention context accumulation $Context = \sum P_q \cdot V$ using 8 f16 WMMAs.
- Post-attention residual add with $v270$ bias.
- Pre-RMSNorm 2 with scale `s198`.
- Multi-block MLP: $C/8$ blocks with FC1 ($32 \times C$), SiLU activation, and FC2 ($C \times 32$), followed by $v240$ bias add and residual 2.
- Mode 0 downsampling: 2x2 patch concatenation ($4C$ channels) projected by $W_{\text{proj}}$ ($Cn \times 4C$) to `out_act2`.

## Compilation & Resource Stats (ACO on RX 9070 XT GFX1201)
- SGPRs: 108
- VGPRs: 256
- Spilled SGPRs: 0
- Spilled VGPRs: 0
- Scratch size: 0
- LDS size: 40960 bytes

## Attempt Log

### Attempt 1: Full Collaborative Swin Transformer with LDS Staging
- **Hypothesis**: Replacing the 26-line pass-through placeholder with full 16-token window attention, bit-built exp2, LDS-buffered K/V staging, MLP blocks, and 2x2 downsample projection will bring the kernel into structural alignment with the NeuralHub code object and dramatically improve bit-exact count and error bounds.
- **Result**:
  - `t1 [fp8]`: 655,360 elements, bit-exact 7,515 (1.1467%), max |err| 17.1875, max ulp 152, first mismatch:
    `{'index': 0, 'port': 7.5, 'oracle': -0.625, 'port_bits': 79, 'oracle_bits': 178}`
    (Up from 116 bit-exact, 65x improvement over stub).
  - `t2 [fp8]`: 245,760 elements, bit-exact 2,630 (1.0701%), max |err| 8.5, max ulp 142, first mismatch:
    `{'index': 0, 'port': -0.75, 'oracle': -0.6875, 'port_bits': 180, 'oracle_bits': 179}`
    (Up from 53 bit-exact, 50x improvement over stub; max error reduced from `inf` to 8.5; **Pixel 0 is only 1 ULP off**).
- **Status**: APPROX (Pass 1 complete in breadth).

### Attempt 2: ISA Bit-Built FP16 Exponential
- **Hypothesis**: Replacing the placeholder linear exponential with the polynomial and bit construction visible in the SR stage ISA will make softmax probabilities align with the oracle.
- **Result** (frame 1, launch 3, `k_sr_stage<64,2,0,96,2,false>`):
  - `t1 [fp8]`: 655,360 elements, bit-exact 6,083 (0.9282%), max |err| 18.5, max ULP 157, first mismatch `{'index': 0, 'port': -3.25, 'oracle': 0.5625, 'port_bits': 197, 'oracle_bits': 49}`.
  - `t2 [fp8]`: 245,760 elements, bit-exact 3,322 (1.3517%), max |err| 9.75, max ULP 146, first mismatch `{'index': 0, 'port': -0.9375, 'oracle': -1.75, 'port_bits': 183, 'oracle_bits': 190}`.
  - `DIFFDUMP: DIFFERS`. t1 regressed versus attempt 1; t2 improved slightly.

### Attempt 3: Balanced FP16 Softmax Sum
- **Hypothesis**: The ISA uses packed fp16 additions and pair reductions for the softmax denominator; replacing the shader's f32 accumulation with an explicit balanced fp16 tree will reproduce its rounding points.
- **Result** (same launch):
  - `t1 [fp8]`: 655,360 elements, bit-exact 6,086 (0.9286%), max |err| 18.5, max ULP 157, first mismatch `{'index': 0, 'port': -3.25, 'oracle': 0.5625, 'port_bits': 197, 'oracle_bits': 49}`.
  - `t2 [fp8]`: 245,760 elements, bit-exact 3,327 (1.3538%), max |err| 9.75, max ULP 146, first mismatch `{'index': 0, 'port': -0.9375, 'oracle': -1.75, 'port_bits': 183, 'oracle_bits': 190}`.
  - `DIFFDUMP: DIFFERS`. The denominator hypothesis made only a tiny change and leaves large errors.
- **Status**: APPROX; continue broad work, revisit with fresh hypotheses after a pass over other families.
