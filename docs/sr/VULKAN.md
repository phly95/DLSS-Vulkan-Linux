# SR in Vulkan: kernel RE notes and port status

Status (2026-10-25): **no shader yet.** `k_sr_stage<64,2,0,96,2,false>` is being read from its ISA
(`tools/re/fn.sh sr <symbol>`, `tools/re/slice.py`); what is established is below, what is not is listed
as open. Accuracy of every kernel: **pending oracle dump** (`tools/gpu/collect_oracle.sh` on the card,
then `tools/verify/diffdump.py`). Nothing here has been compiled, run or compared.

## k_sr_stage<64,2,0,96,2,false> (encoder stage)

Code object statistics (from the code object's metadata): kernarg 56 B, 64 threads (2 waves of 32),
4096 B LDS, 176 VGPRs, 0 scratch. Instruction counts in the ISA: 152 `v_wmma_f32_16x16x16_fp8_fp8`,
8 `v_wmma_f32_16x16x16_f16`, 180 fp8 conversions, 2 `s_barrier`.

### Arguments (`StageArgs`)

| offset | ISA register | meaning |
| --- | --- | --- |
| 0x00 | s[12:13] | packed stage weights (`sr_pack.py` `stageN.bin`) |
| 0x08 | s[14:15] | input activations |
| 0x18 | s[4:5] | output activations |
| 0x20 | s[6:7] | second output (projection to the next stage's width, Cnext = 96) |
| 0x28 | s8, s9 | W, H |
| 0x30 | s10, s11 | tile shift x, y |

### Threads, tile, borders

- A workgroup covers an 8x8 pixel tile at (group*8 - shift). Lane mapping: x = lane & 3 (set A) and
  x + 4 (set B); y = wave*4 + ((lane >> 2) & 3). Out-of-image pixels are mirror-padded (reflect-101:
  `v_max_i32` with the negated coordinate, `v_min3` with dim - 1).
- Activations are `[C/32][H][W][32]` fp8; per 32-channel group a lane loads 8 bytes at storage
  position 8*(lane>>4 & 1) and again at +16.
- Storage position p holds logical channel P(p): P(0..15) = 0,1,16,17,2,3,18,19,4,5,20,21,6,7,22,23;
  P(16+i) = P(i) + 8 (the host's channel permutation, `docs/sr/HOST-RE.md`).

### RMSNorm 1 (rounding points in order)

1. q = x*x in f16 (`v_pk_mul_f16` on the f16-widened fp8 values).
2. A fixed packed f16 add tree over the 16 pair-sums of a lane:
   a=q0+q1 b=q2+q3 c=q4+q5 d=q6+q7 e=q8+q9 g=q10+q11 f=q12+q13 h=q14+q15;
   s1=a+e, s3=c+f, s2=b+g, s4=d+h; s1+=s3; s2+=s4; s=s1+s2 (each an f16 rounding).
3. s += the lane^16 partner's s (`ds_bpermute_b32`), then t = s.lo + s.hi (f16), then t += 0x0800 (f16 bits, 2^-14).
4. rs = `g_sr_lut[t]` (u16 table indexed by the f16 bit pattern; the second table is at +0x20000 bytes).
5. y = fp8(f16(x * f16(rs * gain))), gain = the f16 block at offset 0 of the stage weights (128 B, in
   storage-position order).

### GEMM 1 (64x64)

- Weights: A operand, 256-byte fragments, lane l holds bytes l*8..l*8+7: lane l byte c =
  Wpos[l%16][8*(l/16)+c], which is Mesa's GFX12 A-fragment layout, so a row-major `coopMatLoad` of the
  position-ordered weight matrix gives the same registers.
- Activations: B operand, built with `v_cvt_pk_fp8_f32` (n = pixel, k = position within a 16-chunk).
- Fragment f: output block f//4 (0: group 0 positions 0-15, 1: group 0 16-31, 2: group 1 0-15, 3: group 1
  16-31), input block f%4 in the same order. Weight fragment offsets 128, 384, 640, ...
- K = 64 is two K=32 chains; their f32 accumulators are combined through f16 (`v_fma_mixlo/hi_f16`
  with the per-stage scale in s11).

### Seen but not yet decoded (open)

- From ~0x1a20: a second normalisation: f16 mul by 0x2466, clamp to [0xb873, 0x3873], then
  y*(0x3b6b - y*y) via `v_pk_fma_f16` and a second fma with 0x3d80, `<<5`, `+0x8000`, byte permutes into
  fp8 positions, `ds_bpermute` reductions and a lookup in the second LUT. The function this computes
  and where each rounding happens is **open**; it must be read exactly, not approximated (R5).
- ~0x2d14-0x3420: the eight f16 WMMAs (a residual / diagonal term?) — **open**.
- fp8 WMMAs with weight offsets 2176..17600: the 4C expansion and contraction — operand wiring **open**.
- After the barrier at 0x5cfc: the Cnext = 96 projection, weights staged through LDS
  (`ds_load_2addr_b64`), offsets from 47104 — **open**.
- Output stores (`global_store_b64` near 0x57e4, 0x5c88, 0x6350): layout and which of out / out2 —
  **open**.

## Remaining kernels (HANDOFF order)

`k_sr_tube_split<128,4,false>`, decoder stages, `k_sr_enc0`, `k_sr_dec0`, `k_sr_post`,
`k_sr_downsample`, exposure, then preset L's `k_sr_enc0L_in`, `k_sr_l0`: not started.
