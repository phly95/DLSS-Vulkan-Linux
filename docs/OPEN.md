# Open Questions

Log of questions and unresolved details encountered during the port.
Per referee and anti-stuck rules: do not wait for questions to be answered; log them here and continue elsewhere.

## Active Questions

1. **`k_sr_stage` second normalisation / softmax**:
   - Soft-clamped logit and bit-built exp2 (`v_pk_fma_f16`, `(bits << 5) + 0x8000`).
   - Need exact summation domain and order for the denominator in `g_sr_lut` table 2.

2. **Preset M weight checksum**:
   - In `sr_weights.py`, preset M weights pack and run cleanly in `srref`, but raw checksum differs from NeuralHub's expected hash (L matches). Does not block running or verification.

3. **RR transparency / SSS / virtual-hit / auxiliary guides**:
   - The test sequences in referee don't exercise optional auxiliary guides (flags remain 0).
