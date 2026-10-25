# Open questions and blockers

Continue independent kernel work while these are unresolved.

| Item | Question / evidence | Work that can continue |
|---|---|---|
| `k_sr_stage` second normalization / softmax | Exact denominator summation domain and order from `g_sr_lut` table 2 remain unresolved. | Decode the ISA and compare the next SR launch; do not add a speculative kernel approximation. |
| SR preset M weights | The M raw-weight checksum still does not reproduce NeuralHub's recorded checksum, although packing yields the expected M layout. Is the oracle generated from these weights acceptable as the port's M reference? | Keep comparing M against the checked-in private-storage oracle and report measured results; do not claim NeuralHub checksum parity. |
| RR optional auxiliary guides | The referee frames leave RR transparency, SSS, virtual-hit, and auxiliary guide flags unset. Their host input behavior is not covered by this matrix. | Continue with the required RR preset F path and add these only after the current pipeline reaches the numeric target. |
| Whole-pipeline mismatch | Replaying the restored 256x144 oracle sequence on 2026-10-25 gives 27.79/30.06 dB min/mean for SR M, 26.08/29.45 for SR L, and 26.76/27.15 for RR F. The earlier full referee run reported 37.55 dB for `sr_M_seq`; the gap between these runs and the older status claims is unexplained. | Continue launch-by-launch `diffdump.py` comparisons from the oracle dumps to isolate the first divergent family. |
| V4 ground truth | The compact replay used oracle outputs only and had no corresponding ground-truth directory. It cannot establish the <=0.1 dB ground-truth parity condition. | Use `tools/verify/run_all.sh` after the working tree is clean and its required assets are present. |
