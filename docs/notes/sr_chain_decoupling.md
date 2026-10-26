# The SR network is decoupled from the output image (2026-10-27)

Found while chasing the k_sr_stage attempt-5 softmax question. This is why every
sr_stage / sr_tube_split / sr_dec0 kernel attempt in the record - including bit-exactness
improvements - moved the pipeline number by ~0.00 dB, and why the `DLSS5_AMD_INJECT_DIR`
leverage table attributes ~86% of the error to stage/tube while those kernels' own changes
are pipeline-inert.  **The image the referee sees is produced by `sr_post`'s spatial
reconstruction of `g_col` plus history; the network's outputs never reach it, and the
network's input never contained an image.**

## The three independent breaks

1. **`shaders/sr/sr_enc0.comp` writes a placeholder `t0`** (lines 83-98): 64 per-channel
   bias constants (with a sawtooth fallback pattern when a bias is zero) and *nothing else* -
   no tonemapped colour, no albedo, no motion, no constants.  The whole transformer stack
   runs on a content-free bias pattern.  (Contrast `shaders/rr/rr_enc0_feat.comp`, where the
   analogous feature tensor E0be8 was decoded for real: tonemapped RGB, albedo, mv*3,
   constants - 72.91% bit-exact, `docs/notes/k_enc0_feat.md`.)

2. **The mode-2 (decoder) stages never receive their skip inputs.**  The oracle's dumps show
   them as arguments:
   - launch 11 `k_sr_stage<96,4,2,128>` reads `t6` (in) **and `t3` (skip)** -> writes `t7`
     (pointers 8/16/24 = t6/t3/t7);
   - launch 12 `k_sr_stage<64,2,2,96>` reads `t7` **and `t1`** -> writes `t8`.
   The port's `pipeline.cpp` st8/st9 pass `{wbuf, t_act[6], t_act[7], t_act[7]}` /
   `{wbuf, t_act[7], t_act[8], t_act[8]}` - no skip binding at all - and `sr_stage.comp`
   has no skip input path.  `t1` (stage0's attention output) is written and **never read by
   anything**; `t3` likewise.  This also invalidates the earlier reading of the leverage
   table: stage9/dec0/post looked like the error sources because the decoder reconstructs
   from t6/t7 alone.

3. **The restored attempt-5 `sr_post` ignores gates/factors/aniso** - they are declared as
   bindings but never referenced in the shader body (`grep -n gates shaders/sr/sr_post.comp`:
   binding declaration only).  Even if dec0 produced perfect confidence channels, they would
   not influence `full15`/`dhist`/`ihist1`.

## Measurements that prove it

- `DLSS5_AMD_DUMP_BUF` (new inert diagnostic, `src/vk/runner.cpp`) dumps any named chain
  buffer after GPU completion.  With the probe-verified Q5 softmax x32 applied to
  `sr_stage.comp` + `sr_tube_split.comp`:
  - `kernel_run` on launch 3 (oracle inputs): t1 **changes** (1.1595% -> 0.4305% bit-exact,
    first mismatch port -32.0 vs oracle -0.625) - the shader edit is live;
  - chain `t2` **changes** (byte-compare old vs new spv, `DUMP_BUF=t2:...`);
  - chain `t8` and `gates` **change**;
  - `out_0..8.f32` are **byte-identical** for all 9 frames, old vs new spv, with and without
    oracle-input injection at `enc0`.  Nothing that happens inside the network reaches the
    image.
- `DLSS5_AMD_INJECT_DIR` masks that replace post/downsample outputs still move the image
  (downsample -> 120 dB, post -> 80.79 dB): only the `sr_post`/`sr_downsample` segment is
  live.  With the current tree the entire end-to-end number is
  `post(g_col, history) -> downsample`, i.e. a spatial upscaler with a temporal filter whose
  confidence inputs are disconnected.

## Consequence for the campaign order

The V4 bar for SR requires, in order:
1. real `t0` features in `sr_enc0.comp` (channel layout decodable with the synthetic-frames +
   `srref --dump` probe method of `docs/notes/k_sr_stage.md` attempt 5; the RR `k_enc0_feat`
   decode is the template);
2. the mode-2 skip inputs wired through `pipeline.cpp` + `sr_stage.comp` (binding + add);
3. `sr_post` consuming gates/factors/aniso (the a3a050b attempt-6 shader has the u8vec4
   confidence path; it measured worse *only because the network is dead* - re-evaluate after
   1+2);
4. only then does per-kernel fidelity work (softmax Q5 scale, max-subtraction/dead-zone,
   the denominator table question) start moving the referee number.

The Q5 x32 softmax fix itself is probe-verified (see `docs/notes/k_sr_stage.md` attempt 5)
but was **reverted from the shaders** for now: on real inputs it worsened the launch-3
diffdump (0.43% vs 1.16% bit-exact) because the surrounding attention semantics (dead-zone
at equal logits, denominator table at low Sigma) are not yet settled, and in the current
decoupled chain it cannot move the pipeline number either way.  It lands with the attention
completion, not before.

## Campaign order and rules as set on 2026-10-26 (operator)

Order: **A** (refresh `HANDOFF.md`) -> **D** (profile the ~60 ms/frame, quiet GPU) ->
**B** (wire the network) -> **C** (judge by per-kernel oracle-input fidelity) ->
**E** (decode, probes first).  D precedes B deliberately: today's cost is the *disconnected*
chain (`bf60ac0`), so it is the only clean baseline before wiring adds reads; and if real time
is far out of reach that changes how much fidelity work is worth.  One quick re-profile after
B records wiring's cost delta.

**No-revert rule (pre-committed).**  Wiring `sr_post` to gates/factors/aniso will make it
faithfully consume outputs from kernels that are still wrong (`k_sr_stage` correlation 0.455
with oracle inputs; `k_sr_dec0` ~0.5% exact on `hidden1`), so the end-to-end figure is expected
to drop from 33.76 dB toward ~28 dB.  This already happened once in the other direction: the
faithful post `a3a050b` measured worse (28.15/30.14) and was reverted by `99b33b2`.  **Do not
undo the wiring to recover a number, under any circumstances.**  Until the network kernels
improve, progress is judged per-kernel with oracle inputs (launch-level PSNR/correlation via
`DLSS5_AMD_INJECT_DIR` / `diffdump.py`), not by the end-to-end figure.  The commit message for
the wiring must state the expected drop explicitly so nobody mistakes it for a regression.
Attempt-5 `sr_post` is a **stand-in**, never "the baseline".

**Instrument ranking (unchanged by the emulator-ban lift).**  srref synthetic-weight probing
primary - it settled `k_sr_stage` attempt 5 faster than four attempts of ISA reading and its
results verify directly against oracle dumps; the emulator (`build/work/enc0emu.py`, authorized
2026-10-26) secondary; ISA reading last.  Order by measured leverage.

**Milestone (orchestrator, 2026-11-02):** after A, D and B have landed, at least one large
network kernel - `k_sr_stage` or `k_rr_level` - must reach **>= 40 dB per-launch PSNR against
oracle inputs** (diffdump oracle-input mode / the injection harness, the same machinery as the
chain-leverage tables), with the measurement pasted.  If it misses, that is a condition-(c)
report: come back with the actual number, what was tried, and why - no silent continuation
past the date.  Target and date are fixed until the operator says otherwise.

## Amendments 2026-10-26 (three, from the orchestrator)

**1. The 40 dB per-launch metric is defined here, before any measurement.**  A per-launch
output of `k_sr_stage` / `k_rr_level` is an fp8 activation tensor spanning up to +-448, so
PSNR depends entirely on the chosen peak and an undefined peak would make the milestone
gaming-able and incomparable across sessions.  Fixed definition, all three computed by **one
committed script**:

- **PSNR with peak = the oracle tensor's max |x|** (per dump; the peak is stated in the output),
- **Pearson correlation**,
- **relative RMSE** = RMS(error) / RMS(oracle tensor).

The script lives under `tools/report/` next to `pipeline_report.py` - it is a **new** file, so
`tools/verify/` stays untouched and frozen - and is written in **Step C**, alongside the metric
shift.  It speaks only for per-launch kernel claims; **pipeline-level numbers remain the
exclusive domain of `tools/report/pipeline_report.py`**, as before.  Every subsequent per-launch
claim is produced by this script and no other.  Context for the tripwire: `k_sr_stage` today is
0.455 correlation, i.e. ~0 dB on any sensible definition, so the milestone asks for a large jump
in about four working days - a miss leads to the condition-(c) reassessment report with real
data, not to a rushed push.

**2. Step D's target is distance from NeuralHub's own speed, not NVIDIA-class 1-3 ms.**
`DEEPSEEK-ASSESSMENT.md` measured `rrref` at ~13 ms/frame for 1080p -> 4K on this machine:
NeuralHub itself does not fit 1-3 ms, so that figure must **not** appear in the Step D report
as the bar.  Requirements:

- time **each dispatch** with GPU timestamps, at the **real 1080p -> 4K size**, not 256x144;
- **exclude setup and uploads** - frame time is the frame's render work only;
- time **`srref` and `rrref` the same way, on the same machine, at the same sizes** as the port;
- the report answers exactly one question: **how far is the port from NeuralHub's own speed on
  this card, per dispatch and per frame** - a port/oracle ratio table with pasted numbers.
  Whether either side fits a real-time budget is a separate judgment the operator makes from
  that table, not a conclusion the report draws.

**3. Provenance on every pasted number - an orchestrator gate.**  Every measurement offered as
a claim (mine, or the orchestrator's on re-run) must be traceable to an exact build:

- **commit SHA**,
- **SHA-256 of the shader binaries used**,
- **input directory**,

all three recorded in the paste.  Rationale is already in this project's history:
`docs/notes/k_rr_level.md` says which working tree produced the `inf` dump "is not
recoverable" - that must never happen again.  A number without the trio is not accepted.  The
trio is part of the milestone report format too.
