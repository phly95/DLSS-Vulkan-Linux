# Measurement: the one canonical pipeline command

Pipeline PSNR and GPU-time rows in `docs/PORT-STATUS.md` / `HANDOFF.md` are only trustworthy when
produced by this command:

```bash
build/venv/bin/python -I tools/report/pipeline_report.py PORT_DIR ORACLE_DIR \
    [--frames DIR] [--log LOG]... [--name CELL]
```

- **What it compares**: the port's `out_<i>.f32` (RGBA32F) against the **oracle** — NeuralHub's own
  kernels (`srref` / `rrref` under ROCm) running the same frames on the same card, which is exactly
  the pair `tools/verify/run_all.sh` calls V4 with
  (`check-cell out_port/<cell> out_ref/<cell>`; see `tools/verify/eval_verify.py`).
- **The formula is the referee's, not a new one**: the script *imports* `tools/verify/eval_verify.py`
  (frozen by rule R4, never edited) and calls its `psnr()` — `tone_map(x) = max(x,0)/(1+max(x,0))`,
  `psnr = -10*log10(mean((tm(a)-tm(b))^2))` — so its numbers are directly comparable with V4's
  `>= 50 dB/frame, >= 55 dB mean` bar, which the tool prints under each table.
- **Auditable**: every table prints the exact command and absolute paths of the inputs and the
  `frame.json` used, plus the frame count. Timings are read from real `GPU_FRAME_MS` lines in the
  logs you pass (`n/a` when absent — never estimated).
- **Self-test**: `build/venv/bin/python -I tools/report/pipeline_report.py --self-test` validates the
  tool on synthetic data with no GPU and exits non-zero on failure.

**Why this file exists.** The historical pipeline claims (108.02 dB → 54.16 → 42.40 → 27.79 dB and
0.40 ms → 57.772 ms on the same sequence in one day) were produced by tooling that was never
committed — `grep -rn "Mean PSNR" --include=*.py --include=*.cpp --include=*.sh` found no producer
in the tree (see `docs/OPEN.md` "Whole-pipeline mismatch"). Such numbers are superseded by anything
printed by the command above; this script is the first committed reporter.

## 2026-10-25: the restored 256x144 oracle-sequence replay (`build/breadth-baseline`)

Real inputs were available: the port outputs and the oracle's outputs of the same 9-frame sequence.
Note `build/breadth-baseline/{sr_M,sr_L,rr_F}_ref/out_0.f32` are byte-identical (md5) to
`build/oracle/oracle-8dbf027a9e-s20261025/{sr_M,sr_L,rr_F}/out_0.f32`, i.e. the reference side is
the oracle replay, not another port run.

```bash
build/venv/bin/python -I tools/report/pipeline_report.py \
    build/breadth-baseline/sr_M build/breadth-baseline/sr_M_ref \
    --log build/breadth-baseline/sr_M.log --name sr_M
build/venv/bin/python -I tools/report/pipeline_report.py \
    build/breadth-baseline/sr_L build/breadth-baseline/sr_L_ref \
    --log build/breadth-baseline/sr_L.log --name sr_L
build/venv/bin/python -I tools/report/pipeline_report.py \
    build/breadth-baseline/rr_F build/breadth-baseline/rr_F_ref \
    --log build/breadth-baseline/rr_F.log --name rr_F
```

Pasted output of those three commands, run 2026-10-25:

```text
--- pipeline report: sr_M ---
inputs (absolute paths -- exactly what was measured):
  command: /home/deck/DLSS5-AMD-Linux/build/venv/bin/python -I tools/report/pipeline_report.py build/breadth-baseline/sr_M build/breadth-baseline/sr_M_ref --log build/breadth-baseline/sr_M.log --name sr_M
  port   : /home/deck/DLSS5-AMD-Linux/build/breadth-baseline/sr_M
  oracle : /home/deck/DLSS5-AMD-Linux/build/breadth-baseline/sr_M_ref
  frame  : /home/deck/DLSS5-AMD-Linux/build/breadth-baseline/sr_M_ref/frame.json
  shape  : 9 frames, 512x288 RGBA32F, out_<i>.f32
  psnr   : tools/verify/eval_verify.py tone-mapped PSNR (imported, not copied)
  frame   PSNR dB
      0      27.79
      1      29.63
      2      30.41
      3      31.17
      4      29.58
      5      30.48
      6      31.10
      7      30.32
      8      30.07
  min 27.79  mean 30.06  max 31.17  (9 frames)
  V4 bar (eval_verify.py: min >= 50.0, mean >= 55.0): FAIL
timings (as found in the logs; n/a = not present, never estimated):
  log    : /home/deck/DLSS5-AMD-Linux/build/breadth-baseline/sr_M.log
            GPU_FRAME_MS frames 0,1,2,3,4,5,6,7,8 (9)
            median (frames 1..8): 57.7715 ms
            min    (frames 1..8): 52.4679 ms
--- pipeline report: sr_L ---
inputs (absolute paths -- exactly what was measured):
  command: /home/deck/DLSS5-AMD-Linux/build/venv/bin/python -I tools/report/pipeline_report.py build/breadth-baseline/sr_L build/breadth-baseline/sr_L_ref --log build/breadth-baseline/sr_L.log --name sr_L
  port   : /home/deck/DLSS5-AMD-Linux/build/breadth-baseline/sr_L
  oracle : /home/deck/DLSS5-AMD-Linux/build/breadth-baseline/sr_L_ref
  frame  : /home/deck/DLSS5-AMD-Linux/build/breadth-baseline/sr_L_ref/frame.json
  shape  : 9 frames, 512x288 RGBA32F, out_<i>.f32
  psnr   : tools/verify/eval_verify.py tone-mapped PSNR (imported, not copied)
  frame   PSNR dB
      0      26.08
      1      27.92
      2      30.18
      3      31.21
      4      29.52
      5      29.95
      6      31.04
      7      29.47
      8      29.68
  min 26.08  mean 29.45  max 31.21  (9 frames)
  V4 bar (eval_verify.py: min >= 50.0, mean >= 55.0): FAIL
timings (as found in the logs; n/a = not present, never estimated):
  log    : /home/deck/DLSS5-AMD-Linux/build/breadth-baseline/sr_L.log
            GPU_FRAME_MS frames 0,1,2,3,4,5,6,7,8 (9)
            median (frames 1..8): 58.1453 ms
            min    (frames 1..8): 51.3344 ms
--- pipeline report: rr_F ---
inputs (absolute paths -- exactly what was measured):
  command: /home/deck/DLSS5-AMD-Linux/build/venv/bin/python -I tools/report/pipeline_report.py build/breadth-baseline/rr_F build/breadth-baseline/rr_F_ref --log build/breadth-baseline/rr_F.log --name rr_F
  port   : /home/deck/DLSS5-AMD-Linux/build/breadth-baseline/rr_F
  oracle : /home/deck/DLSS5-AMD-Linux/build/breadth-baseline/rr_F_ref
  frame  : /home/deck/DLSS5-AMD-Linux/build/breadth-baseline/rr_F_ref/frame.json
  shape  : 9 frames, 512x288 RGBA32F, out_<i>.f32
  psnr   : tools/verify/eval_verify.py tone-mapped PSNR (imported, not copied)
  frame   PSNR dB
      0      27.65
      1      27.19
      2      26.93
      3      26.97
      4      27.84
      5      27.29
      6      26.76
      7      26.91
      8      26.78
  min 26.76  mean 27.15  max 27.84  (9 frames)
  V4 bar (eval_verify.py: min >= 50.0, mean >= 55.0): FAIL
timings (as found in the logs; n/a = not present, never estimated):
  log    : /home/deck/DLSS5-AMD-Linux/build/breadth-baseline/rr_F.log
            GPU_FRAME_MS frames 0,1,2,3,4,5,6,7,8 (9)
            median (frames 1..8): 0.7600 ms
            min    (frames 1..8): 0.7582 ms
```

These reproduce the `vk_runner` row in `docs/PORT-STATUS.md` exactly (SR M 27.79/30.06, SR L
26.08/29.45, RR F 26.76/27.15; median 57.772 / 58.145 / 0.760 ms rounded to 3 decimals), so that
row's provenance is this replay — but before this tool existed the same row could not be regenerated
from a clean clone.

Self-test (no GPU, no real data), run 2026-10-25:

```text
$ build/venv/bin/python -I tools/report/pipeline_report.py --self-test
  identical inputs -> PSNR >= 100 dB           ok
  known-noise PSNR matches formula (tol 1e-3 dB) ok
  known-noise PSNR in a sane range (20..80 dB) ok
  missing out_<i>.f32 fails                    ok
  non-finite input fails                       ok
  complete directories -> exit 0               ok
  missing out_<i>.f32 -> exit 1                ok
  missing frame.json -> exit 1                 ok
  GPU_FRAME_MS parsed                          ok
  log without timings -> empty (n/a)           ok
  missing log -> empty (n/a)                   ok
  timed log reports median (frames 1..2): 3.5 ms ok
  empty/absent logs print n/a                  ok
PIPELINE_REPORT SELF-TEST: PASS
```

## Producing a fresh table on the RX 9070 XT

The referee writes each cell's port and oracle outputs plus its logs into its run directory; the
report consumes them directly:

```bash
# 1. the full referee run (V1-V10); prints "DLSS5 Vulkan port referee, run <ID>" and leaves
#    build/verify/run_<ID>/ with out_port/<cell>, out_ref/<cell>, port_<cell>.log, ref_<cell>.log
bash tools/verify/run_all.sh

# 2. per-cell report, one command per cell of interest (example: the moving SR M cell)
build/venv/bin/python -I tools/report/pipeline_report.py \
    build/verify/run_<ID>/out_port/sr_M_seq build/verify/run_<ID>/out_ref/sr_M_seq \
    --log build/verify/run_<ID>/port_sr_M_seq.log \
    --log build/verify/run_<ID>/ref_sr_M_seq.log \
    --name sr_M_seq
```

Cell names are the `CELLS` list of `tools/verify/run_all.sh` (`sr_M_seq`, `sr_L_seq`, `rr_F_seq`,
the static/format/parameter cells, ...). The oracle side `out_ref/<cell>/frame.json` is written by
the referee (`run_all.sh` copies it from the cell's frames directory); if neither side carries one,
pass `--frames <frames dir>`.

Two caveats worth remembering when citing a number:

- `run_all.sh` **deletes every `build/verify/*/out_ref` at the start of the next run** (line 80), so
  copy the `out_ref` directories (or the printed report) right after the run you intend to cite.
- no `out_ref` directory survives in this repository today: the recorded referee runs either failed
  before the oracle stage or had their oracle outputs pruned by a later run, which is why the table
  above uses `build/breadth-baseline` instead of a referee run directory.

## 2026-10-26: the NGX host bridge under Proton (V9, `src/pe/nvngx.cpp`)

The bridge's numbers come from the three commands below; nothing else in the tree produces them.

```bash
# 1. build the port, the package and ngx_client, and install the package into a game folder
#    (install.sh also writes DLSS5_AMD_WEIGHTS into dlss5-amd-launch.env and symlinks
#    build/ngx-weights/{M,L,rr} to the packed weight trees)
bash src/vk/build.sh
bash tools/verify/ngx_client/build.sh build/stage2/verify/game/ngx_client.exe
cp build/re/nvngx_dlss_310.9.1.dll  build/stage2/verify/game/nvngx_dlss.dll
cp build/re/nvngx_dlssd_310.9.1.dll build/stage2/verify/game/nvngx_dlssd.dll
bash build/package/install.sh build/stage2/verify/game optiscaler

# 2. one V9 cell, launched the way tools/verify/run_all.sh's proton_run launches it
#    (same env file, same prefix, same working directory, Z:\ paths for the frames)
PROTON="$HOME/.local/share/Steam/steamapps/common/Proton 10.0/proton" \
bash build/stage2/proton_run.sh <log> --api vulkan --feature rr --frames <Z:\...\base\rr> \
    --out <Z:\...\out_vk_rr3> --test-cycles
#    cells:  --api {vulkan,d3d12} --feature sr --preset {M,L} | --feature rr   (12 frames each)
#    refusal cells: --expect-create-fail in a folder without nvngx_dlss{,d}.dll or with the wrong size

# 3. compare the outputs with the native chain and the ground truth using the referee's own psnr()
build/venv/bin/python build/stage2/evidence.py <bridge_out> <native_out> <gt_dir> --label "..."
```

`evidence.py` imports the frozen `tools/verify/eval_verify.py` and calls its `psnr()` (`tone_map(x) =
max(x,0)/(1+max(x,0))`, `psnr = -10*log10(mean((tm(a)-tm(b))^2))`), so these are V4's numbers and not a
new metric.  "native" is `build/vk/vk_runner` on the same frame directory and the same packed
weights; "gt" is `build/verify/run_<ID>/base/<cell>_gt`, the directory `run_all.sh` passes to
`check-cell --gt`.  `sha256sum out_*.f32` is compared directly for RR, which is reproducible; SR is
not (see `docs/OPEN.md`, "SR chain run-to-run drift"), so SR is quoted by PSNR only.

### Results, RX 9070 XT, Proton 10.0, 12 frames per cell, `--test-cycles` on every cell

All six cells exited 0 with 12 outputs, 0 `radv: GPUVM fault detected` lines and 0 kernel
`amdgpu ... page fault` lines (`sudo journalctl -k --since ... | grep -cE "page fault"` -> 0).

| cell | non-zero floats | bridge PSNR vs gt (min/mean/max, dB) | native PSNR vs gt | largest per-frame gap |
|---|---|---|---|---|
| `--api vulkan --feature sr --preset M`, 1920x1080 -> 3840x2160 | 398131200 / 398131200 | 28.0550 / 33.3818 / 36.0200 | 28.0550 / 33.3818 / 36.0200 | 0.0000 dB |
| `--api vulkan --feature sr --preset L`, 1920x1080 -> 3840x2160 | 398131200 / 398131200 | 26.5742 / 32.9690 / 36.8125 | 26.5742 / 32.9686 / 36.8124 | 0.0022 dB |
| `--api vulkan --feature rr`, 1920x1080 -> 3840x2160 | 398131200 / 398131200 | 27.4949 / 27.7435 / 27.8878 | 27.4949 / 27.7435 / 27.8878 | 0.0000 dB |
| `--api d3d12 --feature sr --preset M`, 1280x720 -> 1920x1080 | 99532800 / 99532800 | 31.2780 / 33.0436 / 34.6731 | 31.2780 / 33.0436 / 34.6730 | 0.0002 dB |
| `--api d3d12 --feature sr --preset L`, 1920x1080 -> 3840x2160 | 398131200 / 398131200 | 26.5739 / 32.9689 / 36.8127 | 26.5742 / 32.9686 / 36.8124 | 0.0013 dB |
| `--api d3d12 --feature rr`, 1920x1080 -> 3840x2160 | 398131200 / 398131200 | byte-identical to the native chain | = native | 0.0000 dB |

The RR cells are stronger than PSNR: for **both** APIs every one of the 12 `out_i.f32` files hashes
to the same value as `build/vk/vk_runner` on the same frames,

```
$ diff build/stage1/before_rr_F.sha256 build/stage2/rr3.sha256 && echo identical
$ sha256sum build/stage2/rr3.sha256 build/stage1/before_rr_F.sha256
4543e334dcad107c52071d5438cca17c876af93eba5ece95055b183b3c48ea50  build/stage2/rr3.sha256
4543e334dcad107c52071d5438cca17c876af93eba5ece95055b183b3c48ea50  build/stage1/before_rr_F.sha256
```

The eight `--expect-create-fail` cells (no `nvngx_dlss*.dll` or a wrong-size one, x
`{vulkan,d3d12}` x `{sr,rr}`) all exit 0, and the refusal is logged by the package itself:

```
[       6 ms] Vulkan CreateFeature refused: nvngx_dlss.dll is missing or not its expected size next to _nvngx.dll
[     118 ms] D3D12 CreateFeature refused: nvngx_dlss.dll is missing or not its expected size next to _nvngx.dll
[       8 ms] Vulkan CreateFeature refused: nvngx_dlssd.dll is missing or not its expected size next to _nvngx.dll
```

**Proton version matters here.**  `run_all.sh` selects Proton with `sort -V | tail -1`, which is
Proton - Experimental on this machine, and every process it starts hangs before loading anything
(`timeout 90 .../Proton - Experimental/proton run ... --nonsense-arg` -> `RC=124`, Proton 11.0 ->
`RC=124`, Proton 10.0 -> `RC=1`).  The table above was therefore measured on Proton 10.0 via the
`PROTON=...` override; see `docs/OPEN.md`, "Proton 11 process startup".
