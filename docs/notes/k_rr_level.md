# Notes: `k_level` / `rr_level` — where the two recorded rows came from

Question raised by `MIMO-ASSESSMENT-2026-10-25_230151.md` §5.6: `docs/PORT-STATUS.md`'s `k_level`
row reads **max |err| inf, 3824/1310720 (0.2917%) bit-exact**, while the same row in commit
`4ec6263` read **max |err| 1.25, 3789 (0.2891%)**. Same launch, or two? Written 2026-10-25 from
`git log -p`, `git show`, and the restored oracle dump; nothing here was guessed.

## Verdict

**Same oracle launch, different port measurements.** The two rows are two diffdumps of the same
oracle file against two *different* port outputs: the port's own first-element bytes differ
(0.140625 / bits 33 vs 1.125 / bits 57), so the port side was rebuilt and re-dumped between them —
it is not one measurement restated. The only committed change to `shaders/rr/rr_level.comp` in
between is `706a63e`'s rewrite (evidence below).

## Evidence: which commit pasted which number

| number | first commit that contains it | commit date/time | what that commit did |
| --- | --- | --- | --- |
| 3789 (0.2891%), max \|err\| 1.25, first mismatch port 0.140625 (bits 33) | `1695015` "docs/PORT-STATUS.md: paste diffdump numbers for k_sr_stage, k_sr_l0, k_level" | 2026-10-25 10:31 | docs-only: replaced `k_level`'s "Offline verification O7 passed" with the pasted diffdump |
| 3824 (0.2917%), max \|err\| inf, first mismatch port 1.125 (bits 57) | `706a63e` "rr_level: implement collaborative 128-thread Swin Transformer layer in LDS with 0 spills on GFX1201" | 2026-10-25 17:08 | rewrote `shaders/rr/rr_level.comp` (+168/-8) **and** updated the row in the same commit; the row's new Notes cell describes exactly that implementation (LDS staging, 2-head Swin attention, 48 VGPRs) |

`4ec6263` (2026-10-25 14:25) did **not** touch the `k_level` row: `git show 4ec6263 --
docs/PORT-STATUS.md` shows only the `k_sr_tube_split` and `k_sr_post` rows changed. Its version of
the row is inherited verbatim from `1695015`.

Between the two pastes (`1695015`, 10:31, and `706a63e`, 17:08) `shaders/rr/rr_level.comp` was
touched by exactly one of the 13 intervening commits — `706a63e` itself:
`git log --oneline 1695015..706a63e -- shaders/rr/rr_level.comp` prints only that commit, and
`git log --all --oneline -- shaders/rr/rr_level.comp` shows the file was created by the reverted
draft `560ca36` (reverted in `96ee848`), committed as breadth code in `6135cee` (00:32), and
rewritten in `706a63e`. So the port shader behind row 1.25 is the breadth version, and the only
committed change that can explain row `inf` is `706a63e`'s rewrite in that same commit.

## Evidence: same oracle launch

Both rows are for buffer `E0b54`, 1310720 fp8 elements, oracle first-mismatch value **0.1015625
(bits 29)** at index 0. In the restored dump (`build/oracle/oracle-8dbf027a9e-s20261025/rr_F/dump/`):

```bash
$ build/venv/bin/python -I tools/verify/diffdump.py launches \
      build/oracle/oracle-8dbf027a9e-s20261025/rr_F/dump/frame0 | sed -n '5p'
  4 _ZN3rrk7k_levelILi32ELi2ELi5ELi0ELi0ELi4EEEvNS_4ArgsE grid [33, 21, 1] block [32, 1, 1] wrote E0b54, E0ba8, E0bc0

$ od -An -tu1 -N1 build/oracle/oracle-8dbf027a9e-s20261025/rr_F/dump/frame0/004.E0b54.bin
 29
$ od -An -tu1 -N1 build/oracle/oracle-8dbf027a9e-s20261025/rr_F/dump/frame1/005.E0b54.bin
 17
$ cmp build/oracle/oracle-8dbf027a9e-s20261025/rr_F/dump/frame1/start.E0b54.bin \
      build/oracle/oracle-8dbf027a9e-s20261025/rr_F/dump/frame0/004.E0b54.bin && echo identical
identical
```

- frame 0, launch 4 is `rrk::k_level<32,2,5,0,0,4>`, the launch that writes `E0b54`; its output's
  first byte is 29 = 0.1015625 — the oracle value **both** rows report.
- frame 1, launch 5's `E0b54` starts with byte 17 = 0.06640625, which matches neither row → ruled out.
- `frame1/start.E0b54.bin` (the buffer's state before frame 1) is byte-identical to frame 0's
  launch-4 output, so if either measurement used the frame-1 copy it is the same content anyway.

The port side differs between the rows (0.140625/33 vs 1.125/57), i.e. two different port dumps of
that launch.

## Why one row says `max |err| inf`

`tools/verify/diffdump.py:123` maps NaN differences to infinity
(`d = np.where(np.isnan(d), np.inf, d)`; both-NaN counts as 0). The oracle file
`frame0/004.E0b54.bin` contains **zero** `0x7f` bytes (E4M3 NaN), measured:

```bash
$ build/venv/bin/python - <<'EOF'
import numpy as np
a = np.fromfile('build/oracle/oracle-8dbf027a9e-s20261025/rr_F/dump/frame0/004.E0b54.bin', dtype=np.uint8)
print(a.size, int((a == 0x7f).sum()))
EOF
1310720 0
```

so at least one `0x7f` (NaN) came from the **port** side of that measurement — the oracle file has
none. The earlier row (`1695015`) reported a finite max error, so the NaN belongs to a port dump
taken after that measurement; whether it came from `706a63e`'s rewrite or an intermediate
uncommitted state is not recoverable (below). (Both rows' bit-exact counts are ~0.29% either way;
the rewrite moved individual values a lot without changing how many words matched.)

## What could not be determined

- Which exact working tree produced the `inf` row's port dump. No port dump for frame 0 launch 4
  exists under `build/` and none was committed, so only the pasted row survives; `706a63e` is the
  only committed change consistent with it, but the diffdump itself may have run on that tree or on
  an uncommitted intermediate state — the history cannot tell which.
- Whether the rewrite made `E0b54` better or worse **overall**: re-running `diffdump.py compare` on
  the RX 9070 XT (port dump of frame 0 launch 4 vs the restored oracle) is required before either
  number is cited as the kernel's current state. Until then the row stays APPROX and both numbers
  stand as two dated measurements of two shader versions, not as a contradiction.
