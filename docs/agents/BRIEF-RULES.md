# Rules for delegated tasks (every subagent brief points here)

These rules apply to every subagent working in this repository. Each brief adds the goal, the files
the agent owns, the spec and the acceptance test. The orchestrating session re-runs every
acceptance command itself before anything is committed.

## Workspace

- Repository: `/home/deck/DLSS5-AMD-Linux` (SteamOS, RX 9070 XT). It is a checkout shared with
  other agents working at the same time.
- **Edit only the files your brief lists as yours.** Everything else is read-only, even when you see
  a bug in it. Report the bug instead.
- **Never edit `tools/verify/`** (frozen referee).
- Write outputs (builds, dumps, logs, scratch scripts) **only under `build/agents/<your-name>/`**.
- Never delete anything outside that directory, and never use `rm -rf` on a path you have not
  just created.
- **Do not delete at all when you can avoid it**: use a fresh directory name per run (`run2/`,
  `run3/`, ...) instead of clearing an old one. Never write `rm` with a shell variable in the
  path (`rm -rf $A/$d`, `rm -rf "$OUT"/*`): the harness stops and asks the user for permission
  every time. If a removal is truly needed, spell the literal path out
  (`rm -rf build/agents/<you>/run1`), or do it from Python with `shutil.rmtree` on a path built
  under your own directory.
- **Do not run `git commit`, `git push`, `git stash`, `git checkout -- <file>` or `git reset`.**
  The orchestrator commits. Other agents' uncommitted work is in the same tree, so `git stash`
  or `git checkout` destroys it. To compare against HEAD, use `git show HEAD:<path> > build/agents/<you>/<file>`.
- Python always runs as `build/venv/bin/python -I ...`.
- The GPU is shared: keep runs short (the 256x144 oracle sequence, single launches). Do not run
  `tools/verify/run_all.sh`, the 1080p/4K profiling inputs, or anything that writes more than
  about 2 GB. The disk is 87% full.
- Shaders: compile **only your own** shader into your directory, e.g.
  `reference-nr-vulkan/toolchain/glslang/bin/glslang -V --target-env vulkan1.3 -Ishaders/include
  -Ireference-nr-vulkan/linux/shaders/rdna4/include shaders/sr/X.comp -o build/agents/<you>/X.spv`,
  and run it with `build/vk/kernel_run --spv build/agents/<you>/X.spv --launch L.json --out DIR`.
  Run `src/vk/build.sh` (which rewrites `build/shaders/` and `build/vk/` for everyone) only if your
  brief says you own the host code.

## Numbers

- Per-launch fidelity comes only from `tools/report/launch_report.py` (peak, PSNR, r, rRMSE),
  measured against oracle inputs. Whole-pipeline numbers come only from
  `tools/report/pipeline_report.py`.
- Every number you report comes with the exact command that produced it and its full output.
- **Never tune a constant to improve a number**, and never write an approximation of a NeuralHub
  kernel (a generic blend, textbook formula, etc.). The spec in your brief is the target. If the
  spec looks wrong, stop and report what you measured; do not invent a fix.
- `docs/sr/STAGE-LAYOUT.md` is the authoritative byte map of the SR stage weights (generated from
  NeuralHub's own packer by `tools/sr/sr_layout.py`).

## Budget and stopping

- Your brief gives an attempt budget. If the acceptance test still fails after it, stop and
  report the data. That is a useful result; a guess dressed up as a fix is not.
- Stop and report instead of continuing if:
  - a needed file or input is missing;
  - a build tool fails in a way unrelated to your change;
  - you would need to edit a file you do not own.

## Final report (your last message)

1. **Changed files**, with a 1-3 line summary each. The orchestrator reads the diff itself.
2. **Acceptance results**: each command, then its output, then PASS/FAIL against the brief's
   threshold.
3. **Open issues / surprises**: anything you measured that contradicts the brief or the docs.
4. **Not done**: anything in the brief you skipped, and why.
