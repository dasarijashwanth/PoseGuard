# Day 24 walkthrough: capture-to-benchmark pipeline

Every piece of the re-run path already existed: the Day 14
capture tool records the session, the Day 16 harness
re-benchmarks it, and the Day 23 combined gate scores both cues
together. What was missing was the wiring: one command that
takes a capture file all the way to a verdict, and a checklist
short enough to follow at the desk. Day 24 builds both. The
user's own capture has still not arrived, so Day 24, like Day
16, is the machinery, proven on the synthetic fixture.

## What was built

**The pipeline: `experiments/model/day24.py`.** One command,
two ways to point it at a capture:

```
python -m experiments.model.day24 --csv <capture.csv> [--manifest <manifest.json>]
python -m experiments.model.day24 --drop-dir <folder>
```

`--drop-dir` auto-discovers the newest `.csv` in the folder;
the manifest is the `.json` with the same stem when present,
else the newest `.json`, else none. The flow:

1. **Pre-flight.** `validate_rerun_csv` runs first and the
   pipeline prints what the file holds in plain English:
   frames, tracked fraction, holds per cue with how many
   reached their depth target, and every warning. When a
   manifest is given it is checked against the CSV too (the
   Day 16 `check_manifest`). A file that fails validation
   stops here: errors printed, exit code 2, nothing
   benchmarked, nothing archived, the Day 16 convention.
2. **Day 16 re-benchmark.** `run_day16` is called as a
   function, not through `day16.main`, so `results_day16.json`
   and `BENCHMARK.md` are left untouched. The head grid and
   model comparison and the shoulder signal and model
   comparison run exactly as Day 16 defined them, with the
   Run 2 timings and rule baselines counted on this file.
3. **Day 23 combined gate.** `evaluate_combined` on the same
   session with the Run 2 timings (5 s ready, 10 s hold, 5 s
   rest): the normalized sum, the isolation scores, and the
   AND control, per cue, at the 5 s gate persistence.
4. **Archive.** `experiments/model/runs/<YYYY-MM-DD-HHMM>/`
   gets `results.json` (the full validation report, the Day
   16 result, and the combined result) and `summary.md` (the
   plain-English verdict: validation line, the Day 16
   verdict, and per cue the best combined variant with its
   threshold, trials, false positives, and whether it passes,
   plus the full variant table). The summary is also printed.

**The checklist: `experiments/CAPTURE_CHECKLIST.md`.** The
Day 14 run instructions, trimmed to what he needs at the desk:
serve `rerun-capture.html` on localhost (the PowerShell
`HttpListener` block from the Day 14 walkthrough), hips in
frame and good light, the guided 8.6 minute session with the
head 15 deg and shoulder 8 deg depth targets, the 5 minute
work sample done as ordinary work, then download the CSV and
manifest and send both. It ends with the one Day 24 command
and what comes back.

## The numbers (synthetic fixture, labeled as such)

All numbers below are from the Day 14 builder's fixture
(`experiments/model/tests/fixtures/rerun-synthetic.csv`):
clean holds stepping to target depth, flat work, 5150 frames,
98.1% tracked, no validator warnings. The fixture passes the
gate by construction; it proves the pipeline runs end to end
and can report a pass. It says nothing about the real capture.

**Pre-flight.** 5150 frames, 98.1% tracked, work sample
299.9 s; head 5/5 holds and shoulder 5/5 holds, all 10
reached their depth target; no warnings.

**Day 16 verdict.** Head passes the gate: rule grid best
(4 deg, 5.0 s, velocity cap none) and all six model rows
(LR, RF, MLP on old and new features). Shoulder passes the
gate: gated signal best (1 deg), LR and RF on the new
features. Same numbers as Day 16, because it is the same
code on the same file.

**Combined gate.** The normalized sum is the best variant on
both cues at threshold 0.5: 5/5 trials, 0 false positives,
F1 1.000, passes. The isolation scores pass too (head 2.0,
shoulder 0.5). The AND control detects 0/5 on both cues, as
Day 23 found on Run 1 and on this fixture alike: a
conjunction cannot see a single-cue hold.

| Cue | Variant | Threshold | Trials | False positives | F1 | Pass |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| head | sum | 0.5 | 5/5 | 0 | 1.000 | yes |
| head | iso | 2.0 | 5/5 | 0 | 1.000 | yes |
| head | and_control | 4.0 | 0/5 | 0 | 0.000 | no |
| shoulder | sum | 0.5 | 5/5 | 0 | 1.000 | yes |
| shoulder | iso | 0.5 | 5/5 | 0 | 1.000 | yes |
| shoulder | and_control | 1.0 | 0/5 | 0 | 0.000 | no |

One labeled archive from the fixture run is committed under
`experiments/model/runs/` (source: "synthetic fixture (Day 14
builder)"), so the archive shape is visible in the repo.

## The honest verdict: the path is wired, the question is open

Nothing about the posture problem changed today. Run 1 still
fails the gate on every rule and model (best joint rule:
shoulder 5/5, head 4/5, both with 8 false positives), and the
fixture pass is built into the fixture. What Day 24 changes
is the distance between the capture and the answer: the
moment the real CSV and manifest land, one command validates
them, re-runs every evaluation the project has built, and
archives the verdict per cue. If the real data passes
nothing, the summary will say so in the same words it uses
for the fixture pass.

## Tests

368/368 tests passing in one combined run (91 eval + 269 + 8 =
277 model), up from 360/360 on Day 23.

8 new Day 24 tests in `experiments/model/tests/test_day24.py`:

- End to end via `--csv` on the fixture: one archive dir,
  `results.json` carries the validation report (5150 frames,
  no warnings), the Day 16 verdict (both cues pass, nothing
  fails), and the combined bests pinned per cue and variant;
  `summary.md` carries the Day 16 verdict line, the per-cue
  combined best lines, and the variant table.
- The Day 16 artifacts are untouched: SHA-256 of
  `results_day16.json` and `BENCHMARK.md` identical before
  and after the run.
- `--drop-dir` discovery: newest CSV wins, same-stem
  manifest wins over a stray `.json`, newest `.json` is the
  fallback, none when there is no `.json`; and an end-to-end
  `--drop-dir` run on a folder holding the fixture pair.
- A corrupted capture (landmark column `lm_24_y` dropped,
  the Day 16 test pattern) exits 2 and writes no archive; a
  manifest whose frame count disagrees with the CSV exits 2
  at the pre-flight, also with no archive.

## Files changed

- `experiments/model/day24.py` (new): the pipeline, CLI,
  pre-flight, combined run, archive writer, summary builder.
- `experiments/model/tests/test_day24.py` (new): 8 tests.
- `experiments/CAPTURE_CHECKLIST.md` (new): the desk
  checklist for the Day 14 capture.
- `experiments/model/runs/<date-time>/` (new): one labeled
  archive from the fixture run (`results.json`,
  `summary.md`).
- `experiments/model/WALKTHROUGH_DAY24.md` (new): this file.
- Root `README.md`: status line to Day 24, Day 24 Build
  Progress entry, "What comes next" item 1 names the Day 24
  command.

## What is still open

- The user's real capture run from localhost (Day 14 tool,
  checklist in `experiments/CAPTURE_CHECKLIST.md`), then the
  real re-benchmark through the Day 24 pipeline.
- Whether anything passes the gate on real data, per cue.
- The live deploy (Render or AWS clicks) is still his.
