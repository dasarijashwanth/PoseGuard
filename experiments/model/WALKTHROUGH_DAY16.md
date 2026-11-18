# Day 16 walkthrough: Run 2 re-benchmark harness

The user's own camera re-run has not arrived yet, so Day 16 builds
the harness that will run on it, not the run itself. Given a Run 2
capture CSV (and optionally its JSON manifest), one command
validates the file and re-runs the Day 15 head evaluation and the
Day 11/12 shoulder evaluation on the new data, with the Run 2
protocol timings and rule baselines counted on the new data. When
the real capture lands, the same command refreshes the results and
the Day 16 section of BENCHMARK.md with its numbers.

## What changed

**The harness, in `experiments/model/day16.py`.** CLI
`python -m experiments.model.day16 --csv <capture.csv>
[--manifest <manifest.json>] [--source <label>]`. It validates
with `experiments.eval.rerun` first and exits 2 with a clear error
when the file cannot be used; a bad file never reaches the
benchmark. Then it re-runs the Day 15 head grid (60
configurations) and the LR / RF / MLP comparison on the old 6 vs
new 8 head features, and the Day 11 shoulder signal evaluation
(all three candidates plus the old level) and the Day 12 model
comparison on the old 8 vs new 11 shoulder features. Same grouped
5-fold CV as Days 8/12 throughout. It writes
`experiments/model/results_day16.json` and appends a Day 16
section to `experiments/model/BENCHMARK.md`, idempotently: a
re-run refreshes the section instead of duplicating it.

**Run 2 timings in the shared evaluation.** This was the real
work of the day. Every shared evaluation hardcoded the Run 1
trial timings: 3 s ready, 6 s hold, 3 s rest, a 12 s cycle. Run 2
holds are 10 s with 5 s ready and rest, a 20 s cycle, so the Run 1
windows would drift a full trial out of phase by the second hold.
`experiments/eval/analysis.py` (`_hold_windows`, `align_trials`,
`analyze_cue`), `experiments/eval/head.py` (`hold_dwells`,
`evaluate_head_gate`), `experiments/eval/signals.py`
(`evaluate_signal`, `evaluate_all`), and
`experiments/model/features.py` (`build_windows`, including the
trial-cycle length used for CV groups) now take the protocol
timings as keyword arguments with the Run 1 values as defaults,
so Days 3-15 are unchanged. The harness passes the Run 2 timings
from `experiments/eval/rerun_protocol`. A test pins the
difference: on the fixture, Run 1 timings label 30 tilted
windows, Run 2 timings label the correct 45.

**Rule baselines counted on the new data.** The Day 12
comparison stored the Day 3-4 Run 1 baselines, which would be
wrong numbers on a Run 2 file. Day 16 counts them fresh: the head
baseline is the session-threshold rule (15 deg, 5 s persistence,
no velocity cap) and the shoulder baseline is the old level at
the session threshold (8 deg, 5 s). The per-feature-set verdict
strings are rebuilt on those baselines too.

**Manifest check.** When a manifest is given, the harness checks
its protocol name, frame count, baselines, and that it lists 10
holds, against the CSV. A mismatch is a validation failure, same
as a bad CSV.

## The numbers

All numbers below are from the synthetic fixture
(`experiments/model/tests/fixtures/rerun-synthetic.csv`, built by
the Day 14 builder): clean holds stepping to target depth, flat
work, 5150 frames, 98.1% tracked, work sample 299.9 s, no
validator warnings, all 10 holds reaching target. The fixture
passes the gate by construction; it proves the harness can report
a pass, it says nothing about the real capture.

**Validation.** 5150 frames, 98.1% tracked, 0 warnings.

**Head rule grid (400 windows, 45 tilted).** Best configuration
is 4 deg, 5.0 s, no velocity cap: 5/5 trials, 0 false positives,
F1 1.0. Every hold dwells 9.9 s above 4 deg at a 16.0 deg peak.
The head rule baseline on this file (15 deg, 5 s) is 5/5, 0 FP.

**Head models.** All six rows are perfect on the fixture: LR,
RF, and MLP each F1 1.0, 5/5 trials, 0 false positives, on both
the old 6 and the new 8 features. The new features change
nothing here because the signal is already trivially separable.

**Shoulder signal.** Gated best is 1 deg, 5/5 trials, 0 false
positives, and the old level best is the same. The shoulder rule
baseline on this file (8 deg, 5 s) is 5/5, 0 FP.

**Shoulder models (400 windows, 45 tilted).** This is where the
fixture earns its keep: the old features still cannot separate
the shoulder cue, exactly the Day 4/5 story reproduced on Run 2
data. Old LR flags all 300 work windows as tilted (accuracy
0.140, F1 0.173, 4/5 trials, 1 false-positive event). The gated
features fix it: new LR F1 0.889 (4/5, 0 FP), new RF F1 1.0
(5/5, 0 FP), new MLP F1 0.750 (3/5, 0 FP).

| Model | Features | F1 | Trials | False positives | Work flagged |
| --- | --- | ---: | ---: | ---: | ---: |
| LR | old 8 | 0.173 | 4/5 | 1 | 300/300 |
| LR | new 11 | 0.889 | 4/5 | 0 | 0/300 |
| RF | old 8 | 0.500 | 0/5 | 0 | 0/300 |
| RF | new 11 | 1.000 | 5/5 | 0 | 0/300 |
| MLP | old 8 | 0.000 | 0/5 | 0 | 0/300 |
| MLP | new 11 | 0.750 | 3/5 | 0 | 0/300 |

## The honest verdict: the harness passes, the question is still open

On the fixture the gate passes on both cues: head by the rule
grid best and all six model rows, shoulder by the gated signal
best and the new-feature LR and RF. That is the harness working,
not the posture problem solved. The fixture holds are clean steps
to target depth with flat work, so a pass there is built in.

The real question is unchanged and still unanswered: whether
longer, deeper holds with raw landmarks separate from ordinary
work in a real session. The harness is ready for that answer the
moment the capture run lands: one command, same timings, same
gate, baselines counted on the real file. If the real data passes
nothing, the verdict will say so in the same words it uses for
the fixture pass.

## Tests

211/211 passing in one combined run (71 eval + 140 model), up
from 194/194 on Day 15. 17 new Day 16 tests, all in
`experiments/model/tests/test_day16.py`:

- Validation rejects a bad file (missing landmark column) before
  any benchmarking, and the CLI exits 2; a mismatched manifest is
  rejected the same way, a matching one accepted.
- The head grid has all 60 configurations with verified
  alignment; the best passes 5/5 with 0 false positives on the
  fixture, and every hold dwells 9.9 s at a 16.0 deg peak.
- All six head model rows are perfect on the fixture, and the
  shoulder gated signal best is 5/5 with 0 false positives.
- The shoulder old-feature LR reproduces the no-signal story
  (F1 0.173, flags all 300 work windows) while the new-feature LR
  (F1 0.889) and RF (F1 1.0) pass.
- The verdict function names the passes on the fixture and reports
  "nothing passes the gate" on fabricated failing numbers.
- Run 1 timings label 30 tilted windows on the fixture, Run 2
  timings label the correct 45.
- The committed results_day16.json schema is pinned, and the
  BENCHMARK.md Day 16 section appends idempotently without
  touching earlier sections.

## Files changed

- `experiments/model/day16.py` (new): the harness, CLI, manifest
  check, Run 2 rule baselines, verdict, BENCHMARK.md section.
- `experiments/model/tests/test_day16.py` (new): 17 tests.
- `experiments/model/tests/fixtures/rerun-synthetic.csv` (new):
  the committed fixture, built by the Day 14 builder.
- `experiments/model/results_day16.json` (new): full numbers
  from the fixture run, labeled as a synthetic fixture.
- `experiments/model/BENCHMARK.md`: Day 16 section appended.
- `experiments/eval/analysis.py`, `experiments/eval/head.py`,
  `experiments/eval/signals.py`,
  `experiments/model/features.py`: protocol timings as keyword
  arguments (Run 1 defaults, Run 2 passed by the harness).
- `experiments/model/WALKTHROUGH_DAY16.md` (new): this file.

## What is still open

- The user's capture run from localhost (Day 14 tool), then the
  real re-benchmark: `python -m experiments.model.day16 --csv
  <capture.csv> --manifest <manifest.json>`.
- Whether anything passes the gate on real data, per cue.
- The live deploy (Render or AWS clicks) is still his.
