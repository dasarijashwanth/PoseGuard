# WALKTHROUGH_DAY22.md

## Day 22 (Nov 23): the full project in one place

Day 21 landed in the morning, and he asked for the live link to show
everything, not just the showcase: all result values, the whole
process, and all the work from the initial days, in one place, and to
do it today. Day 22 is that section in the live demo. One day-slice
per day is the standing rule, and this is the exception he ordered
explicitly ("do it today"), so it is recorded as such here and in the
root README.

## What was built

- `demo/demo_data.py`: three new data blocks, all pure data with no
  Streamlit import, in the Day 18 style.
  - `PIPELINE`: the 8 stages in order (Collect, Evaluate, Validate,
    Learn, Serve, Track, Package, Demo), each with the day it first
    landed and the tool that runs it.
  - `JOURNEY`: one row per day, Days 1 to 22, with the date, what
    was built, the one-line headline result, and the cumulative test
    total after that day (None for Day 7, a docs-only day).
  - `BENCHMARK_TABLES`: headline rows for the Day 8 MLP benchmark
    (head and shoulder), the Day 11 shoulder signal candidates, the
    Day 12 shoulder new-feature comparison, the Day 13 served models,
    and the Day 16 synthetic fixture summary. Each table carries its
    source file.
  - Helpers `journey_frame()`, `pipeline_frame()`, and
    `benchmark_frame(table)` (unknown table raises ValueError).
- `demo/app.py`: `_full_project()` renders the new section after
  "What would fix it" and before Feedback: the pipeline as a numbered
  list, the 22-row journey table, each benchmark table with its
  source, the per-day verdict lines, and the Day 16 fixture labeled
  synthetic. The existing showcase and webcam check are untouched.
- `demo/README.md`: the new section and its source table.
- Root README: the status line and the test-count line now say Day 22,
  and a Day 22 entry leads the Build Progress section.

## Numbers pinned and their sources

Every displayed number is asserted against its source file in
`experiments/model/tests/test_day22.py` (17 tests):

| Demo figure | Source |
| --- | --- |
| Journey dates and test counts per day | Root README Build Progress headings and sections |
| Day 8 head/shoulder model rows | `experiments/model/benchmark.json` old feature sets |
| Day 11 signal rows (old level and gated, 1/6/8 deg) | `experiments/model/BENCHMARK.md` signal table |
| Day 12 shoulder new-feature rows | `experiments/model/benchmark.json` shoulder new feature set |
| Day 13 served LR rows (6 head features, 11 shoulder features) | `experiments/model/results_day13.json` |
| Day 16 fixture (5,150 frames, 98.1% tracked, 299.9 s work, 10/10 holds) | `experiments/model/results_day16.json` |

The honest framing is unchanged: no real-data row in any table passes
the gate (4/5 trials, 0 false positives). The best rule is still the
shoulder 1 deg / 5 s / 5 cap at 4/5 trials with 8 false positives
(F1 0.471), the best head rule 2/5 with 3 false positives (F1 0.400),
and the best models F1 0.172. The Day 16 fixture is the one exception
and stays labeled synthetic everywhere it appears, because it passes
by construction on clean holds with flat work.

## What was deliberately left out

- The full 60-config head grid and 72-config shoulder grid: the demo
  already shows the best configuration per cue, and the full tables
  live in BENCHMARK.md. The journey row for each rework day names
  the grid size instead.
- Per-trial dwell tables beyond the existing showcase: the Day 15 and
  Day 17 dwell tables are already in the demo through the trial
  tables; duplicating them in the new section adds rows, not
  information.
- The Day 16 full model tables (all 1.000 on the fixture): the
  fixture summary and verdict carry the point, and the full tables
  are in BENCHMARK.md. Repeating perfect numbers would read as a
  claim about the real capture.
- No model, threshold, or gate logic changed. The live video stream
  still needs a real browser and camera to verify end to end.

## Verification

342/342 tests passing in one combined run (80 eval + 262 model,
including 18 new Day 22 tests). The new tests pin every journey date
and test count to the root README, every benchmark row to its JSON or
to the BENCHMARK.md table text, and assert no real-data row passes
the gate and the Day 16 fixture stays labeled synthetic. `app.py`
and `demo_data.py` compile cleanly.
