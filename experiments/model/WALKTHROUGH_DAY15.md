# Day 15 walkthrough: head cue rework

Day 13 left the head cue as the worst half of the served model:
logistic regression 1/5 trials, 6 false positives, F1 0.078. Day 4
had said the head tilts were real but too brief for the 5-second
persistence gate. Day 15 tests that claim properly, by separating
the three things it conflates: how far the head moves (the level),
how long it stays there (persistence), and how fast it moves
(velocity). Head only; the shoulder cue is untouched.

## What changed

**Rule grid, in `experiments/eval/head.py`.** A threshold x
persistence x velocity-cap grid: thresholds 4-8 deg, persistence
2/3/5 s, velocity cap none / 5 / 10 / 20 deg/s. A velocity cap
works like the Day 11 jump gate: a frame faster than the cap is
treated as invalid, so a tracker jump breaks a persistence run
instead of starting one. Every one of the 60 configurations is
scored the same way: trials detected out of 5, false-positive
events during ordinary work, and trial-level F1 (the 5 trials are
the positives, every work false positive counts against precision).
The module also reports per-hold dwell: how long each deliberate
hold actually stayed above each threshold, with the sample count in
the hold window so a barely recorded hold is visible as such.

**Two new head features, in `experiments/model/features.py`.**
`vel_h`, the mean speed of the head line in deg/s, and `dwell_h`,
the longest continuous run above 5 deg inside a 1-second window,
in seconds. Both use real elapsed time, not frame counts, and a
tracking gap breaks a dwell run instead of bridging it. The old
6-feature head list is unchanged, so the served v2 artifacts still
match it; Day 15 evaluates the new 8-feature list without
retraining the served models. `experiments/model/day15.py` runs the
grid plus the LR / RF / MLP comparison on old vs new features with
the same grouped 5-fold CV as Days 8 and 12, and writes
`experiments/model/results_day15.json` plus the Day 15 section of
`experiments/model/BENCHMARK.md`.

## The numbers

**Per-hold dwell.** The holds are shallow as well as brief. The
deepest hold peaks at 7.1 deg (trial 1) and the longest run above
4 deg inside any hold is 2.2 s (trials 3 and 4). Trial 2 never gets
above 3.3 deg. Trial 5 is barely recorded: 6 frames in its hold
window, all near zero, so at most 4/5 trials are measurable in
Run 1 at all. Meanwhile ordinary work contains head excursions of
10-18 deg sustained for 2-4 s, plus a 104 deg tracker spike.

| Trial | Samples | Mean |dev| | Max |dev| | Dwell >4 deg | Dwell >5 deg |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 67 | 3.22 | 7.05 | 0.9 s | 0.5 s |
| 2 | 66 | 1.84 | 3.25 | 0.0 s | 0.0 s |
| 3 | 67 | 4.28 | 5.72 | 2.2 s | 1.5 s |
| 4 | 67 | 3.34 | 6.60 | 2.2 s | 0.9 s |
| 5 | 6 | 0.58 | 0.89 | 0.0 s | 0.0 s |

**Rule grid, best 4 of 60 by F1.** Shortening the gate buys
almost nothing on detection and costs a lot on false positives.
At the recorded 8 deg threshold, no persistence detects anything.
The best configuration is the lowest threshold with the shortest
gate and a middle velocity cap, and it still only finds 2 of 5
trials with 3 false positives.

| Threshold | Persistence | Velocity cap | Trials | False positives | F1 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4 deg | 2 s | 10 deg/s | 2/5 | 3 | 0.400 |
| 4 deg | 2 s | 20 deg/s | 2/5 | 5 | 0.333 |
| 4 deg | 2 s | 5 deg/s | 1/5 | 2 | 0.250 |
| 4 deg | 2 s | none | 2/5 | 9 | 0.250 |

The velocity cap does real work on the false-positive side: at
4 deg / 2 s it cuts false positives from 9 (no cap) to 3 (10
deg/s cap) to 2 (5 deg/s cap), because it kills the tracker
jumps, including the 104 deg spike. But it cannot create
detections the holds never earned: no configuration at any
threshold above 4 deg detects even one trial, and no configuration
at 3 or 5 s persistence detects more than 0.

**Model comparison, same grouped 5-fold CV (383 windows, 23
tilted).** The old-feature rows reproduce Day 8/13 exactly.
Velocity and dwell move LR and the MLP a little on F1 and not at
all on the gate.

| Model | Features | F1 | Trials | False positives | Work flagged | Calm-only acc |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| LR | old 6 | 0.078 | 1/5 | 6 | 172/318 | 0.427 |
| LR | new 8 | 0.097 | 2/5 | 4 | 167/318 | 0.448 |
| RF | old 6 | 0.000 | 0/5 | 0 | 1/318 | 0.895 |
| RF | new 8 | 0.000 | 0/5 | 0 | 0/318 | 0.899 |
| MLP | old 6 | 0.060 | 1/5 | 4 | 100/318 | 0.685 |
| MLP | new 8 | 0.105 | 2/5 | 3 | 99/318 | 0.605 |

Two warnings inside the small gains. The MLP's calm-only accuracy
slips 0.685 to 0.605 with the new features, so part of its F1 gain
leans on tracker-noise windows, not cleaner signal. And the new
LR coefficients contradict each other: fraction-above-5 is +1.468
while dwell is -0.962, so the model rewards being above 5 deg
often but punishes staying there, which is fitting noise, not a
tilt. The random forest ignores dwell almost entirely (importance
0.014) and still detects nothing. The decision-threshold sweeps
agree: at 0.3 the new LR detects 4/5 but with 10 false positives,
and the new MLP 4/5 with 4, so no cutoff passes the gate either.

## The honest verdict: nothing passes the gate

The gate is 4/5 trials with 0 false positives, and no head
configuration on Run 1 comes close to it, rule or model.

- Best rule: 2/5 trials, 3 false positives, F1 0.400 (4 deg, 2 s,
  10 deg/s velocity cap).
- Best model: MLP on the new features, F1 0.105, 2/5 trials, 3
  false positives.
- The served models are unchanged: the v2 artifacts still serve
  the Day 13 models (head LR F1 0.078, 1/5, 6 FP), because there
  is nothing better to serve.

Day 4's story needs correcting, not just refining. The head tilts
are not "real but too brief": they are shallow (deepest hold
7.1 deg, against an 8 deg session threshold no hold ever reaches)
and brief (longest run above 4 deg is 2.2 s) and smaller than
ordinary work, where the head moves 10-18 deg for 2-4 s at a time.
Dwell separates in the wrong direction: work dwells longer than
the holds do. No threshold, persistence, velocity cap, or feature
built from these two angle columns can separate a signal that is
smaller than the everyday movement it has to stand out from. The
fix is still the camera re-run with raw landmarks and longer,
deeper holds (Day 14 tool, 10 s holds, 15 deg head target), not a
shorter gate or a cleaner feature.

## Tests

194/194 passing in one combined run (71 eval + 123 model),
up from 166/166 on Day 14. 28 new Day 15 tests:

- `experiments/eval/tests/test_head.py` (11): velocity is zero
  on a flat series and 10 deg/s on a steady 0-9 deg ramp in 0.9 s,
  NaN where the deviation is NaN; the slow mask keeps steady
  frames and rejects a jump; dwell measures a 0.5 s run and 0
  above 7 deg; trial F1 is 0.25 for 2 detected with 9 false
  positives, 0 when nothing is detected, 1 for a clean sweep;
  the grid on the fixture has 60 rows and every configuration
  detects the one clean hold with 0 false positives; capped false
  positives never exceed uncapped ones at the same threshold and
  persistence; the fixture's clean hold dwells above 5 deg for
  more than 5 s and the flat trials dwell 0.
- `experiments/model/tests/test_features.py` (8 new): velocity
  measures a steady tilt rate and halves when the elapsed time
  doubles, dwell measures a 0.5 s run and 0 when nothing exceeds
  5 deg, dwell breaks on a tracking gap, the Day 15 head list is
  the old 6 plus vel_h and dwell_h, and build_windows carries
  both new features finite on Run 1.
- `experiments/model/tests/test_day15.py` (9): the grid has 60
  rows with a verified alignment, the best configuration is
  pinned (4 deg, 2 s, 10 deg/s cap, 2/5, 3 FP, F1 0.400), no
  configuration passes the gate, trial 5 has 6 samples and 0
  dwell, the old head LR reproduces Day 13 exactly (F1 0.078,
  1/5, 6 FP), the new LR is F1 0.097 with 2/5 and 4 FP and the
  new MLP is F1 0.105 with 2/5 and 3 FP, no model passes the gate
  on either feature set, the results JSON schema holds, and the
  report writer keeps the Day 8/11/12 sections and is idempotent.

## Files changed

- `experiments/eval/head.py`: velocity series, slow-mask columns,
  dwell helper, trial-level F1, per-hold dwells, and the 60-row
  threshold x persistence x velocity-cap grid.
- `experiments/eval/tests/test_head.py`: 11 new tests.
- `experiments/model/features.py`: vel_h and dwell_h in
  window_features (real elapsed time, tracking gaps break dwell),
  DAY15_HEAD_FEATURES list; the old 6-feature head list is
  unchanged.
- `experiments/model/tests/test_features.py`: 8 new tests.
- `experiments/model/day15.py`: runs the grid plus the LR / RF /
  MLP old-vs-new comparison and writes results_day15.json and the
  BENCHMARK.md Day 15 section.
- `experiments/model/tests/test_day15.py`: 9 new tests.
- `experiments/model/results_day15.json`,
  `experiments/model/BENCHMARK.md`: full numbers and the Day 15
  section.
- `README.md`: Build Progress entry for Day 15.

## What is still open

- The live URL still needs the Render clicks in `docs/DEPLOY.md`
  or the AWS clicks in `docs/DEPLOY_AWS.md`. The served models
  are unchanged, so no deploy step is needed for Day 15.
- No head configuration passes the gate on Run 1. The camera
  re-run with raw landmarks (hips and shoulder heights) and
  longer, deeper holds is still the missing piece; his capture
  run comes first, then the re-benchmark.
- A real Docker build has still never run (no Docker on the
  build VM), and the 5-second gate has not been exercised
  against a live camera stream, only against recorded frames.
