# Day 17 walkthrough: shoulder cue rework

Day 11 left the shoulder cue with the tracker-noise half fixed:
the jump-gated signal cuts false positives from 7 to 1 at the 6
deg operating threshold. But the level half was untouched: the
gated signal still detects 0/5 trials there, because a deliberate
hold moves the shoulder line only about 2.5 deg while everyday
leaning drifts 4-7 deg. Day 17 gives the shoulder cue the Day 15
treatment, separating the three things the level conflates: how
far the shoulder moves (the level), how long it stays there
(persistence), and how fast it moves (velocity). Shoulder only;
the head cue is untouched.

## What changed

**Rule grid, in `experiments/eval/shoulder.py`.** A threshold x
persistence x velocity-cap grid: thresholds 1-8 deg, persistence
2/3/5 s, velocity cap none / 5 / 10 / 20 deg/s. A velocity cap
works like the Day 11 jump gate: a frame faster than the cap is
treated as invalid, so a tracker jump breaks a persistence run
instead of starting one. Every one of the 72 configurations is
scored the same way: trials detected out of 5, false-positive
events during ordinary work, and trial-level F1 (the 5 trials are
the positives, every work false positive counts against precision).
The module also reports per-hold dwell: how long each deliberate
hold actually stayed above each threshold, with the sample count in
the hold window so a barely recorded hold is visible as such.

Two shoulder specifics. Thresholds start at 1 deg, not 4: the holds
live around 2-3 deg, so a 4 deg floor would test nothing the holds
ever reach. Dwell is measured above 2 deg, not 5: a 5 deg dwell
threshold would be zero on every shoulder window in Run 1, hold
and work alike. The shoulder trial alignment is inferred, not
verified: Run 1 recorded no per-trial summary for the shoulder
cue, so the offset from the head alignment (9.5 s) is reused,
exactly as Days 3-4 and 11 did.

**Two new shoulder features, in `experiments/model/features.py`.**
`vel_s`, the mean speed of the shoulder line in deg/s, and
`dwell_s`, the longest continuous run above 2 deg inside a
1-second window, in seconds. Both use real elapsed time, not frame
counts, and a tracking gap breaks a dwell run instead of bridging
it. Both are dynamics, not level stats: neither is a mean or max
of the shoulder deviation, so the Day 5 ban on shoulder level
features still holds. The old 11-feature shoulder list is
unchanged, so the served v2 artifacts still match it; Day 17
evaluates the new 13-feature list without retraining the served
models. `experiments/model/day17.py` runs the grid plus the LR /
RF / MLP comparison on old vs new features with the same grouped
5-fold CV as Days 8, 12, and 15, and writes
`experiments/model/results_day17.json` plus the Day 17 section of
`experiments/model/BENCHMARK.md`.

## The numbers

**Per-hold dwell.** The holds are small, and one of them is barely
recorded at all. Trial 5 has 8 samples in its hold window, the
same story as the head cue's trial 5 (6 frames), so at most 4/5
shoulder trials are measurable in Run 1 at all. Trials 1 and 3
dwell above 2 deg for nearly the whole 6 s hold, so at the low
threshold the holds are not too brief. Trial 3 carries a 58 deg
tracker spike inside the hold itself (mean |dev| 7.05 against a
2-3 deg hold), which is why the velocity cap matters even inside
trials.

| Trial | Samples | Mean |dev| | Max |dev| | Dwell >1 deg | Dwell >2 deg | Dwell >3 deg |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 67 | 2.82 | 4.33 | 5.9 s | 5.9 s | 1.4 s |
| 2 | 66 | 2.29 | 2.81 | 5.8 s | 2.3 s | 0.0 s |
| 3 | 67 | 7.05 | 58.23 | 5.8 s | 5.8 s | 1.5 s |
| 4 | 67 | 2.67 | 3.89 | 5.9 s | 3.0 s | 2.3 s |
| 5 | 8 | 0.98 | 1.87 | 0.2 s | 0.0 s | 0.0 s |

**Rule grid, best 12 of 72 by F1.** Unlike the head grid, the
shoulder grid reaches the detection half of the gate: 16
configurations detect 4/5 or 5/5 trials, all at 1-2 deg. Every one
of them fails the false-positive half by a wide margin; the
cheapest carries 8 false positives. At the recorded 6 deg
threshold the grid reproduces Day 3/4 exactly: 0/5 trials with 7
false positives.

| Threshold | Persistence | Velocity cap | Trials | False positives | F1 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 deg | 5 s | 5 deg/s | 4/5 | 8 | 0.471 |
| 1 deg | 5 s | 10 deg/s | 4/5 | 8 | 0.471 |
| 1 deg | 5 s | 20 deg/s | 4/5 | 8 | 0.471 |
| 1 deg | 2 s | 5 deg/s | 5/5 | 17 | 0.370 |
| 1 deg | 3 s | 5 deg/s | 4/5 | 13 | 0.364 |
| 1 deg | 2 s | 10 deg/s | 5/5 | 19 | 0.345 |
| 1 deg | 3 s | 10 deg/s | 4/5 | 15 | 0.333 |
| 1 deg | 3 s | 20 deg/s | 4/5 | 15 | 0.333 |
| 1 deg | 2 s | 20 deg/s | 5/5 | 20 | 0.333 |
| 2 deg | 2 s | 5 deg/s | 4/5 | 17 | 0.308 |
| 2 deg | 2 s | 10 deg/s | 4/5 | 18 | 0.296 |
| 2 deg | 3 s | 5 deg/s | 3/5 | 13 | 0.286 |

The velocity cap does real work on the false-positive side, more
than it did for the head cue: at 1 deg / 5 s it cuts false
positives from 19 (no cap) to 8 (any cap), because it kills the
tracker jumps that start most work excursions. But 8 is still 8.
The only zero-false-positive configurations are the four at 8 deg,
and they detect 0 trials. Shortening the gate buys detection and
costs a lot on false positives: at 1 deg / 2 s the grid detects
5/5 but with 17-20 false positives in a 5-minute work sample.

**Model comparison, same grouped 5-fold CV (385 windows, 29
tilted).** The old-feature rows reproduce Day 12 exactly. The new
features make both learning models worse, the opposite of Day 15,
where they helped the head cue a little.

| Model | Features | F1 | Trials | False positives | Work flagged | Calm-only acc |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| LR | old 11 | 0.219 | 5/5 | 3 | 109/318 | 0.585 |
| LR | new 13 | 0.172 | 3/5 | 3 | 89/318 | 0.618 |
| RF | old 11 | 0.000 | 0/5 | 0 | 0/318 | 0.890 |
| RF | new 13 | 0.000 | 0/5 | 0 | 0/318 | 0.890 |
| MLP | old 11 | 0.215 | 4/5 | 3 | 82/318 | 0.638 |
| MLP | new 13 | 0.172 | 3/5 | 4 | 123/318 | 0.524 |

Three warnings inside the regression. The new LR coefficients
contradict the story the features were built to tell: vel_s is
-0.475 and dwell_s is -0.328, so the model punishes the very
speed and persistence it was meant to reward, and the gated
coefficients stay negative (mean_abs_sn -1.197, max_abs_sn
-1.194), so the stillness artifact from Days 5 and 11 is still
doing the work. The MLP's calm-only accuracy slips 0.638 to
0.524 and it flags 123 of 318 work windows, so its detections
lean harder on tracker-noise windows, not cleaner signal. The
random forest ignores the new features almost entirely
(importances 0.067 and 0.059) and still detects nothing. The
decision-threshold sweeps agree: at 0.3 the new MLP detects 5/5
but with 11 false positives, and the new sklearn sweep reaches
1/5 with 0 false positives only at 0.7, so no cutoff passes the
gate either.

## The honest verdict: nothing passes the gate

The gate is 4/5 trials with 0 false positives, and no shoulder
configuration on Run 1 comes close to it, rule or model.

- Best rule: 4/5 trials, 8 false positives, F1 0.471 (1 deg,
  5 s, 5 deg/s velocity cap).
- Best model: LR on the old features, F1 0.219, 5/5 trials, 3
  false positives. The new features make it worse, not better.
- The served models are unchanged: the v2 artifacts still serve
  the Day 13 models (shoulder LR F1 0.219, 5/5, 3 FP), because
  there is nothing better to serve.

The shoulder cue fails differently from the head cue, and the
difference matters. The head holds were too shallow and too brief
for any gate; the shoulder holds are detectable at a 1 deg
threshold (4/5, and 5/5 at 2 s persistence) but the threshold
that detects them also fires 8-20 times during ordinary work,
because work sits 4-7 deg off baseline for seconds at a time.
Velocity and dwell cannot fix that: they separate jumps from
steady movement, and the work excursions are steady, not jumps.
Two of the four measurable holds dwell above 2 deg for nearly
the whole hold, so the holds are not too brief at the threshold
that finds them; they are too small to stand out from work. No
threshold, persistence, velocity cap, or feature built from these
two angle columns can separate a signal that sits inside the
everyday movement it has to stand out from. The fix is still the
camera re-run with raw landmarks and longer, deeper holds (Day 14
tool, 10 s holds, 8 deg shoulder target), not a shorter gate or
a cleaner feature.

## Tests

239/239 passing in one combined run (80 eval + 159 model),
up from 211/211 on Day 16. 28 new Day 17 tests:

- `experiments/eval/tests/test_shoulder.py` (9): velocity is zero
  on a flat series and 10 deg/s on a steady 0-9 deg ramp in 0.9 s,
  NaN where the deviation is NaN; the slow mask keeps steady
  frames and rejects a jump; dwell measures a 0.5 s run above 2
  deg and 0 above 4 deg; the grid on the fixture has 72 rows and
  every configuration detects 0/5 with 1 false positive, best
  1 deg / 5 s / no cap; capped false positives never exceed
  uncapped ones at the same threshold and persistence; the
  fixture's flat shoulder trials dwell 0 with 60 samples each;
  trial_f1 is shared with the head grid (4 detected with 8 false
  positives scores 0.471).
- `experiments/model/tests/test_features.py` (8 new): velocity
  measures a steady tilt rate on the shoulder line and halves
  when the elapsed time doubles, dwell measures a 0.5 s run above
  2 deg and 0 when nothing exceeds 2 deg, dwell breaks on a
  tracking gap, the Day 17 shoulder list is the old 11 plus vel_s
  and dwell_s, the level ban covers the new list too, and
  build_windows carries both new features finite on Run 1.
- `experiments/model/tests/test_day17.py` (11): the grid has 72
  rows with an inferred (not verified) alignment at offset 9.5 s,
  the best configuration is pinned (1 deg, 5 s, 5 deg/s cap, 4/5,
  8 FP, F1 0.471), no configuration passes the gate, 16 reach
  4/5 with a minimum of 8 false positives, the 6 deg / 5 s
  baseline reproduces Day 3/4 (0/5, 7 FP), the velocity cap cuts
  false positives 19 to 8 at 1 deg / 5 s, trial 5 has 8 samples
  and 0 dwell while trial 3 peaks at 58.23 deg, the old shoulder
  LR reproduces Day 12 exactly (F1 0.219, 5/5, 3 FP), the new
  features make LR and MLP worse (F1 0.172, 3/5) with negative
  vel_s and dwell_s coefficients, no model passes the gate on
  either feature set, the results JSON schema holds, and the
  report writer keeps the Day 8/11/12/15/16 sections and is
  idempotent.

## Files changed

- `experiments/eval/shoulder.py`: shoulder velocity series,
  slow-mask columns, per-hold dwells, and the 72-row threshold
  x persistence x velocity-cap grid. Reuses dwell_s and trial_f1
  from head.py.
- `experiments/eval/tests/test_shoulder.py`: 9 new tests.
- `experiments/model/features.py`: vel_s and dwell_s in
  window_features (real elapsed time, tracking gaps break dwell),
  DAY17_SHOULDER_FEATURES list; the old 11-feature shoulder list
  is unchanged.
- `experiments/model/tests/test_features.py`: 8 new tests, and
  the shoulder level-ban test now covers the Day 17 list too.
- `experiments/model/day17.py`: runs the grid plus the LR / RF /
  MLP old-vs-new comparison and writes results_day17.json and
  the BENCHMARK.md Day 17 section.
- `experiments/model/tests/test_day17.py`: 10 new tests.
- `experiments/model/results_day17.json`,
  `experiments/model/BENCHMARK.md`: full numbers and the Day 17
  section.
- `README.md`: Build Progress entry for Day 17.

## What is still open

- The live URL still needs the Render clicks in `docs/DEPLOY.md`
  or the AWS clicks in `docs/DEPLOY_AWS.md`. The served models
  are unchanged, so no deploy step is needed for Day 17.
- No shoulder configuration passes the gate on Run 1. The camera
  re-run with raw landmarks (hips and shoulder heights) and
  longer, deeper holds is still the missing piece; his capture
  run comes first, then the re-benchmark.
- A real Docker build has still never run (no Docker on the
  build VM), and the 5-second gate has not been exercised
  against a live camera stream, only against recorded frames.
