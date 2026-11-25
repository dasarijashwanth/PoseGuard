# PoseGuard re-benchmark run: synthetic fixture (Day 14 builder)

- Capture: experiments/model/tests/fixtures/rerun-synthetic.csv
- Manifest: none

## Validation

5150 frames, 98.1% tracked, work sample 299.9 s. Holds: head 5/5 holds, 5/5 reached their depth target; shoulder 5/5 holds, 5/5 reached their depth target.

Warnings: none.

## Day 16 verdict

Head passes the gate: rule grid best (4 deg, 5.0 s, velocity cap none), logistic_regression (old features), random_forest (old features), mlp (old features), logistic_regression (new features), random_forest (new features), mlp (new features). Shoulder passes the gate: gated signal best (1 deg), logistic_regression (new features), random_forest (new features).

## Combined gate (Day 23)

Best variant per cue by F1 (ties toward more trials, then fewer false positives; the gate is 4/5 trials with 0 false positives):

- head: sum at threshold 0.5, 5/5 trials, 0 false positives, F1 1.000: passes the gate.
- shoulder: sum at threshold 0.5, 5/5 trials, 0 false positives, F1 1.000: passes the gate.

All variants:

| Cue | Variant | Threshold | Trials | False positives | F1 | Pass |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| head | sum | 0.5 | 5/5 | 0 | 1.000 | yes |
| head | iso | 2.0 | 5/5 | 0 | 1.000 | yes |
| head | and_control | 4.0 | 0/5 | 0 | 0.000 | no |
| shoulder | sum | 0.5 | 5/5 | 0 | 1.000 | yes |
| shoulder | iso | 0.5 | 5/5 | 0 | 1.000 | yes |
| shoulder | and_control | 1.0 | 0/5 | 0 | 0.000 | no |
