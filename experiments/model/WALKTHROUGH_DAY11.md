# Day 11 walkthrough: a new shoulder signal

Plain English first. Days 3 and 4 found that the shoulder cue is broken in a
specific way: the shoulder line angle barely moves when he deliberately tilts
(about 2.5 degrees), while ordinary desk work drifts 4 to 7 degrees from the
same baseline for long stretches. On top of that, the tracker sometimes jumps
20 to 100 degrees frame to frame while the head stays steady, and those jumps
create most of the false alarms. No threshold fixes a signal whose deliberate
holds sit inside its ordinary drift. Day 11 builds a different signal instead
of a fancier model on the same one.

## The data said no to two of the three candidates

The plan was shoulder-vs-hip angle and normalized shoulder height asymmetry.
Both need raw landmarks: hip positions (23/24) and shoulder y-coordinates.
The Run 1 CSV has only two angle columns, `head_tilt` and `shoulder_tilt`,
so both candidates are impossible from this data. That is a fact about the
data, not a choice. The walkthrough says it plainly because the next
camera re-run should record raw landmarks, not just angles, if any of those
signals is ever going to be tested.

What the data does support is three angle-only candidates, all built in the
new module `experiments/eval/signals.py`:

- **roll**: shoulder deviation from a trailing 30 second median baseline.
  Adapts when he settles into a new posture, instead of comparing everything
  to one baseline from calibration.
- **rel**: shoulder deviation minus head deviation (circular). Cancels
  whole-body lean: if head and shoulders move together, nothing fires.
- **gated**: the old shoulder level, but a firing only counts on frames a
  jump gate calls clean. The gate marks a frame dirty if the raw shoulder
  angle jumped more than 20 degrees in the last second. This attacks the
  tracker-noise half of the problem directly.

## Rule-level results (5 second persistence, shoulder, 5 trials)

| Signal | Best threshold | Trials | False positives |
| --- | ---: | ---: | ---: |
| old level | 1 deg | 4/5 | 19 |
| old level | 6 deg | 0/5 | 7 |
| old level | 8 deg | 0/5 | 4 |
| roll | 1 deg | 1/5 | 13 |
| roll | 6 deg | 0/5 | 6 |
| rel | 1 deg | 3/5 | 18 |
| rel | 6 deg | 0/5 | 9 |
| **gated** | **1 deg** | **4/5** | **8** |
| **gated** | **6 deg** | **0/5** | **1** |
| **gated** | **8 deg** | **0/5** | **0** |

Persistence curve for gated at 1 degree: 0.5 s gives 5/5, 2 s gives 5/5,
5 s gives 4/5. Two combos were also checked (relative plus gated at 1
degree: 3/5, 6 FP; rolling plus gated at 1 degree: 0/5, 4 FP). Neither
beats gated alone.

**Winner: gated.** It is the only candidate that moves the false-positive
count at the operating threshold: 7 down to 1 at 6 degrees, and 0 at 8
degrees. But it does not pass the gate (4/5 trials, 0 false positives).
The gate fixes the tracker-noise half and not the level half: deliberate
holds move about 2.5 degrees, ordinary work drifts 4 to 7. At any threshold
low enough to catch the holds, work still fires; at any threshold high
enough to silence work, the holds vanish.

## Classifier comparison: logistic regression, old vs new features

The winning signal is wired into the model features as three new
window stats (`mean_abs_sn`, `max_abs_sn`, `std_sn`, the gated signal's
mean, max, and spread over the window). The old shoulder level stays
forbidden, enforced by the existing test. Full numbers are in
`day11_comparison.json`.

| Logistic regression, shoulder | Accuracy | Precision | Recall | F1 | Trials | False positives | Work flagged | Calm-only acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Old features (8) | 0.548 | 0.104 | 0.655 | 0.179 | 5/5 | 5 | 134/318 | 0.447 |
| New features (11) | 0.610 | 0.129 | 0.724 | 0.219 | 5/5 | 3 | 109/318 | 0.585 |

The new features help on every column: F1 0.179 to 0.219, false positives
5 to 3, work windows flagged 134 to 109, calm-only accuracy 0.447 to
0.585. But read the coefficients before celebrating: `mean_abs_sn` -1.775
and `max_abs_sn` -1.627 are negative. The model learned that a larger
gated deviation means less likely tilted, because tilted windows are
quieter than work windows (gated mean 2.04 on tilted windows vs 5.08 on
work windows). The improvement rides the same stillness artifact Day 5
found: deliberate holds are the quietest minute of the session, and the
model spots quiet, not tilt. The gate still fails: 5/5 trials but 3 false
positives, and the 5/5 is the artifact, not detection.

## What changed, file by file

- `experiments/eval/signals.py` (new): the three candidates, the jump
  gate, `add_signals`, `evaluate_signal`, `evaluate_all`, `pick_winner`.
- `experiments/eval/analysis.py`: `sustained_events`, `longest_run`,
  `analyze_cue`, and `false_positives` take optional `dev_col` /
  `valid_col` so any signal can run the same protocol. Old callers are
  unaffected.
- `experiments/model/features.py`: `window_features` takes an optional
  `s_new` series; three gated features join `SHOULDER_FEATURES`;
  `build_windows` passes the new column when present.
- `experiments/model/train.py` and `benchmark.py`: both call
  `add_signals` and set `frames["s_dev_new"]` to the gated signal before
  building windows.
- `experiments/model/api.py`: unchanged. Without `s_new` the new
  features return 0.0, and the saved artifacts were trained on the old
  features, so the live API still serves the Day 5 models. The artifacts
  were deliberately not retrained.
- `experiments/model/day11_comparison.json` (new): the full old-vs-new
  logistic regression numbers.
- Tests: 11 new in `experiments/eval/tests/test_signals.py`, 5 new in
  `experiments/model/tests/test_features.py`, plus the three new columns
  added to the synthetic window builders in `test_train.py`,
  `test_torch_mlp.py`, and `test_api.py`. Full suite: **115/115**.

## Honest verdict

No candidate passes the gate. Gated is the best shoulder signal built so
far: it cuts tracker-noise false positives from 7 to 1 at the operating
threshold and improves the classifier on every metric, but the cue still
fails because the deliberate holds barely move the angle at all. The
classifier improvement is real in the numbers and hollow in the mechanism:
it learned that holds are quiet, not that they tilt. Nothing here changes
the Day 4 conclusion. The fix is still the re-run data, and now the re-run
has a sharper shopping list: longer, deeper shoulder holds; raw landmarks
recorded (hips and shoulder heights), not just angles, so the shoulder-vs-hip
and height-asymmetry candidates can actually be tested; and a camera setup
that does not produce 20-plus degree tracker jumps mid-session.
