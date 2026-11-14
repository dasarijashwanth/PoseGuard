# Day 12 walkthrough: RF + MLP on the gated signal

Day 11 wired the jump-gated shoulder deviation into the classifier
as three new features and showed logistic regression improving on
every metric. Day 12 asks the obvious next question: do the other
two models improve too? The random forest and the PyTorch MLP are
run on both feature sets, old (Day 8 list) and new (Day 11 list
with the three gated stats), for both cues, on the same Run 1 CSV
and the same grouped 5-fold CV by trial as Day 8.

## What was compared

- Models: logistic regression, random forest (200 trees), MLP
  (two hidden layers 32/16, ReLU, dropout 0.3, Adam, early
  stopping, CPU-only, fixed seed).
- Feature sets: old shoulder has 8 features (dynamics and
  coupling only, no shoulder level); new shoulder has 11 (the 8
  plus mean_abs_sn, max_abs_sn, std_sn from the gated signal).
  The head cue is unchanged by Day 11, so its old and new lists
  are the same 6 features by construction.
- Protocol: grouped 5-fold CV by trial, out-of-fold predictions
  for every window. A trial counts as detected when at least half
  its hold windows are predicted tilted. A false-positive event
  is 5+ consecutive tilted work windows. The gate is 4/5 trials
  with 0 false positives.
- Code: `benchmark.py` now evaluates each cue twice, once per
  feature set, through a `features` override on `evaluate_cue`
  and `evaluate_mlp_cue`. Calm-only accuracy is now reported per
  model, not just for the best model.

## The numbers

Shoulder, 385 windows, 29 tilted. Old features first, new second.

| Model | Accuracy | F1 | Trials | False positives | Work flagged | Calm-only |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LR old | 0.548 | 0.179 | 5/5 | 5 | 134/318 | 0.447 |
| LR new | 0.610 | 0.219 | 5/5 | 3 | 109/318 | 0.585 |
| RF old | 0.925 | 0.000 | 0/5 | 0 | 0/318 | 0.898 |
| RF new | 0.919 | 0.000 | 0/5 | 0 | 0/318 | 0.890 |
| MLP old | 0.764 | 0.133 | 2/5 | 1 | 54/318 | 0.663 |
| MLP new | 0.696 | 0.215 | 4/5 | 3 | 82/318 | 0.638 |

Head, 383 windows, 23 tilted. Old and new are identical, as
expected: LR 0.446 / F1 0.078 / 1/5 / 6 FP / 172 flagged, RF
0.932 / F1 0.000 / 0/5 / 0 FP / 1 flagged, MLP 0.671 / F1 0.060
/ 1/5 / 4 FP / 100 flagged.

The old-feature rows reproduce Day 8 exactly, every number, for
all three models on both cues. The comparison is apples to
apples, not a re-run that drifted.

## What the models are actually using

Logistic regression on the new shoulder set has negative
coefficients on the gated features (mean_abs_sn -1.775,
max_abs_sn -1.627, std_sn +0.797). The sign matters: a larger
gated deviation makes the model LESS likely to call a window
tilted. The model still rides the Day 5 stillness artifact,
spotting quiet windows rather than tilted shoulders. The
improvement is real on the scoreboard and hollow in mechanism.

The random forest puts 45 percent of its importance on the
gated features (max_abs_sn 0.186, mean_abs_sn 0.165, std_sn
0.100, the top three), and still detects 0/5 trials with F1
0.000. It splits on the gated features more than on any other
feature, and still never risks a tilted prediction. The gated
signal does not separate the classes enough for the trees.

The MLP is the most interesting move: recall 0.241 to 0.552,
trials 2/5 to 4/5, F1 0.133 to 0.215. At the default 0.5
threshold it is the first model to reach 4/5 trials. But the
gain is bought with false positives, 1 to 3, and work windows
flagged, 54 to 82, and its calm-only accuracy slips from 0.663
to 0.638. The extra detections are not cleaner signal. At
threshold 0.3 it reaches 5/5 trials with 10 false positives.
The Day 8 MLP was the least trigger-happy model; the gated
features made it more trigger-happy.

One more caution on the calm-only column: the random forest
calm accuracy near 0.89 looks like the best number in the
table, but it only says the forest predicts almost everything
upright, and calm windows are mostly upright. The honest
comparison is the best-F1 model: 0.447 to 0.585 on shoulder,
which is the logistic regression improvement, still riding the
stillness artifact.

## Honest verdict

No model passes the gate on either feature set. The closest any
model has come is now two ways: logistic regression on the new
features, 5/5 trials with 3 false positives, and the MLP on
the new features, 4/5 trials with 3 false positives. Both fail
on the same half of the gate, the false positives.

The gated features help logistic regression and the MLP on F1
and leave the random forest where it was, but they do not hand
any model a real tilt signal. The level half of the problem is
untouched: deliberate holds move the angle about 2.5 degrees
while ordinary work drifts 4 to 7, and no feature built from
this CSV can fix that, because the CSV has only angle columns
and no raw landmarks.

## What is still missing

The camera re-run, with longer and deeper shoulder holds and
raw landmarks recorded (hips, shoulder heights), so the two
candidates this dataset could not test get their chance, and so
the next benchmark has more than 29 tilted shoulder windows to
learn from. Until then, every model on this data is grading
itself against the same one session.

## Files

- `benchmark.py`: per-feature-set evaluation, per-model
  calm-only accuracy, Day 12 verdict, append-only Day 12
  report section (Day 8 and Day 11 sections untouched).
- `benchmark.json`: full numbers, old and new feature sets per
  cue, coefficients and importances.
- `BENCHMARK.md`: Day 12 section appended.
- `features.py`: `OLD_SHOULDER_FEATURES` and `OLD_FEATURES`
  (the Day 8 lists, kept for reproduction).
- `train.py`, `torch_mlp.py`: optional `features` override on
  `evaluate_cue` / `evaluate_mlp_cue`; per-model calm-only
  metrics in `evaluate_cue`.
- `tests/test_benchmark.py`: 8 tests, including exact Day 8
  reproduction for LR and RF and the Day 11 LR match.
- `api.py` unchanged: the API still serves the Day 5 models.
  The artifacts were not retrained, and the live API has no
  trailing history to run the gate on.
