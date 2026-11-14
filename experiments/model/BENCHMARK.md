# Day 8 benchmark: PyTorch MLP vs scikit-learn

Same Run 1 CSV, same 1-second window features, same grouped 5-fold cross-validation by trial for every model. The rule baseline from Days 3-4 is the reference in every row. The gate is 4/5 trials detected with 0 false-positive events.

The MLP is a small two-hidden-layer net (32, 16) with ReLU and dropout, Adam with weight decay, early stopping on a validation split taken from the training groups only, and inverse-frequency class weights computed from the training portion only. CPU-only, deterministic at a fixed seed.

## Head (383 windows, 23 tilted)

| Model | Accuracy | Precision (tilted) | Recall (tilted) | F1 (tilted) | Trials detected | False positives | Work windows flagged |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| logistic_regression | 0.446 | 0.043 | 0.391 | 0.078 | 1/5 | 6 | 172/318 |
| random_forest | 0.932 | 0.000 | 0.000 | 0.000 | 0/5 | 0 | 1/318 |
| mlp | 0.671 | 0.036 | 0.174 | 0.060 | 1/5 | 4 | 100/318 |

Rule baseline: 0/5 trials, 0 false positives.

MLP threshold sweep (trials / false positives): 0.3: 4/5, 2 FP, 0.5: 1/5, 4 FP, 0.7: 0/5, 0 FP, 0.9: 0/5, 0 FP

MLP calm-only accuracy: 0.685 (best sklearn: 0.427).

**Verdict:** The MLP does NOT beat the best scikit-learn model (logistic_regression): F1 0.06 vs 0.078. A neural net does not fix this cue on one session of data. It does NOT pass the gate (4/5 trials, 0 false positives): 1/5 trials with 4 false positives, against the rule baseline of 0/5 trials and 0 false positives.

## Shoulder (385 windows, 29 tilted)

| Model | Accuracy | Precision (tilted) | Recall (tilted) | F1 (tilted) | Trials detected | False positives | Work windows flagged |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| logistic_regression | 0.548 | 0.104 | 0.655 | 0.179 | 5/5 | 5 | 134/318 |
| random_forest | 0.925 | 0.000 | 0.000 | 0.000 | 0/5 | 0 | 0/318 |
| mlp | 0.764 | 0.092 | 0.241 | 0.133 | 2/5 | 1 | 54/318 |

Rule baseline: 0/5 trials, 7 false positives.

MLP threshold sweep (trials / false positives): 0.3: 5/5, 11 FP, 0.5: 2/5, 1 FP, 0.7: 0/5, 0 FP, 0.9: 0/5, 0 FP

MLP calm-only accuracy: 0.663 (best sklearn: 0.447).

**Verdict:** The MLP does NOT beat the best scikit-learn model (logistic_regression): F1 0.133 vs 0.179. A neural net does not fix this cue on one session of data. It does NOT pass the gate (4/5 trials, 0 false positives): 2/5 trials with 1 false positives, against the rule baseline of 0/5 trials and 7 false positives.

## What this means

A neural net does not rescue a dataset of one session with 23 tilted windows. Whatever the MLP wins on F1, check the trial and false-positive columns before believing it: the gate, not accuracy, is the honest score. The fix is still the re-run data (longer, deeper holds, more trials), not a fancier model.

Two real differences are worth keeping, even though neither passes the gate. First, the MLP is far less trigger-happy than logistic regression: it flags 100 of 318 work windows on head (vs 172) and 54 on shoulder (vs 134), with 4 and 1 false-positive events (vs 6 and 5). Second, its signal survives the calm-only check: on windows where the shoulder line was steady, MLP accuracy is 0.685 on head and 0.663 on shoulder, against 0.427 and 0.447 for logistic regression, whose accuracy collapsed once tracker noise was removed. The MLP learned something less noise-driven. And at threshold 0.3 the head MLP reaches 4/5 trials with 2 false positives, the closest any model has come to the gate. Close is still not a pass.

## Day 11: new shoulder signal, logistic regression old vs new

Rule-level comparison first (5 second persistence, shoulder, 5 trials).
The Run 1 CSV has only angle columns, no raw landmarks, so the
shoulder-vs-hip and shoulder-height candidates could not be built.
Three angle-only candidates were tested in `experiments/eval/signals.py`.

| Signal | Threshold | Trials detected | False positives |
| --- | ---: | ---: | ---: |
| old level | 1 deg | 4/5 | 19 |
| old level | 6 deg | 0/5 | 7 |
| old level | 8 deg | 0/5 | 4 |
| roll (trailing 30 s median) | 1 deg | 1/5 | 13 |
| roll | 6 deg | 0/5 | 6 |
| rel (shoulder minus head) | 1 deg | 3/5 | 18 |
| rel | 6 deg | 0/5 | 9 |
| gated (jump gate, 20 deg / 1 s) | 1 deg | 4/5 | 8 |
| gated | 6 deg | 0/5 | 1 |
| gated | 8 deg | 0/5 | 0 |

Gated persistence at 1 deg: 0.5 s: 5/5, 2 s: 5/5, 5 s: 4/5.
Combos: rel plus gated at 1 deg: 3/5, 6 FP; roll plus gated at 1 deg:
0/5, 4 FP. Winner: gated. No candidate passes the gate (4/5 trials,
0 false positives).

Logistic regression, shoulder, old features (8) vs new features (11,
old plus three gated-signal stats). Full numbers in
`day11_comparison.json`.

| Model | Accuracy | Precision (tilted) | Recall (tilted) | F1 (tilted) | Trials detected | False positives | Work windows flagged |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| logistic_regression (old) | 0.548 | 0.104 | 0.655 | 0.179 | 5/5 | 5 | 134/318 |
| logistic_regression (new) | 0.610 | 0.129 | 0.724 | 0.219 | 5/5 | 3 | 109/318 |

Threshold sweep (trials / false positives), new: 0.3: 5/5, 14 FP, 0.5:
5/5, 3 FP, 0.7: 0/5, 0 FP, 0.9: 0/5, 0 FP.

Calm-only accuracy: old 0.447, new 0.585.

**Verdict:** The new features improve every column, but the gate still
fails (5/5 trials, 3 false positives), and the gated coefficients are
negative (`mean_abs_sn` -1.775, `max_abs_sn` -1.627): tilted windows
are quieter than work windows (gated mean 2.04 vs 5.08), so the model
still rides the Day 5 stillness artifact. A better signal on the same
one session does not pass the gate. The fix is still the re-run data,
now with raw landmarks recorded.

## Day 12: RF + MLP on the gated signal

Day 11 wired the jump-gated shoulder deviation into the classifier as three new features (mean_abs_sn, max_abs_sn, std_sn) and showed logistic regression improving on every metric. Day 12 asks whether the other two models improve too, on the same grouped 5-fold CV by trial and the same out-of-fold protocol as Day 8. Each cue is run twice: old features (Day 8 list) and new features (Day 11 list). The head cue is unchanged by Day 11, so its old and new rows are the same by construction. The gate is still 4/5 trials with 0 false positives.

### Head (383 windows, 23 tilted)

Old features (6):

| Model | Accuracy | Precision (tilted) | Recall (tilted) | F1 (tilted) | Trials | False positives | Work windows flagged | Calm-only accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| logistic_regression | 0.446 | 0.043 | 0.391 | 0.078 | 1/5 | 6 | 172/318 | 0.427 |
| random_forest | 0.932 | 0.000 | 0.000 | 0.000 | 0/5 | 0 | 1/318 | 0.895 |
| mlp | 0.671 | 0.036 | 0.174 | 0.060 | 1/5 | 4 | 100/318 | 0.685 |

New features (6):

| Model | Accuracy | Precision (tilted) | Recall (tilted) | F1 (tilted) | Trials | False positives | Work windows flagged | Calm-only accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| logistic_regression | 0.446 | 0.043 | 0.391 | 0.078 | 1/5 | 6 | 172/318 | 0.427 |
| random_forest | 0.932 | 0.000 | 0.000 | 0.000 | 0/5 | 0 | 1/318 | 0.895 |
| mlp | 0.671 | 0.036 | 0.174 | 0.060 | 1/5 | 4 | 100/318 | 0.685 |

Rule baseline: 0/5 trials, 0 false positives.

MLP threshold sweep, new features (trials / false positives): 0.3: 4/5, 2 FP, 0.5: 1/5, 4 FP, 0.7: 0/5, 0 FP, 0.9: 0/5, 0 FP

**Verdict:** logistic_regression: F1 unchanged at 0.078, false positives 6 to 6. random_forest: F1 unchanged at 0.0, false positives 0 to 0. mlp: F1 unchanged at 0.06, false positives 4 to 4. Nothing passes the gate (4/5 trials, 0 false positives) on either feature set. Best-F1 model calm-only accuracy 0.427 (logistic_regression, old) to 0.427 (logistic_regression, new). The random forest calm figure is higher only because it predicts almost everything upright, so it is not the honest comparison.

### Shoulder (385 windows, 29 tilted)

Old features (8):

| Model | Accuracy | Precision (tilted) | Recall (tilted) | F1 (tilted) | Trials | False positives | Work windows flagged | Calm-only accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| logistic_regression | 0.548 | 0.104 | 0.655 | 0.179 | 5/5 | 5 | 134/318 | 0.447 |
| random_forest | 0.925 | 0.000 | 0.000 | 0.000 | 0/5 | 0 | 0/318 | 0.898 |
| mlp | 0.764 | 0.092 | 0.241 | 0.133 | 2/5 | 1 | 54/318 | 0.663 |

New features (11):

| Model | Accuracy | Precision (tilted) | Recall (tilted) | F1 (tilted) | Trials | False positives | Work windows flagged | Calm-only accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| logistic_regression | 0.610 | 0.129 | 0.724 | 0.219 | 5/5 | 3 | 109/318 | 0.585 |
| random_forest | 0.919 | 0.000 | 0.000 | 0.000 | 0/5 | 0 | 0/318 | 0.890 |
| mlp | 0.696 | 0.133 | 0.552 | 0.215 | 4/5 | 3 | 82/318 | 0.638 |

Rule baseline: 0/5 trials, 7 false positives.

MLP threshold sweep, new features (trials / false positives): 0.3: 5/5, 10 FP, 0.5: 4/5, 3 FP, 0.7: 0/5, 0 FP, 0.9: 0/5, 0 FP

Logistic regression coefficients on the gated features (new set): {'mean_abs_sn': -1.775, 'max_abs_sn': -1.627, 'std_sn': 0.797}.

Random forest importances on the gated features (new set): {'mean_abs_sn': 0.165, 'max_abs_sn': 0.186, 'std_sn': 0.1}.

**Verdict:** logistic_regression: F1 0.179 to 0.219, false positives 5 to 3, work flagged 134 to 109. random_forest: F1 unchanged at 0.0, false positives 0 to 0. mlp: F1 0.133 to 0.215, false positives 1 to 3, work flagged 54 to 82. Nothing passes the gate (4/5 trials, 0 false positives) on either feature set. Best-F1 model calm-only accuracy 0.447 (logistic_regression, old) to 0.585 (logistic_regression, new). The random forest calm figure is higher only because it predicts almost everything upright, so it is not the honest comparison.

## What the gated features mean

The gated features help logistic regression and the MLP on F1, and leave the random forest where it was, but they do not hand any model a real tilt signal. The MLP gain comes with more false positives (1 to 3) and its calm-only accuracy slips a little, so the extra detections are not cleaner signal. Check the trial and false-positive columns before believing any F1 move: the gate, not accuracy, is the honest score. The fix is still the re-run data (longer, deeper holds, more trials, raw landmarks), not a fancier model or a cleaner feature.
