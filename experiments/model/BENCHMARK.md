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
