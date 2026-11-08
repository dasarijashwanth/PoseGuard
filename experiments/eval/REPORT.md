# PoseGuard validation report: 2026-11-03T19:24:30.699Z

Engine: MediaPipe Tasks Vision Pose Landmarker Lite
Frames: 7820  |  calibration 323  |  head trials 657  |  shoulder trials 673  |  ordinary work 3369  |  landmark loss 1828
Baselines: head -5.622 deg, shoulder 174.173 deg

## Verdict

### head tilt: FAIL (0/5 detected at 8.0 deg, 0 false positives)
- only 0/5 holds detected (need >= 4)
- landmark-loss test never actually ran
- Trial clock offset: 9.5 s (verified against the recorded trial summary)

| trial | samples | mean dev (deg) | std | max | detected |
|---:|---:|---:|---:|---:|---|
| 1 | 67 | 0.54 | 3.73 | 7.05 | no |
| 2 | 66 | -1.70 | 1.22 | 3.25 | no |
| 3 | 67 | -4.28 | 0.83 | 5.72 | no |
| 4 | 67 | -3.26 | 2.00 | 6.60 | no |
| 5 | 6 **SHORT** | -0.58 | 0.25 | 0.89 | no |

### shoulder tilt: FAIL (0/5 detected at 6.0 deg, 7 false positives)
- only 0/5 holds detected (need >= 4)
- 7 false detections during ordinary work (need 0)
- landmark-loss test never actually ran
- Trial clock offset: 9.5 s (INFERRED: no trial summary was recorded for this cue, alignment reused from the other cue)

| trial | samples | mean dev (deg) | std | max | detected |
|---:|---:|---:|---:|---:|---|
| 1 | 67 | 2.82 | 0.60 | 4.33 | no |
| 2 | 66 | 2.29 | 0.38 | 2.81 | no |
| 3 | 67 | -1.94 | 14.00 | 58.23 | no |
| 4 | 67 | 2.67 | 0.79 | 3.89 | no |
| 5 | 8 **SHORT** | 0.31 | 1.14 | 1.87 | no |

## Landmark-loss test

Tracking lost for 18 frames (~1.8 s) out of 1828.
Head firings during loss step: 2; shoulder firings: 0.
Test ran: NO, camera never covered. Clean: no.

## Threshold sensitivity

What each threshold would have decided. Detection needs a 5 s sustained breach inside a 6 s hold.

| cue | threshold (deg) | detected | false positives | longest run (s) |
|---|---:|---:|---:|---:|
| head | 5 | 0/5 | 0 | 1.53 |
| head | 6 | 0/5 | 0 | 0.45 |
| head | 7 | 0/5 | 0 | 0.0 |
| head | 8 | 0/5 | 0 | 0.0 |
| shoulder | 4 | 0/5 | 12 | 1.89 |
| shoulder | 5 | 0/5 | 7 | 1.89 |
| shoulder | 6 | 0/5 | 7 | 1.89 |
| shoulder | 8 | 0/5 | 4 | 1.8 |

## The wraparound bug, in numbers

The file's own deviation columns used plain subtraction. Counting frames where |linear deviation| exceeds 180 degrees shows how often the old math produced phantom jumps:

- Head: 0 frames with |linear deviation| > 180 deg
- Shoulder: 1188 frames with |linear deviation| > 180 deg
- Circular deviation is always within [-180, 180] by construction, so these phantoms disappear entirely.

## Statistical validation (Day 4)

With only 5 trials per cue, a raw 0/5 is not proof the true rate is zero. The Wilson interval below is the honest range: the true detection rate is plausibly anywhere inside it. The gate needs 80%, so a cue passes this check only if the interval's top end clears 80% AND false positives are zero.

### head tilt: salvageable holds

- Detected 0/5, 95% Wilson interval 0% to 43% (gate needs 80%).
- False positives: 0 in 5.1 minutes of ordinary work, about 0 per hour (95% interval 0 to 0).
- Longest hold above 8.0 deg: 0.0 s (rule needs 5 s).
- Lowest threshold with zero false positives: 4 deg, where 0/5 holds are detected.

**Verdict:** No false positives at any tested threshold, but 0/5 holds detected, and the 95% Wilson interval tops out at 43% (the gate needs 80%). The longest hold above 5 deg (the lowest tested threshold) lasted only 1.53 s against the 5 s rule. This is not a threshold problem: at a 0.5 s persistence rule 3/5 holds would have fired, so the tilts were real but far too brief. Salvageable with a re-run of longer, deeper holds.

| threshold (deg) | detected | 95% CI | false positives | FP/hour |
|---:|---:|---:|---:|---:|
| 5 | 0/5 | 0%-43% | 0 | 0 |
| 6 | 0/5 | 0%-43% | 0 | 0 |
| 7 | 0/5 | 0%-43% | 0 | 0 |
| 8 | 0/5 | 0%-43% | 0 | 0 |

Persistence curve at 5 deg (lowest tested threshold): what each persistence rule would have decided.
| persistence rule (s) | detected |
|---:|---:|
| 0.5 | 3/5 |
| 1.0 | 1/5 |
| 1.5 | 1/5 |
| 2.0 | 0/5 |
| 3.0 | 0/5 |
| 5.0 | 0/5 |

### shoulder tilt: metric wrong

- Detected 0/5, 95% Wilson interval 0% to 43% (gate needs 80%).
- False positives: 7 in 5.1 minutes of ordinary work, about 83 per hour (95% interval 22 to 145).
- Longest hold above 6.0 deg: 1.89 s (rule needs 5 s).
- No threshold up to 30 deg kills all false positives.

**Verdict:** False positives at every tested threshold (7 at 6.0 deg, about 83 per hour of desk work), while detection stays 0/5. And even a 30 deg threshold still fires during ordinary work. No knob setting passes both criteria, so the angle itself cannot separate a real tilt from leaning at a desk. The cue needs a different signal, not a different threshold.

| threshold (deg) | detected | 95% CI | false positives | FP/hour |
|---:|---:|---:|---:|---:|
| 4 | 0/5 | 0%-43% | 12 | 142 |
| 5 | 0/5 | 0%-43% | 7 | 83 |
| 6 | 0/5 | 0%-43% | 7 | 83 |
| 8 | 0/5 | 0%-43% | 4 | 48 |

Persistence curve at 4 deg (lowest tested threshold): what each persistence rule would have decided.
| persistence rule (s) | detected |
|---:|---:|
| 0.5 | 1/5 |
| 1.0 | 0/5 |
| 1.5 | 0/5 |
| 2.0 | 0/5 |
| 3.0 | 0/5 |
| 5.0 | 0/5 |

## Machine-learning classifier (Day 5)

One scikit-learn classifier per cue, over 1-second windows of the smoothed circular deviations (10 frames each). Labels come from the protocol: windows inside a trial hold are 'tilted', windows from ordinary work and the ready/rest gaps are 'upright'. Logistic regression and a 200-tree random forest were trained per cue; the better one by out-of-fold F1 is kept. Evaluation is grouped 5-fold cross-validation, grouped by trial, so no trial's windows appear on both sides.

### head tilt: logistic_regression (23 tilted vs 360 upright windows)

- Out-of-fold accuracy 0.446, precision (tilted) 0.043, recall (tilted) 0.391, F1 (tilted) 0.078. Confusion: TP 9, FP 198, FN 14, TN 162.
- Trials detected 1/5 (rule baseline: 0/5), false-positive events 6 (rule baseline: 0). A trial counts when at least half its hold windows are predicted tilted; a false-positive event is 5 consecutive tilted work windows.
- Decision-threshold sweep (trials detected / false-positive events): 0.3: 4/5, 9 FP, 0.5: 1/5, 6 FP, 0.7: 0/5, 0 FP, 0.9: 0/5, 0 FP. No cutoff passes the gate (4/5 trials, 0 false positives).
- Calm-only accuracy 0.427 on 248 windows where the shoulder line was stable (std < 5 deg, i.e. no tracker noise). If the signal were posture, it would survive here.

### shoulder tilt: logistic_regression (29 tilted vs 356 upright windows)

- Out-of-fold accuracy 0.548, precision (tilted) 0.104, recall (tilted) 0.655, F1 (tilted) 0.179. Confusion: TP 19, FP 164, FN 10, TN 192.
- Trials detected 5/5 (rule baseline: 0/5), false-positive events 5 (rule baseline: 7). A trial counts when at least half its hold windows are predicted tilted; a false-positive event is 5 consecutive tilted work windows.
- Decision-threshold sweep (trials detected / false-positive events): 0.3: 5/5, 16 FP, 0.5: 5/5, 5 FP, 0.7: 0/5, 0 FP, 0.9: 0/5, 0 FP. No cutoff passes the gate (4/5 trials, 0 false positives).
- Calm-only accuracy 0.447 on 246 windows where the shoulder line was stable (std < 5 deg, i.e. no tracker noise). If the signal were posture, it would survive here.

**Verdict:** the classifier does not beat the rule baseline on either cue. The head model detects 1/5 trials with 6 false-positive events (the rule: 0/5, 0 FP). The shoulder model detects 5/5 but with precision 0.10 and 5 false-positive events (the rule: 0/5, 7 FP), and the calm-only check shows its separation is tracker stillness, not shoulder tilt: during the holds you sat still, during work the tracker was noisy. Worst of all, the head model learned the relationship backwards: a flat upright window scores 0.56 tilted, while a steady +6 deg tilt scores 0.33, because in Run 1 the biggest head deviations happened during noisy desk work, not during holds. One session of five brief holds is too little data to train a posture classifier, and protocol labels are not ground truth: two of the five head holds barely tilted at all. The classifier is built, tested and served behind FastAPI (see experiments/model/), but it needs the re-run data (longer, deeper holds, more trials) before it can say anything the rule cannot.

## MLflow experiment tracking (Day 6)

Day 5 left its numbers in one `results.json`. Day 6 logs every training run to MLflow (local file store, `experiments/model/mlruns/`), so runs can be compared side by side instead of overwritten. Each run records which model won per cue, the out-of-fold metrics, trials detected and false-positive events, plus the trained model as an artifact. View them with `mlflow ui --backend-store-uri experiments/model/mlruns` (http://127.0.0.1:5000).

| run | head trials | head FP | head F1 (tilted) | shoulder trials | shoulder FP | shoulder F1 (tilted) |
|---|---:|---:|---:|---:|---:|---:|
| random_forest_only | 0/5 | 0 | 0.000 | 0/5 | 0 | 0.000 |
| day5_baseline | 1/5 | 6 | 0.078 | 5/5 | 5 | 0.179 |

**What the comparison shows:** the random-forest-only run looks better at first glance (accuracy about 0.93 on both cues, against 0.45-0.55 for the baseline) but detects 0/5 trials on both cues: it learned to always answer 'upright', which is right 93% of the time because 360 of 383 windows are upright. Accuracy lies on imbalanced data; F1 (tilted) is the honest score, and it is 0.000. The baseline at least detects something (head 1/5, shoulder 5/5), at the cost of false positives. Neither run beats the rule baseline, and no model choice fixes a dataset of five brief holds. That is exactly what tracking is for: the comparison is written down and checkable, instead of a single flattering number.
