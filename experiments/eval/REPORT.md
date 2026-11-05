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
