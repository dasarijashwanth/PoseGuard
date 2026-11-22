# PoseGuard public demo (Day 19)

A Streamlit app for anyone who opens the repo: try the live webcam
check yourself, then read the real Run 1 validation session, gate
verdicts per trial, and the model scoreboard, with the honest verdict
up front.

## Run locally

```bash
pip install -r demo/requirements.txt
streamlit run demo/app.py
```

## Deploy (free)

Streamlit Community Cloud: new app, repo `dasarijashwanth/PoseGuard`,
main file `demo/app.py`. No secrets or environment variables needed;
the app reads only the bundled sample CSV and the bundled pose model
(`demo/pose_landmarker_lite.task`, MediaPipe PoseLandmarker lite).
`demo/requirements.txt` sits next to the app file, and `packages.txt`
at the repo root installs the system libraries MediaPipe needs on
Cloud (libgl1, libglib2.0-0, libegl1, libgles2, Day 20): mediapipe
imports cv2 at module load, and the cv2 that wins links against
libGL, which the Cloud image does not ship.

## What it shows

- **Live webcam check**: real-time head and shoulder angles from your
  camera, with the gate firing on holds. Live video runs through
  streamlit-webrtc with MediaPipe PoseLandmarker (VIDEO mode) on the
  server; if the video stream cannot start, the snapshot check
  (st.camera_input, IMAGE mode) does the same job one photo at a
  time. Calibrate upright first, then hold a tilt or a dropped
  shoulder for 5 seconds. Two gate presets: the operating thresholds
  (head 8 deg / shoulder 6 deg, 5 s) and the best rule-grid configs
  (head 4 deg / 2 s, shoulder 1 deg / 5 s), so visitors can see why
  the best configs fail. No video leaves the browser: landmarks are
  computed from the frames on the spot, only angles are kept.
  `.streamlit/config.toml` turns websocket compression off, which the
  video stream needs on Community Cloud.
- **Purpose and how to use it**: plain words on what PoseGuard is for
  and how to try the check, at the top of the page.
- **Feedback**: a section linking to GitHub issues
  (github.com/dasarijashwanth/PoseGuard/issues) for live-check reports.
- **Session chart**: head and shoulder deviations over time, 1 Hz sample
  of the full Run 1 session (`demo/data/run1_sample.csv`, 727 rows,
  downsampled from 7,820 frames). Head is the circular deviation on
  tracking-valid frames; shoulder is the Day 11 jump-gated deviation,
  NaN where the tracker was noisy or lost. The chart axis is clipped at
  +/-25 deg with the 447 s tracker meltdown disclosed in the caption,
  so one meltdown does not hide the 2-7 deg holds.
- **Gate verdicts**: best rule-grid configuration per cue, with the
  per-trial hold table. The gate is 4/5 trials with 0 false positives.
- **Model scoreboard**: LR / RF / MLP vs the rule baseline, latest
  feature set per cue (head Day 15, shoulder Day 17).
- **The honest verdict**: nothing passes the gate. Best rule 4/5 trials
  with 8 false positives; best model F1 0.172. The holds are shallow as
  well as brief.

## Numbers and their sources

Every number in the demo is pinned to a results file and to tests in
`experiments/model/tests/test_day18.py`:

| Demo figure | Source |
| --- | --- |
| Shoulder best rule 1 deg / 5 s / 5 deg/s, 4/5, 8 FP, F1 0.471 | `experiments/model/results_day17.json` |
| Head best rule 4 deg / 2 s / 10 deg/s, 2/5, 3 FP, F1 0.400 | `experiments/model/results_day15.json` |
| Shoulder models (new features) LR / MLP F1 0.172, RF 0/5 | `experiments/model/results_day17.json` |
| Head models (new features) LR F1 0.097, MLP F1 0.105, RF 0/5 | `experiments/model/results_day15.json` |
| Trial dwell rows | `hold_dwells` in the same two files |

## Layout

- `demo/app.py`: the Streamlit UI only. It imports `demo_data`, never
  the evaluation pipeline, so the app stays light. It probes the
  live-check dependencies through `demo/env_check.py` and shows the
  exact import error when the check cannot load.
- `demo/env_check.py`: pure dependency probing and diagnostics for the
  live check (Day 20). No Streamlit, MediaPipe, or OpenCV import at
  module level.
- `demo/demo_data.py`: pure data + verdict logic, fully tested without
  Streamlit installed.
- `demo/data/run1_sample.csv`: trimmed 1 Hz sample of Run 1. The full
  session CSV is not in the repo.
