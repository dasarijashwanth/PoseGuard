# WALKTHROUGH_DAY19.md

## Day 19 (Nov 21): the demo becomes a real app

Day 18 shipped a showcase: charts and tables you scroll past. The user
looked at it and said it was static and did not do anything, and he was
right. Day 19 makes the demo a real app in the three parts he picked:
(a) a live webcam check, (b) a purpose and how-to-use section, and
(c) a feedback section linking to GitHub issues.

## What was built

### (a) Live webcam check (`demo/app.py`, `demo/webcam.py`)

Two ways to try it, same gate underneath:

- **Live video** (first tab): streamlit-webrtc streams frames to the
  server, where MediaPipe PoseLandmarker (VIDEO mode) finds the pose
  on each frame. The video is annotated live (ear and shoulder lines
  drawn, angle readouts and gate status burned into the frame) and the
  page shows live head/shoulder deviation metrics with the gate run
  time. The visitor presses Calibrate while sitting upright, then
  holds a tilt or a dropped shoulder and the gate fires when the hold
  lasts.
- **Snapshot check** (second tab): st.camera_input takes photos. The
  first photo calibrates the upright pose; each later photo is one
  sample, and repeated photos while holding a pose build the same
  gate the video uses. This is the fallback when the video stream
  cannot start in a visitor's browser.

Both paths use the same pure logic in `demo/webcam.py`, which has no
Streamlit, MediaPipe, or OpenCV import, so the tests cover the exact
code the app runs:

- Angle math copied from the Day 14 capture tool: head raw angle is
  the ear-to-ear line (landmarks 7, 8) minus the shoulder line
  (landmarks 11, 12), circular; shoulder raw angle is the shoulder
  line itself, wrapped to +/-180 deg. Deviations are circular
  differences from an upright calibration baseline.
- `LiveGate`: fires when the absolute deviation stays above a
  threshold for a persistence time. A frame whose speed exceeds the
  velocity cap breaks the run (a jump is tracker noise, not a hold),
  and a lost frame breaks it too. Once fired it latches until reset,
  so the verdict stays on screen while the visitor sits back up.
- Two gate presets. Operating thresholds (head 8 deg / shoulder 6 deg,
  hold 5 s), the same "hold it" instruction the Day 14 tool gives.
  And the best rule-grid configs (head 4 deg / 2 s, shoulder 1 deg /
  5 s), offered on purpose so visitors can see why those configs
  fail: the shoulder best rule fires on everyday leaning.

### (b) Purpose and how-to-use (`demo/demo_data.py`, rendered in `demo/app.py`)

A "What PoseGuard is for" section at the top of the page (on-device
vision AI, webcam angles, no video ever leaves the visitor's machine,
and the honest status: nothing passes the gate yet) and a "How to use
this demo" section with four steps: live check, snapshot check, the
Run 1 session below, and the holds the project is chasing. The copy
lives as constants in `demo_data.py` so tests pin it.

### (c) Feedback (`demo/demo_data.py`, rendered in `demo/app.py`)

A feedback section at the bottom of the page with a button linking to
https://github.com/dasarijashwanth/PoseGuard/issues, asking visitors
to report what the live check did, which check they used, and what
pose they held.

### Supporting changes

- `demo/pose_landmarker_lite.task` (5.8 MB): the MediaPipe
  PoseLandmarker lite model, bundled in the repo so the deployed app
  does not need a model download at runtime.
- `demo/requirements.txt`: mediapipe>=1.0, opencv-python-headless,
  streamlit-webrtc.
- `.streamlit/config.toml`: websocket compression off, which the
  streamlit-webrtc video stream needs on Streamlit Community Cloud.
- `demo/README.md`: documents the live check, the model file, and the
  config file.

## Two things found while building

1. **The legacy MediaPipe Solutions API is gone.** mediapipe 1.x
   removed `mp.solutions.pose`, so the app uses the supported Tasks
   Vision API (`PoseLandmarker`, IMAGE mode for snapshots, VIDEO mode
   for the stream). The old `mp.solutions` code from any tutorial no
   longer runs.
2. **MediaPipe Tasks needs system GL libraries.** The landmarker
   failed to load here until `libegl1` and `libgles2` were installed
   (missing `libEGL.so.1`, then `libGLESv2.so.2`). That is a property
   of this VM, not of the deployed app: Streamlit Community Cloud
   images already carry those libraries, and the app reports the live
   check as unavailable instead of crashing if MediaPipe cannot load.

## Verification

- **286/286 tests passing in one combined run (80 eval + 206 model,
  including 28 new Day 19 tests)** in `experiments/model/tests/test_day19.py`:
  circular angle helpers, landmark-to-angle math pinned to the Day 14
  capture tool (level pose reads 0, a 10 deg ear tilt reads 10, a 6 deg
  shoulder tilt reads -6 head / +6 shoulder), calibration and circular
  deviations, gate firing after persistence, gate breaking on early
  release / velocity-cap jump / tracking loss, fired latching until
  reset, both presets pinned (including the proof that the shoulder
  best rule fires on 2.5 deg of ordinary drift), section copy present,
  and the feedback URL exact.
- **Landmarker end-to-end (by hand, not in the suite):** the lite
  model on a real photo returns 33 landmarks and sensible angles
  (head raw 4.89 deg on the test photo); on a blank frame it returns
  no pose and does not crash.
- **App renders headless (AppTest):** no exception; title, all eight
  section headers, both tabs (Live video, Snapshot check), the camera
  input, the preset radio defaulting to operating, and the feedback
  link button with the exact GitHub issues URL all present.
- **App serves locally:** `streamlit run demo/app.py` answers 200 on
  the health endpoint and the page, with no traceback in the server
  log.

## Honest verdict, unchanged

Nothing passes the gate. Best shoulder rule 4/5 trials with 8 false
positives (F1 0.471); best head rule 2/5 with 3 false positives
(F1 0.400); best models F1 0.172 shoulder and 0.105 head. The live
check does not change that: it is the same gate, now running on the
visitor's own camera. The best-rule preset is the live demonstration
of why the gate fails, not a fix for it.

## What is still untested or uncertain

- **The live video stream has never run in a real browser.** This VM
  has no camera, so the webrtc path was verified only to the point
  of rendering (AppTest) and serving (local run). The frame loop,
  calibration button, and gate readouts in the video tab are
  untested end to end, and the Community Cloud deploy has not been
  exercised yet. If the video stream fails for a visitor, the
  snapshot tab is the working fallback, and the app says so.
- **Streamlit Community Cloud deploy is not part of Day 19.** The
  push happens separately via the browser upload route; the live URL
  updates automatically on push. The new requirements (mediapipe,
  opencv-python-headless, streamlit-webrtc) will install on the
  Cloud image at deploy time, which is the first real test of that
  install.
- **Streamlit 1.67 deprecates `use_container_width`.** The Day 18
  showcase code uses it in four places; the server prints a
  deprecation warning but everything still renders. Left alone on
  purpose: it is a warning, not a break, and Day 19 did not want to
  churn Day 18 lines.
- The harness stress test is still deferred, and the real re-benchmark
  still waits on his camera re-run capture.
