# Day 14 walkthrough: camera re-run capture tool

Day 13 closed with the same honest verdict as Days 11 and 12: the
API now serves the best model we have, and it still does not pass
the gate, because the shoulder classifier fires on quiet windows
instead of tilted ones. The mechanism is known and the data cannot
fix it: deliberate holds move the shoulder angle only about 2.5
degrees while ordinary work drifts 4 to 7, and the Run 1 CSV has
only angle columns, no raw landmarks. Day 14 builds the fix itself:
a camera re-run capture tool that records a new dataset with raw
landmarks and longer, deeper holds, so the Day 15 re-benchmark can
test what this dataset could not.

## What was built

**The capture tool: `experiments/rerun-capture.html`.** A standalone
HTML file, no build step, no install on his laptop beyond a browser.
It reuses the Run 1 experiment's proven patterns: the same MediaPipe
Tasks Vision Pose Landmarker Lite from the CDN with the classic
MediaPipe Pose fallback, the same ear/shoulder angle math, the same
EMA smoothing and circular deviation math, and the same CSV export
layout. What is new:

- **All 33 landmarks per frame.** Every frame records x, y and
  visibility for landmarks 0 to 32, including hips 23/24 and
  shoulder heights. The two signals Run 1 could not test at all,
  shoulder-vs-hip and shoulder height asymmetry, are now in the
  data. Tracking requires ears 7/8, shoulders 11/12 and hips 23/24
  all visible; if the hips leave the frame the tool says so and
  pauses, because a frame without hips cannot test the
  shoulder-vs-hip signal.
- **Longer, deeper holds.** Holds are 10 s (was 6 s), ready and
  rest are 5 s (was 3 s), calibration is 5 s (was 3 s). The hold
  has a depth target shown live: shoulder 8 deg, head 15 deg. The
  target bar turns green past the target, so he can see whether he
  is holding deep enough while holding. Run 1 shoulder holds
  averaged about 2.5 deg of deviation, inside the 4-7 deg everyday
  drift band. An 8 deg shoulder target sits above the drift band:
  if the new data separates, the level signal itself becomes usable
  again. A 10 s hold gives the 5 s persistence gate real margin
  after the move into position and the EMA ramp-up; Run 1's 6 s
  holds left about 1 s.
- **Guided session.** The tool walks through calibration, 5 head
  trials, 5 shoulder trials, a 5 minute work sample and a 10 s
  landmark-loss test, about 8.6 minutes total. Every frame is
  labelled with cue, hold id (1-5) and phase (calib, ready, hold,
  rest, calm, loss). A hold counts as detected when the deviation
  stays beyond its target for at least 5 s inside the hold.
- **Two downloads.** The CSV is a superset of the Run 1 schema: the
  same metadata block and nine frame columns, plus the three label
  columns and the 99 landmark columns, plus the same per-trial
  summary. The existing loader reads it unchanged. The manifest is a
  JSON summary: holds performed with duration, tracked fraction,
  peak deviation and whether the target was reached, the work
  sample, and every tracking-loss gap of 1 s or more with its phase
  and hold context.

**The protocol module: `experiments/eval/rerun_protocol.py`.** The
single source of truth for the re-run numbers: timings, depth
targets, landmark ids, the phase timeline for one cue's trial block
and for the whole 515 s session, and the frame column list. The
HTML implements the same numbers in JavaScript, and a test reads the
HTML and pins the two together so they cannot drift apart.

**The validator: `experiments/eval/rerun.py`.** Checks a captured
CSV before the Day 15 re-benchmark uses it: the Run 1 columns are
all there, the Run 2 metadata keys are there, every landmark column
is there, 5 holds per cue with at least 8 s of frames each, and
labels in range. Data-quality problems are warnings, not errors: a
hold that never reached its target, a hold with heavy tracking loss,
a work sample cut short, and every unexpected tracking-loss gap.
The loss-test gap is expected and is not reported as a gap.

## How he runs it

1. Download `experiments/rerun-capture.html` from the repo and save
   it in a folder by itself, for example `C:\poseguard\`.
2. Open PowerShell in that folder and paste the block printed at the
   top of the page. It serves the folder on
   `http://localhost:8000/` with the built-in `HttpListener`.
   Leave the PowerShell window open while recording.
3. Open `http://localhost:8000/rerun-capture.html` in Chrome or
   Edge. Do not open the file directly with `file://`; the camera
   is blocked outside a secure context, and localhost counts as one.
4. Press Load pose engine, then Enable camera and allow access, then
   Start session and follow the prompts. About 8.5 minutes.
5. At the end, press Download CSV and Download manifest, and send
   both files back. The CSV goes through `validate_rerun_csv` first;
   if it reports errors, the session needs a re-run, and if it
   reports warnings, the re-benchmark notes them.

Nothing is installed on his laptop: the page is one file, the pose
model loads from the CDN, and all processing stays in the tab.

## Tests

166/166 passing in one combined run (60 eval + 106 model), up from
150/150 on Day 13. 16 new Day 14 tests in
`experiments/eval/tests/test_rerun.py`:

- Protocol: timings (5/5/10/5/300/10, 5 trials, 515 s session),
  targets (head 15, shoulder 8), trial-block and session timelines
  (contiguous, 10 holds, one calm sample), 99 landmark columns,
  frame columns a superset of the Run 1 schema.
- Validator: a full synthetic Run 2 session passes clean (5150
  frames, 10 holds, no gaps, no warnings); the Run 1 loader reads
  the Run 2 CSV unchanged; a missing landmark column fails; missing
  Run 2 metadata fails; a 2 s tracking gap in shoulder hold 3 is
  reported with phase, cue and hold id and warns but does not fail;
  a hold below its target warns; a short work sample warns; a 0.5 s
  flicker is not listed as a gap.
- HTML/Python sync: the capture tool declares the same constants as
  the protocol module, the same Run 2 metadata label, the label
  columns, the 33-landmark loop, the required landmark set, and the
  localhost serve instructions.

## What is still open

- The capture has not run yet. The tool produces real data only when
  he runs it; no captured data was invented here, and the Run 1 data
  is untouched.
- Day 15 is the re-benchmark on the new data: run the Day 11/12
  gated models against the new CSV, test shoulder-vs-hip and
  shoulder height asymmetry for the first time, and see whether
  anything passes the gate.
- The live URL still needs the Render clicks in `docs/DEPLOY.md`
  or the AWS clicks in `docs/DEPLOY_AWS.md`. The camera re-run does
  not change that.
- A real Docker build has still never run (no Docker on the build
  VM).

## Files changed

- `experiments/rerun-capture.html`: the capture tool (new).
- `experiments/eval/rerun_protocol.py`: protocol constants and
  timelines (new).
- `experiments/eval/rerun.py`: Run 2 CSV validator (new).
- `experiments/eval/tests/test_rerun.py`: 16 new tests.
- `experiments/WALKTHROUGH_DAY14.md`: this file.
- Root `README.md`: "Built through Day 14", Day 14 Build Progress
  entry.
