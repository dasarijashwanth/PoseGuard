# PoseGuard evaluation harness

Python analysis of the webcam validation experiment's CSV exports.
Day 3 of the reshaped PoseGuard track (Python / AI / ML / Data Science).

## Setup

```bash
python3 -m venv ~/.venv-poseguard-eval
~/.venv-poseguard-eval/bin/pip install -r requirements.txt
```

Needs pandas, numpy, pytest, scikit-learn, fastapi, httpx. Python 3.11+.

## Usage

From the repo root:

```bash
python -m experiments.eval.cli --csv <path/to/validation.csv>
python -m experiments.eval.cli --csv <path> --out experiments/eval/REPORT.md
```

Prints per-cue verdicts (PASS/FAIL vs >= 4/5 detections and zero false
positives) plus the threshold sensitivity table, and writes the full
Markdown report.

## Tests

```bash
python -m pytest experiments/eval/tests/ -q
```

33 tests: circular math, CSV loading and schema validation, smoothing
and persistence detection, trial alignment, false positives, threshold
sensitivity, Wilson intervals and FP rates, threshold and persistence
curves, per-cue statistical verdicts, and end-to-end runs on the real
Run 1 CSV. The Day 5 model package adds 26 more tests under
`../model/tests/` (59 total).

## What it found on Run 1 (2026-11-03)

- Head tilt: FAIL, 0/5 at every threshold 5-8. Longest run above 5 deg
  was 1.5 s vs 5 s required. Tilts too small and too brief.
- Shoulder tilt: FAIL, 0/5 at every threshold 4-8, plus 7 false
  positives during ordinary work. Fixing the wraparound math does not
  fix the cue: leaning while typing looks like a dropped shoulder.
- Landmark-loss test never ran (18 lost frames out of 1,828).
- The old linear deviation produced phantom jumps beyond 180 deg in
  1,188 shoulder frames. Circular math removes all of them.

## Day 4: statistical validation

`stats.py` asks whether ANY threshold or persistence rule could make
a cue pass, instead of just checking the one in the file:

- Wilson 95% intervals: head detection 0/5 is plausibly 0-43%, far
  below the 80% gate. Not bad luck on a small sample.
- False-positive rate: shoulder fires ~83 times per hour of desk work
  (7 in ~5 minutes). Unusable as a reminder.
- Threshold curve: no row gives high detection with zero false
  positives for the shoulder; even a 30 deg threshold still fires.
- Persistence curve: at 5 deg with a 0.5 s rule, head would have
  caught 3/5 holds. The tilts were real but far too brief.
- Verdicts: head is **salvageable** (clean record, re-run with
  longer, deeper holds); shoulder's **metric is wrong for desk work**
  (no knob setting passes both criteria, needs a different signal).

See `WALKTHROUGH.md` for the plain-English version and `REPORT.md`
for the full numbers.

## Day 5: the classifier (experiments/model/)

`experiments/model/` trains one scikit-learn classifier per cue over
1-second windows of the angle stream and serves it behind FastAPI
(`GET /health`, `POST /predict`). The shoulder classifier never sees
the raw shoulder angle level, only dynamics and head/shoulder
coupling, per the Day 4 verdict.

Honest result on Run 1: it does **not** beat the rule baseline.
Head: 1/5 trials, 6 false-positive events (rule: 0/5, 0 FP).
Shoulder: 5/5 trials but precision 0.10 and 5 false-positive
events (rule: 0/5, 7 FP), and the calm-only check shows the
shoulder "win" is tracker stillness, not posture. No decision
threshold passes the gate for either cue. One session of five brief
holds is too little data to train on, and protocol labels are not
ground truth. See `experiments/model/README.md`.
