# PoseGuard evaluation harness

Python analysis of the webcam validation experiment's CSV exports.
Day 3 of the reshaped PoseGuard track (Python / AI / ML / Data Science).

## Setup

```bash
python3 -m venv ~/.venv-poseguard-eval
~/.venv-poseguard-eval/bin/pip install -r requirements.txt
```

Needs only pandas, numpy, pytest. Python 3.11+.

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

21 tests: circular math, CSV loading and schema validation, smoothing
and persistence detection, trial alignment, false positives, threshold
sensitivity, and an end-to-end run on the real Run 1 CSV.

## What it found on Run 1 (2026-11-03)

- Head tilt: FAIL, 0/5 at every threshold 5-8. Longest run above 5 deg
  was 1.5 s vs 5 s required. Tilts too small and too brief.
- Shoulder tilt: FAIL, 0/5 at every threshold 4-8, plus 7 false
  positives during ordinary work. Fixing the wraparound math does not
  fix the cue: leaning while typing looks like a dropped shoulder.
- Landmark-loss test never ran (18 lost frames out of 1,828).
- The old linear deviation produced phantom jumps beyond 180 deg in
  1,188 shoulder frames. Circular math removes all of them.

See `WALKTHROUGH.md` for the plain-English version and `REPORT.md`
for the full numbers.
