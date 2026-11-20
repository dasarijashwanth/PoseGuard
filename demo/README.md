# PoseGuard public demo (Day 18)

A Streamlit showcase of the project for anyone who opens the repo: the
real Run 1 validation session, gate verdicts per trial, and the model
scoreboard, with the honest verdict up front.

## Run locally

```bash
pip install -r demo/requirements.txt
streamlit run demo/app.py
```

## Deploy (free)

Streamlit Community Cloud: new app, repo `dasarijashwanth/PoseGuard`,
main file `demo/app.py`. No secrets or environment variables needed;
the app reads only the bundled sample CSV.

## What it shows

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
  the evaluation pipeline, so the app stays light.
- `demo/demo_data.py`: pure data + verdict logic, fully tested without
  Streamlit installed.
- `demo/data/run1_sample.csv`: trimmed 1 Hz sample of Run 1. The full
  session CSV is not in the repo.
