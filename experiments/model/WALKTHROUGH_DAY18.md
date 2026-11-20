# Day 18 walkthrough: Streamlit public demo showcase (Nov 20)

## What was built

The project had no public face: the API needs his Render/AWS clicks for
a live URL, and the repo is notebooks of numbers. Day 18 ships a
Streamlit app anyone can open, showing the real Run 1 session instead
of a simulation, with the honest verdict as the headline.

- `demo/app.py`: the UI only. Title, verdict banner, session chart,
  gate verdicts per trial, model scoreboard, and what would fix it.
- `demo/demo_data.py`: pure data + verdict logic, no Streamlit import,
  so the test suite exercises it without a server.
- `demo/data/run1_sample.csv`: 727 rows at 1 Hz, downsampled from the
  7,820-frame Run 1 session. Head is the circular deviation on
  tracking-valid frames; shoulder is the Day 11 jump-gated deviation,
  NaN where the tracker was noisy or lost.
- `demo/requirements.txt`, `demo/README.md`: run + free-deploy notes.

## How to run / deploy

```bash
pip install -r demo/requirements.txt
streamlit run demo/app.py
```

Free deploy: Streamlit Community Cloud, repo `dasarijashwanth/PoseGuard`,
main file `demo/app.py`. No secrets needed.

## The honest verdict, unchanged

NOTHING passes the gate (4/5 trials, 0 false positives). The demo
shows why instead of hiding it:

- Best rule, shoulder: 1 deg / 5 s / 5 deg/s cap, 4/5 trials,
  8 false positives, F1 0.471. It reaches the detection half and
  fails the false-positive half.
- Best rule, head: 4 deg / 2 s / 10 deg/s cap, 2/5 trials,
  3 false positives, F1 0.400.
- Best models (latest features): shoulder LR / MLP F1 0.172 (3/5
  trials), head MLP F1 0.105 (2/5), LR F1 0.097. Random forest 0/5
  on both cues; its accuracy is high only because it predicts almost
  everything upright.
- The holds are shallow as well as brief: deepest head hold 7.1 deg
  against an 8 deg threshold no hold ever reaches; trial 5 is barely
  recorded (6 head frames, 8 shoulder), so at most 4/5 trials are
  measurable.

## Something the demo surfaced

Building the sample chart found a 6-second tracker meltdown at 447 s,
mid-work: raw head angle swings -118 to +55 deg frame to frame and
shoulder 142 to -144 deg, with NaN gaps where tracking drops. The
jump gate does not fully contain it (shoulder +141.74 deg survives
into the gated series). The sample keeps it and the chart clips its
axis at +/-25 deg with the meltdown disclosed in the caption, rather
than deleting the ugly seconds or letting one meltdown hide the 2-7
deg holds. The loss step also records ~21 deg sustained head
deviation while the tracker claims to be tracking; that is the
landmark-loss test, not a posture signal.

## Tests

19 new tests in `experiments/model/tests/test_day18.py`: sample loads
with the expected 727 rows and columns, deviations stay inside
circular bounds with the meltdown pinned at 447 s, gate verdicts name
the failing half, best rules and scoreboard numbers are pinned to
`results_day15.json` / `results_day17.json`, trial 5 is barely
recorded, and no scoreboard row passes the gate.

Full suite: 258/258 tests passing in one combined run (80 eval +
178 model, including 19 new Day 18 tests). Streamlit itself is not
in the eval venv and is not needed by any test; the app imports it
only in `demo/app.py`.

## What is still missing

The live URL: the demo is deployable in one click on Streamlit
Community Cloud, but nobody has clicked it yet. The harness stress
test is deferred. The real re-benchmark still waits on his capture
CSV (Day 14 tool, Day 16 harness).
