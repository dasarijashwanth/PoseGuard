# Day 13 walkthrough: serve the gated models in the API

Day 13 puts the Day 11/12 work into the live path. Until now the API
served the Day 5 models (logistic regression on the old 8 shoulder
features) and scored one window at a time, so the 5-second
persistence gate from the PRD could never run live. Day 13 retrains
the serving models on the gated features, versions them as v2, and
teaches the API to score a trailing history and apply the gate.

## What changed

**Serving artifacts, versioned.** `experiments/model/artifacts_v2/`
holds the new serving models: head logistic regression on its 6
features (unchanged, its features never changed), shoulder logistic
regression on the new 11-feature set (the 8 old ones plus the three
gated stats mean_abs_sn, max_abs_sn, std_sn). The retrain was run
with `train.py --save --artifacts-dir experiments/model/artifacts_v2
--out experiments/model/results_day13.json`, and it reproduces the
Day 12 new-feature numbers exactly: shoulder LR F1 0.219, 5/5
trials, 3 false positives; head LR F1 0.078, 1/5 trials, 6 false
positives. The old Day 5 artifacts stay in `artifacts/`,
untouched, for comparison. `train.py` gained a `--artifacts-dir`
flag (default `artifacts/`) so a retrain can be aimed at either
directory without editing code.

**Which model, and why.** Logistic regression, for both cues. On
the gate numbers, which are what the PRD cares about, shoulder LR
new is 5/5 trials and MLP new is 4/5, so LR is the only model that
clears the trials half of the gate. Its F1 is also the best of the
three (0.219 vs MLP 0.215 vs RF 0.000). The random forest is 0/5
on both feature sets, so it was never in the running. The pick is
on the gate numbers, not on F1 alone, and F1 happens to agree.

**Trailing history in the API.** `POST /predict` now accepts an
optional `gate_s` (0 to 10 seconds, default 0). Send more than 10
samples and the API scores every full 1-second window in the
request, then applies the persistence gate: with `gate_s=5` the
response is "tilted" only if the last 5 scored windows are all
tilted. A window with too few tracked samples is reported as
"lost" and resets the run, so tracking loss restarts the 5-second
clock instead of bridging it. The response gains four fields:
`gated` (whether the gate ran), `gate_s`, `windows` (the per-window
states and probabilities), and `consecutive_tilted_windows` (the
trailing run). With `gate_s=0` the endpoint behaves exactly as
before: the last window decides, one entry in `windows`, `gated`
false. `GET /health` and the request shape are unchanged.

**How the gate scores a window.** Each 1-second window is smoothed
(EMA) and jump-gated on its own second of samples, the same as the
Day 5 endpoint did. That is a deliberate trade, not an oversight:
the first attempt smoothed the EMA over the whole request, and the
oldest test in the suite caught it. A window that starts right
after an upright-to-tilted transition ramps up over its first
frames, so its smoothed deviations are smaller and its std and
jerk are larger, and the head model reads that as upright (the
tilted window scored 0.04 against 0.472 for upright-only). The
per-window smoothing keeps the Day 5 contract that the test pins:
send 10 to 100 samples, the most recent 10 are used. Two limits
follow from it and are worth knowing: the gate cannot see a jump
just before a window, and a jump at the end of one window can leak
into the next window's first frames. The training harness smooths
the whole session instead, so there is a small train/serve skew at
window starts. It was there on Day 5 too; Day 13 does not make it
worse, it just scores five windows instead of one.

**Deploy files follow the served model.** `docker-compose.yml`
mounts `./artifacts_v2` over the image's copy now (the Dockerfile
comment says the same), and `deploy/aws/deploy.sh` backs
`artifacts_v2/` up to S3 instead of `artifacts/`. The Docker image
itself already contains v2 through `COPY experiments/`.

## The honest verdict: the served models do NOT pass the gate

The gate is 4/5 trials with 0 false positives, and nothing in the
repo passes it yet.

- Head, served LR: 1/5 trials, 6 false positives, F1 0.078.
  Unchanged from Day 5; the head signal was never fixed.
- Shoulder, served LR on the gated features: 5/5 trials, 3 false
  positives, F1 0.219, calm-only accuracy 0.585. It clears the
  trials half and fails the false-positives half.

So the API now serves the best model we have, and it still fails
the gate, because of the 3 false positives. The mechanism is the
same one Day 11 found: the gated coefficients are negative
(mean_abs_sn -1.775, max_abs_sn -1.627), which means the model
fires on quiet windows, not on tilted ones. The gate logic itself
is verified end to end: the real v2 shoulder model on a real 5
second hold from Run 1 (hold 3 of the shoulder trials) scores all
five windows tilted and the gate fires, and a 2-second tilt
followed by upright does not fire. The plumbing is ready; the
signal is not. The fix is still the camera re-run with raw
landmarks and longer, deeper holds, not a fancier model.

## Tests

150/150 passing in one combined run (44 eval + 106 model),
up from 123/123 on Day 12. 27 new Day 13 tests:

- `tests/test_gate.py` (21): the gate itself (5 sustained tilt
  fires, 4 then upright does not, a brief tilt does not,
  tracking loss resets the run, fractional seconds round up,
  empty and all-lost inputs), scoring windows from a trailing
  history (sustained tilt, brief tilt, loss resets the run,
  leading loss, partial trailing window dropped, all lost
  raises), and the endpoint (gate fires on sustained tilt,
  blocks a brief tilt, blocks a tilt interrupted by tracking
  loss, no `gate_s` is backward compatible, `gate_s` over 10
  seconds rejected, too few samples rejected, all lost
  rejected, and the real v2 shoulder model on a real Run 1 hold
  passes the 5-second gate).
- `tests/test_artifacts_v2.py` (6): the v2 artifacts load by
  default (LR both cues, 6 head features, 11 shoulder features
  including the three gated stats), the v1 artifacts are
  untouched (8 shoulder features, no gated stats), the v2
  numbers match Day 12 exactly, the module-level app serves v2,
  an explicit v1 directory still works, and the
  `--artifacts-dir` CLI flag writes where asked.

One test failed during the build and taught something:
`test_predict_uses_most_recent_window` (a Day 5 test) broke when
the EMA was first run over the whole request, scoring the tilted
window 0.04 against 0.472 for upright-only. The fix was to keep
per-window smoothing, as described above, and the test passes
again with no change to the test itself.

## Files changed

- `experiments/model/train.py`: `--artifacts-dir` flag; the
  trained dict now also records the feature list per cue, so a
  saved artifact can be checked against the code.
- `experiments/model/api.py`: default artifacts dir is now
  `artifacts_v2`; `predict_windows` scores every full 1-second
  window; `apply_gate` runs the persistence rule; the endpoint
  gains `gate_s` and the four gate fields.
- `experiments/model/artifacts_v2/`: new serving models,
  `model_head.joblib` and `model_shoulder.joblib`.
- `experiments/model/results_day13.json`: the retrain metrics.
- `experiments/model/tests/test_gate.py`,
  `tests/test_artifacts_v2.py`: 27 new tests.
- `experiments/model/docker-compose.yml`,
  `experiments/model/Dockerfile`, `deploy/aws/deploy.sh`,
  `tests/test_docker.py`, `tests/test_aws.py`: the compose mount
  and the S3 backup now point at `artifacts_v2/`.
- `experiments/model/README.md`, root `README.md`: Day 13
  section and Build Progress entry.

## What is still open

- The live URL still needs the Render clicks in `docs/DEPLOY.md`
  or the AWS clicks in `docs/DEPLOY_AWS.md`. The Render
  Blueprint and the AWS deploy script now serve the v2 models
  with no extra step.
- No model passes the gate on Run 1. The camera re-run with raw
  landmarks (hips and shoulder heights) and longer, deeper holds
  is still the missing piece.
- A real Docker build has still never run (no Docker on the
  build VM), and the 5-second gate has not been exercised
  against a live camera stream, only against recorded frames.
