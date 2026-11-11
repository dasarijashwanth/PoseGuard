# PoseGuard posture classifier (Day 5)

A scikit-learn classifier per cue, trained on the Run 1 validation
CSV, served behind FastAPI. Day 5 of the reshaped PoseGuard track
(Python / AI / ML / Data Science).

## Setup

Same venv as the eval harness:

```bash
~/.venv-poseguard-eval/bin/pip install -r ../eval/requirements.txt
```

Needs pandas, numpy, scikit-learn, fastapi, httpx, uvicorn, mlflow,
pytest.

## Train

From the repo root:

```bash
python -m experiments.model.train --csv <path/to/validation.csv>
python -m experiments.model.train --csv <path> --save
```

`--save` also writes the winning model per cue to
`artifacts/model_<cue>.joblib` (about 2 KB each) and the full
metrics to `results.json`. The eval CLI picks `results.json` up
automatically and adds the classifier section to `REPORT.md`.

## MLflow tracking (Day 6)

Log a training run instead of overwriting `results.json`:

```bash
python -m experiments.model.train --csv <path> --mlflow
python -m experiments.model.train --csv <path> --model random_forest \
    --mlflow --run-name random_forest_only
```

Runs land in the local file store `mlruns/` (not committed to git).
Compare them with `python -m experiments.model.tracking`, which
writes `mlflow_comparison.json`, or browse them with:

```bash
mlflow ui --backend-store-uri experiments/model/mlruns
```

On Run 1 the comparison is: baseline (best model per cue) head 1/5
trials F1 0.078, shoulder 5/5 F1 0.179; random-forest-only head
0/5 F1 0.000, shoulder 0/5 F1 0.000, despite about 0.93 accuracy,
because it answers "upright" for every window. Accuracy lies on
imbalanced data; F1 is the honest score.























## Serve

```bash
uvicorn experiments.model.api:app --port 8000
```

## Docker (Day 6)

Build from the repo root (the build context):

```bash
docker build -f experiments/model/Dockerfile -t poseguard-api .
docker run -p 8000:8000 poseguard-api
```

Or run the API plus the MLflow UI together (from `experiments/model/`):

```bash
docker compose up --build
```

API on http://localhost:8000, MLflow UI on http://localhost:5000.
The image already contains the trained models; mounting
`./artifacts` over `/app/experiments/model/artifacts` serves a
freshly retrained model without rebuilding. Note: Docker is not
installed on the build VM, so the image was verified by definition
tests, not by a real build.

- `GET /health` returns which models are loaded.
- `POST /predict` takes a window of raw angle samples and returns
  the predicted state plus the probability it is a deliberate tilt:

```json
{
  "cue": "head",
  "baseline_head": -5.622,
  "baseline_shoulder": 174.173,
  "samples": [
    {"elapsed_s": 0.0, "head_tilt": -2.1, "shoulder_tilt": 175.0,
     "tracking": 1}
  ]
}
```

Send 10 to 100 samples; the most recent 10 are used (one 1-second
window). At least 8 of them must be tracked, or the request is
rejected with 422.

## How it works

Each 1-second window (10 frames) of the smoothed circular deviations
becomes a feature vector. The head classifier may use the head angle
level (mean, max, signed mean, fraction past 5 deg) plus steadiness
(std, frame-to-frame change). The shoulder classifier is forbidden
from seeing the shoulder angle level: Day 4 proved that angle cannot
tell a dropped shoulder from leaning at a desk. It gets only
dynamics and coupling (shoulder jumpiness, range, drift slope,
head/shoulder correlation, head steadiness and level as context).

Labels come from the protocol, not from a threshold: windows inside
a trial hold are "tilted", windows from ordinary work and the
ready/rest gaps are "upright". Logistic regression and a 200-tree
random forest are trained per cue; the better one by out-of-fold F1
is kept. Evaluation is grouped 5-fold cross-validation, grouped by
trial, so no trial's windows appear on both sides of a split.

## Tests

```bash
python -m pytest experiments/model/tests/ -q
```

26 tests: feature engineering (including the ban on shoulder-level
features), training determinism, learning on clean synthetic data,
API endpoints via TestClient, and an end-to-end run on the real
Run 1 CSV. Day 6 adds 11 more (MLflow logging and comparison,
Dockerfile/compose definition), 37 model tests in total.

## What it found on Run 1 (2026-11-03)

The honest result: the classifier does **not** beat the rule-based
baseline on either cue.

- Head (logistic regression): 1/5 trials detected, 6 false-positive
  events, F1 (tilted) 0.078. The rule: 0/5, 0 FP. Worse on false
  positives, barely better on detection.
- Shoulder (logistic regression): 5/5 trials detected, but precision
  0.10 and 5 false-positive events, F1 (tilted) 0.179. The rule:
  0/5, 7 FP. The trial "win" is a stillness artifact: during the
  holds you sat still, during work the tracker was noisy, so the
  model learned "steady = tilted". On calm windows only, accuracy
  drops to 0.45.
- No decision threshold passes the gate (4/5 trials, 0 FP) for
  either cue: the sweep trades trials for false positives at every
  cutoff.
- Worst of all, the head model learned the relationship backwards:
  a flat upright window scores 0.56 tilted, a steady +6 deg tilt
  scores 0.33, because in Run 1 the biggest head deviations happened
  during noisy desk work, not during holds.

Why: one session of five brief holds is too little data to train on,
and protocol labels are not ground truth (two of the five head holds
barely tilted at all). The classifier is built, tested and served,
but it needs the re-run data (longer, deeper holds, more trials)
before it can say anything the rule cannot.

See `WALKTHROUGH.md` in `../eval/` for the plain-English version
and `../eval/REPORT.md` for the full numbers.

## Day 8: PyTorch MLP benchmark

A small MLP per cue (`torch_mlp.py`, CPU-only, two hidden layers
32 and 16, ReLU, dropout, Adam with weight decay, early stopping,
inverse-frequency class weights). Same window features and same
grouped 5-fold CV as Day 5, so the comparison is apples-to-apples.
Run it with:

```bash
python -m experiments.model.benchmark --csv <path/to/validation.csv>
```

Torch installs CPU-only from the PyTorch CPU wheel index (see the
note in `../eval/requirements.txt`); the default index fetches the
large CUDA build, which this machine does not need.

Verdict on Run 1: the MLP does **not** beat logistic regression
on either cue (head F1 0.060 vs 0.078, shoulder 0.133 vs 0.179)
and passes the gate on neither (head 1/5 trials, 4 FP; shoulder
2/5 trials, 1 FP). It is still the least trigger-happy model
(head 100/318 work windows flagged vs 172, shoulder 54 vs 134)
and the only one whose signal survives the calm-only check
(accuracy 0.685/0.663 vs 0.427/0.447). Full table and verdicts in
`BENCHMARK.md`, plain-English version in `WALKTHROUGH.md`.

## Day 9: Render deploy prep

The API is ready to go live on Render. `render.yaml` at the repo
root is a Render Blueprint: one web service, `poseguard-api`,
Docker runtime, free plan, health check on `/health`. No secrets
needed, the image already contains the trained models.

Two blockers fixed in the `Dockerfile`: the start command now binds
`$PORT`, which Render sets (10000 by default, not 8000), and the
image now installs the slim runtime-only requirements
(`requirements-api.txt`), because the old file pulled torch and
mlflow and the default torch install is the multi-gigabyte CUDA
build. The MLflow service in `docker-compose.yml` still gets the
full requirements through a build arg.

Full steps, including the exact clicks, are in
`../docs/DEPLOY.md`; plain-English version in
`WALKTHROUGH_DAY9.md`. Verified live locally with uvicorn on a
Render-style port: `/health` 200 with both models loaded,
`/predict` 200 for both cues, bad cue 422. Still unverified: no
real Docker build has ever run (Docker is not installed on the
build VM), and there is no live URL until the Render steps run.
















