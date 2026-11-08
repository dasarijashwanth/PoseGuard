"""FastAPI service for the Day 5 posture classifiers.

Endpoints:
    GET  /health   - which models are loaded
    POST /predict  - predict the posture state of one window of samples

A prediction takes a small window of raw angle samples (the same shape
the webcam experiment logs), smooths them with the same circular EMA
the harness uses, measures circular deviation from the given
baselines, extracts the cue's window features, and asks the trained
model. The reply is the predicted state plus the model's probability
that the window is a deliberate tilt.

The trained models live in experiments/model/artifacts/ (written by
`python -m experiments.model.train --csv <csv> --save`). The app
loads them at startup; pass in-memory models to `create_app` instead
when embedding or testing.

Run it (from the repo root, after training with --save):
    uvicorn experiments.model.api:app --port 8000
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from experiments.eval.analysis import ema_smooth
from experiments.eval.circular import circ_dev

from .features import FEATURES, MIN_TRACKED, WINDOW_N, window_features

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"

# Run 1 baselines, used only when the caller does not send their own.
DEFAULT_BASELINE_HEAD = -5.622
DEFAULT_BASELINE_SHOULDER = 174.173


class Sample(BaseModel):
    elapsed_s: float
    head_tilt: float
    shoulder_tilt: float
    tracking: int = Field(default=1, ge=0, le=1)


class PredictRequest(BaseModel):
    cue: str
    baseline_head: float = DEFAULT_BASELINE_HEAD
    baseline_shoulder: float = DEFAULT_BASELINE_SHOULDER
    samples: list[Sample] = Field(min_length=WINDOW_N, max_length=100)


class PredictResponse(BaseModel):
    cue: str
    state: str
    probability_tilted: float
    n_samples: int
    model: str


def load_models(artifacts_dir: Path = ARTIFACTS_DIR) -> dict:
    """Load the saved model payload per cue (raises if any is missing)."""
    models = {}
    for cue in ("head", "shoulder"):
        path = artifacts_dir / f"model_{cue}.joblib"
        if not path.exists():
            raise RuntimeError(
                f"no trained model for {cue!r} at {path}; run "
                "'python -m experiments.model.train --csv <csv> --save' first")
        models[cue] = joblib.load(path)
    return models


def predict_window(models: dict, cue: str, baseline_head: float,
                   baseline_shoulder: float,
                   samples: list[Sample]) -> dict:
    """Feature-extract one window and predict. Pure function, testable."""
    if cue not in models:
        raise ValueError(f"no model loaded for cue {cue!r}")
    ordered = sorted(samples, key=lambda s: s.elapsed_s)[-WINDOW_N:]
    tracked = [s for s in ordered if s.tracking == 1]
    if len(tracked) < MIN_TRACKED:
        raise ValueError(
            f"only {len(tracked)} tracked samples in the window "
            f"(need {MIN_TRACKED})")
    h_smooth = ema_smooth(pd.Series([s.head_tilt for s in ordered]))
    s_smooth = ema_smooth(pd.Series([s.shoulder_tilt for s in ordered]))
    h_dev = np.array([circ_dev(a, baseline_head) if pd.notna(a) else np.nan
                      for a in h_smooth])
    s_dev = np.array([circ_dev(a, baseline_shoulder) if pd.notna(a) else np.nan
                      for a in s_smooth])
    feats = window_features(h_dev, s_dev)
    payload = models[cue]
    X = np.array([[feats[f] for f in payload["features"]]], dtype=float)
    proba = float(payload["model"].predict_proba(X)[0][1])
    return {"cue": cue,
            "state": "tilted" if proba >= 0.5 else "upright",
            "probability_tilted": round(proba, 3),
            "n_samples": len(ordered),
            "model": payload["model_name"]}


def create_app(models: dict | None = None,
               artifacts_dir: Path = ARTIFACTS_DIR) -> FastAPI:
    """Build the app. `models` overrides the artifact files on disk."""
    loaded = models if models is not None else load_models(artifacts_dir)
    app = FastAPI(title="PoseGuard posture classifier",
                  description="Day 5: scikit-learn tilt classifier "
                              "over webcam angle windows.")

    @app.get("/health")
    def health():
        return {"status": "ok",
                "models": sorted(loaded),
                "model_names": {c: loaded[c]["model_name"]
                                for c in sorted(loaded)}}

    @app.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest):
        if req.cue not in ("head", "shoulder"):
            raise HTTPException(status_code=422,
                                detail="cue must be 'head' or 'shoulder'")
        try:
            return predict_window(loaded, req.cue, req.baseline_head,
                                  req.baseline_shoulder, req.samples)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))

    return app


# Importing this module must not fail before the models exist (tests
# build their own in-memory models), so the ready-to-serve app is
# best-effort: it is None until training has saved the artifacts.
try:
    app = create_app()
except RuntimeError:
    app = None
