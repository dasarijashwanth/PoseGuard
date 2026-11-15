"""FastAPI service for the PoseGuard posture classifiers.

Endpoints:
    GET  /health   - which models are loaded
    POST /predict  - predict the posture state of one window of
                     samples, or of a trailing history with the
                     5-second persistence gate

Day 13 serves the retrained v2 artifacts
(experiments/model/artifacts_v2/): logistic regression per cue,
shoulder trained on the Day 11 gated features (11 features: the
Day 5 dynamics and coupling plus the three gated stats). The Day
5 artifacts stay in artifacts/ untouched for comparison. The API
now also computes the jump-gated shoulder deviation from the
samples it receives, so the v2 shoulder model gets real gated
features instead of the 0.0 placeholders the Day 5 API fed it.

A prediction takes raw angle samples (the same shape the webcam
experiment logs), smooths them with the same circular EMA the
harness uses, measures circular deviation from the given
baselines, extracts the cue's window features, and asks the
trained model. The reply is the predicted state plus the model's
probability that the window is a deliberate tilt.

Trailing history: POST /predict accepts 10 to 100 samples (up to
ten 1-second windows). With gate_s=0 (the default) it scores
every full window and the last one decides, exactly as the
single-window endpoint did. With gate_s set (for example 5.0),
it applies the PRD persistence rule: the state is "tilted" only
when the last K consecutive windows are all tilted, where
K = ceil(gate_s / 1 s). A window with too few tracked samples
counts as "lost" and resets the run, the same way tracking loss
resets the timer in the eval harness. A request with no usable
window at all is still rejected with 422.

The trained models live in experiments/model/artifacts_v2/
(written by `python -m experiments.model.train --csv <csv>
--save --artifacts-dir experiments/model/artifacts_v2`). The app
loads them at startup; pass in-memory models to `create_app`
instead when embedding or testing.

Run it (from the repo root, after training with --save):
    uvicorn experiments.model.api:app --port 8000
"""

from __future__ import annotations

import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from experiments.eval.analysis import ema_smooth
from experiments.eval.circular import circ_dev
from experiments.eval.signals import jump_valid_mask

from .features import FEATURES, MIN_TRACKED, WINDOW_N, WINDOW_S, window_features

ARTIFACTS_DIR = Path(__file__).parent / "artifacts_v2"

# Run 1 baselines, used only when the caller does not send their own.
DEFAULT_BASELINE_HEAD = -5.622
DEFAULT_BASELINE_SHOULDER = 174.173


class Sample(BaseModel):
    elapsed_s: float
    head_tilt: float
    shoulder_tilt: float
    tracking: int = Field(default=1, ge=0, le=1)


class WindowResult(BaseModel):
    start_s: float
    state: str
    probability_tilted: float | None = None


class PredictRequest(BaseModel):
    cue: str
    baseline_head: float = DEFAULT_BASELINE_HEAD
    baseline_shoulder: float = DEFAULT_BASELINE_SHOULDER
    samples: list[Sample] = Field(min_length=WINDOW_N, max_length=100)
    gate_s: float = Field(default=0.0, ge=0.0, le=10.0)


class PredictResponse(BaseModel):
    cue: str
    state: str
    probability_tilted: float
    n_samples: int
    model: str
    gated: bool = False
    gate_s: float = 0.0
    windows: list[WindowResult] | None = None
    consecutive_tilted_windows: int = 0


def load_models(artifacts_dir: Path = ARTIFACTS_DIR) -> dict:
    """Load the saved model payload per cue (raises if any is missing)."""
    models = {}
    for cue in ("head", "shoulder"):
        path = artifacts_dir / f"model_{cue}.joblib"
        if not path.exists():
            raise RuntimeError(
                f"no trained model for {cue!r} at {path}; run "
                "'python -m experiments.model.train --csv <csv> --save "
                "--artifacts-dir experiments/model/artifacts_v2' first")
        models[cue] = joblib.load(path)
    return models


def _deviations(samples, baseline_head: float,
                baseline_shoulder: float):
    """EMA-smoothed circular deviations for a sample list.

    Returns (ordered samples, head deviations, shoulder deviations).
    Each window is smoothed on its own, exactly as the Day 5
    endpoint did: the EMA seeds at the first sample of the window,
    so a window that starts mid-transition ramps up over its first
    frames. That is the documented serve-time approximation; the
    training harness smooths the whole session instead.
    """
    ordered = sorted(samples, key=lambda s: s.elapsed_s)
    h_smooth = ema_smooth(pd.Series([s.head_tilt for s in ordered]))
    s_smooth = ema_smooth(pd.Series([s.shoulder_tilt for s in ordered]))
    h_dev = np.array([circ_dev(a, baseline_head) if pd.notna(a) else np.nan
                      for a in h_smooth])
    s_dev = np.array([circ_dev(a, baseline_shoulder) if pd.notna(a) else np.nan
                      for a in s_smooth])
    return ordered, h_dev, s_dev


def _gated_series(samples, s_dev: np.ndarray) -> np.ndarray:
    """Jump-gated shoulder deviation, aligned with s_dev.

    The gate marks a frame dirty when the raw shoulder angle jumped
    more than 20 degrees in the last second (see
    experiments.eval.signals.jump_valid_mask). Dirty frames become
    NaN, so the window features only ever see the level where the
    tracker was sane. The gate runs on the window alone, so a jump
    just before the window is invisible to it, and a jump at the
    end of one window can leak into the next window's first frames.
    """
    frames = pd.DataFrame({
        "elapsed_s": [s.elapsed_s for s in samples],
        "shoulder_tilt": [s.shoulder_tilt for s in samples]})
    clean = jump_valid_mask(frames).to_numpy(dtype=bool)
    return np.where(clean, s_dev, np.nan)


def _score_window(models: dict, cue: str, h_dev: np.ndarray,
                  s_dev: np.ndarray, s_new: np.ndarray) -> dict:
    """Score one window from precomputed deviation arrays."""
    if cue not in models:
        raise ValueError(f"no model loaded for cue {cue!r}")
    feats = window_features(h_dev, s_dev, s_new=s_new)
    payload = models[cue]
    X = np.array([[feats[f] for f in payload["features"]]], dtype=float)
    proba = float(payload["model"].predict_proba(X)[0][1])
    return {"state": "tilted" if proba >= 0.5 else "upright",
            "probability_tilted": round(proba, 3)}


def predict_window(models: dict, cue: str, baseline_head: float,
                   baseline_shoulder: float,
                   samples: list[Sample],
                   s_new: np.ndarray | None = None) -> dict:
    """Feature-extract one window and predict. Pure function, testable.

    Uses the most recent WINDOW_N samples, smoothed and gated on
    their own, the same as the Day 5 endpoint did. `s_new` is an
    optional precomputed gated series for the window; without it
    the gate runs on this window alone.
    """
    if cue not in models:
        raise ValueError(f"no model loaded for cue {cue!r}")
    ordered = sorted(samples, key=lambda s: s.elapsed_s)[-WINDOW_N:]
    _, h_dev, s_dev = _deviations(ordered, baseline_head,
                                  baseline_shoulder)
    if s_new is None:
        s_new = _gated_series(ordered, s_dev)
    s_new = np.asarray(s_new, dtype=float)[-WINDOW_N:]
    tracked = [s for s in ordered if s.tracking == 1]
    if len(tracked) < MIN_TRACKED:
        raise ValueError(
            f"only {len(tracked)} tracked samples in the window "
            f"(need {MIN_TRACKED})")
    scored = _score_window(models, cue, h_dev, s_dev, s_new)
    return {"cue": cue, **scored, "n_samples": len(ordered),
            "model": models[cue]["model_name"]}


def predict_windows(models: dict, cue: str, baseline_head: float,
                    baseline_shoulder: float,
                    samples: list[Sample]) -> list[dict]:
    """Score every full 1-second window in a trailing history.

    Windows are consecutive, in time order, from the start of the
    request, each smoothed and gated on its own second of samples.
    A trailing partial window (fewer than WINDOW_N samples) is
    dropped: it is less than a second of evidence. A window with
    too few tracked samples is reported as "lost" and resets the
    persistence run. Raises ValueError when no window in the
    request is usable at all.
    """
    if cue not in models:
        raise ValueError(f"no model loaded for cue {cue!r}")
    ordered = sorted(samples, key=lambda s: s.elapsed_s)
    results: list[dict] = []
    for start in range(0, len(ordered) - WINDOW_N + 1, WINDOW_N):
        chunk = ordered[start:start + WINDOW_N]
        entry = {"start_s": float(chunk[0].elapsed_s),
                 "state": "lost", "probability_tilted": None}
        try:
            scored = predict_window(models, cue, baseline_head,
                                    baseline_shoulder, chunk)
        except ValueError:
            results.append(entry)
            continue
        results.append({"start_s": float(chunk[0].elapsed_s),
                        "state": scored["state"],
                        "probability_tilted":
                            scored["probability_tilted"]})
    if not any(r["state"] != "lost" for r in results):
        raise ValueError(
            "no usable window in the request (each 1-second window "
            f"needs {MIN_TRACKED} tracked samples)")
    return results


def apply_gate(window_states: list[str], gate_s: float,
               window_s: float = WINDOW_S) -> tuple[str, int]:
    """Persistence verdict over scored windows, in time order.

    Returns (state, consecutive_tilted). The state is "tilted"
    only when the last K consecutive windows are all tilted, where
    K = ceil(gate_s / window_s). A "lost" window (tracking loss)
    resets the run, the same way tracking loss resets the timer in
    the eval harness. With gate_s 0 the last window alone decides.
    """
    k = 1 if gate_s <= 0 else max(1, int(math.ceil(gate_s / window_s)))
    run = 0
    for state in window_states:
        run = run + 1 if state == "tilted" else 0
    return ("tilted" if run >= k else "upright", run)


def create_app(models: dict | None = None,
               artifacts_dir: Path = ARTIFACTS_DIR) -> FastAPI:
    """Build the app. `models` overrides the artifact files on disk."""
    loaded = models if models is not None else load_models(artifacts_dir)
    app = FastAPI(title="PoseGuard posture classifier",
                  description="Day 13: scikit-learn tilt classifier "
                              "over webcam angle windows, serving the "
                              "gated v2 models with the 5-second "
                              "persistence gate.")

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
            windows = predict_windows(loaded, req.cue, req.baseline_head,
                                      req.baseline_shoulder, req.samples)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        if not windows:
            raise HTTPException(
                status_code=422,
                detail="no full 1-second window in the samples")
        state, run = apply_gate([w["state"] for w in windows],
                                req.gate_s)
        scored = [w for w in windows
                  if w["probability_tilted"] is not None]
        proba = scored[-1]["probability_tilted"] if scored else 0.0
        return {"cue": req.cue, "state": state,
                "probability_tilted": proba,
                "n_samples": len(req.samples),
                "model": loaded[req.cue]["model_name"],
                "gated": req.gate_s > 0, "gate_s": req.gate_s,
                "windows": windows,
                "consecutive_tilted_windows": run}

    return app


# Importing this module must not fail before the models exist (tests
# build their own in-memory models), so the ready-to-serve app is
# best-effort: it is None until training has saved the artifacts.
try:
    app = create_app()
except RuntimeError:
    app = None
