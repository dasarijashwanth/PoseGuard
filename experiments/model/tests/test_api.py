"""Tests for the FastAPI service (TestClient, no server needed).

Behaviour tests (tilt scores higher than upright, most-recent window
wins) train on clean synthetic windows, where the signal is real, so
they test the API rather than Run 1's weak signal. A separate smoke
test checks the models trained on the real Run 1 CSV actually serve.
"""

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from experiments.model.api import create_app, predict_window
from experiments.model.tests.conftest import RUN1_CSV, RUN1_PRESENT
from experiments.model.train import evaluate_cue, train_session


def _synthetic_windows(seed=0):
    """Cleanly separable windows for both cues."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(20):
        rows.append({"label": 1, "step": "head", "start_s": float(i),
                     "group": (i % 5) + 1,
                     "mean_abs_h": 6.0 + rng.normal(0, 0.3),
                     "max_abs_h": 6.5 + rng.normal(0, 0.3),
                     "mean_h": 6.0 + rng.normal(0, 0.3),
                     "std_h": 0.2, "frac_h_gt5": 1.0, "jerk_h": 0.1,
                     "std_s": 0.2, "range_s": 0.5, "jerk_s": 0.1,
                     "slope_s": 0.0, "corr_h_s": 0.0})
    for i in range(40):
        rows.append({"label": 0, "step": "work", "start_s": float(100 + i),
                     "group": (i % 5) + 1,
                     "mean_abs_h": 0.5, "max_abs_h": 1.0,
                     "mean_h": rng.normal(0, 0.5),
                     "std_h": 0.5, "frac_h_gt5": 0.0, "jerk_h": 0.3,
                     "std_s": 0.5, "range_s": 1.0, "jerk_s": 0.2,
                     "slope_s": 0.0, "corr_h_s": 0.0})
    return pd.DataFrame(rows)


def _samples(head_dev, shoulder_dev=0.0, tracking=1):
    """10 raw angle samples around the Run 1 baselines."""
    return [{"elapsed_s": round(i * 0.1, 1),
             "head_tilt": -5.622 + head_dev,
             "shoulder_tilt": 174.173 + shoulder_dev,
             "tracking": tracking} for i in range(10)]


@pytest.fixture(scope="module")
def trained_models():
    """Models trained in memory on clean synthetic windows."""
    windows = _synthetic_windows()
    models = {}
    for cue in ("head", "shoulder"):
        r = evaluate_cue(windows, cue)
        models[cue] = {"cue": cue, "model_name": r["best_model"],
                       "model": r["trained"][r["best_model"]],
                       "features": r["features"]}
    return models


@pytest.fixture(scope="module")
def client(trained_models):
    return TestClient(create_app(models=trained_models))


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert set(r.json()["models"]) == {"head", "shoulder"}


def test_predict_upright_window(client):
    r = client.post("/predict", json={
        "cue": "head", "samples": _samples(0.0)})
    assert r.status_code == 200
    body = r.json()
    assert body["state"] in ("tilted", "upright")
    assert 0.0 <= body["probability_tilted"] <= 1.0
    assert body["n_samples"] == 10
    assert body["model"] in ("logistic_regression", "random_forest")


def test_predict_strong_tilt_scores_higher_than_upright(client):
    tilted = client.post("/predict", json={
        "cue": "head", "samples": _samples(6.0)}).json()
    upright = client.post("/predict", json={
        "cue": "head", "samples": _samples(0.0)}).json()
    assert (tilted["probability_tilted"]
            > upright["probability_tilted"])


def test_predict_rejects_unknown_cue(client):
    r = client.post("/predict", json={
        "cue": "neck", "samples": _samples(0.0)})
    assert r.status_code == 422


def test_predict_rejects_too_few_samples(client):
    r = client.post("/predict", json={
        "cue": "head", "samples": _samples(0.0)[:5]})
    assert r.status_code == 422


def test_predict_rejects_mostly_lost_tracking(client):
    r = client.post("/predict", json={
        "cue": "head", "samples": _samples(0.0, tracking=0)})
    assert r.status_code == 422


def test_predict_window_is_a_pure_function(trained_models):
    out = predict_window(trained_models, "head", -5.622, 174.173,
                         [type("S", (), s)() for s in _samples(6.0)])
    assert out["cue"] == "head"
    assert out["state"] in ("tilted", "upright")
    assert out["n_samples"] == 10


def test_predict_uses_most_recent_window(client):
    # 20 samples: first 10 upright, last 10 strongly tilted.
    samples = _samples(0.0) + [
        {"elapsed_s": round(1.0 + i * 0.1, 1), "head_tilt": 0.378,
         "shoulder_tilt": 174.173, "tracking": 1} for i in range(10)]
    r = client.post("/predict", json={"cue": "head", "samples": samples})
    assert r.status_code == 200
    upright_only = client.post("/predict", json={
        "cue": "head", "samples": _samples(0.0)}).json()
    assert (r.json()["probability_tilted"]
            > upright_only["probability_tilted"])


def test_shoulder_endpoint_serves_without_level_features(client):
    # A big raw shoulder angle alone must not flip the shoulder model:
    # it never sees the level, only dynamics. A flat window at +20 deg
    # shoulder deviation should score near the upright window.
    flat_big = client.post("/predict", json={
        "cue": "shoulder", "samples": _samples(0.0, shoulder_dev=20.0)}).json()
    flat_zero = client.post("/predict", json={
        "cue": "shoulder", "samples": _samples(0.0)}).json()
    assert abs(flat_big["probability_tilted"]
               - flat_zero["probability_tilted"]) < 0.2


@pytest.mark.skipif(not RUN1_PRESENT, reason="Run 1 CSV not present")
def test_real_models_serve():
    """The models trained on the real Run 1 CSV load and answer."""
    result = train_session(RUN1_CSV)
    models = {cue: {"cue": cue,
                    "model_name": result["cues"][cue]["best_model"],
                    "model": result["_trained"][cue][
                        result["cues"][cue]["best_model"]],
                    "features": result["cues"][cue]["features"]}
              for cue in ("head", "shoulder")}
    c = TestClient(create_app(models=models))
    assert c.get("/health").status_code == 200
    for cue in ("head", "shoulder"):
        r = c.post("/predict", json={"cue": cue, "samples": _samples(0.0)})
        assert r.status_code == 200
        assert 0.0 <= r.json()["probability_tilted"] <= 1.0
