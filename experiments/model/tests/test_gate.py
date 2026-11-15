"""Tests for the Day 13 persistence gate and trailing history.

The gate is the PRD rule: a cue fires only after 5 seconds of
sustained tilt, and tracking loss resets the timer. Behaviour
tests use clean synthetic head windows, where the signal is real,
so they test the gate rather than Run 1's weak signal. A separate
smoke test runs the real v2 shoulder model on a real Run 1 hold.
"""

import pytest
from fastapi.testclient import TestClient

from experiments.eval.analysis import _hold_windows, add_deviations, align_trials
from experiments.eval.loader import load_session
from experiments.model.api import (Sample, apply_gate, create_app,
                                   load_models, predict_windows)
from experiments.model.tests.conftest import RUN1_CSV, RUN1_PRESENT
from experiments.model.tests.test_api import _samples, _synthetic_windows
from experiments.model.train import evaluate_cue


def _blocks(blocks):
    """Sample objects from (n, head_dev, shoulder_dev, tracking) blocks."""
    samples = []
    t = 0.0
    for n, hd, sd, tr in blocks:
        for _ in range(n):
            samples.append(Sample(elapsed_s=round(t, 1),
                                  head_tilt=-5.622 + hd,
                                  shoulder_tilt=174.173 + sd,
                                  tracking=tr))
            t += 0.1
    return samples


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


# --- the gate itself, no model involved ---

def test_gate_zero_uses_last_window():
    assert apply_gate(["upright", "tilted"], 0.0) == ("tilted", 1)
    assert apply_gate(["tilted", "upright"], 0.0) == ("upright", 0)


def test_gate_five_sustained_tilts_fire():
    states = ["tilted"] * 5
    assert apply_gate(states, 5.0) == ("tilted", 5)


def test_gate_four_then_upright_does_not_fire():
    states = ["tilted"] * 4 + ["upright"]
    assert apply_gate(states, 5.0) == ("upright", 0)


def test_gate_brief_tilt_does_not_fire():
    states = ["tilted", "tilted", "upright", "upright", "upright"]
    assert apply_gate(states, 5.0) == ("upright", 0)


def test_gate_tracking_loss_resets_the_run():
    states = ["tilted"] * 4 + ["lost"] + ["tilted"] * 5
    assert apply_gate(states, 5.0) == ("tilted", 5)
    states = ["tilted"] * 3 + ["lost"] + ["tilted"] * 2
    assert apply_gate(states, 5.0) == ("upright", 2)


def test_gate_fractional_seconds_round_up():
    # 2.5 s needs 3 whole windows, not 2.
    assert apply_gate(["tilted"] * 2, 2.5) == ("upright", 2)
    assert apply_gate(["tilted"] * 3, 2.5) == ("tilted", 3)


def test_gate_empty_and_all_lost():
    assert apply_gate([], 5.0) == ("upright", 0)
    assert apply_gate(["lost", "lost"], 5.0) == ("upright", 0)


# --- scoring windows from a trailing history ---

def test_windows_sustained_tilt(trained_models):
    out = predict_windows(trained_models, "head", -5.622, 174.173,
                          _blocks([(50, 6.0, 0.0, 1)]))
    assert len(out) == 5
    assert [w["state"] for w in out] == ["tilted"] * 5
    assert all(w["probability_tilted"] > 0.9 for w in out)


def test_windows_brief_tilt(trained_models):
    out = predict_windows(trained_models, "head", -5.622, 174.173,
                          _blocks([(20, 6.0, 0.0, 1), (30, 0.0, 0.0, 1)]))
    assert [w["state"] for w in out] == [
        "tilted", "tilted", "upright", "upright", "upright"]


def test_windows_tracking_loss_resets_run(trained_models):
    out = predict_windows(trained_models, "head", -5.622, 174.173,
                          _blocks([(30, 6.0, 0.0, 1), (10, 6.0, 0.0, 0),
                                   (20, 6.0, 0.0, 1)]))
    assert [w["state"] for w in out] == [
        "tilted", "tilted", "tilted", "lost", "tilted", "tilted"]
    assert out[3]["probability_tilted"] is None


def test_windows_leading_loss_then_tilt(trained_models):
    out = predict_windows(trained_models, "head", -5.622, 174.173,
                          _blocks([(10, 0.0, 0.0, 0), (50, 6.0, 0.0, 1)]))
    assert [w["state"] for w in out] == [
        "lost", "tilted", "tilted", "tilted", "tilted", "tilted"]


def test_windows_drop_partial_trailing_window(trained_models):
    out = predict_windows(trained_models, "head", -5.622, 174.173,
                          _blocks([(55, 6.0, 0.0, 1)]))
    assert len(out) == 5


def test_windows_all_lost_raises(trained_models):
    with pytest.raises(ValueError, match="no usable window"):
        predict_windows(trained_models, "head", -5.622, 174.173,
                        _blocks([(10, 0.0, 0.0, 0)]))


# --- the endpoint with trailing history ---

def _post(client, cue, samples, gate_s=None):
    body = {"cue": cue,
            "samples": [{"elapsed_s": s.elapsed_s,
                         "head_tilt": s.head_tilt,
                         "shoulder_tilt": s.shoulder_tilt,
                         "tracking": s.tracking} for s in samples]}
    if gate_s is not None:
        body["gate_s"] = gate_s
    return client.post("/predict", json=body)


def test_endpoint_gate_fires_on_sustained_tilt(client):
    r = _post(client, "head", _blocks([(50, 6.0, 0.0, 1)]), gate_s=5.0)
    assert r.status_code == 200
    body = r.json()
    assert body["state"] == "tilted"
    assert body["gated"] is True
    assert body["gate_s"] == 5.0
    assert len(body["windows"]) == 5
    assert body["consecutive_tilted_windows"] == 5
    assert body["probability_tilted"] > 0.9


def test_endpoint_gate_blocks_brief_tilt(client):
    r = _post(client, "head",
              _blocks([(20, 6.0, 0.0, 1), (30, 0.0, 0.0, 1)]), gate_s=5.0)
    assert r.status_code == 200
    body = r.json()
    assert body["state"] == "upright"
    assert body["consecutive_tilted_windows"] == 0


def test_endpoint_gate_blocks_tilt_then_loss(client):
    r = _post(client, "head",
              _blocks([(30, 6.0, 0.0, 1), (10, 6.0, 0.0, 0),
                       (20, 6.0, 0.0, 1)]), gate_s=5.0)
    assert r.status_code == 200
    body = r.json()
    assert body["state"] == "upright"
    assert body["consecutive_tilted_windows"] == 2
    assert body["windows"][3]["state"] == "lost"


def test_endpoint_no_gate_is_backward_compatible(client):
    r = _post(client, "head", _blocks([(10, 6.0, 0.0, 1)]))
    assert r.status_code == 200
    body = r.json()
    assert body["state"] == "tilted"
    assert body["gated"] is False
    assert body["gate_s"] == 0.0
    assert len(body["windows"]) == 1
    assert body["consecutive_tilted_windows"] == 1


def test_endpoint_rejects_gate_over_10s(client):
    r = _post(client, "head", _blocks([(50, 6.0, 0.0, 1)]), gate_s=11.0)
    assert r.status_code == 422


def test_endpoint_rejects_too_few_samples(client):
    r = _post(client, "head", _blocks([(5, 0.0, 0.0, 1)]))
    assert r.status_code == 422


def test_endpoint_rejects_all_lost(client):
    r = _post(client, "head", _blocks([(10, 0.0, 0.0, 0)]))
    assert r.status_code == 422


@pytest.mark.skipif(not RUN1_PRESENT, reason="Run 1 CSV not present")
def test_endpoint_real_v2_shoulder_hold_passes_gate():
    """The served v2 shoulder model on a real 5-second hold.

    Hold 3 of the Run 1 shoulder trials, 5 seconds of frames from
    1 second into the hold. All five windows score tilted, so the
    5-second gate fires. This is the end-to-end Day 13 path: real
    data, real served artifacts, real gate.
    """
    session = load_session(RUN1_CSV)
    frames = add_deviations(session)
    align = align_trials(frames, session, "shoulder")
    step_frames = frames[frames["step"] == "shoulder"]
    t0 = float(step_frames["elapsed_s"].iloc[0])
    holds = _hold_windows(t0, align["offset_s"])
    lo, _hi = holds[2]
    d = frames[(frames["step"] == "shoulder")
               & (frames["elapsed_s"] >= lo + 1.0)
               & (frames["elapsed_s"] < lo + 6.0)].sort_values("elapsed_s")
    assert len(d) >= 50
    samples = [{"elapsed_s": float(r.elapsed_s),
                "head_tilt": float(r.head_tilt),
                "shoulder_tilt": float(r.shoulder_tilt),
                "tracking": int(r.tracking)} for r in d.itertuples()]
    models = load_models()
    c = TestClient(create_app(models=models))
    r = c.post("/predict", json={"cue": "shoulder", "samples": samples,
                                 "gate_s": 5.0})
    assert r.status_code == 200
    body = r.json()
    assert body["model"] == "logistic_regression"
    assert len(body["windows"]) == 5
    assert body["state"] == "tilted"
    assert body["consecutive_tilted_windows"] == 5
