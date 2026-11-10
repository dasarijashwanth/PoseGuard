"""Tests for the Day 8 PyTorch MLP classifier."""

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.model_selection import GroupKFold

from experiments.model.features import FEATURES, build_windows
from experiments.model.torch_mlp import (
    MLP,
    class_weights,
    evaluate_mlp_cue,
    train_one_model,
)


def _synthetic_windows(seed=0, n_tilted=20, n_upright=40):
    """Cleanly separable windows: tilted ones sit at +6 deg head dev."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_tilted):
        rows.append({"label": 1, "step": "head", "start_s": float(i),
                     "group": (i % 5) + 1,
                     "mean_abs_h": 6.0 + rng.normal(0, 0.3),
                     "max_abs_h": 6.5 + rng.normal(0, 0.3),
                     "mean_h": 6.0 + rng.normal(0, 0.3),
                     "std_h": 0.2 + abs(rng.normal(0, 0.1)),
                     "frac_h_gt5": 1.0,
                     "jerk_h": 0.1 + abs(rng.normal(0, 0.05)),
                     "std_s": 0.2, "range_s": 0.5, "jerk_s": 0.1,
                     "slope_s": 0.0, "corr_h_s": 0.0})
    for i in range(n_upright):
        rows.append({"label": 0, "step": "work", "start_s": float(100 + i),
                     "group": (i % 5) + 1,
                     "mean_abs_h": 0.5 + abs(rng.normal(0, 0.3)),
                     "max_abs_h": 1.0 + abs(rng.normal(0, 0.3)),
                     "mean_h": rng.normal(0, 0.5),
                     "std_h": 0.5 + abs(rng.normal(0, 0.2)),
                     "frac_h_gt5": 0.0,
                     "jerk_h": 0.3 + abs(rng.normal(0, 0.1)),
                     "std_s": 0.5, "range_s": 1.0, "jerk_s": 0.2,
                     "slope_s": 0.0, "corr_h_s": 0.0})
    return pd.DataFrame(rows)


def test_forward_pass_shapes():
    torch.manual_seed(0)
    model = MLP(n_features=6)
    model.eval()
    x = torch.randn(8, 6)
    with torch.no_grad():
        logits = model(x)
    assert logits.shape == (8, 2)
    probs = torch.softmax(logits, dim=1)
    assert torch.allclose(probs.sum(dim=1), torch.ones(8), atol=1e-5)
    # shoulder cue: 8 features in, same 2-class output
    model_s = MLP(n_features=8)
    assert model_s(torch.randn(4, 8)).shape == (4, 2)


def test_class_weights_come_from_train_only():
    y = np.array([0] * 90 + [1] * 10)
    w = class_weights(y)
    assert float(w[0]) == pytest.approx(100 / 180, rel=1e-5)
    assert float(w[1]) == pytest.approx(100 / 20, rel=1e-5)
    assert float(w[1]) > float(w[0])  # rare tilted class counts more
    with pytest.raises(ValueError):
        class_weights(np.zeros(10, dtype=int))  # no tilted class


def test_training_runs_on_tiny_synthetic_data():
    rng = np.random.default_rng(1)
    X = rng.normal(0, 1, size=(60, 6))
    X[:20] += 3.0  # first 20 are the tilted class
    y = np.array([1] * 20 + [0] * 40)
    model, hist = train_one_model(X[:40], y[:40], X[40:], y[40:], seed=42)
    assert hist["epochs_run"] >= 1
    assert np.isfinite(hist["best_val_loss"])
    model.eval()
    with torch.no_grad():
        pred = torch.argmax(
            model(torch.tensor(X, dtype=torch.float32)), dim=1).numpy()
    assert (pred == y).mean() >= 0.8


def test_mlp_learns_clean_signal():
    w = _synthetic_windows()
    r = evaluate_mlp_cue(w, "head")
    assert r["best_oof_metrics"]["accuracy"] >= 0.9
    assert r["best_oof_metrics"]["recall_tilted"] >= 0.8
    assert r["best_oof_trials"]["detected_trials"] == 5


def test_mlp_evaluation_is_deterministic():
    w = _synthetic_windows()
    r1 = evaluate_mlp_cue(w, "head")
    r2 = evaluate_mlp_cue(w, "head")
    assert (r1["models"]["mlp"]["oof_pred"]
            == r2["models"]["mlp"]["oof_pred"])
    assert (r1["models"]["mlp"]["oof_proba"]
            == r2["models"]["mlp"]["oof_proba"])


def test_cv_folds_never_share_a_trial_group():
    w = _synthetic_windows()
    groups = w["group"].to_numpy()
    y = w["label"].to_numpy()
    cv = GroupKFold(n_splits=5)
    for tr_idx, te_idx in cv.split(w, y, groups):
        assert not (set(groups[tr_idx]) & set(groups[te_idx]))


def test_shoulder_mlp_uses_no_shoulder_level():
    level_names = {"mean_s", "max_abs_s", "mean_abs_s", "frac_s_gt6"}
    assert not (set(FEATURES["shoulder"]) & level_names)
    w = _synthetic_windows()
    r = evaluate_mlp_cue(w, "shoulder")
    assert not (set(r["features"]) & level_names)
    assert "std_s" in r["features"]  # dynamics are allowed


def test_mlp_metrics_structure():
    w = _synthetic_windows()
    r = evaluate_mlp_cue(w, "head")
    m = r["best_oof_metrics"]
    for key in ("accuracy", "precision_tilted", "recall_tilted",
                "f1_tilted", "precision_upright", "recall_upright"):
        assert 0.0 <= m[key] <= 1.0
    assert m["confusion"]["tp"] + m["confusion"]["fn"] == m["n_tilted"]
    assert m["confusion"]["tn"] + m["confusion"]["fp"] == m["n_upright"]
    assert r["rule_baseline"] == {"detected_trials": 0,
                                  "false_positive_events": 0}
    assert "calm_only_metrics" in r and "calm_windows" in r
    assert "threshold_sweep" in r
    assert len(r["models"]["mlp"]["oof_pred"]) == len(w)


def test_real_data_mlp_completes(run1_frames):
    frames, session = run1_frames
    for cue in ("head", "shoulder"):
        windows = build_windows(frames, session, cue)
        r = evaluate_mlp_cue(windows, cue)
        assert r["n_windows"] == len(windows)
        assert r["best_model"] == "mlp"
        assert len(r["models"]["mlp"]["oof_pred"]) == len(windows)
        assert len(r["models"]["mlp"]["fold_histories"]) == 5
