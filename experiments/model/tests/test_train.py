"""Tests for classifier training and evaluation."""

import numpy as np
import pandas as pd
import pytest

from experiments.model.features import FEATURES, build_windows
from experiments.model.train import evaluate_cue, make_models


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


def test_models_are_deterministic():
    w = _synthetic_windows()
    r1 = evaluate_cue(w, "head")
    r2 = evaluate_cue(w, "head")
    assert r1["best_model"] == r2["best_model"]
    for name in r1["models"]:
        assert (r1["models"][name]["oof_pred"]
                == r2["models"][name]["oof_pred"])


def test_clean_signal_is_learned():
    w = _synthetic_windows()
    r = evaluate_cue(w, "head")
    assert r["best_oof_metrics"]["accuracy"] >= 0.95
    assert r["best_oof_metrics"]["recall_tilted"] >= 0.9
    assert r["best_oof_trials"]["detected_trials"] == 5


def test_metrics_structure():
    w = _synthetic_windows()
    r = evaluate_cue(w, "head")
    m = r["best_oof_metrics"]
    for key in ("accuracy", "precision_tilted", "recall_tilted",
                "f1_tilted", "precision_upright", "recall_upright"):
        assert 0.0 <= m[key] <= 1.0
    assert m["confusion"]["tp"] + m["confusion"]["fn"] == m["n_tilted"]
    assert m["confusion"]["tn"] + m["confusion"]["fp"] == m["n_upright"]
    assert r["rule_baseline"] == {"detected_trials": 0,
                                  "false_positive_events": 0}
    assert "calm_only_metrics" in r and "calm_windows" in r


def test_trial_verdicts_count_false_positive_runs():
    w = _synthetic_windows()
    r = evaluate_cue(w, "head")
    t = r["best_oof_trials"]
    assert t["trials_total"] == 5
    assert len(t["per_trial"]) == 5
    assert t["false_positive_events"] == 0  # clean signal, no FP runs


def test_training_is_deterministic_on_real_data(run1_frames):
    frames, session = run1_frames
    windows = build_windows(frames, session, "head")
    r1 = evaluate_cue(windows, "head")
    r2 = evaluate_cue(windows, "head")
    assert r1["best_model"] == r2["best_model"]
    assert (r1["models"][r1["best_model"]]["oof_pred"]
            == r2["models"][r2["best_model"]]["oof_pred"])


def test_real_data_evaluation_completes(run1_frames):
    frames, session = run1_frames
    for cue in ("head", "shoulder"):
        windows = build_windows(frames, session, cue)
        r = evaluate_cue(windows, cue)
        assert r["n_windows"] == len(windows)
        assert r["best_model"] in ("logistic_regression", "random_forest")
        # out-of-fold predictions exist for every window
        assert len(r["models"][r["best_model"]]["oof_pred"]) == len(windows)
