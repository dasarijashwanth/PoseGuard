"""Tests for the Day 6 MLflow tracking (tracking.py) and the
--model wiring in train.py. MLflow runs go to a tmp tracking dir,
never the real experiments/model/mlruns/ store.
"""

import json

import mlflow
import pytest
from mlflow.tracking import MlflowClient

from experiments.model.features import build_windows
from experiments.model.tracking import (EXPERIMENT_NAME, log_training_run,
                                       summarize_runs, tracking_uri,
                                       write_comparison)
from experiments.model.train import evaluate_cue


def _fake_result():
    """Minimal train_session-shaped result, no training needed."""
    cue_result = {
        "best_model": "logistic_regression",
        "n_windows": 100,
        "n_tilted_windows": 10,
        "features": ["head_mean", "head_max"],
        "best_oof_metrics": {"accuracy": 0.9, "precision_tilted": 0.5,
                             "recall_tilted": 0.4, "f1_tilted": 0.44},
        "best_oof_trials": {"detected_trials": 2,
                            "false_positive_events": 3,
                            "work_windows_tilted": 12},
        "calm_only_metrics": {"accuracy": 0.85},
    }
    return {"cues": {"head": dict(cue_result),
                     "shoulder": dict(cue_result)}}


def test_log_run_records_params_and_metrics(tmp_path):
    run_id = log_training_run(_fake_result(), "test_run",
                              tracking_dir=tmp_path)
    client = MlflowClient(tracking_uri=tracking_uri(tmp_path))
    run = client.get_run(run_id)
    assert run.data.params["head_model"] == "logistic_regression"
    assert run.data.params["shoulder_model"] == "logistic_regression"
    assert run.data.metrics["head_f1_tilted"] == pytest.approx(0.44)
    assert run.data.metrics["head_trials_detected"] == 2
    assert run.data.metrics["shoulder_false_positive_events"] == 3


def test_summarize_runs_lists_both_runs(tmp_path):
    log_training_run(_fake_result(), "run_a", tracking_dir=tmp_path)
    other = _fake_result()
    other["cues"]["head"]["best_model"] = "random_forest"
    log_training_run(other, "run_b", tracking_dir=tmp_path)
    runs = summarize_runs(tmp_path)
    assert {r["run_name"] for r in runs} == {"run_a", "run_b"}
    by_name = {r["run_name"]: r for r in runs}
    assert by_name["run_b"]["params"]["head_model"] == "random_forest"


def test_write_comparison_writes_json(tmp_path):
    log_training_run(_fake_result(), "run_a", tracking_dir=tmp_path)
    out = tmp_path / "comparison.json"
    comparison = write_comparison(tmp_path, out=out)
    assert comparison["experiment"] == EXPERIMENT_NAME
    assert len(comparison["runs"]) == 1
    assert json.loads(out.read_text(encoding="utf-8")) == comparison


def test_summarize_empty_store_is_empty(tmp_path):
    assert summarize_runs(tmp_path) == []


def test_forced_model_choice_is_honoured(run1_frames):
    """--model random_forest trains only the forest (Day 6 wiring)."""
    frames, session = run1_frames
    windows = build_windows(frames, session, "head")
    r = evaluate_cue(windows, "head", model_choice="random_forest")
    assert r["best_model"] == "random_forest"
    assert set(r["models"]) == {"random_forest"}


def test_unknown_model_choice_rejected(run1_frames):
    frames, session = run1_frames
    windows = build_windows(frames, session, "head")
    with pytest.raises(ValueError, match="unknown model"):
        evaluate_cue(windows, "head", model_choice="xgboost")
