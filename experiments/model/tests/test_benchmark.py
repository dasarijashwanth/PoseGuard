"""Day 12 benchmark tests: RF + MLP on the gated signal, old vs new.

Old-feature rows must reproduce the Day 8 numbers exactly; the new
shoulder logistic regression row must match the Day 11 comparison;
the committed benchmark.json must hold the new feature-set schema;
and the Day 12 report section must append without touching the
Day 8 and Day 11 sections.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.eval.signals import WINNER_COL, add_signals
from experiments.model.benchmark import DAY12_MARKER, write_report
from experiments.model.features import (
    FEATURES,
    OLD_FEATURES,
    OLD_SHOULDER_FEATURES,
    SHOULDER_FEATURES,
    build_windows,
)
from experiments.model.torch_mlp import evaluate_mlp_cue
from experiments.model.train import evaluate_cue

HERE = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def run1_windows(run1_frames):
    """1-second windows with the Day 11 gated signal attached."""
    frames, session = run1_frames
    frames = add_signals(frames, session)
    frames["s_dev_new"] = frames[WINNER_COL]
    return {cue: build_windows(frames, session, cue)
            for cue in ("head", "shoulder")}


def test_old_shoulder_features_are_the_day8_list():
    assert OLD_SHOULDER_FEATURES == [
        "std_s", "range_s", "jerk_s", "slope_s",
        "corr_h_s", "std_h", "mean_abs_h", "jerk_h"]
    assert set(SHOULDER_FEATURES) - set(OLD_SHOULDER_FEATURES) == {
        "mean_abs_sn", "max_abs_sn", "std_sn"}
    assert OLD_FEATURES["head"] == FEATURES["head"]


def test_old_shoulder_lr_reproduces_day8(run1_windows):
    lr = evaluate_cue(run1_windows["shoulder"], "shoulder",
                      features=OLD_SHOULDER_FEATURES
                      )["models"]["logistic_regression"]
    assert lr["oof_metrics"]["accuracy"] == 0.548
    assert lr["oof_metrics"]["f1_tilted"] == 0.179
    assert lr["oof_trials"]["detected_trials"] == 5
    assert lr["oof_trials"]["false_positive_events"] == 5
    assert lr["oof_trials"]["work_windows_tilted"] == 134


def test_old_shoulder_rf_reproduces_day8(run1_windows):
    rf = evaluate_cue(run1_windows["shoulder"], "shoulder",
                      features=OLD_SHOULDER_FEATURES
                      )["models"]["random_forest"]
    assert rf["oof_metrics"]["accuracy"] == 0.925
    assert rf["oof_metrics"]["f1_tilted"] == 0.0
    assert rf["oof_trials"]["detected_trials"] == 0
    assert rf["oof_trials"]["false_positive_events"] == 0
    assert rf["oof_trials"]["work_windows_tilted"] == 0


def test_new_shoulder_lr_matches_day11(run1_windows):
    lr = evaluate_cue(run1_windows["shoulder"], "shoulder",
                      features=FEATURES["shoulder"]
                      )["models"]["logistic_regression"]
    assert lr["oof_metrics"]["accuracy"] == 0.610
    assert lr["oof_metrics"]["f1_tilted"] == 0.219
    assert lr["oof_trials"]["detected_trials"] == 5
    assert lr["oof_trials"]["false_positive_events"] == 3
    assert lr["oof_trials"]["work_windows_tilted"] == 109
    assert lr["calm_only_metrics"]["accuracy"] == 0.585
    # The stillness artifact persists: the gated coefficients are
    # negative, so the model still spots quiet, not tilt.
    assert lr["coefficients"]["mean_abs_sn"] < 0
    assert lr["coefficients"]["max_abs_sn"] < 0


def test_new_shoulder_rf_uses_gated_features_but_detects_nothing(
        run1_windows):
    rf = evaluate_cue(run1_windows["shoulder"], "shoulder",
                      features=FEATURES["shoulder"]
                      )["models"]["random_forest"]
    assert rf["oof_metrics"]["f1_tilted"] == 0.0
    assert rf["oof_trials"]["detected_trials"] == 0
    assert rf["oof_trials"]["work_windows_tilted"] == 0
    gated = sum(v for k, v in rf["feature_importance"].items()
                if k.endswith("_sn"))
    assert gated > 0.4


def test_new_shoulder_mlp_runs_and_reports_calm(run1_windows):
    res = evaluate_mlp_cue(run1_windows["shoulder"], "shoulder",
                           features=FEATURES["shoulder"])
    m = res["models"]["mlp"]
    assert res["features"] == FEATURES["shoulder"]
    assert m["oof_metrics"]["f1_tilted"] == 0.215
    assert m["oof_trials"]["detected_trials"] == 4
    assert m["oof_trials"]["false_positive_events"] == 3
    assert m["oof_trials"]["work_windows_tilted"] == 82
    assert res["calm_only_metrics"]["accuracy"] == 0.638


def test_benchmark_json_schema_and_day8_reproduction():
    data = json.loads((HERE / "benchmark.json").read_text(
        encoding="utf-8"))
    for cue in ("head", "shoulder"):
        sets = data["cues"][cue]["feature_sets"]
        for name in ("old", "new"):
            rows = sets[name]["rows"]
            assert [r["model"] for r in rows] == [
                "logistic_regression", "random_forest", "mlp"]
            for r in rows:
                for key in ("accuracy", "precision_tilted",
                            "recall_tilted", "f1_tilted",
                            "detected_trials", "false_positive_events",
                            "work_windows_tilted", "work_windows_total",
                            "calm_accuracy"):
                    assert key in r
        old_lr = next(r for r in sets["old"]["rows"]
                      if r["model"] == "logistic_regression")
        new_lr = next(r for r in sets["new"]["rows"]
                      if r["model"] == "logistic_regression")
        if cue == "shoulder":
            assert (old_lr["accuracy"], old_lr["f1_tilted"],
                    old_lr["detected_trials"],
                    old_lr["false_positive_events"]) == (0.548, 0.179, 5, 5)
            assert (new_lr["accuracy"], new_lr["f1_tilted"],
                    new_lr["detected_trials"],
                    new_lr["false_positive_events"]) == (0.610, 0.219, 5, 3)
        else:
            assert sets["old"]["rows"] == sets["new"]["rows"]


def test_write_report_keeps_day8_and_day11_and_is_idempotent(tmp_path):
    data = json.loads((HERE / "benchmark.json").read_text(
        encoding="utf-8"))
    target = tmp_path / "BENCHMARK.md"
    target.write_text((HERE / "BENCHMARK.md").read_text(
        encoding="utf-8"), encoding="utf-8")
    write_report(data, target)
    text = target.read_text(encoding="utf-8")
    assert "# Day 8 benchmark" in text
    assert "## Day 11" in text
    assert text.count(DAY12_MARKER) == 1
    write_report(data, target)
    assert target.read_text(encoding="utf-8").count(DAY12_MARKER) == 1
