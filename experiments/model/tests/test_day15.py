"""Day 15 tests: head cue rework, rule grid plus model comparison.

The old-feature head rows must reproduce the Day 13 served-model
numbers exactly; the rule grid's best configuration is pinned; no
model passes the gate on either feature set; and the Day 15 report
section must append without touching the Day 8, Day 11, and Day 12
sections.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.model.day15 import DAY15_MARKER, run_day15, write_report

HERE = Path(__file__).resolve().parent.parent
RUN1_CSV = Path(
    "~/workspace/user/files/poseguard-validation-2026-11-03T19-24-30-709Z.csv"
).expanduser()


@pytest.fixture(scope="module")
def day15_result():
    if not RUN1_CSV.exists():
        pytest.skip("Run 1 CSV not present")
    return run_day15(str(RUN1_CSV))


def test_rule_grid_has_all_configurations(day15_result):
    grid = day15_result["rule_grid"]
    assert len(grid["rows"]) == 5 * 3 * 4  # thresholds x persists x caps
    assert grid["alignment_verified"] is True


def test_rule_grid_best_configuration(day15_result):
    best = day15_result["rule_grid"]["best"]
    assert best == {"threshold": 4, "persist_s": 2.0,
                    "vel_cap_deg_s": 10.0, "detected": 2,
                    "trials_total": 5, "false_positives": 3,
                    "precision": 0.4, "recall": 0.4, "f1": 0.4}


def test_rule_grid_no_configuration_passes_the_gate(day15_result):
    for row in day15_result["rule_grid"]["rows"]:
        assert not (row["detected"] >= 4 and row["false_positives"] == 0)


def test_hold_dwells_show_the_fifth_hold_is_barely_recorded(
        day15_result):
    dwells = day15_result["rule_grid"]["hold_dwells"]
    assert dwells[4]["samples"] == 6
    assert dwells[4]["dwell_gt4_s"] == 0.0
    assert dwells[2]["dwell_gt4_s"] == 2.2
    assert dwells[0]["max_abs_dev"] == 7.05


def test_old_head_lr_reproduces_day13(day15_result):
    rows = day15_result["model_comparison"]["feature_sets"]["old"]["rows"]
    lr = next(r for r in rows if r["model"] == "logistic_regression")
    assert lr["f1_tilted"] == 0.078
    assert lr["detected_trials"] == 1
    assert lr["false_positive_events"] == 6


def test_new_head_features_move_lr_and_mlp_a_little(day15_result):
    sets = day15_result["model_comparison"]["feature_sets"]
    new_rows = sets["new"]["rows"]
    lr = next(r for r in new_rows if r["model"] == "logistic_regression")
    assert lr["f1_tilted"] == 0.097
    assert lr["detected_trials"] == 2
    assert lr["false_positive_events"] == 4
    mlp = next(r for r in new_rows if r["model"] == "mlp")
    assert mlp["f1_tilted"] == 0.105
    assert mlp["detected_trials"] == 2
    assert mlp["false_positive_events"] == 3


def test_no_model_passes_the_gate_on_either_feature_set(day15_result):
    sets = day15_result["model_comparison"]["feature_sets"]
    for name in ("old", "new"):
        for row in sets[name]["rows"]:
            assert not (row["detected_trials"] >= 4
                        and row["false_positive_events"] == 0)


def test_results_day15_json_schema():
    data = json.loads((HERE / "results_day15.json").read_text(
        encoding="utf-8"))
    assert data["cue"] == "head"
    assert data["n_windows"] == 383
    assert data["n_tilted_windows"] == 23
    assert len(data["rule_grid"]["rows"]) == 60
    comp = data["model_comparison"]
    assert comp["old_features"] == [
        "mean_abs_h", "max_abs_h", "mean_h", "std_h",
        "frac_h_gt5", "jerk_h"]
    assert comp["new_features"] == comp["old_features"] + [
        "vel_h", "dwell_h"]
    for name in ("old", "new"):
        rows = comp["feature_sets"][name]["rows"]
        assert [r["model"] for r in rows] == [
            "logistic_regression", "random_forest", "mlp"]


def test_write_report_keeps_earlier_sections_and_is_idempotent(tmp_path):
    data = json.loads((HERE / "results_day15.json").read_text(
        encoding="utf-8"))
    target = tmp_path / "BENCHMARK.md"
    target.write_text((HERE / "BENCHMARK.md").read_text(
        encoding="utf-8"), encoding="utf-8")
    write_report(data, target)
    text = target.read_text(encoding="utf-8")
    assert "# Day 8 benchmark" in text
    assert "## Day 11" in text
    assert "## Day 12" in text
    assert text.count(DAY15_MARKER) == 1
    write_report(data, target)
    assert target.read_text(encoding="utf-8").count(DAY15_MARKER) == 1
