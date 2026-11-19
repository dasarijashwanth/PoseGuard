"""Day 17 tests: shoulder cue rework, rule grid plus model comparison.

The old-feature shoulder rows must reproduce the Day 12 numbers
exactly; the rule grid's best configuration is pinned; no model
passes the gate on either feature set; and the Day 17 report
section must append without touching the Day 8, Day 11, Day 12,
Day 15, and Day 16 sections.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.model.day17 import DAY17_MARKER, run_day17, write_report

HERE = Path(__file__).resolve().parent.parent
RUN1_CSV = Path(
    "~/workspace/user/files/poseguard-validation-2026-11-03T19-24-30-709Z.csv"
).expanduser()


@pytest.fixture(scope="module")
def day17_result():
    if not RUN1_CSV.exists():
        pytest.skip("Run 1 CSV not present")
    return run_day17(str(RUN1_CSV))


def test_rule_grid_has_all_configurations(day17_result):
    grid = day17_result["rule_grid"]
    assert len(grid["rows"]) == 6 * 3 * 4  # thresholds x persists x caps
    # Run 1 recorded no shoulder trial summary: the alignment is
    # inferred from the head alignment, not verified.
    assert grid["alignment_verified"] is False
    assert grid["alignment_offset_s"] == 9.5


def test_rule_grid_best_configuration(day17_result):
    best = day17_result["rule_grid"]["best"]
    assert best == {"threshold": 1, "persist_s": 5.0,
                    "vel_cap_deg_s": 5.0, "detected": 4,
                    "trials_total": 5, "false_positives": 8,
                    "precision": 0.333, "recall": 0.8, "f1": 0.471}


def test_rule_grid_no_configuration_passes_the_gate(day17_result):
    for row in day17_result["rule_grid"]["rows"]:
        assert not (row["detected"] >= 4 and row["false_positives"] == 0)
    # 16 configurations reach 4/5 trials, all at 1-2 deg, and the
    # cheapest still carries 8 false positives.
    reached = [r for r in day17_result["rule_grid"]["rows"]
               if r["detected"] >= 4]
    assert len(reached) == 16
    assert min(r["false_positives"] for r in reached) == 8


def test_rule_grid_reproduces_the_day3_baseline(day17_result):
    rows = day17_result["rule_grid"]["rows"]
    baseline = next(r for r in rows
                    if r["threshold"] == 6 and r["persist_s"] == 5.0
                    and r["vel_cap_deg_s"] is None)
    assert baseline["detected"] == 0
    assert baseline["false_positives"] == 7


def test_velocity_cap_cuts_false_positives_at_the_best_threshold(
        day17_result):
    rows = {(r["threshold"], r["persist_s"], r["vel_cap_deg_s"]): r
            for r in day17_result["rule_grid"]["rows"]}
    uncapped = rows[(1, 5.0, None)]
    capped = rows[(1, 5.0, 5.0)]
    assert uncapped["false_positives"] == 19
    assert capped["false_positives"] == 8
    assert capped["detected"] == uncapped["detected"] == 4


def test_hold_dwells_show_the_fifth_hold_is_barely_recorded(
        day17_result):
    dwells = day17_result["rule_grid"]["hold_dwells"]
    assert dwells[4]["samples"] == 8
    assert dwells[4]["dwell_gt2_s"] == 0.0
    assert dwells[0]["dwell_gt2_s"] == 5.9
    assert dwells[0]["max_abs_dev"] == 4.33
    # Trial 3 carries a tracker spike inside the hold itself.
    assert dwells[2]["max_abs_dev"] == 58.23
    assert dwells[2]["mean_abs_dev"] == 7.05


def test_old_shoulder_lr_reproduces_day12(day17_result):
    rows = day17_result["model_comparison"]["feature_sets"]["old"]["rows"]
    lr = next(r for r in rows if r["model"] == "logistic_regression")
    assert lr["f1_tilted"] == 0.219
    assert lr["detected_trials"] == 5
    assert lr["false_positive_events"] == 3
    mlp = next(r for r in rows if r["model"] == "mlp")
    assert mlp["f1_tilted"] == 0.215
    assert mlp["detected_trials"] == 4
    assert mlp["false_positive_events"] == 3


def test_new_shoulder_features_make_the_models_worse(day17_result):
    sets = day17_result["model_comparison"]["feature_sets"]
    new_rows = sets["new"]["rows"]
    lr = next(r for r in new_rows if r["model"] == "logistic_regression")
    assert lr["f1_tilted"] == 0.172
    assert lr["detected_trials"] == 3
    assert lr["false_positive_events"] == 3
    mlp = next(r for r in new_rows if r["model"] == "mlp")
    assert mlp["f1_tilted"] == 0.172
    assert mlp["detected_trials"] == 3
    assert mlp["false_positive_events"] == 4
    # The new coefficients contradict the story the features were
    # built to tell: more velocity and dwell score less tilted.
    coefs = sets["new"]["lr_coefficients"]
    assert coefs["vel_s"] < 0
    assert coefs["dwell_s"] < 0


def test_no_model_passes_the_gate_on_either_feature_set(day17_result):
    sets = day17_result["model_comparison"]["feature_sets"]
    for name in ("old", "new"):
        for row in sets[name]["rows"]:
            assert not (row["detected_trials"] >= 4
                        and row["false_positive_events"] == 0)


def test_results_day17_json_schema():
    data = json.loads((HERE / "results_day17.json").read_text(
        encoding="utf-8"))
    assert data["cue"] == "shoulder"
    assert data["n_windows"] == 385
    assert data["n_tilted_windows"] == 29
    assert len(data["rule_grid"]["rows"]) == 72
    comp = data["model_comparison"]
    assert comp["old_features"] == [
        "std_s", "range_s", "jerk_s", "slope_s", "corr_h_s", "std_h",
        "mean_abs_h", "jerk_h", "mean_abs_sn", "max_abs_sn", "std_sn"]
    assert comp["new_features"] == comp["old_features"] + [
        "vel_s", "dwell_s"]
    for name in ("old", "new"):
        rows = comp["feature_sets"][name]["rows"]
        assert [r["model"] for r in rows] == [
            "logistic_regression", "random_forest", "mlp"]


def test_write_report_keeps_earlier_sections_and_is_idempotent(tmp_path):
    data = json.loads((HERE / "results_day17.json").read_text(
        encoding="utf-8"))
    target = tmp_path / "BENCHMARK.md"
    target.write_text((HERE / "BENCHMARK.md").read_text(
        encoding="utf-8"), encoding="utf-8")
    write_report(data, target)
    text = target.read_text(encoding="utf-8")
    assert "# Day 8 benchmark" in text
    assert "## Day 11" in text
    assert "## Day 12" in text
    assert "## Day 15" in text
    assert "## Day 16" in text
    assert text.count(DAY17_MARKER) == 1
    write_report(data, target)
    assert target.read_text(encoding="utf-8").count(DAY17_MARKER) == 1
