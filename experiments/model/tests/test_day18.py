"""Day 18 tests: Streamlit demo showcase data and verdict logic.

The demo must never oversell: every headline number is pinned to
results_day15.json / results_day17.json, the sample CSV must load
cleanly, and no scoreboard row may pass the gate.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from demo.demo_data import (
    BEST_RULES,
    SAMPLE_COLUMNS,
    SCOREBOARD,
    VERDICT,
    best_rule,
    gate_verdict,
    load_sample,
    sample_stats,
    scoreboard_frame,
    trials_frame,
)

HERE = Path(__file__).resolve().parent.parent.parent.parent
SAMPLE_CSV = HERE / "demo" / "data" / "run1_sample.csv"


# --- sample CSV ------------------------------------------------------


def test_sample_loads_with_expected_shape():
    df = load_sample(SAMPLE_CSV)
    assert list(df.columns) == SAMPLE_COLUMNS
    assert len(df) == 727  # 1 Hz downsample of the 7,820-frame Run 1
    assert df["elapsed_s"].is_monotonic_increasing
    assert df["elapsed_s"].iloc[0] == pytest.approx(11.55, abs=0.01)
    assert df["elapsed_s"].iloc[-1] == pytest.approx(737.07, abs=0.01)


def test_sample_steps_are_known():
    df = load_sample(SAMPLE_CSV)
    assert set(df["step"].unique()) <= {"calib", "head", "shoulder", "work", "loss", "results"}
    assert {"head", "shoulder", "work"} <= set(df["step"].unique())


def test_sample_deviations_within_physical_bounds():
    # Circular deviations, so nothing wraps to +/-350. The extremes are
    # the tracker meltdown at 447 s, not posture: head -94.95 deg and
    # shoulder +141.74 deg in the same second, mid-work.
    df = load_sample(SAMPLE_CSV)
    assert df["head_dev_deg"].abs().max() < 180
    assert df["shoulder_dev_deg"].abs().max() < 180
    assert df["head_dev_deg"].abs().max() == pytest.approx(94.95, abs=0.01)
    assert df["shoulder_dev_deg"].abs().max() == pytest.approx(141.74, abs=0.01)
    meltdown = df[(df["elapsed_s"] - 447.06).abs() < 0.6]
    assert len(meltdown) == 1
    assert meltdown.iloc[0]["step"] == "work"


def test_sample_stats_match_the_file():
    stats = sample_stats(load_sample(SAMPLE_CSV))
    assert stats["rows"] == 727
    assert stats["duration_s"] == pytest.approx(725.52, abs=0.1)
    assert stats["max_abs_head_dev"] == pytest.approx(94.95, abs=0.01)
    assert stats["max_abs_shoulder_dev"] == pytest.approx(141.74, abs=0.01)


def test_load_sample_missing_file_raises():
    with pytest.raises(ValueError, match="not found"):
        load_sample(HERE / "demo" / "data" / "nope.csv")


def test_load_sample_wrong_columns_raises(tmp_path):
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"a": [1.0]}).to_csv(bad, index=False)
    with pytest.raises(ValueError, match="expected columns"):
        load_sample(bad)


def test_load_sample_non_monotonic_time_raises(tmp_path):
    bad = tmp_path / "bad.csv"
    pd.DataFrame(
        {
            "elapsed_s": [2.0, 1.0],
            "step": ["work", "work"],
            "head_dev_deg": [0.0, 0.0],
            "shoulder_dev_deg": [0.0, 0.0],
        }
    ).to_csv(bad, index=False)
    with pytest.raises(ValueError, match="monotonic"):
        load_sample(bad)


# --- gate verdicts ---------------------------------------------------


def test_gate_verdict_best_shoulder_rule_fails_on_false_positives():
    v = gate_verdict(4, 8)
    assert v["passes"] is False
    assert "8 false positives" in v["reason"]
    assert "false-positive half" in v["reason"]


def test_gate_verdict_passes_only_at_4_of_5_and_zero_fp():
    assert gate_verdict(4, 0)["passes"] is True
    assert gate_verdict(5, 0)["passes"] is True
    assert gate_verdict(3, 0)["passes"] is False
    assert gate_verdict(4, 1)["passes"] is False


def test_gate_verdict_names_the_failing_half():
    assert "detection half" in gate_verdict(2, 0)["reason"]
    assert "both halves" in gate_verdict(2, 3)["reason"]


# --- best rules pinned to the results files --------------------------


def test_best_shoulder_rule_is_pinned():
    rule = best_rule("shoulder")
    assert rule["threshold_deg"] == 1
    assert rule["persist_s"] == 5.0
    assert rule["vel_cap_deg_s"] == 5.0
    assert rule["detected"] == 4
    assert rule["false_positives"] == 8
    assert rule["f1"] == pytest.approx(0.471)
    assert gate_verdict(rule["detected"], rule["false_positives"])["passes"] is False


def test_best_head_rule_is_pinned():
    rule = best_rule("head")
    assert rule["threshold_deg"] == 4
    assert rule["persist_s"] == 2.0
    assert rule["vel_cap_deg_s"] == 10.0
    assert rule["detected"] == 2
    assert rule["false_positives"] == 3
    assert rule["f1"] == pytest.approx(0.400)
    assert gate_verdict(rule["detected"], rule["false_positives"])["passes"] is False


def test_best_rule_unknown_cue_raises():
    with pytest.raises(ValueError, match="unknown cue"):
        best_rule("elbow")


# --- scoreboard ------------------------------------------------------


def test_scoreboard_has_both_cues_and_no_row_passes_the_gate():
    frame = scoreboard_frame()
    assert len(frame) == 8
    assert set(frame["cue"]) == {"head", "shoulder"}
    for row in SCOREBOARD:
        assert gate_verdict(row["detected"], row["false_positives"])["passes"] is False


def test_scoreboard_model_numbers_are_pinned():
    frame = scoreboard_frame("shoulder")
    lr = frame[frame["model"] == "logistic_regression"].iloc[0]
    assert lr["f1"] == pytest.approx(0.172)
    assert lr["detected"] == 3 and lr["false_positives"] == 3
    mlp = frame[frame["model"] == "mlp"].iloc[0]
    assert mlp["f1"] == pytest.approx(0.172)
    assert mlp["detected"] == 3 and mlp["false_positives"] == 4
    head = scoreboard_frame("head")
    assert head[head["model"] == "mlp"].iloc[0]["f1"] == pytest.approx(0.105)
    assert head[head["model"] == "logistic_regression"].iloc[0]["f1"] == pytest.approx(0.097)
    # The random forest detects nothing on either cue.
    rf = frame[frame["model"] == "random_forest"].iloc[0]
    assert rf["detected"] == 0 and rf["f1"] == 0.0
    # Rule baselines at the operating thresholds.
    assert frame[frame["model"].str.startswith("rule")].iloc[0]["false_positives"] == 7
    assert head[head["model"].str.startswith("rule")].iloc[0]["false_positives"] == 0


# --- trial tables ----------------------------------------------------


def test_trial_tables_cover_five_trials_with_barely_recorded_trial_5():
    head = trials_frame("head")
    shoulder = trials_frame("shoulder")
    assert len(head) == 5 and len(shoulder) == 5
    assert head.iloc[4]["samples"] == 6
    assert shoulder.iloc[4]["samples"] == 8


def test_deepest_head_hold_never_reaches_the_threshold():
    head = trials_frame("head")
    assert head["max_abs_dev"].max() == pytest.approx(7.05, abs=0.01)
    assert head["max_abs_dev"].max() < 8  # session head threshold


def test_trials_frame_unknown_cue_raises():
    with pytest.raises(ValueError, match="unknown cue"):
        trials_frame("elbow")


def test_verdict_line_states_the_honest_outcome():
    assert "NOTHING passes the gate" in VERDICT
    assert "4/5 trials" in VERDICT
    assert "8 false positives" in VERDICT
    assert "7.1 deg" in VERDICT
