"""Tests for the Day 15 head cue rework."""

import numpy as np
import pandas as pd
import pytest

from experiments.eval.head import (
    add_head_features,
    dwell_s,
    evaluate_head_gate,
    head_velocity,
    hold_dwells,
    trial_f1,
)


def _dev_frames(dev, dt=0.1):
    """Frames with a given head deviation series, all tracked."""
    n = len(dev)
    return pd.DataFrame({
        "elapsed_s": [round(i * dt, 1) for i in range(n)],
        "step": ["work"] * n,
        "h_dev": list(dev),
        "tracking": [1] * n,
    })


def test_velocity_is_zero_on_a_flat_series():
    vel = head_velocity(_dev_frames([3.0] * 10))
    assert np.isfinite(vel).all()
    assert (vel.abs() < 1e-9).all()


def test_velocity_measures_a_steady_tilt_rate():
    # 0 to 9 deg over 0.9 s: 10 deg/s.
    vel = head_velocity(_dev_frames(np.linspace(0, 9, 10)))
    assert vel.iloc[5] == pytest.approx(10.0, abs=0.5)


def test_velocity_is_nan_where_the_deviation_is_nan():
    dev = np.array([0.0, 1.0, np.nan, 3.0, 4.0])
    vel = head_velocity(_dev_frames(dev))
    assert np.isnan(vel.iloc[2])
    assert np.isfinite(vel.iloc[1])


def test_slow_mask_keeps_steady_frames_and_rejects_jumps():
    frames = add_head_features(_dev_frames([0.0] * 5 + [20.0] * 5))
    assert bool(frames["h_slow_5"].iloc[2])
    # the jump frame itself is fast: 20 deg in 0.1 s
    assert not bool(frames["h_slow_5"].iloc[5])


def test_dwell_measures_the_longest_run_above_threshold():
    dev = [0.0] * 5 + [6.0] * 6 + [0.0] * 5
    assert dwell_s(_dev_frames(dev), "h_dev", 5.0) == pytest.approx(0.5)
    assert dwell_s(_dev_frames(dev), "h_dev", 7.0) == pytest.approx(0.0)


def test_trial_f1_scores_a_partial_detection():
    f1 = trial_f1(2, 9)
    assert f1["precision"] == pytest.approx(0.182, abs=0.001)
    assert f1["recall"] == pytest.approx(0.4)
    assert f1["f1"] == pytest.approx(0.25)


def test_trial_f1_is_zero_when_nothing_is_detected():
    assert trial_f1(0, 0) == {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    assert trial_f1(0, 3)["f1"] == 0.0


def test_trial_f1_is_one_for_a_clean_sweep():
    assert trial_f1(5, 0) == {"precision": 1.0, "recall": 1.0, "f1": 1.0}


def test_gate_grid_on_the_fixture(frames, session):
    result = evaluate_head_gate(frames, session)
    assert len(result["rows"]) == 5 * 3 * 4  # thresholds x persists x caps
    # Fixture: one clean 6 s hold at +10 deg, four flat trials, and no
    # head excursion during work, so every configuration detects 1/5
    # with 0 false positives.
    for row in result["rows"]:
        assert row["detected"] == 1
        assert row["false_positives"] == 0
        assert row["f1"] == pytest.approx(0.333, abs=0.001)
    best = result["best"]
    assert best["detected"] == 1
    assert best["false_positives"] == 0
    assert best["persist_s"] == 5.0  # ties break toward the live gate


def test_gate_grid_capped_false_positives_never_exceed_uncapped(
        frames, session):
    result = evaluate_head_gate(frames, session)
    by_config = {(r["threshold"], r["persist_s"], r["vel_cap_deg_s"]): r
                 for r in result["rows"]}
    for thr in (4, 5, 6, 7, 8):
        for persist in (2.0, 3.0, 5.0):
            uncapped = by_config[(thr, persist, None)]
            for cap in (5.0, 10.0, 20.0):
                capped = by_config[(thr, persist, cap)]
                assert capped["false_positives"] <= uncapped["false_positives"]
                assert capped["detected"] <= uncapped["detected"]


def test_hold_dwells_show_the_clean_hold(frames, session):
    from experiments.eval.analysis import align_trials
    alignment = align_trials(frames, session, "head")
    dwells = hold_dwells(frames, session, alignment)
    assert dwells[0]["samples"] >= 50
    assert dwells[0]["dwell_gt5_s"] > 5.0
    assert dwells[0]["max_abs_dev"] == pytest.approx(10.0, abs=0.5)
    for row in dwells[1:]:
        assert row["dwell_gt5_s"] == pytest.approx(0.0)
        assert row["max_abs_dev"] == pytest.approx(0.0, abs=0.5)
