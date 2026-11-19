"""Tests for the Day 17 shoulder cue rework."""

import numpy as np
import pandas as pd
import pytest

from experiments.eval.head import dwell_s, trial_f1
from experiments.eval.shoulder import (
    add_shoulder_features,
    evaluate_shoulder_gate,
    shoulder_hold_dwells,
    shoulder_velocity,
)


def _dev_frames(dev, dt=0.1):
    """Frames with a given shoulder deviation series, all tracked."""
    n = len(dev)
    return pd.DataFrame({
        "elapsed_s": [round(i * dt, 1) for i in range(n)],
        "step": ["work"] * n,
        "s_dev": list(dev),
        "tracking": [1] * n,
    })


def test_velocity_is_zero_on_a_flat_series():
    vel = shoulder_velocity(_dev_frames([2.0] * 10))
    assert np.isfinite(vel).all()
    assert (vel.abs() < 1e-9).all()


def test_velocity_measures_a_steady_tilt_rate():
    # 0 to 9 deg over 0.9 s: 10 deg/s.
    vel = shoulder_velocity(_dev_frames(np.linspace(0, 9, 10)))
    assert vel.iloc[5] == pytest.approx(10.0, abs=0.5)


def test_velocity_is_nan_where_the_deviation_is_nan():
    dev = np.array([0.0, 1.0, np.nan, 3.0, 4.0])
    vel = shoulder_velocity(_dev_frames(dev))
    assert np.isnan(vel.iloc[2])
    assert np.isfinite(vel.iloc[1])


def test_slow_mask_keeps_steady_frames_and_rejects_jumps():
    frames = add_shoulder_features(_dev_frames([0.0] * 5 + [20.0] * 5))
    assert bool(frames["s_slow_5"].iloc[2])
    # the jump frame itself is fast: 20 deg in 0.1 s
    assert not bool(frames["s_slow_5"].iloc[5])


def test_dwell_measures_the_longest_run_above_two_degrees():
    dev = [0.0] * 5 + [3.0] * 6 + [0.0] * 5
    assert dwell_s(_dev_frames(dev), "s_dev", 2.0) == pytest.approx(0.5)
    assert dwell_s(_dev_frames(dev), "s_dev", 4.0) == pytest.approx(0.0)


def test_gate_grid_on_the_fixture(frames, session):
    result = evaluate_shoulder_gate(frames, session)
    assert len(result["rows"]) == 6 * 3 * 4  # thresholds x persists x caps
    assert result["alignment_verified"] is True
    # Fixture: five flat shoulder trials and one 6 s shoulder
    # excursion at +10 deg during work, so every configuration
    # detects 0/5 with 1 false positive.
    for row in result["rows"]:
        assert row["detected"] == 0
        assert row["false_positives"] == 1
        assert row["f1"] == pytest.approx(0.0)
    best = result["best"]
    assert best == {"threshold": 1, "persist_s": 5.0,
                    "vel_cap_deg_s": None, "detected": 0,
                    "trials_total": 5, "false_positives": 1,
                    "precision": 0.0, "recall": 0.0, "f1": 0.0}


def test_gate_grid_capped_false_positives_never_exceed_uncapped(
        frames, session):
    result = evaluate_shoulder_gate(frames, session)
    by_config = {(r["threshold"], r["persist_s"], r["vel_cap_deg_s"]): r
                 for r in result["rows"]}
    for thr in (1, 2, 3, 4, 6, 8):
        for persist in (2.0, 3.0, 5.0):
            uncapped = by_config[(thr, persist, None)]
            for cap in (5.0, 10.0, 20.0):
                capped = by_config[(thr, persist, cap)]
                assert capped["false_positives"] <= uncapped["false_positives"]
                assert capped["detected"] <= uncapped["detected"]


def test_hold_dwells_show_the_flat_trials(frames, session):
    from experiments.eval.analysis import align_trials
    head_align = align_trials(frames, session, "head")
    alignment = align_trials(frames, session, "shoulder",
                             fallback_offset=head_align["offset_s"])
    dwells = shoulder_hold_dwells(frames, session, alignment)
    assert len(dwells) == 5
    for row in dwells:
        assert row["samples"] == 60
        assert row["dwell_gt2_s"] == pytest.approx(0.0)
        assert row["max_abs_dev"] == pytest.approx(0.0, abs=0.5)


def test_trial_f1_is_shared_with_the_head_grid():
    # trial_f1 lives in head.py and is reused, not reimplemented.
    assert trial_f1(4, 8)["f1"] == pytest.approx(0.471, abs=0.001)
    assert trial_f1(0, 0)["f1"] == 0.0
