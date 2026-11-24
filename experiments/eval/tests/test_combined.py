"""Tests for the Day 23 combined-cue joint score."""

import numpy as np
import pandas as pd
import pytest

from experiments.eval.analysis import add_deviations
from experiments.eval.combined import (
    AND_HEAD_DEG,
    AND_SHOULDER_DEG,
    add_combined,
    evaluate_combined,
)
from experiments.eval.signals import add_signals


def _two_dev_frames(h_devs, s_devs):
    """Frames carrying only the two deviation columns add_combined needs."""
    return pd.DataFrame({"h_dev": h_devs, "s_dev_gated": s_devs})


def _combined_fixture_frames(frames, session):
    return add_combined(add_signals(frames, session))


def test_sum_is_normalized_sum():
    out = add_combined(_two_dev_frames([5.0, 0.0, 10.0], [1.0, 0.0, 0.0]))
    assert list(out["j_dev_sum"]) == pytest.approx([2.0, 0.0, 2.0])


def test_iso_shoulder_only_hold():
    out = add_combined(_two_dev_frames([0.0], [2.5]))
    assert float(out["j_dev_s"].iloc[0]) == pytest.approx(2.5)
    assert float(out["j_dev_h"].iloc[0]) == 0.0


def test_iso_head_only_hold():
    out = add_combined(_two_dev_frames([10.0], [0.0]))
    assert float(out["j_dev_h"].iloc[0]) == pytest.approx(10.0)
    assert float(out["j_dev_s"].iloc[0]) == 0.0


def test_iso_whole_body_lean_cancels():
    out = add_combined(_two_dev_frames([3.0], [3.0]))
    assert float(out["j_dev_s"].iloc[0]) == 0.0
    assert float(out["j_dev_h"].iloc[0]) == 0.0


def test_iso_clamped_at_zero():
    # Shoulder dominating must not read as a negative head score:
    # sustained_events takes the absolute value, so an unclamped
    # -4 would fire the head cue during a shoulder excursion.
    out = add_combined(_two_dev_frames([1.0], [5.0]))
    assert float(out["j_dev_h"].iloc[0]) == 0.0
    assert float(out["j_dev_s"].iloc[0]) == pytest.approx(4.0)


def test_nan_propagates_to_joint_scores():
    out = add_combined(_two_dev_frames([np.nan, 2.0], [1.0, np.nan]))
    for col in ("j_dev_s", "j_dev_h", "j_dev_sum"):
        assert out[col].isna().all()


def test_and_columns_use_operating_thresholds():
    out = add_combined(_two_dev_frames([5.0, 5.1, 0.0], [1.0, 0.0, 1.1]))
    assert list(out["j_head_above"]) == [False, True, False]
    assert list(out["j_shoulder_above"]) == [False, False, True]
    assert AND_HEAD_DEG == 5.0
    assert AND_SHOULDER_DEG == 1.0


def test_evaluate_combined_structure(frames, session):
    combined = _combined_fixture_frames(frames, session)
    result = evaluate_combined(combined, session)
    # The eval fixture records a shoulder trial summary, so its
    # shoulder alignment is verified; Run 1 records none, so its
    # shoulder alignment is inferred (see test_day23).
    assert result["alignment_verified"] is True
    for cue in ("head", "shoulder"):
        assert set(result["cues"][cue]["variants"]) == {
            "iso", "sum", "and_control"}
        for name, variant in result["cues"][cue]["variants"].items():
            assert "best" in variant
            assert isinstance(variant["best"]["pass"], bool)
        for name in ("iso", "sum"):
            assert len(result["cues"][cue]["variants"][name][
                "persistence_curve"]) == 3


def test_and_control_detects_nothing_on_fixture(frames, session):
    # The fixture holds are single-cue by construction, so a rule
    # that needs both cues above threshold can never fire.
    combined = _combined_fixture_frames(frames, session)
    result = evaluate_combined(combined, session)
    for cue in ("head", "shoulder"):
        best = result["cues"][cue]["variants"]["and_control"]["best"]
        assert best["detected"] == 0
        assert best["false_positives"] == 0


def test_sum_detects_fixture_head_trial_and_work_excursion(frames, session):
    # Fixture head trial 1 is a +10 deg head hold, shoulder flat:
    # the sum (10/5 + 0) crosses 1.0 and holds 6 s. The work
    # excursion is shoulder +10, head flat: the sum fires there too.
    combined = _combined_fixture_frames(frames, session)
    result = evaluate_combined(combined, session)
    head_best = result["cues"]["head"]["variants"]["sum"]["best"]
    assert head_best["detected"] == 1
    assert head_best["false_positives"] == 1
    shoulder_best = result["cues"]["shoulder"]["variants"]["sum"]["best"]
    assert shoulder_best["detected"] == 0
    assert shoulder_best["false_positives"] == 1


def test_iso_fixture_head_clean_shoulder_not(frames, session):
    # Head isolation on the fixture: the +10 head hold fires, and
    # the shoulder work excursion scores 0 after clamping, so no FP.
    # Shoulder isolation fires on the same excursion instead.
    combined = _combined_fixture_frames(frames, session)
    result = evaluate_combined(combined, session)
    head_best = result["cues"]["head"]["variants"]["iso"]["best"]
    assert head_best["detected"] == 1
    assert head_best["false_positives"] == 0
    shoulder_best = result["cues"]["shoulder"]["variants"]["iso"]["best"]
    assert shoulder_best["detected"] == 0
    assert shoulder_best["false_positives"] == 1
