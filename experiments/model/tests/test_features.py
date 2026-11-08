"""Tests for window feature engineering."""

import numpy as np
import pytest

from experiments.model.features import (FEATURES, build_windows,
                                        window_features)


def test_flat_window_features(flat_window):
    h, s = flat_window
    f = window_features(h, s)
    assert f["mean_abs_h"] == pytest.approx(0.0)
    assert f["max_abs_h"] == pytest.approx(0.0)
    assert f["std_h"] == pytest.approx(0.0)
    assert f["frac_h_gt5"] == pytest.approx(0.0)
    assert f["jerk_h"] == pytest.approx(0.0)
    assert f["std_s"] == pytest.approx(0.0)
    assert f["range_s"] == pytest.approx(0.0)
    assert f["corr_h_s"] == pytest.approx(0.0)


def test_tilted_window_features(tilted_window):
    h, s = tilted_window
    f = window_features(h, s)
    assert f["mean_abs_h"] == pytest.approx(6.0)
    assert f["max_abs_h"] == pytest.approx(6.0)
    assert f["mean_h"] == pytest.approx(6.0)
    assert f["frac_h_gt5"] == pytest.approx(1.0)
    assert f["std_h"] == pytest.approx(0.0)  # steady hold, no wobble


def test_jerk_measures_frame_to_frame_change():
    h = np.array([0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0])
    f = window_features(h, np.zeros(10))
    assert f["jerk_h"] == pytest.approx(2.0)
    assert f["std_h"] > 0.9


def test_corr_is_zero_when_one_side_is_flat():
    h = np.linspace(0, 9, 10)
    f = window_features(h, np.zeros(10))
    assert f["corr_h_s"] == pytest.approx(0.0)


def test_corr_is_one_for_perfectly_coupled_motion():
    h = np.linspace(0, 9, 10)
    f = window_features(h, h * 2.0)
    assert f["corr_h_s"] == pytest.approx(1.0)


def test_slope_measures_drift():
    s = np.linspace(0, 9, 10)  # +1 deg per frame
    f = window_features(np.zeros(10), s)
    assert f["slope_s"] == pytest.approx(1.0)
    assert f["range_s"] == pytest.approx(9.0)


def test_shoulder_features_never_include_shoulder_level():
    # Day 4 verdict: the raw shoulder angle is the wrong signal.
    # The feature list must not let the model see it.
    level_names = {"mean_s", "max_abs_s", "mean_abs_s", "frac_s_gt6"}
    assert not (set(FEATURES["shoulder"]) & level_names)
    assert "mean_abs_h" in FEATURES["shoulder"]  # head is context, allowed
    assert set(FEATURES["head"]) != set(FEATURES["shoulder"])


def test_build_windows_labels_and_groups(run1_frames):
    frames, session = run1_frames
    for cue in ("head", "shoulder"):
        w = build_windows(frames, session, cue)
        assert set(w["label"].unique()) <= {0, 1}
        assert w["label"].sum() > 10  # some tilted windows exist
        assert (w["label"] == 0).sum() > 100  # many upright windows
        assert set(w["group"].unique()) <= {1, 2, 3, 4, 5}
        # every feature column is finite
        assert np.isfinite(w[FEATURES[cue]].to_numpy()).all()
        # hold windows come only from the trial step
        assert set(w.loc[w["label"] == 1, "step"].unique()) == {cue}


def test_build_windows_rejects_unknown_cue(run1_frames):
    frames, session = run1_frames
    with pytest.raises(ValueError, match="unknown cue"):
        build_windows(frames, session, "neck")


def test_window_features_rejects_empty_window():
    with pytest.raises(ValueError, match="empty window"):
        window_features(np.array([]), np.array([]))
