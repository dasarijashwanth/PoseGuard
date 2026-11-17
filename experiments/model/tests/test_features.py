"""Tests for window feature engineering."""

import numpy as np
import pytest

from experiments.model.features import (FEATURES, build_windows,
                                        window_features)


def test_gated_features_default_to_zero_without_new_signal(flat_window):
    h, s = flat_window
    f = window_features(h, s)
    assert f["mean_abs_sn"] == pytest.approx(0.0)
    assert f["max_abs_sn"] == pytest.approx(0.0)
    assert f["std_sn"] == pytest.approx(0.0)


def test_gated_features_measure_the_new_signal(tilted_window):
    h, s = tilted_window
    s_new = np.full(len(s), 2.0)
    f = window_features(h, s, s_new)
    assert f["mean_abs_sn"] == pytest.approx(2.0)
    assert f["max_abs_sn"] == pytest.approx(2.0)
    assert f["std_sn"] == pytest.approx(0.0)


def test_gated_features_ignore_nan_frames():
    h = np.zeros(10)
    s = np.zeros(10)
    s_new = np.array([2.0, np.nan, 2.0, np.nan, 2.0,
                      2.0, 2.0, 2.0, 2.0, 2.0])
    f = window_features(h, s, s_new)
    assert f["mean_abs_sn"] == pytest.approx(2.0)
    assert f["max_abs_sn"] == pytest.approx(2.0)


def test_build_windows_carries_the_gated_features(run1_frames):
    frames, session = run1_frames
    frames["s_dev_new"] = frames["s_dev"]
    w = build_windows(frames, session, "shoulder")
    for col in ("mean_abs_sn", "max_abs_sn", "std_sn"):
        assert col in w.columns
        assert np.isfinite(w[col]).all()


def test_build_windows_without_new_signal_is_zero(run1_frames):
    frames, session = run1_frames
    frames = frames.drop(columns=["s_dev_new"])
    w = build_windows(frames, session, "shoulder")
    assert (w["mean_abs_sn"] == 0.0).all()


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


def test_velocity_measures_a_steady_tilt_rate():
    # 0 to 9 deg over 0.9 s: 10 deg/s.
    h = np.linspace(0, 9, 10)
    f = window_features(h, np.zeros(10))
    assert f["vel_h"] == pytest.approx(10.0, abs=0.5)


def test_velocity_is_zero_on_a_flat_window(flat_window):
    h, s = flat_window
    f = window_features(h, s)
    assert f["vel_h"] == pytest.approx(0.0)


def test_velocity_uses_real_time_not_frame_count():
    # Same per-frame steps, twice the elapsed time: half the deg/s.
    h = np.linspace(0, 9, 10)
    t = np.arange(10) * 0.2
    f = window_features(h, np.zeros(10), times=t)
    assert f["vel_h"] == pytest.approx(5.0, abs=0.5)


def test_dwell_measures_the_longest_run_above_five_degrees():
    h = np.array([0.0, 0.0, 0.0, 0.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0])
    f = window_features(h, np.zeros(10))
    assert f["dwell_h"] == pytest.approx(0.5)


def test_dwell_is_zero_when_nothing_exceeds_five_degrees():
    h = np.full(10, 4.0)
    f = window_features(h, np.zeros(10))
    assert f["dwell_h"] == pytest.approx(0.0)


def test_dwell_breaks_on_a_tracking_gap():
    h = np.array([6.0, 6.0, 6.0, 6.0, 6.0, np.nan,
                  6.0, 6.0, 6.0, 6.0])
    f = window_features(h, np.zeros(10))
    assert f["dwell_h"] == pytest.approx(0.4)


def test_day15_head_list_adds_velocity_and_dwell_only():
    from experiments.model.features import DAY15_HEAD_FEATURES
    assert DAY15_HEAD_FEATURES == FEATURES["head"] + ["vel_h", "dwell_h"]
    assert len(FEATURES["head"]) == 6  # served v2 artifacts still match


def test_build_windows_carries_day15_head_features(run1_frames):
    frames, session = run1_frames
    w = build_windows(frames, session, "head")
    for col in ("vel_h", "dwell_h"):
        assert col in w.columns
        assert np.isfinite(w[col]).all()
