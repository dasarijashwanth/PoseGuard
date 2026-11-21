"""Day 19 tests: live webcam check logic, section copy, honest verdict.

The webcam module must stay pure: angle math from landmarks, baseline
calibration, and the persistence gate, all testable without a camera,
a server, or a browser. The angle math is pinned to the Day 14
capture tool (experiments/rerun-capture.html): ear line vs shoulder
line, circular deviations from an upright baseline.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from demo.demo_data import (
    FEEDBACK_BODY,
    FEEDBACK_HEADING,
    FEEDBACK_URL,
    HOWTO_HEADING,
    HOWTO_STEPS,
    PURPOSE_BODY,
    PURPOSE_HEADING,
    VERDICT,
)
from demo.webcam import (
    LIVE_PRESETS,
    LiveGate,
    calibrate,
    circ_dev,
    circ_mean,
    deviations,
    ema_smooth,
    evaluate_stream,
    landmark_angles,
    wrap_angle,
)

HERE = Path(__file__).resolve().parent.parent.parent.parent


def _landmarks(head_tilt_deg: float = 0.0, shoulder_tilt_deg: float = 0.0):
    """33 fake landmarks with a chosen ear-line and shoulder-line tilt.

    Image coordinates (y down): a line tilted by d deg has its right
    endpoint lower by tan(d) over a 100 px run. Ears at y=100, shoulders
    at y=200, so the two lines are independent.
    """
    lm = [(0.0, 0.0)] * 33
    lm = [list(p) for p in lm]
    run = 100.0
    ear_drop = run * math.tan(math.radians(head_tilt_deg))
    sh_drop = run * math.tan(math.radians(shoulder_tilt_deg))
    lm[7] = [100.0, 100.0]
    lm[8] = [200.0, 100.0 + ear_drop]
    lm[11] = [100.0, 200.0]
    lm[12] = [200.0, 200.0 + sh_drop]
    return [tuple(p) for p in lm]


# --- circular angle helpers ------------------------------------------


def test_wrap_angle_matches_capture_tool():
    assert wrap_angle(190) == pytest.approx(-170)
    assert wrap_angle(-190) == pytest.approx(170)
    assert wrap_angle(180) == pytest.approx(180)
    assert wrap_angle(-180) == pytest.approx(-180)
    assert wrap_angle(540) == pytest.approx(180)
    assert wrap_angle(0) == pytest.approx(0)


def test_circ_dev_wraps_across_zero():
    assert circ_dev(10, 350) == pytest.approx(20)
    assert circ_dev(350, 10) == pytest.approx(-20)
    assert circ_dev(170, -170) == pytest.approx(-20)
    assert circ_dev(-170, 170) == pytest.approx(20)
    assert circ_dev(5, 5) == pytest.approx(0)


def test_circ_mean_wraps_across_zero():
    assert circ_mean([350, 10]) == pytest.approx(0)
    assert circ_mean([90, 90]) == pytest.approx(90)
    assert circ_mean([0, 0, 0]) == pytest.approx(0)


def test_circ_mean_empty_raises():
    with pytest.raises(ValueError, match="at least one angle"):
        circ_mean([])


# --- landmarks to angles ----------------------------------------------


def test_landmark_angles_level_pose_reads_zero():
    head_raw, shoulder_raw = landmark_angles(_landmarks())
    assert head_raw == pytest.approx(0)
    assert shoulder_raw == pytest.approx(0)


def test_landmark_angles_head_tilt_is_ear_minus_shoulder():
    # Ears tilted 10 deg, shoulders level: head raw is 10, shoulder 0.
    head_raw, shoulder_raw = landmark_angles(_landmarks(head_tilt_deg=10))
    assert head_raw == pytest.approx(10, abs=0.01)
    assert shoulder_raw == pytest.approx(0)
    # Shoulders tilted 6 deg, ears level: head raw is -6 (ears relative
    # to shoulders), shoulder raw is 6.
    head_raw, shoulder_raw = landmark_angles(_landmarks(shoulder_tilt_deg=6))
    assert head_raw == pytest.approx(-6, abs=0.01)
    assert shoulder_raw == pytest.approx(6, abs=0.01)


def test_landmark_angles_accepts_mediapipe_shaped_objects():
    class Pt:
        def __init__(self, x, y):
            self.x, self.y = x, y

    lm = [Pt(0.0, 0.0)] * 33
    lm[7] = Pt(0.30, 0.40)
    lm[8] = Pt(0.70, 0.40)
    lm[11] = Pt(0.30, 0.60)
    lm[12] = Pt(0.70, 0.60)
    head_raw, shoulder_raw = landmark_angles(lm)
    assert head_raw == pytest.approx(0)
    assert shoulder_raw == pytest.approx(0)


def test_landmark_angles_missing_landmarks_raise():
    with pytest.raises(ValueError, match="at least 13"):
        landmark_angles([(0.0, 0.0)] * 5)


def test_landmark_angles_nan_or_none_raise():
    bad = _landmarks()
    bad[7] = (float("nan"), 100.0)
    with pytest.raises(ValueError, match="NaN"):
        landmark_angles(bad)
    bad = _landmarks()
    bad[11] = (None, 200.0)
    with pytest.raises(ValueError, match="None"):
        landmark_angles(bad)


# --- calibration and deviations ---------------------------------------


def test_calibrate_circular_mean_of_upright_samples():
    baseline = calibrate([(350.0, 5.0), (10.0, 5.0), (0.0, 5.0),
                          (355.0, 5.0), (5.0, 5.0)])
    assert baseline[0] == pytest.approx(0, abs=1.0)
    assert baseline[1] == pytest.approx(5.0)


def test_calibrate_too_few_samples_raises():
    with pytest.raises(ValueError, match="at least 5"):
        calibrate([(0.0, 0.0)] * 4)


def test_deviations_are_circular_from_baseline():
    assert deviations(10.0, 5.0, (350.0, 0.0)) == pytest.approx((20.0, 5.0))
    assert deviations(350.0, 355.0, (10.0, 0.0)) == pytest.approx((-20.0, -5.0))


def test_ema_smooth_seeds_and_follows_circularly():
    assert ema_smooth(None, 30.0) == pytest.approx(30.0)
    # 0.3 of the way from 350 toward 10 across the wrap: 350 + 0.3*20 = 356 -> -4.
    assert ema_smooth(350.0, 10.0) == pytest.approx(-4.0)


# --- live gate ----------------------------------------------------------


def _stream(devs, dt=0.1):
    return [(i * dt, d) for i, d in enumerate(devs)]


def test_live_gate_fires_after_persistence():
    result = evaluate_stream(_stream([10.0] * 60), 8, 5.0)
    assert result["fired"] is True
    assert result["fired_at"] == pytest.approx(5.0, abs=0.11)
    assert result["max_run_s"] == pytest.approx(5.9, abs=0.11)


def test_live_gate_never_fires_below_threshold():
    result = evaluate_stream(_stream([5.0] * 100), 8, 5.0)
    assert result["fired"] is False
    assert result["fired_at"] is None
    assert result["max_run_s"] == pytest.approx(0.0)


def test_live_gate_breaks_when_hold_releases_early():
    # 3 s above, 1 s below, 3 s above: no 5 s run anywhere.
    samples = _stream([10.0] * 30 + [0.0] * 10 + [10.0] * 30)
    result = evaluate_stream(samples, 8, 5.0)
    assert result["fired"] is False
    assert result["max_run_s"] == pytest.approx(2.9, abs=0.11)


def test_live_gate_velocity_cap_breaks_the_run():
    # A jump from 10 to 30 deg in 0.1 s is 200 deg/s, over a 10 deg/s
    # cap, so the run restarts even though every sample is above threshold.
    gate = LiveGate(8, 5.0, vel_cap_deg_s=10.0)
    gate.update(0.0, 10.0)
    gate.update(1.0, 10.0)
    state = gate.update(1.1, 30.0)
    assert state["run_s"] == pytest.approx(0.0)
    assert state["fired"] is False


def test_live_gate_tracking_loss_breaks_the_run():
    gate = LiveGate(8, 5.0)
    gate.update(0.0, 10.0)
    gate.update(1.0, 10.0)
    state = gate.update(2.0, None)
    assert state["run_s"] == pytest.approx(0.0)
    assert state["fired"] is False


def test_live_gate_fired_latches_until_reset():
    gate = LiveGate(8, 2.0)
    for t, d in _stream([10.0] * 30, dt=0.1):
        gate.update(t, d)
    assert gate.fired is True
    gate.update(3.0, 0.0)  # sitting back up does not unfire it
    assert gate.fired is True
    gate.reset()
    assert gate.fired is False
    assert gate.run_s == pytest.approx(0.0)


def test_live_gate_rejects_bad_configs():
    with pytest.raises(ValueError, match="threshold"):
        LiveGate(0, 5.0)
    with pytest.raises(ValueError, match="persist"):
        LiveGate(8, 0)
    with pytest.raises(ValueError, match="vel_cap"):
        LiveGate(8, 5.0, vel_cap_deg_s=0)


def test_live_gate_state_names_its_config():
    gate = LiveGate(8, 5.0)
    state = gate.state()
    assert state["threshold_deg"] == 8
    assert state["persist_s"] == 5.0
    assert state["fired"] is False


# --- presets ------------------------------------------------------------


def test_live_presets_cover_both_cues_and_pin_the_configs():
    assert set(LIVE_PRESETS) == {"operating", "best_rule"}
    op = LIVE_PRESETS["operating"]
    assert op["head"] == {"threshold_deg": 8, "persist_s": 5.0, "vel_cap_deg_s": None}
    assert op["shoulder"] == {"threshold_deg": 6, "persist_s": 5.0, "vel_cap_deg_s": None}
    best = LIVE_PRESETS["best_rule"]
    assert best["head"]["threshold_deg"] == 4
    assert best["shoulder"]["threshold_deg"] == 1
    # The shoulder best rule fires on everyday leaning: 2.5 deg of drift
    # for 5 s at a 1 deg threshold is a hold by that config's lights.
    result = evaluate_stream(_stream([2.5] * 60), **best["shoulder"])
    assert result["fired"] is True


# --- section copy -------------------------------------------------------


def test_purpose_section_says_what_poseguard_is_for():
    assert PURPOSE_HEADING == "What PoseGuard is for"
    assert "slouch" in PURPOSE_BODY
    assert "webcam" in PURPOSE_BODY
    assert "no video ever leaves your machine" in PURPOSE_BODY
    assert "nothing passes the gate yet" in PURPOSE_BODY


def test_howto_steps_cover_live_check_snapshot_and_session():
    assert HOWTO_HEADING == "How to use this demo"
    assert len(HOWTO_STEPS) >= 3
    joined = " ".join(HOWTO_STEPS)
    assert "Calibrate" in joined
    assert "snapshot" in joined
    assert "Run 1" in joined


def test_feedback_section_links_to_github_issues():
    assert FEEDBACK_HEADING == "Feedback"
    assert FEEDBACK_URL == "https://github.com/dasarijashwanth/PoseGuard/issues"
    assert "open an issue" in FEEDBACK_BODY


def test_verdict_still_honest_after_day19():
    assert "NOTHING passes the gate" in VERDICT


def test_webcam_module_has_no_heavy_imports():
    source = (HERE / "demo" / "webcam.py").read_text()
    assert "import streamlit" not in source
    assert "import mediapipe" not in source
    assert "import cv2" not in source


def test_app_mentions_the_new_sections():
    source = (HERE / "demo" / "app.py").read_text()
    for name in ("PURPOSE_HEADING", "HOWTO_HEADING", "FEEDBACK_URL"):
        assert name in source
    assert "webcam" in source.lower()
