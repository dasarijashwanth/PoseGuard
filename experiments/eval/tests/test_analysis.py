"""Tests for the analysis: smoothing, persistence, trials, verdicts."""

import pandas as pd
import pytest

from experiments.eval.analysis import (
    add_deviations,
    align_trials,
    analyze_cue,
    ema_smooth,
    false_positives,
    loss_test,
    sustained_events,
    verdict,
)


def test_ema_smooth_converges_and_wraps():
    # Step from 170 to -170: the short way is +20 through 180, not -340.
    s = ema_smooth(pd.Series([170.0] * 5 + [-170.0] * 20))
    assert s.iloc[-1] == pytest.approx(-170, abs=1.0)
    assert all(-180 <= v <= 180 for v in s)


def test_sustained_events_needs_full_persistence(frames):
    head = frames[frames["step"] == "head"]
    # Trial 1 holds +10 deg (threshold 8) for 6 s: exactly one firing.
    events = sustained_events(head, "h_dev", 8)
    assert len(events) == 1
    # 5.6 s, not 6.0: the EMA ramp needs ~0.3 s to climb past 8 deg
    # after the raw angle jumps, and the hold ends at 11.9 s.
    assert events[0]["duration_s"] == pytest.approx(5.6, abs=0.3)
    # A 4 s requirement would also fire; a 6.5 s requirement would not.
    assert len(sustained_events(head, "h_dev", 8, persist_s=4.0)) == 1
    assert sustained_events(head, "h_dev", 8, persist_s=6.5) == []


def test_sustained_events_reset_on_tracking_loss(frames):
    # Same head data, but knock out tracking mid-hold: the run must break
    # and the remaining stretch is too short to fire.
    head = frames[frames["step"] == "head"].copy()
    head.loc[(head["elapsed_s"] >= 8.0) & (head["elapsed_s"] < 9.0),
             "tracking"] = 0
    assert sustained_events(head, "h_dev", 8) == []


def test_add_deviations_recomputes_circularly(frames):
    # Fixture shoulder baseline is 170, excursion reads 180: circular
    # dev is +10, and here the linear dev agrees (no wrap involved).
    work = frames[frames["step"] == "work"]
    assert work["s_dev"].max() == pytest.approx(10, abs=0.5)


def test_align_trials_recovers_offset(frames, session):
    # Fixture trials start exactly at the step start: offset should be 0.
    result = align_trials(frames, session, "head")
    assert result["verified"] is True
    assert result["offset_s"] == pytest.approx(0.0, abs=0.2)
    assert result["trials"][0]["samples"] == pytest.approx(60, abs=3)
    assert result["trials"][0]["mean"] == pytest.approx(10, abs=1.0)


def test_analyze_cue_fixture_head_is_1_of_5(frames, session):
    alignment = align_trials(frames, session, "head")
    result = analyze_cue(frames, session, "head", 8, alignment)
    assert result["detected"] == 1
    assert [t["detected"] for t in result["per_trial"]] == [
        True, False, False, False, False]


def test_false_positive_counted_in_work(frames):
    fp = false_positives(frames, "shoulder", 6)
    assert fp["count"] == 1
    assert fp["recorded_by_experiment"] == 1
    assert len(fp["events"]) == 1


def test_loss_test_fixture_is_valid(frames, session):
    result = loss_test(frames, session)
    assert result["ran"] is True
    assert result["clean"] is True
    assert result["tracking_lost_frames"] == 100


def test_verdict_fixture(session, frames):
    result = verdict(frames, session)
    # Head: 1/5 detected -> FAIL. Shoulder: 0/5 but 1 false positive -> FAIL.
    assert result["cues"]["head"]["pass"] is False
    assert result["cues"]["head"]["detected"] == 1
    assert result["cues"]["shoulder"]["pass"] is False
    assert result["cues"]["shoulder"]["false_positives"] == 1
