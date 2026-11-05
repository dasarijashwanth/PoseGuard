"""Tests for threshold sensitivity, plus the real Run 1 CSV end to end."""

import os

import pytest

from experiments.eval.analysis import add_deviations, sensitivity, verdict
from experiments.eval.loader import load_session

RUN1 = os.path.expanduser(
    "~/workspace/user/files/poseguard-validation-2026-11-03T19-24-30-709Z.csv")


def test_sensitivity_fixture_head_thresholds(frames, session):
    rows = sensitivity(frames, session)
    head = {r["threshold"]: r for r in rows if r["cue"] == "head"}
    # Trial 1 holds at +10 deg: detected at every threshold 5-8.
    assert all(head[t]["detected"] == "1/5" for t in (5, 6, 7, 8))
    assert all(head[t]["false_positives"] == 0 for t in (5, 6, 7, 8))
    shoulder = {r["threshold"]: r for r in rows if r["cue"] == "shoulder"}
    # Work excursion is +10 deg: fires at 4, 5, 6, 8 alike.
    assert all(shoulder[t]["false_positives"] == 1 for t in (4, 5, 6, 8))
    assert all(shoulder[t]["detected"] == "0/5" for t in (4, 5, 6, 8))


@pytest.mark.skipif(not os.path.exists(RUN1), reason="Run 1 CSV not present")
def test_run1_csv_end_to_end():
    """The real session: both cues fail, and lowering the threshold
    does not rescue either one. This pins the headline findings."""
    session = load_session(RUN1)
    frames = add_deviations(session)
    result = verdict(frames, session)
    assert result["cues"]["head"]["detected"] == 0
    assert result["cues"]["shoulder"]["detected"] == 0
    assert result["cues"]["head"]["pass"] is False
    assert result["cues"]["shoulder"]["pass"] is False
    # Shoulder false positives survive the circular-math fix.
    assert result["cues"]["shoulder"]["false_positives"] > 0
    # Landmark-loss test never ran.
    assert result["loss_test"]["ran"] is False
    # Sensitivity: head is 0/5 at every threshold 5-8.
    head_rows = [r for r in result["sensitivity"] if r["cue"] == "head"]
    assert all(r["detected"] == "0/5" for r in head_rows)
    # Shoulder is 0/5 at every threshold 4-8 too.
    sh_rows = [r for r in result["sensitivity"] if r["cue"] == "shoulder"]
    assert all(r["detected"] == "0/5" for r in sh_rows)
