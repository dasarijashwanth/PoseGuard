"""Tests for the CSV loader: schema validation and section parsing."""

import pytest

from experiments.eval.loader import load_session
from conftest import build_csv


def test_loads_fixture_session(session):
    assert session.baseline_head == 0
    assert session.baseline_shoulder == 170
    assert session.threshold_head == 8
    assert session.threshold_shoulder == 6
    assert len(session.frames) == 30 + 600 + 600 + 660 + 100
    assert set(session.frames["step"].unique()) == {
        "calib", "head", "shoulder", "work", "loss"}
    assert len(session.trial_summary) == 10
    head_trials = [r for r in session.trial_summary if r["cue"] == "head"]
    assert head_trials[0]["detected"] is True
    assert all(r["detected"] is False for r in head_trials[1:])


def test_rejects_missing_metadata(tmp_path):
    path = build_csv(tmp_path,
                     meta_overrides={"Baseline head (deg)": None})
    # remove the baseline-head line entirely
    text = open(path).read().replace("Baseline head (deg),None\n", "")
    open(path, "w").write(text)
    with pytest.raises(ValueError, match="missing metadata"):
        load_session(path)


def test_rejects_missing_column(tmp_path):
    path = build_csv(tmp_path)
    text = open(path).read().replace(
        "timestamp,elapsed_s,step,head_tilt,shoulder_tilt,"
        "cue_detected,tracking,head_deviation,shoulder_deviation",
        "timestamp,elapsed_s,step,head_tilt,shoulder_tilt,"
        "cue_detected,tracking,head_deviation")
    open(path, "w").write(text)
    with pytest.raises(ValueError, match="missing frame columns"):
        load_session(path)


def test_rejects_unknown_step(tmp_path):
    path = build_csv(tmp_path)
    text = open(path).read().replace(",head,10.0,", ",dance,10.0", 1)
    open(path, "w").write(text)
    with pytest.raises(ValueError, match="unknown step"):
        load_session(path)


def test_rejects_empty_file(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("")
    with pytest.raises(ValueError, match="empty file"):
        load_session(str(path))
