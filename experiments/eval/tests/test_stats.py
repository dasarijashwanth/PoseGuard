"""Tests for Day 4 statistical validation."""

import os

import pytest

from experiments.eval.analysis import add_deviations
from experiments.eval.loader import load_session
from experiments.eval.stats import (
    cue_stats,
    fp_rate_interval,
    fp_rate_per_hour,
    persistence_curve,
    statistical_verdict,
    threshold_curve,
    wilson_interval,
    work_seconds,
    zero_fp_threshold,
)

RUN1 = os.path.expanduser(
    "~/workspace/user/files/poseguard-validation-2026-11-03T19-24-30-709Z.csv")


def test_wilson_known_values():
    # 0/5: the true rate could be as high as ~43%, but not 80%.
    lo, hi = wilson_interval(0, 5)
    assert lo == pytest.approx(0.0, abs=1e-9)
    assert hi == pytest.approx(0.4345, abs=0.005)
    # 5/5: mirror image, lower bound ~57%.
    lo, hi = wilson_interval(5, 5)
    assert lo == pytest.approx(0.5655, abs=0.005)
    assert hi == pytest.approx(1.0, abs=1e-9)
    # 4/5: the gate value, interval (0.38, 0.96), wide with 5 trials.
    lo, hi = wilson_interval(4, 5)
    assert lo == pytest.approx(0.3755, abs=0.005)
    assert hi == pytest.approx(0.9638, abs=0.005)
    # 1/5: (0.04, 0.62).
    lo, hi = wilson_interval(1, 5)
    assert lo == pytest.approx(0.036, abs=0.005)
    assert hi == pytest.approx(0.624, abs=0.005)


def test_wilson_rejects_bad_inputs():
    with pytest.raises(ValueError):
        wilson_interval(0, 0)
    with pytest.raises(ValueError):
        wilson_interval(6, 5)
    with pytest.raises(ValueError):
        wilson_interval(-1, 5)


def test_fp_rate_per_hour():
    # 7 false positives in 336.9 s of work is about 75 per hour.
    assert fp_rate_per_hour(7, 336.9) == pytest.approx(74.8, abs=0.5)
    assert fp_rate_per_hour(0, 336.9) == 0.0


def test_fp_rate_interval():
    lo, hi = fp_rate_interval(7, 336.9)
    # Normal approx: 74.8 +/- 1.96 * sqrt(7) / 336.9 * 3600.
    assert lo == pytest.approx(18.6, abs=1.0)
    assert hi == pytest.approx(131.0, abs=1.0)
    assert lo >= 0.0
    lo0, hi0 = fp_rate_interval(0, 100.0)
    assert lo0 == 0.0 and hi0 == 0.0


def test_work_seconds_fixture(frames):
    # Fixture work sample: 660 frames at 0.1 s, span 65.9 s.
    assert work_seconds(frames) == pytest.approx(65.9, abs=0.5)


def test_threshold_curve_fixture(frames, session):
    from experiments.eval.analysis import align_trials
    head_align = align_trials(frames, session, "head")
    rows = threshold_curve(frames, session, "head", head_align)
    # Trial 1 holds at +10 deg: detected 1/5 at every threshold.
    assert all(r["detected"] == 1 for r in rows)
    assert all(r["false_positives"] == 0 for r in rows)
    # Wilson interval for 1/5 tops out at ~62%, below the 80% gate.
    assert rows[0]["ci_high"] == pytest.approx(0.624, abs=0.005)
    assert rows[0]["ci_low"] == pytest.approx(0.036, abs=0.005)

    shoulder_align = align_trials(frames, session, "shoulder",
                                  fallback_offset=head_align["offset_s"])
    rows = threshold_curve(frames, session, "shoulder", shoulder_align)
    assert all(r["detected"] == 0 for r in rows)
    assert all(r["false_positives"] == 1 for r in rows)
    # One FP in 65.9 s of work is ~55 per hour.
    assert rows[0]["fp_per_hour"] == pytest.approx(54.6, abs=1.0)


def test_persistence_curve_fixture(frames, session):
    from experiments.eval.analysis import align_trials
    head_align = align_trials(frames, session, "head")
    rows = persistence_curve(frames, session, "head", 8, head_align)
    # The clean 6 s hold fires under every persistence rule.
    assert all(r["detected"] == 1 for r in rows)
    # Shorter persistence can only detect more, never fewer.
    assert [r["detected"] for r in rows] == sorted(
        [r["detected"] for r in rows], reverse=True)


def test_zero_fp_threshold_fixture(frames, session):
    from experiments.eval.analysis import align_trials
    head_align = align_trials(frames, session, "head")
    shoulder_align = align_trials(frames, session, "shoulder",
                                  fallback_offset=head_align["offset_s"])
    # Head work is flat: threshold 1 already has zero FPs, 1/5 detected.
    zfp = zero_fp_threshold(frames, session, "head", head_align)
    assert zfp == {"threshold": 1, "detected": 1, "trials_total": 5}
    # Shoulder work excursion is +10 deg sustained 6 s: silence needs
    # threshold 10, where detection is still 0/5.
    zfp = zero_fp_threshold(frames, session, "shoulder", shoulder_align)
    assert zfp == {"threshold": 10, "detected": 0, "trials_total": 5}


def test_cue_stats_fixture(frames, session):
    from experiments.eval.analysis import align_trials
    head_align = align_trials(frames, session, "head")
    shoulder_align = align_trials(frames, session, "shoulder",
                                  fallback_offset=head_align["offset_s"])
    head = cue_stats(frames, session, "head", 8, head_align)
    assert head["verdict"] == "salvageable_holds"
    assert head["ci_high"] == pytest.approx(0.624, abs=0.005)
    assert head["false_positives"] == 0
    assert "not a threshold problem" in head["verdict_text"]

    shoulder = cue_stats(frames, session, "shoulder", 6, shoulder_align)
    assert shoulder["verdict"] == "metric_wrong"
    assert shoulder["false_positives"] == 1
    assert shoulder["zero_fp_threshold"]["threshold"] == 10
    assert "different signal" in shoulder["verdict_text"]


def test_statistical_verdict_fixture(frames, session):
    result = statistical_verdict(frames, session)
    assert result["cues"]["head"]["verdict"] == "salvageable_holds"
    assert result["cues"]["shoulder"]["verdict"] == "metric_wrong"
    assert result["work_seconds"] == pytest.approx(65.9, abs=0.5)


def test_build_markdown_includes_day4_with_enriched_frames(frames, session):
    """Regression: build_markdown crashed (KeyError 'h_dev') when it
    ran the Day 4 stats on the raw session frames. With enriched
    frames it must render the section; without them it must skip it
    instead of crashing."""
    from experiments.eval.analysis import verdict
    from experiments.eval.report import build_markdown

    result = verdict(frames, session)
    md = build_markdown(session, result, frames)
    assert "## Statistical validation (Day 4)" in md
    assert "salvageable" in md
    md_plain = build_markdown(session, result)
    assert "## Statistical validation (Day 4)" not in md_plain
    assert "## Verdict" in md_plain


@pytest.mark.skipif(not os.path.exists(RUN1), reason="Run 1 CSV not present")
def test_run1_statistical_verdict():
    """Run 1, Day 4: head is salvageable by better holds, shoulder's
    metric is wrong for desk work. Both claims are pinned here."""
    session = load_session(RUN1)
    frames = add_deviations(session)
    result = statistical_verdict(frames, session)

    head = result["cues"]["head"]
    assert head["verdict"] == "salvageable_holds"
    # 0/5: Wilson tops out at ~43%, far below the 80% gate.
    assert head["ci_low"] == pytest.approx(0.0, abs=1e-9)
    assert head["ci_high"] == pytest.approx(0.4345, abs=0.005)
    assert head["false_positives"] == 0
    # Longest run at the lowest tested threshold (5 deg): 1.53 s.
    assert head["longest_run_lowest_threshold_s"] == pytest.approx(
        1.53, abs=0.05)
    # No threshold in the tested range rescues detection.
    assert all(r["detected"] == 0 for r in head["threshold_curve"])
    # Persistence curve at 5 deg: the tilts were real but brief.
    # At 0.5 s persistence 3/5 would have fired; at 2 s and up, 0/5.
    persist = {r["persist_s"]: r["detected"]
               for r in head["persistence_curve"]}
    assert persist[0.5] == 3
    assert persist[1.0] == 1
    assert persist[2.0] == 0
    assert persist[5.0] == 0
    assert "0.5 s persistence rule 3/5 holds" in head["verdict_text"]

    shoulder = result["cues"]["shoulder"]
    assert shoulder["verdict"] == "metric_wrong"
    assert shoulder["false_positives"] == 7
    # ~7 FPs in ~5.0 minutes of work is ~83 per hour.
    assert shoulder["fp_per_hour"] == pytest.approx(83.1, abs=2.0)
    assert all(r["detected"] == 0 for r in shoulder["threshold_curve"])
    # Whatever threshold silences the false alarms also silences
    # every real hold (or none exists up to 30 deg).
    zfp = shoulder["zero_fp_threshold"]
    assert zfp is None or zfp["detected"] < 4
