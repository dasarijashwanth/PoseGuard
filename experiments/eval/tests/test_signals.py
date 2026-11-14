"""Tests for the Day 11 replacement shoulder signals."""

import numpy as np
import pandas as pd
import pytest

from experiments.eval.analysis import add_deviations, align_trials, analyze_cue
from experiments.eval.signals import (
    CANDIDATES,
    add_signals,
    evaluate_all,
    evaluate_signal,
    gated_dev,
    jump_valid_mask,
    pick_winner,
    relative_dev,
    rolling_baseline_dev,
)


def _jump_frames():
    """Frames with one 25 deg shoulder jump, otherwise flat."""
    rows = []
    for i in range(30):
        rows.append({"elapsed_s": round(i * 0.1, 1),
                     "step": "work",
                     "head_tilt": 0.0,
                     "shoulder_tilt": 170.0,
                     "tracking": 1})
    rows.append({"elapsed_s": 3.0, "step": "work", "head_tilt": 0.0,
                 "shoulder_tilt": 145.0, "tracking": 1})
    for i in range(31, 60):
        rows.append({"elapsed_s": round(i * 0.1, 1),
                     "step": "work",
                     "head_tilt": 0.0,
                     "shoulder_tilt": 145.0,
                     "tracking": 1})
    df = pd.DataFrame(rows)
    df["s_smooth"] = df["shoulder_tilt"]
    df["h_smooth"] = df["head_tilt"]
    return df


def _shoulder_alignment(frames, session):
    head_align = align_trials(frames, session, "head")
    return align_trials(frames, session, "shoulder",
                        fallback_offset=head_align["offset_s"])


def test_rolling_baseline_needs_history_then_tracks_flat(frames, session):
    dev = rolling_baseline_dev(frames, session)
    assert pd.isna(dev.iloc[0])
    assert np.isfinite(dev.iloc[100])
    flat = dev.iloc[100:200]
    assert float(flat.abs().median()) < 1.0


def test_rolling_baseline_sees_a_step_then_absorbs_it(frames, session):
    dev = rolling_baseline_dev(frames, session)
    work = frames[frames["step"] == "work"]
    exc = dev[work.index[300:360]]
    assert float(exc.max()) > 5.0
    late = dev[work.index[400:500]]
    assert float(late.abs().median()) < 2.0


def test_relative_dev_cancels_joint_motion(frames, session):
    dev = relative_dev(frames, session)
    head = frames[frames["step"] == "head"]
    hold1 = dev[head.index[30:90]]
    assert float(hold1.mean()) < -5.0
    work = frames[frames["step"] == "work"]
    exc = dev[work.index[300:360]]
    assert float(exc.mean()) > 5.0


def test_jump_mask_flags_the_jump_and_the_settle_second():
    df = _jump_frames()
    clean = jump_valid_mask(df)
    assert bool(clean.iloc[29])
    assert not bool(clean.iloc[30])
    assert not bool(clean.iloc[35])
    assert not bool(clean.iloc[40])
    assert bool(clean.iloc[41])


def test_jump_mask_ignores_small_jumps(frames):
    clean = jump_valid_mask(frames)
    assert bool(clean.all())


def test_gated_dev_is_nan_only_on_contaminated_frames():
    df = _jump_frames()
    df["s_dev"] = df["shoulder_tilt"] - 170.0
    df["s_clean"] = jump_valid_mask(df)
    gated = gated_dev(df)
    assert pd.isna(gated.iloc[30])
    assert pd.isna(gated.iloc[35])
    assert float(gated.iloc[10]) == pytest.approx(0.0)
    assert float(gated.iloc[50]) == pytest.approx(-25.0)


def test_add_signals_adds_four_columns(frames, session):
    out = add_signals(frames, session)
    for col in ("s_dev_roll", "s_dev_rel", "s_clean", "s_dev_gated"):
        assert col in out.columns
    assert out["s_clean"].dtype == bool


def test_evaluate_signal_returns_the_full_protocol(frames, session):
    out = add_signals(frames, session)
    alignment = _shoulder_alignment(out, session)
    for name in CANDIDATES:
        result = evaluate_signal(out, session, alignment, name)
        assert len(result["threshold_curve"]) == 6
        assert len(result["persistence_curve"]) == 3
        assert result["best"]["threshold"] in (1, 2, 3, 4, 6, 8)


def test_evaluate_all_covers_old_level_and_candidates(frames, session):
    out = add_signals(frames, session)
    ev = evaluate_all(out, session)
    assert set(ev["candidates"]) == {"roll", "rel", "gated"}
    assert "old_level" in ev
    assert "alignment_offset_s" in ev


def test_pick_winner_prefers_gated_on_a_tie(frames, session):
    out = add_signals(frames, session)
    ev = evaluate_all(out, session)
    assert pick_winner(ev) == "gated"


def test_analyze_cue_accepts_a_deviation_override(frames, session):
    out = add_signals(frames, session)
    alignment = _shoulder_alignment(out, session)
    a = analyze_cue(out, session, "shoulder", 6, alignment,
                    dev_col="s_dev_rel")
    assert a["detected"] == 0
    assert a["trials_total"] == 5
