"""Day 23 tests: the combined-cue joint score on Run 1.

The joint score is the first rule to beat the Day 22 bests on F1:
the normalized sum reaches shoulder 5/5 trials with 8 false
positives (F1 0.556) and head 4/5 with 8 false positives (F1
0.471), all at the 5 s gate persistence. It still fails the gate
on false positives, the isolation scores are worse than the
single-cue rules, and the AND control detects nothing by
construction. Every number here is pinned to
experiments/model/results_day23.json, which is generated from the
same evaluation code, and the Day 22 bests come from the demo's
BEST_RULES (the published record).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from demo.demo_data import BEST_RULES, VERDICT
from experiments.eval.analysis import add_deviations, false_positives
from experiments.eval.combined import add_combined, evaluate_combined
from experiments.eval.loader import load_session
from experiments.eval.signals import add_signals

HERE = Path(__file__).resolve().parent.parent
MODEL_DIR = HERE
RESULTS = MODEL_DIR / "results_day23.json"
FIXTURE_CSV = HERE / "tests" / "fixtures" / "rerun-synthetic.csv"
R2_TIMING = {"ready_s": 5.0, "hold_s": 10.0, "rest_s": 5.0}

PINNED = {
    ("head", "iso"): {"threshold": 2.0, "detected": 1,
                      "false_positives": 0, "f1": 0.333},
    ("head", "sum"): {"threshold": 1.0, "detected": 4,
                      "false_positives": 8, "f1": 0.471},
    ("head", "and_control"): {"threshold": 4.0, "detected": 0,
                              "false_positives": 0, "f1": 0.0},
    ("shoulder", "iso"): {"threshold": 0.5, "detected": 3,
                          "false_positives": 9, "f1": 0.353},
    ("shoulder", "sum"): {"threshold": 1.0, "detected": 5,
                          "false_positives": 8, "f1": 0.556},
    ("shoulder", "and_control"): {"threshold": 1.0, "detected": 0,
                                  "false_positives": 0, "f1": 0.0},
}


def _load_results():
    return json.loads(RESULTS.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def run1_evaluation(run1_frames):
    frames, session = run1_frames
    combined = add_combined(add_signals(frames, session))
    return evaluate_combined(combined, session)


def test_results_day23_matches_recomputation(run1_evaluation):
    saved = _load_results()["evaluation"]
    for cue in ("head", "shoulder"):
        for name, variant in run1_evaluation["cues"][cue]["variants"].items():
            saved_best = saved["cues"][cue]["variants"][name]["best"]
            assert variant["best"] == saved_best, (cue, name)


def test_run1_bests_pinned():
    saved = _load_results()["evaluation"]
    # Run 1 recorded no shoulder trial summary, so the shoulder
    # alignment is inferred from the head offset, not verified.
    assert saved["alignment_verified"] is False
    assert saved["cues"]["shoulder"]["alignment_verified"] is False
    assert saved["cues"]["head"]["alignment_verified"] is True
    for (cue, name), pinned in PINNED.items():
        best = saved["cues"][cue]["variants"][name]["best"]
        for key, value in pinned.items():
            assert best[key] == value, (cue, name, key)


def test_nothing_passes_gate_run1():
    saved = _load_results()["evaluation"]
    for cue in ("head", "shoulder"):
        for name, variant in saved["cues"][cue]["variants"].items():
            assert variant["best"]["pass"] is False, (cue, name)
            assert all(not row["pass"] for row in variant["rows"]), (cue, name)


def test_sum_beats_day22_bests():
    saved = _load_results()["evaluation"]
    for cue in ("head", "shoulder"):
        assert (saved["cues"][cue]["variants"]["sum"]["best"]["f1"]
                > BEST_RULES[cue]["f1"])


def test_sum_passes_on_fixture_at_threshold_1():
    # The Day 16 synthetic fixture has clean holds to target depth
    # and flat work: the joint score passes the gate there, which
    # proves the machinery can report a pass. The AND control still
    # detects nothing, even on perfect data, because the fixture
    # holds are single-cue too.
    session = load_session(str(FIXTURE_CSV))
    frames = add_combined(add_signals(add_deviations(session), session))
    result = evaluate_combined(frames, session, **R2_TIMING)
    for cue in ("head", "shoulder"):
        rows = result["cues"][cue]["variants"]["sum"]["rows"]
        row = next(r for r in rows if r["threshold"] == 1.0)
        assert row["detected"] == 5
        assert row["false_positives"] == 0
        assert row["pass"] is True
        and_best = result["cues"][cue]["variants"]["and_control"]["best"]
        assert and_best["detected"] == 0


def test_verdict_names_joint_best():
    assert "NOTHING passes the gate" in VERDICT
    assert "5/5" in VERDICT
    assert "8 false positives" in VERDICT
    assert "joint" in VERDICT


def test_sum_fp_events_are_shoulder_drift(run1_frames):
    # The sum's 8 false positives are the same shoulder-drift
    # episodes the jump-gated rule fires on: 7 start at the exact
    # same second, the 8th within one frame. The sum buys
    # detection, not clean work.
    frames, session = run1_frames
    combined = add_combined(add_signals(frames, session))
    sum_fp = false_positives(combined, "shoulder", 1.0,
                             dev_col="j_dev_sum")["events"]
    gated_fp = false_positives(combined, "shoulder", 1,
                               dev_col="s_dev_gated")["events"]
    assert len(sum_fp) == 8
    assert len(gated_fp) == 8
    sum_starts = sorted(e["start_s"] for e in sum_fp)
    gated_starts = sorted(e["start_s"] for e in gated_fp)
    exact = sum(1 for s in sum_starts if s in gated_starts)
    assert exact >= 7
    for s in sum_starts:
        assert min(abs(s - g) for g in gated_starts) < 0.2
