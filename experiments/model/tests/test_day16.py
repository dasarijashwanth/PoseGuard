"""Day 16 tests: Run 2 re-benchmark harness.

The harness runs on a committed synthetic fixture
(experiments/model/tests/fixtures/rerun-synthetic.csv, built by the
Day 14 builder in experiments.eval.tests.test_rerun): a full Run 2
session with clean holds to target depth and flat work. On that
fixture the gate passes by construction, which is the point: it
proves the harness can report a pass, and the verdict function is
tested separately on fabricated failing numbers so it cannot hide
a real failure behind fixture optimism.

Also pinned: the Run 1 trial timings mislabel the fixture (30
tilted windows instead of 45), which is why the shared evaluation
takes the Run 2 timings as arguments; and validation rejects a
bad file before any benchmarking happens.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.eval.analysis import add_deviations
from experiments.eval.loader import load_session
from experiments.eval.rerun import validate_rerun_csv
from experiments.eval.signals import WINNER_COL, add_signals
from experiments.eval.tests.test_rerun import build_rerun_csv
from experiments.model.day16 import (
    DAY16_MARKER,
    check_manifest,
    main,
    run_day16,
    validate_capture,
    verdict_day16,
)
from experiments.model.features import build_windows

HERE = Path(__file__).resolve().parent.parent
FIXTURE_CSV = HERE / "tests" / "fixtures" / "rerun-synthetic.csv"
FIXTURE_SOURCE = ("synthetic fixture (experiments.eval.tests.test_rerun "
                  "builder), not a real capture")


def _manifest_dict(frames=5150, protocol="Run 2 re-run capture"):
    return {
        "tool": "poseguard-rerun-capture",
        "protocol": protocol,
        "exported": "2026-11-16T00:00:00.000Z",
        "engine": "Test Landmarker",
        "baseline_head_deg": 0,
        "baseline_shoulder_deg": 170,
        "protocol_settings": {
            "calib_s": 5, "ready_s": 5, "hold_s": 10, "rest_s": 5,
            "work_s": 300, "loss_s": 10, "trials_per_cue": 5,
            "target_head_deg": 15, "target_shoulder_deg": 8,
            "detect_persist_s": 5,
        },
        "frames": frames,
        "frames_tracked": 5050,
        "tracked_fraction": 0.981,
        "holds": [{"cue": cue, "hold": i, "duration_s": 10,
                   "frames": 100, "tracked_fraction": 1.0,
                   "max_abs_dev_deg": 16.0 if cue == "head" else 9.0,
                   "target_deg": 15 if cue == "head" else 8,
                   "detected": True}
                  for cue in ("head", "shoulder") for i in range(1, 6)],
        "gaps": [],
        "loss_test": "PASS",
    }


@pytest.fixture(scope="module")
def manifest_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("day16") / "manifest.json"
    path.write_text(json.dumps(_manifest_dict()), encoding="utf-8")
    return str(path)


@pytest.fixture(scope="module")
def day16_result(manifest_path):
    return run_day16(str(FIXTURE_CSV), manifest_path,
                     source=FIXTURE_SOURCE)


def test_fixture_validates_clean(day16_result):
    val = day16_result["validation"]
    assert val["frames"] == 5150
    assert val["tracked_fraction"] == pytest.approx(0.981, abs=0.001)
    assert val["work_s"] == pytest.approx(299.9, abs=0.2)
    assert val["warnings"] == []
    assert len(val["holds"]) == 10
    assert day16_result["source"] == FIXTURE_SOURCE


def test_validation_rejects_bad_file(tmp_path):
    path = build_rerun_csv(tmp_path, drop_columns=("lm_24_y",))
    report = validate_rerun_csv(path)
    assert not report["ok"]
    with pytest.raises(ValueError, match="cannot be used"):
        validate_capture(path)
    assert main(["--csv", path]) == 2


def test_manifest_mismatch_rejected(tmp_path):
    session = load_session(str(FIXTURE_CSV))
    report = validate_rerun_csv(str(FIXTURE_CSV))
    assert report["ok"]
    bad = tmp_path / "manifest.json"
    bad.write_text(json.dumps(_manifest_dict(frames=9999)),
                   encoding="utf-8")
    errors = check_manifest(str(bad), session, report)
    assert any("9999" in e for e in errors)
    with pytest.raises(ValueError, match="cannot be used"):
        validate_capture(str(FIXTURE_CSV), str(bad))


def test_manifest_matching_accepted(tmp_path, day16_result):
    session = load_session(str(FIXTURE_CSV))
    report = validate_rerun_csv(str(FIXTURE_CSV))
    good = tmp_path / "manifest.json"
    good.write_text(json.dumps(_manifest_dict()), encoding="utf-8")
    assert check_manifest(str(good), session, report) == []
    assert day16_result["manifest"] is not None


def test_head_grid_has_all_configurations(day16_result):
    grid = day16_result["cues"]["head"]["rule_grid"]
    assert len(grid["rows"]) == 5 * 3 * 4  # thresholds x persists x caps
    assert grid["alignment_verified"] is True
    assert grid["alignment_offset_s"] == pytest.approx(0.1, abs=0.05)


def test_head_grid_best_passes_on_fixture(day16_result):
    best = day16_result["cues"]["head"]["rule_grid"]["best"]
    assert best == {"threshold": 4, "persist_s": 5.0,
                    "vel_cap_deg_s": None, "detected": 5,
                    "trials_total": 5, "false_positives": 0,
                    "precision": 1.0, "recall": 1.0, "f1": 1.0}


def test_head_hold_dwells_are_full_holds(day16_result):
    dwells = day16_result["cues"]["head"]["rule_grid"]["hold_dwells"]
    assert len(dwells) == 5
    for row in dwells:
        assert row["samples"] == 100
        assert row["max_abs_dev"] == 16.0
        assert row["dwell_gt4_s"] == 9.9


def test_head_rule_baseline_is_counted_on_fixture(day16_result):
    baseline = day16_result["cues"]["head"]["rule_baseline"]
    assert baseline == {"detected_trials": 5, "false_positive_events": 0}


def test_head_models_all_pass_on_fixture(day16_result):
    sets = day16_result["cues"]["head"]["model_comparison"]["feature_sets"]
    assert day16_result["cues"]["head"]["n_windows"] == 400
    assert day16_result["cues"]["head"]["n_tilted_windows"] == 45
    for name in ("old", "new"):
        for row in sets[name]["rows"]:
            assert row["f1_tilted"] == 1.0
            assert row["detected_trials"] == 5
            assert row["false_positive_events"] == 0


def test_shoulder_gated_signal_best_passes_on_fixture(day16_result):
    signal = day16_result["cues"]["shoulder"]["signal"]
    assert signal["alignment_verified"] is True
    gated = signal["candidates"]["gated"]["best"]
    assert gated["detected"] == 5
    assert gated["false_positives"] == 0
    old = signal["old_level"]["best"]
    assert old["detected"] == 5
    assert old["false_positives"] == 0


def test_shoulder_rule_baseline_is_counted_on_fixture(day16_result):
    baseline = day16_result["cues"]["shoulder"]["rule_baseline"]
    assert baseline == {"detected_trials": 5, "false_positive_events": 0}


def test_shoulder_models_old_features_cannot_separate(day16_result):
    sets = day16_result["cues"]["shoulder"]["model_comparison"][
        "feature_sets"]
    old_lr = next(r for r in sets["old"]["rows"]
                  if r["model"] == "logistic_regression")
    assert old_lr["f1_tilted"] == 0.173
    assert old_lr["detected_trials"] == 4
    assert old_lr["false_positive_events"] == 1
    # Without the gated level stats the old features flag all work.
    assert old_lr["work_windows_tilted"] == 300
    new_lr = next(r for r in sets["new"]["rows"]
                  if r["model"] == "logistic_regression")
    assert new_lr["f1_tilted"] == 0.889
    assert new_lr["detected_trials"] == 4
    assert new_lr["false_positive_events"] == 0
    new_rf = next(r for r in sets["new"]["rows"]
                  if r["model"] == "random_forest")
    assert new_rf["f1_tilted"] == 1.0
    assert new_rf["detected_trials"] == 5
    assert new_rf["false_positive_events"] == 0


def test_verdict_names_the_passes_on_fixture(day16_result):
    verdict = day16_result["verdict"]
    assert verdict.startswith("Head passes the gate:")
    assert "Shoulder passes the gate:" in verdict
    assert "nothing passes the gate" not in verdict


def test_verdict_reports_no_pass_honestly():
    failing = {
        "cues": {
            "head": {
                "rule_grid": {"best": {"threshold": 8, "persist_s": 5.0,
                                       "vel_cap_deg_s": None,
                                       "detected": 2,
                                       "false_positives": 3}},
                "model_comparison": {"feature_sets": {
                    "old": {"rows": [
                        {"model": "logistic_regression",
                         "detected_trials": 2,
                         "false_positive_events": 3}]},
                    "new": {"rows": [
                        {"model": "logistic_regression",
                         "detected_trials": 2,
                         "false_positive_events": 3}]}}},
            },
            "shoulder": {
                "signal": {"candidates": {"gated": {"best": {
                    "threshold": 6, "detected": 1,
                    "false_positives": 2}}}},
                "model_comparison": {"feature_sets": {
                    "old": {"rows": [
                        {"model": "logistic_regression",
                         "detected_trials": 1,
                         "false_positive_events": 2}]},
                    "new": {"rows": [
                        {"model": "logistic_regression",
                         "detected_trials": 1,
                         "false_positive_events": 2}]}}},
            },
        },
    }
    verdict = verdict_day16(failing)
    assert verdict.startswith(
        "Head: nothing passes the gate (4/5 trials, "
        "0 false positives) on the new data.")
    assert ("Shoulder: nothing passes the gate (4/5 trials, "
            "0 false positives) on the new data." in verdict)


def test_run1_timings_mislabel_run2_windows():
    session = load_session(str(FIXTURE_CSV))
    frames = add_signals(add_deviations(session), session)
    frames["s_dev_new"] = frames[WINNER_COL]
    run1 = build_windows(frames, session, "head")
    run2 = build_windows(frames, session, "head",
                         ready_s=5.0, hold_s=10.0, rest_s=5.0)
    assert len(run1) == len(run2) == 400
    assert int((run1["label"] == 1).sum()) == 30
    assert int((run2["label"] == 1).sum()) == 45


def test_results_day16_json_schema():
    data = json.loads((HERE / "results_day16.json").read_text(
        encoding="utf-8"))
    assert data["source"] == FIXTURE_SOURCE
    assert data["frames"] == 5150
    assert set(data["cues"]) == {"head", "shoulder"}
    head = data["cues"]["head"]
    assert head["n_windows"] == 400
    assert head["n_tilted_windows"] == 45
    assert len(head["rule_grid"]["rows"]) == 60
    comp = head["model_comparison"]
    assert comp["old_features"] == [
        "mean_abs_h", "max_abs_h", "mean_h", "std_h",
        "frac_h_gt5", "jerk_h"]
    assert comp["new_features"] == comp["old_features"] + [
        "vel_h", "dwell_h"]
    for name in ("old", "new"):
        rows = comp["feature_sets"][name]["rows"]
        assert [r["model"] for r in rows] == [
            "logistic_regression", "random_forest", "mlp"]
    shoulder = data["cues"]["shoulder"]
    assert shoulder["n_windows"] == 400
    assert shoulder["n_tilted_windows"] == 45
    assert "Head passes the gate:" in data["verdict"]
    assert "Shoulder passes the gate:" in data["verdict"]


def test_write_report_keeps_earlier_sections_and_is_idempotent(tmp_path):
    from experiments.model.day16 import write_report

    data = json.loads((HERE / "results_day16.json").read_text(
        encoding="utf-8"))
    target = tmp_path / "BENCHMARK.md"
    target.write_text((HERE / "BENCHMARK.md").read_text(
        encoding="utf-8"), encoding="utf-8")
    write_report(data, target)
    text = target.read_text(encoding="utf-8")
    assert "# Day 8 benchmark" in text
    assert "## Day 11" in text
    assert "## Day 12" in text
    assert "## Day 15" in text
    assert text.count(DAY16_MARKER) == 1
    assert FIXTURE_SOURCE in text
    write_report(data, target)
    assert target.read_text(encoding="utf-8").count(DAY16_MARKER) == 1
