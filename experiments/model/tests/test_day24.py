"""Day 24 tests: the capture-to-benchmark pipeline.

The pipeline runs on the same committed synthetic fixture as
Day 16 (experiments/model/tests/fixtures/rerun-synthetic.csv,
clean holds to target depth, flat work), so the gate passes by
construction: the tests prove the pipeline validates, runs the
Day 16 re-benchmark and the Day 23 combined gate, and archives
both verdicts. They say nothing about a real capture.

Also pinned: --drop-dir discovery picks the newest CSV and its
same-stem manifest, a corrupted capture exits 2 and archives
nothing, a mismatched manifest exits 2 at the pre-flight, and
the pipeline never touches the Day 16 artifacts
(results_day16.json, BENCHMARK.md).
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import pytest

from experiments.eval.tests.test_rerun import build_rerun_csv
from experiments.model.day24 import discover_inputs, main

HERE = Path(__file__).resolve().parent.parent
FIXTURE_CSV = HERE / "tests" / "fixtures" / "rerun-synthetic.csv"
FIXTURE_SOURCE = "synthetic fixture (Day 14 builder)"

# The fixture's Day 23 combined bests at the 5 s gate persistence,
# Run 2 timings: isolation and the normalized sum pass both cues,
# the AND control detects nothing, exactly as Day 23 found.
PINNED_COMBINED = {
    ("head", "iso"): {"threshold": 2.0, "detected": 5,
                      "false_positives": 0, "f1": 1.0, "pass": True},
    ("head", "sum"): {"threshold": 0.5, "detected": 5,
                      "false_positives": 0, "f1": 1.0, "pass": True},
    ("head", "and_control"): {"threshold": 4.0, "detected": 0,
                              "false_positives": 0, "f1": 0.0,
                              "pass": False},
    ("shoulder", "iso"): {"threshold": 0.5, "detected": 5,
                          "false_positives": 0, "f1": 1.0,
                          "pass": True},
    ("shoulder", "sum"): {"threshold": 0.5, "detected": 5,
                          "false_positives": 0, "f1": 1.0,
                          "pass": True},
    ("shoulder", "and_control"): {"threshold": 1.0, "detected": 0,
                                  "false_positives": 0, "f1": 0.0,
                                  "pass": False},
}


def _manifest_dict(frames=5150):
    """A manifest matching the fixture (the Day 16 test pattern)."""
    return {
        "tool": "poseguard-rerun-capture",
        "protocol": "Run 2 re-run capture",
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


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def csv_run(tmp_path_factory):
    """One full --csv pipeline run on the fixture.

    Hashes of the Day 16 artifacts are captured before the run so
    a test can prove the pipeline left them untouched.
    """
    work = tmp_path_factory.mktemp("day24")
    manifest = work / "manifest.json"
    manifest.write_text(json.dumps(_manifest_dict()), encoding="utf-8")
    out_dir = work / "runs"
    before = {name: _hash(HERE / name)
              for name in ("results_day16.json", "BENCHMARK.md")}
    code = main(["--csv", str(FIXTURE_CSV),
                 "--manifest", str(manifest),
                 "--source", FIXTURE_SOURCE,
                 "--out-dir", str(out_dir)])
    run_dirs = sorted(out_dir.iterdir()) if out_dir.exists() else []
    return {"code": code, "run_dirs": run_dirs, "before": before}


def _archive(csv_run):
    assert csv_run["code"] == 0
    assert len(csv_run["run_dirs"]) == 1
    run_dir = csv_run["run_dirs"][0]
    results = json.loads((run_dir / "results.json").read_text(
        encoding="utf-8"))
    summary = (run_dir / "summary.md").read_text(encoding="utf-8")
    return results, summary


def test_end_to_end_csv_archives_both_verdicts(csv_run):
    results, _ = _archive(csv_run)
    assert results["source"] == FIXTURE_SOURCE
    assert results["csv"] == str(FIXTURE_CSV)
    assert results["manifest"] is not None
    val = results["validation"]
    assert val["ok"] is True
    assert val["frames"] == 5150
    assert val["tracked_fraction"] == pytest.approx(0.981, abs=0.001)
    assert val["warnings"] == []
    verdict = results["day16"]["verdict"]
    assert verdict.startswith("Head passes the gate:")
    assert "Shoulder passes the gate:" in verdict
    assert "nothing passes the gate" not in verdict


def test_combined_bests_pinned_on_fixture(csv_run):
    results, _ = _archive(csv_run)
    combined = results["combined"]
    assert combined["alignment_verified"] is True
    for (cue, name), pinned in PINNED_COMBINED.items():
        best = combined["cues"][cue]["variants"][name]["best"]
        for key, value in pinned.items():
            assert best[key] == value, (cue, name, key)


def test_summary_names_verdicts_and_combined_lines(csv_run):
    results, summary = _archive(csv_run)
    assert FIXTURE_SOURCE in summary
    assert results["day16"]["verdict"] in summary
    assert ("- head: sum at threshold 0.5, 5/5 trials, "
            "0 false positives, F1 1.000: passes the gate."
            in summary)
    assert ("- shoulder: sum at threshold 0.5, 5/5 trials, "
            "0 false positives, F1 1.000: passes the gate."
            in summary)
    assert "| head | and_control | 4.0 | 0/5 | 0 | 0.000 | no |" in summary
    assert "Warnings: none." in summary


def test_day16_artifacts_untouched(csv_run):
    _archive(csv_run)
    for name, digest in csv_run["before"].items():
        assert _hash(HERE / name) == digest, name


def test_drop_dir_discovers_newest_pair(tmp_path):
    older = tmp_path / "capture-old.csv"
    newer = tmp_path / "capture-new.csv"
    older.write_text("old", encoding="utf-8")
    newer.write_text("new", encoding="utf-8")
    manifest = tmp_path / "capture-new.json"
    manifest.write_text("{}", encoding="utf-8")
    stray = tmp_path / "notes.json"
    stray.write_text("{}", encoding="utf-8")
    os.utime(older, (1000, 1000))
    os.utime(stray, (1500, 1500))
    os.utime(manifest, (2000, 2000))
    os.utime(newer, (3000, 3000))
    csv_path, manifest_path = discover_inputs(str(tmp_path))
    assert csv_path == str(newer)
    # Same stem wins over the stray .json.
    assert manifest_path == str(manifest)
    manifest.unlink()
    csv_path, manifest_path = discover_inputs(str(tmp_path))
    assert csv_path == str(newer)
    # No same-stem manifest: the newest .json is used.
    assert manifest_path == str(stray)
    stray.unlink()
    csv_path, manifest_path = discover_inputs(str(tmp_path))
    assert manifest_path is None


def test_drop_dir_end_to_end(tmp_path):
    drop = tmp_path / "drop"
    drop.mkdir()
    shutil.copy(FIXTURE_CSV, drop / "capture.csv")
    (drop / "capture.json").write_text(
        json.dumps(_manifest_dict()), encoding="utf-8")
    out_dir = tmp_path / "runs"
    code = main(["--drop-dir", str(drop),
                 "--source", FIXTURE_SOURCE,
                 "--out-dir", str(out_dir)])
    assert code == 0
    run_dirs = sorted(out_dir.iterdir())
    assert len(run_dirs) == 1
    results = json.loads((run_dirs[0] / "results.json").read_text(
        encoding="utf-8"))
    assert results["csv"] == str(drop / "capture.csv")
    assert results["manifest"] == str(drop / "capture.json")
    assert "Head passes the gate:" in results["day16"]["verdict"]


def test_corrupted_csv_exits_2_and_archives_nothing(tmp_path):
    path = build_rerun_csv(tmp_path, drop_columns=("lm_24_y",))
    out_dir = tmp_path / "runs"
    code = main(["--csv", path, "--out-dir", str(out_dir)])
    assert code == 2
    assert not out_dir.exists() or not list(out_dir.iterdir())


def test_manifest_mismatch_exits_2_at_preflight(tmp_path):
    bad = tmp_path / "manifest.json"
    bad.write_text(json.dumps(_manifest_dict(frames=9999)),
                   encoding="utf-8")
    out_dir = tmp_path / "runs"
    code = main(["--csv", str(FIXTURE_CSV),
                 "--manifest", str(bad),
                 "--out-dir", str(out_dir)])
    assert code == 2
    assert not out_dir.exists() or not list(out_dir.iterdir())
