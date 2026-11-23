"""Day 22 tests: the full project in one place.

The demo now shows the whole journey (Days 1-22), the pipeline, and
every benchmark table. Every displayed headline number is pinned to
its source file: the root README Build Progress, BENCHMARK.md, or a
results JSON. Nothing in the new section may pass the gate on real
data; the Day 16 fixture is the one exception and must stay labeled
synthetic.
"""

from __future__ import annotations

import json
import py_compile
import re
from pathlib import Path

import pytest

from demo.demo_data import (
    BENCHMARK_TABLES,
    BEST_RULES,
    JOURNEY,
    PIPELINE,
    benchmark_frame,
    gate_verdict,
    journey_frame,
    pipeline_frame,
)

HERE = Path(__file__).resolve().parent.parent.parent.parent
README = HERE / "README.md"
BENCHMARK_MD = HERE / "experiments" / "model" / "BENCHMARK.md"
MODEL_DIR = HERE / "experiments" / "model"
APP_PY = HERE / "demo" / "app.py"
DEMO_DATA_PY = HERE / "demo" / "demo_data.py"

SOURCE_PATHS = {
    "README.md": README,
    "BENCHMARK.md": BENCHMARK_MD,
    "results.json": MODEL_DIR / "results.json",
    "benchmark.json": MODEL_DIR / "benchmark.json",
    "day11_comparison.json": MODEL_DIR / "day11_comparison.json",
    "results_day13.json": MODEL_DIR / "results_day13.json",
    "results_day15.json": MODEL_DIR / "results_day15.json",
    "results_day16.json": MODEL_DIR / "results_day16.json",
    "results_day17.json": MODEL_DIR / "results_day17.json",
    "mlflow_comparison.json": MODEL_DIR / "mlflow_comparison.json",
}


def _load(name: str) -> dict:
    return json.loads((MODEL_DIR / name).read_text(encoding="utf-8"))


def _readme_sections() -> dict[int, str]:
    """Root README Build Progress, split per day heading."""
    text = README.read_text(encoding="utf-8")
    parts = re.split(r"### Day (\d+) \(", text)
    sections = {}
    for i in range(1, len(parts), 2):
        sections[int(parts[i])] = parts[i + 1]
    return sections


# --- journey ---------------------------------------------------------


def test_journey_has_22_days_in_order():
    assert [row["day"] for row in JOURNEY] == list(range(1, 23))
    assert len({row["day"] for row in JOURNEY}) == 22


def test_journey_dates_match_readme():
    text = README.read_text(encoding="utf-8")
    for row in JOURNEY:
        heading = f"### Day {row['day']} ({row['date']})"
        assert heading in text, f"missing README heading: {heading}"


def test_journey_test_counts_match_readme():
    sections = _readme_sections()
    for row in JOURNEY:
        if row["tests"] is None:
            continue
        section = sections[row["day"]]
        assert f"{row['tests']}/{row['tests']}" in section
        assert "tests passing" in section


def test_journey_results_are_nonempty():
    for row in JOURNEY:
        assert row["title"].strip()
        assert row["built"].strip()
        assert row["result"].strip()
        assert row["date"].startswith("Nov ")


def test_journey_sources_are_real_files():
    for row in JOURNEY:
        assert row["source"] in SOURCE_PATHS, f"unknown source: {row['source']}"
        assert SOURCE_PATHS[row["source"]].exists()


def test_journey_frame_columns():
    df = journey_frame()
    assert list(df.columns) == ["day", "date", "title", "built", "result", "tests", "source"]
    assert len(df) == 22
    assert list(df["day"]) == list(range(1, 23))


# --- pipeline --------------------------------------------------------


def test_pipeline_has_eight_steps_in_order():
    assert [row["step"] for row in PIPELINE] == list(range(1, 9))
    assert [row["stage"] for row in PIPELINE] == [
        "Collect", "Evaluate", "Validate", "Learn",
        "Serve", "Track", "Package", "Demo",
    ]


def test_pipeline_covers_collect_to_demo():
    df = pipeline_frame()
    assert len(df) == 8
    for row in PIPELINE:
        assert row["what"].strip()
        assert row["tool"].strip()
        assert 1 <= row["day"] <= 22


# --- benchmark tables ------------------------------------------------


def _model_rows(json_rows):
    return {
        r["model"]: r for r in json_rows
    }


def test_day8_head_matches_benchmark_json():
    bench = _load("benchmark.json")
    want = _model_rows(bench["cues"]["head"]["feature_sets"]["old"]["rows"])
    for row in BENCHMARK_TABLES["day8_head"]["rows"]:
        src = want[row["model"]]
        assert row["accuracy"] == pytest.approx(src["accuracy"])
        assert row["f1"] == pytest.approx(src["f1_tilted"])
        assert row["detected"] == src["detected_trials"]
        assert row["false_positives"] == src["false_positive_events"]
        assert row["work_flagged"] == f"{src['work_windows_tilted']} / {src['work_windows_total']}"
        assert row["calm_accuracy"] == pytest.approx(src["calm_accuracy"])


def test_day8_shoulder_matches_benchmark_json():
    bench = _load("benchmark.json")
    want = _model_rows(bench["cues"]["shoulder"]["feature_sets"]["old"]["rows"])
    for row in BENCHMARK_TABLES["day8_shoulder"]["rows"]:
        src = want[row["model"]]
        assert row["accuracy"] == pytest.approx(src["accuracy"])
        assert row["f1"] == pytest.approx(src["f1_tilted"])
        assert row["detected"] == src["detected_trials"]
        assert row["false_positives"] == src["false_positive_events"]
        assert row["work_flagged"] == f"{src['work_windows_tilted']} / {src['work_windows_total']}"
        assert row["calm_accuracy"] == pytest.approx(src["calm_accuracy"])


def test_day11_signals_match_benchmark_md():
    text = BENCHMARK_MD.read_text(encoding="utf-8")
    for row in BENCHMARK_TABLES["day11_signals"]["rows"]:
        line = (
            f"| {row['signal']} | {row['threshold_deg']} deg | "
            f"{row['detected']}/5 | {row['false_positives']} |"
        )
        assert line in text, f"signal row missing from BENCHMARK.md: {line}"


def test_day12_shoulder_new_matches_benchmark_json():
    bench = _load("benchmark.json")
    want = _model_rows(bench["cues"]["shoulder"]["feature_sets"]["new"]["rows"])
    for row in BENCHMARK_TABLES["day12_shoulder_new"]["rows"]:
        src = want[row["model"]]
        assert row["accuracy"] == pytest.approx(src["accuracy"])
        assert row["f1"] == pytest.approx(src["f1_tilted"])
        assert row["detected"] == src["detected_trials"]
        assert row["false_positives"] == src["false_positive_events"]
        assert row["work_flagged"] == f"{src['work_windows_tilted']} / {src['work_windows_total']}"
        assert row["calm_accuracy"] == pytest.approx(src["calm_accuracy"])


def test_day13_served_matches_results_day13():
    day13 = _load("results_day13.json")
    for row in BENCHMARK_TABLES["day13_served"]["rows"]:
        cue = day13["cues"][row["cue"]]
        assert len(cue["features"]) == row["features"]
        lr = cue["models"]["logistic_regression"]
        assert row["accuracy"] == pytest.approx(lr["oof_metrics"]["accuracy"])
        assert row["f1"] == pytest.approx(lr["oof_metrics"]["f1_tilted"])
        assert row["detected"] == lr["oof_trials"]["detected_trials"]
        assert row["false_positives"] == lr["oof_trials"]["false_positive_events"]


def test_day16_fixture_matches_results_day16():
    day16 = _load("results_day16.json")
    fixture = BENCHMARK_TABLES["day16_fixture"]
    val = day16["validation"]
    assert fixture["frames"] == val["frames"]
    assert fixture["tracked_fraction"] == pytest.approx(val["tracked_fraction"])
    assert fixture["work_s"] == pytest.approx(val["work_s"])
    reached = sum(1 for h in val["holds"] if h["reached_target"])
    assert fixture["holds_reached_target"] == f"{reached} / {len(val['holds'])}"
    assert fixture["verdict"] == day16["verdict"]
    assert "5/5, 0 false positives, F1 1.000" in fixture["head_grid_best"]


# --- honest framing --------------------------------------------------


def test_no_real_row_passes_the_gate():
    for key in ("day8_head", "day8_shoulder", "day12_shoulder_new", "day13_served"):
        for row in BENCHMARK_TABLES[key]["rows"]:
            verdict = gate_verdict(row["detected"], row["false_positives"])
            assert verdict["passes"] is False, f"{key} {row} passes the gate"
    for row in BENCHMARK_TABLES["day11_signals"]["rows"]:
        assert gate_verdict(row["detected"], row["false_positives"])["passes"] is False
    for cue, rule in BEST_RULES.items():
        assert gate_verdict(rule["detected"], rule["false_positives"])["passes"] is False


def test_day16_fixture_is_labeled_synthetic():
    fixture = BENCHMARK_TABLES["day16_fixture"]
    assert fixture["synthetic"] is True
    assert "synthetic" in fixture["title"]
    assert "not a real capture" in fixture["title"]


# --- wiring ----------------------------------------------------------


def test_app_calls_full_project_section():
    py_compile.compile(str(APP_PY), doraise=True)
    py_compile.compile(str(DEMO_DATA_PY), doraise=True)
    text = APP_PY.read_text(encoding="utf-8")
    assert "def _full_project() -> None:" in text
    assert "    _full_project()\n" in text
    assert "from demo_data import" in text
    assert "benchmark_frame" in text
    assert "journey_frame" in text
    assert "pipeline_frame" in text


def test_benchmark_frame_rejects_unknown_table():
    with pytest.raises(ValueError, match="unknown benchmark table"):
        benchmark_frame("day99")
