"""PoseGuard demo data: pure functions behind the Day 18 Streamlit showcase.

No Streamlit import here, so the test suite can exercise this module
without a server. The UI lives in demo/app.py and only calls these
helpers. Every number below is pinned to a results file, never invented:

- shoulder best rule and model rows: experiments/model/results_day17.json
- head best rule and model rows: experiments/model/results_day15.json
- trial dwell rows: the rule_grid hold_dwells of those same files.

Sample columns: head is the circular deviation from the calibration
baseline on tracking-valid frames only; shoulder is the Day 11
jump-gated deviation (NaN on tracker-noise frames). Both are NaN where
the tracker was lost, so the chart has honest gaps instead of invented
values. One 6-second tracker meltdown survives in the sample at 447 s
(head -94.95 deg, shoulder +141.74 deg, mid-work); the app clips the
chart axis at +/-25 deg and says so, rather than letting one meltdown
hide the 2-7 deg holds.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DEMO_DIR = Path(__file__).resolve().parent
SAMPLE_CSV = DEMO_DIR / "data" / "run1_sample.csv"

SAMPLE_COLUMNS = ["elapsed_s", "step", "head_dev_deg", "shoulder_dev_deg"]
KNOWN_STEPS = {"calib", "head", "shoulder", "work", "loss", "results"}

# The promotion gate, unchanged since Day 3: detect at least 4 of 5
# trials with zero false positives on the work sample.
GATE_TRIALS_TOTAL = 5
GATE_MIN_DETECTED = 4
GATE_MAX_FALSE_POSITIVES = 0

# Operating thresholds shown on the charts (session thresholds from Run 1).
HEAD_THRESHOLD_DEG = 8
SHOULDER_THRESHOLD_DEG = 6

# Best rule-grid configuration per cue (threshold deg / persistence s /
# velocity cap deg/s -> trials detected, false positives, trial F1).
BEST_RULES = {
    "head": {
        "threshold_deg": 4,
        "persist_s": 2.0,
        "vel_cap_deg_s": 10.0,
        "detected": 2,
        "trials_total": 5,
        "false_positives": 3,
        "f1": 0.400,
        "source": "results_day15.json",
    },
    "shoulder": {
        "threshold_deg": 1,
        "persist_s": 5.0,
        "vel_cap_deg_s": 5.0,
        "detected": 4,
        "trials_total": 5,
        "false_positives": 8,
        "f1": 0.471,
        "source": "results_day17.json",
    },
}

# Model scoreboard, latest feature set per cue (head: Day 15 new features
# vel_h/dwell_h; shoulder: Day 17 new features vel_s/dwell_s). The rule
# baseline rows are the operating-threshold baselines from the same days.
SCOREBOARD = [
    {
        "cue": "head",
        "model": "rule baseline (8 deg / 5 s)",
        "f1": 0.0,
        "detected": 0,
        "trials_total": 5,
        "false_positives": 0,
        "work_flagged": None,
        "calm_accuracy": None,
    },
    {
        "cue": "head",
        "model": "logistic_regression",
        "f1": 0.097,
        "detected": 2,
        "trials_total": 5,
        "false_positives": 4,
        "work_flagged": "167 / 318",
        "calm_accuracy": 0.448,
    },
    {
        "cue": "head",
        "model": "random_forest",
        "f1": 0.0,
        "detected": 0,
        "trials_total": 5,
        "false_positives": 0,
        "work_flagged": "0 / 318",
        "calm_accuracy": 0.899,
    },
    {
        "cue": "head",
        "model": "mlp",
        "f1": 0.105,
        "detected": 2,
        "trials_total": 5,
        "false_positives": 3,
        "work_flagged": "99 / 318",
        "calm_accuracy": 0.605,
    },
    {
        "cue": "shoulder",
        "model": "rule baseline (6 deg / 5 s)",
        "f1": 0.0,
        "detected": 0,
        "trials_total": 5,
        "false_positives": 7,
        "work_flagged": None,
        "calm_accuracy": None,
    },
    {
        "cue": "shoulder",
        "model": "logistic_regression",
        "f1": 0.172,
        "detected": 3,
        "trials_total": 5,
        "false_positives": 3,
        "work_flagged": "89 / 318",
        "calm_accuracy": 0.618,
    },
    {
        "cue": "shoulder",
        "model": "random_forest",
        "f1": 0.0,
        "detected": 0,
        "trials_total": 5,
        "false_positives": 0,
        "work_flagged": "0 / 318",
        "calm_accuracy": 0.89,
    },
    {
        "cue": "shoulder",
        "model": "mlp",
        "f1": 0.172,
        "detected": 3,
        "trials_total": 5,
        "false_positives": 4,
        "work_flagged": "123 / 318",
        "calm_accuracy": 0.524,
    },
]

# Per-trial hold summary. dwell_s is the longest run above the dwell
# threshold for that cue (head: 4 deg, shoulder: 2 deg). Trial 5 is
# barely recorded in Run 1, so at most 4/5 trials are measurable.
HEAD_TRIALS = [
    {"trial": 1, "samples": 67, "mean_abs_dev": 3.22, "max_abs_dev": 7.05, "dwell_s": 0.9, "dwell_threshold_deg": 4},
    {"trial": 2, "samples": 66, "mean_abs_dev": 1.84, "max_abs_dev": 3.25, "dwell_s": 0.0, "dwell_threshold_deg": 4},
    {"trial": 3, "samples": 67, "mean_abs_dev": 4.28, "max_abs_dev": 5.72, "dwell_s": 2.2, "dwell_threshold_deg": 4},
    {"trial": 4, "samples": 67, "mean_abs_dev": 3.34, "max_abs_dev": 6.60, "dwell_s": 2.2, "dwell_threshold_deg": 4},
    {"trial": 5, "samples": 6, "mean_abs_dev": 0.58, "max_abs_dev": 0.89, "dwell_s": 0.0, "dwell_threshold_deg": 4},
]
SHOULDER_TRIALS = [
    {"trial": 1, "samples": 67, "mean_abs_dev": 2.82, "max_abs_dev": 4.33, "dwell_s": 5.9, "dwell_threshold_deg": 2},
    {"trial": 2, "samples": 66, "mean_abs_dev": 2.29, "max_abs_dev": 2.81, "dwell_s": 2.3, "dwell_threshold_deg": 2},
    {"trial": 3, "samples": 67, "mean_abs_dev": 7.05, "max_abs_dev": 58.23, "dwell_s": 5.8, "dwell_threshold_deg": 2},
    {"trial": 4, "samples": 67, "mean_abs_dev": 2.67, "max_abs_dev": 3.89, "dwell_s": 3.0, "dwell_threshold_deg": 2},
    {"trial": 5, "samples": 8, "mean_abs_dev": 0.98, "max_abs_dev": 1.87, "dwell_s": 0.0, "dwell_threshold_deg": 2},
]

VERDICT = (
    "NOTHING passes the gate. The best rule reaches 4/5 trials but "
    "carries 8 false positives, and the best model reaches F1 0.172. "
    "The holds are shallow as well as brief: the deepest head hold peaks "
    "at 7.1 deg against an 8 deg threshold no hold ever reaches."
)

# --- Day 19: purpose, how to use it, feedback -----------------------
# The demo stopped being a static showcase on Day 19, so the page says
# in plain words what PoseGuard is for and how to try it. Kept here
# (pure data, no Streamlit import) so tests can pin the copy.

PURPOSE_HEADING = "What PoseGuard is for"
PURPOSE_BODY = (
    "PoseGuard is on-device vision AI that watches head and shoulder "
    "angles while you work at a desk and nudges you when you slouch. "
    "It runs on your webcam, in your browser, and no video ever leaves "
    "your machine: the angles are computed from pose landmarks on the "
    "spot, and only the angles are kept. This page is the honest "
    "status report for that idea. The validation session below shows "
    "the real numbers, including the ones that do not flatter it: "
    "nothing passes the gate yet."
)

HOWTO_HEADING = "How to use this demo"
HOWTO_STEPS = [
    "Try it live: open the webcam check below, sit upright, and press "
    "Calibrate so the app learns your upright pose. Then tilt your head "
    "or drop one shoulder and hold it for 5 seconds. The gate fires when "
    "the hold lasts, and the live readouts show your angles as you move.",
    "If the live video does not start in your browser, use the snapshot "
    "check instead: take a photo, then take another while holding the "
    "pose. Each snapshot is one sample, and repeated snapshots build the "
    "same gate the video uses.",
    "Read the session below it: the real Run 1 validation session, gate "
    "verdicts per trial, and the model scoreboard. The chart axis is "
    "clipped at +/-25 deg and the 447 s tracker meltdown is disclosed, "
    "not deleted.",
    "The holds this project is chasing are 10 seconds at 8 deg (shoulder) "
    "and 15 deg (head). The holds in Run 1 moved the angle only about "
    "2.5 deg on the shoulder line, which is why the gate stays unpassed.",
]

FEEDBACK_HEADING = "Feedback"
FEEDBACK_URL = "https://github.com/dasarijashwanth/PoseGuard/issues"
FEEDBACK_BODY = (
    "Tried the live check and it fired on your everyday lean, or never "
    "fired at all? That is useful data: open an issue and say what "
    "happened, which check you used (live video or snapshot), and what "
    "pose you held. Bug reports, better thresholds, and a working "
    "camera re-run are all welcome."
)


def load_sample(path: str | Path = SAMPLE_CSV) -> pd.DataFrame:
    """Load the trimmed Run 1 sample (1 Hz, circular deviations).

    Raises ValueError on a missing file, wrong columns, or a
    non-monotonic time axis, so a broken sample fails loudly instead
    of rendering a misleading chart.
    """
    path = Path(path)
    if not path.exists():
        raise ValueError(f"sample CSV not found: {path}")
    df = pd.read_csv(path)
    if list(df.columns) != SAMPLE_COLUMNS:
        raise ValueError(f"expected columns {SAMPLE_COLUMNS}, got {list(df.columns)}")
    if df.empty:
        raise ValueError("sample CSV is empty")
    if not df["elapsed_s"].is_monotonic_increasing:
        raise ValueError("elapsed_s is not monotonic increasing")
    unknown = set(df["step"].unique()) - KNOWN_STEPS
    if unknown:
        raise ValueError(f"unknown step values in sample: {sorted(unknown)}")
    return df


def sample_stats(df: pd.DataFrame) -> dict:
    """One-line summary of a loaded sample, for the demo header."""
    return {
        "rows": len(df),
        "duration_s": round(float(df["elapsed_s"].iloc[-1] - df["elapsed_s"].iloc[0]), 1),
        "steps": sorted(df["step"].unique()),
        "max_abs_head_dev": round(float(df["head_dev_deg"].abs().max()), 2),
        "max_abs_shoulder_dev": round(float(df["shoulder_dev_deg"].abs().max()), 2),
    }


def gate_verdict(detected: int, false_positives: int,
                 trials_total: int = GATE_TRIALS_TOTAL) -> dict:
    """Apply the promotion gate: >= 4/5 trials, 0 false positives.

    Returns {"passes": bool, "reason": str}. The reason names the
    failing half so the demo never shows a bare red/green without
    the numbers behind it.
    """
    if detected >= GATE_MIN_DETECTED and false_positives <= GATE_MAX_FALSE_POSITIVES:
        return {
            "passes": True,
            "reason": f"{detected}/{trials_total} trials, {false_positives} false positives: passes the gate.",
        }
    if detected < GATE_MIN_DETECTED and false_positives > GATE_MAX_FALSE_POSITIVES:
        reason = (
            f"only {detected}/{trials_total} trials and "
            f"{false_positives} false positives: fails both halves of the gate."
        )
    elif detected < GATE_MIN_DETECTED:
        reason = (
            f"only {detected}/{trials_total} trials "
            f"(needs {GATE_MIN_DETECTED}): fails the detection half."
        )
    else:
        reason = (
            f"{detected}/{trials_total} trials but {false_positives} false "
            f"positives (needs {GATE_MAX_FALSE_POSITIVES}): fails the "
            "false-positive half."
        )
    return {"passes": False, "reason": reason}


def scoreboard_frame(cue: str | None = None) -> pd.DataFrame:
    """Scoreboard rows as a DataFrame, optionally filtered to one cue."""
    rows = [r for r in SCOREBOARD if cue is None or r["cue"] == cue]
    return pd.DataFrame(rows)


def trials_frame(cue: str) -> pd.DataFrame:
    """Per-trial hold table for one cue ("head" or "shoulder")."""
    if cue == "head":
        return pd.DataFrame(HEAD_TRIALS)
    if cue == "shoulder":
        return pd.DataFrame(SHOULDER_TRIALS)
    raise ValueError(f"unknown cue: {cue!r} (expected 'head' or 'shoulder')")


def best_rule(cue: str) -> dict:
    """Best rule-grid configuration for one cue (copy, so callers cannot mutate it)."""
    if cue not in BEST_RULES:
        raise ValueError(f"unknown cue: {cue!r} (expected 'head' or 'shoulder')")
    return dict(BEST_RULES[cue])


# --- Day 22: the full project in one place ---------------------------
# The live demo showed the Day 18 showcase and the Day 19/20 webcam
# check, but nothing from the first 17 days. Day 22 puts the whole
# project on the page: the pipeline, one row per day with its key
# result, and every benchmark table. Every number below is pinned to
# the root README Build Progress, BENCHMARK.md, or a results JSON,
# never invented. Tests in experiments/model/tests/test_day22.py
# assert each one against its source file.

# The pipeline, in order. "day" is the day the step first landed.
PIPELINE = [
    {"step": 1, "stage": "Collect", "day": 1, "tool": "React demo shell + MediaPipe pose landmarks",
     "what": "Webcam landmarks in the browser, exported as angle CSVs. Day 14 records all 33 landmarks per frame."},
    {"step": 2, "stage": "Evaluate", "day": 3, "tool": "experiments/eval/ (pandas)",
     "what": "Scores each cue against the CSV with circular angle math that handles the +/-180 deg wraparound."},
    {"step": 3, "stage": "Validate", "day": 4, "tool": "experiments/eval/stats.py",
     "what": "Wilson confidence intervals, false-positive rates, and threshold/persistence curves for every rule."},
    {"step": 4, "stage": "Learn", "day": 5, "tool": "experiments/model/ (scikit-learn, PyTorch)",
     "what": "One classifier per cue on 1-second windows, grouped 5-fold cross-validation by trial."},
    {"step": 5, "stage": "Serve", "day": 5, "tool": "experiments/model/api.py (FastAPI)",
     "what": "GET /health and POST /predict. Day 13 serves the gated models and adds the 5-second persistence gate."},
    {"step": 6, "stage": "Track", "day": 6, "tool": "experiments/model/tracking.py (MLflow)",
     "what": "Every training run logged to a local file store so runs are compared instead of overwritten."},
    {"step": 7, "stage": "Package", "day": 6, "tool": "Dockerfile, docker-compose, deploy/",
     "what": "Slim Docker image for the API; Render (Day 9, free) and AWS Fargate (Day 10, about $25/month) deploy prep."},
    {"step": 8, "stage": "Demo", "day": 18, "tool": "demo/ (Streamlit)",
     "what": "Public demo: Run 1 showcase, live webcam check, and this full-project section (Day 22)."},
]

# One row per day. "tests" is the cumulative suite total after that
# day (None for Day 7, a docs-only day). "result" is the one-line
# headline from the root README Build Progress entry for that day.
JOURNEY = [
    {"day": 1, "date": "Nov 2", "title": "Project scaffold and posture math",
     "built": "Vite + React + TypeScript scaffold; head and shoulder angle math plus EMA smoothing in src/posture/angles.ts.",
     "result": "14/14 Vitest tests passing.", "tests": 14, "source": "README.md"},
    {"day": 2, "date": "Nov 4", "title": "Camera pipeline",
     "built": "Webcam access with permission states, frame capture in a Web Worker, getUserMedia fallbacks, FPS counters.",
     "result": "About 15 checks per second; 36/36 Vitest tests passing.", "tests": 36, "source": "README.md"},
    {"day": 3, "date": "Nov 5", "title": "Evaluation harness",
     "built": "Python eval package (pandas): circular angle math, CSV loading, trial analysis, report generator, run on the real Nov 3 CSV.",
     "result": "Head 0/5 at every threshold; shoulder 0/5 with 7 false positives.", "tests": 21, "source": "README.md"},
    {"day": 4, "date": "Nov 6", "title": "Statistical validation",
     "built": "Wilson confidence intervals, false-positive rates, and threshold/persistence curves.",
     "result": "Head salvageable, shoulder metric wrong for desk work; head Wilson 95% CI 0-43%.", "tests": 33, "source": "README.md"},
    {"day": 5, "date": "Nov 7", "title": "Classifier and FastAPI service",
     "built": "Logistic regression and 200-tree random forest per cue, grouped 5-fold CV; FastAPI GET /health and POST /predict.",
     "result": "Head LR 1/5, 6 FP, F1 0.078; shoulder LR 5/5, 5 FP, F1 0.179 (stillness artifact).", "tests": 59, "source": "results.json"},
    {"day": 6, "date": "Nov 8", "title": "MLflow tracking and Docker",
     "built": "MLflow run logging and comparison; Docker image for the API plus MLflow UI.",
     "result": "Random forest about 93% accuracy but 0/5 trials on both cues.", "tests": 70, "source": "mlflow_comparison.json"},
    {"day": 7, "date": "Nov 9", "title": "README rewritten",
     "built": "Root README now leads with the Python evaluation pipeline, statistics, and classifier.",
     "result": "Docs-only day; no test count.", "tests": None, "source": "README.md"},
    {"day": 8, "date": "Nov 10", "title": "PyTorch MLP benchmark",
     "built": "Small PyTorch MLP per cue (32, 16) vs logistic regression and random forest, same grouped CV.",
     "result": "Head MLP F1 0.060, shoulder 0.133; neither beats logistic regression.", "tests": 79, "source": "benchmark.json"},
    {"day": 9, "date": "Nov 10", "title": "Render deploy prep",
     "built": "Render Blueprint, slim runtime-only Docker image, click-by-click deploy guide.",
     "result": "API verified locally on a Render-style port; no live URL yet.", "tests": 87, "source": "README.md"},
    {"day": 10, "date": "Nov 12", "title": "AWS deploy prep",
     "built": "ECS Fargate task and service definitions, deploy script, deploy guide.",
     "result": "About $25 a month; never run end to end.", "tests": 99, "source": "README.md"},
    {"day": 11, "date": "Nov 13", "title": "New shoulder signal",
     "built": "Three angle-only candidates; the jump-gated deviation wins and becomes three new features.",
     "result": "Gated signal cuts false positives 7 to 1 at 6 deg; LR F1 0.179 to 0.219.", "tests": 115, "source": "day11_comparison.json"},
    {"day": 12, "date": "Nov 14", "title": "RF + MLP on the gated signal",
     "built": "Random forest and MLP on old vs new features, both cues, same grouped CV.",
     "result": "Shoulder MLP F1 0.215, 4/5 trials; random forest still 0/5.", "tests": 123, "source": "benchmark.json"},
    {"day": 13, "date": "Nov 15", "title": "Serve the gated models",
     "built": "Serving models retrained on gated features in artifacts_v2; POST /predict gains gate_s.",
     "result": "Shoulder LR F1 0.219, 5/5, 3 FP; head LR F1 0.078; gate verified end to end.", "tests": 150, "source": "results_day13.json"},
    {"day": 14, "date": "Nov 16", "title": "Camera re-run capture tool",
     "built": "Standalone capture page: all 33 landmarks per frame, 10 s holds, live depth targets (shoulder 8 deg, head 15 deg).",
     "result": "No captured data invented; the capture itself is the next step.", "tests": 166, "source": "README.md"},
    {"day": 15, "date": "Nov 17", "title": "Head cue rework",
     "built": "60-config rule grid (threshold x persistence x velocity cap) plus velocity and dwell features.",
     "result": "Best rule 4 deg / 2 s / 10 cap: 2/5, 3 FP, F1 0.400.", "tests": 194, "source": "results_day15.json"},
    {"day": 16, "date": "Nov 18", "title": "Run 2 re-benchmark harness",
     "built": "Validates a Run 2 capture, re-runs the head and shoulder evaluations on the new data with Run 2 timings.",
     "result": "Synthetic fixture passes the gate by construction; real capture still missing.", "tests": 211, "source": "results_day16.json"},
    {"day": 17, "date": "Nov 19", "title": "Shoulder cue rework",
     "built": "72-config rule grid plus velocity and dwell features.",
     "result": "Best rule 1 deg / 5 s / 5 cap: 4/5, 8 FP, F1 0.471.", "tests": 239, "source": "results_day17.json"},
    {"day": 18, "date": "Nov 20", "title": "Streamlit public demo showcase",
     "built": "Streamlit app with the real Run 1 session, gate verdicts per trial, and the model scoreboard.",
     "result": "727-row sample; tracker meltdown at 447 s disclosed, not deleted.", "tests": 258, "source": "README.md"},
    {"day": 19, "date": "Nov 21", "title": "Demo becomes a real app",
     "built": "Live webcam check (video + snapshot), purpose and how-to-use section, feedback link to GitHub issues.",
     "result": "Same numbers; the live check runs the same gate on your camera.", "tests": 286, "source": "README.md"},
    {"day": 20, "date": "Nov 22", "title": "MediaPipe loads on Streamlit Cloud",
     "built": "packages.txt system libraries, pinned requirements, per-dependency probing with a diagnostic expander.",
     "result": "Root cause was cv2/libGL, not MediaPipe; both webcam tabs verified live.", "tests": 302, "source": "README.md"},
    {"day": 21, "date": "Nov 23", "title": "Harness stress test",
     "built": "Messy inputs fed to the Day 16 harness: malformed CSVs, NaN and inf values, ragged rows, a 19.3 MB file.",
     "result": "11 bugs fixed; harness now fails gracefully; suite 324/324.", "tests": 324, "source": "README.md"},
    {"day": 22, "date": "Nov 23", "title": "Full project in one place",
     "built": "This section: the whole journey, the pipeline, and every benchmark table in the live demo.",
     "result": "Every number pinned to its source file; nothing passes the gate.", "tests": 342, "source": "README.md"},
]

# Benchmark tables, headline rows only. Each row carries its source
# file; the Day 22 tests assert every figure against that file.
# "work_flagged" is "flagged / total" work windows; None means the
# source table does not report it.
BENCHMARK_TABLES = {
    # Day 8, benchmark.json old feature sets (head new == old).
    "day8_head": {
        "title": "Day 8: MLP benchmark, head (383 windows, 23 tilted)",
        "source": "benchmark.json",
        "rows": [
            {"model": "logistic_regression", "accuracy": 0.446, "f1": 0.078, "detected": 1, "false_positives": 6, "work_flagged": "172 / 318", "calm_accuracy": 0.427},
            {"model": "random_forest", "accuracy": 0.932, "f1": 0.0, "detected": 0, "false_positives": 0, "work_flagged": "1 / 318", "calm_accuracy": 0.895},
            {"model": "mlp", "accuracy": 0.671, "f1": 0.06, "detected": 1, "false_positives": 4, "work_flagged": "100 / 318", "calm_accuracy": 0.685},
        ],
    },
    "day8_shoulder": {
        "title": "Day 8: MLP benchmark, shoulder (385 windows, 29 tilted)",
        "source": "benchmark.json",
        "rows": [
            {"model": "logistic_regression", "accuracy": 0.548, "f1": 0.179, "detected": 5, "false_positives": 5, "work_flagged": "134 / 318", "calm_accuracy": 0.447},
            {"model": "random_forest", "accuracy": 0.925, "f1": 0.0, "detected": 0, "false_positives": 0, "work_flagged": "0 / 318", "calm_accuracy": 0.898},
            {"model": "mlp", "accuracy": 0.764, "f1": 0.133, "detected": 2, "false_positives": 1, "work_flagged": "54 / 318", "calm_accuracy": 0.663},
        ],
    },
    # Day 11, BENCHMARK.md signal table (5 s persistence, 5 trials).
    # No JSON carries these numbers; the tests pin them to the table text.
    "day11_signals": {
        "title": "Day 11: shoulder signal candidates (5 s persistence)",
        "source": "BENCHMARK.md",
        "rows": [
            {"signal": "old level", "threshold_deg": 1, "detected": 4, "false_positives": 19},
            {"signal": "old level", "threshold_deg": 6, "detected": 0, "false_positives": 7},
            {"signal": "old level", "threshold_deg": 8, "detected": 0, "false_positives": 4},
            {"signal": "gated (jump gate, 20 deg / 1 s)", "threshold_deg": 1, "detected": 4, "false_positives": 8},
            {"signal": "gated", "threshold_deg": 6, "detected": 0, "false_positives": 1},
            {"signal": "gated", "threshold_deg": 8, "detected": 0, "false_positives": 0},
        ],
    },
    # Day 12, benchmark.json shoulder new feature sets.
    "day12_shoulder_new": {
        "title": "Day 12: shoulder, new gated features (11)",
        "source": "benchmark.json",
        "rows": [
            {"model": "logistic_regression", "accuracy": 0.61, "f1": 0.219, "detected": 5, "false_positives": 3, "work_flagged": "109 / 318", "calm_accuracy": 0.585},
            {"model": "random_forest", "accuracy": 0.919, "f1": 0.0, "detected": 0, "false_positives": 0, "work_flagged": "0 / 318", "calm_accuracy": 0.89},
            {"model": "mlp", "accuracy": 0.696, "f1": 0.215, "detected": 4, "false_positives": 3, "work_flagged": "82 / 318", "calm_accuracy": 0.638},
        ],
    },
    # Day 13, results_day13.json served logistic regressions.
    "day13_served": {
        "title": "Day 13: served models (artifacts_v2)",
        "source": "results_day13.json",
        "rows": [
            {"cue": "head", "model": "logistic_regression", "features": 6, "accuracy": 0.446, "f1": 0.078, "detected": 1, "false_positives": 6},
            {"cue": "shoulder", "model": "logistic_regression", "features": 11, "accuracy": 0.61, "f1": 0.219, "detected": 5, "false_positives": 3},
        ],
    },
    # Day 16, results_day16.json synthetic fixture. Labeled synthetic
    # everywhere it appears: the gate passes here by construction, and
    # that says nothing about the real capture.
    "day16_fixture": {
        "title": "Day 16: re-benchmark harness on the synthetic fixture (not a real capture)",
        "source": "results_day16.json",
        "synthetic": True,
        "frames": 5150,
        "tracked_fraction": 0.981,
        "work_s": 299.9,
        "holds_reached_target": "10 / 10",
        "head_grid_best": "4 deg / 5 s / no cap: 5/5, 0 false positives, F1 1.000",
        "shoulder_gated_best": "1 deg: 5/5, 0 false positives",
        "verdict": "Head passes the gate: rule grid best (4 deg, 5.0 s, velocity cap none), logistic_regression (old features), random_forest (old features), mlp (old features), logistic_regression (new features), random_forest (new features), mlp (new features). Shoulder passes the gate: gated signal best (1 deg), logistic_regression (new features), random_forest (new features).",
    },
}

JOURNEY_HEADING = "The full project in one place"
JOURNEY_INTRO = (
    "Everything PoseGuard has done, Days 1 to 22, on one page: the "
    "pipeline, one row per day with its key result, and every benchmark "
    "table. Every number is pinned to a results file or the benchmark "
    "document, and the honest verdict has not changed since Day 3: "
    "nothing passes the gate on the real data. The Day 16 fixture is "
    "the one exception, and it is labeled synthetic, because it passes "
    "by construction."
)


def journey_frame() -> pd.DataFrame:
    """The Day 1-22 journey as a DataFrame, in day order."""
    return pd.DataFrame(JOURNEY)


def pipeline_frame() -> pd.DataFrame:
    """The pipeline stages as a DataFrame, in step order."""
    return pd.DataFrame(PIPELINE)


def benchmark_frame(table: str) -> pd.DataFrame:
    """One benchmark table as a DataFrame (copy, so callers cannot mutate it)."""
    if table not in BENCHMARK_TABLES:
        raise ValueError(f"unknown benchmark table: {table!r} (expected one of {sorted(BENCHMARK_TABLES)})")
    entry = BENCHMARK_TABLES[table]
    if "rows" in entry:
        return pd.DataFrame(entry["rows"])
    raise ValueError(f"benchmark table {table!r} has no rows")
