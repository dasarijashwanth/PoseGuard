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
