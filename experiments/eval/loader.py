"""Load and validate a PoseGuard validation export CSV.

File layout (written by the web experiment, see
experiments/webcam-validation-experiment.html):

    Line 0:    title line
    Lines 1-8: metadata, one "key,value" pair per line
               (export time, engine, baselines, thresholds,
                false-detection count, landmark-loss result)
    Line 9:    blank
    Line 10:   per-frame header:
               timestamp,elapsed_s,step,head_tilt,shoulder_tilt,
               cue_detected,tracking,head_deviation,shoulder_deviation
    Next:      one row per video frame. `step` is one of
               calib / head / shoulder / work / loss / results.
    Tail:      "Per-trial summary ..." section with one row per
               completed trial: cue,trial,detected,samples,
               head_mean,head_std,shoulder_mean,shoulder_std

Note: the deviation columns in the file were computed by the experiment
with plain subtraction, so near +/-180 they contain phantom jumps
(e.g. -351 degrees). This loader keeps them for comparison but the
analysis recomputes deviations from the raw angles with circular math.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd

REQUIRED_META = {
    "Exported",
    "Engine",
    "Baseline head (deg)",
    "Baseline shoulder (deg)",
    "Threshold head (deg)",
    "Threshold shoulder (deg)",
    "False detections (work sample)",
    "Landmark-loss test",
}

REQUIRED_COLUMNS = [
    "timestamp", "elapsed_s", "step", "head_tilt", "shoulder_tilt",
    "cue_detected", "tracking", "head_deviation", "shoulder_deviation",
]

KNOWN_STEPS = {"calib", "head", "shoulder", "work", "loss", "results"}

_TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:")


@dataclass
class SessionData:
    """One validation session, parsed and validated."""

    path: str
    meta: dict
    frames: pd.DataFrame          # one row per video frame
    trial_summary: list[dict] = field(default_factory=list)

    @property
    def baseline_head(self) -> float:
        return float(self.meta["Baseline head (deg)"])

    @property
    def baseline_shoulder(self) -> float:
        return float(self.meta["Baseline shoulder (deg)"])

    @property
    def threshold_head(self) -> float:
        return float(self.meta["Threshold head (deg)"])

    @property
    def threshold_shoulder(self) -> float:
        return float(self.meta["Threshold shoulder (deg)"])


def _parse_meta(lines: list[str]) -> dict:
    meta = {}
    for line in lines:
        if not line.strip():
            continue
        key, _, value = line.partition(",")
        meta[key.strip()] = value.strip().strip('"')
    missing = REQUIRED_META - set(meta)
    if missing:
        raise ValueError(f"missing metadata keys: {sorted(missing)}")
    return meta


def load_session(path: str) -> SessionData:
    """Read an export CSV, validate its schema, return a SessionData."""
    lines = open(path, encoding="utf-8").read().splitlines()
    if not lines:
        raise ValueError("empty file")

    # --- metadata: everything up to the first blank line ---
    try:
        blank = lines.index("")
    except ValueError:
        raise ValueError("no blank line after metadata block")
    meta = _parse_meta(lines[:blank])

    # --- frame header: first line starting with "timestamp," ---
    try:
        header_i = next(i for i, l in enumerate(lines) if l.startswith("timestamp,"))
    except StopIteration:
        raise ValueError("no per-frame header line (timestamp,elapsed_s,...)")
    header = [c.strip() for c in lines[header_i].split(",")]
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in header]
    if missing_cols:
        raise ValueError(f"missing frame columns: {missing_cols}")

    # --- frame rows: consecutive lines starting with an ISO timestamp ---
    frame_lines, summary_lines = [], []
    i = header_i + 1
    while i < len(lines) and _TIMESTAMP_RE.match(lines[i]):
        frame_lines.append(lines[i])
        i += 1
    summary_lines = lines[i:]
    if not frame_lines:
        raise ValueError("no per-frame data rows found")

    import io
    frames = pd.read_csv(io.StringIO("\n".join([lines[header_i]] + frame_lines)))
    for col in ["elapsed_s", "head_tilt", "shoulder_tilt",
                "head_deviation", "shoulder_deviation"]:
        frames[col] = pd.to_numeric(frames[col], errors="coerce")
    frames["tracking"] = pd.to_numeric(frames["tracking"], errors="coerce")
    frames["timestamp"] = pd.to_datetime(frames["timestamp"], errors="coerce")
    frames = frames.sort_values("elapsed_s").reset_index(drop=True)

    unknown = set(frames["step"].dropna().unique()) - KNOWN_STEPS
    if unknown:
        raise ValueError(f"unknown step values: {sorted(unknown)}")
    if frames["elapsed_s"].isna().any():
        raise ValueError("elapsed_s has missing values")
    if not frames["elapsed_s"].is_monotonic_increasing:
        raise ValueError("elapsed_s is not monotonic increasing")

    # --- per-trial summary section (may be empty for a cue) ---
    trial_summary = []
    for line in summary_lines:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 8 and parts[0] in ("head", "shoulder") and parts[1].isdigit():
            trial_summary.append({
                "cue": parts[0],
                "trial": int(parts[1]),
                "detected": parts[2] == "yes",
                "samples": int(parts[3]),
                "head_mean": float(parts[4]),
                "head_std": float(parts[5]),
                "shoulder_mean": float(parts[6]),
                "shoulder_std": float(parts[7]),
            })

    return SessionData(path=path, meta=meta, frames=frames,
                       trial_summary=trial_summary)
