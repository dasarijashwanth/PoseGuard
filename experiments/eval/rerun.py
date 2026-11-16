"""Validate a Run 2 re-run capture CSV (Day 14).

The capture tool (experiments/rerun-capture.html) writes a CSV that is
a superset of the Run 1 schema: the same metadata block and nine
frame columns, plus per-frame labels (cue, hold_id, phase) and all 33
pose landmarks. This module checks a captured file before it is used
for the Day 15 re-benchmark:

- schema: the Run 1 columns are all there, the Run 2 metadata keys
  are there, and every landmark column is there;
- structure: 5 holds per cue, each hold long enough and with enough
  tracked frames to test;
- data quality, reported as warnings, not errors: holds that never
  reached their depth target, holds with heavy tracking loss, a work
  sample cut short, and every tracking-loss gap of 1 s or more with
  its phase and hold context.

Errors mean the file cannot be used. Warnings mean it can be used
but the re-benchmark should know about the weak spots.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .loader import REQUIRED_COLUMNS, REQUIRED_META, load_session
from .rerun_protocol import (
    CUES,
    HOLD_S,
    PHASES,
    TRIALS_PER_CUE,
    WORK_S,
    frame_columns,
    landmark_columns,
    target_deg,
)

# Metadata keys the Run 2 tool writes on top of the Run 1 set.
RUN2_META_KEYS = {
    "Protocol",
    "Hold duration (s)",
    "Target head (deg)",
    "Target shoulder (deg)",
}

MIN_HOLD_DURATION_S = 8.0   # a 10 s hold must span at least 8 s of frames
MIN_HOLD_FRAMES = 50        # at ~12 fps, 10 s is about 120 frames
MIN_HOLD_TRACKED = 0.5      # below this, the hold is mostly lost
MIN_WORK_S = 240.0          # work sample cut short below 4 min is a warning
MIN_GAP_S = 1.0             # tracking-loss gaps of 1 s or more are listed


def tracking_gaps(frames: pd.DataFrame,
                  min_gap_s: float = MIN_GAP_S) -> list[dict]:
    """Runs of lost tracking spanning at least `min_gap_s` seconds.

    Each gap reports its start and end time, duration, and the phase,
    cue and hold it falls in (from the first lost frame of the run).
    """
    lost = frames["tracking"].to_numpy() != 1
    t = frames["elapsed_s"].to_numpy(dtype=float)
    gaps: list[dict] = []
    i = 0
    while i < len(frames):
        if not lost[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(frames) and lost[j + 1]:
            j += 1
        start, end = float(t[i]), float(t[j])
        # The gap ends when tracking resumes: the next frame's time.
        # A gap that runs to the last frame has no closing frame; use
        # the frame spacing as its tail instead of reporting 0 s.
        if j + 1 < len(frames):
            end = float(t[j + 1])
        elif len(frames) > 1:
            end = float(t[j]) + float(np.median(np.diff(t)))
        if end - start >= min_gap_s:
            row = frames.iloc[i]
            gaps.append({
                "start_s": round(start, 1),
                "end_s": round(end, 1),
                "duration_s": round(end - start, 1),
                "step": str(row["step"]),
                "phase": str(row["phase"]),
                "cue": str(row["cue"]) if pd.notna(row["cue"]) else "",
                "hold_id": (int(row["hold_id"])
                            if pd.notna(row["hold_id"]) else None),
            })
        i = j + 1
    return gaps


def _hold_stats(frames: pd.DataFrame, cue: str, hold_id: int) -> dict:
    """Duration, frame count, tracked fraction and peak deviation of one hold."""
    h = frames[(frames["cue"] == cue)
               & (frames["hold_id"] == hold_id)
               & (frames["phase"] == "hold")]
    dev_col = "head_deviation" if cue == "head" else "shoulder_deviation"
    devs = h[dev_col].to_numpy(dtype=float) if len(h) else np.array([])
    devs = devs[np.isfinite(devs)]
    tracked = float((h["tracking"] == 1).mean()) if len(h) else 0.0
    duration = (float(h["elapsed_s"].max()) - float(h["elapsed_s"].min())
                if len(h) > 1 else 0.0)
    return {
        "cue": cue,
        "hold_id": hold_id,
        "frames": int(len(h)),
        "duration_s": round(duration, 1),
        "tracked_fraction": round(tracked, 2),
        "max_abs_dev_deg": (round(float(np.max(np.abs(devs))), 1)
                            if len(devs) else None),
        "target_deg": target_deg(cue),
        "reached_target": bool(len(devs)
                               and float(np.max(np.abs(devs))) >= target_deg(cue)),
    }


def validate_rerun_csv(path: str) -> dict:
    """Validate a Run 2 capture CSV. Returns a report dict.

    Keys: ok (bool), errors (list of str), warnings (list of str),
    frames (int), tracked_fraction (float), holds (per-hold stats),
    gaps (tracking-loss gaps), work_s (float, calm sample duration).
    `ok` is False only on schema or structure errors; warnings never
    fail the file, they describe what the re-benchmark should know.
    """
    errors: list[str] = []
    warnings: list[str] = []
    try:
        session = load_session(path)
    except ValueError as e:
        return {"ok": False, "errors": [str(e)], "warnings": [],
                "frames": 0, "tracked_fraction": 0.0, "holds": [],
                "gaps": [], "work_s": 0.0}
    frames = session.frames
    meta = session.meta

    # --- Run 2 metadata -------------------------------------------------
    for key in sorted(RUN2_META_KEYS):
        if key not in meta:
            errors.append(f"missing Run 2 metadata key: {key}")
    if meta.get("Protocol", "") != "Run 2 re-run capture":
        errors.append(f"Protocol is not a Run 2 capture: "
                      f"{meta.get('Protocol', '')!r}")
    if meta.get("Hold duration (s)", "") not in ("10", "10.0"):
        errors.append(f"Hold duration is not 10 s: "
                      f"{meta.get('Hold duration (s)', '')!r}")

    # --- Run 2 columns ----------------------------------------------------
    missing = [c for c in frame_columns() if c not in frames.columns]
    if missing:
        errors.append(f"missing frame columns: {missing[:5]}"
                      + (f" (+{len(missing) - 5} more)"
                         if len(missing) > 5 else ""))
    # The base loader already guarantees REQUIRED_COLUMNS, so a missing
    # landmark or label column is the only way this fires.
    if errors:
        return {"ok": False, "errors": errors, "warnings": warnings,
                "frames": int(len(frames)), "tracked_fraction": 0.0,
                "holds": [], "gaps": [], "work_s": 0.0}

    # --- label values -------------------------------------------------------
    bad_phase = sorted(set(frames["phase"].dropna().unique()) - set(PHASES))
    if bad_phase:
        errors.append(f"unknown phase values: {bad_phase}")
    bad_cue = sorted(set(frames["cue"].dropna().unique()) - set(CUES) - {""})
    if bad_cue:
        errors.append(f"unknown cue values: {bad_cue}")
    hold_ids = frames["hold_id"].dropna().unique()
    bad_hold = sorted(int(h) for h in hold_ids
                      if int(h) < 1 or int(h) > TRIALS_PER_CUE)
    if bad_hold:
        errors.append(f"hold_id out of range 1-5: {bad_hold}")

    # --- holds ---------------------------------------------------------------
    holds: list[dict] = []
    for cue in CUES:
        for hold_id in range(1, TRIALS_PER_CUE + 1):
            stat = _hold_stats(frames, cue, hold_id)
            holds.append(stat)
            if stat["frames"] == 0:
                errors.append(f"missing hold: {cue} {hold_id}")
                continue
            if stat["duration_s"] < MIN_HOLD_DURATION_S:
                errors.append(
                    f"{cue} hold {hold_id} spans only "
                    f"{stat['duration_s']} s (need {MIN_HOLD_DURATION_S} s)")
            if stat["frames"] < MIN_HOLD_FRAMES:
                warnings.append(
                    f"{cue} hold {hold_id} has only {stat['frames']} frames")
            if stat["tracked_fraction"] < MIN_HOLD_TRACKED:
                warnings.append(
                    f"{cue} hold {hold_id} tracked only "
                    f"{stat['tracked_fraction']:.0%} of frames")
            if not stat["reached_target"]:
                warnings.append(
                    f"{cue} hold {hold_id} never reached its "
                    f"{stat['target_deg']:.0f} deg target "
                    f"(peak {stat['max_abs_dev_deg']} deg)")

    # --- work sample ----------------------------------------------------------
    calm = frames[frames["phase"] == "calm"]
    work_s = (float(calm["elapsed_s"].max()) - float(calm["elapsed_s"].min())
              if len(calm) > 1 else 0.0)
    if work_s < MIN_WORK_S:
        warnings.append(f"work sample is only {work_s:.0f} s "
                        f"(full sample is {WORK_S:.0f} s)")

    # --- tracking-loss gaps -----------------------------------------------------
    # A gap inside the loss test is the test working, not a gap: the
    # validator reports unexpected loss only.
    gaps = [g for g in tracking_gaps(frames) if g["phase"] != "loss"]
    for gap in gaps:
        warnings.append(
            f"tracking lost {gap['duration_s']} s at "
            f"{gap['start_s']} s ({gap['phase']}"
            + (f" {gap['cue']} hold {gap['hold_id']}" if gap["hold_id"] else "")
            + ")")

    tracked_fraction = float((frames["tracking"] == 1).mean()) \
        if len(frames) else 0.0
    return {"ok": not errors, "errors": errors, "warnings": warnings,
            "frames": int(len(frames)),
            "tracked_fraction": round(tracked_fraction, 3),
            "holds": holds, "gaps": gaps, "work_s": round(work_s, 1)}


def landmark_columns_present(header: list[str]) -> list[str]:
    """Landmark columns missing from a frame header (empty when complete)."""
    return [c for c in landmark_columns() if c not in header]
