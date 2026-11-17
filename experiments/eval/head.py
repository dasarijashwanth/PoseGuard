"""Day 15: head cue rework.

Day 13 left the head cue as the worst half of the served model:
logistic regression 1/5 trials, 6 false positives, F1 0.078. Day 4
said the head tilts were real but too brief for the 5 s persistence
gate. Day 15 checks that claim properly and retunes the cue on the
three axes the earlier days never separated:

- angle deviation: the level itself, at thresholds 4-8 deg.
- velocity: how fast the smoothed head line moves, in deg/s. A
  deliberate tilt moves steadily; tracker noise jumps. Frames faster
  than a cap are treated as invalid, so a jump breaks a persistence
  run instead of starting one.
- dwell: how long the deviation stays above a threshold inside one
  hold, in seconds. The gate needs sustained presence, not a peak.

What Run 1 actually shows (full numbers in
experiments/model/WALKTHROUGH_DAY15.md): the holds are shallow as
well as brief. The deepest hold peaks at 7.1 deg and the longest run
above 4 deg inside any hold is 2.2 s, while ordinary work contains
head excursions of 10-18 deg sustained for 2-4 s. No threshold,
persistence, or velocity cap separates them, and the 5th hold is
barely recorded at all (6 frames), so at most 4/5 trials are
measurable in Run 1. The grid below reports that honestly: each
configuration gets trials detected, work false positives, and a
trial-level F1 (precision = detected / (detected + FP), recall =
detected / 5).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .analysis import (
    TRIALS_PER_CUE,
    _hold_windows,
    align_trials,
    analyze_cue,
    sustained_events,
)
from .loader import SessionData

HEAD_THRESHOLDS = (4, 5, 6, 7, 8)
PERSIST_GRID_S = (2.0, 3.0, 5.0)
VEL_CAPS_DEG_S = (None, 5.0, 10.0, 20.0)
DWELL_THRESHOLD = 5.0  # dwell is measured above this head deviation


def head_velocity(frames: pd.DataFrame) -> pd.Series:
    """Speed of the smoothed head line, in degrees per second.

    The gradient of h_dev against elapsed_s. Frames where the tracker
    was lost carry NaN deviations, and the gradient is NaN there too:
    a jump back from a gap is not a real head movement. NaN gaps are
    interpolated only so the gradient itself is defined across them;
    the NaN is restored afterwards.
    """
    f = frames.sort_values("elapsed_s").reset_index(drop=True)
    t = f["elapsed_s"].to_numpy(dtype=float)
    dev = f["h_dev"].to_numpy(dtype=float)
    filled = pd.Series(dev).interpolate(limit_direction="both").to_numpy()
    vel = np.gradient(filled, t)
    vel = np.where(np.isfinite(dev), vel, np.nan)
    return pd.Series(vel, index=f.index)


def add_head_features(frames: pd.DataFrame,
                      vel_caps: tuple = VEL_CAPS_DEG_S) -> pd.DataFrame:
    """Return frames with the Day 15 head columns added.

    Adds h_vel (deg/s) and one boolean h_slow_<cap> column per cap:
    True where |velocity| is at or under the cap. A None cap adds
    nothing; it means "no velocity gate" in the grid. The input must
    already carry h_dev (see experiments.eval.analysis.add_deviations).
    """
    out = frames.copy()
    out["h_vel"] = head_velocity(out)
    for cap in vel_caps:
        if cap is None:
            continue
        col = f"h_slow_{int(cap)}"
        out[col] = np.abs(out["h_vel"]) <= cap
    return out


def dwell_s(frames: pd.DataFrame, dev_col: str,
            threshold: float) -> float:
    """Longest continuous run above `threshold`, in seconds.

    Same run rules as the persistence gate (a gap over 0.35 s or a
    tracking loss breaks a run), with no minimum length: this is the
    "how long did it actually stay tilted" number, used per hold.
    """
    events = sustained_events(frames, dev_col, threshold, persist_s=0.0)
    return max((e["duration_s"] for e in events), default=0.0)


def trial_f1(detected: int, false_positives_count: int) -> dict:
    """Trial-level precision, recall, and F1 for one configuration.

    The 5 trials are the positives; every work false-positive event
    is a false positive. A configuration that detects nothing scores
    F1 0, however clean its work sample is.
    """
    precision = (detected / (detected + false_positives_count)
                 if detected + false_positives_count > 0 else 0.0)
    recall = detected / TRIALS_PER_CUE
    f1 = (2 * precision * recall / (precision + recall)
          if precision + recall > 0 else 0.0)
    return {"precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3)}


def hold_dwells(frames: pd.DataFrame, session: SessionData,
                alignment: dict,
                thresholds: tuple = HEAD_THRESHOLDS) -> list[dict]:
    """Per-hold dwell above each threshold, in seconds.

    One row per trial, with the sample count in the hold window so a
    barely recorded hold (Run 1 trial 5: 6 samples) is visible as
    such instead of looking like a missed tilt.
    """
    step_frames = frames[frames["step"] == "head"].sort_values("elapsed_s")
    t0 = float(step_frames["elapsed_s"].iloc[0])
    rows: list[dict] = []
    for k, (lo, hi) in enumerate(_hold_windows(t0, alignment["offset_s"]), 1):
        w = step_frames[(step_frames["elapsed_s"] >= lo)
                        & (step_frames["elapsed_s"] < hi)]
        row = {"trial": k, "samples": int(len(w)),
               "mean_abs_dev": (round(float(w["h_dev"].abs().mean()), 2)
                                if len(w) else float("nan")),
               "max_abs_dev": (round(float(w["h_dev"].abs().max()), 2)
                               if len(w) else float("nan"))}
        for thr in thresholds:
            row[f"dwell_gt{thr}_s"] = round(dwell_s(w, "h_dev", thr), 1)
        rows.append(row)
    return rows


def evaluate_head_gate(frames: pd.DataFrame, session: SessionData,
                       thresholds: tuple = HEAD_THRESHOLDS,
                       persists: tuple = PERSIST_GRID_S,
                       vel_caps: tuple = VEL_CAPS_DEG_S) -> dict:
    """Threshold x persistence x velocity-cap grid for the head cue.

    Every configuration is scored the same way: trials detected out
    of 5 (a firing must overlap the hold window), false-positive
    events during the ordinary-work sample, and trial-level F1. The
    velocity cap is applied as a validity column, so a fast frame
    breaks a persistence run. `best` is the highest F1, breaking
    ties toward more trials detected, then fewer false positives,
    then the longest persistence (the gate the API actually runs).
    The shoulder cue is untouched: this grid is head only.
    """
    frames = add_head_features(frames, vel_caps)
    alignment = align_trials(frames, session, "head")
    rows: list[dict] = []
    for cap in vel_caps:
        valid_col = None if cap is None else f"h_slow_{int(cap)}"
        for threshold in thresholds:
            for persist in persists:
                a = analyze_cue(frames, session, "head", threshold,
                                alignment, persist_s=persist,
                                dev_col="h_dev", valid_col=valid_col)
                work = frames[frames["step"] == "work"].sort_values(
                    "elapsed_s")
                fp_events = sustained_events(work, "h_dev", threshold,
                                             persist_s=persist,
                                             valid_col=valid_col)
                f1 = trial_f1(a["detected"], len(fp_events))
                rows.append({"threshold": threshold,
                             "persist_s": persist,
                             "vel_cap_deg_s": cap,
                             "detected": a["detected"],
                             "trials_total": TRIALS_PER_CUE,
                             "false_positives": len(fp_events),
                             **f1})
    best = max(rows, key=lambda r: (r["f1"], r["detected"],
                                    -r["false_positives"],
                                    r["persist_s"]))
    return {"alignment_offset_s": alignment["offset_s"],
            "alignment_verified": alignment["verified"],
            "hold_dwells": hold_dwells(frames, session, alignment),
            "rows": rows, "best": best}
