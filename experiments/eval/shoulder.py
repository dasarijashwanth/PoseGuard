"""Day 17: shoulder cue rework.

Day 4 proved the shoulder angle level is the wrong signal for desk
work: the rule baseline detects 0/5 holds and fires 7 times during
ordinary work, and no threshold fixes it, because a deliberate hold
moves the shoulder line only about 2.5 deg while everyday leaning
drifts 4-7 deg. Day 11 fixed the tracker-noise half with the
jump-gated signal (false positives 7 to 1 at the 6 deg operating
threshold) but still detected 0/5 trials there, so the level half
remained. Day 17 gives the shoulder cue the Day 15 treatment: it
separates the three things the level conflates, the level itself,
how long it stays there (persistence), and how fast it moves
(velocity), and scores a threshold x persistence x velocity-cap
grid the same way Day 15 scored the head cue.

Two shoulder specifics, and why they differ from the head grid:

- Thresholds are 1-8 deg, not 4-8: the holds live around 2-3 deg,
  so a 4 deg floor would test nothing the holds ever reach. The
  grid is 6 thresholds x 3 persistences x 4 velocity caps, 72
  configurations.
- Dwell is measured above 2 deg, not 5: a 5 deg dwell threshold
  would be zero on every shoulder window in Run 1, hold and work
  alike, and carry no information. 2 deg sits just below the
  typical hold depth, the same place 5 deg sat for the head cue.

The shoulder trial alignment is inferred, not verified: Run 1
recorded no per-trial summary for the shoulder cue, so the offset
from the head alignment is reused, exactly as Days 3-4 and 11 did.
The grid reports that honestly (alignment_verified False).

What Run 1 actually shows (full numbers in
experiments/model/WALKTHROUGH_DAY17.md): the holds are shallow and
brief, the longest run above 2 deg inside any hold is under 2 s,
while ordinary work contains shoulder excursions of 4-7 deg
sustained for seconds at a time. No threshold, persistence, or
velocity cap separates them. The grid below reports that honestly:
each configuration gets trials detected, work false positives, and
a trial-level F1 (precision = detected / (detected + FP), recall =
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
from .head import dwell_s, trial_f1
from .loader import SessionData

SHOULDER_THRESHOLDS_GRID = (1, 2, 3, 4, 6, 8)
PERSIST_GRID_S = (2.0, 3.0, 5.0)
VEL_CAPS_DEG_S = (None, 5.0, 10.0, 20.0)
DWELL_THRESHOLD = 2.0  # dwell is measured above this shoulder deviation


def shoulder_velocity(frames: pd.DataFrame) -> pd.Series:
    """Speed of the smoothed shoulder line, in degrees per second.

    Same construction as head_velocity in experiments.eval.head: the
    gradient of s_dev against elapsed_s. Frames where the tracker
    was lost carry NaN deviations, and the gradient is NaN there
    too: a jump back from a gap is not a real shoulder movement.
    NaN gaps are interpolated only so the gradient itself is defined
    across them; the NaN is restored afterwards.
    """
    f = frames.sort_values("elapsed_s").reset_index(drop=True)
    t = f["elapsed_s"].to_numpy(dtype=float)
    dev = f["s_dev"].to_numpy(dtype=float)
    filled = pd.Series(dev).interpolate(limit_direction="both").to_numpy()
    vel = np.gradient(filled, t)
    vel = np.where(np.isfinite(dev), vel, np.nan)
    return pd.Series(vel, index=f.index)


def add_shoulder_features(frames: pd.DataFrame,
                          vel_caps: tuple = VEL_CAPS_DEG_S) -> pd.DataFrame:
    """Return frames with the Day 17 shoulder columns added.

    Adds s_vel (deg/s) and one boolean s_slow_<cap> column per cap:
    True where |velocity| is at or under the cap. A None cap adds
    nothing; it means "no velocity gate" in the grid. The input must
    already carry s_dev (see experiments.eval.analysis.add_deviations).
    """
    out = frames.copy()
    out["s_vel"] = shoulder_velocity(out)
    for cap in vel_caps:
        if cap is None:
            continue
        col = f"s_slow_{int(cap)}"
        out[col] = np.abs(out["s_vel"]) <= cap
    return out


def shoulder_hold_dwells(frames: pd.DataFrame, session: SessionData,
                         alignment: dict,
                         thresholds: tuple = SHOULDER_THRESHOLDS_GRID,
                         ready_s: float = 3.0, hold_s: float = 6.0,
                         rest_s: float = 3.0) -> list[dict]:
    """Per-hold dwell above each threshold, in seconds.

    One row per trial, with the sample count in the hold window so
    a barely recorded hold is visible as such instead of looking
    like a missed tilt. The timing args are the protocol timings,
    Run 1 by default, Run 2 (5 / 10 / 5) for a re-run capture.
    The shoulder alignment is inferred from the head alignment on
    Run 1 (no shoulder trial summary was recorded); the caller
    passes whatever alignment evaluate_shoulder_gate computed.
    """
    step_frames = frames[frames["step"] == "shoulder"].sort_values(
        "elapsed_s")
    t0 = float(step_frames["elapsed_s"].iloc[0])
    rows: list[dict] = []
    for k, (lo, hi) in enumerate(_hold_windows(
            t0, alignment["offset_s"], ready_s, hold_s, rest_s), 1):
        w = step_frames[(step_frames["elapsed_s"] >= lo)
                        & (step_frames["elapsed_s"] < hi)]
        mean_abs = w["s_dev"].abs().mean() if len(w) else float("nan")
        max_abs = w["s_dev"].abs().max() if len(w) else float("nan")
        row = {"trial": k, "samples": int(len(w)),
               # No data in the window means no mean, not a NaN mean:
               # NaN would leak into results JSON (invalid strict JSON).
               "mean_abs_dev": (round(float(mean_abs), 2)
                                if pd.notna(mean_abs) else None),
               "max_abs_dev": (round(float(max_abs), 2)
                               if pd.notna(max_abs) else None)}
        for thr in thresholds:
            row[f"dwell_gt{thr}_s"] = round(dwell_s(w, "s_dev", thr), 1)
        rows.append(row)
    return rows


def evaluate_shoulder_gate(frames: pd.DataFrame, session: SessionData,
                           thresholds: tuple = SHOULDER_THRESHOLDS_GRID,
                           persists: tuple = PERSIST_GRID_S,
                           vel_caps: tuple = VEL_CAPS_DEG_S,
                           ready_s: float = 3.0, hold_s: float = 6.0,
                           rest_s: float = 3.0) -> dict:
    """Threshold x persistence x velocity-cap grid for the shoulder cue.

    Every configuration is scored the same way as the Day 15 head
    grid: trials detected out of 5 (a firing must overlap the hold
    window), false-positive events during the ordinary-work sample,
    and trial-level F1. The velocity cap is applied as a validity
    column, so a fast frame breaks a persistence run. `best` is the
    highest F1, breaking ties toward more trials detected, then
    fewer false positives, then the longest persistence (the gate
    the API actually runs). The head cue is untouched: this grid is
    shoulder only. The timing args are the protocol timings, Run 1
    by default, Run 2 (5 / 10 / 5) for a re-run capture.
    """
    frames = add_shoulder_features(frames, vel_caps)
    head_align = align_trials(frames, session, "head",
                              ready_s=ready_s, hold_s=hold_s, rest_s=rest_s)
    alignment = align_trials(frames, session, "shoulder",
                             fallback_offset=head_align["offset_s"],
                             ready_s=ready_s, hold_s=hold_s, rest_s=rest_s)
    rows: list[dict] = []
    for cap in vel_caps:
        valid_col = None if cap is None else f"s_slow_{int(cap)}"
        for threshold in thresholds:
            for persist in persists:
                a = analyze_cue(frames, session, "shoulder", threshold,
                                alignment, persist_s=persist,
                                dev_col="s_dev", valid_col=valid_col,
                                ready_s=ready_s, hold_s=hold_s,
                                rest_s=rest_s)
                work = frames[frames["step"] == "work"].sort_values(
                    "elapsed_s")
                fp_events = sustained_events(work, "s_dev", threshold,
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
            "hold_dwells": shoulder_hold_dwells(frames, session, alignment,
                                                ready_s=ready_s,
                                                hold_s=hold_s,
                                                rest_s=rest_s),
            "rows": rows, "best": best}
