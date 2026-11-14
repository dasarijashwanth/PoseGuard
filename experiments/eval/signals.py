"""Day 11: replacement shoulder signals.

Day 4 proved the shoulder angle level is the wrong signal: the rule
baseline detects 0/5 holds and fires 7 times during ordinary work,
and no threshold fixes it. The Day 5 classifiers worked around the
same bad metric and did no better (logistic regression 5/5 trials
but 5 false positives, and that 5/5 is a stillness artifact).

What the Run 1 traces actually show, and why the level fails:

- The deliberate holds moved the shoulder angle only about 2.5 deg
  from the calibration baseline (hold means 2.8, 2.3, -1.9, 2.7, 0.3).
- Ordinary desk work sits 4-7 deg from that same baseline for long
  stable stretches (120 of 318 one-second work windows are stable
  with |deviation| above 2 deg). The signal is smaller than the
  everyday drift, so no level threshold can separate them.
- The 7 false positives are not drift at all: they are tracker
  noise episodes where the raw shoulder angle jumps 20-100+ deg
  frame to frame while the head stays rock steady (498 work frames
  have a jump above 20 deg, 94 above 100 deg). The EMA smoothing
  follows those jumps and stays displaced for 5+ seconds.

Two of the planned candidates turned out to be impossible from the
Run 1 export, which records only head_tilt and shoulder_tilt angles,
no raw landmarks: shoulder-vs-hip needs hip landmarks 23/24, and
shoulder height asymmetry needs shoulder y-coordinates and shoulder
width. Neither exists in the data, so both were replaced by
candidates computable from the two angle series alone:

- roll: deviation from a trailing 30 s median baseline. A slow
  baseline absorbs everyday drift, so only fast moves away from
  recent posture register.
- rel: shoulder angle minus head angle (circular), deviation from
  its own baseline. Whole-body lean moves head and shoulders
  together, so the difference should cancel it.
- gated: the old level, but a firing only counts on frames the
  jump gate calls clean: no raw frame-to-frame jump above 20 deg
  in the last second. A real shoulder cannot move 20 deg in 0.1 s,
  so those frames are tracker noise by construction.

Each candidate is evaluated with the same Day 3-4 protocol:
threshold sweep, persistence curve, trials detected out of 5,
false-positive events during ordinary work. The gate is 4/5 trials
with 0 false positives.
"""

from __future__ import annotations

import bisect
from collections import deque

import numpy as np
import pandas as pd

from .analysis import (
    PERSIST_S,
    TRIALS_PER_CUE,
    align_trials,
    analyze_cue,
    false_positives,
)
from .circular import circ_dev, wrap_angle
from .loader import SessionData

ROLLING_WINDOW_S = 30.0     # trailing baseline window, seconds
ROLLING_MIN_SAMPLES = 50    # need this many frames before a baseline exists
JUMP_GATE_DEG = 20.0        # one-frame jump above this is tracker noise
JUMP_SETTLE_S = 1.0         # frames stay gated out this long after a jump
CANDIDATE_THRESHOLDS = (1, 2, 3, 4, 6, 8)
PERSIST_GRID_S = (0.5, 2.0, 5.0)

# name -> (deviation column, validity column or None)
CANDIDATES = {
    "roll": ("s_dev_roll", None),
    "rel": ("s_dev_rel", None),
    "gated": ("s_dev", "s_clean"),
}

# Winner of the Day 11 evaluation on Run 1 (see
# experiments/model/WALKTHROUGH_DAY11.md): the jump-gated level
# detects 4/5 trials at a 1 deg threshold with 8 false positives,
# and 0/5 with 1 false positive at the 6 deg operating threshold.
# No candidate passes the gate, but gated wins on trials detected
# first, then false positives. The model features use its series.
WINNER = "gated"
WINNER_COL = "s_dev_gated"


def rolling_baseline_dev(frames: pd.DataFrame, session: SessionData,
                         window_s: float = ROLLING_WINDOW_S,
                         min_samples: int = ROLLING_MIN_SAMPLES) -> pd.Series:
    """Deviation from a trailing median baseline, in degrees.

    For each frame, the baseline is the median of the circular
    deviations of the smoothed shoulder angle over the trailing
    `window_s` seconds (past frames only). The deviation is then
    measured from that moving baseline instead of the calibration
    baseline. Slow posture drift is absorbed; only moves away from
    recent posture register. Frames without enough history get NaN.
    """
    f = frames.sort_values("elapsed_s").reset_index(drop=True)
    t = f["elapsed_s"].to_numpy(dtype=float)
    smooth = f["s_smooth"].to_numpy(dtype=float)
    base = session.baseline_shoulder
    devs = np.array([circ_dev(a, base) if np.isfinite(a) else np.nan
                     for a in smooth])
    out = np.full(len(f), np.nan)
    window: deque = deque()       # (time, dev) pairs inside the window
    sorted_devs: list[float] = []  # same devs, kept sorted for the median
    for i in range(len(f)):
        while window and window[0][0] < t[i] - window_s:
            _, old = window.popleft()
            j = bisect.bisect_left(sorted_devs, old)
            if j < len(sorted_devs) and sorted_devs[j] == old:
                sorted_devs.pop(j)
        if np.isfinite(devs[i]):
            window.append((t[i], devs[i]))
            bisect.insort(sorted_devs, devs[i])
        if len(sorted_devs) >= min_samples and np.isfinite(smooth[i]):
            med = sorted_devs[len(sorted_devs) // 2] \
                if len(sorted_devs) % 2 == 1 else 0.5 * (
                    sorted_devs[len(sorted_devs) // 2 - 1]
                    + sorted_devs[len(sorted_devs) // 2])
            out[i] = circ_dev(smooth[i], base + med)
    return pd.Series(out, index=f.index)


def relative_dev(frames: pd.DataFrame, session: SessionData) -> pd.Series:
    """Shoulder-minus-head deviation, in degrees.

    The relative angle (shoulder line minus head line, wrapped) and
    its own baseline from the calibration baselines. When the whole
    body leans, head and shoulders rotate together and the difference
    stays put; a shoulder-only tilt moves the difference alone.
    """
    rel = np.array([wrap_angle(s - h) if np.isfinite(s) and np.isfinite(h)
                    else np.nan
                    for s, h in zip(frames["s_smooth"], frames["h_smooth"])])
    rel_base = wrap_angle(session.baseline_shoulder - session.baseline_head)
    out = np.array([circ_dev(r, rel_base) if np.isfinite(r) else np.nan
                    for r in rel])
    return pd.Series(out, index=frames.index)


def jump_valid_mask(frames: pd.DataFrame,
                    jump_deg: float = JUMP_GATE_DEG,
                    settle_s: float = JUMP_SETTLE_S) -> pd.Series:
    """Which frames are free of tracker jump noise.

    A frame is clean when no raw frame-to-frame shoulder jump above
    `jump_deg` happened in the last `settle_s` seconds. A real
    shoulder cannot rotate 20 deg in one 0.1 s frame, so a jump that
    big is tracker noise by construction, and the EMA needs about a
    second to settle back after one. Frames with no raw angle (NaN)
    are left clean here; tracking loss already breaks runs.
    """
    f = frames.sort_values("elapsed_s").reset_index(drop=True)
    raw = f["shoulder_tilt"].to_numpy(dtype=float)
    t = f["elapsed_s"].to_numpy(dtype=float)
    jumps = np.zeros(len(f), dtype=bool)
    for i in range(1, len(f)):
        if np.isfinite(raw[i]) and np.isfinite(raw[i - 1]):
            jumps[i] = abs(circ_dev(raw[i], raw[i - 1])) > jump_deg
    clean = np.ones(len(f), dtype=bool)
    last_jump_t: float | None = None
    for i in range(len(f)):
        if jumps[i]:
            last_jump_t = t[i]
        if last_jump_t is not None and t[i] - last_jump_t <= settle_s:
            clean[i] = False
    return pd.Series(clean, index=f.index)


def gated_dev(frames: pd.DataFrame) -> pd.Series:
    """Old shoulder deviation, NaN on jump-contaminated frames.

    The level itself, with tracker-noise frames removed instead of
    smoothed over. Lets the classifier see the level only where the
    tracker was sane; window features drop NaN frames before
    computing anything.
    """
    return frames["s_dev"].where(frames["s_clean"])


def add_signals(frames: pd.DataFrame, session: SessionData) -> pd.DataFrame:
    """Return frames with the Day 11 candidate columns added.

    Adds s_dev_roll, s_dev_rel, s_clean and s_dev_gated. The input
    must already carry s_smooth / h_smooth / s_dev (see
    experiments.eval.analysis.add_deviations).
    """
    out = frames.copy()
    out["s_dev_roll"] = rolling_baseline_dev(out, session)
    out["s_dev_rel"] = relative_dev(out, session)
    out["s_clean"] = jump_valid_mask(out)
    out["s_dev_gated"] = gated_dev(out)
    return out


def evaluate_signal(frames: pd.DataFrame, session: SessionData,
                    alignment: dict, name: str,
                    thresholds: tuple = CANDIDATE_THRESHOLDS) -> dict:
    """Day 3-4 protocol for one candidate signal.

    Threshold sweep at the 5 s persistence rule, then a persistence
    curve (0.5 / 2 / 5 s) at the best threshold: most trials
    detected, fewest false positives. `alignment` is the shoulder
    trial alignment (same protocol timing for every candidate).
    Returns the per-threshold rows, the persistence rows, and the
    best row by (detected, -false_positives).
    """
    if name not in CANDIDATES:
        raise ValueError(f"unknown candidate {name!r}, "
                         f"expected one of {sorted(CANDIDATES)}")
    dev_col, valid_col = CANDIDATES[name]
    rows: list[dict] = []
    for threshold in thresholds:
        a = analyze_cue(frames, session, "shoulder", threshold, alignment,
                        dev_col=dev_col, valid_col=valid_col)
        fp = false_positives(frames, "shoulder", threshold,
                             dev_col=dev_col, valid_col=valid_col)
        rows.append({"threshold": threshold,
                     "detected": a["detected"],
                     "trials_total": TRIALS_PER_CUE,
                     "false_positives": fp["count"],
                     "longest_run_s": a["longest_run_s"]})
    best = max(rows, key=lambda r: (r["detected"], -r["false_positives"]))
    persist: list[dict] = []
    for persist_s in PERSIST_GRID_S:
        a = analyze_cue(frames, session, "shoulder", best["threshold"],
                        alignment, persist_s=persist_s,
                        dev_col=dev_col, valid_col=valid_col)
        persist.append({"persist_s": persist_s,
                        "threshold": best["threshold"],
                        "detected": a["detected"],
                        "trials_total": TRIALS_PER_CUE})
    return {"candidate": name, "dev_col": dev_col,
            "valid_col": valid_col, "threshold_curve": rows,
            "persistence_curve": persist, "best": best}


def evaluate_all(frames: pd.DataFrame, session: SessionData) -> dict:
    """All three candidates plus the old level, same protocol.

    The old level is the Day 3-4 rule baseline for reference. The
    shoulder alignment is inferred from the head alignment (Run 1
    recorded no shoulder trial summary), exactly as Days 3-4 did.
    """
    head_align = align_trials(frames, session, "head")
    shoulder_align = align_trials(frames, session, "shoulder",
                                  fallback_offset=head_align["offset_s"])
    out = {"alignment_offset_s": shoulder_align["offset_s"],
           "alignment_verified": shoulder_align["verified"],
           "candidates": {}}
    old_rows: list[dict] = []
    for threshold in CANDIDATE_THRESHOLDS:
        a = analyze_cue(frames, session, "shoulder", threshold,
                        shoulder_align)
        fp = false_positives(frames, "shoulder", threshold)
        old_rows.append({"threshold": threshold,
                         "detected": a["detected"],
                         "trials_total": TRIALS_PER_CUE,
                         "false_positives": fp["count"],
                         "longest_run_s": a["longest_run_s"]})
    out["old_level"] = {"candidate": "old_level", "dev_col": "s_dev",
                        "valid_col": None, "threshold_curve": old_rows,
                        "best": max(old_rows,
                                    key=lambda r: (r["detected"],
                                                   -r["false_positives"]))}
    for name in CANDIDATES:
        out["candidates"][name] = evaluate_signal(
            frames, session, shoulder_align, name)
    return out


def pick_winner(evaluation: dict) -> str:
    """Best candidate by the gate's own terms: most trials, fewest FP.

    Ties break toward the candidate with the fewest false positives
    at its best threshold, then toward the simplest signal (gated,
    then roll, then rel). Returns the candidate name.
    """
    order = {"gated": 0, "roll": 1, "rel": 2}
    best_name, best_key = None, None
    for name, result in evaluation["candidates"].items():
        b = result["best"]
        key = (b["detected"], -b["false_positives"], -order[name])
        if best_key is None or key > best_key:
            best_name, best_key = name, key
    return best_name
