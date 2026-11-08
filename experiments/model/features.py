"""Window features for the Day 5 posture classifier.

The classifier does not look at single frames. It looks at a 1-second
window (10 frames at ~10 fps) of the smoothed circular deviations the
Day 3 harness computes, and predicts one of two states:

    tilted  - the user is deliberately holding a tilt
    upright - ordinary posture (sitting still, or typing at a desk)

Two different feature sets, on purpose. Day 4 proved the raw shoulder
angle cannot tell a dropped shoulder from leaning at a desk, so the
shoulder classifier is FORBIDDEN from seeing the shoulder level
(no mean, no max of the shoulder deviation). It gets only dynamics
and coupling: how jumpy the shoulder line is, how fast it drifts,
and how it moves together with the head. If those features cannot
separate the holds from desk work, the honest answer is "no signal",
not a better threshold.

Labels come from the protocol, not from a threshold: windows fully
inside a trial hold window are labelled tilted (the user was told to
hold a tilt); windows from the ordinary-work step, and the ready/rest
windows inside a trial step, are labelled upright.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.eval.analysis import _hold_windows, align_trials

WINDOW_S = 1.0        # window length in seconds
WINDOW_N = 10         # frames per window at ~10 fps
MIN_TRACKED = 8       # need at least this many tracked frames per window

# Head cue: the angle itself is salvageable (Day 4), so level features
# are fair game, plus steadiness features.
HEAD_FEATURES = [
    "mean_abs_h",    # mean |head deviation| in the window
    "max_abs_h",     # largest |head deviation|
    "mean_h",        # signed mean head deviation (tilts go one way)
    "std_h",         # how steady the head line is
    "frac_h_gt5",    # fraction of frames |head dev| > 5 deg
    "jerk_h",        # mean |frame-to-frame change| of head deviation
]

# Shoulder cue: NO shoulder level. Only dynamics, drift, and coupling.
SHOULDER_FEATURES = [
    "std_s",         # how jumpy the shoulder line is
    "range_s",       # max - min shoulder deviation in the window
    "jerk_s",        # mean |frame-to-frame change| of shoulder deviation
    "slope_s",       # linear drift of the shoulder line (deg per frame)
    "corr_h_s",      # head/shoulder coupling inside the window
    "std_h",         # head steadiness (context: still vs moving)
    "mean_abs_h",    # head level (context: head tilting along?)
    "jerk_h",        # head jumpiness (context)
]

FEATURES = {"head": HEAD_FEATURES, "shoulder": SHOULDER_FEATURES}


def window_features(h_dev: np.ndarray, s_dev: np.ndarray) -> dict:
    """Features for one window of deviations.

    Frames where the tracker was lost carry NaN deviations; they are
    dropped before any feature is computed (at least 3 finite frames
    are needed, otherwise the window says nothing). All eleven
    features are always returned, even though each cue's model only
    trains on its own list: the calm-only check in train.py needs
    std_s for the head cue too."""
    h = np.asarray(h_dev, dtype=float)
    s = np.asarray(s_dev, dtype=float)
    if len(h) == 0:
        raise ValueError("empty window")
    valid = np.isfinite(h) & np.isfinite(s)
    if int(valid.sum()) < 3:
        raise ValueError("too few valid samples in window")
    h, s = h[valid], s[valid]
    corr = 0.0
    if np.std(h) > 1e-9 and np.std(s) > 1e-9:
        with np.errstate(invalid="ignore"):
            corr = float(np.corrcoef(h, s)[0, 1])
        if np.isnan(corr):
            corr = 0.0
    slope_s = float(np.polyfit(np.arange(len(s)), s, 1)[0]) \
        if len(s) > 1 else 0.0
    return {
        "mean_abs_h": float(np.mean(np.abs(h))),
        "max_abs_h": float(np.max(np.abs(h))),
        "mean_h": float(np.mean(h)),
        "std_h": float(np.std(h)),
        "frac_h_gt5": float(np.mean(np.abs(h) > 5)),
        "jerk_h": float(np.mean(np.abs(np.diff(h)))) if len(h) > 1 else 0.0,
        "std_s": float(np.std(s)),
        "range_s": float(np.max(s) - np.min(s)),
        "jerk_s": float(np.mean(np.abs(np.diff(s)))) if len(s) > 1 else 0.0,
        "slope_s": slope_s,
        "corr_h_s": corr,
    }


def build_windows(frames: pd.DataFrame, session, cue: str) -> pd.DataFrame:
    """1-second labelled windows for one cue from enriched frames.

    `frames` must already carry h_dev / s_dev (see
    experiments.eval.analysis.add_deviations). Returns one row per
    window: the cue's features, label (1 tilted, 0 upright), the step
    it came from, its start time, and a `group` (trial 1-5, or work
    time-block 1-5) used for grouped cross-validation so no trial's
    windows ever straddle train and test. Windows with fewer than
    MIN_TRACKED tracked frames are skipped (the tracker was mostly
    lost there, so the window says nothing about posture).
    """
    if cue not in FEATURES:
        raise ValueError(f"unknown cue {cue!r}, expected 'head' or 'shoulder'")
    step = cue  # trial step name matches the cue name
    alignment = align_trials(frames, session, cue)
    step_frames = frames[frames["step"] == step].sort_values("elapsed_s")
    t0 = float(step_frames["elapsed_s"].iloc[0])
    holds = _hold_windows(t0, alignment["offset_s"])

    rows: list[dict] = []
    work_frames = frames[frames["step"] == "work"].sort_values("elapsed_s")
    work_t0 = float(work_frames["elapsed_s"].iloc[0]) if len(work_frames) else 0.0
    work_span = (float(work_frames["elapsed_s"].iloc[-1]) - work_t0
                 if len(work_frames) > 1 else 1.0)
    for source, label_of in (("trial", None), ("work", 0)):
        d = frames[frames["step"] == (step if source == "trial" else "work")]
        d = d.sort_values("elapsed_s").reset_index(drop=True)
        for i in range(0, len(d) - WINDOW_N + 1, WINDOW_N):
            w = d.iloc[i:i + WINDOW_N]
            if int((w["tracking"] == 1).sum()) < MIN_TRACKED:
                continue
            try:
                feats = window_features(w["h_dev"].to_numpy(),
                                        w["s_dev"].to_numpy())
            except ValueError:
                continue  # too few valid samples: window says nothing
            t_start = float(w["elapsed_s"].iloc[0])
            if source == "trial":
                t_end = float(w["elapsed_s"].iloc[-1])
                label = int(any(t_start >= lo and t_end <= hi
                                for lo, hi in holds))
                # which 12 s trial cycle the window belongs to (1-5);
                # used as the CV group so a trial's windows never
                # straddle train and test.
                cycle = 3.0 + 6.0 + 3.0
                group = int((t_start - t0 - alignment["offset_s"]) // cycle) + 1
                group = max(1, min(5, group))
            else:
                label = 0
                # work windows split into 5 time blocks, same idea.
                group = int(5 * (t_start - work_t0) / max(work_span, 1e-9)) + 1
                group = max(1, min(5, group))
            rows.append({"label": label, "step": w["step"].iloc[0],
                         "start_s": t_start, "group": group,
                         **feats})
    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError(f"no usable windows for cue {cue!r}")
    return out.reset_index(drop=True)
