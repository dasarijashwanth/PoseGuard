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

Day 11 adds three gated-signal features (mean_abs_sn, max_abs_sn,
std_sn) from the winning replacement signal, the jump-gated
deviation (see experiments.eval.signals). Those are stats of the
gated signal, not of the raw level: the gated series is NaN wherever
the tracker was jumping, so the classifier only ever sees the level
on frames the tracker was sane. The old level names stay banned
(see test_shoulder_features_never_include_shoulder_level).

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

# Shoulder cue: NO raw shoulder level. Only dynamics, drift, and
# coupling, plus Day 11 gated-signal stats (see the module docstring).
SHOULDER_FEATURES = [
    "std_s",         # how jumpy the shoulder line is
    "range_s",       # max - min shoulder deviation in the window
    "jerk_s",        # mean |frame-to-frame change| of shoulder deviation
    "slope_s",       # linear drift of the shoulder line (deg per frame)
    "corr_h_s",      # head/shoulder coupling inside the window
    "std_h",         # head steadiness (context: still vs moving)
    "mean_abs_h",    # head level (context: head tilting along?)
    "jerk_h",        # head jumpiness (context)
    "mean_abs_sn",   # mean |gated shoulder deviation| (Day 11 signal)
    "max_abs_sn",    # largest |gated shoulder deviation|
    "std_sn",        # steadiness of the gated shoulder deviation
]

FEATURES = {"head": HEAD_FEATURES, "shoulder": SHOULDER_FEATURES}

# Shoulder cue before Day 11: the same 8 dynamics and coupling
# features, without the three gated-signal stats. Day 12 keeps this
# list so the old-feature benchmark reproduces Day 8 exactly.
OLD_SHOULDER_FEATURES = [
    "std_s",         # how jumpy the shoulder line is
    "range_s",       # max - min shoulder deviation in the window
    "jerk_s",        # mean |frame-to-frame change| of shoulder deviation
    "slope_s",       # linear drift of the shoulder line (deg per frame)
    "corr_h_s",      # head/shoulder coupling inside the window
    "std_h",         # head steadiness (context: still vs moving)
    "mean_abs_h",    # head level (context: head tilting along?)
    "jerk_h",        # head jumpiness (context)
]

# Day 15 adds two head features (velocity and dwell) to the head list.
# HEAD_FEATURES stays the Day 5/13 list so the served v2 artifacts keep
# matching; Day 15 evaluates the 8-feature list without retraining
# the served models.
DAY15_HEAD_FEATURES = HEAD_FEATURES + ["vel_h", "dwell_h"]
DWELL_THRESHOLD_DEG = 5.0  # dwell is measured above this head deviation

# Day 17 adds the same two features for the shoulder cue, mirroring
# Day 15. vel_s is the mean |speed| of the shoulder line in deg/s
# and dwell_s is the longest continuous run above 2 deg in seconds.
# The dwell threshold is 2 deg, not 5: the shoulder holds live
# around 2-3 deg, so a 5 deg threshold would be zero on every
# window, hold and work alike. Both are dynamics, not level stats:
# neither is a mean or max of the shoulder deviation, so the Day 5
# ban on shoulder level features still holds (see
# test_shoulder_features_never_include_shoulder_level).
DAY17_SHOULDER_FEATURES = SHOULDER_FEATURES + ["vel_s", "dwell_s"]
DWELL_THRESHOLD_SHOULDER_DEG = 2.0

# Day 11 changed nothing for the head cue, so its old and new lists
# are the same by construction.
OLD_FEATURES = {"head": HEAD_FEATURES, "shoulder": OLD_SHOULDER_FEATURES}


def window_features(h_dev: np.ndarray, s_dev: np.ndarray,
                    s_new: np.ndarray | None = None,
                    times: np.ndarray | None = None) -> dict:
    """Features for one window of deviations.

    Frames where the tracker was lost carry NaN deviations; they are
    dropped before any feature is computed (at least 3 finite frames
    are needed, otherwise the window says nothing). All eighteen
    features are always returned, even though each cue's model only
    trains on its own list: the calm-only check in train.py needs
    std_s for the head cue too. `s_new` is the Day 11 gated
    deviation series for the same window (NaN where the jump gate
    rejected a frame); without it the three gated features are 0.0,
    which is what the live API sees, since it has no trailing
    history to run the gate on. `times` is the elapsed seconds per
    frame (Day 15: velocity and dwell need real time, not a frame
    count); without it frames are assumed 0.1 s apart.

    Day 15 adds two head features. vel_h is the mean |speed| of the
    head line in deg/s: a deliberate tilt moves steadily, tracker
    noise jumps. dwell_h is the longest continuous run above 5 deg
    inside the window, in seconds: the gate needs sustained
    presence, not a peak. Both are computed on the valid frames
    only, so a tracking gap breaks a dwell run instead of bridging
    it.

    Day 17 adds the same two features for the shoulder cue. vel_s
    is the mean |speed| of the shoulder line in deg/s and dwell_s
    is the longest continuous run above 2 deg in seconds. The
    shoulder dwell threshold is 2 deg, not 5: the holds live around
    2-3 deg, so a 5 deg threshold would be zero on every window.
    Both are dynamics, not level stats, so the shoulder classifier
    still never sees the shoulder level itself.
    """
    h = np.asarray(h_dev, dtype=float)
    s = np.asarray(s_dev, dtype=float)
    if len(h) == 0:
        raise ValueError("empty window")
    valid = np.isfinite(h) & np.isfinite(s)
    if int(valid.sum()) < 3:
        raise ValueError("too few valid samples in window")
    t = (np.asarray(times, dtype=float) if times is not None
         else np.arange(len(h)) * 0.1)
    if len(t) != len(h) or (len(t) > 1 and np.any(np.diff(t) <= 0)):
        t = np.arange(len(h)) * 0.1
    h_filled = pd.Series(h).interpolate(limit_direction="both").to_numpy()
    vel = np.gradient(h_filled, t)
    vel_h = float(np.mean(np.abs(vel[valid])))
    s_filled = pd.Series(s).interpolate(limit_direction="both").to_numpy()
    s_vel = np.gradient(s_filled, t)
    vel_s = float(np.mean(np.abs(s_vel[valid])))

    def _dwell(series: np.ndarray, threshold: float) -> float:
        """Longest continuous run above `threshold`, in seconds.

        Computed on the valid frames only, so a tracking gap breaks
        a run instead of bridging it. Duration is the span from the
        first to the last frame of the run.
        """
        above = valid & (np.abs(series) > threshold)
        longest = 0.0
        run_start = None
        for i, flag in enumerate(above):
            if flag:
                if run_start is None:
                    run_start = i
            elif run_start is not None:
                longest = max(longest, float(t[i - 1] - t[run_start]))
                run_start = None
        if run_start is not None:
            longest = max(longest, float(t[-1] - t[run_start]))
        return longest

    dwell_h = _dwell(h, DWELL_THRESHOLD_DEG)
    dwell_s = _dwell(s, DWELL_THRESHOLD_SHOULDER_DEG)
    h, s = h[valid], s[valid]
    corr = 0.0
    if np.std(h) > 1e-9 and np.std(s) > 1e-9:
        with np.errstate(invalid="ignore"):
            corr = float(np.corrcoef(h, s)[0, 1])
        if np.isnan(corr):
            corr = 0.0
    slope_s = float(np.polyfit(np.arange(len(s)), s, 1)[0]) \
        if len(s) > 1 else 0.0
    if s_new is None:
        gated = {"mean_abs_sn": 0.0, "max_abs_sn": 0.0, "std_sn": 0.0}
    else:
        sn = np.asarray(s_new, dtype=float)[valid]
        sn = sn[np.isfinite(sn)]
        if len(sn) < 3:
            gated = {"mean_abs_sn": 0.0, "max_abs_sn": 0.0, "std_sn": 0.0}
        else:
            gated = {"mean_abs_sn": float(np.mean(np.abs(sn))),
                     "max_abs_sn": float(np.max(np.abs(sn))),
                     "std_sn": float(np.std(sn))}
    return {
        "mean_abs_h": float(np.mean(np.abs(h))),
        "max_abs_h": float(np.max(np.abs(h))),
        "mean_h": float(np.mean(h)),
        "std_h": float(np.std(h)),
        "frac_h_gt5": float(np.mean(np.abs(h) > 5)),
        "jerk_h": float(np.mean(np.abs(np.diff(h)))) if len(h) > 1 else 0.0,
        "vel_h": vel_h,
        "dwell_h": dwell_h,
        "vel_s": vel_s,
        "dwell_s": dwell_s,
        "std_s": float(np.std(s)),
        "range_s": float(np.max(s) - np.min(s)),
        "jerk_s": float(np.mean(np.abs(np.diff(s)))) if len(s) > 1 else 0.0,
        "slope_s": slope_s,
        "corr_h_s": corr,
        **gated,
    }


def build_windows(frames: pd.DataFrame, session, cue: str,
                  ready_s: float = 3.0, hold_s: float = 6.0,
                  rest_s: float = 3.0) -> pd.DataFrame:
    """1-second labelled windows for one cue from enriched frames.

    `frames` must already carry h_dev / s_dev (see
    experiments.eval.analysis.add_deviations). Returns one row per
    window: the cue's features, label (1 tilted, 0 upright), the step
    it came from, its start time, and a `group` (trial 1-5, or work
    time-block 1-5) used for grouped cross-validation so no trial's
    windows ever straddle train and test. Windows with fewer than
    MIN_TRACKED tracked frames are skipped (the tracker was mostly
    lost there, so the window says nothing about posture). The
    timing args are the protocol timings, Run 1 by default, Run 2
    (5 / 10 / 5) for a re-run capture: the hold windows and the
    trial cycle (ready + hold + rest) both change, so a Run 1
    cycle would label the wrong frames tilted.
    """
    if cue not in FEATURES:
        raise ValueError(f"unknown cue {cue!r}, expected 'head' or 'shoulder'")
    step = cue  # trial step name matches the cue name
    alignment = align_trials(frames, session, cue,
                             ready_s=ready_s, hold_s=hold_s, rest_s=rest_s)
    step_frames = frames[frames["step"] == step].sort_values("elapsed_s")
    t0 = float(step_frames["elapsed_s"].iloc[0])
    holds = _hold_windows(t0, alignment["offset_s"],
                          ready_s, hold_s, rest_s)

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
                s_new = (w["s_dev_new"].to_numpy()
                         if "s_dev_new" in w.columns else None)
                feats = window_features(w["h_dev"].to_numpy(),
                                        w["s_dev"].to_numpy(),
                                        s_new=s_new,
                                        times=w["elapsed_s"].to_numpy())
            except ValueError:
                continue  # too few valid samples: window says nothing
            t_start = float(w["elapsed_s"].iloc[0])
            if source == "trial":
                t_end = float(w["elapsed_s"].iloc[-1])
                label = int(any(t_start >= lo and t_end <= hi
                                for lo, hi in holds))
                # which trial cycle the window belongs to (1-5);
                # used as the CV group so a trial's windows never
                # straddle train and test. The cycle is the full
                # ready + hold + rest length of the protocol.
                cycle = ready_s + hold_s + rest_s
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
