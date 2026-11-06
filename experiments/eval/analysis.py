"""Per-cue analysis: detection rates, false positives, sensitivity.

This mirrors the web experiment's rules exactly, so the numbers here are
directly comparable to what the page showed live:

- Angles are EMA-smoothed first (alpha 0.4, circular EMA, same as the
  page), then deviation from baseline is measured with circular math.
- A cue "fires" only after its deviation stays beyond the threshold
  continuously for 5 seconds (independent timer per cue, reset on
  tracking loss or a frame gap over 0.35 s).
- A trial counts as detected if a firing overlaps its 6-second hold.
- Pass criteria per cue: at least 4 of 5 holds detected AND zero false
  detections during ordinary work AND the landmark-loss test ran clean
  (landmarks actually lost, no firings while they were gone).

One honest complication: the export does not record where each hold
starts. The trial clock (3 s ready, 6 s hold, 3 s rest, x5) starts when
the owner clicks "Begin trials", a few seconds after the step's first
frame. `align_trials` recovers that offset by fitting it against the
per-trial summary the experiment recorded. When no summary exists for a
cue (Run 1 recorded none for shoulder), the offset from the other cue
is reused and the trial stats are flagged as inferred, not verified.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .circular import circ_dev, wrap_angle
from .loader import SessionData

READY_S = 3.0    # "get ready" before each hold
HOLD_S = 6.0     # hold the tilt this long
REST_S = 3.0     # rest upright between holds
TRIALS_PER_CUE = 5
PERSIST_S = 5.0  # deviation must hold this long to fire
EMA_ALPHA = 0.4  # smoothing, same as the web experiment
MAX_GAP_S = 0.35  # a bigger frame gap breaks a persistence run

HEAD_THRESHOLDS = [5, 6, 7, 8]
SHOULDER_THRESHOLDS = [4, 5, 6, 8]

CUE_STEP = {"head": "head", "shoulder": "shoulder"}
CUE_BASELINE = {"head": "baseline_head", "baseline_shoulder": "baseline_shoulder"}
CUE_ANGLE = {"head": "head_tilt", "shoulder": "shoulder_tilt"}


def ema_smooth(angles: pd.Series, alpha: float = EMA_ALPHA) -> pd.Series:
    """Circular exponential moving average, seeded at the first valid angle.

    Plain EMA breaks at the wrap boundary (averaging +179 and -179 gives
    0). This one steps along the circle: each update moves a fraction
    `alpha` of the short-way distance from the current smooth value to
    the new reading. Identical to the smoothing in the web experiment.
    """
    out: list[float] = []
    smooth: float | None = None
    for a in angles:
        if a is None or (isinstance(a, float) and math.isnan(a)):
            out.append(float("nan"))
            continue
        if smooth is None:
            smooth = float(a)
        else:
            smooth = wrap_angle(smooth + alpha * circ_dev(float(a), smooth))
        out.append(smooth)
    return pd.Series(out, index=angles.index)


def add_deviations(session: SessionData) -> pd.DataFrame:
    """Return frames with smoothed angles and circular deviations added.

    Columns added: h_smooth, s_smooth (EMA-smoothed angles),
    h_dev, s_dev (circular deviation from baseline, the honest numbers),
    h_dev_linear, s_dev_linear (plain subtraction, kept only to show
    the bug: near +/-180 these jump to ~+/-350 degrees).
    """
    frames = session.frames.copy()
    frames["h_smooth"] = ema_smooth(frames["head_tilt"])
    frames["s_smooth"] = ema_smooth(frames["shoulder_tilt"])
    frames["h_dev"] = [circ_dev(a, session.baseline_head) if pd.notna(a) else np.nan
                       for a in frames["h_smooth"]]
    frames["s_dev"] = [circ_dev(a, session.baseline_shoulder) if pd.notna(a) else np.nan
                       for a in frames["s_smooth"]]
    frames["h_dev_linear"] = frames["head_tilt"] - session.baseline_head
    frames["s_dev_linear"] = frames["shoulder_tilt"] - session.baseline_shoulder
    return frames


def sustained_events(frames: pd.DataFrame, dev_col: str, threshold: float,
                     persist_s: float = PERSIST_S) -> list[dict]:
    """All runs where |deviation| stayed beyond `threshold` for >= persist.

    A run breaks when tracking is lost, the deviation drops back inside
    the threshold, or frames gap by more than MAX_GAP_S. Returns one
    dict per firing: start/end elapsed seconds, duration, peak |dev|.
    """
    f = frames.sort_values("elapsed_s").reset_index(drop=True)
    events: list[dict] = []
    start: float | None = None
    prev_t: float | None = None
    peak = 0.0
    for row in f.itertuples():
        t = float(row.elapsed_s)
        dev = getattr(row, dev_col)
        ok = (row.tracking == 1 and pd.notna(dev) and abs(dev) > threshold)
        gap = (t - prev_t) if prev_t is not None else 0.0
        if ok and start is None:
            start, peak = t, abs(dev)
        if ok:
            peak = max(peak, abs(dev))
        if (not ok or gap > MAX_GAP_S) and start is not None:
            if prev_t - start >= persist_s:
                events.append({"start_s": round(start, 2), "end_s": round(prev_t, 2),
                               "duration_s": round(prev_t - start, 2),
                               "peak_dev": round(peak, 2)})
            start = None
        prev_t = t
    if start is not None and prev_t - start >= persist_s:
        events.append({"start_s": round(start, 2), "end_s": round(prev_t, 2),
                       "duration_s": round(prev_t - start, 2),
                       "peak_dev": round(peak, 2)})
    return events


def longest_run(frames: pd.DataFrame, dev_col: str, threshold: float) -> float:
    """Longest continuous stretch (seconds) with |deviation| > threshold.

    Useful for "how close did it come": a 1.5 s run against a 5 s
    requirement is nowhere near a detection, however big the peak was.
    """
    events = sustained_events(frames, dev_col, threshold, persist_s=0.0)
    return max((e["duration_s"] for e in events), default=0.0)


def _hold_windows(step_start: float, offset: float) -> list[tuple[float, float]]:
    """The 5 hold windows for a cue step, from protocol timing."""
    cycle = READY_S + HOLD_S + REST_S
    return [(step_start + offset + k * cycle + READY_S,
             step_start + offset + k * cycle + READY_S + HOLD_S)
            for k in range(TRIALS_PER_CUE)]


def _window_stats(frames: pd.DataFrame, dev_col: str,
                  lo: float, hi: float) -> dict:
    w = frames[(frames["elapsed_s"] >= lo) & (frames["elapsed_s"] < hi)][dev_col].dropna()
    if len(w) == 0:
        return {"samples": 0, "mean": float("nan"), "std": float("nan"),
                "max_abs": float("nan")}
    return {"samples": int(len(w)), "mean": round(float(w.mean()), 3),
            "std": round(float(w.std(ddof=1)), 3) if len(w) > 1 else 0.0,
            "max_abs": round(float(w.abs().max()), 3)}


def align_trials(frames: pd.DataFrame, session: SessionData, cue: str,
                 fallback_offset: float | None = None) -> dict:
    """Recover the trial clock offset and per-trial stats for one cue.

    Grid-searches the offset (0-12 s in 0.1 s steps) and keeps the one
    whose hold windows best reproduce the per-trial summary the
    experiment recorded (sample counts, means, stds). Returns the offset,
    the fit error, the per-trial stats, and whether the alignment is
    inferred (no recorded summary to fit against) rather than verified.
    """
    step = CUE_STEP[cue]
    dev_col = "h_dev" if cue == "head" else "s_dev"
    step_frames = frames[frames["step"] == step].sort_values("elapsed_s")
    if step_frames.empty:
        raise ValueError(f"no frames for step {step!r}")
    t0 = float(step_frames["elapsed_s"].iloc[0])
    summary = [r for r in session.trial_summary if r["cue"] == cue]

    def score(offset: float) -> tuple[float, list[dict]]:
        stats, err = [], 0.0
        for k, (lo, hi) in enumerate(_hold_windows(t0, offset)):
            s = _window_stats(step_frames, dev_col, lo, hi)
            stats.append(s)
            if k < len(summary):
                r = summary[k]
                err += abs(s["samples"] - r["samples"]) / 10.0
                err += abs((s["mean"] or 0) - r["head_mean" if cue == "head" else "shoulder_mean"])
                err += abs((s["std"] or 0) - r["head_std" if cue == "head" else "shoulder_std"])
        return err, stats

    if summary:
        best_offset, best_err, best_stats = 0.0, float("inf"), []
        for i in range(0, 121):
            err, stats = score(i / 10.0)
            if err < best_err:
                best_offset, best_err, best_stats = i / 10.0, err, stats
        return {"offset_s": best_offset, "fit_error": round(best_err, 2),
                "verified": True, "trials": best_stats}
    offset = fallback_offset if fallback_offset is not None else 0.0
    _, stats = score(offset)
    return {"offset_s": offset, "fit_error": None,
            "verified": False, "trials": stats}


def analyze_cue(frames: pd.DataFrame, session: SessionData, cue: str,
                threshold: float, alignment: dict,
                persist_s: float = PERSIST_S) -> dict:
    """Detection verdict for one cue at one threshold.

    A trial counts as detected if a sustained firing overlaps its hold
    window. Also reports the longest sustained run anywhere in the trial
    step, which shows how close a near-miss came. `persist_s` lets the
    Day 4 statistics ask \"what if the persistence rule were shorter?\"
    without changing the 5 s rule the experiment itself used.
    """
    step = CUE_STEP[cue]
    dev_col = "h_dev" if cue == "head" else "s_dev"
    step_frames = frames[frames["step"] == step].sort_values("elapsed_s")
    t0 = float(step_frames["elapsed_s"].iloc[0])
    events = sustained_events(step_frames, dev_col, threshold, persist_s=persist_s)
    detected = 0
    per_trial = []
    for k, (lo, hi) in enumerate(_hold_windows(t0, alignment["offset_s"])):
        hit = any(e["start_s"] < hi and e["end_s"] > lo for e in events)
        detected += int(hit)
        per_trial.append({**alignment["trials"][k], "detected": hit})
    return {
        "cue": cue, "threshold": threshold,
        "detected": detected, "trials_total": TRIALS_PER_CUE,
        "per_trial": per_trial,
        "events_in_step": events,
        "longest_run_s": longest_run(step_frames, dev_col, threshold),
    }


def false_positives(frames: pd.DataFrame, cue: str, threshold: float) -> dict:
    """Sustained firings during the ordinary-work sample (must be zero).

    Also returns what the experiment itself recorded, so the two can be
    compared: the page counted firings with its (then buggy) linear
    deviation, this harness recounts them with circular math.
    """
    dev_col = "h_dev" if cue == "head" else "s_dev"
    work = frames[frames["step"] == "work"].sort_values("elapsed_s")
    events = sustained_events(work, dev_col, threshold)
    recorded = frames[(frames["step"] == "work")
                      & (frames["cue_detected"] == cue)]
    return {"count": len(events), "events": events,
            "recorded_by_experiment": int(len(recorded)),
            "recorded_times_s": [round(float(t), 1)
                                 for t in recorded["elapsed_s"]]}


def loss_test(frames: pd.DataFrame, session: SessionData) -> dict:
    """Did the landmark-loss test actually run, and stay clean?

    A valid test needs landmarks to actually disappear (tracking == 0)
    for a real stretch while the camera is covered, and no cue firing
    during the loss step. Run 1 fails both: tracking was lost for only
    18 scattered frames, and head firings ran through the step.
    """
    loss = frames[frames["step"] == "loss"].sort_values("elapsed_s")
    lost = loss[loss["tracking"] == 0]
    head_events = sustained_events(loss, "h_dev", session.threshold_head)
    shoulder_events = sustained_events(loss, "s_dev", session.threshold_shoulder)
    return {
        "frames_total": int(len(loss)),
        "tracking_lost_frames": int(len(lost)),
        "tracking_lost_seconds": round(float(len(lost)) * 0.1, 1),
        "head_events": head_events,
        "shoulder_events": shoulder_events,
        "ran": bool(len(lost) >= 20),
        "clean": not head_events and not shoulder_events,
    }


def sensitivity(frames: pd.DataFrame, session: SessionData) -> list[dict]:
    """What would each threshold have decided? One row per threshold.

    The key learning artifact: if dropping the threshold from 8 to 5
    still gives 0/5 detections, the problem is not the threshold, it
    is the movement itself (too small, too brief, or too noisy).
    """
    head_align = align_trials(frames, session, "head")
    shoulder_align = align_trials(frames, session, "shoulder",
                                  fallback_offset=head_align["offset_s"])
    rows: list[dict] = []
    for threshold in HEAD_THRESHOLDS:
        a = analyze_cue(frames, session, "head", threshold, head_align)
        fp = false_positives(frames, "head", threshold)
        rows.append({"cue": "head", "threshold": threshold,
                     "detected": f"{a['detected']}/{TRIALS_PER_CUE}",
                     "false_positives": fp["count"],
                     "longest_run_s": a["longest_run_s"]})
    for threshold in SHOULDER_THRESHOLDS:
        a = analyze_cue(frames, session, "shoulder", threshold, shoulder_align)
        fp = false_positives(frames, "shoulder", threshold)
        rows.append({"cue": "shoulder", "threshold": threshold,
                     "detected": f"{a['detected']}/{TRIALS_PER_CUE}",
                     "false_positives": fp["count"],
                     "longest_run_s": a["longest_run_s"]})
    return rows


def verdict(frames: pd.DataFrame, session: SessionData) -> dict:
    """Full verdict for a session at the thresholds recorded in the file."""
    head_align = align_trials(frames, session, "head")
    shoulder_align = align_trials(frames, session, "shoulder",
                                  fallback_offset=head_align["offset_s"])
    loss = loss_test(frames, session)
    cues: dict[str, dict] = {}
    for cue, threshold, alignment in [
            ("head", session.threshold_head, head_align),
            ("shoulder", session.threshold_shoulder, shoulder_align)]:
        a = analyze_cue(frames, session, cue, threshold, alignment)
        fp = false_positives(frames, cue, threshold)
        reasons = []
        if a["detected"] < 4:
            reasons.append(f"only {a['detected']}/5 holds detected (need >= 4)")
        if fp["count"] > 0:
            reasons.append(f"{fp['count']} false detections during ordinary work (need 0)")
        if not loss["ran"]:
            reasons.append("landmark-loss test never actually ran")
        elif not loss["clean"]:
            reasons.append("cue fired during the landmark-loss step")
        cues[cue] = {
            "threshold": threshold, "detected": a["detected"],
            "false_positives": fp["count"], "pass": not reasons,
            "reasons": reasons, "analysis": a, "false_positive_detail": fp,
            "alignment": alignment,
        }
    return {"cues": cues, "loss_test": loss,
            "sensitivity": sensitivity(frames, session)}
