"""Day 23: combined-cue joint score.

Days 3-17 scored each cue alone. The best single-cue rules are the
shoulder jump-gated level (1 deg, 5 s: 4/5 trials, 8 false
positives, F1 0.471) and the head level (4 deg, 2 s, 10 deg/s cap:
2/5 trials, 3 false positives, F1 0.400). Both fail the gate on
false positives, and at the gate's own 5 s persistence the head
level detects 0/5 at every threshold 4-8 deg: the holds are too
brief as well as too shallow (Day 15). Day 23 asks whether the two
cues together do better than either alone.

Three combinations, and why only one of them is a real candidate:

- isolation (difference): |shoulder| minus |head| (and the mirror
  image for the head cue). The idea is that whole-body lean moves
  both cues together, so subtracting the other cue should cancel
  it. It does not work on Run 1: during a shoulder hold the head
  moves a little too, so the subtraction eats the 2.5 deg hold,
  while during ordinary work the head stays relatively still
  (mean |head deviation| 0.9-3.3 deg inside the gated rule's 8
  false-positive episodes), so the subtraction barely touches the
  false positives. Best: shoulder 3/5 with 9 FP, head 1/5 with
  0 FP. Both are worse than the single-cue rules.
- normalized sum: |head| / 5 + |shoulder gated| / 1. Each cue is
  scaled by its own operating threshold (head 5 deg, shoulder
  1 deg), so 1.0 means "one cue at its operating level". A hold
  moves both angles a little, and the two small deviations add
  up to cross a low threshold neither cue sustains alone. This
  is the best rule on Run 1 by F1 on both cues (shoulder 5/5,
  8 FP, F1 0.556; head 4/5, 8 FP, F1 0.471, all at the 5 s
  gate persistence). It still fails the gate: the same 8
  shoulder-drift episodes fire it during work, and the head cue
  inherits all 8 of them.
- AND (control): a firing needs both cues above their operating
  thresholds at once. It detects 0/5 on both cues, 0 FP: the
  deliberate holds are single-cue by construction, so a rule
  that requires both can never see them. This is the control
  that shows why the combination has to be a joint score, not
  a conjunction.

The isolation scores are floored at 0: a negative difference
(shoulder dominating on the head score, or the reverse) is not a
tilt in the other direction, it is the other cue firing, and
`sustained_events` takes the absolute value, so an unclamped
negative score would fire the wrong cue. Clamping changed the
head isolation result (4 false positives at 2 deg became 0);
the shoulder numbers are unchanged.

The gate is 4/5 trials with 0 false positives, per cue. The
evaluation below runs the same Day 3-4 protocol (threshold sweep
at the 5 s persistence, persistence curve at the best threshold)
for each combination, on both cues, with the head alignment
verified and the shoulder alignment inferred from it, exactly as
Days 3-4 and 11 did. Timing args are the protocol timings, Run 1
by default, Run 2 (5 / 10 / 5) for a re-run capture.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .analysis import (
    PERSIST_S,
    TRIALS_PER_CUE,
    align_trials,
    analyze_cue,
    false_positives,
)
from .head import trial_f1
from .loader import SessionData

H_SCALE_DEG = 5.0    # head operating scale (head grid thresholds 4-8 deg)
S_SCALE_DEG = 1.0    # shoulder operating scale (gated 1 deg threshold)
SUM_THRESHOLDS = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
ISO_S_THRESHOLDS = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
ISO_H_THRESHOLDS = (2.0, 4.0, 6.0, 8.0, 10.0)
PERSIST_GRID_S = (0.5, 2.0, 5.0)
AND_HEAD_DEG = 5.0        # AND control: head must exceed this too
AND_SHOULDER_DEG = 1.0    # AND control: shoulder must exceed this too

# cue -> variant -> (deviation column, thresholds, validity column or None)
# The AND control is not a column: it is the base cue's own deviation
# with a validity column requiring the other cue above its operating
# threshold. See evaluate_combined.
VARIANTS = {
    "shoulder": {
        "iso": ("j_dev_s", ISO_S_THRESHOLDS, None),
        "sum": ("j_dev_sum", SUM_THRESHOLDS, None),
    },
    "head": {
        "iso": ("j_dev_h", ISO_H_THRESHOLDS, None),
        "sum": ("j_dev_sum", SUM_THRESHOLDS, None),
    },
}

# cue -> (base deviation column, base threshold, other-cue validity column)
AND_CONTROL = {
    "shoulder": ("s_dev_gated", AND_SHOULDER_DEG, "j_head_above"),
    "head": ("h_dev", 4.0, "j_shoulder_above"),
}


def add_combined(frames: pd.DataFrame) -> pd.DataFrame:
    """Return frames with the Day 23 joint-score columns added.

    Adds j_dev_s (shoulder excess over head, floored at 0), j_dev_h
    (head excess over shoulder, floored at 0), j_dev_sum (the
    normalized sum), and the two AND-control boolean columns
    j_head_above / j_shoulder_above. The input must already carry
    h_dev (see experiments.eval.analysis.add_deviations) and
    s_dev_gated (see experiments.eval.signals.add_signals). A NaN
    in either input makes the joint score NaN, so a tracking loss
    breaks a persistence run instead of bridging it.
    """
    out = frames.copy()
    h_abs = out["h_dev"].abs()
    s_abs = out["s_dev_gated"].abs()
    out["j_dev_s"] = np.maximum(0.0, s_abs - h_abs)
    out["j_dev_h"] = np.maximum(0.0, h_abs - s_abs)
    out["j_dev_sum"] = h_abs / H_SCALE_DEG + s_abs / S_SCALE_DEG
    out["j_head_above"] = h_abs > AND_HEAD_DEG
    out["j_shoulder_above"] = s_abs > AND_SHOULDER_DEG
    return out


def _sweep(frames: pd.DataFrame, session: SessionData, cue: str,
           alignment: dict, dev_col: str, thresholds: tuple,
           valid_col: str | None = None,
           persist_s: float = PERSIST_S,
           ready_s: float = 3.0, hold_s: float = 6.0,
           rest_s: float = 3.0) -> dict:
    """Threshold sweep for one variant at one persistence.

    Returns the per-threshold rows and the best row by F1 (ties
    toward more trials detected, then fewer false positives), each
    row carrying whether it passes the gate (4/5 trials, 0 FP).
    The timing args are the protocol timings, Run 1 by default,
    Run 2 (5 / 10 / 5) for a re-run capture.
    """
    rows: list[dict] = []
    for threshold in thresholds:
        a = analyze_cue(frames, session, cue, threshold, alignment,
                        persist_s=persist_s, dev_col=dev_col,
                        valid_col=valid_col,
                        ready_s=ready_s, hold_s=hold_s, rest_s=rest_s)
        fp = false_positives(frames, cue, threshold,
                             dev_col=dev_col, valid_col=valid_col)
        f1 = trial_f1(a["detected"], fp["count"])
        rows.append({"threshold": threshold,
                     "detected": a["detected"],
                     "trials_total": TRIALS_PER_CUE,
                     "false_positives": fp["count"],
                     **f1,
                     "pass": a["detected"] >= 4 and fp["count"] == 0})
    best = max(rows, key=lambda r: (r["f1"], r["detected"],
                                    -r["false_positives"]))
    return {"dev_col": dev_col, "valid_col": valid_col,
            "persist_s": persist_s, "rows": rows, "best": best}


def _persistence_curve(frames: pd.DataFrame, session: SessionData,
                       cue: str, alignment: dict, dev_col: str,
                       threshold: float,
                       valid_col: str | None = None,
                       ready_s: float = 3.0, hold_s: float = 6.0,
                       rest_s: float = 3.0) -> list[dict]:
    """Detection and FP at 0.5 / 2 / 5 s persistence, one threshold."""
    curve: list[dict] = []
    for persist_s in PERSIST_GRID_S:
        a = analyze_cue(frames, session, cue, threshold, alignment,
                        persist_s=persist_s, dev_col=dev_col,
                        valid_col=valid_col,
                        ready_s=ready_s, hold_s=hold_s, rest_s=rest_s)
        fp = false_positives(frames, cue, threshold,
                             dev_col=dev_col, valid_col=valid_col)
        curve.append({"persist_s": persist_s, "threshold": threshold,
                      "detected": a["detected"],
                      "trials_total": TRIALS_PER_CUE,
                      "false_positives": fp["count"],
                      "pass": a["detected"] >= 4 and fp["count"] == 0})
    return curve


def evaluate_combined(frames: pd.DataFrame, session: SessionData,
                      ready_s: float = 3.0, hold_s: float = 6.0,
                      rest_s: float = 3.0) -> dict:
    """Joint-score evaluation on both cues, same protocol as Days 11-17.

    The head alignment is verified against the recorded per-trial
    summary; the shoulder alignment is inferred from the head offset,
    exactly as Days 3-4 and 11 did (Run 1 recorded no shoulder
    trial summary). Per cue, each variant (isolation, normalized
    sum) gets a threshold sweep at the 5 s gate persistence plus a
    persistence curve at its best threshold, and the AND control
    gets its single configuration. `best` per variant is the
    highest F1, ties toward more trials, then fewer false
    positives. The timing args are the protocol timings, Run 1 by
    default, Run 2 (5 / 10 / 5) for a re-run capture.
    """
    head_align = align_trials(frames, session, "head",
                              ready_s=ready_s, hold_s=hold_s, rest_s=rest_s)
    shoulder_align = align_trials(
        frames, session, "shoulder",
        fallback_offset=head_align["offset_s"],
        ready_s=ready_s, hold_s=hold_s, rest_s=rest_s)
    alignments = {"head": head_align, "shoulder": shoulder_align}
    cues: dict[str, dict] = {}
    for cue in ("head", "shoulder"):
        alignment = alignments[cue]
        variants: dict[str, dict] = {}
        for name, (dev_col, thresholds, valid_col) in VARIANTS[cue].items():
            result = _sweep(frames, session, cue, alignment, dev_col,
                            thresholds, valid_col=valid_col,
                            ready_s=ready_s, hold_s=hold_s, rest_s=rest_s)
            result["persistence_curve"] = _persistence_curve(
                frames, session, cue, alignment, dev_col,
                result["best"]["threshold"], valid_col=valid_col,
                ready_s=ready_s, hold_s=hold_s, rest_s=rest_s)
            variants[name] = result
        dev_col, threshold, valid_col = AND_CONTROL[cue]
        and_result = _sweep(frames, session, cue, alignment, dev_col,
                            (threshold,), valid_col=valid_col,
                            ready_s=ready_s, hold_s=hold_s, rest_s=rest_s)
        variants["and_control"] = and_result
        cues[cue] = {
            "alignment_offset_s": alignment["offset_s"],
            "alignment_verified": alignment["verified"],
            "variants": variants,
        }
    return {"alignment_offset_s": shoulder_align["offset_s"],
            "alignment_verified": shoulder_align["verified"],
            "cues": cues}
