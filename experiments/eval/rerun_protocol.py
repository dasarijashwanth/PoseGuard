"""Run 2 re-run protocol (Day 14).

The Day 11-13 verdict is that no feature built from the Run 1 CSV can
fix the signal: deliberate holds move the shoulder angle only about
2.5 deg while ordinary work drifts 4-7 deg, and the CSV has only
angle columns, no raw landmarks. The fix is a camera re-run with raw
landmarks and longer, deeper holds. This module is the protocol the
re-run capture tool implements (see experiments/rerun-capture.html)
and the validator checks (see experiments/eval/rerun.py).

What changed vs Run 1, and why:

- Holds are 10 s instead of 6 s. The 5 s persistence gate needs a
  hold comfortably longer than 5 s after the move into position and
  the EMA ramp-up. Run 1's 6 s holds left about 1 s of margin; a
  10 s hold leaves 4-5 s of clean sustained signal per hold.
- Ready and rest are 5 s instead of 3 s, so the move out of a hold
  and back to upright finishes before the next hold starts, and the
  ready window gives the tracker time to settle.
- Calibration is 5 s instead of 3 s, for a steadier baseline.
- Holds have a depth target, shown live while holding: shoulder
  8 deg, head 15 deg. Run 1 shoulder holds averaged about 2.5 deg
  of deviation, inside the 4-7 deg everyday drift band, so no level
  threshold could separate them. An 8 deg shoulder target sits above
  the drift band: if the new data separates, the level signal itself
  becomes usable again. A 15 deg head tilt is a clear deliberate
  tilt, and the head fix is duration more than depth (Run 1 head
  tilts were real but too brief for the 5 s rule).
- Every frame records all 33 pose landmarks (x, y, visibility), not
  just the two angles. Hips 23/24 and shoulder heights are the two
  signals Run 1 could not test at all: shoulder-vs-hip and shoulder
  height asymmetry. Tracking now requires hips 23/24 visible too,
  because a frame without hips cannot test the shoulder-vs-hip
  signal; the tool tells the user to step back when hips leave the
  frame.
- The work sample stays 5 min and the landmark-loss test stays 10 s,
  so false positives and loss behaviour stay comparable to Run 1.
  Trials stay 5 per cue, so the gate (4/5 trials, 0 false
  positives) is unchanged.

The session runs about 8.6 minutes: calibration 5 s, head trials
100 s, shoulder trials 100 s, work 300 s, loss test 10 s.
"""

from __future__ import annotations

# --- protocol timings, seconds -------------------------------------------
CALIB_S = 5.0       # calibration, tracked time only (was 3 s in Run 1)
READY_S = 5.0       # get-ready before each hold (was 3 s)
HOLD_S = 10.0       # hold duration (was 6 s)
REST_S = 5.0        # rest between holds (was 3 s)
WORK_S = 300.0      # ordinary-work sample (unchanged)
LOSS_S = 10.0       # landmark-loss test (unchanged)
TRIALS_PER_CUE = 5  # holds per cue (unchanged, the gate is 4/5)

# --- depth targets, degrees of deviation from the calibration baseline ---
TARGET_HEAD_DEG = 15.0      # head tilt target during a head hold
TARGET_SHOULDER_DEG = 8.0   # shoulder tilt target during a shoulder hold

# A hold counts as detected when the deviation stays beyond its target
# for at least this long inside the hold.
DETECT_PERSIST_S = 5.0

# --- landmarks ------------------------------------------------------------
# MediaPipe Pose Landmarker indices. 7/8 are the ears (head angle),
# 11/12 the shoulders (shoulder angle), 23/24 the hips (the two
# signals Run 1 could not test). All 33 landmarks are recorded.
LANDMARK_IDS = list(range(33))
REQUIRED_LANDMARK_IDS = (7, 8, 11, 12, 23, 24)
EAR_IDS = (7, 8)
SHOULDER_IDS = (11, 12)
HIP_IDS = (23, 24)

# Phases a frame can carry, and the cue values it can carry.
PHASES = ("calib", "ready", "hold", "rest", "calm", "loss")
CUES = ("head", "shoulder")

# Frame columns the re-run CSV adds after the nine Run 1 columns
# (timestamp, elapsed_s, step, head_tilt, shoulder_tilt, cue_detected,
# tracking, head_deviation, shoulder_deviation). The new CSV is a
# superset of the Run 1 schema, so the existing loader still reads it.
EXTRA_COLUMNS = ["cue", "hold_id", "phase"]


def landmark_columns() -> list[str]:
    """All 33 landmarks as x/y/visibility columns, in index order."""
    cols: list[str] = []
    for i in LANDMARK_IDS:
        cols.extend([f"lm_{i}_x", f"lm_{i}_y", f"lm_{i}_v"])
    return cols


def frame_columns() -> list[str]:
    """Full Run 2 frame header: Run 1 columns, labels, landmarks."""
    from .loader import REQUIRED_COLUMNS

    return REQUIRED_COLUMNS + EXTRA_COLUMNS + landmark_columns()


def target_deg(cue: str) -> float:
    """Depth target in degrees for one cue's holds."""
    if cue == "head":
        return TARGET_HEAD_DEG
    if cue == "shoulder":
        return TARGET_SHOULDER_DEG
    raise ValueError(f"unknown cue {cue!r}, expected 'head' or 'shoulder'")


def trial_block_timeline(cue: str) -> list[dict]:
    """Phase timeline for one cue's trial block, relative to its start.

    Five trials, each ready + hold + rest: 5 x (5 + 10 + 5) = 100 s.
    Each entry has phase, hold_id (1-5), start_s and end_s.
    """
    out: list[dict] = []
    t = 0.0
    for hold_id in range(1, TRIALS_PER_CUE + 1):
        for phase, dur in (("ready", READY_S), ("hold", HOLD_S),
                           ("rest", REST_S)):
            out.append({"cue": cue, "phase": phase, "hold_id": hold_id,
                        "start_s": t, "end_s": t + dur, "duration_s": dur})
            t += dur
    return out


def session_timeline() -> list[dict]:
    """Full session timeline, in order, relative to session start.

    Calibration, head trials, shoulder trials, work sample, loss test:
    5 + 100 + 100 + 300 + 10 = 515 s, about 8.6 minutes.
    """
    out: list[dict] = [
        {"cue": "", "phase": "calib", "hold_id": None,
         "start_s": 0.0, "end_s": CALIB_S, "duration_s": CALIB_S},
    ]
    t = CALIB_S
    for cue in ("head", "shoulder"):
        for entry in trial_block_timeline(cue):
            out.append({**entry, "start_s": entry["start_s"] + t,
                        "end_s": entry["end_s"] + t})
        t += TRIALS_PER_CUE * (READY_S + HOLD_S + REST_S)
    out.append({"cue": "", "phase": "calm", "hold_id": None,
                "start_s": t, "end_s": t + WORK_S, "duration_s": WORK_S})
    t += WORK_S
    out.append({"cue": "", "phase": "loss", "hold_id": None,
                "start_s": t, "end_s": t + LOSS_S, "duration_s": LOSS_S})
    return out


def session_duration_s() -> float:
    """Total session length in seconds (515 s)."""
    return CALIB_S + 2 * TRIALS_PER_CUE * (READY_S + HOLD_S + REST_S) \
        + WORK_S + LOSS_S
