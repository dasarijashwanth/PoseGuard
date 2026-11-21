"""PoseGuard live webcam check: pure angle and gate logic (Day 19).

No Streamlit, MediaPipe, or OpenCV import here, so the test suite can
exercise this module without a camera, a server, or a browser. The UI
lives in demo/app.py and only calls these helpers.

Angle math matches the Day 14 capture tool
(experiments/rerun-capture.html), so a live check measures the same
quantities Run 1 measured:

- head raw angle: the angle of the ear-to-ear line (MediaPipe pose
  landmarks 7 and 8) minus the angle of the shoulder line
  (landmarks 11 and 12), taken circularly. A level head on level
  shoulders reads 0 deg.
- shoulder raw angle: the shoulder line angle itself, wrapped to
  +/-180 deg. It is a level, not a deviation: leaning at a desk moves
  it, which is exactly why the Day 4 finding still applies and the
  live check reports it as a level alongside the deviation.
- deviations are circular differences from a calibration baseline: the
  circular mean of the upright pose over the first seconds of the
  check. Calibrate upright, then tilt or drop a shoulder and hold it.

The live gate is the same rule the grid search scored (Days 15/17):
a deviation above a threshold, sustained for a persistence time, with
a velocity cap that breaks the run when the line jumps too fast.
"""

from __future__ import annotations

import math

# MediaPipe pose landmark indices used by the live check. Only four
# landmarks matter: the ears (7, 8) and the shoulders (11, 12).
EAR_LEFT = 7
EAR_RIGHT = 8
SHOULDER_LEFT = 11
SHOULDER_RIGHT = 12
MIN_LANDMARKS = 13

# Live-check defaults: the session operating thresholds (head 8 deg,
# shoulder 6 deg) with a 5 s hold, the same "hold it" instruction the
# Day 14 capture tool gives. The best rule-grid configs are offered as
# an alternative preset in the UI so visitors can see why they fail
# (the shoulder best rule is 1 deg, which everyday leaning clears).
LIVE_PRESETS = {
    "operating": {
        "head": {"threshold_deg": 8, "persist_s": 5.0, "vel_cap_deg_s": None},
        "shoulder": {"threshold_deg": 6, "persist_s": 5.0, "vel_cap_deg_s": None},
    },
    "best_rule": {
        "head": {"threshold_deg": 4, "persist_s": 2.0, "vel_cap_deg_s": 10.0},
        "shoulder": {"threshold_deg": 1, "persist_s": 5.0, "vel_cap_deg_s": 5.0},
    },
}

# Minimum calibration samples before a baseline is trusted (about 1 s
# at the snapshot rate, a few frames of video).
MIN_CALIBRATION_SAMPLES = 5


def wrap_angle(deg: float) -> float:
    """Wrap an angle to [-180, 180], matching rerun-capture.html."""
    a = float(deg)
    while a > 180:
        a -= 360
    while a < -180:
        a += 360
    return a


def circ_dev(a: float, b: float) -> float:
    """Circular deviation of angle a from angle b, in deg, in [-180, 180]."""
    d = (float(a) - float(b)) * math.pi / 180
    return math.atan2(math.sin(d), math.cos(d)) * 180 / math.pi


def circ_mean(angles) -> float:
    """Circular mean of angles in deg. Raises ValueError on empty input."""
    xs = [float(a) for a in angles]
    if not xs:
        raise ValueError("circ_mean needs at least one angle")
    s = sum(math.sin(x * math.pi / 180) for x in xs)
    c = sum(math.cos(x * math.pi / 180) for x in xs)
    return math.atan2(s / len(xs), c / len(xs)) * 180 / math.pi


def angle_deg(ax: float, ay: float, bx: float, by: float) -> float:
    """Angle in deg of the line from (ax, ay) to (bx, by).

    Image coordinates: y grows downward, so a line that drops to the
    right has a positive angle. Only the difference between two such
    angles matters for the head cue, so the sign convention cancels.
    """
    return math.degrees(math.atan2(by - ay, bx - ax))


def _xy(landmark):
    """One landmark as an (x, y) pair.

    Accepts a (x, y) tuple/list or an object with .x and .y (the
    MediaPipe landmark shape), so tests can pass plain tuples while
    the app passes real landmarks unchanged.
    """
    if hasattr(landmark, "x") and hasattr(landmark, "y"):
        return float(landmark.x), float(landmark.y)
    x, y = landmark[0], landmark[1]
    if x is None or y is None:
        raise ValueError("landmark coordinate is None")
    return float(x), float(y)


def landmark_angles(landmarks) -> tuple[float, float]:
    """Head and shoulder raw angles in deg from pose landmarks.

    landmarks is a sequence with at least 13 entries (MediaPipe pose
    returns 33). Returns (head_raw_deg, shoulder_raw_deg). Raises
    ValueError when the needed landmarks are missing or unusable, so
    the app can report "tracking lost" instead of inventing angles.
    """
    lm = list(landmarks)
    if len(lm) < MIN_LANDMARKS:
        raise ValueError(
            f"need at least {MIN_LANDMARKS} landmarks, got {len(lm)}"
        )
    try:
        ear_l = _xy(lm[EAR_LEFT])
        ear_r = _xy(lm[EAR_RIGHT])
        sh_l = _xy(lm[SHOULDER_LEFT])
        sh_r = _xy(lm[SHOULDER_RIGHT])
    except (TypeError, IndexError, ValueError) as exc:
        raise ValueError(f"unusable landmark: {exc}") from exc
    if any(math.isnan(v) for p in (ear_l, ear_r, sh_l, sh_r) for v in p):
        raise ValueError("landmark coordinate is NaN")
    ear_a = angle_deg(*ear_l, *ear_r)
    sh_a = angle_deg(*sh_l, *sh_r)
    head_raw = circ_dev(ear_a, sh_a)
    shoulder_raw = wrap_angle(sh_a)
    return head_raw, shoulder_raw


def calibrate(samples) -> tuple[float, float]:
    """Calibration baseline from upright samples.

    samples is a sequence of (head_raw_deg, shoulder_raw_deg) taken
    while the visitor sits upright. Returns (head_base, shoulder_base)
    as circular means. Raises ValueError on too few samples.
    """
    rows = list(samples)
    if len(rows) < MIN_CALIBRATION_SAMPLES:
        raise ValueError(
            f"need at least {MIN_CALIBRATION_SAMPLES} calibration samples, "
            f"got {len(rows)}"
        )
    head_base = circ_mean([r[0] for r in rows])
    shoulder_base = circ_mean([r[1] for r in rows])
    return head_base, shoulder_base


def deviations(head_raw: float, shoulder_raw: float,
               baseline: tuple[float, float]) -> tuple[float, float]:
    """Circular deviations from the calibration baseline, in deg."""
    head_base, shoulder_base = baseline
    return circ_dev(head_raw, head_base), circ_dev(shoulder_raw, shoulder_base)


def ema_smooth(prev: float | None, new: float, alpha: float = 0.3) -> float:
    """Circular exponential moving average, as in rerun-capture.html.

    The first sample seeds the smoother. Alpha 0.3 matches the capture
    tool; a smaller alpha is steadier, a larger one is more responsive.
    """
    if prev is None:
        return float(new)
    return wrap_angle(prev + alpha * circ_dev(new, prev))


class LiveGate:
    """One cue's persistence gate over a timestamped angle stream.

    Fires when the absolute deviation stays above threshold_deg for
    persist_s continuously. A sample whose frame-to-frame speed exceeds
    vel_cap_deg_s breaks the run (a jump is tracker noise, not a
    hold), and a NaN deviation (tracking lost) breaks it too. Once
    fired, the gate stays fired until reset(), so the UI can hold the
    verdict on screen while the visitor sits back up.
    """

    def __init__(self, threshold_deg: float, persist_s: float,
                 vel_cap_deg_s: float | None = None):
        if threshold_deg <= 0:
            raise ValueError("threshold_deg must be positive")
        if persist_s <= 0:
            raise ValueError("persist_s must be positive")
        if vel_cap_deg_s is not None and vel_cap_deg_s <= 0:
            raise ValueError("vel_cap_deg_s must be positive when set")
        self.threshold_deg = float(threshold_deg)
        self.persist_s = float(persist_s)
        self.vel_cap_deg_s = vel_cap_deg_s
        self.run_s = 0.0
        self.fired = False
        self.fired_at: float | None = None
        self.last_t: float | None = None
        self.last_dev: float | None = None

    def reset(self) -> None:
        self.run_s = 0.0
        self.fired = False
        self.fired_at = None
        self.last_t = None
        self.last_dev = None

    def update(self, t: float, dev: float | None) -> dict:
        """Feed one sample. Returns the current gate state as a dict."""
        if dev is None or (isinstance(dev, float) and math.isnan(dev)):
            self.run_s = 0.0
            self.last_t = None
            self.last_dev = None
            return self.state()
        dev = float(dev)
        if self.last_t is not None and self.last_dev is not None:
            dt = float(t) - self.last_t
            if dt > 0:
                speed = abs(circ_dev(dev, self.last_dev)) / dt
                if self.vel_cap_deg_s is not None and speed > self.vel_cap_deg_s:
                    self.run_s = 0.0
                elif abs(dev) >= self.threshold_deg:
                    self.run_s += dt
                else:
                    self.run_s = 0.0
            # dt <= 0 (duplicate timestamp) leaves the run untouched.
        elif abs(dev) >= self.threshold_deg:
            self.run_s = 0.0  # first sample starts the run, no time yet
        self.last_t = float(t)
        self.last_dev = dev
        if not self.fired and self.run_s >= self.persist_s:
            self.fired = True
            self.fired_at = float(t)
        return self.state()

    def state(self) -> dict:
        return {
            "threshold_deg": self.threshold_deg,
            "persist_s": self.persist_s,
            "vel_cap_deg_s": self.vel_cap_deg_s,
            "run_s": round(self.run_s, 2),
            "fired": self.fired,
            "fired_at": self.fired_at,
        }


def evaluate_stream(samples, threshold_deg: float, persist_s: float,
                    vel_cap_deg_s: float | None = None) -> dict:
    """Run a LiveGate over a finished (t, dev) sample list.

    Returns {"fired": bool, "fired_at": float or None, "max_run_s": float}.
    The snapshot path uses this: each camera snapshot is one sample, and
    repeated snapshots while holding a pose build the stream.
    """
    gate = LiveGate(threshold_deg, persist_s, vel_cap_deg_s)
    max_run = 0.0
    for t, dev in samples:
        state = gate.update(t, dev)
        max_run = max(max_run, state["run_s"])
    return {
        "fired": gate.fired,
        "fired_at": gate.fired_at,
        "max_run_s": round(max_run, 2),
    }
