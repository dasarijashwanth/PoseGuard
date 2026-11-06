"""Day 4: statistical validation of the two posture cues.

Day 3 asked "did it work at each threshold?" Day 4 asks the harder
question: "could ANY threshold or persistence rule make it work, or is
the cue itself wrong?" Three tools, all standard library plus numpy:

- Wilson score intervals on the 0/5-style detection counts. With only
  5 trials, a raw 0/5 does not mean "the true rate is zero", it means
  "the true rate is plausibly anywhere up to ~43%". The interval makes
  that honest, and it also shows 0/5 cannot plausibly be the >= 4/5
  the gate needs (the interval's top end sits far below 80%).
- False-positive rates per hour of ordinary work, with a normal
  approximation interval. A cue that fires 7 times in 5 minutes of
  typing is unusable no matter what the detection rate says.
- Threshold and persistence curves: detection and false positives as
  functions of the two knobs, plus the lowest threshold that kills all
  false positives and what detection looks like there. If no knob
  setting passes both criteria, the metric itself is the problem, not
  the settings.
"""

from __future__ import annotations

import math

from .analysis import (
    HEAD_THRESHOLDS,
    PERSIST_S,
    SHOULDER_THRESHOLDS,
    TRIALS_PER_CUE,
    analyze_cue,
    false_positives,
    verdict,
)

PASS_DETECTED = 4          # gate: at least 4 of 5 holds detected
Z_95 = 1.96                # 95% confidence
SCAN_MAX_DEG = 30          # highest threshold tried when hunting zero FPs
PERSIST_GRID_S = (0.5, 1.0, 1.5, 2.0, 3.0, 5.0)
CUE_THRESHOLDS = {"head": HEAD_THRESHOLDS, "shoulder": SHOULDER_THRESHOLDS}


def wilson_interval(k: int, n: int, z: float = Z_95) -> tuple[float, float]:
    """Wilson score interval for a proportion k/n (95% by default).

    The honest interval for small samples. A plain k/n with 5 trials
    pretends to know more than it does; Wilson widens correctly near
    0 and 1, where the normal approximation breaks. E.g. 0/5 gives
    (0.0, 0.43): the true rate could be as high as 43%, but not 80%.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 <= k <= n:
        raise ValueError("k must be between 0 and n")
    p = k / n
    z2 = z * z
    denom = 1 + z2 / n
    centre = (p + z2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def fp_rate_per_hour(count: int, work_seconds: float) -> float:
    """False positives per hour of ordinary work."""
    if work_seconds <= 0:
        return float("nan")
    return count / work_seconds * 3600.0


def fp_rate_interval(count: int, work_seconds: float,
                     z: float = Z_95) -> tuple[float, float]:
    """Approximate 95% interval for the FP rate per hour.

    Normal approximation to the Poisson count (rate +/- z * sqrt(count)
    over the exposure), floored at zero. Good enough at the counts we
    see (4-12); with 0 counts it correctly returns (0, 0) only as a
    point statement, so callers should read 0 FPs as "none seen in
    this much work", not "the true rate is zero".
    """
    if work_seconds <= 0:
        return (float("nan"), float("nan"))
    rate = fp_rate_per_hour(count, work_seconds)
    half = z * math.sqrt(count) / work_seconds * 3600.0
    return (max(0.0, rate - half), rate + half)


def work_seconds(frames) -> float:
    """Length of the ordinary-work sample, in seconds."""
    work = frames[frames["step"] == "work"]
    if len(work) == 0:
        return 0.0
    return float(work["elapsed_s"].max() - work["elapsed_s"].min())


def threshold_curve(frames, session, cue: str, alignment: dict) -> list[dict]:
    """Detection rate and FP rate at every tested threshold, with intervals.

    One row per threshold: detected count, Wilson interval on the
    detection rate, FP count during ordinary work, FP rate per hour
    with interval, and the longest sustained run in the trial step.
    This is the curve the verdict reads off: a usable cue needs some
    row where detection is high AND false positives are zero.
    """
    rows: list[dict] = []
    work_s = work_seconds(frames)
    for threshold in CUE_THRESHOLDS[cue]:
        a = analyze_cue(frames, session, cue, threshold, alignment)
        fp = false_positives(frames, cue, threshold)
        lo, hi = wilson_interval(a["detected"], TRIALS_PER_CUE)
        fp_lo, fp_hi = fp_rate_interval(fp["count"], work_s)
        rows.append({
            "cue": cue, "threshold": threshold,
            "detected": a["detected"], "trials_total": TRIALS_PER_CUE,
            "detection_rate": round(a["detected"] / TRIALS_PER_CUE, 3),
            "ci_low": round(lo, 3), "ci_high": round(hi, 3),
            "false_positives": fp["count"],
            "fp_per_hour": round(fp_rate_per_hour(fp["count"], work_s), 1),
            "fp_per_hour_low": round(fp_lo, 1),
            "fp_per_hour_high": round(fp_hi, 1),
            "longest_run_s": a["longest_run_s"],
        })
    return rows


def persistence_curve(frames, session, cue: str, threshold: float,
                      alignment: dict,
                      grid: tuple[float, ...] = PERSIST_GRID_S) -> list[dict]:
    """What would each persistence rule have decided?

    Detection count if the firing rule required 0.5, 1, 1.5, 2, 3 or
    5 seconds of sustained breach instead of 5. If detection only
    appears at very short persistence, the holds were too brief for
    the 5 s rule, and the cue is salvageable by longer holds (or a
    shorter rule), not by a different threshold.
    """
    rows: list[dict] = []
    for persist_s in grid:
        a = analyze_cue(frames, session, cue, threshold, alignment,
                        persist_s=persist_s)
        rows.append({"persist_s": persist_s, "threshold": threshold,
                     "detected": a["detected"],
                     "trials_total": TRIALS_PER_CUE})
    return rows


def zero_fp_threshold(frames, session, cue: str, alignment: dict,
                      scan_max: int = SCAN_MAX_DEG) -> dict | None:
    """Lowest threshold (1..scan_max deg) with zero false positives.

    Returns the threshold and the detection count there, or None if no
    threshold up to scan_max kills all false positives. The point of
    the scan: if the threshold that silences the false alarms also
    silences every real hold, the cue cannot pass at any setting.
    """
    for threshold in range(1, scan_max + 1):
        fp = false_positives(frames, cue, threshold)
        if fp["count"] == 0:
            a = analyze_cue(frames, session, cue, threshold, alignment)
            return {"threshold": threshold, "detected": a["detected"],
                    "trials_total": TRIALS_PER_CUE}
    return None


def cue_stats(frames, session, cue: str, threshold: float,
              alignment: dict) -> dict:
    """All Day 4 statistics for one cue, plus a plain-English verdict.

    Verdict labels, decided by the numbers, not by hand:
    - "pass": detected >= 4 and zero false positives at the threshold.
    - "salvageable_holds": no false positives, but detection is 0/5
      with a Wilson upper bound below 0.8. The cue fires cleanly; the
      holds were too small or too brief. A re-run with longer, deeper
      holds (or a shorter persistence rule) can rescue it.
    - "metric_wrong": false positives at the operating threshold, and
      no threshold up to 30 deg gives both >= 4 detections and zero
      false positives. The angle alone cannot tell the signal apart
      from ordinary desk posture; a different signal is needed.
    - "inconclusive": none of the above fits cleanly; more data needed.
    """
    curve = threshold_curve(frames, session, cue, alignment)
    op = next(r for r in curve if r["threshold"] == threshold)
    # Longest run at the LOWEST tested threshold: the most generous
    # reading of "how close did a hold come". The operating threshold
    # (8 deg for head) shows 0.0 s, which hides the real near-miss.
    low_row = curve[0]
    longest_run_lowest_threshold_s = low_row["longest_run_s"]
    # The persistence curve is also read at the lowest threshold: at
    # the operating threshold the head cue never fires at all, so the
    # curve would be flat zeros and say nothing.
    persist = persistence_curve(frames, session, cue,
                                low_row["threshold"], alignment)
    zfp = zero_fp_threshold(frames, session, cue, alignment)
    work_s = work_seconds(frames)

    if op["detected"] >= PASS_DETECTED and op["false_positives"] == 0:
        label = "pass"
        text = (f"Passes at {threshold} deg: {op['detected']}/5 holds "
                f"detected, zero false positives in {work_s / 60:.1f} "
                f"minutes of ordinary work.")
    elif op["false_positives"] == 0 and op["ci_high"] < 0.8:
        short = [r for r in persist if r["persist_s"] < PERSIST_S]
        best = max(short, key=lambda r: r["detected"])
        text = (f"No false positives at any tested threshold, but "
                f"0/5 holds detected, and the 95% Wilson interval tops "
                f"out at {op['ci_high']:.0%} (the gate needs 80%). The "
                f"longest hold above {low_row['threshold']} deg (the "
                f"lowest tested threshold) lasted only "
                f"{longest_run_lowest_threshold_s} s against the 5 s "
                f"rule. This is "
                f"not a threshold problem: at a {best['persist_s']} s "
                f"persistence rule {best['detected']}/5 holds would "
                f"have fired, so the tilts were real but far too "
                f"brief. Salvageable with a re-run of longer, deeper "
                f"holds.")
        label = "salvageable_holds"
    elif op["false_positives"] > 0 and (zfp is None
                                        or zfp["detected"] < PASS_DETECTED):
        if zfp is None:
            zfp_text = (f"even a {SCAN_MAX_DEG} deg threshold still "
                        f"fires during ordinary work")
        else:
            zfp_text = (f"the lowest threshold with zero false "
                        f"positives is {zfp['threshold']} deg, where "
                        f"only {zfp['detected']}/5 holds are detected")
        text = (f"False positives at every tested threshold "
                f"({op['false_positives']} at {threshold} deg, "
                f"about {op['fp_per_hour']:.0f} per hour of desk work), "
                f"while detection stays 0/5. And {zfp_text}. No knob "
                f"setting passes both criteria, so the angle itself "
                f"cannot separate a real tilt from leaning at a desk. "
                f"The cue needs a different signal, not a different "
                f"threshold.")
        label = "metric_wrong"
    else:
        label = "inconclusive"
        text = ("The numbers do not decide this cue cleanly. More data "
                "is needed, ideally a re-run with the full protocol.")
    return {
        "cue": cue, "threshold": threshold,
        "detected": op["detected"], "trials_total": TRIALS_PER_CUE,
        "detection_rate": op["detection_rate"],
        "ci_low": op["ci_low"], "ci_high": op["ci_high"],
        "false_positives": op["false_positives"],
        "fp_per_hour": op["fp_per_hour"],
        "fp_per_hour_low": op["fp_per_hour_low"],
        "fp_per_hour_high": op["fp_per_hour_high"],
        "longest_run_s": op["longest_run_s"],
        "longest_run_lowest_threshold_s": longest_run_lowest_threshold_s,
        "work_seconds": round(work_s, 1),
        "threshold_curve": curve,
        "persistence_curve": persist,
        "persistence_threshold": low_row["threshold"],
        "zero_fp_threshold": zfp,
        "verdict": label,
        "verdict_text": text,
    }


def statistical_verdict(frames, session) -> dict:
    """Day 4 verdict for a session: Day 3 verdict plus statistics.

    Reuses the Day 3 verdict (alignments computed once) and adds, per
    cue, the Wilson interval, FP rate, threshold and persistence
    curves, the zero-FP threshold, and the plain-English verdict.
    """
    base = verdict(frames, session)
    cues: dict[str, dict] = {}
    for cue in ("head", "shoulder"):
        c = base["cues"][cue]
        cues[cue] = cue_stats(frames, session, cue, c["threshold"],
                              c["alignment"])
    return {"cues": cues, "loss_test": base["loss_test"],
            "work_seconds": work_seconds(frames)}
