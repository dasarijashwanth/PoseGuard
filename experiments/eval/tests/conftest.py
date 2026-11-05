"""Shared fixtures: a small synthetic validation export with known answers.

The fixture session: baseline head 0 deg, shoulder 170 deg, thresholds
head 8 / shoulder 6. The head trials contain one clean 6 s hold at +10
degrees (trial 1, detected) and four flat trials (not detected). The
work sample contains one 6 s shoulder excursion at +10 degrees (one
false positive). The loss step has tracking == 0 throughout, no firings.
"""

import textwrap
from datetime import datetime, timedelta

import pytest

from experiments.eval.loader import load_session


def _frame_rows():
    """(elapsed, step, head, shoulder, cue, tracking) rows, 0.1 s apart."""
    rows = []
    t = 0.0

    def add(n, step, head, shoulder, cue="", tracking=1):
        nonlocal t
        for _ in range(n):
            rows.append((round(t, 1), step, head, shoulder, cue, tracking))
            t = round(t + 0.1, 1)

    add(30, "calib", 0.0, 170.0)                    # 0.0-2.9 s
    # head trials: ready 3 s, hold 6 s, rest 3 s, x5 (12 s each)
    add(30, "head", 0.0, 170.0)                     # ready
    add(60, "head", 10.0, 170.0)                    # hold 1: tilted, detected
    add(30, "head", 0.0, 170.0)                     # rest
    add(90, "head", 0.0, 170.0)                     # trials 2-4 flat
    add(30, "head", 0.0, 170.0)
    add(60, "head", 0.0, 170.0)
    add(30, "head", 0.0, 170.0)
    add(90, "head", 0.0, 170.0)
    add(30, "head", 0.0, 170.0)
    add(60, "head", 0.0, 170.0)
    add(30, "head", 0.0, 170.0)
    add(60, "head", 0.0, 170.0)                     # trial 5 flat
    # shoulder trials: all flat
    add(600, "shoulder", 0.0, 170.0)
    # work: flat, then one 6 s shoulder excursion, then flat
    add(300, "work", 0.0, 170.0)
    add(60, "work", 0.0, 180.0)                     # false positive
    add(300, "work", 0.0, 170.0)
    # loss: no tracking at all
    add(100, "loss", 0.0, 170.0, tracking=0)
    # mark exactly one excursion row as the experiment's recorded firing
    for j, r in enumerate(rows):
        if r[1] == "work" and r[3] == 180.0:
            rows[j] = (r[0], r[1], r[2], r[3], "shoulder", r[5])
            break
    return rows


def build_csv(tmp_path, rows=None, meta_overrides=None):
    meta = {
        "Exported": "2026-11-05T00:00:00.000Z",
        "Engine": "Test Landmarker",
        "Baseline head (deg)": "0",
        "Baseline shoulder (deg)": "170",
        "Threshold head (deg)": "8",
        "Threshold shoulder (deg)": "6",
        "False detections (work sample)": "1",
        "Landmark-loss test": "PASS",
    }
    meta.update(meta_overrides or {})
    lines = ["PoseGuard validation experiment - session export"]
    lines += [f"{k},{v}" for k, v in meta.items()]
    lines += ["",
              "timestamp,elapsed_s,step,head_tilt,shoulder_tilt,"
              "cue_detected,tracking,head_deviation,shoulder_deviation"]
    for (el, step, head, sh, cue, tr) in (rows or _frame_rows()):
        ts = (datetime(2026, 11, 5) + timedelta(seconds=float(el))).strftime(
            "%Y-%m-%dT%H:%M:%S.000Z")
        lines.append(f"{ts},{el},{step},{head},{sh},{cue},{tr},"
                     f"{head - 0},{sh - 170}")
    # trial summary: head trial 1 detected, rest not; shoulder none detected
    lines += ["",
              "Per-trial summary (angles are deviation from baseline, deg)",
              "cue,trial,detected,samples,head_mean,head_std,"
              "shoulder_mean,shoulder_std",
              "head,1,yes,60,10.0,0.0,0.0,0.0",
              "head,2,no,60,0.0,0.0,0.0,0.0",
              "head,3,no,60,0.0,0.0,0.0,0.0",
              "head,4,no,60,0.0,0.0,0.0,0.0",
              "head,5,no,60,0.0,0.0,0.0,0.0",
              "shoulder,1,no,60,0.0,0.0,0.0,0.0",
              "shoulder,2,no,60,0.0,0.0,0.0,0.0",
              "shoulder,3,no,60,0.0,0.0,0.0,0.0",
              "shoulder,4,no,60,0.0,0.0,0.0,0.0",
              "shoulder,5,no,60,0.0,0.0,0.0,0.0"]
    path = tmp_path / "session.csv"
    path.write_text("\n".join(lines), encoding="utf-8")
    return str(path)


@pytest.fixture
def session_csv(tmp_path):
    return build_csv(tmp_path)


@pytest.fixture
def session(session_csv):
    return load_session(session_csv)


@pytest.fixture
def frames(session):
    from experiments.eval.analysis import add_deviations
    return add_deviations(session)
