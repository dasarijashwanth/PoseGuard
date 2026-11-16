"""Day 14 tests: Run 2 protocol, capture CSV schema, and validator.

The protocol module is the single source of truth for the re-run
timings and targets; the capture tool (experiments/rerun-capture.html)
implements the same numbers in JavaScript, and one test here reads
the HTML and pins the two together so they cannot drift apart.
"""

from datetime import datetime, timedelta
from pathlib import Path

import pytest

from experiments.eval.loader import REQUIRED_COLUMNS, load_session
from experiments.eval.rerun import (
    MIN_GAP_S,
    RUN2_META_KEYS,
    tracking_gaps,
    validate_rerun_csv,
)
from experiments.eval.rerun_protocol import (
    CALIB_S,
    DETECT_PERSIST_S,
    HOLD_S,
    LANDMARK_IDS,
    LOSS_S,
    READY_S,
    REST_S,
    TARGET_HEAD_DEG,
    TARGET_SHOULDER_DEG,
    TRIALS_PER_CUE,
    WORK_S,
    frame_columns,
    landmark_columns,
    session_duration_s,
    session_timeline,
    target_deg,
    trial_block_timeline,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
HTML_PATH = REPO_ROOT / "experiments" / "rerun-capture.html"


# --- protocol ---------------------------------------------------------------

def test_protocol_timings():
    assert CALIB_S == 5.0
    assert READY_S == 5.0
    assert HOLD_S == 10.0
    assert REST_S == 5.0
    assert WORK_S == 300.0
    assert LOSS_S == 10.0
    assert TRIALS_PER_CUE == 5
    assert DETECT_PERSIST_S == 5.0
    assert session_duration_s() == 515.0


def test_protocol_targets():
    assert target_deg("head") == 15.0
    assert target_deg("shoulder") == 8.0
    assert TARGET_HEAD_DEG == 15.0
    assert TARGET_SHOULDER_DEG == 8.0
    with pytest.raises(ValueError):
        target_deg("elbow")


def test_trial_block_timeline():
    block = trial_block_timeline("shoulder")
    assert len(block) == 15  # 5 trials x ready/hold/rest
    assert [e["phase"] for e in block] == ["ready", "hold", "rest"] * 5
    assert [e["hold_id"] for e in block] == [i for i in range(1, 6)
                                             for _ in range(3)]
    assert block[0]["start_s"] == 0.0
    assert block[-1]["end_s"] == 100.0
    hold = block[1]
    assert hold["phase"] == "hold" and hold["duration_s"] == 10.0
    assert hold["start_s"] == 5.0 and hold["end_s"] == 15.0


def test_session_timeline():
    timeline = session_timeline()
    assert timeline[0]["phase"] == "calib"
    assert timeline[0]["end_s"] == 5.0
    assert timeline[-1]["phase"] == "loss"
    assert timeline[-1]["end_s"] == 515.0
    phases = [e["phase"] for e in timeline]
    assert phases.count("hold") == 10
    assert phases.count("calm") == 1
    # phases are contiguous: each entry starts where the last ended
    for prev, cur in zip(timeline, timeline[1:]):
        assert cur["start_s"] == prev["end_s"]


def test_landmark_columns():
    cols = landmark_columns()
    assert len(cols) == 99  # 33 landmarks x, y, visibility
    assert cols[0] == "lm_0_x"
    for i in (7, 8, 11, 12, 23, 24):
        assert f"lm_{i}_x" in cols and f"lm_{i}_y" in cols
        assert f"lm_{i}_v" in cols
    assert set(LANDMARK_IDS) == set(range(33))


def test_frame_columns_superset_of_run1():
    cols = frame_columns()
    assert len(cols) == 9 + 3 + 99
    for c in REQUIRED_COLUMNS:
        assert c in cols
    assert cols[9:12] == ["cue", "hold_id", "phase"]
    assert "lm_23_x" in cols and "lm_24_y" in cols


# --- synthetic Run 2 CSV -----------------------------------------------------

def _landmark_values(tracked=True):
    """Plausible landmark values: hips and shoulders in frame."""
    vals = []
    for i in range(33):
        if not tracked:
            vals.extend(["", "", ""])
        else:
            x, y = 0.5, 0.5
            if i == 11:
                x, y = 0.40, 0.50
            elif i == 12:
                x, y = 0.60, 0.50
            elif i == 23:
                x, y = 0.45, 0.80
            elif i == 24:
                x, y = 0.55, 0.80
            elif i in (7, 8):
                x, y = (0.45, 0.30) if i == 7 else (0.55, 0.30)
            vals.extend([f"{x:.3f}", f"{y:.3f}", "0.9"])
    return vals


def build_rerun_csv(tmp_path, drop_columns=(), drop_meta=(),
                    gap_hold=None, weak_hold=None, work_s=300.0):
    """Write a synthetic Run 2 capture CSV.

    Full session at 10 fps: calib 5 s, head trials (ready 5 / hold 10
    at +16 deg / rest 5) x5, shoulder trials (hold at +9 deg) x5,
    work `work_s` seconds flat, loss 10 s with no tracking.
    `gap_hold` is ("shoulder", 3) style: 2 s of tracking loss in the
    middle of that hold. `weak_hold` is ("shoulder", 2): that hold
    only reaches +3 deg, below its 8 deg target.
    """
    meta = {
        "Exported": "2026-11-16T00:00:00.000Z",
        "Engine": "Test Landmarker",
        "Protocol": "Run 2 re-run capture",
        "Baseline head (deg)": "0",
        "Baseline shoulder (deg)": "170",
        "Threshold head (deg)": "15",
        "Threshold shoulder (deg)": "8",
        "Target head (deg)": "15",
        "Target shoulder (deg)": "8",
        "Hold duration (s)": "10",
        "False detections (work sample)": "0",
        "Landmark-loss test": "PASS",
    }
    for key in drop_meta:
        meta.pop(key, None)
    header = frame_columns()
    header = [c for c in header if c not in drop_columns]
    rows = []
    t = 0.0

    def add(n, step, head, sh, cue="", hold_id="", phase="",
            tracking=1):
        nonlocal t
        for _ in range(n):
            ts = (datetime(2026, 11, 16)
                  + timedelta(seconds=float(t))).strftime(
                      "%Y-%m-%dT%H:%M:%S.000Z")
            tracked = tracking == 1
            vals = {
                "timestamp": ts, "elapsed_s": f"{t:.1f}", "step": step,
                "head_tilt": "" if not tracked else f"{head:.3f}",
                "shoulder_tilt": "" if not tracked else f"{sh:.3f}",
                "cue_detected": "",
                "tracking": str(tracking),
                "head_deviation": "" if not tracked else f"{head:.3f}",
                "shoulder_deviation": "" if not tracked else f"{sh - 170:.3f}",
                "cue": cue,
                "hold_id": "" if hold_id == "" else str(hold_id),
                "phase": phase,
            }
            vals.update(dict(zip(landmark_columns(),
                                 _landmark_values(tracked))))
            rows.append(",".join(vals[c] for c in header))
            t = round(t + 0.1, 1)

    add(50, "calib", 0.0, 170.0, phase="calib")
    for trial in range(1, 6):
        add(50, "head", 0.0, 170.0, cue="head", hold_id=trial,
            phase="ready")
        add(100, "head", 16.0, 170.0, cue="head", hold_id=trial,
            phase="hold")
        add(50, "head", 0.0, 170.0, cue="head", hold_id=trial,
            phase="rest")
    for trial in range(1, 6):
        add(50, "shoulder", 0.0, 170.0, cue="shoulder", hold_id=trial,
            phase="ready")
        if weak_hold == ("shoulder", trial):
            hold_sh = 173.0  # +3 deg, below the 8 deg target
        else:
            hold_sh = 179.0  # +9 deg, past the target
        if gap_hold == ("shoulder", trial):
            add(40, "shoulder", 0.0, hold_sh, cue="shoulder",
                hold_id=trial, phase="hold")
            add(20, "shoulder", 0.0, 170.0, cue="shoulder",
                hold_id=trial, phase="hold", tracking=0)
            add(40, "shoulder", 0.0, hold_sh, cue="shoulder",
                hold_id=trial, phase="hold")
        else:
            add(100, "shoulder", 0.0, hold_sh, cue="shoulder",
                hold_id=trial, phase="hold")
        add(50, "shoulder", 0.0, 170.0, cue="shoulder", hold_id=trial,
            phase="rest")
    add(int(work_s * 10), "work", 0.0, 170.0, phase="calm")
    add(100, "loss", 0.0, 170.0, phase="loss", tracking=0)
    lines = ["PoseGuard re-run capture - session export"]
    lines += [f"{k},{v}" for k, v in meta.items()]
    lines += ["", ",".join(header)]
    lines += rows
    lines += ["",
              "Per-trial summary (angles are deviation from baseline, deg)",
              "cue,trial,detected,samples,head_mean,head_std,"
              "shoulder_mean,shoulder_std"]
    for trial in range(1, 6):
        lines.append(f"head,{trial},yes,100,16.0,0.0,0.0,0.0")
        lines.append(f"shoulder,{trial},yes,100,0.0,0.0,9.0,0.0")
    path = tmp_path / "rerun.csv"
    path.write_text("\n".join(lines), encoding="utf-8")
    return str(path)


# --- validator ----------------------------------------------------------------

def test_valid_rerun_csv_passes(tmp_path):
    path = build_rerun_csv(tmp_path)
    report = validate_rerun_csv(path)
    assert report["ok"], report["errors"]
    assert report["warnings"] == []
    assert report["frames"] == 5150
    assert len(report["holds"]) == 10
    assert report["work_s"] == pytest.approx(300.0, abs=0.2)
    assert report["gaps"] == []  # the loss-test gap is expected, not listed
    head1 = [h for h in report["holds"]
             if h["cue"] == "head" and h["hold_id"] == 1][0]
    assert head1["frames"] == 100
    assert head1["duration_s"] == 9.9
    assert head1["tracked_fraction"] == 1.0
    assert head1["max_abs_dev_deg"] == 16.0
    assert head1["reached_target"] is True


def test_run2_csv_loads_with_run1_loader(tmp_path):
    path = build_rerun_csv(tmp_path)
    session = load_session(path)
    assert len(session.frames) == 5150
    assert "lm_23_x" in session.frames.columns
    assert "lm_24_y" in session.frames.columns
    assert session.meta["Protocol"] == "Run 2 re-run capture"
    assert set(session.frames["phase"].unique()) == {
        "calib", "ready", "hold", "rest", "calm", "loss"}


def test_missing_landmark_column_fails(tmp_path):
    path = build_rerun_csv(tmp_path, drop_columns=("lm_24_y",))
    report = validate_rerun_csv(path)
    assert not report["ok"]
    assert any("lm_24_y" in e for e in report["errors"])


def test_missing_protocol_meta_fails(tmp_path):
    path = build_rerun_csv(tmp_path, drop_meta=("Protocol",))
    report = validate_rerun_csv(path)
    assert not report["ok"]
    assert any("Protocol" in e for e in report["errors"])


def test_tracking_gap_reported_with_context(tmp_path):
    path = build_rerun_csv(tmp_path, gap_hold=("shoulder", 3))
    gaps = [g for g in tracking_gaps(load_session(path).frames)
            if g["phase"] != "loss"]  # the loss-test gap is expected
    assert len(gaps) == 1
    gap = gaps[0]
    assert gap["duration_s"] == 2.0
    assert gap["phase"] == "hold"
    assert gap["cue"] == "shoulder"
    assert gap["hold_id"] == 3
    report = validate_rerun_csv(path)
    assert report["ok"]  # a gap warns, it does not fail the file
    assert any("tracking lost 2.0 s" in w for w in report["warnings"])
    hold3 = [h for h in report["holds"]
             if h["cue"] == "shoulder" and h["hold_id"] == 3][0]
    assert hold3["tracked_fraction"] == 0.8


def test_hold_below_target_warns(tmp_path):
    path = build_rerun_csv(tmp_path, weak_hold=("shoulder", 2))
    report = validate_rerun_csv(path)
    assert report["ok"]
    assert any("shoulder hold 2 never reached its 8 deg target" in w
               for w in report["warnings"])
    hold2 = [h for h in report["holds"]
             if h["cue"] == "shoulder" and h["hold_id"] == 2][0]
    assert hold2["max_abs_dev_deg"] == 3.0
    assert hold2["reached_target"] is False


def test_short_work_sample_warns(tmp_path):
    path = build_rerun_csv(tmp_path, work_s=100.0)
    report = validate_rerun_csv(path)
    assert report["ok"]
    assert any("work sample is only 100 s" in w for w in report["warnings"])


def test_short_gap_below_threshold_ignored(tmp_path):
    # 0.5 s of loss is tracker flicker, not a gap worth listing.
    frames = load_session(build_rerun_csv(tmp_path)).frames
    frames.loc[1000:1004, "tracking"] = 0
    gaps = [g for g in tracking_gaps(frames, min_gap_s=MIN_GAP_S)
            if g["phase"] != "loss"]  # the loss-test gap is expected
    assert gaps == []


def test_run2_meta_keys_covered():
    assert RUN2_META_KEYS == {
        "Protocol", "Hold duration (s)",
        "Target head (deg)", "Target shoulder (deg)"}


# --- HTML / Python sync ---------------------------------------------------------

def test_capture_tool_matches_protocol():
    html = HTML_PATH.read_text(encoding="utf-8")
    for name, value in (
            ("CALIB_S", "5"), ("READY_S", "5"), ("HOLD_S", "10"),
            ("REST_S", "5"), ("WORK_S", "300"), ("LOSS_S", "10"),
            ("TRIALS_PER_CUE", "5"), ("TARGET_HEAD_DEG", "15"),
            ("TARGET_SHOULDER_DEG", "8")):
        assert f"var {name} = {value};" in html, \
            f"{name} in HTML does not match the Python protocol"
    assert "Run 2 re-run capture" in html
    assert "cue,hold_id,phase" in html
    # Landmark columns are generated by a loop over all 33 landmarks;
    # pin the loop, the count, and the required set (ears, shoulders,
    # hips) instead of 99 literals.
    assert "var LM_COUNT = 33;" in html
    assert "for (var li = 0; li < LM_COUNT; li++)" in html
    assert '"lm_" + li + "_x"' in html
    assert '"lm_" + li + "_y"' in html
    assert '"lm_" + li + "_v"' in html
    assert "var REQUIRED_LM = [7, 8, 11, 12, 23, 24];" in html
    assert "localhost:8000" in html
    assert "HttpListener" in html
