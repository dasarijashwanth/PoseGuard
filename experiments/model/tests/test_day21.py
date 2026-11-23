"""Day 21 tests: the Day 16 harness must fail gracefully on messy
inputs, never with a raw traceback.

Every case here broke or lied during the Day 21 stress probe, or is
the regression test for a fix it forced. The probe itself was a
scratch script; these tests are the permanent record.
"""

import json
import math
from pathlib import Path

import pandas as pd
import pytest

from experiments.eval.analysis import align_trials, ema_smooth
from experiments.eval.loader import SessionData
from experiments.eval.rerun import validate_rerun_csv
from experiments.eval.tests.test_rerun import build_rerun_csv
from experiments.model.day16 import main, run_day16, validate_capture


# ----------------------------------------------------------------------
# variant builders
# ----------------------------------------------------------------------

def _rows(base_path):
    lines = Path(base_path).read_text(encoding="utf-8").splitlines()
    hdr = next(i for i, l in enumerate(lines) if l.startswith("timestamp,"))
    cols = {c: i for i, c in enumerate(lines[hdr].split(","))}
    return lines, hdr, cols


def _frame_rows(lines, hdr):
    rows = lines[hdr + 1:]
    n = next((i for i, l in enumerate(rows) if l == ""), len(rows))
    return rows, n


def _edit_rows(lines, hdr, fn):
    rows, n = _frame_rows(lines, hdr)
    for i in range(n):
        parts = rows[i].split(",")
        fn(parts)
        rows[i] = ",".join(parts)
    lines[hdr + 1:] = rows


def _variant(base_path, tmp_path, name, mutate):
    lines, hdr, cols = _rows(base_path)
    mutate(lines, hdr, cols)
    path = tmp_path / name
    path.write_text("\n".join(lines), encoding="utf-8")
    return str(path)


def _untrack(parts, cols, step=None, hold=None):
    if step is not None and parts[cols["step"]] != step:
        return
    if hold is not None and parts[cols["hold_id"]] != str(hold):
        return
    if parts[cols["phase"]] == "hold":
        parts[cols["tracking"]] = "0"
        for col in ("head_tilt", "shoulder_tilt",
                    "head_deviation", "shoulder_deviation"):
            parts[cols[col]] = ""


# ----------------------------------------------------------------------
# unreadable or unparseable files: clear error, no traceback
# ----------------------------------------------------------------------

def test_missing_csv_rejected_with_clear_error(tmp_path):
    report = validate_rerun_csv(str(tmp_path / "nope.csv"))
    assert report["ok"] is False
    assert "cannot read capture file" in report["errors"][0]
    assert "No such file" in report["errors"][0]


def test_directory_path_rejected_with_clear_error(tmp_path):
    report = validate_rerun_csv(str(tmp_path))
    assert report["ok"] is False
    assert "cannot read capture file" in report["errors"][0]
    assert "Is a directory" in report["errors"][0]


def test_empty_file_rejected(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("", encoding="utf-8")
    report = validate_rerun_csv(str(path))
    assert report["ok"] is False
    assert report["errors"] == ["empty file"]


def test_binary_file_rejected(tmp_path):
    path = tmp_path / "binary.csv"
    path.write_bytes(bytes(range(256)) * 4)
    report = validate_rerun_csv(str(path))
    assert report["ok"] is False
    assert report["errors"]


def test_ragged_row_rejected_with_field_count(tmp_path):
    base = build_rerun_csv(tmp_path)

    def ragged(lines, hdr, cols):
        lines[hdr + 1] = lines[hdr + 1] + ",9"

    report = validate_rerun_csv(_variant(base, tmp_path, "ragged.csv",
                                         ragged))
    assert report["ok"] is False
    assert "fields" in report["errors"][0]
    assert "expected 111" in report["errors"][0]


# ----------------------------------------------------------------------
# label and value errors: named, not a raw int() or sorted() crash
# ----------------------------------------------------------------------

def test_hold_id_text_rejected(tmp_path):
    base = build_rerun_csv(tmp_path)

    def text_hold(lines, hdr, cols):
        _edit_rows(lines, hdr,
                   lambda p: p.__setitem__(cols["hold_id"], "abc"))

    report = validate_rerun_csv(_variant(base, tmp_path, "abchold.csv",
                                         text_hold))
    assert report["ok"] is False
    assert any("hold_id values must be integers 1-5" in e
               for e in report["errors"])


def test_hold_id_fraction_rejected_not_truncated(tmp_path):
    base = build_rerun_csv(tmp_path)

    def frac_hold(lines, hdr, cols):
        _edit_rows(lines, hdr,
                   lambda p: p.__setitem__(cols["hold_id"], "1.5"))

    report = validate_rerun_csv(_variant(base, tmp_path, "frachold.csv",
                                         frac_hold))
    assert report["ok"] is False
    assert any("hold_id values must be integers 1-5, got: ['1.5']" in e
               for e in report["errors"])


def test_out_of_order_elapsed_rejected(tmp_path):
    base = build_rerun_csv(tmp_path)

    def swap(lines, hdr, cols):
        rows, n = _frame_rows(lines, hdr)
        rows[:200] = rows[100:200] + rows[:100]
        lines[hdr + 1:] = rows

    report = validate_rerun_csv(_variant(base, tmp_path, "ooo.csv", swap))
    assert report["ok"] is False
    assert any("not monotonic increasing" in e for e in report["errors"])


def test_elapsed_s_inf_rejected(tmp_path):
    base = build_rerun_csv(tmp_path)

    def inf_time(lines, hdr, cols):
        rows, n = _frame_rows(lines, hdr)
        parts = rows[1000].split(",")
        parts[cols["elapsed_s"]] = "inf"
        rows[1000] = ",".join(parts)
        lines[hdr + 1:] = rows

    report = validate_rerun_csv(_variant(base, tmp_path, "inftime.csv",
                                         inf_time))
    assert report["ok"] is False
    assert any("missing or non-finite" in e for e in report["errors"])


# ----------------------------------------------------------------------
# data quality on tracked frames: error for corrupt, warning for missing
# ----------------------------------------------------------------------

def test_adjacent_duplicate_rows_warn(tmp_path):
    base = build_rerun_csv(tmp_path)

    def dup(lines, hdr, cols):
        rows, n = _frame_rows(lines, hdr)
        rows[1001:1001] = [rows[1000]]
        lines[hdr + 1:] = rows

    report = validate_rerun_csv(_variant(base, tmp_path, "dups.csv", dup))
    assert report["ok"] is True
    assert any("duplicate" in w for w in report["warnings"])


def test_inf_tilt_on_one_tracked_frame_rejected(tmp_path):
    base = build_rerun_csv(tmp_path)

    def inf_tilt(lines, hdr, cols):
        def set_inf(p):
            if (p[cols["step"]] == "head" and p[cols["tracking"]] == "1"
                    and p[cols["elapsed_s"]] == "100.0"):
                p[cols["head_tilt"]] = "inf"
        _edit_rows(lines, hdr, set_inf)

    report = validate_rerun_csv(_variant(base, tmp_path, "inftilt.csv",
                                         inf_tilt))
    assert report["ok"] is False
    assert any("head_tilt has 1 infinite values" in e
               for e in report["errors"])


def test_missing_tilt_on_tracked_frames_warns(tmp_path):
    base = build_rerun_csv(tmp_path)

    def blank_tilt(lines, hdr, cols):
        def blank(p):
            if (p[cols["step"]] == "head" and p[cols["tracking"]] == "1"
                    and float(p[cols["elapsed_s"]]) < 6.0):
                p[cols["head_tilt"]] = ""
                p[cols["head_deviation"]] = ""
        _edit_rows(lines, hdr, blank)

    report = validate_rerun_csv(_variant(base, tmp_path, "blanktilt.csv",
                                         blank_tilt))
    assert report["ok"] is True
    assert any("missing values on tracked frames" in w
               for w in report["warnings"])


def test_nan_landmark_on_tracked_frames_warns(tmp_path):
    base = build_rerun_csv(tmp_path)

    def nan_lm(lines, hdr, cols):
        def blank(p):
            if p[cols["tracking"]] == "1":
                p[cols["lm_5_x"]] = "nan"
        _edit_rows(lines, hdr, blank)

    report = validate_rerun_csv(_variant(base, tmp_path, "nanlm.csv",
                                         nan_lm))
    assert report["ok"] is True
    assert any("landmark" in w for w in report["warnings"])


def test_nan_landmark_on_untracked_frames_only_is_silent(tmp_path):
    base = build_rerun_csv(tmp_path)

    def nan_lm(lines, hdr, cols):
        def blank(p):
            if p[cols["tracking"]] == "0":
                p[cols["lm_5_x"]] = "nan"
        _edit_rows(lines, hdr, blank)

    report = validate_rerun_csv(_variant(base, tmp_path, "nanlm2.csv",
                                         nan_lm))
    assert report["ok"] is True
    assert not any("landmark" in w for w in report["warnings"])


# ----------------------------------------------------------------------
# manifest handling
# ----------------------------------------------------------------------

def test_validate_capture_bad_csv_raises_aggregated_error(tmp_path):
    base = build_rerun_csv(tmp_path)
    manifest = _manifest(tmp_path)
    bad = tmp_path / "bad.csv"
    bad.write_text("not a capture file\n", encoding="utf-8")
    with pytest.raises(ValueError, match="cannot be used"):
        validate_capture(str(bad), str(manifest))


def _manifest(tmp_path, frames=5150):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({
        "protocol": "Run 2 re-run capture", "frames": frames,
        "baseline_head_deg": 0, "baseline_shoulder_deg": 170,
        "holds": [{"cue": c, "hold": i} for c in ("head", "shoulder")
                  for i in range(1, 6)],
    }), encoding="utf-8")
    return path


def test_manifest_frames_as_string_accepted(tmp_path):
    base = build_rerun_csv(tmp_path)
    data = {"protocol": "Run 2 re-run capture", "frames": "5150",
            "baseline_head_deg": 0, "baseline_shoulder_deg": 170,
            "holds": [{"cue": c, "hold": i} for c in ("head", "shoulder")
                      for i in range(1, 6)]}
    string_manifest = tmp_path / "manifest-str.json"
    string_manifest.write_text(json.dumps(data), encoding="utf-8")
    report = validate_capture(base, str(string_manifest))
    assert report["ok"] is True


# ----------------------------------------------------------------------
# analysis fixes: inf must not poison the EMA, NaN must not empty trials
# ----------------------------------------------------------------------

def test_ema_smooth_ignores_inf():
    out = ema_smooth(pd.Series([10.0, float("inf"), 12.0]), alpha=0.5)
    assert out[0] == 10.0
    assert math.isnan(out[1])
    # the run resumes from the last good value, not from NaN
    assert out[2] == 11.0


def test_align_trials_all_nan_step_returns_five_trials():
    frames = pd.DataFrame({
        "step": ["head"] * 601,
        "elapsed_s": [i / 10.0 for i in range(601)],
        "h_dev": [float("nan")] * 601,
    })
    session = SessionData(
        path="test", meta={}, frames=frames,
        trial_summary=[
            {"cue": "head", "trial": k + 1, "samples": 100,
             "head_mean": 15.0, "head_std": 1.0,
             "shoulder_mean": 0.0, "shoulder_std": 0.0}
            for k in range(5)])
    out = align_trials(frames, session, "head",
                       ready_s=5.0, hold_s=10.0, rest_s=5.0)
    assert len(out["trials"]) == 5
    assert math.isfinite(out["fit_error"])
    assert all(t["samples"] == 0 for t in out["trials"])


# ----------------------------------------------------------------------
# end to end: run_day16 fails gracefully, and stays strict-JSON valid
# ----------------------------------------------------------------------

def test_run_day16_untracked_head_step_fails_gracefully(tmp_path):
    base = build_rerun_csv(tmp_path)

    def untrack_head(lines, hdr, cols):
        def untrack(p):
            if p[cols["step"]] == "head":
                p[cols["tracking"]] = "0"
                for col in ("head_tilt", "shoulder_tilt",
                            "head_deviation", "shoulder_deviation"):
                    p[cols[col]] = ""
        _edit_rows(lines, hdr, untrack)

    path = _variant(base, tmp_path, "untracked_head.csv", untrack_head)
    with pytest.raises(ValueError,
                       match="no usable tracked frames for cue head"):
        run_day16(path)


def test_run_day16_one_untracked_hold_is_strict_json(tmp_path):
    base = build_rerun_csv(tmp_path)

    def untrack_hold3(lines, hdr, cols):
        _edit_rows(lines, hdr,
                   lambda p: _untrack(p, cols, step="shoulder", hold=3))

    path = _variant(base, tmp_path, "untracked_hold3.csv", untrack_hold3)
    result = run_day16(path)
    # an empty hold window used to leak NaN into the result, which is
    # not valid strict JSON
    json.dumps(result, allow_nan=False)


def test_run_day16_sparse_head_fails_gracefully(tmp_path):
    base = build_rerun_csv(tmp_path)
    seen = {}

    def sparse(lines, hdr, cols):
        def keep_five(p):
            if p[cols["step"]] == "head" and p[cols["phase"]] == "hold":
                key = p[cols["hold_id"]]
                seen[key] = seen.get(key, 0) + 1
                if seen[key] > 5:
                    _untrack(p, cols)
        _edit_rows(lines, hdr, keep_five)

    path = _variant(base, tmp_path, "sparse_head.csv", sparse)
    with pytest.raises(ValueError, match="no tilted windows for cue"):
        run_day16(path)


def test_main_missing_csv_returns_2(tmp_path):
    assert main(["--csv", str(tmp_path / "nope.csv")]) == 2
