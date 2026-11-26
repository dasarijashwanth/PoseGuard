"""Day 25 tests: the capture-session launcher.

Everything here is fast and camera-free: the HTTP test talks to
127.0.0.1 on an ephemeral port, the checklist tests parse the
real CAPTURE_CHECKLIST.md, the drop-dir tests use tmp folders,
the wait tests inject a fake clock, and the pipeline handoff is
mocked (the real Day 24 run is already covered by test_day24).
"""

from __future__ import annotations

import os
import urllib.error
import urllib.request
from pathlib import Path

import pytest

import experiments.model.day25 as day25
from experiments.model.day25 import (
    CAPTURE_HTML,
    checklist_steps,
    fallback_command,
    main,
    run_pipeline,
    scan_drop_dir,
    session_timeline,
    start_server,
    wait_for_capture,
)


def _get(url: str):
    with urllib.request.urlopen(url, timeout=5) as resp:
        return resp.status, resp.headers.get_content_type(), \
            resp.read().decode("utf-8")


def test_handler_serves_capture_page():
    server, _thread, url = start_server(port=0)
    try:
        status, ctype, body = _get(url)
        assert status == 200
        assert ctype == "text/html"
        assert "PoseGuard Run 2: camera re-run capture" in body
        assert body == CAPTURE_HTML.read_text(encoding="utf-8")
        root = url.rsplit("/", 1)[0] + "/"
        status, _ctype, body = _get(root)
        assert status == 200
        assert "PoseGuard Run 2" in body
        with pytest.raises(urllib.error.HTTPError) as exc:
            _get(root + "not-the-capture-page")
        assert exc.value.code == 404
    finally:
        server.shutdown()
        server.server_close()


def test_checklist_steps_order_and_content():
    steps = checklist_steps()
    sections = []
    for section, _text in steps:
        if section not in sections:
            sections.append(section)
    assert sections[0] == "Before you start"
    assert sections[1].startswith("During the session")
    assert sections[2] == "After"
    counts = {s: sum(1 for sec, _ in steps if sec == s)
              for s in sections}
    assert counts["Before you start"] == 4
    assert counts[sections[1]] == 5
    assert counts["After"] == 2
    joined = " ".join(text for _s, text in steps)
    assert "hips" in joined
    assert "head 15 deg" in joined
    assert "shoulder 8" in joined
    assert "Download CSV" in joined
    assert "Download manifest" in joined
    # The session steps come in desk order: calibration before
    # the holds, the holds before the work sample.
    during = [t for s, t in steps if s == sections[1]]
    assert "Calibration" in during[1]
    assert "head trials" in during[2]
    assert "Work sample" in during[3]


def test_session_timeline_order_and_total():
    timeline = session_timeline()
    labels = [label for label, _s in timeline]
    assert labels[0].startswith("Calibration")
    assert labels[1].startswith("Head holds x5")
    assert labels[2].startswith("Shoulder holds x5")
    assert labels[3].startswith("Work sample")
    assert labels[4].startswith("Landmark-loss test")
    assert "15 deg" in labels[1]
    assert "8 deg" in labels[2]
    # 5 + 5x(5+10+5) + 5x(5+10+5) + 300 + 10 = 515 s, the
    # checklist's "about 8.6 minutes".
    assert sum(s for _l, s in timeline) == 515.0


def _touch(path: Path, mtime: float) -> Path:
    path.write_text("x", encoding="utf-8")
    os.utime(path, (mtime, mtime))
    return path


def test_scan_drop_dir_states(tmp_path):
    assert scan_drop_dir(str(tmp_path))["state"] == "empty"
    assert scan_drop_dir(str(tmp_path / "missing"))["state"] == "empty"
    _touch(tmp_path / "poseguard-rerun-2026-11-26.csv", 1000)
    scan = scan_drop_dir(str(tmp_path))
    assert scan["state"] == "csv_only"
    assert scan["manifest"] is None
    # The capture tool's manifest has a different stem, so the
    # pairing falls back to the newest .json (Day 24's rule).
    manifest = _touch(
        tmp_path / "poseguard-rerun-manifest-2026-11-26.json", 2000)
    scan = scan_drop_dir(str(tmp_path))
    assert scan["state"] == "ready"
    assert scan["manifest"] == str(manifest)


def test_scan_drop_dir_manifest_only(tmp_path):
    _touch(tmp_path / "poseguard-rerun-manifest.json", 1000)
    scan = scan_drop_dir(str(tmp_path))
    assert scan["state"] == "manifest_only"
    assert scan["csv"] is None


def test_wait_for_capture_finds_files_after_polls(tmp_path):
    calls = {"n": 0}

    def fake_sleep(_seconds):
        calls["n"] += 1
        if calls["n"] == 2:
            _touch(tmp_path / "capture.csv", 1000)
            _touch(tmp_path / "capture.json", 2000)

    found = wait_for_capture(str(tmp_path), timeout_s=60,
                             poll_s=0.01, sleep=fake_sleep)
    assert found is not None
    csv_path, manifest_path = found
    assert csv_path == str(tmp_path / "capture.csv")
    assert manifest_path == str(tmp_path / "capture.json")


def test_wait_for_capture_timeout(tmp_path, capsys):
    clock = [0.0]

    def fake_now():
        return clock[0]

    def fake_sleep(seconds):
        clock[0] += seconds

    found = wait_for_capture(str(tmp_path), timeout_s=5,
                             poll_s=2.0, sleep=fake_sleep,
                             now=fake_now)
    assert found is None
    assert "No capture files yet" in capsys.readouterr().out


def _fake_archive():
    def variant(threshold, detected, fp, f1, passed):
        return {"best": {"threshold": threshold,
                         "detected": detected, "trials_total": 5,
                         "false_positives": fp, "f1": f1,
                         "pass": passed}}

    cues = {}
    for cue in ("head", "shoulder"):
        cues[cue] = {"variants": {
            "sum": variant(0.5, 5, 0, 1.0, True),
            "iso": variant(0.5, 5, 0, 1.0, True),
            "and_control": variant(1.0, 0, 0, 0.0, False),
        }}
    return {
        "run_dir": "/tmp/runs/2026-11-26-0900",
        "day16": {"verdict": "Head passes the gate: fake. "
                             "Shoulder passes the gate: fake."},
        "combined": {"cues": cues},
    }


def test_run_pipeline_handoff_mocked(monkeypatch, capsys):
    seen = {}

    def fake_run_day24(csv_path, manifest_path, source=None,
                       out_dir=None):
        seen.update(csv=csv_path, manifest=manifest_path,
                    source=source, out_dir=out_dir)
        return _fake_archive()

    monkeypatch.setattr(day25, "run_day24", fake_run_day24)
    archive = run_pipeline("cap.csv", "cap.json",
                           source="test source", out_dir="runs-x")
    assert seen == {"csv": "cap.csv", "manifest": "cap.json",
                    "source": "test source", "out_dir": "runs-x"}
    assert archive["run_dir"] == "/tmp/runs/2026-11-26-0900"
    out = capsys.readouterr().out
    assert "Archive: /tmp/runs/2026-11-26-0900" in out
    assert "Day 16 verdict: Head passes the gate: fake." in out
    assert ("Combined head: sum at threshold 0.5, 5/5 trials, "
            "0 false positives, F1 1.000: passes the gate." in out)


def test_main_no_wait_ready_runs_pipeline(tmp_path, monkeypatch):
    _touch(tmp_path / "capture.csv", 1000)
    _touch(tmp_path / "capture.json", 2000)
    seen = {}

    def fake_run_day24(csv_path, manifest_path, source=None,
                       out_dir=None):
        seen.update(csv=csv_path, manifest=manifest_path)
        return _fake_archive()

    monkeypatch.setattr(day25, "run_day24", fake_run_day24)
    code = main(["--no-wait", "--drop-dir", str(tmp_path)])
    assert code == 0
    assert seen["csv"] == str(tmp_path / "capture.csv")
    assert seen["manifest"] == str(tmp_path / "capture.json")


def test_main_no_wait_empty_prints_fallback(tmp_path, capsys):
    code = main(["--no-wait", "--drop-dir", str(tmp_path)])
    assert code == 2
    out = capsys.readouterr().out
    assert "No capture files yet" in out
    assert fallback_command(str(tmp_path)) in out


def test_fallback_command_exact():
    assert fallback_command("/some/folder") == (
        "python -m experiments.model.day24 --drop-dir /some/folder")
