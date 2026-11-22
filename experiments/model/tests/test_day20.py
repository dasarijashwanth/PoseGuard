"""Day 20 tests: MediaPipe actually loading on Streamlit Cloud.

Root cause (reproduced in a fresh venv with the Cloud's missing
system library hidden): mediapipe 1.2.3 imports cv2 at module load
(tasks vision drawing_utils), and the cv2 that wins is
opencv-contrib-python, mediapipe's own dependency, which links
against libGL.so.1. The Cloud image does not ship libGL, so
`import mediapipe` failed there while it worked locally, and the old
app swallowed the error in one try/except around four imports. The
fix is packages.txt at the repo root (libgl1, libglib2.0-0, libegl1,
libgles2), pinned live-check versions, per-module import probing
with a diagnostic in the app, and an actionable error when creating
a landmarker fails (import can succeed while creation fails: the
native library loads lazily, so a missing libEGL only shows up
there).
"""

from __future__ import annotations

import importlib
import re
from pathlib import Path

from demo.env_check import (
    diagnostic_text,
    landmarker_error_hint,
    live_check_status,
    probe_live_deps,
)

HERE = Path(__file__).resolve().parent.parent.parent.parent


def _requirement_lines():
    text = (HERE / "demo" / "requirements.txt").read_text()
    return [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def _failing_importer(fail_on, message):
    """An importer that fails for the named modules, real otherwise."""

    def fake(name):
        if any(name == f or name.startswith(f + ".") for f in fail_on):
            raise ImportError(message)
        return importlib.import_module(name)

    return fake


def test_requirements_pins_mediapipe():
    lines = _requirement_lines()
    assert "mediapipe==1.2.3" in lines


def test_requirements_has_no_plain_opencv_python():
    for line in _requirement_lines():
        assert not (
            line.startswith("opencv-python")
            and not line.startswith("opencv-python-headless")
        ), f"non-headless opencv line would pull libGL in: {line}"


def test_requirements_pins_the_live_check_stack():
    lines = _requirement_lines()
    assert "streamlit-webrtc==0.79.1" in lines
    assert "streamlit==1.67.0" in lines
    assert "opencv-python-headless==5.1.0.95" in lines


def test_packages_txt_lists_the_system_libraries():
    text = (HERE / "packages.txt").read_text()
    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    for package in ("libgl1", "libglib2.0-0", "libegl1", "libgles2"):
        assert package in lines, f"packages.txt is missing {package}"


def test_probe_loads_everything_in_this_environment():
    modules, errors = probe_live_deps()
    assert errors == {}
    assert set(modules) == {"av", "cv2", "mediapipe", "streamlit_webrtc"}
    assert all(m is not None for m in modules.values())
    assert hasattr(modules["mediapipe"], "Image")
    assert hasattr(modules["mediapipe"].tasks.vision, "PoseLandmarker")


def test_probe_records_each_failure_separately():
    modules, errors = probe_live_deps(
        _failing_importer(
            {"cv2"},
            "libGL.so.1: cannot open shared object file: No such file",
        )
    )
    assert set(errors) == {"cv2"}
    assert modules["cv2"] is None
    assert modules["mediapipe"] is not None
    assert "libGL.so.1" in errors["cv2"]


def test_probe_mediapipe_failure_blocks_snapshot():
    _, errors = probe_live_deps(
        _failing_importer(
            {"mediapipe"},
            "libGL.so.1: cannot open shared object file: No such file",
        )
    )
    assert set(errors) == {"mediapipe"}
    status = live_check_status(errors)
    assert status == {"snapshot_ok": False, "video_ok": False}


def test_status_snapshot_needs_mediapipe_and_cv2():
    assert live_check_status({}) == {"snapshot_ok": True, "video_ok": True}
    assert live_check_status({"cv2": "boom"}) == {
        "snapshot_ok": False,
        "video_ok": False,
    }
    assert live_check_status({"mediapipe": "boom"}) == {
        "snapshot_ok": False,
        "video_ok": False,
    }


def test_status_video_needs_av_and_webrtc_too():
    assert live_check_status({"av": "boom"}) == {
        "snapshot_ok": True,
        "video_ok": False,
    }
    assert live_check_status({"streamlit_webrtc": "boom"}) == {
        "snapshot_ok": True,
        "video_ok": False,
    }


def test_diagnostic_names_each_failed_module():
    text = diagnostic_text(
        {
            "cv2": "ImportError: libGL.so.1: cannot open shared object file",
            "av": "ImportError: No module named 'av'",
        }
    )
    assert "cv2: ImportError: libGL.so.1" in text
    assert "av: ImportError: No module named 'av'" in text


def test_diagnostic_points_libgl_failures_at_packages_txt():
    text = diagnostic_text(
        {"cv2": "ImportError: libGL.so.1: cannot open shared object file"}
    )
    assert "libgl1" in text
    assert "packages.txt" in text


def test_diagnostic_points_egl_failures_at_libegl1():
    text = diagnostic_text(
        {"mediapipe": "OSError: libEGL.so.1: cannot open shared object file"}
    )
    assert "libegl1" in text
    assert "packages.txt" in text


def test_diagnostic_empty_when_nothing_failed():
    assert diagnostic_text({}) == ""


def test_landmarker_hint_names_the_missing_library_and_packages_txt():
    text = landmarker_error_hint(
        OSError("libEGL.so.1: cannot open shared object file")
    )
    assert "libEGL.so.1" in text
    assert "packages.txt" in text
    assert "libegl1" in text


def test_env_check_has_no_heavy_top_level_imports():
    source = (HERE / "demo" / "env_check.py").read_text()
    for line in source.splitlines():
        assert not re.match(
            r"\s*(import|from)\s+(cv2|av|mediapipe|streamlit_webrtc)\b", line
        ), f"env_check must probe, not import, heavy modules: {line}"
    assert "import streamlit" not in source


def test_app_uses_probe_and_shows_the_diagnostic():
    source = (HERE / "demo" / "app.py").read_text()
    assert "probe_live_deps" in source
    assert "diagnostic_text" in source
    assert "landmarker_error_hint" in source
    assert "st.expander" in source
    # The old swallow-all block is gone: no bare False assignment.
    assert "MEDIAPIPE_AVAILABLE = False" not in source
    assert "WEBRTC_AVAILABLE = False" not in source
