"""Day 20: probe the live-check dependencies and explain failures.

No Streamlit, MediaPipe, or OpenCV import at module level here, so
the test suite can exercise this module anywhere. The app
(demo/app.py) calls probe_live_deps() once at startup and uses the
result to decide what the live check can do and what to tell the
visitor when it cannot run.

Why this exists: on Nov 21 the deployed demo showed only "The live
check needs MediaPipe, which did not load in this environment" with
no tabs at all, and the old code swallowed the real error in one
try/except around four imports. Reproduced locally in a fresh venv
with the Cloud's missing system library hidden, the chain is:

- mediapipe 1.2.3 imports cv2 at module load: its tasks vision
  drawing_utils does `import cv2`, reached through
  mediapipe -> tasks.python -> vision -> drawing_styles.
- The cv2 that wins is opencv-contrib-python, mediapipe's own
  dependency, installed after opencv-python-headless, whose files
  overwrite the headless ones. That cv2 links against libGL.so.1.
- Streamlit Community Cloud does not ship libGL.so.1, so `import cv2`
  raises ImportError, `import mediapipe` raises with it, and the
  whole guarded block in the old app failed. The fix is packages.txt
  at the repo root (libgl1, libglib2.0-0, libegl1, libgles2).

If it still fails after that, the app now shows the exact import
error in an expander instead of the bare warning, so the next fix
does not need another round of guessing from the outside.
"""

from __future__ import annotations

import importlib

# (label, module whose import is probed, module handed back to the app)
# for each live-check dependency. mediapipe is probed through its
# tasks vision submodule, because that is the import that pulls cv2
# in, while the app uses the top-level mediapipe module.
LIVE_DEPS = (
    ("av", "av", "av"),
    ("cv2", "cv2", "cv2"),
    ("mediapipe", "mediapipe.tasks.python.vision", "mediapipe"),
    ("streamlit_webrtc", "streamlit_webrtc", "streamlit_webrtc"),
)

# System libraries behind the common failure modes, and the packages.txt
# entry that supplies each one on Streamlit Community Cloud.
SYSTEM_LIB_HINTS = (
    ("libGL.so", "libgl1"),
    ("libglib", "libglib2.0-0"),
    ("libEGL.so", "libegl1"),
    ("libGLESv2.so", "libgles2"),
)


def probe_live_deps(importer=importlib.import_module):
    """Try importing each live-check dependency.

    Returns (modules, errors): modules maps label -> module or None,
    errors maps label -> "ExcType: message" for each failed import.
    importer is injectable so tests can simulate a missing system
    library without touching sys.modules.
    """
    modules = {}
    errors = {}
    for label, probe_name, return_name in LIVE_DEPS:
        try:
            importer(probe_name)
            modules[label] = importer(return_name)
        except Exception as exc:  # noqa: BLE001 - recorded, not raised
            modules[label] = None
            errors[label] = f"{type(exc).__name__}: {exc}"
    return modules, errors


def live_check_status(errors):
    """What the live check can do, given a probe errors dict.

    The snapshot check needs MediaPipe and cv2; the video stream needs
    those plus av (video frames) and streamlit-webrtc.
    """
    snapshot_ok = "mediapipe" not in errors and "cv2" not in errors
    video_ok = (
        snapshot_ok
        and "av" not in errors
        and "streamlit_webrtc" not in errors
    )
    return {"snapshot_ok": snapshot_ok, "video_ok": video_ok}


def diagnostic_text(errors):
    """Human-readable lines for a probe errors dict.

    One line per failed import, plus the system-library hint when the
    error names one. Empty string when nothing failed.
    """
    if not errors:
        return ""
    lines = []
    for label, error in errors.items():
        lines.append(f"{label}: {error}")
        for needle, package in SYSTEM_LIB_HINTS:
            if needle in error:
                lines.append(
                    f"  -> {label} needs the system library {needle}, "
                    "which Streamlit Community Cloud does not ship. "
                    f"packages.txt must list {package}."
                )
    lines.append(
        "The Run 1 session below still works; only the live check is off."
    )
    return "\n".join(lines)


def landmarker_error_hint(exc):
    """Actionable message when creating a PoseLandmarker fails.

    Import can succeed while creation fails: the native library is
    loaded lazily with ctypes, so a missing libEGL only shows up here
    (on this VM, import works but create_from_options raises
    OSError: libEGL.so.1: cannot open shared object file). The raw
    error names the missing library; the hint names the fix.
    """
    return (
        "Could not start the pose landmarker.\n"
        f"{type(exc).__name__}: {exc}\n"
        "On Streamlit Community Cloud this means a system library is "
        "missing; packages.txt must list libgl1, libglib2.0-0, libegl1 "
        "and libgles2."
    )
