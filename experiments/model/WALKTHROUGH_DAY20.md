# WALKTHROUGH_DAY20.md

## Day 20 (Nov 22): make MediaPipe load on Streamlit Cloud

Day 19 shipped the live webcam check and the user tried it the same
evening. The new sections were on the live page, but the check
itself was not: under "Try it live: webcam check" the app showed
only "The live check needs MediaPipe, which did not load in this
environment", with no Live video or Snapshot tabs at all. Locally
MediaPipe loads fine. Day 20 is the dependency and environment fix
that makes it load on Cloud too.

## Root cause: cv2, not MediaPipe

The old `demo/app.py` wrapped four imports in one try/except, so the
deployed warning named MediaPipe and hid which import actually
failed. Reproduced in a fresh venv installing only
`demo/requirements.txt`, with the Cloud's missing system library
hidden (bind-mounting /dev/null over libGL.so.1), the chain is:

1. mediapipe 1.2.3 imports cv2 at module load. `mediapipe/__init__.py`
   eagerly imports `mediapipe.tasks.python`, which imports `vision`,
   which imports `drawing_styles`, which imports `drawing_utils`,
   which does `import cv2` at its top level.
2. The cv2 that wins is opencv-contrib-python, mediapipe's own
   dependency. pip installs it after opencv-python-headless, and its
   cv2 files overwrite the headless ones. That cv2 links against the
   system libraries libGL.so.1 and libglib-2.0.so.0 (shown by ldd).
3. Streamlit Community Cloud does not ship libGL.so.1, so `import cv2`
   raises ImportError, `import mediapipe` raises with it, and the
   whole guarded block in the old app failed. `opencv-python-headless`
   in the requirements never had a chance to help.

So the warning text was accurate about the symptom and wrong about
the cause: MediaPipe did not load because cv2 could not load.

A second failure hides behind the first. Import can succeed while
creating a landmarker fails: mediapipe's native library loads lazily
with ctypes, so a missing libEGL only shows up at
`PoseLandmarker.create_from_options`. On this VM, import works but
creation raises `OSError: libEGL.so.1: cannot open shared object
file` (ldd on libmediapipe.so shows libEGL.so.1 not found, while
libGLESv2.so.2 is present). Day 19's walkthrough said the Cloud
image "already carries" libegl1/libgles2; that was unverified, and
this VM does not carry libEGL either, so Day 20 does not rely on it.

## What was built

### packages.txt (repo root)

Streamlit Community Cloud installs this file with apt-get before pip,
from the repository root:

- `libgl1` (libGL.so.1) and `libglib2.0-0` (libglib-2.0.so.0): what
  the winning cv2 needs to import at all.
- `libegl1` (libEGL.so.1) and `libgles2` (libGLESv2.so.2): what
  mediapipe's native library needs when a landmarker is created.

### demo/requirements.txt: pinned live-check stack

`mediapipe==1.2.3`, `streamlit-webrtc==0.79.1`, `streamlit==1.67.0`,
`opencv-python-headless==5.1.0.95`, the exact versions verified to
import together on Python 3.12 in a fresh venv. pandas and altair
stay ranged (they resolved to pandas 3 and altair 6 without breaking
the app). A comment in the file records why headless alone cannot
fix the Cloud: mediapipe's opencv-contrib-python overwrites it.

The requirements location is not the problem: Streamlit's docs say
the dependency file may sit at the repo root or next to the entrypoint
file, and `demo/requirements.txt` sits next to `demo/app.py`.

### demo/env_check.py (new): probe, status, diagnostic

A pure module with no Streamlit, MediaPipe, or OpenCV import at
module level, so tests can exercise it anywhere:

- `probe_live_deps()` imports each of av, cv2, mediapipe (through
  its tasks vision submodule, the import that pulls cv2 in), and
  streamlit_webrtc on its own, and returns the modules plus an
  errors dict mapping each failed label to "ExcType: message". The
  importer is injectable so tests can simulate a missing system
  library.
- `live_check_status(errors)`: the snapshot check needs MediaPipe and
  cv2; the video stream needs those plus av and streamlit_webrtc.
- `diagnostic_text(errors)`: one line per failed import, plus the
  system-library hint when the error names one (libGL -> libgl1,
  libglib -> libglib2.0-0, libEGL -> libegl1, libGLESv2 -> libgles2).
- `landmarker_error_hint(exc)`: the actionable message for when
  creating a landmarker fails after import succeeded.

### demo/app.py: the failure now names itself

- The single swallow-all try/except is gone. The app probes with
  `probe_live_deps()` and unpacks the returned modules, so a failure
  in one dependency cannot take the others down silently. The
  mediapipe attribute access goes through `_mp.tasks` (which is
  mediapipe.tasks.python, bound as `tasks` in mediapipe's __init__),
  not `_mp.tasks.python`, which does not exist.
- When the live check cannot run, the warning stays, and an expander
  "Why the live check is unavailable" shows the exact import errors.
  The same expander appears when only live video is unavailable.
- `PoseLandmarker.create_from_options` failures in both cached
  factories now raise a RuntimeError built by
  `landmarker_error_hint`, so the snapshot tab's existing warning
  shows the missing library and the packages.txt fix instead of a
  bare OSError.
- Two latent crashes fixed along the way: `RTC_CONFIG` was built
  unconditionally from `RTCConfiguration`, and `PoseProcessor`
  inherited from `VideoProcessorBase`, so a failed webrtc import
  would have crashed the whole app at import time instead of
  degrading to the snapshot check. Both are now guarded, and the
  class is only ever built when video is available.

## Verification

- Fresh venv (`~/.venv-day20`, Python 3.12) installing only
  `demo/requirements.txt` resolves to exactly the pinned versions
  and imports av, cv2, mediapipe, and streamlit_webrtc cleanly.
- With libGL.so.1 hidden, the same venv reproduces the Cloud
  failure: `import cv2` and `import mediapipe` both fail on libGL,
  and the new app module reports `MEDIAPIPE_AVAILABLE: False` with a
  diagnostic naming cv2 and mediapipe and pointing at libgl1 and
  packages.txt.
- With libGL present, `import app` in both the project venv and the
  fresh venv reports `MEDIAPIPE_AVAILABLE: True` and
  `WEBRTC_AVAILABLE: True` with no import errors.
- Landmarker creation on this VM still fails on the missing libEGL
  (the VM has no libegl1), which is exactly what packages.txt now
  installs on Cloud. The creation error is covered by the new
  RuntimeError message.
- 302/302 tests passing in one combined run (80 eval + 222 model,
  including 16 new Day 20 tests).

## What still cannot be verified without a real browser or camera

- That Streamlit Cloud's apt install of packages.txt actually
  supplies all four libraries and the redeploy then loads MediaPipe.
  The fix is the documented mechanism and the exact missing files,
  but only the next deploy proves it; the app's diagnostic expander
  is the tripwire if it does not.
- That the live video stream runs in a real browser: no camera on
  this VM, and the video path needs a visitor's camera and WebRTC.
  The snapshot path is the fallback and is exercised only by the
  import and gate tests, not by a real photo.
- That libEGL/libGLESv2 on Cloud match the versions mediapipe
  1.2.3 was built against. If creation still fails after the
  deploy, the RuntimeError message will name the missing library.

## Numbers (unchanged)

Best shoulder rule 4/5 trials with 8 false positives (F1 0.471),
best head rule 2/5 with 3 false positives (F1 0.400), best models
F1 0.172 shoulder and 0.105 head. Nothing passes the gate, and Day
20 changes no model, no threshold, and no gate logic: it only makes
the live check load where visitors actually open it.
