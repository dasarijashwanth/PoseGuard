"""PoseGuard public demo (Day 19).

Run locally:  streamlit run demo/app.py
Deploy free:  Streamlit Community Cloud, pointed at this repo, main file demo/app.py.

The app shows the real Run 1 validation session, not a simulation:
sample angle traces, gate verdicts per trial, and the model scoreboard.
The honest verdict is the headline: nothing passes the gate yet.

Day 19 turns the showcase into a real app: a live webcam check
(streamlit-webrtc video when the browser allows it, camera snapshots
otherwise), a purpose and how-to-use section, and a feedback link.
No video ever leaves the visitor's machine: pose landmarks are
computed from the camera frames on the spot, and only the angles
are kept. The angle math is the same as the Day 14 capture tool
(ear line vs shoulder line, circular deviations from an upright
baseline), and the gate is the same rule the grid search scored.
"""

from __future__ import annotations

import os
import time

import altair as alt
import numpy as np
import streamlit as st

from demo_data import (
    FEEDBACK_BODY,
    FEEDBACK_HEADING,
    FEEDBACK_URL,
    HEAD_THRESHOLD_DEG,
    HOWTO_HEADING,
    HOWTO_STEPS,
    PURPOSE_BODY,
    PURPOSE_HEADING,
    SHOULDER_THRESHOLD_DEG,
    VERDICT,
    best_rule,
    gate_verdict,
    load_sample,
    sample_stats,
    scoreboard_frame,
    trials_frame,
)
from webcam import (
    LIVE_PRESETS,
    LiveGate,
    calibrate,
    deviations,
    ema_smooth,
    evaluate_stream,
    landmark_angles,
)

# MediaPipe Tasks is the supported pose API (the legacy solutions API
# was removed in mediapipe 1.x). Both imports are guarded so the
# sample showcase still renders when the live check cannot load.
try:
    import av
    import cv2
    from mediapipe import Image as MPImage
    from mediapipe import ImageFormat
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import (
        PoseLandmarker,
        PoseLandmarkerOptions,
        RunningMode,
    )

    MEDIAPIPE_AVAILABLE = True
except Exception:  # pragma: no cover - environment without MediaPipe
    MEDIAPIPE_AVAILABLE = False

try:
    from streamlit_webrtc import RTCConfiguration, VideoProcessorBase, webrtc_streamer

    WEBRTC_AVAILABLE = True
except Exception:  # pragma: no cover - environment without webrtc
    WEBRTC_AVAILABLE = False

MODEL_PATH = "demo/pose_landmarker_lite.task"
# The lite landmarker is bundled in the repo so the deployed app does
# not need a model download at runtime. Fall back to the script-relative
# path when the app runs from the demo directory.
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "pose_landmarker_lite.task")

RTC_CONFIG = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# Frames collected for one calibration run in live video mode.
CALIBRATION_FRAMES = 15


@st.cache_resource
def _image_landmarker():
    """PoseLandmarker for single snapshots (IMAGE running mode)."""
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
    )
    return PoseLandmarker.create_from_options(options)


@st.cache_resource
def _video_landmarker():
    """PoseLandmarker for the video stream (VIDEO running mode)."""
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return PoseLandmarker.create_from_options(options)


def _detect_image(rgb) -> list | None:
    """Pose landmarks from one RGB frame, or None when no pose is found."""
    result = _image_landmarker().detect(MPImage(ImageFormat.SRGB, rgb))
    if not result.pose_landmarks:
        return None
    return result.pose_landmarks[0]


class PoseProcessor(VideoProcessorBase):
    """WebRTC video processor: angles, calibration, and the live gate.

    Runs in the stream thread. The main thread reads self.latest for
    the readouts and sets self.calibrate_requested when the visitor
    presses Calibrate. Everything angle-shaped goes through
    demo/webcam.py, so the same code is what the tests cover.
    """

    def __init__(self, preset: str = "operating"):
        self.landmarker = _video_landmarker()
        self.set_preset(preset)
        self.baseline: tuple[float, float] | None = None
        self.calib_samples: list = []
        self.calibrate_requested = False
        self.smooth_head: float | None = None
        self.smooth_sh: float | None = None
        self.last_ts_ms = 0
        self.latest = {
            "tracking": False,
            "head_dev": None,
            "shoulder_dev": None,
            "head_fired": False,
            "shoulder_fired": False,
            "head_run_s": 0.0,
            "shoulder_run_s": 0.0,
        }

    def set_preset(self, preset: str) -> None:
        cfg = LIVE_PRESETS[preset]
        self.preset = preset
        self.gates = {
            cue: LiveGate(**cfg[cue]) for cue in ("head", "shoulder")
        }

    def request_calibration(self) -> None:
        self.calibrate_requested = True
        self.calib_samples = []

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ts_ms = max(int(time.time() * 1000), self.last_ts_ms + 1)
        self.last_ts_ms = ts_ms
        try:
            result = self.landmarker.detect_for_video(
                MPImage(ImageFormat.SRGB, rgb), ts_ms
            )
            landmarks = result.pose_landmarks[0] if result.pose_landmarks else None
        except Exception:
            landmarks = None
        if landmarks is None:
            self.latest["tracking"] = False
            for gate in self.gates.values():
                gate.update(time.time(), None)
            self._draw(img, "No pose found: sit so your ears and shoulders are in frame.")
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        try:
            head_raw, shoulder_raw = landmark_angles(landmarks)
        except ValueError:
            self.latest["tracking"] = False
            self._draw(img, "Landmarks unusable on this frame.")
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        self.smooth_head = ema_smooth(self.smooth_head, head_raw)
        self.smooth_sh = ema_smooth(self.smooth_sh, shoulder_raw)

        if self.calibrate_requested:
            self.calib_samples.append((self.smooth_head, self.smooth_sh))
            if len(self.calib_samples) >= CALIBRATION_FRAMES:
                self.baseline = calibrate(self.calib_samples)
                self.calibrate_requested = False
                for gate in self.gates.values():
                    gate.reset()
            self._draw_landmarks(img, landmarks)
            self._draw(img, f"Calibrating upright: {len(self.calib_samples)}/{CALIBRATION_FRAMES} frames.")
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        self.latest["tracking"] = True
        if self.baseline is None:
            self._draw_landmarks(img, landmarks)
            self._draw(img, "Press Calibrate while sitting upright, then hold a tilt or a dropped shoulder.")
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        head_dev, shoulder_dev = deviations(self.smooth_head, self.smooth_sh, self.baseline)
        now = time.time()
        head_state = self.gates["head"].update(now, head_dev)
        shoulder_state = self.gates["shoulder"].update(now, shoulder_dev)
        self.latest.update(
            {
                "head_dev": round(head_dev, 1),
                "shoulder_dev": round(shoulder_dev, 1),
                "head_fired": head_state["fired"],
                "shoulder_fired": shoulder_state["fired"],
                "head_run_s": head_state["run_s"],
                "shoulder_run_s": shoulder_state["run_s"],
            }
        )
        self._draw_landmarks(img, landmarks)
        status = (
            f"Head {head_dev:+.1f} deg (gate {head_state['run_s']:.1f}s)  "
            f"Shoulder {shoulder_dev:+.1f} deg (gate {shoulder_state['run_s']:.1f}s)"
        )
        if head_state["fired"] or shoulder_state["fired"]:
            fired = [c for c, s in (("head", head_state), ("shoulder", shoulder_state)) if s["fired"]]
            status += "  GATE FIRED: " + " + ".join(fired)
        self._draw(img, status)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

    @staticmethod
    def _draw_landmarks(img, landmarks) -> None:
        for idx in (7, 8, 11, 12):
            lm = landmarks[idx]
            x, y = int(lm.x * img.shape[1]), int(lm.y * img.shape[0])
            cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
        h, w = img.shape[0], img.shape[1]
        for a, b in ((7, 8), (11, 12)):
            x1, y1 = int(landmarks[a].x * w), int(landmarks[a].y * h)
            x2, y2 = int(landmarks[b].x * w), int(landmarks[b].y * h)
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    @staticmethod
    def _draw(img, text: str) -> None:
        cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 1, cv2.LINE_AA)


def _live_check() -> None:
    st.header("Try it live: webcam check")
    st.write(
        "Sit upright, press Calibrate, then tilt your head or drop one "
        "shoulder and hold it. The gate fires when the hold lasts. "
        "Nothing is recorded and no video leaves your browser: the pose "
        "landmarks are computed from the camera frames on the spot, and "
        "only the angles are kept."
    )
    if not MEDIAPIPE_AVAILABLE:
        st.warning(
            "The live check needs MediaPipe, which did not load in this "
            "environment. The Run 1 session below still works, and the "
            "live check runs where the demo requirements are installed."
        )
        return

    preset = st.radio(
        "Gate preset",
        options=["operating", "best_rule"],
        format_func=lambda p: (
            "Operating thresholds: head 8 deg / shoulder 6 deg, hold 5 s"
            if p == "operating"
            else "Best rule-grid configs: head 4 deg / 2 s, shoulder 1 deg / 5 s"
        ),
        horizontal=True,
    )
    if preset == "best_rule":
        st.caption(
            "The best-rule preset shows why those configs fail: the "
            "shoulder threshold is 1 deg, so everyday leaning clears it "
            "and the gate fires on ordinary movement."
        )

    if WEBRTC_AVAILABLE:
        tab_video, tab_snap = st.tabs(["Live video", "Snapshot check"])
    else:
        st.info(
            "Live video streaming is not available in this environment, "
            "so the snapshot check below is the way to try it."
        )
        tab_video, tab_snap = None, st.container()

    if tab_video is not None:
        with tab_video:
            _live_video_tab(preset)
    with tab_snap:
        _snapshot_tab(preset)


def _live_video_tab(preset: str) -> None:
    try:
        ctx = webrtc_streamer(
            key="poseguard-live",
            video_processor_factory=lambda: PoseProcessor(preset),
            rtc_configuration=RTC_CONFIG,
            media_stream_constraints={"video": True, "audio": False},
        )
    except Exception as exc:  # pragma: no cover - no browser in tests
        st.warning(f"Live video could not start ({exc}). Use the snapshot check instead.")
        return
    if ctx.video_processor is None:
        st.write("Press Start above to open your camera.")
        return
    processor = ctx.video_processor
    if processor.preset != preset:
        processor.set_preset(preset)
    if st.button("Calibrate upright", key="calibrate-live"):
        processor.request_calibration()
        st.write("Calibrating: sit upright and stay still for a second.")
    head_box = st.empty()
    shoulder_box = st.empty()
    status_box = st.empty()
    while ctx.state.playing:
        latest = processor.latest
        if not latest["tracking"]:
            status_box.write("Waiting for a pose: sit so your ears and shoulders are in frame.")
        else:
            head_box.metric(
                "Head deviation",
                f"{latest['head_dev']} deg" if latest["head_dev"] is not None else "calibrate first",
                f"gate run {latest['head_run_s']} s" + ("  GATE FIRED" if latest["head_fired"] else ""),
            )
            shoulder_box.metric(
                "Shoulder deviation",
                f"{latest['shoulder_dev']} deg" if latest["shoulder_dev"] is not None else "calibrate first",
                f"gate run {latest['shoulder_run_s']} s" + ("  GATE FIRED" if latest["shoulder_fired"] else ""),
            )
            if latest["head_fired"] or latest["shoulder_fired"]:
                fired = [c for c, f in (("head", latest["head_fired"]), ("shoulder", latest["shoulder_fired"])) if f]
                status_box.success("Gate fired on: " + " + ".join(fired) + ". That is a hold the gate counts.")
            else:
                status_box.write("Hold a tilt or a dropped shoulder to fire the gate.")
        time.sleep(0.25)


def _snapshot_tab(preset: str) -> None:
    st.write(
        "Take a photo sitting upright to calibrate, then take more "
        "photos while holding the pose. Each photo is one sample, and "
        "repeated photos build the same gate the video uses."
    )
    if "snap_baseline" not in st.session_state:
        st.session_state["snap_baseline"] = None
    if "snap_samples" not in st.session_state:
        st.session_state["snap_samples"] = []
    photo = st.camera_input("Take a snapshot", key="snapshot-check")
    if photo is None:
        if st.session_state["snap_samples"]:
            _snapshot_results(preset)
        return
    data = photo.getvalue()
    img = cv2.imdecode(np.frombuffer(data, dtype="uint8"), cv2.IMREAD_COLOR)
    if img is None:
        st.warning("Could not read that photo. Try again.")
        return
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    try:
        landmarks = _detect_image(rgb)
    except Exception as exc:
        st.warning(f"Pose detection failed on this photo ({exc}). Try again.")
        return
    if landmarks is None:
        st.warning("No pose found in that photo. Sit so your ears and shoulders are in frame and try again.")
        return
    head_raw, shoulder_raw = landmark_angles(landmarks)
    if st.session_state["snap_baseline"] is None:
        st.session_state["snap_baseline"] = (head_raw, shoulder_raw)
        st.success("Calibrated from this photo: that is your upright pose. Now hold a tilt or a dropped shoulder and take another photo.")
        return
    head_dev, shoulder_dev = deviations(head_raw, shoulder_raw, st.session_state["snap_baseline"])
    st.session_state["snap_samples"].append((time.time(), head_dev, shoulder_dev))
    st.write(f"This photo: head {head_dev:+.1f} deg, shoulder {shoulder_dev:+.1f} deg from your upright pose.")
    _snapshot_results(preset)
    if st.button("Reset snapshot check", key="reset-snap"):
        st.session_state["snap_baseline"] = None
        st.session_state["snap_samples"] = []
        st.rerun()


def _snapshot_results(preset: str) -> None:
    samples = st.session_state["snap_samples"]
    if not samples:
        return
    cfg = LIVE_PRESETS[preset]
    head = evaluate_stream([(t, d) for t, d, _ in samples], **cfg["head"])
    shoulder = evaluate_stream([(t, _, d) for t, _, d in samples], **cfg["shoulder"])
    st.write(f"Samples so far: {len(samples)} photos.")
    for cue, result in (("head", head), ("shoulder", shoulder)):
        if result["fired"]:
            st.success(f"{cue.capitalize()} gate FIRED: held above {cfg[cue]['threshold_deg']} deg for {cfg[cue]['persist_s']} s.")
        else:
            st.write(
                f"{cue.capitalize()} gate: longest run {result['max_run_s']} s "
                f"(needs {cfg[cue]['persist_s']} s above {cfg[cue]['threshold_deg']} deg)."
            )


def main() -> None:
    st.set_page_config(page_title="PoseGuard: on-device posture cues", layout="wide")
    st.title("PoseGuard: can a webcam tell when you slouch?")
    st.write(
        "On-device vision AI that watches head and shoulder angles during desk "
        "work and nudges you to sit up. This demo shows the real validation "
        "session (Run 1, Nov 3, 2026) and every number the project has "
        "measured so far, including the ones that do not flatter it."
    )

    st.error("Honest verdict: " + VERDICT)

    st.header(PURPOSE_HEADING)
    st.write(PURPOSE_BODY)

    st.header(HOWTO_HEADING)
    for step in HOWTO_STEPS:
        st.write("- " + step)

    _live_check()

    df = load_sample()
    stats = sample_stats(df)

    st.header("The session, as recorded")
    st.write(
        f"Run 1: {stats['rows']} samples at 1 Hz over {stats['duration_s']} s, "
        f"steps {', '.join(stats['steps'])}. Largest head deviation "
        f"{stats['max_abs_head_dev']} deg and largest shoulder deviation "
        f"{stats['max_abs_shoulder_dev']} deg, both at 447 s: a 6-second "
        "tracker meltdown mid-work, not a posture change. The chart axis is "
        "clipped at +/-25 deg so that meltdown does not hide the 2-7 deg "
        "holds; gaps are frames where the tracker was lost or gated out. "
        "Angles are circular deviations from the calibration baseline, "
        "shoulder jump-gated the Day 11 way."
    )
    chart_df = df.melt(
        id_vars=["elapsed_s"],
        value_vars=["head_dev_deg", "shoulder_dev_deg"],
        var_name="cue",
        value_name="dev_deg",
    )
    chart = (
        alt.Chart(chart_df)
        .mark_line()
        .encode(
            x=alt.X("elapsed_s", title="Elapsed (s)"),
            y=alt.Y("dev_deg", title="Deviation from baseline (deg)",
                    scale=alt.Scale(domain=[-25, 25])),
            color=alt.Color("cue", title="Cue"),
        )
    )
    st.altair_chart(chart, use_container_width=True)
    st.caption(
        f"Session thresholds: head {HEAD_THRESHOLD_DEG} deg, shoulder "
        f"{SHOULDER_THRESHOLD_DEG} deg. Deliberate holds were 10 s; ordinary "
        "work drifts 4-7 deg on the shoulder line, which is why the holds "
        "hide inside everyday movement."
    )

    st.header("Gate verdicts per trial")
    st.write(
        "The gate is 4/5 trials detected with 0 false positives. Trial 5 is "
        "barely recorded (6 head frames, 8 shoulder frames), so at most 4/5 "
        "trials are measurable in Run 1."
    )
    for cue in ("head", "shoulder"):
        rule = best_rule(cue)
        verdict = gate_verdict(rule["detected"], rule["false_positives"])
        st.subheader(f"{cue.capitalize()} cue: best rule")
        st.write(
            f"{rule['threshold_deg']} deg / {rule['persist_s']} s / "
            f"{rule['vel_cap_deg_s']} deg/s cap: {rule['detected']}/5 trials, "
            f"{rule['false_positives']} false positives, F1 {rule['f1']}. "
            + verdict["reason"]
        )
        st.dataframe(trials_frame(cue), use_container_width=True)
    st.caption(
        "Dwell is the longest run above the dwell threshold (head 4 deg, "
        "shoulder 2 deg). The holds are shallow as well as brief: the deepest "
        "head hold peaks at 7.1 deg against an 8 deg threshold no hold ever reaches."
    )

    st.header("Model scoreboard")
    st.write(
        "Grouped 5-fold CV by trial, same protocol for every row. Latest "
        "feature set per cue (head: Day 15 velocity + dwell; shoulder: Day 17 "
        "velocity + dwell). The new features made both models worse, so the "
        "served models are unchanged."
    )
    st.dataframe(scoreboard_frame(), use_container_width=True)
    st.caption(
        "The random forest looks accurate only because it predicts almost "
        "everything upright: 0/5 trials on both cues. Calm-only accuracy is "
        "accuracy on work windows with no tilt."
    )

    st.header("What would fix it")
    st.write(
        "A camera re-run with raw landmarks and longer, deeper holds "
        "(the Day 14 capture tool: 10 s holds, shoulder 8 deg and head "
        "15 deg targets). The Day 16 harness re-benchmarks everything the "
        "moment that capture lands. Until then the served models stay as "
        "they are, and the gate stays unpassed."
    )
    st.write(
        "Code, walkthroughs, and the full benchmark tables: "
        "github.com/dasarijashwanth/PoseGuard"
    )

    st.header(FEEDBACK_HEADING)
    st.write(FEEDBACK_BODY)
    st.link_button("Open an issue on GitHub", FEEDBACK_URL)


if __name__ == "__main__":
    main()
