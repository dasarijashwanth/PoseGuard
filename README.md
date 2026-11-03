![PoseGuard — Posture awareness, illustrated with a man at a laptop and ear and shoulder tracking landmarks.](assets/poseguard-readme-banner.png)

# PoseGuard

A browser-based posture awareness app designed to process laptop webcam images locally and give gentle reminders about sustained changes in sitting position.

**Status:** Planning stage. The features below describe the intended first version; camera reliability, performance, and offline behavior have not been validated yet.

## Why PoseGuard?

While working or studying, it is easy to stay in a shifted sitting position without noticing. PoseGuard aims to help users notice those changes through live visual feedback and optional sound reminders, while keeping camera images on their laptop.

## Planned Features

- Laptop webcam access with clear permission and camera states.
- Live body landmark and skeleton overlay.
- A short calibration step to establish the user's reference sitting position.
- Sustained head tilt relative to the shoulder line, compared with calibration.
- Sustained shoulder tilt relative to the image horizontal, compared with calibration. This measures visible alignment, not the cause or whole-body leaning.
- A reminder when a supported deviation persists for five continuous seconds, with an option to mute sound.
- Local session summaries showing time within and outside the configured posture range, with untracked time reported separately.
- CSV and JSON export of session summaries.
- Offline use after the app, model, and runtime have been successfully cached.

The MVP assumes one user facing a fixed laptop webcam. Head tilt uses the line between the ears relative to the shoulder line; shoulder tilt uses the line between the shoulders relative to the image horizontal. Both cues still require validation on the development laptop. Forward neck flexion, spinal slouch, and back curvature are outside the MVP pending separate testing.

## How It Works

```mermaid
flowchart TD
    A[Allow webcam access] --> B[Check body landmark visibility]
    B --> C[Calibrate reference sitting position]
    C --> D[Estimate body landmarks locally]
    D --> E[Smooth estimates and compare with reference]
    E --> F{Deviation lasts five seconds?}
    F -->|No| D
    F -->|Yes| G[Show reminder and optional chime]
    G --> D
    E --> H[Store session summaries locally]
```

A pretrained pose model estimates body points such as ears and shoulders. Simple geometry and explicit rules compare those points with the calibrated reference. The initial calibration target is about three seconds.

When required landmarks are not visible or confidence is too low, posture evaluation pauses and the reminder timer resets. Monitoring resumes when tracking is reliable again.

## Planned Tech Stack

- **Frontend — React + TypeScript:** interface and application logic.
- **Build tool — Vite:** local development server and static production build.
- **Webcam — Browser `getUserMedia`:** laptop camera access with user permission.
- **Pose estimation — MediaPipe Tasks Vision, Pose Landmarker Lite:** local CPU inference using `delegate: "CPU"`, with optional GPU acceleration on supported devices. A dedicated GPU is not required. [Supported delegates](https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/tasks/web/core/task_runner_options.d.ts).
- **Background processing — Web Worker + OffscreenCanvas:** frame processing and inference away from the interface thread, subject to browser compatibility testing.
- **Posture rules — TypeScript:** alignment calculations, smoothing, calibration comparisons, and reminder timing.
- **Visual feedback — Canvas 2D:** body landmarks and skeleton overlay.
- **Sound — Web Audio:** optional reminder chimes and mute control.
- **Local storage — IndexedDB:** session summaries and settings.
- **Offline support — Service worker + Cache Storage:** cached application, model, and runtime assets.
- **Testing — Vitest + Playwright:** posture rule unit tests and browser flow tests.
- **Deployment — Vercel static hosting:** serves the built application without an application backend.
- **Experimentation and evaluation — Python:** local landmark experiments and validation of posture calculations; not required on users' laptops.

No application backend, cloud database, LLM, or custom model training is planned for the first version. Python + FastAPI would be considered later only if features such as accounts or optional cross-device syncing require a backend.

### CPU Inference Plan

Start with Pose Landmarker Lite and target 10–15 posture checks per second, then adjust based on measurements on the development laptop. This is an initial target, not a verified performance claim.

Process one frame at a time and skip incoming frames while inference is busy to avoid a queue of stale frames. Camera preview and posture checking can run at different rates. Reminder timing uses elapsed time rather than a fixed number of frames; tracking loss or stale results must reset the timer.

## Intended Usage

1. Open the app and allow webcam access.
2. Adjust the camera so the required body parts are visible.
3. Sit comfortably upright and calibrate your reference position.
4. Start monitoring and receive reminders for sustained, supported changes.
5. End the session to review or export its summary.

Recalibrate after moving the camera or substantially changing your seating setup.

## Privacy and Limitations

The design calls for local frame processing, no recording or uploading of camera images, and storage of session summaries and settings only. No accounts or application backend are planned. Initial loading and updates require downloading the application and model assets from a static host.

Detection depends on camera angle, lighting, clothing, and body visibility. A laptop webcam may not show the hips or reliably distinguish forward slouching. Feedback describes deviation from a reference position; it is not a medical assessment or a validated health score.

Browser storage can be cleared or evicted. Offline availability, CPU performance, and optional GPU acceleration must be tested on the supported browsers and devices.

## Development and Validation

The first milestone is a fixed-camera experiment: calibrate while comfortably upright, hold a head tilt and then a shoulder tilt for six seconds each, returning to the reference between attempts. Repeat five times per cue, then test five minutes of ordinary typing, looking around, and brief reaching. Also test loss of required landmarks.

The initial engineering gate is at least four of five deliberate holds detected per cue, no reminders during the five-minute ordinary-work sample, and paused evaluation when required landmarks disappear. These are prototype criteria, not measured results or evidence of general accuracy. Angle thresholds, visibility cutoffs, and smoothing settings will be selected during the experiment and recorded with results.

See [PRD.md](PRD.md) for the MVP requirements and acceptance criteria.

Validation will cover landmark reliability, false reminders during ordinary movement, tracking loss, reminder timing, inference speed, interface responsiveness, offline reloads, and network checks for frame uploads. Performance figures will be published only after measurement on identified hardware and browsers.

Setup commands, test instructions, screenshots, and a deployed demo link will be added as the implementation becomes available.
