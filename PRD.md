# PoseGaurd — Product Requirements

**Status:** Phase 1 scope agreed; prototype validation pending. Phase 2 is a planned optional extension, not implemented.

## Problem and Goal

Desk workers and students may not notice sustained changes in their sitting alignment. Provide gentle awareness reminders using a laptop webcam. Camera processing always stays on the device; session summaries stay local by default, with optional syncing in Phase 2.

## Delivery Plan

- **Phase 1 — Browser-only MVP:** local detection, reminders, session summaries, exports, and offline use. Validate head and shoulder tilt before completing the reminder flow.
- **Phase 2 — Optional history syncing:** Python + FastAPI for consent-based summary syncing, cross-device history, and weekly reports. Begin after Phase 1 acceptance checks pass.

```text
Webcam → browser inference → posture rules → reminders
                                  ↓
                         IndexedDB summaries
                                  ↓ Optional, with consent
                         FastAPI → database
                                  ↓
                       History and weekly reports
```

## Phase 1: MVP Scope

One user faces a fixed laptop webcam with ears and shoulders visible. Support two cues, each compared with a calibrated reference:

- **Head tilt:** change in the angle of the ear line relative to the shoulder line.
- **Shoulder tilt:** change in the angle of the shoulder line relative to the image horizontal. This reports visible alignment, not its cause or proof of whole-body leaning.

Forward neck flexion, spinal slouch, and back curvature are outside the MVP. Accounts and cloud syncing are deferred to Phase 2. Medical assessment, health scores, and custom model training are outside both phases. Forward slouch detection requires a separate validation effort and is not part of Phase 2.

## Phase 1: User Flow

Open → allow camera → check visibility → calibrate → monitor → end session → review or export summary.

## Phase 1: Requirements

- Show camera loading, denied permission, unavailable camera, and tracking-loss states. Stop camera capture when the session ends.
- Calibrate from about three seconds of stable, sufficiently visible landmarks. Require recalibration after the camera or seating setup changes.
- Run Pose Landmarker Lite locally on CPU; GPU acceleration is optional. Keep inference off the interface thread and avoid queuing stale frames. Initially target 10–15 checks per second, subject to measurement.
- Smooth landmark estimates and evaluate each cue against its own calibrated baseline and threshold.
- Show a visual reminder after five continuous seconds beyond threshold; optionally play a chime with a mute control. Time each cue independently using elapsed time. Brief deviations must not accumulate; avoid repeated chimes for the same ongoing event.
- If required landmarks become unreliable or results become stale, pause the affected cue and reset its timer. Display an adjustment prompt; resume only with reliable tracking.
- Display a live landmark overlay. Save session summaries and settings in IndexedDB; report tracked time within/outside the configured range and untracked time separately. Count simultaneous deviations once in combined outside-range time. Export summaries as CSV or JSON.
- Do not record or upload frames. Support offline reload and monitoring after all required application, model, and runtime assets are cached.

## Phase 1: Initial Validation Gate

With the laptop camera fixed, calibrate comfortably upright. Hold a head tilt, return to reference, then hold a shoulder tilt; hold each for six seconds. Repeat five times per cue. Test five minutes of ordinary typing, looking around, and brief reaching, plus deliberate loss of required landmarks.

Proceed with each cue only if it detects at least four of five deliberate holds, produces no reminders during the ordinary-work sample, and pauses when required points disappear. Record angle thresholds, visibility cutoffs, smoothing, browser, hardware, and results. These are initial engineering criteria, not evidence of general accuracy. If either cue fails, investigate and retest before claiming support.

## Phase 1: Acceptance Checks

- Verify calibration, five-second timing, timer resets, mute, and camera shutdown; brief movements and tracking gaps must not trigger reminders.
- Verify session accounting and both export formats against a known test session.
- Verify offline reload with networking disabled and inspect network traffic to confirm no frame uploads.
- Record CPU inference rate and interface responsiveness on the development laptop; publish measured results and supported browsers.

## Implementation Stack

React, TypeScript, Vite, MediaPipe Tasks Vision, Web Worker/OffscreenCanvas, Canvas 2D, Web Audio, IndexedDB, and a service worker/cache. Vitest and Playwright for testing; Python for local experiments; Vercel for static deployment. No application backend is required for Phase 1. Phase 2 adds Python + FastAPI and a database; database, authentication provider, and backend hosting will be selected before that phase. Browser compatibility remains to be verified.

## Proposed Phase 1 Project Structure

Keep one Vite application at the repository root. Create folders as their features are implemented.

```text
pose-guard/
├── README.md
├── PRD.md
├── package.json
├── index.html
├── vite.config.ts
├── tsconfig.json
├── public/
│   ├── models/           # Pose Landmarker Lite model
│   └── wasm/             # Matching MediaPipe runtime assets
├── src/
│   ├── main.tsx          # Application entry point
│   ├── App.tsx           # Session flow and application state
│   ├── components/       # Camera view, controls, status, session summary
│   ├── camera/           # Permissions, capture, frame scheduling, cleanup
│   ├── pose/             # Model initialization, inference worker, message types
│   ├── posture/          # Angles, smoothing, calibration, visibility, cue timers
│   ├── feedback/         # Skeleton drawing and sound reminders
│   ├── sessions/         # Time accounting, IndexedDB, CSV/JSON export
│   ├── offline/          # Service worker, cache versions, offline readiness
│   └── styles/           # Application styles
├── tests/
│   ├── e2e/              # Playwright browser flows and offline checks
│   └── fixtures/         # Synthetic landmark sequences for repeatable tests
├── experiments/          # Local Python exploration and evaluation scripts
└── docs/
    └── validation.md     # Experiment settings, hardware, browsers, results
```

Keep posture calculations independent of React and MediaPipe, using plain landmark data and timestamps. Place Vitest unit tests beside the modules they verify. Use explicit worker message types to connect inference to the interface.

Python experiments support development only and are excluded from the deployed application. Do not commit camera recordings or personal session exports; use synthetic fixtures for shared tests. Pin model and runtime versions together and include required assets in offline caching. No backend folder is needed for the MVP.

## Phase 2: Optional Summary Syncing

**Flow:** sign in → explicitly enable syncing → upload completed summaries → view history and weekly reports. Monitoring remains usable without an account or API connection.

### Requirements

- Keep syncing off by default. Obtain consent before uploading summaries, including any previously saved local sessions. Allow disabling syncing and deleting cloud history; disabling stops pending uploads but does not itself delete existing cloud history.
- Authenticate users and restrict all history, reports, and deletion operations to the requesting user's data.
- Save completed summaries locally first. Upload session identifiers, timestamps, duration, tracked/untracked time, time within/outside the configured range, and reminder counts only. Never upload camera frames or individual landmark sequences.
- Retry failed uploads with a stable session identifier so retries cannot create duplicates. API failures must not interrupt monitoring or local summaries.
- Calculate weekly reports from saved summaries. Use tracked duration to calculate time-in-range percentages and report untracked time separately.

### API and Structure

- `POST /sessions` — save a completed summary without duplication on retry.
- `GET /sessions` — retrieve the authenticated user's history.
- `GET /reports/weekly` — retrieve the user's weekly aggregates.
- `DELETE /sessions` — delete the user's cloud session history.

Add these folders only when Phase 2 starts:

```text
src/api/                 # Consent-aware summary sync client
backend/
├── pyproject.toml       # Python dependencies and tooling
├── app/
│   ├── main.py          # FastAPI application
│   ├── schemas.py       # Summary request/response validation
│   ├── db.py            # Database access
│   ├── routes/          # Session and report endpoints
│   └── services/        # Weekly aggregations
└── tests/               # API, ownership, retries, reports, deletion
```

### Acceptance Checks

- Verify no summary uploads before consent or after syncing is disabled, including queued retries.
- Verify offline monitoring, local saving, and duplicate-free retry after reconnection.
- Verify one user cannot read, overwrite, report on, or delete another user's data; check summary-only upload payloads.
- Verify weekly totals against known sessions and confirm deleted cloud history is not silently restored from the local sync queue.

A Python service that users must install locally is outside this plan; live inference and posture rules remain in the browser in both phases.
