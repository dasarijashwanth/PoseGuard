# src/camera

Day 2: camera capture pipeline.

- `types.ts` — shared types: permission states, constraint attempts, classified
  errors, frame-stats snapshots, and the main-thread/worker message protocol.
- `constraints.ts` — pure helpers: front-camera device selection, ordered
  getUserMedia fallbacks (HD → SD → any camera), and DOMException-name to
  plain-English error classification.
- `stats.ts` — `FrameStats`: FPS and frame-drop counters with elapsed-time
  timing and an injectable clock (no fixed frame-rate assumptions).
- `capture.ts` — `CameraCapture`: owns permission query, device selection,
  fallback attempts, the hidden video element, the frame worker, and the
  dispatch loop. One frame in flight at a time; due frames are dropped (not
  queued) while the worker is busy. Browser globals are injected via
  `CameraCaptureDeps` so the lifecycle is mock-testable.
- `frameWorker.ts` — Web Worker: draws posted ImageBitmaps onto its own
  OffscreenCanvas and acknowledges each frame. Day 3 plugs PoseLandmarker
  inference into the frame handler.

Frames are handed to the pose worker; no frames are recorded or uploaded.
