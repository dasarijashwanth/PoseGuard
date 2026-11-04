/**
 * Shared types for the camera pipeline (main thread + frame worker).
 *
 * Kept free of React and MediaPipe types so the worker bundle and the
 * main-thread code share one protocol definition.
 */

/** Camera permission state as reported by the Permissions API. */
export type CameraPermission = "prompt" | "granted" | "denied" | "unknown";

/** A single getUserMedia attempt. Tried in order until one succeeds. */
export interface ConstraintAttempt {
  label: string;
  constraints: MediaStreamConstraints;
}

export type CameraErrorCode =
  | "permission-denied"
  | "no-device"
  | "overconstrained"
  | "in-use"
  | "unknown";

/** Classified camera failure with plain-English guidance for the user. */
export interface CameraErrorInfo {
  code: CameraErrorCode;
  title: string;
  guidance: string;
}

/** Point-in-time view of the frame pipeline counters. */
export interface FrameStatsSnapshot {
  fps: number;
  frames: number;
  dropped: number;
  elapsedMs: number;
}

// ---------------------------------------------------------------------------
// Worker protocol. Day 3 extends the frame-ack message with pose landmarks.
// ---------------------------------------------------------------------------

/** Main -> worker: allocate the worker-side OffscreenCanvas. */
export interface WorkerInitMessage {
  type: "init";
  width: number;
  height: number;
}

/** Main -> worker: one video frame as a transferable ImageBitmap. */
export interface WorkerFrameMessage {
  type: "frame";
  id: number;
  bitmap: ImageBitmap;
  timestamp: number;
}

export type MainToWorkerMessage = WorkerInitMessage | WorkerFrameMessage;

/** Worker -> main: the worker is initialized and ready for frames. */
export interface WorkerReadyMessage {
  type: "ready";
}

/** Worker -> main: frame processed; the worker accepts the next frame. */
export interface WorkerFrameAckMessage {
  type: "frame-ack";
  id: number;
  processedMs: number;
}

export type WorkerToMainMessage = WorkerReadyMessage | WorkerFrameAckMessage;
