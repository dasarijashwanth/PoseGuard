/**
 * Camera pipeline for PoseGuard (Day 2).
 *
 * Owns the full capture lifecycle: permission query, device selection,
 * ordered getUserMedia fallbacks, a hidden video element, the frame worker,
 * and the frame-dispatch loop. Exactly one frame is ever in flight: while
 * the worker is busy, due frames are counted as dropped instead of queued.
 * Frame pacing targets ~15 checks/sec; Day 3's PoseLandmarker consumes the
 * worker-side OffscreenCanvas without changing this class.
 *
 * Browser globals are injected through CameraCaptureDeps so the lifecycle
 * is testable with mocks.
 */
import FrameWorker from "./frameWorker?worker";
import {
  classifyCameraError,
  constraintFallbacks,
  preferredDeviceId,
} from "./constraints";
import { FrameStats } from "./stats";
import type {
  CameraErrorInfo,
  CameraPermission,
  FrameStatsSnapshot,
  MainToWorkerMessage,
  WorkerToMainMessage,
} from "./types";

export interface CameraCaptureEvents {
  onPermission?: (state: CameraPermission) => void;
  onError?: (info: CameraErrorInfo) => void;
  onStats?: (snapshot: FrameStatsSnapshot) => void;
}

export interface CameraCaptureDeps {
  mediaDevices?: Pick<MediaDevices, "enumerateDevices" | "getUserMedia">;
  queryPermission?: () => Promise<CameraPermission>;
  createWorker?: () => Worker;
  createVideoElement?: () => HTMLVideoElement;
  scheduleFrame?: (video: HTMLVideoElement, callback: () => void) => void;
  now?: () => number;
}

type VideoWithRvfc = HTMLVideoElement & {
  requestVideoFrameCallback?: (cb: () => void) => number;
};

async function defaultQueryPermission(): Promise<CameraPermission> {
  try {
    const status = await navigator.permissions.query({
      name: "camera",
    } as PermissionDescriptor);
    return status.state as CameraPermission;
  } catch {
    return "unknown";
  }
}

/** Prefer requestVideoFrameCallback (per presented video frame); fall back to rAF. */
function defaultScheduleFrame(
  video: HTMLVideoElement,
  callback: () => void,
): void {
  const v = video as VideoWithRvfc;
  if (typeof v.requestVideoFrameCallback === "function") {
    v.requestVideoFrameCallback(() => callback());
  } else {
    requestAnimationFrame(() => callback());
  }
}

export class CameraCapture {
  private readonly deps: Required<CameraCaptureDeps>;
  private readonly events: CameraCaptureEvents;
  private readonly stats: FrameStats;
  private readonly targetIntervalMs: number;

  private stream: MediaStream | null = null;
  private video: HTMLVideoElement | null = null;
  private worker: Worker | null = null;
  private running = false;
  private workerBusy = false;
  private frameId = 0;
  private lastSentAt = 0;

  constructor(
    deps: CameraCaptureDeps = {},
    events: CameraCaptureEvents = {},
    targetFps = 15,
  ) {
    this.deps = {
      mediaDevices: deps.mediaDevices ?? navigator.mediaDevices,
      queryPermission: deps.queryPermission ?? defaultQueryPermission,
      createWorker: deps.createWorker ?? (() => new FrameWorker()),
      createVideoElement:
        deps.createVideoElement ?? (() => document.createElement("video")),
      scheduleFrame: deps.scheduleFrame ?? defaultScheduleFrame,
      now: deps.now ?? (() => performance.now()),
    };
    this.events = events;
    this.stats = new FrameStats(this.deps.now);
    this.targetIntervalMs = 1000 / targetFps;
  }

  get isRunning(): boolean {
    return this.running;
  }

  async start(): Promise<void> {
    if (this.running) {
      return;
    }
    this.setPermission(await this.deps.queryPermission());

    const devices = await this.deps.mediaDevices.enumerateDevices();
    const deviceId = preferredDeviceId(devices);

    let stream: MediaStream | null = null;
    let lastErrorName = "UnknownError";
    for (const attempt of constraintFallbacks(deviceId)) {
      try {
        stream = await this.deps.mediaDevices.getUserMedia(attempt.constraints);
        break;
      } catch (error) {
        lastErrorName =
          error instanceof DOMException ? error.name : "UnknownError";
      }
    }
    if (!stream) {
      const info = classifyCameraError(lastErrorName);
      this.events.onError?.(info);
      throw new Error(info.title);
    }

    this.setPermission("granted");
    this.stream = stream;

    const video = this.deps.createVideoElement();
    video.muted = true;
    video.playsInline = true;
    video.style.display = "none";
    if (typeof document !== "undefined") {
      document.body.appendChild(video);
    }
    video.srcObject = stream;
    await video.play();
    this.video = video;

    const worker = this.deps.createWorker();
    worker.onmessage = (event: MessageEvent<WorkerToMainMessage>) => {
      this.handleWorkerMessage(event.data);
    };
    this.worker = worker;

    const init: MainToWorkerMessage = {
      type: "init",
      width: video.videoWidth || 640,
      height: video.videoHeight || 480,
    };
    worker.postMessage(init);

    this.running = true;
    this.frameId = 0;
    this.lastSentAt = 0;
    this.stats.reset();
    // The frame loop starts when the worker posts its 'ready' message.
  }

  stop(): void {
    this.running = false;
    this.workerBusy = false;
    if (this.stream) {
      for (const track of this.stream.getTracks()) {
        track.stop();
      }
      this.stream = null;
    }
    if (this.video) {
      this.video.pause();
      this.video.srcObject = null;
      this.video.remove();
      this.video = null;
    }
    if (this.worker) {
      this.worker.terminate();
      this.worker = null;
    }
  }

  private setPermission(state: CameraPermission): void {
    this.events.onPermission?.(state);
  }

  private handleWorkerMessage(message: WorkerToMainMessage): void {
    if (message.type === "ready") {
      this.pump();
      return;
    }
    // frame-ack: the worker is free for the next frame.
    this.workerBusy = false;
    this.events.onStats?.(this.stats.snapshot());
  }

  private pump(): void {
    if (!this.running || !this.video || !this.worker) {
      return;
    }
    this.deps.scheduleFrame(this.video, () => this.onVideoFrame());
  }

  private onVideoFrame(): void {
    if (!this.running || !this.video || !this.worker) {
      return;
    }
    const now = this.deps.now();
    if (now - this.lastSentAt < this.targetIntervalMs) {
      this.pump(); // Not due yet; not a drop.
      return;
    }
    if (this.workerBusy) {
      this.stats.recordDrop();
      this.events.onStats?.(this.stats.snapshot());
      this.pump();
      return;
    }
    this.workerBusy = true;
    this.lastSentAt = now;
    const id = ++this.frameId;
    const worker = this.worker;
    createImageBitmap(this.video)
      .then((bitmap) => {
        const message: MainToWorkerMessage = {
          type: "frame",
          id,
          bitmap,
          timestamp: now,
        };
        worker.postMessage(message, [bitmap]);
        this.stats.recordFrame();
      })
      .catch(() => {
        // Frame could not be captured; free the worker for the next one.
        this.workerBusy = false;
      })
      .finally(() => this.pump());
  }
}
