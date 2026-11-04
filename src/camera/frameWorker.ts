/**
 * PoseGuard frame worker (Day 2).
 *
 * Runs in a Web Worker. The main thread posts video frames as transferable
 * ImageBitmaps; the worker draws each frame onto its own OffscreenCanvas and
 * acknowledges it. Day 3 plugs MediaPipe PoseLandmarker inference into the
 * frame handler, consuming the OffscreenCanvas directly.
 */
import type {
  MainToWorkerMessage,
  WorkerToMainMessage,
} from "./types";

// Typed narrowly so this file compiles regardless of how the DOM and
// WebWorker libs declare the global `self`. Guarded so the module can be
// loaded outside a worker (e.g. by the test runner) without throwing.
const workerScope = (
  typeof self !== "undefined" ? self : null
) as unknown as {
  onmessage: ((event: MessageEvent<MainToWorkerMessage>) => void) | null;
  postMessage: (message: WorkerToMainMessage) => void;
} | null;

if (workerScope) {
  let canvas: OffscreenCanvas | null = null;
  let ctx: OffscreenCanvasRenderingContext2D | null = null;

  workerScope.onmessage = (event) => {
    const message = event.data;
    if (message.type === "init") {
      canvas = new OffscreenCanvas(message.width, message.height);
      ctx = canvas.getContext("2d");
      workerScope.postMessage({ type: "ready" });
      return;
    }
    // Frame message: draw it, release the bitmap, acknowledge.
    const started = performance.now();
    if (ctx && canvas) {
      ctx.drawImage(message.bitmap, 0, 0, canvas.width, canvas.height);
    }
    message.bitmap.close();
    workerScope.postMessage({
      type: "frame-ack",
      id: message.id,
      processedMs: performance.now() - started,
    });
  };
}
