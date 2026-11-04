import { afterEach, describe, expect, it, vi } from "vitest";
import { CameraCapture } from "./capture";
import type { CameraErrorInfo, CameraPermission } from "./types";

function fakeStream(): MediaStream {
  return {
    getTracks: () => [{ stop: vi.fn() }],
  } as unknown as MediaStream;
}

function makeMediaDevices(
  getUserMediaImpl: (c: MediaStreamConstraints) => Promise<MediaStream>,
) {
  const getUserMedia = vi.fn(getUserMediaImpl);
  const mediaDevices = {
    enumerateDevices: vi.fn(async () => [
      { kind: "videoinput", label: "USB Webcam", deviceId: "usb-1" },
      { kind: "videoinput", label: "FaceTime HD Camera", deviceId: "front-1" },
    ]),
    getUserMedia,
  } as unknown as Pick<MediaDevices, "enumerateDevices" | "getUserMedia">;
  return { mediaDevices, getUserMedia };
}

function makeVideo() {
  const el = {
    muted: false,
    playsInline: false,
    style: {} as Record<string, string>,
    srcObject: null,
    videoWidth: 640,
    videoHeight: 480,
    play: vi.fn(async () => undefined),
    pause: vi.fn(),
    remove: vi.fn(),
  };
  return el as unknown as HTMLVideoElement & {
    play: ReturnType<typeof vi.fn>;
  };
}

interface MockWorker {
  postMessage: ReturnType<typeof vi.fn>;
  terminate: ReturnType<typeof vi.fn>;
  onmessage: ((event: { data: unknown }) => void) | null;
}

function makeWorker(): MockWorker {
  return {
    postMessage: vi.fn(),
    terminate: vi.fn(),
    onmessage: null,
  };
}

function domError(name: string): DOMException {
  // Node 20+ exposes DOMException globally.
  return new DOMException("mock", name);
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("CameraCapture lifecycle", () => {
  it("tries fallbacks in order and starts on the first success", async () => {
    const { mediaDevices, getUserMedia } = makeMediaDevices(async (c) => {
      if (JSON.stringify(c).includes("1280")) {
        throw domError("OverconstrainedError");
      }
      return fakeStream();
    });
    const worker = makeWorker();
    const events = {
      onPermission: vi.fn(),
      onError: vi.fn(),
    };
    const capture = new CameraCapture(
      {
        mediaDevices,
        queryPermission: async () => "prompt" as CameraPermission,
        createWorker: () => worker as unknown as Worker,
        createVideoElement: makeVideo,
        scheduleFrame: () => {},
      },
      events,
    );

    await capture.start();

    expect(getUserMedia).toHaveBeenCalledTimes(2);
    expect(events.onError).not.toHaveBeenCalled();
    expect(events.onPermission).toHaveBeenCalledWith("granted");
    expect(capture.isRunning).toBe(true);
    capture.stop();
  });

  it("pins the preferred front camera on the first attempt", async () => {
    const { mediaDevices, getUserMedia } = makeMediaDevices(async () => fakeStream());
    const capture = new CameraCapture(
      {
        mediaDevices,
        queryPermission: async () => "granted" as CameraPermission,
        createWorker: () => makeWorker() as unknown as Worker,
        createVideoElement: makeVideo,
        scheduleFrame: () => {},
      },
      {},
    );
    await capture.start();
    const firstCall = getUserMedia.mock.calls[0][0];
    expect(JSON.stringify(firstCall)).toContain("front-1");
    capture.stop();
  });

  it("classifies the error and rejects when every attempt fails", async () => {
    const { mediaDevices, getUserMedia } = makeMediaDevices(async () => {
      throw domError("NotAllowedError");
    });
    let seen: { info: CameraErrorInfo | null } = { info: null };
    const capture = new CameraCapture(
      {
        mediaDevices,
        queryPermission: async () => "prompt" as CameraPermission,
        createWorker: () => makeWorker() as unknown as Worker,
        createVideoElement: makeVideo,
        scheduleFrame: () => {},
      },
      { onError: (info) => (seen.info = info) },
    );
    await expect(capture.start()).rejects.toThrow();
    expect(getUserMedia).toHaveBeenCalledTimes(3);
    expect(seen.info?.code).toBe("permission-denied");
    expect(capture.isRunning).toBe(false);
  });

  it("skips a frame while the worker is busy instead of queueing", async () => {
    vi.stubGlobal(
      "createImageBitmap",
      vi.fn(async () => ({ close: vi.fn() })),
    );
    let t = 0;
    const now = () => t;
    const { mediaDevices, getUserMedia: _getUserMedia } = makeMediaDevices(async () => fakeStream());
    const worker = makeWorker();
    const scheduled: Array<() => void> = [];
    const drops: number[] = [];
    const capture = new CameraCapture(
      {
        mediaDevices,
        queryPermission: async () => "granted" as CameraPermission,
        createWorker: () => worker as unknown as Worker,
        createVideoElement: makeVideo,
        scheduleFrame: (_video, cb) => scheduled.push(cb),
        now,
      },
      { onStats: (s) => drops.push(s.dropped) },
      15,
    );
    await capture.start();

    // Worker signals ready -> first frame scheduled.
    worker.onmessage?.({ data: { type: "ready" } });
    expect(scheduled).toHaveLength(1);

    // Fire the first frame; the worker is now busy.
    t = 100;
    scheduled.shift()?.();
    await new Promise((r) => setTimeout(r, 0)); // flush the bitmap promise chain
    expect(worker.postMessage).toHaveBeenCalledTimes(2); // init + frame

    // A second due frame while busy is dropped, not queued.
    t = 200;
    scheduled.shift()?.();
    expect(worker.postMessage).toHaveBeenCalledTimes(2);
    expect(drops[drops.length - 1]).toBe(1);

    // The ack frees the worker; the next frame goes through.
    worker.onmessage?.({ data: { type: "frame-ack", id: 1, processedMs: 5 } });
    t = 300;
    scheduled.shift()?.();
    await new Promise((r) => setTimeout(r, 0)); // flush the bitmap promise chain
    expect(worker.postMessage).toHaveBeenCalledTimes(3);

    capture.stop();
    expect(worker.terminate).toHaveBeenCalled();
  });

  it("stop() tears down the stream, video, and worker", async () => {
    const { mediaDevices, getUserMedia: _getUserMedia } = makeMediaDevices(async () => fakeStream());
    const worker = makeWorker();
    const video = makeVideo();
    const capture = new CameraCapture(
      {
        mediaDevices,
        queryPermission: async () => "granted" as CameraPermission,
        createWorker: () => worker as unknown as Worker,
        createVideoElement: () => video,
        scheduleFrame: () => {},
      },
      {},
    );
    await capture.start();
    capture.stop();
    expect(capture.isRunning).toBe(false);
    expect(worker.terminate).toHaveBeenCalled();
    expect(video.pause).toHaveBeenCalled();
    expect(video.remove).toHaveBeenCalled();
  });
});
