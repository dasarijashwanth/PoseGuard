import { useEffect, useRef, useState } from "react";
import { CameraCapture } from "./camera/capture";
import type {
  CameraPermission,
  FrameStatsSnapshot,
} from "./camera/types";

export default function App() {
  const captureRef = useRef<CameraCapture | null>(null);
  const [permission, setPermission] = useState<CameraPermission>("unknown");
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [stats, setStats] = useState<FrameStatsSnapshot | null>(null);

  useEffect(() => {
    return () => {
      captureRef.current?.stop();
    };
  }, []);

  async function startCamera(): Promise<void> {
    setError(null);
    const capture = new CameraCapture(
      {},
      {
        onPermission: setPermission,
        onError: (info) => setError(`${info.title}. ${info.guidance}`),
        onStats: setStats,
      },
    );
    captureRef.current = capture;
    try {
      await capture.start();
      setRunning(true);
    } catch {
      setRunning(false);
    }
  }

  function stopCamera(): void {
    captureRef.current?.stop();
    captureRef.current = null;
    setRunning(false);
  }

  return (
    <main className="app">
      <h1>PoseGuard</h1>
      <p>
        Day 2: camera pipeline. Frames stream to a Web Worker as ImageBitmaps;
        pose inference lands on Day 3.
      </p>
      <p>Camera permission: {permission}</p>
      {running ? (
        <button type="button" onClick={stopCamera}>
          Stop camera
        </button>
      ) : (
        <button type="button" onClick={startCamera}>
          Start camera
        </button>
      )}
      {error && <p role="alert">{error}</p>}
      {stats && (
        <p>
          FPS: {stats.fps} · frames: {stats.frames} · dropped: {stats.dropped}
        </p>
      )}
    </main>
  );
}
