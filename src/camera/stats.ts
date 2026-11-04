/**
 * Frame-rate and frame-drop counters with elapsed-time timing.
 *
 * No assumptions about a fixed frame rate: FPS is measured from actual
 * frame timestamps over a trailing window. The clock is injectable so the
 * counters are unit-testable without a browser.
 */
import type { FrameStatsSnapshot } from "./types";

export class FrameStats {
  private frameTimes: number[] = [];
  private dropped = 0;
  private startTime: number | null = null;

  constructor(
    private readonly now: () => number = () => performance.now(),
    private readonly windowMs = 1000,
  ) {}

  /** Record a successfully dispatched frame. */
  recordFrame(): void {
    const t = this.now();
    if (this.startTime === null) {
      this.startTime = t;
    }
    this.frameTimes.push(t);
    const cutoff = t - this.windowMs;
    while (this.frameTimes.length > 0 && this.frameTimes[0] < cutoff) {
      this.frameTimes.shift();
    }
  }

  /** Record a frame skipped because the worker was still busy. */
  recordDrop(): void {
    this.dropped += 1;
  }

  snapshot(): FrameStatsSnapshot {
    const t = this.now();
    const times = this.frameTimes;
    let fps = 0;
    if (times.length >= 2) {
      const spanMs = Math.min(
        this.windowMs,
        times[times.length - 1] - times[0],
      );
      fps = spanMs > 0 ? ((times.length - 1) / spanMs) * 1000 : 0;
    }
    return {
      fps: Math.round(fps * 10) / 10,
      frames: times.length,
      dropped: this.dropped,
      elapsedMs: this.startTime === null ? 0 : Math.round(t - this.startTime),
    };
  }

  reset(): void {
    this.frameTimes = [];
    this.dropped = 0;
    this.startTime = null;
  }
}
