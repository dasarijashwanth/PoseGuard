import { describe, expect, it } from "vitest";
import { FrameStats } from "./stats";

/** Controllable clock for deterministic FPS assertions. */
function fakeClock() {
  let t = 0;
  return {
    now: () => t,
    advance: (ms: number) => {
      t += ms;
    },
  };
}

describe("FrameStats", () => {
  it("measures ~10 fps for frames 100ms apart", () => {
    const clock = fakeClock();
    const stats = new FrameStats(clock.now);
    for (let i = 0; i < 10; i++) {
      stats.recordFrame();
      clock.advance(100);
    }
    const snap = stats.snapshot();
    expect(snap.frames).toBe(10);
    expect(snap.fps).toBeCloseTo(10, 0);
  });

  it("counts drops separately from frames", () => {
    const clock = fakeClock();
    const stats = new FrameStats(clock.now);
    stats.recordFrame();
    stats.recordDrop();
    stats.recordDrop();
    const snap = stats.snapshot();
    expect(snap.frames).toBe(1);
    expect(snap.dropped).toBe(2);
  });

  it("reports 0 fps before two frames exist", () => {
    const stats = new FrameStats(fakeClock().now);
    stats.recordFrame();
    expect(stats.snapshot().fps).toBe(0);
  });

  it("tracks elapsed time from the first frame", () => {
    const clock = fakeClock();
    const stats = new FrameStats(clock.now);
    stats.recordFrame();
    clock.advance(2500);
    expect(stats.snapshot().elapsedMs).toBe(2500);
  });

  it("resets all counters", () => {
    const clock = fakeClock();
    const stats = new FrameStats(clock.now);
    stats.recordFrame();
    stats.recordDrop();
    stats.reset();
    const snap = stats.snapshot();
    expect(snap.frames).toBe(0);
    expect(snap.dropped).toBe(0);
    expect(snap.elapsedMs).toBe(0);
  });
});
