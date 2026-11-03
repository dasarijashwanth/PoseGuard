import { describe, expect, it } from "vitest";
import {
  EMA,
  calculateAngle,
  headTiltAngle,
  shoulderTiltAngle,
} from "./angles.ts";

describe("calculateAngle", () => {
  it("returns 90 for a right angle", () => {
    const angle = calculateAngle({ x: 0, y: 0 }, { x: 1, y: 0 }, { x: 1, y: 1 });
    expect(angle).toBeCloseTo(90, 6);
  });

  it("returns 180 for a straight line", () => {
    const angle = calculateAngle({ x: 0, y: 0 }, { x: 1, y: 0 }, { x: 2, y: 0 });
    expect(angle).toBeCloseTo(180, 6);
  });

  it("returns 0 for a degenerate segment", () => {
    const angle = calculateAngle({ x: 1, y: 1 }, { x: 1, y: 1 }, { x: 2, y: 2 });
    expect(angle).toBe(0);
  });

  it("returns 135 for the interior angle of a 45-degree bend", () => {
    const angle = calculateAngle({ x: 0, y: 0 }, { x: 1, y: 0 }, { x: 2, y: 1 });
    expect(angle).toBeCloseTo(135, 6); // interior angle at b
  });
});

describe("headTiltAngle", () => {
  it("returns 0 for a level head relative to level shoulders", () => {
    const angle = headTiltAngle(
      { x: 0, y: 0 },
      { x: 10, y: 0 },
      { x: 0, y: 10 },
      { x: 10, y: 10 },
    );
    expect(angle).toBeCloseTo(0, 6);
  });

  it("returns the ear-line tilt when ears are tilted over level shoulders", () => {
    // Ear line rises 2 units over 10 units of run: atan2(2, 10) ≈ 11.31°.
    const angle = headTiltAngle(
      { x: 0, y: 0 },
      { x: 10, y: 2 },
      { x: 0, y: 10 },
      { x: 10, y: 10 },
    );
    expect(angle).toBeCloseTo(11.3099, 4);
  });

  it("returns 0 when either line is degenerate", () => {
    const angle = headTiltAngle(
      { x: 5, y: 5 },
      { x: 5, y: 5 },
      { x: 0, y: 10 },
      { x: 10, y: 10 },
    );
    expect(angle).toBe(0);
  });
});

describe("shoulderTiltAngle", () => {
  it("returns 0 for level shoulders", () => {
    const angle = shoulderTiltAngle({ x: 0, y: 5 }, { x: 10, y: 5 });
    expect(angle).toBeCloseTo(0, 6);
  });

  it("returns 45 for a 45-degree shoulder line", () => {
    const angle = shoulderTiltAngle({ x: 0, y: 0 }, { x: 10, y: 10 });
    expect(angle).toBeCloseTo(45, 6);
  });

  it("returns 0 for coincident shoulder points", () => {
    const angle = shoulderTiltAngle({ x: 3, y: 3 }, { x: 3, y: 3 });
    expect(angle).toBe(0);
  });
});

describe("EMA", () => {
  it("returns the first input unchanged", () => {
    const ema = new EMA(0.4);
    expect(ema.update(10)).toBe(10);
  });

  it("converges toward a constant input", () => {
    const ema = new EMA(0.5);
    let value = 0;
    for (let i = 0; i < 50; i++) {
      value = ema.update(10);
    }
    expect(value).toBeCloseTo(10, 6);
  });

  it("responds to a step change without jumping instantly", () => {
    const ema = new EMA(0.5);
    for (let i = 0; i < 20; i++) {
      ema.update(0);
    }
    const first = ema.update(10);
    // With alpha 0.5: 0.5*10 + 0.5*0 = 5, not 10.
    expect(first).toBeCloseTo(5, 6);
    // Eventually converges to the new level.
    let value = first;
    for (let i = 0; i < 50; i++) {
      value = ema.update(10);
    }
    expect(value).toBeCloseTo(10, 6);
  });

  it("rejects an invalid alpha", () => {
    expect(() => new EMA(0)).toThrow();
    expect(() => new EMA(1.5)).toThrow();
  });
});
