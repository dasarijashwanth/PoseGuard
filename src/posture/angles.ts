/**
 * Pure posture geometry for PoseGuard.
 *
 * These functions take plain {x, y} landmark data and timestamps only.
 * They are intentionally independent of React and of MediaPipe types so they
 * can be unit-tested in isolation and reused in a Web Worker.
 */

/** A 2D landmark point in normalized image coordinates. */
export interface Point {
  x: number;
  y: number;
}

/**
 * Interior angle (0-180 degrees) at vertex `b` formed by segments b->a and b->c.
 * Returns 0 when either segment is degenerate (zero length).
 */
export function calculateAngle(a: Point, b: Point, c: Point): number {
  const v1x = a.x - b.x;
  const v1y = a.y - b.y;
  const v2x = c.x - b.x;
  const v2y = c.y - b.y;

  const dot = v1x * v2x + v1y * v2y;
  const mag1 = Math.hypot(v1x, v1y);
  const mag2 = Math.hypot(v2x, v2y);

  if (mag1 === 0 || mag2 === 0) {
    return 0;
  }

  // Clamp to avoid NaN from floating-point rounding.
  const cos = Math.min(1, Math.max(-1, dot / (mag1 * mag2)));
  return (Math.acos(cos) * 180) / Math.PI;
}

/**
 * Angle (0-180 degrees) between the ear line (earL -> earR) and the
 * shoulder line (shoulderL -> shoulderR). 0 means the ear line is parallel
 * to the shoulder line (level head relative to shoulders).
 */
export function headTiltAngle(
  earL: Point,
  earR: Point,
  shoulderL: Point,
  shoulderR: Point,
): number {
  const earDirX = earR.x - earL.x;
  const earDirY = earR.y - earL.y;
  const shDirX = shoulderR.x - shoulderL.x;
  const shDirY = shoulderR.y - shoulderL.y;

  const dot = earDirX * shDirX + earDirY * shDirY;
  const magEar = Math.hypot(earDirX, earDirY);
  const magSh = Math.hypot(shDirX, shDirY);

  if (magEar === 0 || magSh === 0) {
    return 0;
  }

  const cos = Math.min(1, Math.max(-1, dot / (magEar * magSh)));
  return (Math.acos(cos) * 180) / Math.PI;
}

/**
 * Angle (0-180 degrees) of the shoulder line (shoulderL -> shoulderR)
 * relative to the image horizontal. 0 means level shoulders.
 */
export function shoulderTiltAngle(shoulderL: Point, shoulderR: Point): number {
  const dx = shoulderR.x - shoulderL.x;
  const dy = shoulderR.y - shoulderL.y;

  const mag = Math.hypot(dx, dy);
  if (mag === 0) {
    return 0;
  }

  const cos = Math.min(1, Math.max(-1, dx / mag));
  return (Math.acos(cos) * 180) / Math.PI;
}

/**
 * Exponential moving average for smoothing scalar values such as angles.
 * The first update() call returns the input unchanged (no prior state).
 */
export class EMA {
  private value: number | null = null;

  constructor(private readonly alpha: number = 0.4) {
    if (alpha <= 0 || alpha > 1) {
      throw new Error("EMA alpha must be in (0, 1].");
    }
  }

  update(input: number): number {
    if (this.value === null) {
      this.value = input;
    } else {
      this.value = this.alpha * input + (1 - this.alpha) * this.value;
    }
    return this.value;
  }

  reset(): void {
    this.value = null;
  }
}
