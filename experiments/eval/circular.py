"""Circular angle math.

Angles in PoseGuard live on a circle, not a number line. The shoulder
angle is reported in [-180, 180] degrees, so a baseline of +174 degrees
and a reading of -179 degrees are only 7 degrees apart, but plain
subtraction says -179 - 174 = -353 degrees. That phantom jump is the
bug that caused 6 false shoulder detections in Run 1.

The fix, here and in the web experiment, is to always measure the
*short way round the circle*:

    circ_dev(a, b) = atan2(sin(a - b), cos(a - b))   in degrees

sin/cos wrap the difference into [-180, 180] automatically, so the
answer is always the smallest signed rotation from b to a.
"""

import math


def wrap_angle(deg: float) -> float:
    """Wrap an angle into [-180, 180). 174 stays 174; 190 becomes -170."""
    return math.degrees(math.atan2(math.sin(math.radians(deg)),
                                   math.cos(math.radians(deg))))


def circ_dev(angle: float, baseline: float) -> float:
    """Signed deviation of `angle` from `baseline`, the short way round.

    Always in [-180, 180]. Example: circ_dev(-179, 174) == +7, not -353.
    """
    return wrap_angle(angle - baseline)


def circ_mean(angles) -> float:
    """Mean of angles, done on the circle.

    A plain average of +179 and -179 gives 0 (wrong: the two angles both
    point almost due left, mean should be ~180). Averaging the unit
    vectors instead gives the right answer.
    """
    angles = [a for a in angles if a is not None and not math.isnan(a)]
    if not angles:
        raise ValueError("circ_mean needs at least one angle")
    s = sum(math.sin(math.radians(a)) for a in angles) / len(angles)
    c = sum(math.cos(math.radians(a)) for a in angles) / len(angles)
    return math.degrees(math.atan2(s, c))
