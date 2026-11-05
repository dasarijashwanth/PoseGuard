"""PoseGuard evaluation harness (Day 3).

Reads the CSV exported by the webcam validation experiment and answers,
with numbers, whether each posture cue actually works:

- detection rate per cue (how many of the 5 holds were detected),
- false positives during ordinary work (must be zero to pass),
- threshold sensitivity (what each threshold would have decided),
- whether the landmark-loss test really ran.

The headline lesson of Run 1: the web experiment's deviation column used
plain subtraction (raw angle minus baseline). Near the +/-180 degree
boundary that produces phantom jumps of ~350 degrees. This harness
recomputes every deviation with circular math, so the numbers here are
the honest ones.

Run from the repo root:
    python -m experiments.eval.cli --csv <path/to/validation.csv>
"""

from .circular import circ_dev, circ_mean, wrap_angle
from .loader import SessionData, load_session

__all__ = ["circ_dev", "circ_mean", "wrap_angle", "SessionData", "load_session"]
