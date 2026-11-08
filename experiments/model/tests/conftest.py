"""Fixtures for the Day 5 model tests.

Two fixtures: tiny synthetic windows built by hand (known answers for
feature tests), and the real Run 1 CSV loaded once per module (for
determinism and end-to-end tests). The real CSV is read-only test
data, never written to.
"""

import os

import numpy as np
import pandas as pd
import pytest

from experiments.eval.analysis import add_deviations
from experiments.eval.loader import load_session

RUN1_CSV = os.path.expanduser(
    "~/workspace/user/files/poseguard-validation-2026-11-03T19-24-30-709Z.csv")
RUN1_PRESENT = os.path.exists(RUN1_CSV)


@pytest.fixture
def flat_window():
    """A perfectly steady upright window: all deviations zero."""
    return np.zeros(10), np.zeros(10)


@pytest.fixture
def tilted_window():
    """A steady head tilt: head dev +6 deg, shoulder steady."""
    return np.full(10, 6.0), np.zeros(10)


@pytest.fixture(scope="module")
def run1_frames():
    if not RUN1_PRESENT:
        pytest.skip("Run 1 CSV not present")
    session = load_session(RUN1_CSV)
    return add_deviations(session), session


@pytest.fixture(scope="module")
def run1_session(run1_frames):
    return run1_frames[1]
