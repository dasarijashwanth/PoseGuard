"""Tests for circular angle math: the core of the Run 1 bug fix."""

import math

import pytest

from experiments.eval.circular import circ_dev, circ_mean, wrap_angle


def test_wrap_angle():
    assert wrap_angle(190) == pytest.approx(-170)
    assert wrap_angle(-190) == pytest.approx(170)
    assert wrap_angle(174) == pytest.approx(174)
    assert wrap_angle(360) == pytest.approx(0)
    assert wrap_angle(540) == pytest.approx(180)


def test_circ_dev_across_wrap_boundary():
    # The Run 1 bug: baseline 174, reading -179. Plain subtraction says
    # -353; the short way round the circle it is +7.
    assert circ_dev(-179, 174) == pytest.approx(7)
    assert circ_dev(179, -174) == pytest.approx(-7)
    assert circ_dev(10, -5) == pytest.approx(15)
    assert circ_dev(-5, 10) == pytest.approx(-15)


def test_circ_dev_is_always_within_180():
    for a in range(-720, 721, 7):
        for b in range(-720, 721, 11):
            assert -180 <= circ_dev(a, b) <= 180


def test_circ_mean_across_wrap_boundary():
    # Plain average of +179 and -179 is 0 (wrong). Both point almost
    # due left, so the mean should be ~180.
    assert circ_mean([179, -179]) == pytest.approx(180)
    assert circ_mean([170, 180, -170]) == pytest.approx(180)
    assert circ_mean([0, 10, -10]) == pytest.approx(0)


def test_circ_mean_needs_angles():
    with pytest.raises(ValueError):
        circ_mean([])
