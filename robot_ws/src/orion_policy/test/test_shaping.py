"""Gating bands, rate limiting and the safety clamp."""

import math

from orion_learning import obs_spec
from orion_policy.shaping import Band
from orion_policy.shaping import classify
from orion_policy.shaping import CommandShaper
from orion_policy.shaping import scale_to_limits
import pytest


@pytest.mark.parametrize('rho,band', [
    (0.0, Band.EXPLORE), (0.2999, Band.EXPLORE), (0.3, Band.CAUTION), (0.4999, Band.CAUTION),
    (0.5, Band.RETREAT), (0.7, Band.RETREAT), (0.7001, Band.ESCAPE), (1.0, Band.ESCAPE),
    (math.nan, Band.EXPLORE)])
def test_gating_thresholds(rho, band):
    assert classify(rho) == band


def test_scale_to_limits_keeps_turn_radius():
    v, w = scale_to_limits(0.4, 0.8, 0.2, 1.0)
    assert (v, w) == pytest.approx((0.2, 0.4))
    v, w = scale_to_limits(-0.1, -0.9, 0.5, 0.3)
    assert (v, w) == pytest.approx((-0.1 / 3, -0.3))
    assert scale_to_limits(0.1, 0.1, 0.5, 1.0) == (0.1, 0.1)


@pytest.mark.parametrize('bad', [0.0, -1.0, math.nan])
def test_zero_negative_or_nan_limit_stops(bad):
    assert scale_to_limits(0.3, 0.0, bad, 1.0) == (0.0, 0.0)


def test_shaper_rate_limits_then_clamps():
    s = CommandShaper(0.05)
    v, w = s.shape((0.5, 1.0), 0.5, 1.0)
    assert (v, w) == pytest.approx((obs_spec.MAX_LINEAR_ACCEL * 0.05,
                                    obs_spec.MAX_ANGULAR_ACCEL * 0.05))
    for _ in range(20):
        v, w = s.shape((0.5, 0.0), 0.5, 1.0)
    assert v == pytest.approx(0.5) and w == pytest.approx(0.0)
    v, w = s.shape((0.5, 0.0), 0.2, 1.0)          # safety LIMITED wins at once
    assert v == pytest.approx(0.2)
    assert s.hold() == (0.0, 0.0) and s.last == (0.0, 0.0)
