"""Critic gating bands and command shaping (rate limit, then safety limits)."""

from __future__ import annotations

from enum import IntEnum
import math
from typing import Sequence

from orion_learning import obs_spec


class Band(IntEnum):
    """What the critic's entrapment risk rho allows (SPEC section 6, runtime gating)."""

    EXPLORE = 0   # rho < 0.3: run the policy
    CAUTION = 1   # 0.3 <= rho < 0.5: cap v at 0.2 m/s and probe
    RETREAT = 2   # 0.5 <= rho <= 0.7: retreat along the entry path
    ESCAPE = 3    # rho > 0.7: run the escape library


def classify(rho: float, thresholds: Sequence[float] = (0.3, 0.5, 0.7)) -> Band:
    """Gating band for rho in [0, 1]. NaN (no critic) means no gating: EXPLORE."""
    caution, retreat, escape = thresholds
    if math.isnan(rho) or rho < caution:
        return Band.EXPLORE
    if rho < retreat:
        return Band.CAUTION
    if rho <= escape:
        return Band.RETREAT
    return Band.ESCAPE


def scale_to_limits(v: float, w: float, max_v: float, max_w: float) -> tuple[float, float]:
    """
    Scale (v, w) by one common factor so |v| <= max_v and |w| <= max_w.

    A common factor keeps the turn radius v / w, so a limited rover still follows the same
    arc. Negative or NaN limits are treated as 0 (stop).
    """
    max_v = max_v if max_v > 0.0 else 0.0   # NaN compares False -> 0
    max_w = max_w if max_w > 0.0 else 0.0
    f = 1.0
    if abs(v) > max_v:
        f = min(f, max_v / abs(v))
    if abs(w) > max_w:
        f = min(f, max_w / abs(w))
    return v * f, w * f


class CommandShaper:
    """
    Rate-limit with obs_spec.rate_limit, then apply the SafetyStatus limits.

    last is the command actually published (v m/s, w rad/s); it is the next tick's rate
    limit origin and the observation's prev_cmd.
    """

    def __init__(self, dt: float):
        self.dt = dt
        self.last = (0.0, 0.0)

    def shape(self, target: Sequence[float], max_v: float, max_w: float) -> tuple[float, float]:
        v, w = obs_spec.rate_limit(self.last, target, self.dt)
        self.last = scale_to_limits(v, w, max_v, max_w)
        return self.last

    def hold(self) -> tuple[float, float]:
        """Immediate zero (safety stop: not rate limited)."""
        self.last = (0.0, 0.0)
        return self.last
