"""Small causal filters shared by the runtime estimator and the calibration tool."""

from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np


class WheelAccel:
    """
    dw/dt per wheel (rad/s^2): finite difference of successive speeds, low-passed (tau).

    The SAME estimator is used at runtime and by fit_effort_model, so c2 means the same
    thing in both.
    """

    def __init__(self, tau_s: float):
        self.tau = tau_s
        self._t: Optional[float] = None
        self._w: Optional[np.ndarray] = None
        self.value: Optional[np.ndarray] = None

    def update(self, t: float, w: Sequence[float]) -> np.ndarray:
        """Add wheel speeds w (rad/s) at time t (s); return the filtered dw/dt."""
        w = np.asarray(w, dtype=np.float64)
        if self.value is None:
            self.value = np.zeros_like(w)
        if self._t is not None and t > self._t:
            dt = t - self._t
            raw = (w - self._w) / dt
            alpha = 1.0 - math.exp(-dt / self.tau) if self.tau > 0 else 1.0
            self.value = self.value + alpha * (raw - self.value)
        if self._t is None or t > self._t:
            self._t, self._w = t, w
        return self.value


class LowPass:
    """First-order low-pass y += (1 - exp(-dt / tau)) (x - y); starts at the first sample."""

    def __init__(self, tau_s: float):
        self.tau = tau_s
        self.value: Optional[np.ndarray] = None
        self._t: Optional[float] = None

    def update(self, t: float, x: Sequence[float]) -> np.ndarray:
        """Add sample x at time t (s); return the filtered value."""
        x = np.asarray(x, dtype=np.float64)
        if self.value is None or self._t is None:
            self.value = x.copy()
        elif t > self._t:
            alpha = 1.0 - math.exp(-(t - self._t) / self.tau) if self.tau > 0 else 1.0
            self.value = self.value + alpha * (x - self.value)
        self._t = t if self._t is None else max(self._t, t)
        return self.value

    def reset(self) -> None:
        self.value, self._t = None, None
