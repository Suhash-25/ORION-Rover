"""
Time handling around the shared filters: sample-time bookkeeping for irregular messages.

The filter maths (gain 1 - exp(-dt / tau), hold where masked) is orion_learning.estimator's,
the same code the learning env runs at its fixed 0.05 s step.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from orion_learning import estimator as shared


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
            self.value = shared.lowpass(self.value, (w - self._w) / dt, dt, self.tau)
        if self._t is None or t > self._t:
            self._t, self._w = t, w
        return self.value


class LowPass:
    """First-order low-pass (orion_learning.estimator.lowpass); starts at the first sample."""

    def __init__(self, tau_s: float = shared.LOWPASS_TAU_S):
        self.tau = tau_s
        self.value: Optional[np.ndarray] = None
        self._t: Optional[float] = None

    def update(self, t: float, x: Sequence[float], mask=None) -> np.ndarray:
        """
        Add sample x at time t (s); return the filtered value.

        Entries where ``mask`` is False are held (they start at 0 if never updated). A sample
        not newer than the last one does not move the filter.
        """
        x = np.asarray(x, dtype=np.float64)
        if self.value is None or self._t is None:
            self.value = x if mask is None else np.where(mask, x, 0.0)
        else:
            self.value = shared.lowpass(self.value, x, t - self._t, self.tau, mask)
        self._t = t if self._t is None else max(self._t, t)
        return self.value
