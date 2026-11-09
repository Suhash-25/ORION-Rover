"""
Clock-offset check between message header stamps and their receipt time (council M6).

The estimator runs on header stamps but ages its sources against the node clock, so a camera
PC or IMU driver whose clock is off by more than a source timeout (0.25 s for the external
pose) makes that source permanently stale, silently. This check warns instead: when
|stamp - receipt| exceeds max_skew_s continuously for hold_s on one input, it returns a
warning, at most once per warn_period_s per input. Plain Python, no rclpy.
"""

from __future__ import annotations

from typing import Optional

# Stamped inputs whose clock must agree with the node clock (terrain_node checks these).
CLOCK_CHECKED = ('ground_truth_pose', 'imu', 'joint_states')


class ClockSkewMonitor:
    """Per-input tracker of |header stamp - receipt time| (both in s, one per message)."""

    def __init__(self, max_skew_s: float = 0.1, hold_s: float = 1.0,
                 warn_period_s: float = 10.0):
        self.max_skew_s = max_skew_s
        self.hold_s = hold_s
        self.warn_period_s = warn_period_s
        self._since: dict[str, Optional[float]] = {}     # receipt time the skew began
        self._warned_at: dict[str, float] = {}

    def update(self, role: str, stamp: float, receipt: float) -> Optional[str]:
        """Record one stamped message; return a warning text when one is due, else None."""
        skew = stamp - receipt
        if abs(skew) <= self.max_skew_s:
            self._since[role] = None
            return None
        since = self._since.get(role)
        if since is None:
            self._since[role] = since = receipt
        if receipt - since < self.hold_s:
            return None
        last = self._warned_at.get(role)
        if last is not None and receipt - last < self.warn_period_s:
            return None
        self._warned_at[role] = receipt
        return (f'{role}: header stamps are {skew:+.3f} s off the receipt time for '
                f'{receipt - since:.1f} s (> {self.max_skew_s:g} s): clocks are not synchronised '
                '(chrony / use_sim_time); stamped inputs will be treated as stale or early')
