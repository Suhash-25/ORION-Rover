"""
Scripted motions: the escape library E, the critic probe, and the retreat along the entry path.

Each motion is a Plan: a fixed list of (v m/s, w rad/s) targets, one per control tick. Plans
are counted in ticks, not wall time, so they are deterministic under sim time and in tests.

The escape library is orion_learning.escape_library.E itself (council B3): the manoeuvres the
critic's rho is labelled under and that B0 runs in simulation are the ones the rover runs, and
an escape plays exactly escape_library.manoeuvre_ticks, as the simulator does.
"""

from __future__ import annotations

from collections import deque
from typing import Optional, Sequence

from orion_learning.escape_library import E, Manoeuvre, manoeuvre_ticks

# The escape library E (SPEC section 1: rho is defined "under the escape library E").
ESCAPE_LIBRARY = E

PROBE_SPEED = 0.1        # m/s, SPEC section 6: probe = 0.1 m/s straight for 1 s
PROBE_DURATION_S = 1.0
PROBE = Manoeuvre('probe', ((PROBE_DURATION_S, PROBE_SPEED, 0.0),))


class Plan:
    """A finite list of per-tick (v, w) targets, consumed with next_target()."""

    def __init__(self, targets: Sequence[tuple[float, float]], label: str):
        if not targets:
            raise ValueError(f'plan {label!r} is empty')
        self._targets = list(targets)
        self._i = 0
        self.label = label

    def next_target(self) -> tuple[float, float]:
        if self.done:
            raise RuntimeError(f'plan {self.label!r} is finished')
        target = self._targets[self._i]
        self._i += 1
        return target

    @property
    def done(self) -> bool:
        return self._i >= len(self._targets)

    def __len__(self) -> int:
        return len(self._targets)


def escape_plan(dt: float, index: int = 0) -> Plan:
    """Plan of manoeuvre E[index % len(E)] at control period dt (s)."""
    manoeuvre = ESCAPE_LIBRARY[index % len(ESCAPE_LIBRARY)]
    return Plan(manoeuvre_ticks(manoeuvre, dt), f'escape {manoeuvre.name}')


def probe_plan(dt: float) -> Plan:
    return Plan(manoeuvre_ticks(PROBE, dt), 'probe')


class EntryPath:
    """
    The commands the rover executed while driving normally, newest last.

    At most max_s seconds are kept; a retreat replays them back along the way it came in.
    """

    def __init__(self, dt: float, max_s: float = 2.0):
        self.dt = dt
        self._cmds: deque = deque(maxlen=max(1, round(max_s / dt)))

    def record(self, v: float, w: float) -> None:
        """Record one executed tick; standstill ticks carry no path and are skipped."""
        if abs(v) > 1e-3 or abs(w) > 1e-3:
            self._cmds.append((v, w))

    def clear(self) -> None:
        self._cmds.clear()

    def __len__(self) -> int:
        return len(self._cmds)

    def retreat_plan(self, max_speed: float) -> Optional[Plan]:
        """
        Replay the path backwards (reverse order, negated commands), or None if empty.

        Ticks faster than max_speed (m/s) are slowed by a factor f and stretched to 1/f
        ticks, so curvature and arc length are kept while |v| <= max_speed.
        """
        if not self._cmds:
            return None
        targets: list[tuple[float, float]] = []
        carry = 0.0
        for v, w in reversed(self._cmds):
            f = min(1.0, max_speed / abs(v)) if abs(v) > 1e-9 else 1.0
            carry += 1.0 / f
            n = int(carry + 1e-9)
            carry -= n
            targets += [(-v * f, -w * f)] * n
        return Plan(targets, 'retreat')
