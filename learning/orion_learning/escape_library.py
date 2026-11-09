"""The escape library E: the one list of scripted escape manoeuvres (SPEC section 1, F1).

The entrapment risk rho is defined "under the escape library E", so the manoeuvres that label
rho in simulation (risk.py), that B0 and the critic gate run in simulation (baselines.py) and
that the rover runs (orion_policy.sequences builds its plans from E) must be the same object.
orion_policy imports E from here; a test checks the identity.

Each manoeuvre is a named tuple of (duration_s, v_mps, w_radps) segments. Successive escapes
of one stuck episode run the next manoeuvre (cyclic); after MAX_ESCAPE_ATTEMPTS escapes
without a clean driving step in between, the controller gives up and stops (on the rover
orion_safety's hard-stuck rule then ends the trial). |v| <= 0.2 m/s and |w| <= 0.8 rad/s;
on the rover the commands are still rate-limited and scaled to the SafetyStatus limits.

Order = order of use, so the rover's three attempts run the first three. Chosen on env data
(docs/engineering-log/agents/fix-shared.md, B3): from states where B0 had just been flagged
stuck, the settle-reverse-wiggle-creep sequence (the former orion_policy library) was best or
tied in every condition tested; forward-wiggle was worst in every condition and comes last.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Manoeuvre:
    """One escape manoeuvre: segments of (duration s, v m/s, w rad/s)."""

    name: str
    segments: tuple[tuple[float, float, float], ...]

    @property
    def duration_s(self) -> float:
        return sum(seg[0] for seg in self.segments)


E: tuple[Manoeuvre, ...] = (
    Manoeuvre('settle-reverse-wiggle-creep', (
        (1.0, 0.0, 0.0),        # stop: a wheel at rest no longer digs
        (2.0, -0.15, 0.0),      # reverse out along the ruts
        (0.5, 0.0, 0.6), (0.5, 0.0, -0.6), (0.5, 0.0, 0.6), (0.5, 0.0, -0.6),  # break rut walls
        (2.0, 0.12, 0.0),       # creep forward on the loosened ground
    )),
    Manoeuvre('straight-reverse', ((2.0, -0.20, 0.0),)),
    Manoeuvre('rocking', ((0.6, -0.15, 0.0), (0.6, 0.15, 0.0)) * 3),
    Manoeuvre('reverse-s-arc', ((1.5, -0.20, 0.6), (1.5, -0.20, -0.6))),
    Manoeuvre('forward-wiggle', ((0.75, 0.10, 0.8), (0.75, 0.10, -0.8)) * 2),
)

MAX_ESCAPE_ATTEMPTS = 3      # escapes in a row (no clean driving step between) before giving up


def manoeuvre_ticks(manoeuvre: Manoeuvre, dt: float) -> tuple[tuple[float, float], ...]:
    """Per-control-tick (v m/s, w rad/s) targets of a manoeuvre at period dt (s).

    Each segment lasts round(duration / dt) ticks, at least one. Counting ticks, not seconds,
    makes the plan deterministic under sim time, and the env and the rover play the same
    tick sequence.
    """
    out: list[tuple[float, float]] = []
    for duration, v, w in manoeuvre.segments:
        out += [(float(v), float(w))] * max(1, round(duration / dt))
    return tuple(out)
