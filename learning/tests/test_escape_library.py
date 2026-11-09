"""The escape library E is defined once and played the same everywhere (council B3)."""

import numpy as np

from orion_learning import baselines, risk
from orion_learning import escape_library as EL


def test_E_is_named_manoeuvres_of_duration_speed_segments():
    names = [m.name for m in EL.E]
    assert len(set(names)) == len(names) >= 4
    for m in EL.E:
        assert m.segments and all(len(seg) == 3 and seg[0] > 0 for seg in m.segments)
        assert all(abs(v) <= 0.2 and abs(w) <= 0.8 for _, v, w in m.segments)
    assert EL.E[0].name == 'settle-reverse-wiggle-creep'
    assert EL.E[0].segments[0] == (1.0, 0.0, 0.0)


def test_every_user_plays_the_same_E():
    assert baselines.EscapeLibrary().library is EL.E
    assert baselines.B0Controller().escape.library is EL.E
    assert baselines.CriticGate().escape.library is EL.E
    assert risk.B0Controller is baselines.B0Controller          # critic labels under E


def test_manoeuvre_ticks_count_ticks_per_segment():
    m = EL.Manoeuvre('t', ((0.5, 0.1, 0.0), (0.01, -0.1, 0.2)))
    ticks = EL.manoeuvre_ticks(m, 0.05)
    assert ticks == ((0.1, 0.0),) * 10 + ((-0.1, 0.2),)       # short segments last one tick
    assert len(EL.manoeuvre_ticks(EL.E[0], 0.05)) == round(EL.E[0].duration_s / 0.05)
    assert np.isclose(EL.E[0].duration_s, 7.0)
