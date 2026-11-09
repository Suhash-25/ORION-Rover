"""B0 slip governor, escape library, probe and retreat plans."""

import numpy as np
from orion_policy.controllers import BaselineB0
from orion_policy.controllers import ControlInput
from orion_policy.sequences import EntryPath
from orion_policy.sequences import ESCAPE_LIBRARY
from orion_policy.sequences import escape_plan
from orion_policy.sequences import probe_plan
import pytest

DT = 0.05


def run_b0(b0, slip_side, seconds):
    out = None
    for _ in range(round(seconds / DT)):
        out = b0.step(ControlInput(obs=None, history=None, slip_side=slip_side, dt=DT))
    return out


def test_b0_cruises_at_0_3():
    out = run_b0(BaselineB0(), (0.1, 0.2), 1.0)
    assert out.v == pytest.approx(0.3) and out.w == 0.0


def test_b0_governor_cuts_20_percent_per_second_and_recovers_5():
    b0 = BaselineB0()
    assert run_b0(b0, (0.5, 0.0), 1.0).v == pytest.approx(0.3 * 0.8)
    assert run_b0(b0, (0.5, 0.0), 1.0).v == pytest.approx(0.3 * 0.6)
    assert run_b0(b0, (0.0, 0.0), 2.0).v == pytest.approx(0.3 * 0.7)


def test_b0_governor_uses_worse_side_and_strict_threshold():
    b0 = BaselineB0()
    assert run_b0(b0, (0.0, 0.31), 1.0).v == pytest.approx(0.24)
    b0.reset()
    assert run_b0(b0, (0.3, 0.3), 1.0).v == pytest.approx(0.3)   # not above 0.3


def test_b0_floor_keeps_stuck_detector_armed():
    out = run_b0(BaselineB0(), (0.9, 0.9), 10.0)
    assert out.v == pytest.approx(0.12) and out.v > 0.1


def test_escape_plan_follows_library():
    plan = escape_plan(DT)
    total = sum(s.duration_s for s in ESCAPE_LIBRARY)
    assert len(plan) == round(total / DT)
    targets = [plan.next_target() for _ in range(len(plan))]
    assert plan.done
    assert targets[0] == (0.0, 0.0)                 # settle first
    assert targets[round(1.0 / DT)] == (-0.15, 0.0)  # then reverse
    assert any(w > 0 for _, w in targets) and any(w < 0 for _, w in targets)
    assert targets[-1] == (0.12, 0.0)
    with pytest.raises(RuntimeError):
        plan.next_target()


def test_probe_is_0_1_mps_straight_for_1_s():
    plan = probe_plan(DT)
    assert [plan.next_target() for _ in range(len(plan))] == [(0.1, 0.0)] * 20


def test_retreat_replays_entry_path_backwards():
    path = EntryPath(DT, max_s=2.0)
    assert path.retreat_plan(0.2) is None
    path.record(0.0, 0.0)                           # standstill is not path
    for k in range(10):
        path.record(0.1, 0.01 * k)
    plan = path.retreat_plan(0.2)
    targets = [plan.next_target() for _ in range(len(plan))]
    assert targets == [(-0.1, -0.01 * k) for k in reversed(range(10))]


def test_retreat_slows_fast_segments_but_keeps_distance_and_curvature():
    path = EntryPath(DT, max_s=2.0)
    for _ in range(40):
        path.record(0.3, 0.6)
    plan = path.retreat_plan(0.2)
    targets = np.array([plan.next_target() for _ in range(len(plan))])
    assert np.all(np.abs(targets[:, 0]) <= 0.2 + 1e-12)
    np.testing.assert_allclose(targets[:, 1] / targets[:, 0], 2.0)
    assert targets[:, 0].sum() * DT == pytest.approx(-0.3 * 40 * DT, abs=0.2 * DT)


def test_entry_path_keeps_only_the_last_max_s():
    path = EntryPath(DT, max_s=1.0)
    for k in range(100):
        path.record(0.1 + k * 1e-3, 0.0)
    assert len(path) == 20
