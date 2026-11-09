"""
Runtime vs learning code on the same scripted trace (council B2, B3).

* B0: PolicyCore in BASELINE (BaselineB0 + escape plans + attempt limit, FLAG_STUCK from the
  trace) publishes the same commands as orion_learning.baselines.B0Controller fed the same
  observations and stuck flags, followed by obs_spec.rate_limit (what the env executes).
* History: the adapter input the core hands a learnt controller is bit-identical to what
  orion_learning.history.History gives for the same observation trace.
"""

import numpy as np
from orion_learning import obs_spec
from orion_learning.baselines import B0Controller
from orion_learning.escape_library import E
from orion_learning.history import History
from orion_policy.controllers import BaselineB0
from orion_policy.controllers import Controller
from orion_policy.controllers import ControlOutput
from orion_policy.core import CoreConfig
from orion_policy.core import FLAG_STUCK
from orion_policy.core import MODE_BASELINE
from orion_policy.core import MODE_ESCAPE
from orion_policy.core import MODE_IDLE
from orion_policy.core import MODE_POLICY
from orion_policy.core import PolicyCore
from orion_policy.core import SafetyInput
from orion_policy.observation import ObservationBuilder
from orion_policy.sequences import escape_plan
import pytest

DT = 0.05
JOINTS = ['fl', 'fr', 'rl', 'rr']


class RecordingBuilder(ObservationBuilder):
    """Keeps every observation the core built."""

    def __init__(self):
        super().__init__(JOINTS, effort_max=3.0)
        self.built = []

    def build(self, now, prev_cmd):
        obs = super().build(now, prev_cmd)
        self.built.append(obs.copy())
        return obs


def tick(core, builder, t, slip, flags, speed=0.1):
    builder.update_joint_state(t, JOINTS, [1.0, 1.1, 0.9, 1.0], [0.5, 0.6, 0.4, 0.5])
    builder.update_imu(t, (0, 0, 0, 1), (0, 0, 0.01), (0.1, 0, 9.81))
    builder.update_terrain(t, speed, 0.9, slip, 0.2)
    # Limits far above anything B0 commands: the comparison is of the controllers, not the clamp.
    return core.tick(t, builder, SafetyInput(t, 0, flags, 5.0, 5.0, 'test'))


def scripted_trace():
    """(slip_side, stuck flag) per tick: cruise, slip, one escape, recovery, stuck for good."""
    trace = [((0.1, 0.0), False)] * 40 + [((0.5, 0.2), False)] * 40 + [((0.5, 0.2), True)]
    trace += [((0.0, 0.0), False)] * (len(escape_plan(DT, 0)) + 40)
    trace += [((0.6, 0.6), True)] * sum(len(escape_plan(DT, i)) + 1 for i in (1, 2, 3)) + \
        [((0.6, 0.6), True)] * 10
    return trace


def test_runtime_b0_matches_learning_b0_on_a_scripted_trace():
    builder = RecordingBuilder()
    core = PolicyCore(CoreConfig(dt=DT), BaselineB0(dt=DT))
    assert core.request_mode(MODE_BASELINE)[0]
    sim = B0Controller(1, dt=DT)
    sim_last = (0.0, 0.0)
    modes, gave_up_at = [], None
    t = 50.0
    for k, (slip, stuck) in enumerate(scripted_trace()):
        t += DT
        n_built = len(builder.built)
        res = tick(core, builder, t, slip, FLAG_STUCK if stuck else 0)
        modes.append(res.mode)
        if core.mode == MODE_IDLE:
            gave_up_at = k
            break
        assert len(builder.built) == n_built + 1
        action = sim.act(builder.built[-1], stuck=[stuck])[0]
        sim_last = obs_spec.rate_limit(sim_last, obs_spec.action_to_twist(action), DT)
        _, v, w = res.publish[-1]
        assert (v, w) == pytest.approx(sim_last, abs=1e-6), f'tick {k}'
    # Every manoeuvre was exercised: E[0], then E[1..3] back to back, then the give-up.
    assert modes.count(MODE_ESCAPE) == sum(len(escape_plan(DT, i)) for i in range(4))
    assert gave_up_at is not None and not sim.arbiter.gave_up[0]
    action = sim.act(builder.built[-1], stuck=[True])[0]
    assert sim.arbiter.gave_up[0] and not action.any()
    # The governor restarted at scale_after_escape after E[0], as in simulation.
    assert sim.governor.scale[0] == pytest.approx(core.baseline.governor.scale[0])
    assert len(E) >= 4


class RecordingPolicy(Controller):
    """Drives at a varying command so prev_cmd changes, and keeps each adapter input."""

    name = 'student'
    policy_sha = '0' * 16

    def __init__(self):
        self.seen = []

    def step(self, inp):
        self.seen.append(np.array(inp.history, copy=True))
        k = len(self.seen)
        return ControlOutput(0.1 + 0.01 * (k % 7), 0.05 * ((k % 5) - 2), 0.1,
                             np.zeros(obs_spec.LATENT_DIM, np.float32))


def test_history_adapter_input_is_bit_identical_to_learning_history():
    builder = RecordingBuilder()
    policy = RecordingPolicy()
    core = PolicyCore(CoreConfig(dt=DT), BaselineB0(dt=DT), policy)
    assert core.request_mode(MODE_POLICY)[0]
    t = 10.0
    for k in range(obs_spec.HISTORY_LEN + 15):
        t += DT
        tick(core, builder, t, (0.05 * (k % 3), 0.0), 0, speed=0.1 + 0.01 * k)
    hist = History(1)
    assert len(policy.seen) == len(builder.built) > obs_spec.HISTORY_LEN
    for obs, seen in zip(builder.built, policy.seen):
        expected = hist.push(obs[None])[0]
        assert seen.dtype == expected.dtype and seen.shape == expected.shape
        assert np.array_equal(seen, expected)
    # The newest row's action column is the command published on the tick before.
    assert policy.seen[-1][-1, obs_spec.OBS_DIM:].any()
