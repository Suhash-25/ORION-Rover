"""Mode machine: transitions, channels, critic gating, escape, safety clamp and stale safety."""

import numpy as np
from orion_learning import obs_spec
from orion_policy.controllers import BaselineB0
from orion_policy.controllers import Controller
from orion_policy.controllers import ControlOutput
from orion_policy.core import CoreConfig
from orion_policy.core import ESCAPE_CHANNEL
from orion_policy.core import FLAG_STUCK
from orion_policy.core import MODE_BASELINE
from orion_policy.core import MODE_ESCAPE
from orion_policy.core import MODE_IDLE
from orion_policy.core import MODE_POLICY
from orion_policy.core import MODE_PROBE
from orion_policy.core import POLICY_CHANNEL
from orion_policy.core import PolicyCore
from orion_policy.core import SafetyInput
from orion_policy.observation import ObservationBuilder
from orion_policy.sequences import escape_plan
import pytest

DT = 0.05
JOINTS = ['fl', 'fr', 'rl', 'rr']
ESCAPE_TICKS = len(escape_plan(DT))


class FakePolicy(Controller):
    """Drives at (v, w) and reports whatever risk the test sets."""

    name = 'student'
    policy_sha = 'abcdef0123456789'

    def __init__(self, v=0.4, w=0.0):
        self.v, self.w, self.risk = v, w, 0.1
        self.seen_history = None

    def step(self, inp):
        self.seen_history = inp.history.copy()
        return ControlOutput(self.v, self.w, self.risk, np.full(8, 0.5, np.float32))


class Rig:
    """A core plus fresh inputs, advanced one 50 ms tick at a time."""

    def __init__(self, policy=None, **cfg):
        self.builder = ObservationBuilder(JOINTS, effort_max=3.0)
        self.policy = policy
        self.core = PolicyCore(CoreConfig(**cfg), BaselineB0(), policy)
        self.t = 100.0
        self.flags = 0
        self.state = 0
        self.limits = (0.5, 1.0)
        self.slip = (0.0, 0.0)
        self.safety_fresh = True

    def step(self):
        self.t += DT
        self.builder.update_joint_state(self.t, JOINTS, [1.0] * 4, [0.5] * 4)
        self.builder.update_imu(self.t, (0, 0, 0, 1), (0, 0, 0), (0, 0, 9.81))
        self.builder.update_terrain(self.t, 0.1, 0.9, self.slip, 0.0)
        safety = None
        if self.safety_fresh:
            self._safety = SafetyInput(self.t, self.state, self.flags, *self.limits, 'test')
        safety = getattr(self, '_safety', None)
        return self.core.tick(self.t, self.builder, safety)

    def run(self, n):
        return [self.step() for _ in range(n)]


def commands(results, channel=None):
    return [(c, v, w) for r in results for c, v, w in r.publish
            if channel is None or c == channel]


def test_idle_publishes_nothing():
    rig = Rig()
    assert rig.core.mode == MODE_IDLE
    assert commands(rig.run(5)) == []


def test_baseline_drives_on_policy_channel_with_rate_limit():
    rig = Rig()
    assert rig.core.request_mode(MODE_BASELINE)[0]
    res = rig.run(20)
    cmds = commands(res)
    assert all(c == POLICY_CHANNEL for c, _, _ in cmds)
    assert cmds[0][1] == pytest.approx(obs_spec.MAX_LINEAR_ACCEL * DT)
    assert cmds[-1][1] == pytest.approx(0.3)
    assert all(r.mode == MODE_BASELINE and r.policy_sha == '' for r in res)
    assert np.isnan(res[-1].risk) and not res[-1].latent.any()


def test_history_holds_obs_and_previous_executed_action():
    rig = Rig()
    rig.core.request_mode(MODE_BASELINE)
    rig.run(3)
    h = rig.core.history.buffer[0]
    # Row of tick 3: its obs carries prev_cmd = the command executed on tick 2, and so does the
    # action column (orion_learning.history convention).
    assert h[-1, -2] == pytest.approx(2 * obs_spec.MAX_LINEAR_ACCEL * DT / 0.5)
    assert h[-1, 0] == h[-1, -2] and h[-1, 1] == h[-1, -1]


def test_policy_mode_requires_a_model():
    ok, msg = Rig().core.request_mode(MODE_POLICY)
    assert not ok and 'no ONNX controller' in msg
    assert not Rig().core.request_mode(9)[0]


def test_stuck_flag_arms_escape_then_returns_to_baseline():
    rig = Rig()
    rig.core.request_mode(MODE_BASELINE)
    rig.run(10)
    rig.flags = FLAG_STUCK
    first = rig.step()
    assert first.mode == MODE_ESCAPE and first.publish[-1][0] == ESCAPE_CHANNEL
    assert first.publish[0] == (POLICY_CHANNEL, 0.0, 0.0)
    rig.flags = 0
    res = rig.run(ESCAPE_TICKS - 1)
    assert all(r.mode == MODE_ESCAPE for r in res)
    assert min(v for _, v, _ in commands(res)) == pytest.approx(-0.15)
    after = rig.step()
    assert rig.core.mode == MODE_BASELINE
    assert after.publish[0] == (ESCAPE_CHANNEL, 0.0, 0.0)   # release twist_mux at once
    assert after.publish[-1][0] == POLICY_CHANNEL


def test_escape_gives_up_to_idle_after_max_attempts():
    rig = Rig(max_escape_attempts=2)
    rig.core.request_mode(MODE_BASELINE)
    rig.flags = FLAG_STUCK
    res = rig.run(ESCAPE_TICKS + len(escape_plan(DT, 1)))      # E[0], then E[1]
    assert all(r.mode == MODE_ESCAPE for r in res)
    assert not rig.core.escapes_exhausted
    res = rig.step()
    assert rig.core.mode == MODE_IDLE and 'escape failed 2 times' in res.reason
    assert rig.core.escapes_exhausted
    assert commands(rig.run(3)) == []
    assert rig.core.escapes_exhausted, 'latched until the next SetMode'
    rig.core.request_mode(MODE_BASELINE)
    assert not rig.core.escapes_exhausted


def test_escapes_exhausted_is_not_set_by_idle_from_set_mode():
    rig = Rig()
    rig.core.request_mode(MODE_BASELINE)
    rig.run(5)
    rig.core.request_mode(MODE_IDLE)
    rig.run(3)
    assert rig.core.mode == MODE_IDLE and not rig.core.escapes_exhausted


def test_set_mode_escape_from_idle_returns_to_idle():
    rig = Rig()
    assert rig.core.request_mode(MODE_ESCAPE)[0]
    rig.run(ESCAPE_TICKS)
    assert rig.core.mode == MODE_IDLE
    assert rig.step().publish == [(ESCAPE_CHANNEL, 0.0, 0.0)]


def test_set_mode_idle_sends_one_zero():
    rig = Rig()
    rig.core.request_mode(MODE_BASELINE)
    rig.run(5)
    rig.core.request_mode(MODE_IDLE)
    assert commands(rig.run(3)) == [(POLICY_CHANNEL, 0.0, 0.0)]


def test_safety_limits_clamp_every_command():
    rig = Rig(policy=FakePolicy(v=0.5, w=1.0))
    rig.core.request_mode(MODE_POLICY)
    rig.limits = (0.1, 0.1)
    cmds = commands(rig.run(40))
    assert max(abs(v) for _, v, _ in cmds) <= 0.1 + 1e-9
    assert max(abs(w) for _, _, w in cmds) <= 0.1 + 1e-9


def test_stale_safety_publishes_zero_and_says_why():
    rig = Rig()
    rig.core.request_mode(MODE_BASELINE)
    rig.run(20)
    rig.safety_fresh = False
    res = rig.run(12)
    assert res[8].publish[-1][1] > 0.0                 # 0.45 s old: still fine
    assert res[10].publish == [(POLICY_CHANNEL, 0.0, 0.0)]
    assert 'SafetyStatus is 0.55 s old' in res[10].reason


def test_no_safety_ever_publishes_zero():
    rig = Rig()
    rig.core.request_mode(MODE_BASELINE)
    rig.safety_fresh = False
    res = rig.step()
    assert res.publish == [(POLICY_CHANNEL, 0.0, 0.0)] and 'no SafetyStatus' in res.reason


def test_halt_publishes_zero_and_aborts_escape():
    rig = Rig()
    rig.core.request_mode(MODE_BASELINE)
    rig.flags = FLAG_STUCK
    rig.run(3)
    assert rig.core.mode == MODE_ESCAPE
    rig.state = 2
    res = rig.step()
    assert rig.core.mode == MODE_BASELINE
    assert (ESCAPE_CHANNEL, 0.0, 0.0) in res.publish
    assert res.publish[-1] == (POLICY_CHANNEL, 0.0, 0.0) and 'safety state 2' in res.reason


def test_stale_input_publishes_zero():
    rig = Rig()
    rig.core.request_mode(MODE_BASELINE)
    rig.step()
    res = rig.core.tick(rig.t + 0.4, rig.builder, rig._safety)
    assert res.publish == [(POLICY_CHANNEL, 0.0, 0.0)] and 'joint_states' in res.reason


def test_policy_low_risk_explores():
    pol = FakePolicy(v=0.4)
    rig = Rig(policy=pol)
    assert rig.core.request_mode(MODE_POLICY)[0]
    res = rig.run(20)
    assert commands(res)[-1][1] == pytest.approx(0.4)
    assert res[-1].risk == pytest.approx(0.1) and res[-1].policy_sha == 'abcdef0123456789'
    assert res[-1].latent[0] == pytest.approx(0.5)


def test_caution_band_probes_then_caps_speed():
    pol = FakePolicy(v=0.4)
    rig = Rig(policy=pol)
    rig.core.request_mode(MODE_POLICY)
    rig.run(20)
    pol.risk = 0.4
    res = rig.run(20)
    assert all(r.mode == MODE_PROBE for r in res)
    probe = commands(res, ESCAPE_CHANNEL)
    assert len(probe) == 20 and probe[-1][1] == pytest.approx(0.1) and probe[-1][2] == 0.0
    res = rig.run(30)
    assert rig.core.mode == MODE_POLICY
    assert res[0].publish[0] == (ESCAPE_CHANNEL, 0.0, 0.0)
    assert max(v for _, v, _ in commands(res, POLICY_CHANNEL)) == pytest.approx(0.2)


def test_retreat_band_backs_along_entry_path():
    pol = FakePolicy(v=0.2, w=0.1)
    rig = Rig(policy=pol)
    rig.core.request_mode(MODE_POLICY)
    rig.run(60)
    pol.risk = 0.6
    res = rig.run(10)
    assert all(r.mode == MODE_POLICY for r in res)
    cmds = commands(res, POLICY_CHANNEL)
    assert cmds[-1][1] == pytest.approx(-0.2) and cmds[-1][2] == pytest.approx(-0.1)


def test_retreat_without_entry_path_escapes():
    pol = FakePolicy()
    pol.risk = 0.6
    rig = Rig(policy=pol)
    rig.core.request_mode(MODE_POLICY)
    res = rig.step()
    assert res.mode == MODE_ESCAPE and 'no entry path' in res.reason


def test_high_risk_escapes_and_returns_to_policy():
    pol = FakePolicy()
    rig = Rig(policy=pol)
    rig.core.request_mode(MODE_POLICY)
    rig.run(5)
    pol.risk = 0.8
    assert rig.step().mode == MODE_ESCAPE
    pol.risk = 0.1
    rig.run(ESCAPE_TICKS)
    assert rig.core.mode == MODE_POLICY


def test_gating_off_is_b3():
    pol = FakePolicy(v=0.4)
    rig = Rig(policy=pol, critic_gating=False)
    rig.core.request_mode(MODE_POLICY)
    pol.risk = 0.9
    res = rig.run(20)
    assert all(r.mode == MODE_POLICY for r in res)
    assert commands(res)[-1][1] == pytest.approx(0.4)


def test_adapter_sees_current_obs_and_executed_command_pushed_before_inference():
    """Council B2: the newest row is [obs_t, a_(t-1) executed], zero rows before it."""
    pol = FakePolicy()
    rig = Rig(policy=pol)
    rig.core.request_mode(MODE_POLICY)
    first = rig.step()
    row = pol.seen_history[-1]
    assert row[:obs_spec.OBS_DIM].any() and not pol.seen_history[:-1].any()
    assert not row[obs_spec.OBS_DIM:].any()                # nothing executed before t = 0
    rig.step()
    v0, w0 = commands([first])[-1][1:]
    np.testing.assert_allclose(pol.seen_history[-1][obs_spec.OBS_DIM:], [v0 / 0.5, w0 / 1.0],
                               rtol=1e-6)
    assert pol.seen_history[-2].any() and not pol.seen_history[-3].any()


def test_history_is_cleared_when_a_driving_mode_is_selected():
    pol = FakePolicy()
    rig = Rig(policy=pol)
    rig.core.request_mode(MODE_POLICY)
    rig.run(5)
    rig.core.request_mode(MODE_POLICY)
    rig.step()
    assert not pol.seen_history[:-1].any()
