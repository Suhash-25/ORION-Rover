"""
Unit tests for every rule of docs/SPEC.md section 5, with synthetic inputs and a fake clock.

Each test drives SafetyMonitor at 50 Hz (DT = 0.02 s). Durations are asserted to within one
tick of the threshold.
"""

import math

from orion_safety.monitor import FLAG_FIRMWARE_FAULT
from orion_safety.monitor import FLAG_GEOFENCE
from orion_safety.monitor import FLAG_HARD_STUCK
from orion_safety.monitor import FLAG_HEARTBEAT
from orion_safety.monitor import FLAG_OVERCURRENT
from orion_safety.monitor import FLAG_STUCK
from orion_safety.monitor import FLAG_TILT
from orion_safety.monitor import SafetyConfig
from orion_safety.monitor import SafetyMonitor
from orion_safety.monitor import STATE_ESTOP
from orion_safety.monitor import STATE_HALT
from orion_safety.monitor import STATE_LIMITED
from orion_safety.monitor import STATE_NOMINAL
import pytest

DT = 0.02
TICK = DT + 1e-9
OK = 'CLEAR ESTOP'


class Rig:
    """Feeds the monitor healthy required inputs plus whatever a test sets, at 50 Hz."""

    def __init__(self, **cfg):
        self.m = SafetyMonitor(SafetyConfig(**cfg))
        self.k = 0
        self.feed_joint = True
        self.feed_imu = True
        self.tilt_deg = (0.0, 0.0)            # roll, pitch
        self.efforts = [0.0, 0.0, 0.0, 0.0]
        self.terrain = None                   # (speed, confidence, (slip_l, slip_r)[, t_fix])
        self.pose = None                      # external (x, y, yaw)
        self.odom = None                      # odom-frame (x, y, yaw)
        self.cmd = None                       # (v, w)
        self.health = None
        self.d = None

    @property
    def t(self):
        return self.k * DT

    def tick(self):
        self.k += 1
        t, m = self.t, self.m
        if self.feed_joint:
            m.on_joint_states(t, self.efforts)
        if self.feed_imu:
            m.on_imu(t, *(math.radians(a) for a in self.tilt_deg))
        if self.terrain is not None:
            m.on_terrain(t, *self.terrain)
        if self.pose is not None:
            m.on_external_pose(t, *self.pose)
        if self.odom is not None:
            m.on_odom_pose(t, *self.odom)
        if self.cmd is not None:
            m.on_command(t, *self.cmd)
        if self.health is not None:
            m.on_health(t, self.health)
        self.d = m.step(t)
        return self.d

    def run(self, seconds):
        for _ in range(int(round(seconds / DT))):
            self.tick()
        return self.d

    def time_until(self, predicate, max_s):
        """Seconds until predicate(decision) first holds (None if it never does)."""
        t0 = self.t
        for _ in range(int(round(max_s / DT))):
            if predicate(self.tick()):
                return self.t - t0
        return None

    def reset(self, phrase=OK):
        return self.m.request_reset(self.t, phrase)


def state_is(state):
    return lambda d: d.state == state


# ---------------------------------------------------------------- constants and nominal

def test_constants_match_orion_msgs():
    msg = pytest.importorskip('orion_msgs.msg')
    from orion_safety import monitor
    for name in ('STATE_NOMINAL', 'STATE_LIMITED', 'STATE_HALT', 'STATE_ESTOP', 'FLAG_TILT',
                 'FLAG_OVERCURRENT', 'FLAG_HEARTBEAT', 'FLAG_GEOFENCE', 'FLAG_STUCK',
                 'FLAG_FIRMWARE_FAULT', 'FLAG_HARD_STUCK'):
        assert getattr(monitor, name) == getattr(msg.SafetyStatus, name), name
    for bit, name in monitor.FW_FAULT_NAMES.items():
        assert getattr(msg.RoverHealth, 'FAULT_' + name) == bit


def test_nominal_with_healthy_inputs():
    d = Rig().run(1.0)
    assert (d.state, d.flags, d.reason) == (STATE_NOMINAL, 0, '')
    assert (d.max_linear_speed, d.max_angular_speed, d.max_current) == (0.5, 1.0, 12.0)
    assert not d.stops_robot


# ---------------------------------------------------------------- tilt

def test_tilt_limited_enter_and_exit():
    r = Rig()
    r.tilt_deg = (17.9, 0.0)
    assert r.run(1.0).state == STATE_NOMINAL
    r.tilt_deg = (0.0, -18.5)                # pitch, nose down: absolute value counts
    d = r.tick()
    assert d.state == STATE_LIMITED and d.flags == FLAG_TILT
    assert (d.max_linear_speed, d.max_angular_speed) == (0.2, 0.5)
    r.tilt_deg = (10.0, 0.0)
    assert r.tick().state == STATE_NOMINAL


def test_tilt_halt_needs_200ms():
    r = Rig()
    r.tilt_deg = (26.0, 0.0)
    r.run(0.16)
    assert r.d.state == STATE_LIMITED       # above 25 deg, not long enough yet
    r.tilt_deg = (20.0, 0.0)                # dip below 25 restarts the timer
    r.tick()
    r.tilt_deg = (26.0, 0.0)
    elapsed = r.time_until(state_is(STATE_HALT), 1.0)
    assert 0.2 - TICK <= elapsed <= 0.2 + TICK
    assert r.d.flags & FLAG_TILT and r.d.stops_robot
    assert (r.d.max_linear_speed, r.d.max_angular_speed, r.d.max_current) == (0, 0, 0)


def test_tilt_halt_latches_until_reset_when_level():
    r = Rig()
    r.tilt_deg = (0.0, 30.0)
    r.run(0.5)
    r.tilt_deg = (0.0, 0.0)
    assert r.run(5.0).state == STATE_HALT, 'tilt HALT must stay latched'
    r.tilt_deg = (0.0, 19.0)
    r.tick()
    ok, msg = r.reset()
    assert not ok and 'tilted' in msg
    r.tilt_deg = (0.0, 2.0)
    r.tick()
    assert r.reset()[0]
    assert r.tick().state == STATE_NOMINAL


def test_tilt_reset_refused_without_fresh_imu():
    r = Rig()
    r.tilt_deg = (30.0, 0.0)
    r.run(0.5)
    r.feed_imu = False
    r.run(1.0)
    ok, msg = r.reset()
    assert not ok and 'imu' in msg


# ---------------------------------------------------------------- current

def test_current_limited_after_2s_and_releases_after_2s():
    r = Rig()
    r.efforts = [1.0, -7.0, math.nan, 2.0]  # sign does not matter, NaN = no reading
    elapsed = r.time_until(state_is(STATE_LIMITED), 3.0)
    assert 2.0 - TICK <= elapsed <= 2.0 + TICK
    d = r.d
    assert d.flags == FLAG_OVERCURRENT and d.max_current == 6.0 and d.max_linear_speed == 0.3
    assert d.max_angular_speed == 1.0
    r.efforts = [3.0] * 4
    elapsed = r.time_until(state_is(STATE_NOMINAL), 3.0)
    assert 2.0 - TICK <= elapsed <= 2.0 + TICK


def test_current_brief_overload_is_ignored():
    r = Rig()
    r.efforts = [7.0] * 4
    r.run(1.9)
    r.efforts = [5.0] * 4
    r.tick()
    r.efforts = [7.0] * 4
    assert r.run(1.9).state == STATE_NOMINAL


def test_current_halt_after_300ms_then_5s_cooldown():
    r = Rig()
    r.efforts = [0.0, 13.0, 0.0, 0.0]
    r.run(0.26)
    assert r.d.state != STATE_HALT
    elapsed = r.time_until(state_is(STATE_HALT), 1.0)
    assert r.t == pytest.approx(0.30, abs=TICK) and elapsed is not None
    assert r.d.flags & FLAG_OVERCURRENT
    r.efforts = [0.0] * 4                   # wheels stop: current falls, cool-down runs
    elapsed = r.time_until(lambda d: d.state != STATE_HALT, 10.0)
    assert 5.0 - TICK <= elapsed <= 5.0 + TICK
    assert r.d.state == STATE_NOMINAL       # no reset needed


def test_current_cooldown_restarts_on_new_peak():
    r = Rig()
    r.efforts = [13.0] * 4
    r.run(0.4)
    r.efforts = [0.0] * 4
    r.run(4.0)
    r.efforts = [13.0] * 4                  # a single sample above I_peak restarts it
    r.tick()
    r.efforts = [0.0] * 4
    elapsed = r.time_until(lambda d: d.state != STATE_HALT, 10.0)
    assert 5.0 - TICK <= elapsed <= 5.0 + 2 * TICK


# ---------------------------------------------------------------- heartbeat

@pytest.mark.parametrize('attr,timeout', [('feed_joint', 0.3), ('feed_imu', 0.5)])
def test_heartbeat_halts_on_stale_input_and_recovers(attr, timeout):
    r = Rig()
    r.run(1.0)
    setattr(r, attr, False)
    elapsed = r.time_until(state_is(STATE_HALT), 2.0)
    assert timeout - TICK <= elapsed <= timeout + TICK
    assert r.d.flags == FLAG_HEARTBEAT and 'stale' in r.d.reason
    setattr(r, attr, True)
    assert r.tick().state == STATE_NOMINAL  # not latched


def test_heartbeat_grace_when_input_never_arrives():
    r = Rig()
    r.feed_joint = False
    elapsed = r.time_until(state_is(STATE_HALT), 20.0)
    assert 10.0 - TICK <= elapsed <= 10.0 + 2 * TICK
    assert 'since start' in r.d.reason and r.d.flags == FLAG_HEARTBEAT
    r.feed_joint = True                     # late controller: HALT clears by itself
    assert r.tick().state == STATE_NOMINAL


def test_heartbeat_slow_launch_never_halts():
    r = Rig()
    r.feed_joint = r.feed_imu = False

    def not_nominal(d):
        return d.state != STATE_NOMINAL
    assert r.time_until(not_nominal, 8.0) is None     # IMU bridge up at 8 s
    r.feed_imu = True
    assert r.time_until(not_nominal, 1.5) is None     # controllers up at 9.5 s
    r.feed_joint = True
    assert r.time_until(not_nominal, 20.0) is None


def test_heartbeat_input_lost_during_grace_still_halts():
    r = Rig()
    r.run(1.0)
    r.feed_imu = False                      # seen once, then lost: grace no longer applies
    assert r.run(0.6).state == STATE_HALT


# ---------------------------------------------------------------- geofence

def fence_rig(**cfg):
    r = Rig(**cfg)
    r.pose = (0.0, 0.0, 0.0)
    r.cmd = (0.0, 0.0)
    return r


def test_geofence_nominal_inside():
    r = fence_rig()
    r.pose = (0.80, 0.0, 0.0)               # keep-in edge is x = 1.0 - 0.15 = 0.85
    assert r.run(0.2).state == STATE_NOMINAL


def test_geofence_prediction_limits_and_clears():
    r = fence_rig()
    r.pose = (0.60, 0.0, 0.0)
    r.cmd = (0.3, 0.0)                       # predicted x = 0.90 > 0.85
    d = r.tick()
    assert d.state == STATE_LIMITED and d.flags == FLAG_GEOFENCE
    assert (d.max_linear_speed, d.max_angular_speed) == (0.2, 0.5)
    r.pose = (0.60, 0.0, math.pi)            # same speed heading inwards
    assert r.tick().state == STATE_NOMINAL
    r.pose, r.cmd = (0.0, 0.70, 0.0), (0.5, 1.0)  # arc ends at y = 0.93
    assert r.tick().state == STATE_LIMITED


def test_geofence_outside_halts_and_clears_when_back():
    r = fence_rig()
    r.pose = (0.0, -0.90, 0.0)
    d = r.tick()
    assert d.state == STATE_HALT and d.flags == FLAG_GEOFENCE
    r.pose = (2.0, 0.0, 0.0)                 # truly outside the polygon
    assert r.tick().state == STATE_HALT
    r.pose = (0.0, 0.0, 0.0)
    assert r.tick().state == STATE_NOMINAL


def test_geofence_odom_fallback_uses_start_pose_and_wider_margin():
    r = fence_rig()
    r.pose = None
    r.odom = (1.50, 0.0, 0.0)                # fence x = -0.75 + 1.50 = 0.75, 0.25 m in
    assert r.tick().state == STATE_NOMINAL
    r.odom = (1.56, 0.0, 0.0)                # fence x = 0.81: fine with the 0.15 m margin...
    d = r.tick()
    assert d.state == STATE_HALT and 'odom' in d.reason  # ...but not with 0.25 m for odom
    r.odom = (0.0, 0.0, math.pi / 2)         # yaw is composed with the start pose too
    r.cmd = (0.5, 0.0)
    assert r.tick().state == STATE_NOMINAL


def test_geofence_prefers_ground_truth_over_odom():
    r = fence_rig()
    r.odom = (5.0, 5.0, 0.0)
    assert r.run(0.2).state == STATE_NOMINAL


def test_geofence_inactive_without_pose_or_when_disabled():
    r = fence_rig()
    r.pose = None
    assert r.run(1.0).state == STATE_NOMINAL
    r = fence_rig(geofence_enabled=False)
    r.pose = (5.0, 0.0, 0.0)
    assert r.run(0.2).state == STATE_NOMINAL


def test_geofence_ignores_stale_pose():
    r = fence_rig()
    r.pose = (5.0, 0.0, 0.0)
    assert r.tick().state == STATE_HALT
    r.pose = None
    assert r.run(0.6).state == STATE_NOMINAL


def test_geofence_custom_polygon():
    # L-shaped keep-in: the notch at x > 1, y > 1 is outside although inside the bounding box.
    r = fence_rig(geofence_polygon=[0.0, 0.0, 2.0, 0.0, 2.0, 1.0, 1.0, 1.0, 1.0, 2.0, 0.0, 2.0])
    r.pose = (0.5, 0.5, 0.0)
    assert r.tick().state == STATE_NOMINAL
    r.pose = (1.5, 1.5, 0.0)
    assert r.tick().state == STATE_HALT


# ---------------------------------------------------------------- stuck

def stuck_rig(body_speed=0.0, confidence=0.9, slip=(0.0, 0.0), cmd=(0.3, 0.0)):
    r = Rig()
    r.terrain = (body_speed, confidence, slip)
    r.cmd = cmd
    return r


def test_stuck_by_speed_flags_after_3s_without_halting():
    r = stuck_rig()
    elapsed = r.time_until(lambda d: d.flags & FLAG_STUCK, 5.0)
    assert 3.0 - TICK <= elapsed <= 3.0 + TICK
    assert r.d.state == STATE_NOMINAL and r.d.max_linear_speed == 0.5
    r.terrain = (0.2, 0.9, (0.0, 0.0))        # moving again: flag clears
    assert not r.tick().flags & FLAG_STUCK


@pytest.mark.parametrize('kwargs', [
    {'confidence': 0.2},                    # body speed not trusted
    {'cmd': (0.05, 0.0)},                   # not commanded to move
    {'body_speed': 0.05},                   # it is moving
])
def test_stuck_by_speed_needs_all_conditions(kwargs):
    assert not stuck_rig(**kwargs).run(5.0).flags & FLAG_STUCK


@pytest.mark.parametrize('slip,cmd,stuck', [
    ((0.9, 0.1), (0.3, 0.0), True),          # left wheels spinning forward
    ((-0.1, -0.9), (-0.3, 0.0), True),       # reversing: spinning backwards is negative slip
    ((-0.9, 0.0), (0.3, 0.0), False),        # skidding forward is not spinning
    ((0.0, 0.85), (0.0, 1.0), True),         # turning in place: magnitude counts
    ((0.7, 0.7), (0.3, 0.0), False),
])
def test_stuck_by_slip(slip, cmd, stuck):
    r = stuck_rig(body_speed=0.3, slip=slip, cmd=cmd)
    r.run(2.9)
    assert not r.d.flags & FLAG_STUCK
    assert bool(r.run(0.2).flags & FLAG_STUCK) == stuck


def test_stuck_by_speed_with_fresh_fix_needs_no_confidence():
    # Council M7: stuck, wheels spinning at 0.5 m/s rim speed. The wheel/body disagreement
    # (0.5 m/s) drives the confidence to 0, but the camera fix is 0.05 s old.
    r = stuck_rig(confidence=0.0, slip=(1.0, 1.0), cmd=(0.5, 0.0))
    r.terrain += (0.05,)
    elapsed = r.time_until(lambda d: 'body 0.000 m/s (fresh fix)' in d.reason, 5.0)
    assert 3.0 - TICK <= elapsed <= 3.0 + TICK
    assert r.d.flags & FLAG_STUCK and r.d.state == STATE_NOMINAL


@pytest.mark.parametrize('t_fix', [2.0, math.inf, math.nan, 0.5])
def test_stuck_by_speed_disabled_with_stale_fix_and_low_confidence(t_fix):
    r = stuck_rig(confidence=0.0, cmd=(0.5, 0.0))      # slip 0: only the speed rule could fire
    r.terrain += (t_fix,)
    assert not r.run(5.0).flags & FLAG_STUCK
    r.terrain = r.terrain[:1] + (0.9,) + r.terrain[2:]  # confidence restores the rule
    assert r.run(3.1).flags & FLAG_STUCK


def test_fix_age_includes_the_age_of_the_terrain_message():
    # Fix 0.45 s old when sent; the message then stops arriving but stays valid for 0.5 s.
    r = stuck_rig(confidence=0.0, cmd=(0.5, 0.0))
    r.terrain += (0.45,)
    r.run(2.8)
    r.terrain = None
    assert not r.run(0.4).flags & FLAG_STUCK


def test_stuck_needs_terrain_state():
    r = Rig()
    r.cmd = (0.3, 0.0)
    assert r.run(5.0).flags == 0


# ---------------------------------------------------------------- hard-stuck

def test_hard_stuck_halts_after_20s_stuck_without_progress():
    r = stuck_rig()
    r.pose = (0.0, 0.0, 0.0)
    elapsed = r.time_until(state_is(STATE_HALT), 40.0)
    assert 23.0 - TICK <= elapsed <= 23.0 + 2 * TICK   # 3 s to flag stuck + 20 s window
    assert r.d.flags & FLAG_HARD_STUCK and 'ENTRAPPED' in r.d.reason
    r.cmd = (0.0, 0.0)
    assert r.run(5.0).state == STATE_HALT, 'hard-stuck must stay latched'
    assert r.reset()[0]
    assert r.tick().state == STATE_NOMINAL


def test_hard_stuck_not_when_creeping_forward():
    r = stuck_rig(body_speed=0.01)
    r.pose = (0.0, 0.0, 0.0)
    for _ in range(int(40 / DT)):
        r.pose = (r.pose[0] + 0.01 * DT, 0.0, 0.0)   # 0.2 m per 20 s, still "stuck"
        r.tick()
        assert r.d.state == STATE_NOMINAL
    assert r.d.flags & FLAG_STUCK


def test_hard_stuck_uses_integrated_body_speed_without_ground_truth():
    r = stuck_rig(body_speed=0.0)
    r.odom = (0.0, 0.0, 0.0)                 # wheel odometry must not count as progress
    elapsed = r.time_until(state_is(STATE_HALT), 40.0)
    assert 23.0 - TICK <= elapsed <= 23.0 + 2 * TICK
    r = stuck_rig(body_speed=0.02)           # 0.4 m per 20 s from the terrain estimator
    assert r.time_until(state_is(STATE_HALT), 40.0) is None


def test_hard_stuck_window_starts_with_the_stuck_episode():
    r = stuck_rig(cmd=(0.0, 0.0))
    r.pose = (0.0, 0.0, 0.0)
    r.run(30.0)                              # parked: no progress, but not trying either
    r.cmd = (0.3, 0.0)
    elapsed = r.time_until(state_is(STATE_HALT), 40.0)
    assert elapsed >= 23.0 - TICK


def test_hard_stuck_survives_escape_back_and_forth():
    r = stuck_rig()
    r.pose = (0.0, 0.0, 0.0)
    for i in range(int(30 / DT)):
        r.cmd = (0.3 if (i // 50) % 2 == 0 else -0.3, 0.0)
        r.tick()
    assert r.d.state == STATE_HALT and r.d.flags & FLAG_HARD_STUCK


# ---------------------------------------------------------------- firmware faults

def test_firmware_overcurrent_flag_only():
    r = Rig()
    r.health = 2
    d = r.run(0.2)
    assert d.state == STATE_NOMINAL and d.flags == FLAG_FIRMWARE_FAULT
    assert 'OVERCURRENT' in d.reason


def test_firmware_encoder_fault_halts_while_present():
    r = Rig()
    r.health = 4
    assert r.tick().state == STATE_HALT
    r.health = 0
    assert r.tick().state == STATE_NOMINAL


def test_firmware_health_optional_and_stale_ignored():
    r = Rig()
    r.health = 8
    assert r.tick().state == STATE_HALT
    r.health = None
    assert r.run(1.1).state == STATE_NOMINAL


# ---------------------------------------------------------------- e-stop

def test_estop_latches_and_needs_exact_phrase():
    r = Rig()
    r.run(0.5)
    r.m.on_estop(r.t, True)
    d = r.tick()
    assert d.state == STATE_ESTOP and d.stops_robot and d.max_linear_speed == 0.0
    assert 'e-stop' in d.reason
    r.m.on_estop(r.t, False)
    assert r.run(2.0).state == STATE_ESTOP, 'ESTOP must stay latched after /orion/estop false'
    for phrase in ('', 'clear estop', 'CLEAR ESTOP ', 'CLEAR'):
        ok, msg = r.reset(phrase)
        assert not ok and 'CLEAR ESTOP' in msg
    assert r.tick().state == STATE_ESTOP
    ok, msg = r.reset()
    assert ok and 'ESTOP' in msg
    assert r.tick().state == STATE_NOMINAL


def test_estop_reset_refused_while_input_still_true():
    r = Rig()
    r.m.on_estop(r.t, True)
    r.tick()
    ok, msg = r.reset()
    assert not ok and 'still true' in msg
    assert r.tick().state == STATE_ESTOP


def test_reset_with_nothing_latched():
    ok, msg = Rig().reset()
    assert ok and 'nothing' in msg


# ---------------------------------------------------------------- priority

def test_priority_estop_over_halt_over_limited():
    r = Rig()
    r.efforts = [7.0] * 4                    # current LIMITED after 2 s
    r.run(2.1)
    assert r.d.state == STATE_LIMITED
    r.tilt_deg = (30.0, 0.0)                 # tilt HALT after 0.2 s
    r.run(0.3)
    assert r.d.state == STATE_HALT
    assert r.d.flags == FLAG_TILT | FLAG_OVERCURRENT
    assert r.d.max_linear_speed == 0.0 and r.d.max_current == 0.0
    r.m.on_estop(r.t, True)
    d = r.tick()
    assert d.state == STATE_ESTOP and d.flags == FLAG_TILT | FLAG_OVERCURRENT
    assert d.reason.startswith('e-stop')     # most severe reason first
    assert d.reason.index('tilt') < d.reason.index('current > 6')


def test_priority_tightest_limit_wins():
    r = Rig()
    r.efforts = [7.0] * 4
    r.tilt_deg = (20.0, 0.0)
    d = r.run(2.1)
    assert d.state == STATE_LIMITED and d.flags == FLAG_TILT | FLAG_OVERCURRENT
    assert (d.max_linear_speed, d.max_angular_speed, d.max_current) == (0.2, 0.5, 6.0)


def test_stuck_flag_does_not_mask_limits():
    r = stuck_rig()
    r.tilt_deg = (20.0, 0.0)
    d = r.run(3.1)
    assert d.state == STATE_LIMITED and d.flags == FLAG_TILT | FLAG_STUCK


# ---------------------------------------------------------------- config validation

@pytest.mark.parametrize('bad', [
    {'tilt_limited_deg': 25.0},
    {'current_cont': 12.0},
    {'imu_timeout_s': -1.0},
    {'geofence_margin_m': math.nan},
    {'geofence_polygon': [0.0, 0.0, 1.0, 0.0]},
    {'geofence_polygon': [0.0, 0.0, 1.0, 0.0, 1.0]},
    {'odom_start_pose': [0.0, 0.0]},
    {'firmware_halt_mask': 256},
])
def test_config_rejects_inconsistent_values(bad):
    with pytest.raises(ValueError):
        SafetyConfig(**bad)
