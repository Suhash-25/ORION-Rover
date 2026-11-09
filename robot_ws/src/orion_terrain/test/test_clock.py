"""Clock-offset warning (council M6): stamp vs receipt > 0.1 s for 1 s warns, throttled."""

from orion_learning import estimator as shared
from orion_terrain.clock import CLOCK_CHECKED, ClockSkewMonitor
from orion_terrain.estimator import stamp_seconds, TerrainEstimator
from orion_terrain.params import TerrainParams
import pytest


def feed(mon, role, offset, t0, seconds, hz=20.0):
    """Messages stamped `offset` s off their receipt time; returns the warnings issued."""
    out = []
    for k in range(int(round(seconds * hz))):
        receipt = t0 + k / hz
        w = mon.update(role, receipt + offset, receipt)
        if w:
            out.append((receipt, w))
    return out


def test_sustained_offset_warns_after_one_second():
    mon = ClockSkewMonitor()
    warnings = feed(mon, 'ground_truth_pose', 0.3, 0.0, 1.5)
    assert len(warnings) == 1
    t, text = warnings[0]
    assert t == 1.0 and 'ground_truth_pose' in text and '+0.300 s' in text


def test_small_or_brief_offsets_do_not_warn():
    mon = ClockSkewMonitor()
    assert not feed(mon, 'imu', 0.09, 0.0, 5.0, hz=100.0)
    assert not feed(mon, 'imu', -0.2, 5.0, 0.9, hz=100.0)        # 0.9 s, then good again
    assert not feed(mon, 'imu', 0.0, 5.9, 0.1, hz=100.0)
    assert not feed(mon, 'imu', -0.2, 6.0, 0.9, hz=100.0)        # the hold restarted


def test_warning_is_throttled_per_input():
    mon = ClockSkewMonitor(warn_period_s=10.0)
    warnings = feed(mon, 'joint_states', -0.5, 0.0, 25.0)
    assert [t for t, _ in warnings] == [1.0, 11.0, 21.0]
    assert len(feed(mon, 'imu', -0.5, 25.0, 2.0)) == 1           # independent per input


def test_node_checks_the_three_stamped_inputs_with_the_configured_limits():
    assert set(CLOCK_CHECKED) == {'ground_truth_pose', 'imu', 'joint_states'}
    p = TerrainParams()
    assert (p.clock_skew_max_s, p.clock_skew_hold_s) == (0.1, 1.0)


def _pose(stamp_s, x):
    sec = int(stamp_s)
    return {'header': {'stamp': {'sec': sec, 'nanosec': int(round((stamp_s - sec) * 1e9))}},
            'pose': {'position': {'x': x, 'y': 0.0, 'z': 0.0},
                     'orientation': {'x': 0.0, 'y': 0.0, 'z': 0.0, 'w': 1.0}}}


def test_time_since_fix_is_the_fix_age_on_the_stamp_clock():
    """
    TerrainState.time_since_fix = publish time - the fix's header stamp (council M7).

    The node hands the estimator header stamps, and its publish time (also the TerrainState
    header stamp) is on the same node clock. So a pose stamped at capture and received 0.3 s
    later is 0.3 s old when published; orion_safety then adds the TerrainState's own age. An
    unstamped pose falls back to its receipt time.
    """
    for stamped, expected in ((True, 0.3), (False, 0.0)):
        est = TerrainEstimator()
        receipt = 100.0
        for k in range(21):
            capture = 100.0 + k * 0.05
            receipt = capture + 0.3
            msg = _pose(capture if stamped else 0.0, 0.2 * k * 0.05)
            est.on_message(stamp_seconds(msg, receipt), 'ground_truth_pose', msg)
        e = est.estimate(receipt)
        assert e.time_since_fix == pytest.approx(expected, abs=1e-6), stamped
        assert e.confidence == float(shared.speed_confidence(e.time_since_fix,
                                                             e.body_speed_std))
