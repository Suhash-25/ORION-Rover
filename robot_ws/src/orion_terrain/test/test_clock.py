"""Clock-offset warning (council M6): stamp vs receipt > 0.1 s for 1 s warns, throttled."""

from orion_terrain.clock import CLOCK_CHECKED, ClockSkewMonitor
from orion_terrain.params import TerrainParams


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
