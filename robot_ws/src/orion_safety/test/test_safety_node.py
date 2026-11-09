"""safety_node wiring that needs rclpy: the stalled-clock warning (council m11)."""

import time

from orion_safety.safety_node import clock_stalled
from orion_safety.safety_node import SafetyNode
import pytest
import rclpy
from rclpy.parameter import Parameter


def test_clock_stalled():
    assert clock_stalled(0, 5.1, 5.0)
    assert not clock_stalled(0, 4.9, 5.0)               # still starting up
    assert not clock_stalled(1, 60.0, 5.0)              # clock runs


@pytest.mark.parametrize('use_sim_time,warns', [(True, True), (False, False)])
def test_node_warns_while_sim_time_never_starts(use_sim_time, warns):
    rclpy.init()
    try:
        node = SafetyNode(parameter_overrides=[
            Parameter('use_sim_time', value=use_sim_time),
            Parameter('clock_warn_after_s', value=0.5)])
        end = time.monotonic() + 2.5
        while time.monotonic() < end:
            rclpy.spin_once(node, timeout_sec=0.05)
        assert (node.clock_warnings > 0) == warns
        node.destroy_node()
    finally:
        rclpy.shutdown()
