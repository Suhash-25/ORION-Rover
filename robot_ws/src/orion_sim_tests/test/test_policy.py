"""orion_policy drives the simulated rover in BASELINE (B0) mode.

The test publishes a nominal SafetyStatus and, when nothing else publishes it yet, a
synthetic TerrainState (orion_terrain is developed in parallel). orion_policy must then
drive: odometry advances for 10 s and PolicyStatus reports MODE_BASELINE.

TEMPORARY (integrator): twist_mux is not wired in this branch, so /cmd_vel/policy is remapped
straight to /diff_drive_controller/cmd_vel below. Remove the remapping once twist_mux is in
the bringup.

Run with: colcon test --packages-select orion_sim_tests --ctest-args -R test_policy
"""

import os
import sys
import time
import unittest

from geometry_msgs.msg import TwistStamped
import launch
from launch_ros.actions import Node
import launch_testing.actions
import launch_testing.markers
from nav_msgs.msg import Odometry
from orion_msgs.msg import PolicyStatus
from orion_msgs.msg import SafetyStatus
from orion_msgs.msg import TerrainState
import pytest
import rclpy
from rclpy.clock import Clock
from rclpy.clock import ClockType

sys.path.insert(0, os.path.dirname(__file__))
from sim_helpers import include_bringup, kill_leftover_gazebo, Probe  # noqa: E402, I100

CMD_TOPIC = '/diff_drive_controller/cmd_vel'
ODOM_TOPIC = '/diff_drive_controller/odom'
STATUS_TOPIC = '/orion/policy_status'


@pytest.mark.launch_test
@launch_testing.markers.keep_alive
def generate_test_description():
    policy = Node(
        package='orion_policy', executable='policy_node', name='orion_policy',
        output='screen',
        parameters=[{'initial_mode': 'BASELINE', 'use_sim_time': True}],
        # TEMPORARY until twist_mux is in the bringup (see module docstring).
        remappings=[('/cmd_vel/policy', CMD_TOPIC)],
    )
    return launch.LaunchDescription([
        include_bringup('gazebo.launch.py'),
        policy,
        launch_testing.actions.ReadyToTest(),
    ])


class TestPolicyBaseline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        rclpy.init()
        cls.node = Probe('orion_policy_sim_probe')
        cls.node.watch(Odometry, ODOM_TOPIC)
        cls.node.watch(PolicyStatus, STATUS_TOPIC)
        cls.node.watch(TwistStamped, CMD_TOPIC)
        cls.safety_pub = cls.node.create_publisher(SafetyStatus, '/orion/safety_status', 10)
        cls.terrain_pub = cls.node.create_publisher(TerrainState, '/orion/terrain_state', 10)
        cls.terrain_stub = None
        # Wall-clock timer: inputs keep flowing even before /clock arrives.
        cls.node.create_timer(0.05, cls._publish_inputs,
                              clock=Clock(clock_type=ClockType.STEADY_TIME))

    @classmethod
    def tearDownClass(cls):
        cls.node.destroy_node()
        rclpy.shutdown()

    @classmethod
    def _publish_inputs(cls):
        cls.safety_pub.publish(SafetyStatus(state=SafetyStatus.STATE_NOMINAL,
                                            max_linear_speed=0.5, max_angular_speed=1.0,
                                            max_current=6.0))
        if cls.terrain_stub is None:
            # Stub only if no orion_terrain (other than our own publisher) is running.
            cls.terrain_stub = cls.node.count_publishers('/orion/terrain_state') <= 1
        if cls.terrain_stub:
            cls.terrain_pub.publish(TerrainState(body_speed=0.3, confidence=0.9))

    def test_1_policy_comes_up_in_baseline(self):
        ok = self.node.spin_until(
            lambda: ODOM_TOPIC in self.node.latest and STATUS_TOPIC in self.node.latest,
            timeout_s=120)
        self.assertTrue(ok, 'diff_drive_controller odometry or PolicyStatus never published')
        self.assertEqual(self.node.latest[STATUS_TOPIC].mode, PolicyStatus.MODE_BASELINE)

    def test_2_drives_for_10_s(self):
        ok = self.node.spin_until(lambda: self.node.counts.get(CMD_TOPIC, 0) > 0, 60)
        self.assertTrue(ok, 'orion_policy never commanded the rover')
        x0 = self.node.latest[ODOM_TOPIC].pose.pose.position.x
        modes = set()
        t_end = time.monotonic() + 10.0
        while time.monotonic() < t_end:
            rclpy.spin_once(self.node, timeout_sec=0.05)
            modes.add(self.node.latest[STATUS_TOPIC].mode)
        moved = self.node.latest[ODOM_TOPIC].pose.pose.position.x - x0
        self.assertEqual(modes, {PolicyStatus.MODE_BASELINE})
        # B0 cruises at 0.3 m/s; even at a real-time factor of 0.5 that is 1.5 m in 10 s.
        self.assertGreater(moved, 1.0, f'odometry advanced only {moved:.2f} m in 10 s')
        cmd = self.node.latest[CMD_TOPIC].twist
        self.assertAlmostEqual(cmd.linear.x, 0.3, delta=0.01)
        self.assertAlmostEqual(cmd.angular.z, 0.0, delta=1e-6)


@launch_testing.post_shutdown_test()
class TestShutdown(unittest.TestCase):

    def test_processes_exit(self, proc_info):
        kill_leftover_gazebo()
        for info in proc_info:
            self.assertNotEqual(info.returncode, -11, f'{info.process_name} segfaulted')
