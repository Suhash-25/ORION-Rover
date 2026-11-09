"""Safety chain end to end: policy -> twist_mux -> wheels, and a latched e-stop that wins.

The policy keeps publishing on /cmd_vel/policy while /orion/estop goes true; the wheels must
stop within 0.5 s of sim time and orion_safety must report ESTOP until a deliberate reset.
Run with: colcon test --packages-select orion_sim_tests --ctest-args -R safety
"""

import os
import sys
import time
import unittest

from geometry_msgs.msg import TwistStamped
import launch
import launch_testing.actions
import launch_testing.markers
from nav_msgs.msg import Odometry
from orion_msgs.msg import SafetyStatus
from orion_msgs.srv import EstopReset
import pytest
import rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool

sys.path.insert(0, os.path.dirname(__file__))
from sim_helpers import include_bringup, kill_leftover_gazebo, Probe  # noqa: E402, I100

ODOM = '/diff_drive_controller/odom'
STATUS = '/orion/safety_status'
STOP_LIMIT_S = 0.5          # SPEC: e-stop zeroes wheel velocity within 0.5 s
STOPPED_RAD_S = 0.1         # |wheel speed| below this counts as stopped (0.012 m/s at r=0.12)


@pytest.mark.launch_test
@launch_testing.markers.keep_alive
def generate_test_description():
    return launch.LaunchDescription([
        include_bringup('gazebo.launch.py'),
        launch_testing.actions.ReadyToTest(),
    ])


def sim_seconds(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


class TestSafety(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        rclpy.init()
        cls.node = Probe('orion_safety_probe')
        cls.node.watch(Odometry, ODOM)
        cls.node.watch(SafetyStatus, STATUS)
        cls.wheels = []     # (sim stamp s, max |wheel speed| rad/s)
        cls.statuses = []   # (sim stamp s, state)
        cls.node.create_subscription(JointState, '/joint_states', cls._on_joints,
                                     qos_profile_sensor_data)
        cls.node.create_subscription(SafetyStatus, STATUS, cls._on_status, 10)
        cls.policy = cls.node.create_publisher(TwistStamped, '/cmd_vel/policy', 10)
        cls.estop = cls.node.create_publisher(Bool, '/orion/estop', 10)
        cls.reset = cls.node.create_client(EstopReset, '/orion/estop_reset')

    @classmethod
    def tearDownClass(cls):
        cls.node.destroy_node()
        rclpy.shutdown()

    @classmethod
    def _on_joints(cls, msg):
        if msg.velocity:
            cls.wheels.append((sim_seconds(msg.header.stamp), max(abs(v) for v in msg.velocity)))

    @classmethod
    def _on_status(cls, msg):
        cls.statuses.append((sim_seconds(msg.header.stamp), msg.state))

    def _state(self):
        msg = self.node.latest.get(STATUS)
        return None if msg is None else msg.state

    def _x(self):
        return self.node.latest[ODOM].pose.pose.position.x

    def _drive_policy(self, seconds, v=0.3):
        """Publish v on /cmd_vel/policy at 20 Hz for `seconds` of wall time."""
        msg = TwistStamped()
        msg.header.frame_id = 'base_link'
        msg.twist.linear.x = v
        t_end = time.monotonic() + seconds
        while time.monotonic() < t_end:
            msg.header.stamp = self.node.get_clock().now().to_msg()
            self.policy.publish(msg)
            self.node.spin_for(0.05)

    def _call_reset(self, phrase):
        self.assertTrue(self.reset.wait_for_service(timeout_sec=10), 'no /orion/estop_reset')
        fut = self.reset.call_async(EstopReset.Request(confirm=phrase))
        self.assertTrue(self.node.spin_until(fut.done, 10), 'reset call timed out')
        return fut.result()

    def test_1_safety_nominal_at_50hz(self):
        ok = self.node.spin_until(
            lambda: self._state() == SafetyStatus.STATE_NOMINAL and ODOM in self.node.latest,
            timeout_s=120)
        self.assertTrue(ok, f'safety never NOMINAL with odom up; last status '
                            f'{self.node.latest.get(STATUS)}')
        self.statuses.clear()
        self.node.spin_for(3.0)
        span = self.statuses[-1][0] - self.statuses[0][0]
        rate = (len(self.statuses) - 1) / span
        self.assertAlmostEqual(rate, 50.0, delta=10.0, msg=f'SafetyStatus at {rate:.1f} Hz sim')

    def test_2_policy_reaches_wheels_through_mux(self):
        x0 = self._x()
        self._drive_policy(3.0)
        moved = self._x() - x0
        self.assertGreater(moved, 0.4, f'0.3 m/s on /cmd_vel/policy for 3 s moved {moved:.2f} m')
        self.assertEqual(self._state(), SafetyStatus.STATE_NOMINAL)

    def test_3_estop_stops_wheels_while_policy_publishes(self):
        self._drive_policy(1.5)
        self.assertGreater(self.wheels[-1][1], 1.0, 'wheels not turning before the e-stop')
        t_estop = sim_seconds(self.node.get_clock().now().to_msg())
        wall_estop = time.monotonic()
        self.estop.publish(Bool(data=True))
        self._drive_policy(3.0)             # the policy does not know and keeps commanding

        after = [(t, w) for t, w in self.wheels if t >= t_estop]
        self.assertTrue(after, 'no /joint_states after the e-stop')
        moving = [t for t, w in after if w >= STOPPED_RAD_S]
        t_stopped = after[0][0] if not moving else next(
            (t for t, _ in after if t > moving[-1]), None)
        self.assertIsNotNone(t_stopped, 'wheels still turning 3 s after the e-stop')
        latency = t_stopped - t_estop
        print(f'e-stop -> wheels < {STOPPED_RAD_S} rad/s in {latency:.3f} s sim '
              f'(test wall clock {time.monotonic() - wall_estop:.1f} s for the window)')
        self.assertLess(latency, STOP_LIMIT_S, f'wheels stopped {latency:.3f} s after e-stop')

        self.assertEqual(self._state(), SafetyStatus.STATE_ESTOP)
        self.assertEqual(self.node.latest[STATUS].max_linear_speed, 0.0)
        x0 = self._x()
        self._drive_policy(2.0)
        self.assertLess(abs(self._x() - x0), 0.02, 'rover moved during ESTOP')

    def test_4_reset_is_deliberate(self):
        self.assertFalse(self._call_reset('clear estop').success, 'wrong phrase accepted')
        self.assertFalse(self._call_reset('CLEAR ESTOP').success,
                         'reset accepted while /orion/estop is still true')
        self.estop.publish(Bool(data=False))
        self.node.spin_for(0.5)
        self.assertEqual(self._state(), SafetyStatus.STATE_ESTOP, 'ESTOP must stay latched')
        result = self._call_reset('CLEAR ESTOP')
        self.assertTrue(result.success, result.message)
        self.assertTrue(self.node.spin_until(
            lambda: self._state() == SafetyStatus.STATE_NOMINAL, 5), 'not NOMINAL after reset')
        x0 = self._x()
        self._drive_policy(3.0)             # the mux lock is released too
        self.assertGreater(self._x() - x0, 0.3, 'policy cannot drive after the reset')


@launch_testing.post_shutdown_test()
class TestShutdown(unittest.TestCase):

    def test_processes_exit(self, proc_info):
        kill_leftover_gazebo()
        for info in proc_info:
            self.assertNotEqual(info.returncode, -11, f'{info.process_name} segfaulted')
