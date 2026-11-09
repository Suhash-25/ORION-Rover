"""Smoke test: the rover spawns in an empty world, its controllers come up, and it drives.

Run with: colcon test --packages-select orion_sim_tests
"""

import math
import os
import sys
import time
import unittest

from geometry_msgs.msg import TwistStamped
import launch
import launch_testing.actions
import launch_testing.markers
from nav_msgs.msg import Odometry
import pytest
import rclpy
from sensor_msgs.msg import Imu
from sensor_msgs.msg import JointState
from sensor_msgs.msg import LaserScan
from tf2_ros import Buffer
from tf2_ros import TransformListener

sys.path.insert(0, os.path.dirname(__file__))
from sim_helpers import include_bringup, kill_leftover_gazebo, Probe  # noqa: E402, I100

# Through twist_mux like every command source (teleop slot, priority 200).
CMD_TOPIC = '/cmd_vel/teleop'
ODOM_TOPIC = '/diff_drive_controller/odom'


@pytest.mark.launch_test
@launch_testing.markers.keep_alive
def generate_test_description():
    return launch.LaunchDescription([
        include_bringup('gazebo.launch.py'),
        launch_testing.actions.ReadyToTest(),
    ])


class TestDrive(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        rclpy.init()
        cls.node = Probe()
        cls.node.watch(JointState, '/joint_states')
        cls.node.watch(Odometry, ODOM_TOPIC)
        cls.node.watch(LaserScan, '/scan', sensor_qos=True)
        cls.node.watch(Imu, '/imu/data', sensor_qos=True)
        cls.cmd_pub = cls.node.create_publisher(TwistStamped, CMD_TOPIC, 10)
        cls.tf_buffer = Buffer()
        cls.tf_listener = TransformListener(cls.tf_buffer, cls.node)

    @classmethod
    def tearDownClass(cls):
        cls.node.destroy_node()
        rclpy.shutdown()

    def test_1_controllers_come_up(self):
        ok = self.node.spin_until(
            lambda: '/joint_states' in self.node.latest and ODOM_TOPIC in self.node.latest,
            timeout_s=90)
        self.assertTrue(ok, 'joint_state_broadcaster / diff_drive_controller never published')
        js = self.node.latest['/joint_states']
        self.assertEqual(len(js.name), 4, f'expected 4 wheel joints, got {js.name}')
        self.assertEqual(len(js.effort), 4, 'effort state interface missing (needed for sinkage)')

    def test_2_lidar_publishes_above_5hz(self):
        self.node.spin_until(lambda: self.node.counts.get('/scan', 0) > 0, timeout_s=60)
        start = self.node.counts.get('/scan', 0)
        t0 = time.monotonic()
        self.node.spin_for(5.0)
        rate = (self.node.counts.get('/scan', 0) - start) / (time.monotonic() - t0)
        self.assertGreater(rate, 5.0, f'/scan at {rate:.1f} Hz')

    def test_2b_imu_reports_gravity(self):
        self.assertTrue(self.node.spin_until(lambda: '/imu/data' in self.node.latest, 60),
                        '/imu/data never published')
        acc = self.node.latest['/imu/data'].linear_acceleration
        self.assertAlmostEqual(acc.z, 9.8, delta=0.5, msg=f'IMU z accel {acc.z:.2f} at rest')
        self.assertEqual(self.node.latest['/imu/data'].header.frame_id, 'imu_link')

    def test_3_tf_chain(self):
        def has_tf():
            return self.tf_buffer.can_transform('odom', 'lidar_link', rclpy.time.Time())
        self.assertTrue(self.node.spin_until(has_tf, timeout_s=30),
                        'TF odom -> base_link -> lidar_link does not resolve')

    def test_4_drives_forward(self):
        self.assertTrue(self.node.spin_until(lambda: ODOM_TOPIC in self.node.latest, 60))
        x0 = self.node.latest[ODOM_TOPIC].pose.pose.position.x
        msg = TwistStamped()
        msg.twist.linear.x = 0.3
        t_end = time.monotonic() + 4.0
        while time.monotonic() < t_end:
            msg.header.stamp = self.node.get_clock().now().to_msg()
            self.cmd_pub.publish(msg)
            self.node.spin_for(0.1)
        self.node.spin_for(1.0)
        pose = self.node.latest[ODOM_TOPIC].pose.pose.position
        moved = pose.x - x0
        self.assertGreater(moved, 0.5, f'commanded 0.3 m/s for 4 s but odom moved {moved:.2f} m')
        self.assertLess(abs(pose.y), 0.2 + 0.1 * abs(moved), 'drifted sideways driving straight')
        self.assertFalse(math.isnan(moved))


@launch_testing.post_shutdown_test()
class TestShutdown(unittest.TestCase):

    def test_processes_exit(self, proc_info):
        kill_leftover_gazebo()
        # Gazebo is killed by launch at teardown; only check nothing crashed hard (segfault).
        for info in proc_info:
            self.assertNotEqual(info.returncode, -11, f'{info.process_name} segfaulted')
