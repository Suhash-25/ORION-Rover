"""Smoke test: Nav2 localizes on the saved map and drives the rover to a goal.

This is the test that would have caught the Nav2 Twist/TwistStamped topic mismatch.
"""

import math
import os
import sys
import time
import unittest

from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import PoseWithCovarianceStamped
import launch
import launch_testing.actions
import launch_testing.markers
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import Odometry
import pytest
import rclpy
from rclpy.action import ActionClient
from std_srvs.srv import Trigger
from tf2_ros import Buffer
from tf2_ros import TransformListener

sys.path.insert(0, os.path.dirname(__file__))
from sim_helpers import include_bringup, kill_leftover_gazebo, Probe  # noqa: E402, I100

GOAL_X, GOAL_Y = 1.5, 0.0      # free space in mapping_world, 1.5 m ahead of spawn
GOAL_TOLERANCE_M = 0.5


@pytest.mark.launch_test
@launch_testing.markers.keep_alive
def generate_test_description():
    return launch.LaunchDescription([
        include_bringup('navigation.launch.py'),
        launch_testing.actions.ReadyToTest(),
    ])


class TestNavigation(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        rclpy.init()
        cls.node = Probe('orion_nav_probe')
        cls.node.watch(Odometry, '/diff_drive_controller/odom')
        cls.init_pub = cls.node.create_publisher(
            PoseWithCovarianceStamped, '/initialpose', 10)
        cls.nav = ActionClient(cls.node, NavigateToPose, 'navigate_to_pose')
        cls.tf_buffer = Buffer()
        cls.tf_listener = TransformListener(cls.tf_buffer, cls.node)

    @classmethod
    def tearDownClass(cls):
        cls.node.destroy_node()
        rclpy.shutdown()

    def _map_pose(self):
        try:
            t = self.tf_buffer.lookup_transform('map', 'base_link', rclpy.time.Time())
        except Exception:
            return None
        return t.transform.translation

    def test_reaches_goal(self):
        node = self.node
        self.assertTrue(node.spin_until(
            lambda: '/diff_drive_controller/odom' in node.latest, 120),
            'controllers never came up')

        # AMCL has set_initial_pose: false, so tell it the rover starts at the map origin.
        def localized():
            pose = PoseWithCovarianceStamped()
            pose.header.frame_id = 'map'
            pose.header.stamp = node.get_clock().now().to_msg()
            pose.pose.pose.orientation.w = 1.0
            pose.pose.covariance[0] = pose.pose.covariance[7] = 0.05
            pose.pose.covariance[35] = 0.05
            self.init_pub.publish(pose)
            node.spin_for(1.0)
            return self._map_pose() is not None
        self.assertTrue(node.spin_until(localized, 90), 'AMCL never published map -> odom')

        # The action server exists before Nav2 is active and rejects goals until then.
        active = node.create_client(Trigger, '/lifecycle_manager_navigation/is_active')

        def nav2_active():
            if not active.wait_for_service(timeout_sec=0.5):
                return False
            fut = active.call_async(Trigger.Request())
            node.spin_until(fut.done, 5)
            return fut.done() and fut.result().success
        self.assertTrue(node.spin_until(nav2_active, 120), 'Nav2 lifecycle never became active')
        self.assertTrue(self.nav.wait_for_server(timeout_sec=30),
                        'navigate_to_pose action server never came up')
        goal = NavigateToPose.Goal()
        goal.pose = PoseStamped()
        goal.pose.header.frame_id = 'map'
        goal.pose.pose.position.x = GOAL_X
        goal.pose.pose.position.y = GOAL_Y
        goal.pose.pose.orientation.w = 1.0

        send = self.nav.send_goal_async(goal)
        self.assertTrue(node.spin_until(send.done, 30), 'goal was never acknowledged')
        handle = send.result()
        self.assertTrue(handle.accepted, 'Nav2 rejected the goal')

        result = handle.get_result_async()
        t0 = time.monotonic()
        finished = node.spin_until(result.done, 150)
        pose = self._map_pose()
        dist = math.hypot(pose.x - GOAL_X, pose.y - GOAL_Y) if pose else float('nan')
        self.assertTrue(finished, f'Nav2 did not finish within 150 s; {dist:.2f} m from goal')
        self.assertLess(dist, GOAL_TOLERANCE_M,
                        f'Nav2 finished after {time.monotonic() - t0:.0f} s but rover is '
                        f'{dist:.2f} m from the goal')


@launch_testing.post_shutdown_test()
class TestShutdown(unittest.TestCase):

    def test_cleanup(self):
        kill_leftover_gazebo()
