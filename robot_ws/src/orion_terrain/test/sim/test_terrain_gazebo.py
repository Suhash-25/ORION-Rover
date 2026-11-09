"""
Headless Gazebo check of terrain_node (launch_testing, pytest).

Not collected by `colcon test` (test/conftest.py) because it boots Gazebo. Run with:

    cd robot_ws/src/orion_terrain
    ORION_RUN_SIM=1 ROS_DOMAIN_ID=73 GZ_PARTITION=orion_terrain_test \
        python3 -m pytest test/sim/test_terrain_gazebo.py -s

1. TerrainState at 20 Hz (sim time), IMU_ONLY (no camera / lidar odometry in the bringup),
   effort_source NONE (the shipped sim effort model is a labelled zero model).
2. IMU-only while driving straight at 0.3 m/s: body_speed is reported, NOT asserted against
   truth. Gazebo's 100 Hz IMU samples the instantaneous acceleration of a body that the
   velocity-commanded wheels spin up through physics-rate contact impulses, so its integral
   recovers only ~77% of the speed change (0.230 vs true 0.299 m/s, measured with
   `gz model -p`; the estimator reproduces the raw IMU integral exactly). See the WP3 log.
3. Gazebo's true model pose relayed to /orion/ground_truth_pose (the sim analogue of the
   overhead camera): source EXTERNAL, body speed = wheel odometry, slip ~ 0 on rigid ground.
"""

import os
import subprocess
import sys
import time
import unittest

os.environ.setdefault('ROS_DOMAIN_ID', '73')
os.environ.setdefault('GZ_PARTITION', 'orion_terrain_test')

from ament_index_python.packages import get_package_share_directory  # noqa: E402
from geometry_msgs.msg import PoseStamped, TwistStamped  # noqa: E402
import launch  # noqa: E402
from launch.actions import IncludeLaunchDescription  # noqa: E402
from launch.launch_description_sources import PythonLaunchDescriptionSource  # noqa: E402
from launch_ros.actions import Node as LaunchNode  # noqa: E402
import launch_testing.actions  # noqa: E402
import launch_testing.markers  # noqa: E402
from nav_msgs.msg import Odometry  # noqa: E402
import numpy as np  # noqa: E402
from orion_msgs.msg import TerrainState  # noqa: E402
import pytest  # noqa: E402
import rclpy  # noqa: E402
from rclpy.node import Node  # noqa: E402
from tf2_msgs.msg import TFMessage  # noqa: E402

GZ_POSES = '/world/orion_world/dynamic_pose/info'


def report(text):
    """Print a result line on stderr (launch_testing captures the tests' stdout)."""
    sys.stderr.write(f'[terrain_gazebo] {text}\n')


def include(pkg, launch_file, **args):
    path = os.path.join(get_package_share_directory(pkg), 'launch', launch_file)
    return IncludeLaunchDescription(PythonLaunchDescriptionSource(path),
                                    launch_arguments=args.items())


@pytest.mark.launch_test
@launch_testing.markers.keep_alive
def generate_test_description():
    pose_bridge = LaunchNode(
        package='ros_gz_bridge', executable='parameter_bridge', output='screen',
        arguments=[f'{GZ_POSES}@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V'],
        remappings=[(GZ_POSES, '/terrain_test/gz_poses')])
    return launch.LaunchDescription([
        include('orion_bringup', 'gazebo.launch.py', headless='true', teleop='false'),
        include('orion_terrain', 'terrain.launch.py', sim='true'),
        pose_bridge,
        launch_testing.actions.ReadyToTest(),
    ])


class TestTerrainInGazebo(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        rclpy.init()
        cls.node = Node('terrain_probe', parameter_overrides=[
            rclpy.parameter.Parameter('use_sim_time', value=True)])
        cls.msgs, cls.odom, cls.relayed = [], [], []
        cls.relay = False
        cls.node.create_subscription(TerrainState, '/orion/terrain_state',
                                     cls.msgs.append, 50)
        cls.node.create_subscription(Odometry, '/diff_drive_controller/odom',
                                     lambda m: cls.odom.append(m.twist.twist.linear.x), 50)
        cls.node.create_subscription(TFMessage, '/terrain_test/gz_poses', cls.on_gz_poses, 50)
        cls.cmd = cls.node.create_publisher(TwistStamped, '/diff_drive_controller/cmd_vel', 10)
        cls.gt = cls.node.create_publisher(PoseStamped, '/orion/ground_truth_pose', 10)

    @classmethod
    def on_gz_poses(cls, msg):
        """Relay the model pose as an overhead-camera fix (Pose_V names/stamps are not bridged)."""
        if not cls.relay or not msg.transforms:
            return
        tr = msg.transforms[0].transform           # entry 0 is the model; links are relative
        out = PoseStamped()
        out.header.stamp = cls.node.get_clock().now().to_msg()
        out.header.frame_id = 'sandbox'
        out.pose.position.x, out.pose.position.y = tr.translation.x, tr.translation.y
        out.pose.position.z = tr.translation.z
        out.pose.orientation = tr.rotation
        cls.gt.publish(out)
        cls.relayed.append(tr.translation.x)

    @classmethod
    def tearDownClass(cls):
        cls.node.destroy_node()
        rclpy.shutdown()

    def spin_for(self, seconds, v=None):
        """Spin for `seconds` of SIM time, commanding v m/s if given."""
        clock = self.node.get_clock()
        while clock.now().nanoseconds == 0:
            rclpy.spin_once(self.node, timeout_sec=0.1)
        end = clock.now().nanoseconds * 1e-9 + seconds
        last = 0.0
        while clock.now().nanoseconds * 1e-9 < end:
            rclpy.spin_once(self.node, timeout_sec=0.05)
            now = clock.now().nanoseconds * 1e-9
            if v is not None and now - last >= 0.05:
                msg = TwistStamped()
                msg.header.stamp = clock.now().to_msg()
                msg.twist.linear.x = v
                self.cmd.publish(msg)
                last = now

    def drive_straight(self, v=0.3, settle=1.5, measure=4.0):
        self.spin_for(settle, v=v)
        self.msgs.clear()
        self.odom.clear()
        self.spin_for(measure, v=v)
        speed = np.array([m.body_speed for m in self.msgs])
        slip = np.array([m.slip_side for m in self.msgs])
        odom = float(np.mean(self.odom))
        self.spin_for(2.0, v=0.0)
        return speed, slip, odom

    def test_1_publishes_at_20hz_imu_only(self):
        deadline = time.monotonic() + 120
        while (not self.msgs or len(self.odom) < 50) and time.monotonic() < deadline:
            rclpy.spin_once(self.node, timeout_sec=0.1)
        self.assertTrue(self.msgs, 'no TerrainState within 120 s')
        self.msgs.clear()
        self.spin_for(5.0)
        stamps = np.array([m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
                           for m in self.msgs])
        rate = (len(stamps) - 1) / (stamps[-1] - stamps[0])
        report(f'TerrainState rate {rate:.2f} Hz (sim time), {len(stamps)} msgs')
        self.assertAlmostEqual(rate, 20.0, delta=1.0)
        last = self.msgs[-1]
        self.assertEqual(last.body_speed_source, TerrainState.SPEED_SOURCE_IMU_ONLY)
        self.assertEqual(last.effort_source, TerrainState.EFFORT_SOURCE_NONE)
        self.assertEqual(last.sinkage_proxy, 0.0)
        self.assertEqual(last.confidence, 0.0)          # never had a fix
        self.assertLess(abs(last.body_speed), 0.02)     # at rest, no drift

    def test_2_imu_only_straight_drive_is_reported(self):
        speed, slip, odom = self.drive_straight()
        report(f'IMU_ONLY: body_speed {speed.mean():.3f} m/s vs wheel odom {odom:.3f} m/s, '
               f'slip_side mean {slip.mean(0).round(3)} (Gazebo IMU impulse aliasing, see doc)')
        self.assertAlmostEqual(odom, 0.3, delta=0.03, msg='rover did not drive at 0.3 m/s')
        self.assertTrue(all(m.body_speed_source == TerrainState.SPEED_SOURCE_IMU_ONLY
                            for m in self.msgs))
        self.assertGreater(float(speed.mean()), 0.1)    # integrates, sign right

    def test_3_external_fix_gives_zero_slip_on_rigid_ground(self):
        type(self).relay = True
        self.spin_for(1.0)
        self.assertGreater(len(self.relayed), 20, 'no Gazebo model poses bridged')
        speed, slip, odom = self.drive_straight()
        type(self).relay = False
        sources = {m.body_speed_source for m in self.msgs}
        conf = np.array([m.confidence for m in self.msgs])
        report(f'EXTERNAL: body_speed {speed.mean():.3f} m/s vs wheel odom {odom:.3f} m/s, '
               f'slip_side mean {slip.mean(0).round(3)}, max |slip_side| '
               f'{np.abs(slip).max():.3f}, confidence min {conf.min():.2f} mean '
               f'{conf.mean():.2f}, time_since_fix max '
               f'{max(m.time_since_fix for m in self.msgs):.3f} s, |v - odom| max '
               f'{np.abs(speed - odom).max():.3f}')
        self.assertEqual(sources, {TerrainState.SPEED_SOURCE_EXTERNAL})
        self.assertAlmostEqual(float(speed.mean()), odom, delta=0.02)
        self.assertLess(float(np.abs(slip.mean(0)).max()), 0.05)
        self.assertLess(float(np.abs(slip).max()), 0.15)
        # Single-tick wheel-odometry spikes from Gazebo contacts dip the disagreement factor
        # (observed minima 0.67-0.83 over runs); on average the fix is fully trusted.
        self.assertGreater(conf.min(), 0.5)
        self.assertGreater(conf.mean(), 0.9)


@launch_testing.post_shutdown_test()
class TestShutdown(unittest.TestCase):

    def test_kill_leftover_gazebo(self, proc_info):
        partition = os.environ['GZ_PARTITION']
        out = subprocess.run(['pgrep', '-f', 'gz sim'], capture_output=True, text=True).stdout
        for pid in out.split():
            try:
                with open(f'/proc/{pid}/environ', 'rb') as f:
                    env = f.read().split(b'\0')
            except OSError:
                continue
            if f'GZ_PARTITION={partition}'.encode() in env:
                subprocess.run(['kill', '-9', pid], check=False)
        for info in proc_info:
            self.assertNotEqual(info.returncode, -11, f'{info.process_name} segfaulted')
