"""orion_policy drives the simulated rover from an ONNX artefact in MODE_POLICY (council m10).

An artefact (adapter + policy + critic ONNX, manifest with sha256 and obs_spec_hash) is
exported into a temporary directory with orion_policy's test fixture, the same contract the
learning exporter follows. The policy is constant: action (0.6, 0.0) = 0.3 m/s straight, so
the expected motion is known; the adapter and critic are the fixture's random tiny nets and
still run every tick (latent and risk are checked). Critic gating is off so the risk cannot
switch the rover to PROBE / ESCAPE. The test publishes a nominal SafetyStatus and, when
nothing else does, a synthetic TerrainState. The command reaches the wheels through twist_mux
(priority 100), as on the robot.

Run with: colcon test --packages-select orion_sim_tests --ctest-args -R test_policy_onnx
"""

import json
import math
import os
import shutil
import sys
import tempfile
import time
import unittest

from geometry_msgs.msg import TwistStamped
import launch
from launch_ros.actions import Node
import launch_testing.actions
import launch_testing.markers
from nav_msgs.msg import Odometry
from orion_learning import obs_spec
from orion_msgs.msg import PolicyStatus
from orion_msgs.msg import SafetyStatus
from orion_msgs.msg import TerrainState
import pytest
import rclpy
from rclpy.clock import Clock
from rclpy.clock import ClockType

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
# The ONNX fixture of orion_policy's own tests (source tree; not installed).
sys.path.insert(0, os.path.join(HERE, '..', '..', 'orion_policy', 'test'))
from onnx_fixture import export_artefact  # noqa: E402, I100
from sim_helpers import include_bringup, kill_leftover_gazebo, Probe  # noqa: E402, I100

CMD_TOPIC = '/diff_drive_controller/cmd_vel'
ODOM_TOPIC = '/diff_drive_controller/odom'
STATUS_TOPIC = '/orion/policy_status'
MODEL_DIR = tempfile.mkdtemp(prefix='orion_sim_onnx_')
POLICY_ACTION = (0.6, 0.0)          # v = 0.5 * 0.6 = 0.3 m/s, w = 0


@pytest.mark.launch_test
@launch_testing.markers.keep_alive
def generate_test_description():
    export_artefact(MODEL_DIR, policy_action=POLICY_ACTION)
    policy = Node(
        package='orion_policy', executable='policy_node', name='orion_policy',
        output='screen',
        parameters=[{'initial_mode': 'POLICY', 'model_dir': MODEL_DIR,
                     'critic_gating': False, 'use_sim_time': True}],
    )
    return launch.LaunchDescription([
        include_bringup('gazebo.launch.py'),
        policy,
        launch_testing.actions.ReadyToTest(),
    ])


class TestPolicyOnnx(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        rclpy.init()
        with open(os.path.join(MODEL_DIR, 'manifest.json')) as f:
            cls.manifest = json.load(f)
        cls.node = Probe('orion_policy_onnx_sim_probe')
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
            cls.terrain_stub = cls.node.count_publishers('/orion/terrain_state') <= 1
        if cls.terrain_stub:
            cls.terrain_pub.publish(TerrainState(body_speed=0.3, confidence=0.9))

    def test_1_policy_loads_the_artefact(self):
        ok = self.node.spin_until(
            lambda: ODOM_TOPIC in self.node.latest and STATUS_TOPIC in self.node.latest,
            timeout_s=120)
        self.assertTrue(ok, 'diff_drive_controller odometry or PolicyStatus never published')
        status = self.node.latest[STATUS_TOPIC]
        self.assertEqual(status.mode, PolicyStatus.MODE_POLICY)
        self.assertEqual(status.obs_spec_hash, obs_spec.OBS_SPEC_HASH)
        self.assertEqual(self.manifest['obs_spec_hash'], obs_spec.OBS_SPEC_HASH)
        self.assertEqual(status.policy_sha, self.manifest['sha256']['policy.onnx'][:16])
        self.assertFalse(status.critic_gating)

    def test_2_drives_for_10_s_in_policy_mode(self):
        ok = self.node.spin_until(lambda: self.node.counts.get(CMD_TOPIC, 0) > 0, 60)
        self.assertTrue(ok, 'orion_policy never commanded the rover')
        x0 = self.node.latest[ODOM_TOPIC].pose.pose.position.x
        modes, hashes = set(), set()
        t_end = time.monotonic() + 10.0
        while time.monotonic() < t_end:
            rclpy.spin_once(self.node, timeout_sec=0.05)
            modes.add(self.node.latest[STATUS_TOPIC].mode)
            hashes.add(self.node.latest[STATUS_TOPIC].obs_spec_hash)
        moved = self.node.latest[ODOM_TOPIC].pose.pose.position.x - x0
        self.assertEqual(modes, {PolicyStatus.MODE_POLICY})
        self.assertEqual(hashes, {obs_spec.OBS_SPEC_HASH})
        # 0.3 m/s; even at a real-time factor of 0.5 that is 1.5 m in 10 s.
        self.assertGreater(moved, 1.0, f'odometry advanced only {moved:.2f} m in 10 s')
        cmd = self.node.latest[CMD_TOPIC].twist
        self.assertAlmostEqual(cmd.linear.x, 0.5 * POLICY_ACTION[0], delta=0.01)
        self.assertAlmostEqual(cmd.angular.z, 0.0, delta=1e-3)
        # The adapter and critic ran: a latent and a finite risk in [0, 1].
        status = self.node.latest[STATUS_TOPIC]
        self.assertTrue(any(abs(z) > 0.0 for z in status.latent), 'adapter latent is zero')
        self.assertTrue(math.isfinite(status.entrapment_risk))
        self.assertTrue(0.0 <= status.entrapment_risk <= 1.0)


@launch_testing.post_shutdown_test()
class TestShutdown(unittest.TestCase):

    def test_processes_exit(self, proc_info):
        kill_leftover_gazebo()
        shutil.rmtree(MODEL_DIR, ignore_errors=True)
        for info in proc_info:
            self.assertNotEqual(info.returncode, -11, f'{info.process_name} segfaulted')
