"""The ROS node end to end: topics, 20 Hz PolicyStatus, SetMode, stale safety, ONNX load."""

import time

from geometry_msgs.msg import TwistStamped
from onnx_fixture import export_artefact
from orion_learning import obs_spec
from orion_msgs.msg import PolicyStatus
from orion_msgs.msg import SafetyStatus
from orion_msgs.msg import TerrainState
from orion_msgs.srv import SetMode
from orion_policy import core
from orion_policy.artefact import ArtefactError
from orion_policy.policy_node import DEFAULT_WHEEL_JOINTS
from orion_policy.policy_node import PolicyNode
import pytest
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.parameter import Parameter
from sensor_msgs.msg import Imu
from sensor_msgs.msg import JointState


def test_mode_and_flag_constants_match_messages():
    assert (core.MODE_IDLE, core.MODE_BASELINE, core.MODE_POLICY, core.MODE_PROBE,
            core.MODE_ESCAPE) == (PolicyStatus.MODE_IDLE, PolicyStatus.MODE_BASELINE,
                                  PolicyStatus.MODE_POLICY, PolicyStatus.MODE_PROBE,
                                  PolicyStatus.MODE_ESCAPE)
    assert core.FLAG_STUCK == SafetyStatus.FLAG_STUCK
    assert core.STATE_HALT == SafetyStatus.STATE_HALT


@pytest.fixture
def ros():
    rclpy.init()
    yield
    rclpy.shutdown()


class Feeder:
    """Publishes fake sensors and safety, records what orion_policy publishes."""

    def __init__(self):
        self.node = rclpy.create_node('orion_policy_test_feeder')
        n = self.node
        self.js = n.create_publisher(JointState, '/joint_states', 10)
        self.imu = n.create_publisher(Imu, '/imu/data', 10)
        self.terrain = n.create_publisher(TerrainState, '/orion/terrain_state', 10)
        self.safety = n.create_publisher(SafetyStatus, '/orion/safety_status', 10)
        self.got = {'policy': [], 'escape': [], 'status': []}
        n.create_subscription(TwistStamped, '/cmd_vel/policy',
                              lambda m: self.got['policy'].append((time.monotonic(), m)), 10)
        n.create_subscription(TwistStamped, '/cmd_vel/escape',
                              lambda m: self.got['escape'].append((time.monotonic(), m)), 10)
        n.create_subscription(PolicyStatus, '/orion/policy_status',
                              lambda m: self.got['status'].append((time.monotonic(), m)), 10)
        self.send_safety = True
        n.create_timer(0.02, self._publish)

    def _publish(self):
        js = JointState(name=DEFAULT_WHEEL_JOINTS, velocity=[1.0] * 4, effort=[0.3] * 4)
        self.js.publish(js)
        imu = Imu()
        imu.orientation.w = 1.0
        imu.linear_acceleration.z = 9.81
        self.imu.publish(imu)
        self.terrain.publish(TerrainState(body_speed=0.1, confidence=0.9))
        if self.send_safety:
            self.safety.publish(SafetyStatus(state=0, max_linear_speed=0.5,
                                             max_angular_speed=1.0))


def spin(executor, seconds):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        executor.spin_once(timeout_sec=0.01)


def make(params):
    node = PolicyNode(parameter_overrides=[Parameter(k, value=v) for k, v in params.items()])
    feeder = Feeder()
    ex = SingleThreadedExecutor()
    ex.add_node(node)
    ex.add_node(feeder.node)
    return node, feeder, ex


def call_set_mode(feeder, ex, mode):
    client = feeder.node.create_client(SetMode, '/orion/set_mode')
    assert client.wait_for_service(timeout_sec=5.0)
    future = client.call_async(SetMode.Request(mode=mode))
    ex.spin_until_future_complete(future, timeout_sec=5.0)
    return future.result()


def test_baseline_drives_reports_and_stops_on_stale_safety(ros):
    node, feeder, ex = make({'initial_mode': 'BASELINE'})
    spin(ex, 2.0)
    cmds = [m.twist.linear.x for _, m in feeder.got['policy']]
    assert cmds and max(cmds) == pytest.approx(0.3, abs=1e-6)
    assert feeder.got['escape'] == []
    status = [m for _, m in feeder.got['status']]
    assert {s.mode for s in status} == {PolicyStatus.MODE_BASELINE}
    assert status[-1].obs_spec_hash == obs_spec.OBS_SPEC_HASH and status[-1].policy_sha == ''
    stamps = [t for t, _ in feeder.got['status']]
    rate = (len(stamps) - 1) / (stamps[-1] - stamps[0])
    assert 15.0 < rate < 25.0, f'PolicyStatus at {rate:.1f} Hz'

    feeder.send_safety = False
    t_stop = time.monotonic()
    spin(ex, 1.0)
    late = [m.twist.linear.x for t, m in feeder.got['policy'] if t > t_stop + 0.7]
    assert late and all(v == 0.0 for v in late), 'stale SafetyStatus must give zero'
    node.destroy_node()
    feeder.node.destroy_node()


def test_set_mode_service(ros, tmp_path):
    node, feeder, ex = make({})
    res = call_set_mode(feeder, ex, PolicyStatus.MODE_POLICY)
    assert not res.success and 'no ONNX controller' in res.message
    assert call_set_mode(feeder, ex, PolicyStatus.MODE_BASELINE).success
    spin(ex, 0.5)
    assert feeder.got['status'][-1][1].mode == PolicyStatus.MODE_BASELINE
    node.destroy_node()
    feeder.node.destroy_node()


def test_policy_mode_with_onnx_artefact(ros, tmp_path):
    export_artefact(str(tmp_path))
    node, feeder, ex = make({'model_dir': str(tmp_path), 'critic_gating': False})
    assert call_set_mode(feeder, ex, PolicyStatus.MODE_POLICY).success
    spin(ex, 1.0)
    status = feeder.got['status'][-1][1]
    assert status.mode == PolicyStatus.MODE_POLICY
    assert status.policy_sha == node.core.policy.policy_sha and len(status.policy_sha) == 16
    assert 0.0 <= status.entrapment_risk <= 1.0
    assert any(status.latent)
    assert feeder.got['policy']
    node.destroy_node()
    feeder.node.destroy_node()


def test_node_refuses_tampered_artefact(ros, tmp_path):
    export_artefact(str(tmp_path))
    with open(tmp_path / 'critic.onnx', 'ab') as f:
        f.write(b'\0')
    with pytest.raises(ArtefactError, match='sha256'):
        PolicyNode(parameter_overrides=[Parameter('model_dir', value=str(tmp_path))])
