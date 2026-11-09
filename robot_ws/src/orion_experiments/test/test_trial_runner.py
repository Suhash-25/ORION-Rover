"""Trial runner: validation, outcomes, controller hand-off, metadata, real bag recording."""

import json
import math
import os
import subprocess
import threading
import time

from geometry_msgs.msg import PoseStamped
from orion_experiments.analysis import find_msg_dir
from orion_experiments.analysis import make_typestore
from orion_experiments.analysis import read_bag
from orion_experiments.metadata import find_repo_root
from orion_experiments.recorder import BagRecorder
from orion_experiments.trial_runner import TrialRunner
from orion_learning import obs_spec
from orion_msgs.msg import PolicyStatus
from orion_msgs.msg import SafetyStatus
from orion_msgs.msg import TerrainState
from orion_msgs.srv import SetMode
from orion_msgs.srv import StartTrial
from orion_msgs.srv import StopTrial
import pytest
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.parameter import Parameter

REPO = find_repo_root(os.path.dirname(__file__))


class FakeRecorder:
    """Stands in for `ros2 bag record` (one real-recorder test is at the end)."""

    def __init__(self, bag_path, topics, use_sim_time=False):
        self.bag_path, self.topics = bag_path, topics
        self.started = self.stopped = False

    def start(self):
        os.makedirs(self.bag_path)
        self.started = True

    def running(self):
        return self.started and not self.stopped

    def stop(self):
        self.stopped = True
        return 0


class Rig:

    def __init__(self, tmp_path, set_mode_server=True,
                 speed_source=TerrainState.SPEED_SOURCE_EXTERNAL, **params):
        defaults = {'bag_root': str(tmp_path), 'record_warmup_s': 0.0, 'operator': 'ana',
                    'cone_depths_mm': [11.0, 12.5, 10.0], 'set_mode_timeout_s': 1.0,
                    'external_speed_hold_s': 0.3}
        defaults.update(params)
        self.recorders = []

        def factory(*a, **k):
            self.recorders.append(FakeRecorder(*a, **k))
            return self.recorders[-1]

        self.runner = TrialRunner(recorder_factory=factory, parameter_overrides=[
            Parameter(k, value=v) for k, v in defaults.items()])
        self.node = rclpy.create_node('trial_runner_test_helper')
        self.modes = []
        if set_mode_server:
            self.node.create_service(SetMode, '/orion/set_mode', self._on_set_mode)
        self.safety = self.node.create_publisher(SafetyStatus, '/orion/safety_status', 10)
        self.pose = self.node.create_publisher(PoseStamped, '/orion/ground_truth_pose', 10)
        self.status = self.node.create_publisher(PolicyStatus, '/orion/policy_status', 10)
        # orion_terrain at 20 Hz; speed_source None publishes nothing.
        self.speed_source = speed_source
        self.terrain = self.node.create_publisher(TerrainState, '/orion/terrain_state', 10)
        self.node.create_timer(0.05, self._publish_terrain)
        self.start_cli = self.node.create_client(StartTrial, '/orion/start_trial')
        self.stop_cli = self.node.create_client(StopTrial, '/orion/stop_trial')
        self.executor = MultiThreadedExecutor(num_threads=4)
        self.executor.add_node(self.runner)
        self.executor.add_node(self.node)
        self.thread = threading.Thread(target=self.executor.spin, daemon=True)
        self.thread.start()
        if speed_source == TerrainState.SPEED_SOURCE_EXTERNAL:
            self.wait_external(self.runner.param('external_speed_hold_s') + 0.1)

    def _publish_terrain(self):
        if self.speed_source is not None:
            self.terrain.publish(TerrainState(body_speed_source=self.speed_source))

    def wait_external(self, seconds, timeout=10.0):
        """Wait until the runner has seen EXTERNAL for `seconds` (discovery takes a while)."""
        end = time.monotonic() + timeout
        while self.runner.external_speed_for_s() < seconds:
            assert time.monotonic() < end, 'runner never saw EXTERNAL TerrainState'
            time.sleep(0.02)

    def _on_set_mode(self, req, res):
        self.modes.append(req.mode)
        res.success, res.message = True, 'ok'
        return res

    def call(self, client, request, timeout=10.0):
        assert client.wait_for_service(timeout_sec=5.0)
        future = client.call_async(request)
        end = time.monotonic() + timeout
        while not future.done() and time.monotonic() < end:
            time.sleep(0.01)
        assert future.done(), 'service call timed out'
        return future.result()

    def start(self, trial_id='t1', cell='light-packed', controller='B0', duration=30.0):
        return self.call(self.start_cli, StartTrial.Request(
            trial_id=trial_id, cell=cell, controller=controller, duration_s=duration))

    def stop(self, outcome, notes=''):
        return self.call(self.stop_cli, StopTrial.Request(outcome=outcome, notes=notes))

    def meta(self, trial_id='t1'):
        with open(os.path.join(self.runner.param('bag_root'), trial_id, 'metadata.json')) as f:
            return json.load(f)

    def wait_outcome(self, trial_id='t1', timeout=5.0):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if self.meta(trial_id)['outcome'] is not None:
                return self.meta(trial_id)
            time.sleep(0.05)
        raise AssertionError('trial did not end')

    def close(self):
        self.executor.shutdown()
        self.runner.destroy_node()
        self.node.destroy_node()


@pytest.fixture
def rig_factory(tmp_path):
    rclpy.init()
    rigs = []

    def make(**params):
        rigs.append(Rig(tmp_path, **params))
        return rigs[-1]

    yield make
    for r in rigs:
        r.close()
    rclpy.shutdown()


def test_start_stop_writes_complete_metadata(rig_factory):
    rig = rig_factory(firmware_hash='')
    res = rig.start()
    assert res.success, res.message
    assert res.bag_path.endswith(os.path.join('t1', 'bag'))
    m = rig.meta()
    assert m['outcome'] is None and m['start_wall'] is not None        # written at start
    assert rig.modes == [PolicyStatus.MODE_BASELINE]
    rig.status.publish(PolicyStatus(mode=1, obs_spec_hash=obs_spec.OBS_SPEC_HASH))
    time.sleep(0.3)
    assert rig.stop(StopTrial.Request.OUTCOME_ABORT, 'operator tripped').success
    m = rig.meta()
    head = subprocess.run(['git', '-C', REPO, 'rev-parse', 'HEAD'], capture_output=True,
                          text=True).stdout.strip()
    assert m['git_sha'] == head and isinstance(m['git_dirty'], bool)
    assert m['outcome'] == 'ABORT' and m['outcome_code'] == 3
    assert m['outcome_source'] == 'operator' and m['notes'] == 'operator tripped'
    assert m['cell'] == 'light-packed' and m['controller'] == 'B0'
    assert m['operator'] == 'ana' and m['cone_depths_mm'] == [11.0, 12.5, 10.0]
    assert m['ballast_kg'] == 0.0 and m['firmware_hash'] is None and m['onnx'] is None
    assert m['obs_spec_hash'] == obs_spec.OBS_SPEC_HASH
    assert m['policy_status']['obs_spec_hash'] == obs_spec.OBS_SPEC_HASH
    assert m['modes_seen'] == [1]
    assert m['bag_path'] == res.bag_path and m['bag_exit_code'] == 0
    assert m['elapsed_s'] >= 0.3
    assert rig.modes[-1] == PolicyStatus.MODE_IDLE and rig.recorders[0].stopped
    assert not rig.stop(StopTrial.Request.OUTCOME_ABORT).success   # nothing running


def test_timeout(rig_factory):
    rig = rig_factory()
    assert rig.start(duration=0.5).success
    m = rig.wait_outcome()
    assert m['outcome'] == 'TIMEOUT' and m['outcome_source'] == 'timeout'
    assert m['elapsed_s'] == pytest.approx(0.5, abs=0.25)


def test_hard_stuck_is_entrapped_and_halt_is_safety(rig_factory):
    rig = rig_factory()
    assert rig.start().success
    rig.safety.publish(SafetyStatus(state=0, flags=SafetyStatus.FLAG_STUCK))
    time.sleep(0.3)
    assert rig.meta()['outcome'] is None      # stuck alone does not end a trial
    rig.safety.publish(SafetyStatus(state=2, flags=SafetyStatus.FLAG_STUCK
                                    | SafetyStatus.FLAG_HARD_STUCK, reason='hard stuck'))
    m = rig.wait_outcome()
    assert m['outcome'] == 'ENTRAPPED' and m['notes'] == 'hard stuck'

    assert rig.start(trial_id='t2').success
    rig.safety.publish(SafetyStatus(state=2, reason='tilt 26 deg'))
    m = rig.wait_outcome('t2')
    assert m['outcome'] == 'SAFETY' and m['notes'] == 'tilt 26 deg'


def test_policy_give_up_is_entrapped(rig_factory):
    rig = rig_factory()
    # A flag still latched from before the trial is not a give-up of this trial.
    rig.status.publish(PolicyStatus(mode=0, escapes_exhausted=True))
    time.sleep(0.2)
    assert rig.start().success
    rig.status.publish(PolicyStatus(mode=0, escapes_exhausted=True))
    time.sleep(0.3)
    assert rig.meta()['outcome'] is None
    rig.status.publish(PolicyStatus(mode=1))
    time.sleep(0.1)
    gave_up = PolicyStatus(mode=0, escapes_exhausted=True)
    gave_up.header.stamp.sec, gave_up.header.stamp.nanosec = 1234, 500000000
    rig.status.publish(gave_up)
    m = rig.wait_outcome()
    assert m['outcome'] == 'ENTRAPPED' and m['outcome_source'] == 'policy gave up'
    assert m['policy_gave_up_ros_s'] == pytest.approx(1234.5)
    assert rig.modes[-1] == PolicyStatus.MODE_IDLE


def test_ground_truth_goal_line_is_success(rig_factory):
    rig = rig_factory()
    assert rig.start().success
    yaw = 0.5
    for k in range(18):
        d = 0.1 * k
        msg = PoseStamped()
        msg.header.frame_id = 'sandbox'
        msg.pose.position.x, msg.pose.position.y = 2.0 + d * math.cos(yaw), d * math.sin(yaw)
        msg.pose.orientation.z, msg.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
        rig.pose.publish(msg)
        time.sleep(0.03)
    m = rig.wait_outcome()
    assert m['outcome'] == 'SUCCESS' and m['outcome_source'] == 'ground truth goal line'
    assert m['final_progress_m'] == pytest.approx(1.5, abs=0.11)
    assert m['start_pose']['frame'] == 'sandbox' and m['start_pose']['yaw'] == pytest.approx(yaw)


@pytest.mark.parametrize('kwargs,params,problem', [
    ({'cell': 'mud'}, {}, 'cell'),
    ({'controller': 'B9'}, {}, 'controller'),
    ({'trial_id': '../x'}, {}, 'trial_id'),
    ({'duration': 0.0}, {}, 'duration_s'),
    ({'cell': 'heavy-loose'}, {}, 'ballast_kg > 0'),
    ({}, {'ballast_kg': 8.0}, 'ballast_kg = 0'),
    ({'controller': 'student'}, {}, 'model_dir'),
])
def test_start_is_validated(rig_factory, kwargs, params, problem):
    rig = rig_factory(**params)
    res = rig.start(**kwargs)
    assert not res.success and problem in res.message
    assert rig.recorders == []


def test_external_speed_gate_is_on_by_default(rig_factory):
    rig = rig_factory()
    runner = TrialRunner(recorder_factory=FakeRecorder)    # no overrides
    try:
        assert runner.param('require_external_speed') is True
        assert runner.param('external_speed_hold_s') == 2.0
    finally:
        runner.destroy_node()
    assert rig.start().success
    assert rig.meta()['require_external_speed'] and rig.meta()['external_speed_for_s'] >= 0.3


@pytest.mark.parametrize('source,problem', [
    (TerrainState.SPEED_SOURCE_LIDAR_ODOM, 'body_speed_source is LIDAR_ODOM'),
    (TerrainState.SPEED_SOURCE_IMU_ONLY, 'body_speed_source is IMU_ONLY'),
    (None, 'no /orion/terrain_state received'),
])
def test_start_refused_without_external_speed(rig_factory, source, problem):
    rig = rig_factory(speed_source=source)
    time.sleep(0.5)
    res = rig.start()
    assert not res.success and problem in res.message and 'EXTERNAL' in res.message
    assert 'require_external_speed' in res.message
    assert rig.recorders == []
    assert not os.path.exists(os.path.join(rig.runner.param('bag_root'), 't1'))


def test_start_needs_external_continuously_for_the_hold_time(rig_factory):
    rig = rig_factory(speed_source=TerrainState.SPEED_SOURCE_LIDAR_ODOM,
                      external_speed_hold_s=1.0)
    time.sleep(0.3)
    rig.speed_source = TerrainState.SPEED_SOURCE_EXTERNAL
    rig.wait_external(0.1)
    res = rig.start()
    assert not res.success and 'EXTERNAL for only' in res.message
    rig.wait_external(1.0)
    assert rig.start().success
    rig.stop(StopTrial.Request.OUTCOME_ABORT)
    rig.speed_source = None                     # camera (or its clock) lost
    time.sleep(0.5)
    res = rig.start(trial_id='t2')
    assert not res.success and 's old' in res.message


def test_external_speed_gate_can_be_disabled(rig_factory):
    rig = rig_factory(speed_source=None, require_external_speed=False)
    assert rig.start().success
    assert rig.meta()['require_external_speed'] is False


def test_one_trial_at_a_time_and_no_overwrite(rig_factory):
    rig = rig_factory()
    assert rig.start().success
    res = rig.start(trial_id='t2')
    assert not res.success and 'still running' in res.message
    rig.stop(StopTrial.Request.OUTCOME_SUCCESS)
    res = rig.start()
    assert not res.success and 'already exists' in res.message


def test_learnt_controller_records_onnx_sha256(rig_factory, tmp_path):
    model = tmp_path / 'model'
    model.mkdir()
    (model / 'policy.onnx').write_bytes(b'policy-bytes')
    import hashlib
    sha = hashlib.sha256(b'policy-bytes').hexdigest()
    (model / 'manifest.json').write_text(json.dumps({
        'controller': 'student', 'obs_spec_hash': obs_spec.OBS_SPEC_HASH,
        'files': {'adapter': None, 'policy': 'policy.onnx', 'critic': None},
        'sha256': {'policy.onnx': sha}}))
    rig = rig_factory(model_dir=str(model), ballast_kg=8.0)
    assert rig.start(cell='heavy-loose', controller='student').success
    assert rig.modes == [PolicyStatus.MODE_POLICY]
    rig.stop(StopTrial.Request.OUTCOME_SUCCESS)
    onnx = rig.meta()['onnx']
    assert onnx['files']['policy.onnx'] == {'role': 'policy', 'sha256': sha,
                                            'manifest_match': True}
    assert onnx['obs_spec_hash'] == obs_spec.OBS_SPEC_HASH
    assert rig.meta()['ballast_kg'] == 8.0


def test_missing_controller_aborts_the_trial(rig_factory):
    rig = rig_factory(set_mode_server=False, set_mode_timeout_s=0.3)
    res = rig.start()
    assert not res.success and 'could not start the controller' in res.message
    m = rig.meta()
    assert m['outcome'] == 'ABORT' and '/orion/set_mode not available' in m['notes']
    assert rig.recorders[0].stopped


def test_real_ros2_bag_record_round_trip(tmp_path):
    """`ros2 bag record` writes an mcap bag that the analysis reads back."""
    rclpy.init()
    node = rclpy.create_node('bag_round_trip')
    pub = node.create_publisher(SafetyStatus, '/orion/safety_status', 10)
    rec = BagRecorder(str(tmp_path / 'bag'), ['/orion/safety_status'])
    rec.start()
    try:
        time.sleep(3.0)                          # discovery
        for k in range(30):
            pub.publish(SafetyStatus(state=0, flags=16 if k >= 15 else 0, reason=str(k)))
            time.sleep(0.05)
        time.sleep(0.5)
    finally:
        code = rec.stop()
        node.destroy_node()
        rclpy.shutdown()
    assert code == 0, (tmp_path / 'bag.record.log').read_text()
    assert (tmp_path / 'bag' / 'metadata.yaml').exists()
    msgs = read_bag(str(tmp_path / 'bag'), make_typestore(find_msg_dir()))
    got = msgs['/orion/safety_status']
    assert len(got) >= 25, f'only {len(got)} of 30 messages recorded'
    assert got[-1][1].flags == 16 and got[-1][1].reason == '29'
