"""
Trial runner: one service call starts a recorded, timed trial; it ends on its own.

Services
  /orion/start_trial (orion_msgs/srv/StartTrial): validate, start `ros2 bag record` into
      <bag_root>/<trial_id>/bag, wait record_warmup_s, start the controller (SetMode
      BASELINE for B0, POLICY otherwise), start the timeout.
  /orion/stop_trial (orion_msgs/srv/StopTrial): the operator's outcome (SUCCESS when the goal
      line is crossed and no ground truth is available, ABORT, SAFETY).
The trial also ends by itself on
  timeout                              -> OUTCOME_TIMEOUT
  SafetyStatus FLAG_HARD_STUCK          -> OUTCOME_ENTRAPPED
  PolicyStatus.escapes_exhausted rises  -> OUTCOME_ENTRAPPED (the controller gave up after
      max_escape_attempts escapes and stopped; orion_safety's hard-stuck rule needs a command
      > 0.1 m/s and would never fire, SPEC section 8). metadata policy_gave_up_ros_s = the
      stamp of the first PolicyStatus with the flag.
  SafetyStatus HALT / ESTOP otherwise   -> OUTCOME_SAFETY
  ground truth progress >= goal_distance_m along the start heading -> OUTCOME_SUCCESS
Ending a trial sets the controller to IDLE, stops the bag and writes metadata.json next to it
(<bag_root>/<trial_id>/metadata.json). metadata.json is also written at the start, so a crash
leaves a record with outcome null.

Per-trial conditions that StartTrial does not carry are parameters, read at each StartTrial:
operator, ballast_kg, cone_depths_mm (the 3 drop-cone readings of this soil prep),
firmware_hash (from orion_hardware's PONG; RoverHealth has no hash field), model_dir.

Ground-truth gate (council M6): StartTrial is refused unless /orion/terrain_state has reported
body_speed_source EXTERNAL (overhead camera) continuously for external_speed_hold_s (2 s),
with no gap longer than terrain_max_gap_s. A camera whose clock is offset from the terrain
node's goes stale silently; this makes it loud. require_external_speed:=false is only for
sessions that are run without ground truth (failure rate only, SPEC section 9).
"""

from __future__ import annotations

import math
import os
import re
import threading
import time
from typing import Optional

from geometry_msgs.msg import PoseStamped
from orion_learning import obs_spec
from orion_msgs.msg import PolicyStatus
from orion_msgs.msg import SafetyStatus
from orion_msgs.msg import TerrainState
from orion_msgs.srv import SetMode
from orion_msgs.srv import StartTrial
from orion_msgs.srv import StopTrial
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data

from .metadata import find_repo_root
from .metadata import git_state
from .metadata import onnx_provenance
from .metadata import OUTCOME_NAMES
from .metadata import SCHEMA_VERSION
from .metadata import utc_now_iso
from .metadata import write_json_atomic
from .protocol import CELLS
from .protocol import CONTROLLERS
from .recorder import BagRecorder
from .recorder import SPEC_TOPICS

TRIAL_ID_RE = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$')
MODE_FOR_CONTROLLER = {'B0': PolicyStatus.MODE_BASELINE}
SPEED_SOURCE_NAMES = {TerrainState.SPEED_SOURCE_NONE: 'NONE',
                      TerrainState.SPEED_SOURCE_IMU_ONLY: 'IMU_ONLY',
                      TerrainState.SPEED_SOURCE_LIDAR_ODOM: 'LIDAR_ODOM',
                      TerrainState.SPEED_SOURCE_EXTERNAL: 'EXTERNAL'}


def yaw_of(q) -> float:
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class TrialRunner(Node):

    def __init__(self, recorder_factory=BagRecorder, **kwargs):
        super().__init__('orion_trial_runner', **kwargs)
        p = self.declare_parameter
        p('bag_root', os.path.expanduser('~/orion_trials'))
        p('operator', '')
        p('ballast_kg', 0.0)
        p('cone_depths_mm', Parameter.Type.DOUBLE_ARRAY)
        p('firmware_hash', '')
        p('model_dir', '')
        p('repo_path', '')
        p('topics', list(SPEC_TOPICS))
        p('goal_distance_m', 1.5)
        p('auto_success', True)
        p('drive_policy', True)
        p('record_warmup_s', 2.0)
        p('set_mode_timeout_s', 3.0)
        p('require_external_speed', True)
        p('external_speed_hold_s', 2.0)
        p('terrain_max_gap_s', 0.25)       # TerrainState is 20 Hz: 5 missed messages
        self._recorder_factory = recorder_factory
        self._lock = threading.RLock()
        self._trial: Optional[dict] = None
        self._recorder = None
        self._timer = None
        self._start_pose = None
        self._status: Optional[PolicyStatus] = None
        # (receipt time of the last TerrainState, its speed source, start of the current
        # continuous EXTERNAL run or None), node clock in s; replaced as one tuple.
        self._terrain = (None, None, None)
        # Services and subscriptions each run one callback at a time; the SetMode client has
        # its own group so its response is handled while a service callback waits for it
        # (needs a MultiThreadedExecutor, see main()).
        services, subs = MutuallyExclusiveCallbackGroup(), MutuallyExclusiveCallbackGroup()
        self._set_mode = self.create_client(SetMode, '/orion/set_mode',
                                            callback_group=ReentrantCallbackGroup())
        self.create_service(StartTrial, '/orion/start_trial', self._on_start,
                            callback_group=services)
        self.create_service(StopTrial, '/orion/stop_trial', self._on_stop,
                            callback_group=services)
        self.create_subscription(SafetyStatus, '/orion/safety_status', self._on_safety, 10,
                                 callback_group=subs)
        self.create_subscription(PolicyStatus, '/orion/policy_status', self._on_policy, 10,
                                 callback_group=subs)
        self.create_subscription(PoseStamped, '/orion/ground_truth_pose', self._on_pose, 10,
                                 callback_group=subs)
        self.create_subscription(TerrainState, '/orion/terrain_state', self._on_terrain,
                                 qos_profile_sensor_data, callback_group=subs)

    def param(self, name):
        return self.get_parameter(name).value

    def _ros_now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    # --- start ------------------------------------------------------------------------------

    def _validate(self, req) -> Optional[str]:
        if not TRIAL_ID_RE.match(req.trial_id):
            return f'trial_id {req.trial_id!r} must match {TRIAL_ID_RE.pattern}'
        if req.cell not in CELLS:
            return f'cell {req.cell!r} not in {CELLS}'
        if req.controller not in CONTROLLERS:
            return f'controller {req.controller!r} not in {CONTROLLERS}'
        if not (req.duration_s > 0.0 and math.isfinite(req.duration_s)):
            return f'duration_s must be > 0, got {req.duration_s}'
        ballast = self.param('ballast_kg')
        if req.cell.startswith('heavy') and not ballast > 0.0:
            return f'cell {req.cell} needs ballast_kg > 0 (parameter is {ballast})'
        if req.cell.startswith('light') and ballast != 0.0:
            return f'cell {req.cell} needs ballast_kg = 0 (parameter is {ballast})'
        if req.controller != 'B0' and not self.param('model_dir'):
            return f'controller {req.controller} needs the model_dir parameter (ONNX sha256)'
        return self._speed_source_problem()

    def _on_terrain(self, msg: TerrainState):
        now = self._ros_now()
        last, _, since = self._terrain
        if msg.body_speed_source != TerrainState.SPEED_SOURCE_EXTERNAL:
            since = None
        elif since is None or now - last > self.param('terrain_max_gap_s'):
            since = now
        self._terrain = (now, int(msg.body_speed_source), since)

    def external_speed_for_s(self) -> float:
        """Seconds the body-speed source has been EXTERNAL without a gap (0 if not now)."""
        last, _, since = self._terrain
        now = self._ros_now()
        if since is None or now - last > self.param('terrain_max_gap_s'):
            return 0.0
        return now - since

    def _speed_source_problem(self) -> Optional[str]:
        if not self.param('require_external_speed'):
            return None
        hold = self.param('external_speed_hold_s')
        last, source, _ = self._terrain
        held = self.external_speed_for_s()
        if held >= hold:
            return None
        if last is None:
            why = 'no /orion/terrain_state received'
        elif self._ros_now() - last > self.param('terrain_max_gap_s'):
            why = f'/orion/terrain_state is {self._ros_now() - last:.2f} s old'
        elif source != TerrainState.SPEED_SOURCE_EXTERNAL:
            why = f'body_speed_source is {SPEED_SOURCE_NAMES.get(source, source)}'
        else:
            why = f'body_speed_source EXTERNAL for only {held:.2f} s'
        return (f'refused: {why}; a trial needs EXTERNAL (overhead camera) continuously for '
                f'{hold:g} s (council M6). Check orion_ground_truth and the camera clock, or '
                f'set require_external_speed:=false for a session without ground truth')

    def _on_start(self, req, res):
        with self._lock:
            res.success = False
            if self._trial is not None:
                res.message = f'trial {self._trial["trial_id"]} is still running'
                return res
            problem = self._validate(req)
            if problem:
                res.message = problem
                return res
            trial_dir = os.path.join(self.param('bag_root'), req.trial_id)
            if os.path.exists(trial_dir):
                res.message = f'{trial_dir} already exists'
                return res
            try:
                onnx = (None if req.controller == 'B0'
                        else onnx_provenance(self.param('model_dir')))
            except (OSError, ValueError) as e:
                res.message = f'cannot read model_dir: {e}'
                return res
            repo = self.param('repo_path') or find_repo_root(os.path.dirname(__file__))
            sha, dirty = git_state(repo)
            os.makedirs(trial_dir)
            bag_path = os.path.join(trial_dir, 'bag')
            self._trial = {
                'schema_version': SCHEMA_VERSION, 'trial_id': req.trial_id, 'cell': req.cell,
                'controller': req.controller, 'duration_s': float(req.duration_s),
                'operator': self.param('operator') or None,
                'ballast_kg': float(self.param('ballast_kg')),
                'cone_depths_mm': list(self.param('cone_depths_mm') or []),
                'git_sha': sha, 'git_dirty': dirty,
                'firmware_hash': self.param('firmware_hash') or None,
                'onnx': onnx, 'obs_spec_hash': obs_spec.OBS_SPEC_HASH,
                'policy_status': None, 'modes_seen': [],
                'require_external_speed': bool(self.param('require_external_speed')),
                'external_speed_for_s': self.external_speed_for_s(),
                'goal_distance_m': float(self.param('goal_distance_m')),
                'start_pose': None, 'final_progress_m': None, 'policy_gave_up_ros_s': None,
                'bag_path': os.path.abspath(bag_path), 'bag_exit_code': None,
                'start_wall': None, 'start_ros_s': None, 'end_wall': None, 'end_ros_s': None,
                'elapsed_s': None, 'outcome': None, 'outcome_code': None,
                'outcome_source': None, 'notes': '',
            }
            self._meta_path = os.path.join(trial_dir, 'metadata.json')
            self._start_pose = None
            self._recorder = self._recorder_factory(
                bag_path, self.param('topics'),
                use_sim_time=self.get_parameter('use_sim_time').value)
            self._recorder.start()
            time.sleep(self.param('record_warmup_s'))   # let the recorder subscribe
            if not self._recorder.running():
                self._finish(StopTrial.Request.OUTCOME_ABORT, 'runner',
                             'ros2 bag record exited during warm-up; see bag.record.log')
                res.message = 'bag recorder failed to start (see bag.record.log)'
                return res
            self._trial['start_wall'] = utc_now_iso()
            self._trial['start_ros_s'] = self._ros_now()
            write_json_atomic(self._meta_path, self._trial)
            if self.param('drive_policy'):
                mode = MODE_FOR_CONTROLLER.get(req.controller, PolicyStatus.MODE_POLICY)
                ok, why = self._call_set_mode(mode)
                if not ok:
                    self._finish(StopTrial.Request.OUTCOME_ABORT, 'runner',
                                 f'SetMode {mode} failed: {why}')
                    res.message = f'could not start the controller: {why}'
                    return res
            self._timer = self.create_timer(float(req.duration_s), self._on_timeout)
            res.success = True
            res.bag_path = self._trial['bag_path']
            res.message = f'trial {req.trial_id} running for {req.duration_s:.0f} s'
            self.get_logger().info(res.message)
            return res

    def _call_set_mode(self, mode: int) -> tuple[bool, str]:
        timeout = self.param('set_mode_timeout_s')
        if not self._set_mode.wait_for_service(timeout_sec=timeout):
            return False, '/orion/set_mode not available'
        future = self._set_mode.call_async(SetMode.Request(mode=mode))
        done = threading.Event()
        future.add_done_callback(lambda _: done.set())
        if not done.wait(timeout):
            return False, 'SetMode timed out'
        result = future.result()
        return result.success, result.message

    # --- end ---------------------------------------------------------------------------------

    def _on_stop(self, req, res):
        if req.outcome not in OUTCOME_NAMES:
            res.success, res.message = False, f'unknown outcome {req.outcome}'
            return res
        res.success = self._finish(req.outcome, 'operator', req.notes)
        res.message = 'trial stopped' if res.success else 'no trial is running'
        return res

    def _on_timeout(self):
        self._finish(StopTrial.Request.OUTCOME_TIMEOUT, 'timeout', '')

    def _on_safety(self, msg: SafetyStatus):
        if self._trial is None or self._trial['start_ros_s'] is None:
            return
        if msg.flags & SafetyStatus.FLAG_HARD_STUCK:
            self._finish(StopTrial.Request.OUTCOME_ENTRAPPED, 'safety hard-stuck', msg.reason)
        elif msg.state in (SafetyStatus.STATE_HALT, SafetyStatus.STATE_ESTOP):
            self._finish(StopTrial.Request.OUTCOME_SAFETY, f'safety state {msg.state}',
                         msg.reason)

    def _on_policy(self, msg: PolicyStatus):
        prev, self._status = self._status, msg
        # Only a rising edge: a flag still latched from before the trial's SetMode is not a
        # give-up of this trial.
        gave_up = msg.escapes_exhausted and not (prev is not None and prev.escapes_exhausted)
        with self._lock:
            if self._trial is None:
                return
            if msg.mode not in self._trial['modes_seen']:
                self._trial['modes_seen'].append(int(msg.mode))
            if not gave_up or self._trial['start_ros_s'] is None:
                return
            stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            self._trial['policy_gave_up_ros_s'] = stamp
            self._finish(StopTrial.Request.OUTCOME_ENTRAPPED, 'policy gave up',
                         'escape library exhausted (PolicyStatus.escapes_exhausted)')

    def _on_pose(self, msg: PoseStamped):
        with self._lock:
            if self._trial is None or self._trial['start_ros_s'] is None:
                return
            pos = msg.pose.position
            if self._start_pose is None:
                self._start_pose = (pos.x, pos.y, yaw_of(msg.pose.orientation))
                self._trial['start_pose'] = {'frame': msg.header.frame_id, 'x': pos.x,
                                             'y': pos.y, 'yaw': self._start_pose[2]}
            x0, y0, yaw0 = self._start_pose
            progress = (pos.x - x0) * math.cos(yaw0) + (pos.y - y0) * math.sin(yaw0)
            self._trial['final_progress_m'] = progress
            if self.param('auto_success') and progress >= self._trial['goal_distance_m']:
                self._finish(StopTrial.Request.OUTCOME_SUCCESS, 'ground truth goal line', '')

    def _finish(self, outcome: int, source: str, notes: str) -> bool:
        """End the running trial once; returns False when no trial is running."""
        with self._lock:
            trial = self._trial
            if trial is None:
                return False
            if self._timer is not None:
                self._timer.cancel()
                self.destroy_timer(self._timer)
                self._timer = None
            if self.param('drive_policy') and trial['start_ros_s'] is not None:
                ok, why = self._call_set_mode(PolicyStatus.MODE_IDLE)
                if not ok:
                    self.get_logger().error(f'could not set the controller to IDLE: {why}')
            trial['end_ros_s'] = self._ros_now()
            trial['end_wall'] = utc_now_iso()
            if trial['start_ros_s'] is not None:
                trial['elapsed_s'] = trial['end_ros_s'] - trial['start_ros_s']
            trial['outcome'] = OUTCOME_NAMES[outcome]
            trial['outcome_code'] = int(outcome)
            trial['outcome_source'] = source
            trial['notes'] = notes
            if self._status is not None:
                trial['policy_status'] = {'policy_sha': self._status.policy_sha,
                                          'obs_spec_hash': self._status.obs_spec_hash}
            trial['bag_exit_code'] = self._recorder.stop()
            write_json_atomic(self._meta_path, trial)
            self._trial = None
            self.get_logger().info(f'trial {trial["trial_id"]} ended: {trial["outcome"]} '
                                   f'({source}) after {trial["elapsed_s"]} s')
            return True

    def destroy_node(self):
        if self._trial is not None:
            self._finish(StopTrial.Request.OUTCOME_ABORT, 'runner shutdown', '')
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = TrialRunner()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
