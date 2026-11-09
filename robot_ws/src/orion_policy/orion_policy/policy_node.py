"""
orion_policy node: ROS wiring around PolicyCore.

Subscribes /joint_states, /imu/data, /orion/terrain_state, /orion/safety_status.
Publishes TwistStamped on /cmd_vel/policy and /cmd_vel/escape and PolicyStatus on
/orion/policy_status, every 50 ms of ROS time (sim time in Gazebo).
Service /orion/set_mode (orion_msgs/srv/SetMode).
"""

from __future__ import annotations

from geometry_msgs.msg import TwistStamped
from orion_learning import obs_spec
from orion_learning.baselines import B0Params
from orion_learning.escape_library import MAX_ESCAPE_ATTEMPTS
from orion_msgs.msg import PolicyStatus
from orion_msgs.msg import SafetyStatus
from orion_msgs.msg import TerrainState
from orion_msgs.srv import SetMode
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu
from sensor_msgs.msg import JointState

from .artefact import load_artefact
from .controllers import BaselineB0
from .controllers import OnnxController
from .core import CoreConfig
from .core import ESCAPE_CHANNEL
from .core import MODE_NAMES
from .core import POLICY_CHANNEL
from .core import PolicyCore
from .core import SafetyInput
from .observation import ObservationBuilder

DEFAULT_WHEEL_JOINTS = ['front_left_wheel_joint', 'front_right_wheel_joint',
                        'rear_left_wheel_joint', 'rear_right_wheel_joint']


class PolicyNode(Node):

    def __init__(self, **kwargs):
        super().__init__('orion_policy', **kwargs)
        p = self.declare_parameter
        wheel_joints = p('wheel_joints', DEFAULT_WHEEL_JOINTS).value
        effort_max = p('effort_max', 3.0).value   # N.m in Gazebo (tau_max); A on the rover
        gravity = p('gravity', 9.81).value
        initial_mode = p('initial_mode', 'IDLE').value
        model_dir = p('model_dir', '').value
        d = B0Params()      # defaults = the simulator's B0 (orion_learning.baselines)
        b0 = BaselineB0(B0Params(
            cruise_v=p('b0.cruise_speed', d.cruise_v).value,
            governor_slip=p('b0.slip_threshold', d.governor_slip).value,
            cut_per_s=p('b0.cut_rate', d.cut_per_s).value,
            recover_per_s=p('b0.recover_rate', d.recover_per_s).value,
            min_scale=p('b0.min_fraction', d.min_scale).value,
            scale_after_escape=p('b0.scale_after_escape', d.scale_after_escape).value))
        cfg = CoreConfig(
            safety_timeout_s=p('safety_timeout_s', 0.5).value,
            critic_gating=p('critic_gating', True).value,
            risk_thresholds=tuple(p('risk_thresholds', [0.3, 0.5, 0.7]).value),
            caution_max_speed=p('caution_max_speed', 0.2).value,
            probe_cooldown_s=p('probe_cooldown_s', 5.0).value,
            retreat_path_s=p('retreat_path_s', 2.0).value,
            max_escape_attempts=p('max_escape_attempts', MAX_ESCAPE_ATTEMPTS).value)

        policy = None
        if model_dir:
            artefact = load_artefact(model_dir)   # raises ArtefactError: refuse to start
            policy = OnnxController(artefact)
            self.get_logger().info(
                f'loaded {artefact.controller} from {model_dir} (policy sha '
                f'{artefact.policy_sha}, adapter {artefact.adapter is not None}, '
                f'critic {artefact.critic is not None})')

        self.builder = ObservationBuilder(wheel_joints, effort_max, gravity)
        self.core = PolicyCore(cfg, b0, policy)
        self.safety = None

        self.create_subscription(JointState, '/joint_states', self._on_joints, 10)
        self.create_subscription(Imu, '/imu/data', self._on_imu, qos_profile_sensor_data)
        self.create_subscription(TerrainState, '/orion/terrain_state', self._on_terrain, 10)
        self.create_subscription(SafetyStatus, '/orion/safety_status', self._on_safety, 10)
        self.pubs = {POLICY_CHANNEL: self.create_publisher(TwistStamped, '/cmd_vel/policy', 10),
                     ESCAPE_CHANNEL: self.create_publisher(TwistStamped, '/cmd_vel/escape', 10)}
        self.status_pub = self.create_publisher(PolicyStatus, '/orion/policy_status', 10)
        self.create_service(SetMode, '/orion/set_mode', self._on_set_mode)
        self.timer = self.create_timer(cfg.dt, self._on_tick)

        modes = {v: k for k, v in MODE_NAMES.items()}
        if initial_mode not in modes:
            raise ValueError(f'initial_mode {initial_mode!r} not in {sorted(modes)}')
        ok, msg = self.core.request_mode(modes[initial_mode])
        if not ok:
            raise RuntimeError(f'initial_mode {initial_mode}: {msg}')
        self._last_reason = ''

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_joints(self, msg: JointState) -> None:
        try:
            self.builder.update_joint_state(self._now(), msg.name, msg.velocity, msg.effort)
        except ValueError as e:
            self.get_logger().error(str(e), throttle_duration_sec=2.0)

    def _on_imu(self, msg: Imu) -> None:
        q, av, la = msg.orientation, msg.angular_velocity, msg.linear_acceleration
        self.builder.update_imu(self._now(), (q.x, q.y, q.z, q.w), (av.x, av.y, av.z),
                                (la.x, la.y, la.z))

    def _on_terrain(self, msg: TerrainState) -> None:
        self.builder.update_terrain(self._now(), msg.body_speed, msg.confidence,
                                    msg.slip_side, msg.sinkage_proxy)

    def _on_safety(self, msg: SafetyStatus) -> None:
        self.safety = SafetyInput(t=self._now(), state=msg.state, flags=msg.flags,
                                  max_linear_speed=msg.max_linear_speed,
                                  max_angular_speed=msg.max_angular_speed, reason=msg.reason)

    def _on_set_mode(self, request, response):
        before = self.core.mode
        response.success, response.message = self.core.request_mode(request.mode)
        text = f'SetMode {request.mode}: {response.message} (was {MODE_NAMES[before]})'
        # rclpy loggers forbid two severities from one call site, so two calls.
        if response.success:
            self.get_logger().info(text)
        else:
            self.get_logger().warn(text)
        return response

    def _on_tick(self) -> None:
        result = self.core.tick(self._now(), self.builder, self.safety)
        stamp = self.get_clock().now().to_msg()
        for channel, v, w in result.publish:
            msg = TwistStamped()
            msg.header.stamp = stamp
            msg.header.frame_id = 'base_link'
            msg.twist.linear.x = float(v)
            msg.twist.angular.z = float(w)
            self.pubs[channel].publish(msg)
        status = PolicyStatus()
        status.header.stamp = stamp
        status.mode = result.mode
        status.entrapment_risk = float(result.risk)
        status.latent = [float(x) for x in result.latent]
        status.policy_sha = result.policy_sha
        status.obs_spec_hash = obs_spec.OBS_SPEC_HASH
        status.critic_gating = bool(self.core.cfg.critic_gating)
        status.escapes_exhausted = bool(self.core.escapes_exhausted)
        self.status_pub.publish(status)
        if result.reason and not self._last_reason:
            self.get_logger().warn(f'[{MODE_NAMES[result.mode]}] {result.reason}')
        elif result.reason:
            self.get_logger().warn(result.reason, throttle_duration_sec=2.0)
        self._last_reason = result.reason


def main(args=None):
    rclpy.init(args=args)
    node = PolicyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
