"""
ROS 2 node: publishes orion_msgs/TerrainState on /orion/terrain_state at 20 Hz.

Thin wrapper around orion_terrain.estimator.TerrainEstimator. Message times are header stamps
(receipt time for unstamped messages) and the publish tick uses the node clock, so every input
must be stamped from the same clock (use_sim_time in Gazebo; chrony between the rover and the
overhead-camera PC on hardware).
"""

from __future__ import annotations

from dataclasses import fields
import os
from typing import Optional

from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from orion_msgs.msg import RoverHealth, TerrainState
from orion_terrain.effort_model import EffortModel
from orion_terrain.estimator import stamp_seconds, TerrainEstimate, TerrainEstimator
from orion_terrain.params import DEFAULT_JOINT_NAMES, DEFAULT_TOPICS, TerrainParams
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu, JointState

INPUT_TYPES = {
    'joint_states': JointState,
    'imu': Imu,
    'wheel_odom': Odometry,
    'ground_truth_pose': PoseStamped,
    'lidar_odom': Odometry,
    'rover_health': RoverHealth,
}


def default_model_path(sim: bool) -> str:
    """Shipped sim model, or the hardware model written by fit_effort_model (not shipped)."""
    name = 'effort_model_sim.yaml' if sim else 'effort_model.yaml'
    return os.path.join(get_package_share_directory('orion_terrain'), 'config', name)


def to_msg(est: TerrainEstimate, stamp, frame_id: str = 'base_link') -> TerrainState:
    msg = TerrainState()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.slip = [float(s) for s in est.slip]
    msg.slip_side = [float(s) for s in est.slip_side]
    msg.effort_residual = [float(e) for e in est.effort_residual]
    msg.sinkage_proxy = float(est.sinkage_proxy)
    msg.body_speed = float(est.body_speed)
    msg.body_yaw_rate = float(est.body_yaw_rate)
    msg.body_speed_source = int(est.body_speed_source)
    msg.effort_source = int(est.effort_source)
    msg.confidence = float(est.confidence)
    msg.time_since_fix = float(est.time_since_fix)
    return msg


class TerrainNode(Node):

    def __init__(self):
        super().__init__('terrain_node')
        defaults = TerrainParams()
        for f in fields(TerrainParams):
            self.declare_parameter(f.name, getattr(defaults, f.name))
        sim = bool(self.declare_parameter('sim', False).value)
        model_path = str(self.declare_parameter('effort_model_path', '').value)
        joint_names = list(self.declare_parameter('joint_names',
                                                  list(DEFAULT_JOINT_NAMES)).value)
        topics = {role: str(self.declare_parameter(f'topics.{role}', topic).value)
                  for role, topic in DEFAULT_TOPICS.items()}
        params = TerrainParams.from_mapping(
            {f.name: self.get_parameter(f.name).value for f in fields(TerrainParams)})

        model = self._load_model(model_path or default_model_path(sim))
        self.est = TerrainEstimator(params, model, sim, joint_names, logger=self.get_logger())
        for role, msg_type in INPUT_TYPES.items():
            self.create_subscription(msg_type, topics[role],
                                     lambda msg, role=role: self._on_input(role, msg),
                                     qos_profile_sensor_data)
        self.pub = self.create_publisher(TerrainState, topics['terrain_state'], 10)
        self.create_timer(1.0 / params.rate_hz, self._tick)
        self.get_logger().info(
            f'terrain_node: sim={sim}, {params.rate_hz:.0f} Hz on {topics["terrain_state"]}, '
            f'effort model {"none" if model is None else model.effort_source.name}')

    def _load_model(self, path: str) -> Optional[EffortModel]:
        if not os.path.isfile(path):
            self.get_logger().warning(
                f'no effort model at {path}: sinkage_proxy will be 0 (effort_source NONE). '
                'Calibrate with `ros2 run orion_terrain fit_effort_model`.')
            return None
        try:
            return EffortModel.load(path)
        except (OSError, ValueError) as e:
            self.get_logger().error(f'cannot load effort model {path}: {e}')
            return None

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_input(self, role: str, msg) -> None:
        self.est.on_message(stamp_seconds(msg, self._now()), role, msg)

    def _tick(self) -> None:
        now = self.get_clock().now()
        self.pub.publish(to_msg(self.est.estimate(now.nanoseconds * 1e-9), now.to_msg()))


def main(args=None):
    rclpy.init(args=args)
    node = TerrainNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
