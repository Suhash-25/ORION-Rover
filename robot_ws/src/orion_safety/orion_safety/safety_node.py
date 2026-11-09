"""
orion_safety node: thin ROS wrapper around SafetyMonitor (docs/SAFETY.md).

Publishes orion_msgs/SafetyStatus on /orion/safety_status at rate_hz (50 Hz), zero
TwistStamped on /cmd_vel/safety (twist_mux priority 150) while HALT or ESTOP and on
/cmd_vel/estop (priority 255) while ESTOP, and serves /orion/estop_reset.
Input arrival times are taken from the node clock (sim time in Gazebo), not header stamps,
so a sensor with a skewed clock cannot fake freshness.
"""

from dataclasses import fields
import math

from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
from orion_msgs.msg import RoverHealth
from orion_msgs.msg import SafetyStatus
from orion_msgs.msg import TerrainState
from orion_msgs.srv import EstopReset
from orion_safety.geometry import accel_to_roll_pitch
from orion_safety.geometry import quaternion_to_rpy
from orion_safety.monitor import SafetyConfig
from orion_safety.monitor import SafetyMonitor
from orion_safety.monitor import STATE_ESTOP
from orion_safety.monitor import STATE_NAMES
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool

DEFAULT_WHEELS = ['front_left_wheel_joint', 'front_right_wheel_joint',
                  'rear_left_wheel_joint', 'rear_right_wheel_joint']


def imu_roll_pitch(msg):
    """(roll, pitch) in rad: from the orientation if the IMU provides one, else from gravity."""
    q = msg.orientation
    has_orientation = (msg.orientation_covariance[0] != -1.0
                       and abs(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w - 1.0) < 0.1)
    if has_orientation:
        roll, pitch, _ = quaternion_to_rpy(q.x, q.y, q.z, q.w)
        return roll, pitch
    a = msg.linear_acceleration
    return accel_to_roll_pitch(a.x, a.y, a.z)


def yaw_of(q):
    return quaternion_to_rpy(q.x, q.y, q.z, q.w)[2]


class SafetyNode(Node):

    def __init__(self):
        super().__init__('orion_safety')
        self.monitor = SafetyMonitor(self._load_config())
        rate = self.declare_parameter('rate_hz', 50.0).value
        self._wheels = list(self.declare_parameter('wheel_joints', DEFAULT_WHEELS).value)
        # Frame of the geofence polygon = frame of /orion/ground_truth_pose (orion_ground_truth).
        self._fence_frame = self.declare_parameter('geofence_frame', 'sandbox').value
        self._last_state = None

        self.status_pub = self.create_publisher(SafetyStatus, '/orion/safety_status', 10)
        self.safety_pub = self.create_publisher(TwistStamped, '/cmd_vel/safety', 10)
        self.estop_pub = self.create_publisher(TwistStamped, '/cmd_vel/estop', 10)

        sensor = qos_profile_sensor_data
        self.create_subscription(Imu, '/imu/data', self._on_imu, sensor)
        self.create_subscription(JointState, '/joint_states', self._on_joint_states, sensor)
        self.create_subscription(TerrainState, '/orion/terrain_state', self._on_terrain, sensor)
        self.create_subscription(PoseStamped, '/orion/ground_truth_pose', self._on_pose, sensor)
        self.create_subscription(Odometry, '/diff_drive_controller/odom', self._on_odom, sensor)
        self.create_subscription(
            TwistStamped, '/diff_drive_controller/cmd_vel', self._on_command, sensor)
        self.create_subscription(RoverHealth, '/orion/rover_health', self._on_health, sensor)
        # Reliable: an e-stop must not be dropped. Publishers must be reliable too.
        self.create_subscription(Bool, '/orion/estop', self._on_estop, 10)
        self.create_service(EstopReset, '/orion/estop_reset', self._on_reset)
        self.create_timer(1.0 / rate, self.tick)
        cfg = self.monitor.cfg
        self.get_logger().info(
            f'orion_safety at {rate:g} Hz; current limits {cfg.current_cont:g}/'
            f'{cfg.current_peak:g} (effort unit), geofence '
            f'{"on" if cfg.geofence_enabled else "OFF"}')
        if not cfg.geofence_enabled:
            self.get_logger().warn('geofence disabled by parameter geofence_enabled')

    def _load_config(self):
        kwargs = {}
        for f in fields(SafetyConfig):
            default = getattr(SafetyConfig(), f.name)
            value = self.declare_parameter(f.name, default).value
            kwargs[f.name] = (list(value) if isinstance(default, list)
                              else type(default)(value))
        return SafetyConfig(**kwargs)

    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_imu(self, msg):
        self.monitor.on_imu(self._now(), *imu_roll_pitch(msg))

    def _on_joint_states(self, msg):
        # Without an effort interface the message still counts as a heartbeat (efforts NaN).
        by_name = dict(zip(msg.name, msg.effort))
        efforts = [by_name.get(w, math.nan) for w in self._wheels]
        self.monitor.on_joint_states(self._now(), efforts)

    def _on_terrain(self, msg):
        self.monitor.on_terrain(self._now(), msg.body_speed, msg.confidence, msg.slip_side)

    def _on_pose(self, msg):
        if msg.header.frame_id != self._fence_frame:
            self.get_logger().error(
                f'/orion/ground_truth_pose in frame "{msg.header.frame_id}", geofence polygon '
                f'is in "{self._fence_frame}": pose ignored', throttle_duration_sec=5.0)
            return
        p = msg.pose
        self.monitor.on_external_pose(self._now(), p.position.x, p.position.y,
                                      yaw_of(p.orientation))

    def _on_odom(self, msg):
        p = msg.pose.pose
        self.monitor.on_odom_pose(self._now(), p.position.x, p.position.y,
                                  yaw_of(p.orientation))

    def _on_command(self, msg):
        self.monitor.on_command(self._now(), msg.twist.linear.x, msg.twist.angular.z)

    def _on_health(self, msg):
        self.monitor.on_health(self._now(), msg.fw_fault_flags)

    def _on_estop(self, msg):
        self.monitor.on_estop(self._now(), msg.data)
        if msg.data:
            self.tick()         # stop now, not at the next timer tick

    def _on_reset(self, request, response):
        response.success, response.message = self.monitor.request_reset(
            self._now(), request.confirm)
        log = self.get_logger().info if response.success else self.get_logger().warn
        log(f'/orion/estop_reset: {response.message}')
        return response

    def tick(self):
        now_msg = self.get_clock().now()
        if now_msg.nanoseconds == 0:
            return      # use_sim_time and no /clock yet
        decision = self.monitor.step(now_msg.nanoseconds * 1e-9)
        status = SafetyStatus()
        status.header.stamp = now_msg.to_msg()
        status.state = decision.state
        status.flags = decision.flags
        status.max_linear_speed = float(decision.max_linear_speed)
        status.max_angular_speed = float(decision.max_angular_speed)
        status.max_current = float(decision.max_current)
        status.reason = decision.reason
        self.status_pub.publish(status)

        if decision.stops_robot:
            zero = TwistStamped()
            zero.header.stamp = status.header.stamp
            zero.header.frame_id = 'base_link'
            self.safety_pub.publish(zero)
            if decision.state == STATE_ESTOP:
                self.estop_pub.publish(zero)

        if decision.state != self._last_state:
            log = self.get_logger().info if decision.state <= 1 else self.get_logger().warn
            log(f'state {STATE_NAMES[decision.state]}: {decision.reason or "ok"}')
            self._last_state = decision.state


def main(args=None):
    rclpy.init(args=args)
    node = SafetyNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
