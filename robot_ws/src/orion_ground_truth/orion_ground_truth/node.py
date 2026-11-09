"""ROS node: AprilTag TF from the overhead camera -> /orion/ground_truth_pose (PoseStamped).

The AprilTag detector (apriltag_ros) publishes TF from the camera frame to each tag frame.
This node looks those up, solves the rover pose in the `sandbox` frame (see geometry.py) and
publishes it stamped with the image time, so latency is visible downstream.
"""

import math

from geometry_msgs.msg import PoseStamped
import numpy as np
from orion_ground_truth.geometry import GroundTruthSolver, make_tf, SolverConfig
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from tf2_ros import Buffer, TransformException, TransformListener


def _tf_to_matrix(msg) -> np.ndarray:
    t, q = msg.transform.translation, msg.transform.rotation
    return make_tf((t.x, t.y, t.z), (q.x, q.y, q.z, q.w))


class GroundTruthNode(Node):

    def __init__(self):
        super().__init__('orion_ground_truth')
        p = self.declare_parameter
        self.camera_frame = p('camera_frame', 'overhead_camera').value
        self.ref_frame = p('ref_tag_frame', 'tag36h11:0').value
        self.rover_frame = p('rover_tag_frame', 'tag36h11:1').value
        ref_xyz = p('ref_tag_in_sandbox.xyz', [0.0, 0.0, 0.0]).value
        ref_quat = p('ref_tag_in_sandbox.quat_xyzw', [0.0, 0.0, 0.0, 1.0]).value
        base_xyz = p('base_in_rover_tag.xyz', [0.0, 0.0, -0.30]).value
        base_quat = p('base_in_rover_tag.quat_xyzw', [0.0, 0.0, 0.0, 1.0]).value
        rate = p('rate_hz', 30.0).value
        self.solver = GroundTruthSolver(SolverConfig(
            sandbox_from_ref=make_tf(ref_xyz, ref_quat),
            tag_to_base=make_tf(base_xyz, base_quat),
            max_speed_mps=p('max_speed_mps', 1.0).value,
            max_yaw_rate_rps=p('max_yaw_rate_rps', 4.0).value,
            max_tilt_rad=p('max_tilt_rad', 0.6).value,
        ))
        self.pub = self.create_publisher(PoseStamped, '/orion/ground_truth_pose', 10)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.last_rover_stamp = None
        self.create_timer(1.0 / rate, self._tick)

    def _tick(self):
        try:
            ref = self.tf_buffer.lookup_transform(self.camera_frame, self.ref_frame, Time())
            rover = self.tf_buffer.lookup_transform(self.camera_frame, self.rover_frame, Time())
        except TransformException:
            return
        stamp = Time.from_msg(rover.header.stamp)
        if self.last_rover_stamp is not None and stamp <= self.last_rover_stamp:
            return  # no new image of the rover tag
        self.last_rover_stamp = stamp
        pose = self.solver.solve(stamp.nanoseconds * 1e-9,
                                 _tf_to_matrix(ref), _tf_to_matrix(rover))
        if pose is None:
            self.get_logger().warn(
                f'rejected implausible tag detection ({self.solver.rejected} so far)',
                throttle_duration_sec=2.0)
            return
        msg = PoseStamped()
        msg.header.stamp = rover.header.stamp
        msg.header.frame_id = 'sandbox'
        msg.pose.position.x, msg.pose.position.y = pose.x, pose.y
        msg.pose.orientation.z = math.sin(pose.yaw / 2.0)
        msg.pose.orientation.w = math.cos(pose.yaw / 2.0)
        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = GroundTruthNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
