"""
Drive the Gazebo rover through a varied flat-ground pattern for effort-model calibration.

Procedure used to try to fit a Gazebo effort model (result in the header of
config/effort_model_sim.yaml and docs/engineering-log/agents/wp3-terrain.md):

    ros2 launch orion_bringup gazebo.launch.py headless:=true teleop:=false &
    ros2 bag record -o sim_fit /joint_states /imu/data &
    python3 test/sim/calibration_drive.py --duration 180
    ros2 run orion_terrain fit_effort_model sim_fit --sim --i-stall 3.0 \
        --max-yaw-rate 0.05 -o effort_model_sim.yaml

Segments of 2-4 s with random v in [-0.5, 0.5] m/s and w in [-1, 1] rad/s, mixed with
sinusoidal speed sweeps, published as TwistStamped on /diff_drive_controller/cmd_vel.
"""

import argparse
import math

from geometry_msgs.msg import TwistStamped
import numpy as np
import rclpy
from rclpy.node import Node


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--duration', type=float, default=180.0, help='sim seconds')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    rclpy.init()
    node = Node('orion_calibration_drive', parameter_overrides=[
        rclpy.parameter.Parameter('use_sim_time', value=True)])
    pub = node.create_publisher(TwistStamped, '/diff_drive_controller/cmd_vel', 10)
    rng = np.random.default_rng(args.seed)
    while node.get_clock().now().nanoseconds == 0:
        rclpy.spin_once(node, timeout_sec=0.1)
    t0 = node.get_clock().now().nanoseconds * 1e-9
    seg_end, v, w, sweep, last = 0.0, 0.0, 0.0, False, -1.0
    msg = TwistStamped()
    while rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.02)
        t = node.get_clock().now().nanoseconds * 1e-9 - t0
        if t > args.duration:
            break
        if t - last < 0.05:                       # 20 Hz in sim time
            continue
        last = t
        if t >= seg_end:
            seg_end = t + rng.uniform(2.0, 4.0)
            sweep = rng.random() < 0.3
            v, w = rng.uniform(-0.5, 0.5), rng.uniform(-1.0, 1.0) * (rng.random() < 0.5)
        msg.header.stamp = node.get_clock().now().to_msg()
        msg.twist.linear.x = 0.4 * math.sin(2.0 * t) if sweep else v
        msg.twist.angular.z = w
        pub.publish(msg)
    msg.twist.linear.x = msg.twist.angular.z = 0.0
    pub.publish(msg)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
