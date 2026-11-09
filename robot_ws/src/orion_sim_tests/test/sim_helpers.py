"""Shared helpers for ORION headless-Gazebo launch tests."""

import os
import subprocess
import time

from ament_index_python.packages import get_package_share_directory
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data


def include_bringup(launch_file, **args):
    """Include an orion_bringup launch file headless, without the teleop terminal."""
    path = os.path.join(
        get_package_share_directory('orion_bringup'), 'launch', launch_file)
    launch_args = {'headless': 'true', 'teleop': 'false'}
    launch_args.update(args)
    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(path),
        launch_arguments=launch_args.items(),
    )


class Probe(Node):
    """A test node that records the latest message on chosen topics."""

    def __init__(self, name='orion_sim_probe'):
        super().__init__(name, parameter_overrides=[
            rclpy.parameter.Parameter('use_sim_time', value=True)])
        self.latest = {}
        self.counts = {}
        self.stamps = {}

    def watch(self, msg_type, topic, sensor_qos=False):
        qos = qos_profile_sensor_data if sensor_qos else 10

        def cb(msg, topic=topic):
            self.latest[topic] = msg
            self.counts[topic] = self.counts.get(topic, 0) + 1
            self.stamps.setdefault(topic, []).append(time.monotonic())

        self.create_subscription(msg_type, topic, cb, qos)

    def spin_until(self, predicate, timeout_s):
        """Spin until predicate() is true; return whether it became true."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            if predicate():
                return True
        return False

    def spin_for(self, seconds):
        self.spin_until(lambda: False, seconds)


def kill_leftover_gazebo():
    """Kill Gazebo servers left behind by `gz sim`'s ruby wrapper.

    launch stops the wrapper, but the server it forked can survive and keep
    publishing into the next test. Only processes in this test's partition die.
    """
    partition = os.environ.get('GZ_PARTITION', '')
    out = subprocess.run(['pgrep', '-f', 'gz sim'], capture_output=True, text=True).stdout
    for pid in out.split():
        try:
            with open(f'/proc/{pid}/environ', 'rb') as f:
                env = f.read().split(b'\0')
        except OSError:
            continue
        if not partition or f'GZ_PARTITION={partition}'.encode() in env:
            subprocess.run(['kill', '-9', pid], check=False)
