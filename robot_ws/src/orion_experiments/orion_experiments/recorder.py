"""
rosbag2 recording of a trial through a `ros2 bag record` subprocess.

Why a subprocess and not rosbag2_py.Recorder in-process: Recorder.record() blocks its thread,
owns its own rclcpp context and signal handling, and a crash in the recorder would take the
trial runner (and the trial's metadata) down with it. The CLI is the tool every ROS user
already trusts, finalises the bag (metadata.yaml) on SIGINT, and can be killed on its own.
"""

from __future__ import annotations

import os
import signal
import subprocess
from typing import Optional, Sequence

# Every topic SPEC sections 4-8 need to compute the metrics and to replay a trial.
SPEC_TOPICS = (
    '/joint_states', '/imu/data', '/scan', '/tf', '/tf_static',
    '/orion/terrain_state', '/orion/safety_status', '/orion/policy_status',
    '/orion/rover_health', '/orion/estop',
    '/orion/ground_truth_pose', '/odom_rf2o', '/diff_drive_controller/odom',
    '/cmd_vel/estop', '/cmd_vel/teleop', '/cmd_vel/safety', '/cmd_vel/escape',
    '/cmd_vel/policy', '/cmd_vel/nav', '/diff_drive_controller/cmd_vel',
)


class RecorderError(RuntimeError):
    pass


class BagRecorder:
    """Start/stop one `ros2 bag record -s mcap` process writing to bag_path."""

    def __init__(self, bag_path: str, topics: Sequence[str] = SPEC_TOPICS,
                 use_sim_time: bool = False, stop_timeout_s: float = 15.0):
        self.bag_path = bag_path
        self.topics = list(topics)
        self.use_sim_time = use_sim_time
        self.stop_timeout_s = stop_timeout_s
        self._proc: Optional[subprocess.Popen] = None
        self._log = None

    def command(self) -> list:
        cmd = ['ros2', 'bag', 'record', '-s', 'mcap', '-o', self.bag_path]
        if self.use_sim_time:
            cmd.append('--use-sim-time')
        return cmd + ['--topics'] + self.topics

    def start(self) -> None:
        if os.path.exists(self.bag_path):
            raise RecorderError(f'{self.bag_path} already exists; refusing to overwrite')
        self._log = open(self.bag_path + '.record.log', 'w')
        # Own process group, so SIGINT reaches ros2 and the recorder it starts.
        self._proc = subprocess.Popen(self.command(), stdout=self._log,
                                      stderr=subprocess.STDOUT, start_new_session=True)

    def running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def stop(self) -> int:
        """SIGINT (bag finalised), escalating to SIGTERM/SIGKILL; returns the exit code."""
        if self._proc is None:
            raise RecorderError('recorder was never started')
        code = self._proc.poll()
        if code is None:
            for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGKILL):
                try:
                    os.killpg(self._proc.pid, sig)
                except ProcessLookupError:
                    break
                try:
                    code = self._proc.wait(timeout=self.stop_timeout_s)
                    break
                except subprocess.TimeoutExpired:
                    continue
        self._log.close()
        return code if code is not None else self._proc.wait()
