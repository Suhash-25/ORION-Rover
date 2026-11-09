"""
Turn ROS sensor data into the observation defined by orion_learning.obs_spec.

Nothing here re-defines the observation layout: the builder only converts ROS inputs to the
physical units that ``obs_spec.build_observation`` takes and calls it. Pure Python (no rclpy)
so it can be unit-tested against obs_spec directly.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from typing import Optional, Sequence

import numpy as np
from orion_learning import obs_spec

STANDARD_GRAVITY = 9.81  # m/s^2


class InputNotReady(RuntimeError):
    """An input needed for the observation is missing or stale; the message says which."""


def quat_to_rpy(x: float, y: float, z: float, w: float) -> tuple[float, float, float]:
    """Convert a unit quaternion (x, y, z, w) to (roll, pitch, yaw) in rad, ZYX order."""
    sinr = 2.0 * (w * x + y * z)
    cosr = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr, cosr)
    sinp = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    pitch = math.asin(sinp)
    siny = 2.0 * (w * z + x * y)
    cosy = 1.0 - 2.0 * (y * y + z * z)
    return roll, pitch, math.atan2(siny, cosy)


def gravity_in_body(roll: float, pitch: float, g: float = STANDARD_GRAVITY) -> np.ndarray:
    """
    Return the reaction to gravity an accelerometer reads at rest, body frame, m/s^2.

    For a level IMU this is (0, 0, +g). Yaw does not matter.
    """
    return g * np.array([-math.sin(pitch),
                         math.sin(roll) * math.cos(pitch),
                         math.cos(roll) * math.cos(pitch)])


def gravity_free_accel(linear_acceleration: Sequence[float], roll: float, pitch: float,
                       g: float = STANDARD_GRAVITY) -> np.ndarray:
    """Body-frame specific force (m/s^2, as sensor_msgs/Imu reports it) minus gravity."""
    return np.asarray(linear_acceleration, dtype=np.float64) - gravity_in_body(roll, pitch, g)


@dataclass
class _Stamped:
    t: float      # s, receive time on the node clock
    data: dict


class ObservationBuilder:
    """
    Caches the latest inputs and builds one obs_spec observation per control tick.

    wheel_joints: joint names in obs_spec.WHEEL_ORDER (FL, FR, RL, RR).
    effort_max: normaliser for /joint_states effort, same unit as the effort (A on the rover,
        N.m in Gazebo).
    gravity: local g in m/s^2, used only to remove gravity from the accelerometer.
    timeouts: maximum input age in s before the observation is refused.
    """

    def __init__(self, wheel_joints: Sequence[str], effort_max: float,
                 gravity: float = STANDARD_GRAVITY, joint_timeout: float = 0.3,
                 imu_timeout: float = 0.5, terrain_timeout: float = 0.5):
        if len(wheel_joints) != len(obs_spec.WHEEL_ORDER):
            raise ValueError(f'need {len(obs_spec.WHEEL_ORDER)} wheel joints in order '
                             f'{obs_spec.WHEEL_ORDER}, got {list(wheel_joints)}')
        if effort_max <= 0.0:
            raise ValueError('effort_max must be > 0')
        self.wheel_joints = tuple(wheel_joints)
        self.effort_max = float(effort_max)
        self.gravity = float(gravity)
        self._timeouts = {'joint_states': joint_timeout, 'imu': imu_timeout,
                          'terrain_state': terrain_timeout}
        self._latest: dict[str, _Stamped] = {}

    def update_joint_state(self, t: float, names: Sequence[str], velocity: Sequence[float],
                           effort: Sequence[float]) -> None:
        """
        Store one /joint_states message. velocity in rad/s, effort in effort_max units.

        Joints are matched by name; a message missing a wheel or its effort is rejected with
        ValueError so that a misconfigured driver is loud, not a silent zero.
        """
        index = {n: i for i, n in enumerate(names)}
        missing = [j for j in self.wheel_joints if j not in index]
        if missing:
            raise ValueError(f'/joint_states lacks wheel joints {missing}')
        if len(velocity) != len(names) or len(effort) != len(names):
            raise ValueError('/joint_states needs velocity and effort for every joint '
                             f'(names {len(names)}, velocity {len(velocity)}, '
                             f'effort {len(effort)})')
        order = [index[j] for j in self.wheel_joints]
        self._latest['joint_states'] = _Stamped(t, {
            'wheel_speed': [float(velocity[i]) for i in order],
            'wheel_effort': [float(effort[i]) for i in order]})

    def update_imu(self, t: float, orientation_xyzw: Sequence[float],
                   angular_velocity: Sequence[float],
                   linear_acceleration: Sequence[float]) -> None:
        """Store one /imu/data message (rad/s, m/s^2 specific force, body = IMU frame)."""
        roll, pitch, _ = quat_to_rpy(*orientation_xyzw)
        accel = gravity_free_accel(linear_acceleration, roll, pitch, self.gravity)
        self._latest['imu'] = _Stamped(t, {
            'roll': roll, 'pitch': pitch, 'yaw_rate': float(angular_velocity[2]),
            'body_accel_xy': [float(accel[0]), float(accel[1])]})

    def update_terrain(self, t: float, body_speed: float, confidence: float,
                       slip_side: Sequence[float], sinkage_proxy: float) -> None:
        """Store one TerrainState (body_speed m/s, slip_side [-1, 1], sinkage [0, 1])."""
        self._latest['terrain_state'] = _Stamped(t, {
            'body_speed': float(body_speed), 'speed_confidence': float(confidence),
            'slip_side': [float(slip_side[0]), float(slip_side[1])],
            'sinkage_proxy': float(sinkage_proxy)})

    def terrain(self, now: float) -> dict:
        """Latest TerrainState fields; raises InputNotReady when missing or stale."""
        return self._fresh('terrain_state', now)

    def _fresh(self, key: str, now: float) -> dict:
        item: Optional[_Stamped] = self._latest.get(key)
        if item is None:
            raise InputNotReady(f'no {key} received yet')
        age = now - item.t
        if age > self._timeouts[key]:
            raise InputNotReady(f'{key} is {age:.2f} s old (limit {self._timeouts[key]} s)')
        return item.data

    def build(self, now: float, prev_cmd: Sequence[float]) -> np.ndarray:
        """
        Observation at time now (s) given our last published (v m/s, w rad/s).

        Raises InputNotReady when any input is missing or stale.
        """
        joints = self._fresh('joint_states', now)
        imu = self._fresh('imu', now)
        terrain = self._fresh('terrain_state', now)
        return obs_spec.build_observation(
            prev_cmd=prev_cmd, effort_max=self.effort_max, **joints, **imu, **terrain)


class History:
    """
    The adapter's input: the last HISTORY_LEN rows of [obs, action], oldest first.

    action is the command that was executed after obs, in obs_spec action units
    (v / MAX_LINEAR_SPEED, w / MAX_ANGULAR_SPEED). Until HISTORY_LEN steps have been seen the
    missing oldest rows are zeros.
    """

    ROW = obs_spec.OBS_DIM + obs_spec.ACTION_DIM

    def __init__(self, length: int = obs_spec.HISTORY_LEN):
        self._rows: deque = deque(maxlen=length)
        self.length = length

    def push(self, obs: np.ndarray, action: Sequence[float]) -> None:
        row = np.concatenate([np.asarray(obs, dtype=np.float32),
                              np.asarray(action, dtype=np.float32)])
        if row.shape != (self.ROW,):
            raise ValueError(f'history row has shape {row.shape}, expected ({self.ROW},)')
        self._rows.append(row)

    def clear(self) -> None:
        self._rows.clear()

    def __len__(self) -> int:
        return len(self._rows)

    def array(self) -> np.ndarray:
        """float32 [length, OBS_DIM + ACTION_DIM], zero rows first while filling."""
        out = np.zeros((self.length, self.ROW), dtype=np.float32)
        if self._rows:
            out[self.length - len(self._rows):] = np.stack(self._rows)
        return out


def twist_to_action(v: float, w: float) -> np.ndarray:
    """(v m/s, w rad/s) in obs_spec action units, clipped to [-1, 1]."""
    return np.clip(np.array([v / obs_spec.MAX_LINEAR_SPEED, w / obs_spec.MAX_ANGULAR_SPEED],
                            dtype=np.float32), -1.0, 1.0)
