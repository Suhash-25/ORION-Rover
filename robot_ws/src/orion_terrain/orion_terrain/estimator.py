"""
Terrain estimator core: plain Python, no rclpy (docs/SPEC.md section 4).

The ROS node, the offline replay (`estimate_from_messages`) and the tests all drive the same
`TerrainEstimator` through `on_message(t, role, msg)`. ``msg`` may be a ROS message, a
rosbags-deserialised message or a plain nested dict with the ROS field names, e.g.
``{'twist': {'twist': {'linear': {'x': 0.3}}}}``.

All times are seconds in ONE time base chosen by the caller (the node uses header stamps,
falling back to receipt time for unstamped messages). Speeds m/s, rates rad/s.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import math
from operator import itemgetter
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np
from orion_learning import estimator as shared
from orion_learning.obs_spec import side_mean, WHEEL_ORDER, wheel_slip

from orion_terrain.effort_model import EffortModel, EffortSource
from orion_terrain.filters import LowPass, WheelAccel
from orion_terrain.imu import body_acceleration, roll_pitch
from orion_terrain.params import DEFAULT_JOINT_NAMES, DEFAULT_TOPICS, TerrainParams
from orion_terrain.speed import BodySpeedKalman, PoseDifferentiator, SpeedSource


@dataclass
class TerrainEstimate:
    """One TerrainState worth of output (field meanings in orion_msgs/TerrainState.msg)."""

    t: float
    slip: np.ndarray                 # (4,) FL FR RL RR, [-1, 1]
    slip_side: np.ndarray            # (2,) left, right
    effort_residual: np.ndarray      # (4,) effort unit of the model
    sinkage_proxy: float             # [0, 1]
    body_speed: float                # m/s
    body_speed_std: float            # m/s, Kalman 1-sigma (not in the message)
    body_yaw_rate: float             # rad/s
    body_speed_source: SpeedSource
    effort_source: EffortSource
    confidence: float                # [0, 1]
    time_since_fix: float            # s, inf before the first fix


def msg_get(msg, path: str):
    """Field at dotted ``path`` of a ROS message, rosbags message or nested dict."""
    for key in path.split('.'):
        msg = msg[key] if isinstance(msg, Mapping) else getattr(msg, key)
    return msg


def msg_vec3(msg, path: str) -> tuple[float, float, float]:
    v = msg_get(msg, path)
    return float(msg_get(v, 'x')), float(msg_get(v, 'y')), float(msg_get(v, 'z'))


def msg_quat(msg, path: str) -> tuple[float, float, float, float]:
    q = msg_get(msg, path)
    return tuple(float(msg_get(q, k)) for k in ('x', 'y', 'z', 'w'))


class TerrainEstimator:
    """
    Body speed, slip, effort residual and sinkage proxy from the rover's sensor streams.

    sim=True: effort is Gazebo joint torque (EFFORT_SOURCE_SIM_TORQUE). sim=False (rover):
    effort is CURRENT or DUTY_ESTIMATE according to RoverHealth.current_sensed, and the
    sinkage proxy is only published with a calibrated model of that same source; otherwise
    sinkage_proxy = 0 and effort_source = NONE, with a warning logged once per reason.

    The maths (Kalman filter, confidence, low-pass, effort residual and its informative-window
    hold, sinkage proxy) is orion_learning.estimator, the code the learning env trains with;
    this class only routes messages and keeps the sample times.
    """

    def __init__(self, params: Optional[TerrainParams] = None,
                 effort_model: Optional[EffortModel] = None, sim: bool = False,
                 joint_names: Sequence[str] = DEFAULT_JOINT_NAMES,
                 logger: Optional[logging.Logger] = None):
        self.p = params or TerrainParams()
        if len(joint_names) != len(WHEEL_ORDER):
            raise ValueError(f'need one joint name per wheel {WHEEL_ORDER}, got {joint_names}')
        self.joint_names = tuple(joint_names)
        self.model = effort_model
        self.sim = sim
        self._log = logger or logging.getLogger('orion_terrain')
        self._warned: set = set()

        self.kf = BodySpeedKalman(shared.KF_ACCEL_SIGMA, 0.0, self.p.initial_speed_std_mps)
        self._pose_diff = PoseDifferentiator(self.p.sg_window, self.p.external_timeout_s)
        self._wheel_accel = WheelAccel(self.p.wheel_accel_tau_s)
        self._slip_lp = LowPass(shared.LOWPASS_TAU_S)
        self._residual_lp = LowPass(shared.LOWPASS_TAU_S)

        self._ax = 0.0                     # gravity-free body a_x, m/s^2 (zero-order hold)
        self._yaw_rate = 0.0               # rad/s, gyro z
        self._pitch = 0.0                  # rad, nose down positive
        self._current_sensed: Optional[bool] = None
        self._t_imu = None
        self._t_external = self._t_lidar = self._t_fix = None

    # --- helpers -------------------------------------------------------------------------
    def warn_once(self, text: str) -> None:
        if text not in self._warned:
            self._warned.add(text)
            self._log.warning(text)

    @staticmethod
    def _fresh(t_last: Optional[float], t: float, timeout: float) -> bool:
        return t_last is not None and t - t_last <= timeout

    def _advance(self, t: float) -> None:
        accel = self._ax if self._fresh(self._t_imu, t, self.p.imu_timeout_s) else 0.0
        self.kf.predict_to(t, accel)

    def speed_source(self, t: float) -> SpeedSource:
        """Highest-priority source that is not stale at time t."""
        if self._fresh(self._t_external, t, self.p.external_timeout_s):
            return SpeedSource.EXTERNAL
        if self._fresh(self._t_lidar, t, self.p.lidar_timeout_s):
            return SpeedSource.LIDAR_ODOM
        if self._fresh(self._t_imu, t, self.p.imu_timeout_s):
            return SpeedSource.IMU_ONLY
        return SpeedSource.NONE

    def effort_source(self) -> EffortSource:
        """Effort source to publish; NONE whenever the sinkage proxy would be unfounded."""
        m = self.model
        if m is None or not m.calibrated:
            what = 'no effort model' if m is None else 'the effort model is not calibrated'
            self.warn_once(f'{what}: publishing sinkage_proxy 0, effort_source NONE')
            return EffortSource.NONE
        if self.sim:
            live = EffortSource.SIM_TORQUE
        elif self._current_sensed is None:
            self.warn_once('no RoverHealth yet, effort source unknown: sinkage_proxy 0')
            return EffortSource.NONE
        else:
            live = EffortSource.CURRENT if self._current_sensed else EffortSource.DUTY_ESTIMATE
        if m.effort_source != live:
            self.warn_once(f'effort model was calibrated on {m.effort_source.name} but the '
                           f'live effort is {live.name}: sinkage_proxy 0, effort_source NONE')
            return EffortSource.NONE
        return live

    # --- inputs --------------------------------------------------------------------------
    def on_message(self, t: float, role: str, msg) -> None:
        """Dispatch one message by role (a key of params.DEFAULT_TOPICS)."""
        if role == 'imu':
            self.on_imu(t, msg_quat(msg, 'orientation'), msg_vec3(msg, 'angular_velocity'),
                        msg_vec3(msg, 'linear_acceleration'))
        elif role == 'joint_states':
            self.on_joint_state(t, list(msg_get(msg, 'name')), msg_get(msg, 'velocity'),
                                msg_get(msg, 'effort'))
        elif role == 'ground_truth_pose':
            self.on_ground_truth_pose(t, msg_vec3(msg, 'pose.position'),
                                      msg_quat(msg, 'pose.orientation'))
        elif role == 'lidar_odom':
            self.on_lidar_odom(t, float(msg_get(msg, 'twist.twist.linear.x')))
        elif role == 'rover_health':
            self.on_rover_health(t, bool(msg_get(msg, 'current_sensed')))
        else:
            raise KeyError(f'unknown input role {role!r}')

    def on_imu(self, t: float, quat, gyro, specific_force) -> None:
        """IMU sample: orientation (x y z w), gyro (rad/s), specific force (m/s^2), IMU frame."""
        try:
            ax = float(body_acceleration(quat, specific_force, self.p.gravity_mps2)[0])
            _, self._pitch = roll_pitch(quat)
        except ValueError:
            self.warn_once('IMU publishes no orientation: cannot remove gravity, '
                           'body speed is held between fixes')
            ax = 0.0
        self._advance(t)
        self._ax, self._yaw_rate = ax, float(gyro[2])
        self._t_imu = t if self._t_imu is None else max(self._t_imu, t)

    def on_joint_state(self, t: float, names: Sequence[str], velocity, effort) -> None:
        """Wheel speeds (rad/s) and efforts, mapped to FL FR RL RR by joint NAME."""
        try:
            idx = [names.index(j) for j in self.joint_names]
        except ValueError:
            self.warn_once(f'/joint_states names {list(names)} lack one of {self.joint_names}')
            return
        w = np.asarray(velocity, dtype=np.float64)[idx]
        self._advance(t)
        dwdt = self._wheel_accel.update(t, w)
        self._slip_lp.update(t, wheel_slip(w, self.kf.v, self._yaw_rate))
        if self.model is None or not self.model.calibrated:
            return
        if len(effort) != len(names):
            self.warn_once('/joint_states carries no effort: no effort residual')
            return
        e = np.asarray(effort, dtype=np.float64)[idx]
        self._residual_lp.update(t, self.model.residual(e, w, dwdt, self._pitch),
                                 mask=self.model.informative(w, self._yaw_rate))

    def on_ground_truth_pose(self, t: float, position, quat) -> None:
        """Overhead-camera pose (world frame, m): EXTERNAL fix, highest priority."""
        fix = self._pose_diff.push(t, position, quat)
        if fix is None:
            return
        t_centre, v = fix
        self._advance(t)
        self.kf.update(v, shared.FIX_SIGMA_EXTERNAL, t_meas=t_centre)
        self._t_external = self._t_fix = t

    def on_lidar_odom(self, t: float, v: float) -> None:
        """rf2o body speed (m/s): LIDAR_ODOM fix, fused only while EXTERNAL is stale."""
        self._t_lidar = t
        if self.speed_source(t) == SpeedSource.EXTERNAL:
            return
        self._advance(t)
        self.kf.update(v, shared.FIX_SIGMA_LIDAR, t_meas=t)
        self._t_fix = t

    def on_rover_health(self, t: float, current_sensed: bool) -> None:
        self._current_sensed = current_sensed

    # --- output --------------------------------------------------------------------------
    def estimate(self, t: float) -> TerrainEstimate:
        """State at time t (s); call at the publish rate."""
        self._advance(t)
        t_since = max(0.0, t - self._t_fix) if self._t_fix is not None else math.inf
        slip = (self._slip_lp.value.copy() if self._slip_lp.value is not None
                else np.zeros(len(WHEEL_ORDER)))
        source = self.effort_source()
        if source == EffortSource.NONE or self._residual_lp.value is None:
            residual, sinkage = np.zeros(len(WHEEL_ORDER)), 0.0
        else:
            residual = self._residual_lp.value.copy()
            sinkage = self.model.sinkage_proxy(residual)
        return TerrainEstimate(
            t=t, slip=slip, slip_side=side_mean(slip), effort_residual=residual,
            sinkage_proxy=sinkage, body_speed=self.kf.v, body_speed_std=self.kf.std,
            body_yaw_rate=self._yaw_rate, body_speed_source=self.speed_source(t),
            effort_source=source,
            confidence=float(shared.speed_confidence(t_since, self.kf.std)),
            time_since_fix=t_since)


def estimate_from_messages(messages: Iterable[tuple[float, str, object]],
                           params: Optional[TerrainParams] = None,
                           effort_model: Optional[EffortModel] = None, sim: bool = False,
                           joint_names: Sequence[str] = DEFAULT_JOINT_NAMES,
                           topics: Optional[Mapping[str, str]] = None,
                           logger: Optional[logging.Logger] = None) -> list[TerrainEstimate]:
    """
    Offline replay: (t s, topic, msg) in any order -> estimates at params.rate_hz.

    Messages are processed sorted by (t, topic), so the result does not depend on input order.

    Messages on topics not in ``topics`` (default DEFAULT_TOPICS) are ignored. The first
    estimate is one period after the first message; each is computed from all messages with
    t <= its tick, exactly as the node would at that instant.
    """
    est = TerrainEstimator(params, effort_model, sim, joint_names, logger)
    role_of = {topic: role for role, topic in (topics or DEFAULT_TOPICS).items()}
    period = 1.0 / est.p.rate_hz
    out: list[TerrainEstimate] = []
    t0, k = None, 1
    for t, topic, msg in sorted(messages, key=itemgetter(0, 1)):
        if t0 is None:
            t0 = t
        while t > t0 + k * period:
            out.append(est.estimate(t0 + k * period))
            k += 1
        role = role_of.get(topic)
        if role is not None and role != 'terrain_state':
            est.on_message(t, role, msg)
    if t0 is not None:
        out.append(est.estimate(t0 + k * period))
    return out


def stamp_seconds(msg, fallback: float) -> float:
    """header.stamp in seconds, or ``fallback`` for an unstamped (zero) or headerless msg."""
    try:
        stamp = msg_get(msg, 'header.stamp')
        sec, nanosec = int(msg_get(stamp, 'sec')), int(msg_get(stamp, 'nanosec'))
    except (AttributeError, KeyError):
        return fallback
    return sec + nanosec * 1e-9 if (sec or nanosec) else fallback
