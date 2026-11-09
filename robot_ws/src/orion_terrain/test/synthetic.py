"""
Small kinematic generator of rover sensor streams with known truth (tests only).

Produces (t, topic, msg_dict) tuples in the ROS field layout that
orion_terrain.estimator.estimate_from_messages accepts, plus the truth.

Model: planar skid-steer with known body speed v(t), yaw rate r(t), pitch p(t). Left and right
wheels spin with a known slip s_side(t) >= 0: r_wheel w = v_side / (1 - s). The IMU sees
a_body = (dv/dt, v r, 0) plus gravity, white noise and a random-walk bias; the overhead camera
sees the pose at 20 Hz with noise and dropouts; wheel odometry reports r mean(w) like
diff_drive_controller.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np
from orion_learning.obs_spec import TRACK_WIDTH_M, WHEEL_RADIUS_M

from orion_terrain.imu import quat_from_rpy, rotation_matrix
from orion_terrain.params import DEFAULT_JOINT_NAMES, DEFAULT_TOPICS

T = DEFAULT_TOPICS


def _xyz(v):
    return {'x': float(v[0]), 'y': float(v[1]), 'z': float(v[2])}


def _q(q):
    return {'x': q[0], 'y': q[1], 'z': q[2], 'w': q[3]}


@dataclass
class Scenario:
    duration_s: float = 30.0
    speed: callable = lambda t: min(t, 1.0) * (0.3 + 0.05 * math.sin(0.5 * t))
    yaw_rate: callable = lambda t: 0.2 * math.sin(0.3 * t)
    pitch: callable = lambda t: 0.05 * math.sin(0.2 * t)
    slip_left: callable = lambda t: 0.5 * t / 30.0
    slip_right: callable = lambda t: 0.25 * t / 30.0
    imu_noise: float = 0.05            # m/s^2 per sample
    gyro_noise: float = 0.002          # rad/s
    imu_bias0: float = 0.02            # m/s^2
    bias_walk: float = 0.02            # m/s^2 per sqrt(s)
    wheel_noise: float = 0.05          # rad/s
    pose_noise: float = 0.001          # m
    dropouts: list = field(default_factory=lambda: [(10.0, 12.0), (18.0, 23.0)])
    imu_hz: float = 100.0
    joint_hz: float = 100.0
    pose_hz: float = 20.0
    odom_hz: float = 50.0
    shuffle_joints: bool = True
    seed: int = 1


def in_dropout(t, dropouts):
    return any(a <= t < b for a, b in dropouts)


def generate(sc: Scenario):
    """Return (messages, truth) where truth(t) -> dict of true values at time t."""
    rng = np.random.default_rng(sc.seed)
    dt = 1e-3
    ts = np.arange(0.0, sc.duration_s, dt)
    v = np.array([sc.speed(t) for t in ts])
    r = np.array([sc.yaw_rate(t) for t in ts])
    p = np.array([sc.pitch(t) for t in ts])
    dvdt = np.gradient(v, dt)
    yaw = np.cumsum(r) * dt
    pos = np.zeros((len(ts), 3))
    for k in range(1, len(ts)):
        rot = rotation_matrix(quat_from_rpy(0.0, p[k - 1], yaw[k - 1]))
        pos[k] = pos[k - 1] + rot[:, 0] * v[k - 1] * dt

    def state(t):
        k = min(int(round(t / dt)), len(ts) - 1)
        return k

    msgs = []
    bias = sc.imu_bias0
    for t in np.arange(0.0, sc.duration_s, 1.0 / sc.imu_hz):
        k = state(t)
        q = quat_from_rpy(0.0, p[k], yaw[k])
        rot = rotation_matrix(q)
        a_body = np.array([dvdt[k], v[k] * r[k], 0.0])
        f = a_body + 9.81 * rot[2, :]
        bias += sc.bias_walk * math.sqrt(1.0 / sc.imu_hz) * rng.standard_normal()
        f = f + rng.normal(0.0, sc.imu_noise, 3) + np.array([bias, 0.0, 0.0])
        gyro = np.array([0.0, 0.0, r[k] + rng.normal(0.0, sc.gyro_noise)])
        msgs.append((t, T['imu'], {'orientation': _q(q), 'angular_velocity': _xyz(gyro),
                                   'linear_acceleration': _xyz(f)}))

    def wheel_speeds(k):
        v_l = v[k] - r[k] * TRACK_WIDTH_M / 2
        v_r = v[k] + r[k] * TRACK_WIDTH_M / 2
        s_l, s_r = sc.slip_left(ts[k]), sc.slip_right(ts[k])
        rim_l, rim_r = v_l / (1 - s_l), v_r / (1 - s_r)
        return np.array([rim_l, rim_r, rim_l, rim_r]) / WHEEL_RADIUS_M

    names = list(DEFAULT_JOINT_NAMES)
    for t in np.arange(0.0, sc.duration_s, 1.0 / sc.joint_hz):
        k = state(t)
        w = wheel_speeds(k) + rng.normal(0.0, sc.wheel_noise, 4)
        order = rng.permutation(4) if sc.shuffle_joints else np.arange(4)
        msgs.append((t, T['joint_states'], {
            'name': [names[i] for i in order], 'velocity': [float(w[i]) for i in order],
            'effort': [0.0] * 4}))
    for t in np.arange(0.0, sc.duration_s, 1.0 / sc.odom_hz):
        k = state(t)
        v_wheel = WHEEL_RADIUS_M * float(np.mean(wheel_speeds(k)))
        msgs.append((t, '/diff_drive_controller/odom',      # not an estimator input any more
                     {'twist': {'twist': {'linear': _xyz([v_wheel, 0, 0])}}}))
    for t in np.arange(0.0, sc.duration_s, 1.0 / sc.pose_hz):
        if in_dropout(t, sc.dropouts):
            continue
        k = state(t)
        noisy = pos[k] + rng.normal(0.0, sc.pose_noise, 3)
        msgs.append((t, T['ground_truth_pose'], {'pose': {
            'position': _xyz(noisy), 'orientation': _q(quat_from_rpy(0.0, p[k], yaw[k]))}}))

    def truth(t):
        k = state(t)
        return {'v': v[k], 'yaw_rate': r[k],
                'slip_side': np.array([sc.slip_left(ts[k]), sc.slip_right(ts[k])])}

    return msgs, truth
