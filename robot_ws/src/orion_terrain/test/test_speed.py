"""Body-speed Kalman filter, pose differentiation, source priority/hand-over, confidence."""

import math

import numpy as np
from orion_terrain.estimator import TerrainEstimator
from orion_terrain.imu import quat_from_rpy
from orion_terrain.params import DEFAULT_JOINT_NAMES
from orion_terrain.speed import BodySpeedKalman, confidence, PoseDifferentiator, SpeedSource
import pytest

LEVEL = (0.0, 0.0, 0.0, 1.0)
G = 9.81


# --- Kalman filter -------------------------------------------------------------------------
def test_variance_grows_linearly_while_imu_only():
    kf = BodySpeedKalman(q_accel=0.1, std0=0.0)
    kf.predict_to(0.0, 0.0)
    for k in range(1, 401):
        kf.predict_to(k * 0.01, 0.0)
    assert kf.var == pytest.approx(0.1 ** 2 * 4.0)          # P = q^2 t
    assert kf.std == pytest.approx(0.2)


def test_variance_contracts_on_fix_below_prior_and_measurement():
    kf = BodySpeedKalman(q_accel=0.1, std0=0.2)
    kf.predict_to(0.0, 0.0)
    kf.update(0.3, 0.02)
    assert kf.std < 0.02
    assert kf.v == pytest.approx(0.3 * 0.04 / (0.04 + 0.0004))
    prior = kf.var
    kf.update(0.3, 0.05)
    assert kf.var < prior


def test_constant_bias_error_is_b_t_and_inside_one_sigma_for_documented_horizon():
    """Q models white accel noise only; a 0.02 m/s^2 bias gives error 0.02 t (documented)."""
    kf = BodySpeedKalman(q_accel=0.1, std0=0.0)
    kf.predict_to(0.0, 0.0)
    bias = 0.02
    for k in range(1, 401):
        kf.predict_to(k * 0.01, bias)                       # true accel 0
    assert kf.v == pytest.approx(bias * 4.0)
    assert kf.v < kf.std                                    # 0.08 < 0.2
    # The band stops being conservative only after (q / b)^2 = 25 s.
    t_cross = (0.1 / bias) ** 2
    assert bias * t_cross == pytest.approx(0.1 * math.sqrt(t_cross))


def test_delayed_fix_is_propagated_with_integrated_acceleration():
    kf = BodySpeedKalman(q_accel=0.1, std0=1.0)
    a = 0.5
    kf.predict_to(0.0, a)
    for k in range(1, 101):
        kf.predict_to(k * 0.01, a)
    # True v(t) = a t. A fix measured at t = 0.9 (0.45 m/s) arrives at t = 1.0.
    kf.update(0.45, 0.001, t_meas=0.9)
    assert kf.v == pytest.approx(0.5, abs=1e-3)


# --- pose differentiation --------------------------------------------------------------------
def test_savgol_is_exact_for_constant_acceleration_and_projects_on_body_x():
    yaw = 0.7
    q = quat_from_rpy(0, 0, yaw)
    d = PoseDifferentiator(5, 0.25)
    a, v0 = 0.4, 0.2
    out = None
    for k in range(5):
        t = 10.0 + k * 0.05 + (0.004 if k == 2 else 0.0)    # timestamp jitter is fine
        s = v0 * t + 0.5 * a * t * t
        out = d.push(t, (s * math.cos(yaw), s * math.sin(yaw), 0.0), q)
    t_c, v = out
    assert t_c == pytest.approx(10.104)
    assert v == pytest.approx(v0 + a * t_c, abs=1e-6)


def test_savgol_needs_full_window_and_resets_after_gap():
    d = PoseDifferentiator(5, 0.25)
    results = [d.push(k * 0.05, (k * 0.01, 0, 0), LEVEL) for k in range(5)]
    assert results[:4] == [None] * 4 and results[4] is not None
    assert d.push(1.0, (0.1, 0, 0), LEVEL) is None           # 0.8 s gap -> buffer cleared


def test_savgol_rejects_even_window():
    with pytest.raises(ValueError):
        PoseDifferentiator(4)


# --- source priority and hand-over -------------------------------------------------------------
class Rig:
    """Rover at constant speed v with level IMU, feeding the chosen fix sources."""

    def __init__(self, v=0.3):
        self.v = v
        self.est = TerrainEstimator()
        self.t = 0.0

    def run(self, seconds, external=True, lidar=True, lidar_value=None, imu=True):
        for _ in range(int(round(seconds * 100))):
            t = self.t = self.t + 0.01
            k = int(round(t * 100))
            if imu:
                self.est.on_imu(t, LEVEL, (0, 0, 0), (0, 0, G))
            if external and k % 5 == 0:
                self.est.on_ground_truth_pose(t, (self.v * t, 0, 0), LEVEL)
            if lidar and k % 10 == 0:
                self.est.on_lidar_odom(t, self.v if lidar_value is None else lidar_value)
            self.est.on_wheel_odom(t, self.v)
        return self.est.estimate(self.t)


def test_external_has_priority_and_lidar_is_not_fused_meanwhile():
    rig = Rig()
    e = rig.run(3.0, lidar_value=1.0)                         # lidar lies, external is right
    assert e.body_speed_source == SpeedSource.EXTERNAL
    assert e.body_speed == pytest.approx(0.3, abs=0.005)


def test_handover_external_to_lidar_to_imu_to_none():
    rig = Rig()
    rig.run(2.0)
    e = rig.run(0.2, external=False)                          # < 0.25 s: still EXTERNAL
    assert e.body_speed_source == SpeedSource.EXTERNAL
    e = rig.run(0.5, external=False)
    assert e.body_speed_source == SpeedSource.LIDAR_ODOM
    assert e.time_since_fix < 0.11
    e = rig.run(0.6, external=False, lidar=False)
    assert e.body_speed_source == SpeedSource.IMU_ONLY
    assert e.time_since_fix == pytest.approx(0.6, abs=0.11)
    e = rig.run(0.6, external=False, lidar=False, imu=False)
    assert e.body_speed_source == SpeedSource.NONE


def test_lidar_is_fused_after_handover():
    rig = Rig()
    rig.run(2.0)
    e = rig.run(3.0, external=False, lidar_value=0.5)
    assert e.body_speed_source == SpeedSource.LIDAR_ODOM
    assert e.body_speed == pytest.approx(0.5, abs=0.01)


def test_external_takes_over_again_when_it_returns():
    rig = Rig()
    rig.run(2.0, external=False, lidar_value=0.5)
    e = rig.run(1.0)
    assert e.body_speed_source == SpeedSource.EXTERNAL
    assert e.body_speed == pytest.approx(0.3, abs=0.01)


def test_wheel_odometry_is_never_a_speed_source():
    est = TerrainEstimator()
    for k in range(1000):
        t = k * 0.01
        est.on_imu(t, LEVEL, (0, 0, 0), (0, 0, G))
        est.on_wheel_odom(t, 0.4)                             # wheels spinning, rover stuck
        est.on_joint_state(t, list(DEFAULT_JOINT_NAMES), [0.4 / 0.12] * 4, [0.0] * 4)
    e = est.estimate(10.0)
    assert e.body_speed == 0.0
    assert e.body_speed_source == SpeedSource.IMU_ONLY
    assert np.allclose(e.slip, 1.0)


# --- confidence ------------------------------------------------------------------------------
def test_confidence_formula():
    assert confidence(0.0, 0.3, 0.3) == pytest.approx(1.0)
    assert confidence(3.0, 0.3, 0.3) == pytest.approx(math.exp(-1))
    assert confidence(0.0, 0.55, 0.3) == pytest.approx(0.5)
    assert confidence(0.0, 1.0, 0.3) == 0.0
    assert confidence(math.inf, 0.3, 0.3) == 0.0
    assert confidence(0.0, None, 0.3) == 0.0


def test_confidence_drops_below_0p3_within_4s_of_fix_dropout():
    rig = Rig()
    e = rig.run(2.0)
    assert e.confidence > 0.95
    history = [rig.run(0.05, external=False, lidar=False) for _ in range(80)]
    first_low = next(i for i, h in enumerate(history) if h.confidence < 0.3)
    assert (first_low + 1) * 0.05 <= 4.0
    assert all(h.confidence < 0.3 for h in history[first_low:])
    assert history[-1].time_since_fix == pytest.approx(4.0, abs=0.06)


def test_confidence_zero_before_any_fix():
    rig = Rig()
    e = rig.run(1.0, external=False, lidar=False)
    assert e.confidence == 0.0
    assert math.isinf(e.time_since_fix)


def test_wheel_speed_from_joints_when_wheel_odom_is_missing():
    est = TerrainEstimator()
    for k in range(100):
        t = k * 0.01
        est.on_imu(t, LEVEL, (0, 0, 0), (0, 0, G))
        est.on_lidar_odom(t, 0.3)
        est.on_joint_state(t, list(DEFAULT_JOINT_NAMES), [0.55 / 0.12] * 4, [0.0] * 4)
    assert est.estimate(1.0).confidence == pytest.approx(0.5, abs=0.02)   # |0.55-0.3|/0.5
