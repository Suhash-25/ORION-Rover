"""Slip: signs, floor, standstill, turning, low-pass, and that obs_spec is the formula used."""

import math

import numpy as np
from orion_learning import obs_spec
from orion_learning.obs_spec import TRACK_WIDTH_M, WHEEL_RADIUS_M
from orion_terrain.estimator import TerrainEstimator
from orion_terrain.filters import LowPass
from orion_terrain.params import DEFAULT_JOINT_NAMES
import pytest

LEVEL = (0.0, 0.0, 0.0, 1.0)
REST_F = (0.0, 0.0, 9.81)


def run_constant(w, v_fix, yaw_rate=0.0, seconds=3.0):
    """Feed constant wheel speeds, IMU and a lidar speed fix; return the final estimate."""
    est = TerrainEstimator()
    for k in range(int(seconds * 100)):
        t = k * 0.01
        est.on_imu(t, LEVEL, (0.0, 0.0, yaw_rate), REST_F)
        est.on_lidar_odom(t, v_fix)
        est.on_joint_state(t, list(DEFAULT_JOINT_NAMES), w, [0.0] * 4)
    return est.estimate(seconds)


def test_spinning_is_positive_skidding_is_negative():
    v = 0.3
    spin = run_constant([v / WHEEL_RADIUS_M * 1.5] * 4, v)
    skid = run_constant([v / WHEEL_RADIUS_M * 0.5] * 4, v)
    assert np.allclose(spin.slip, 1 - 1 / 1.5, atol=0.01)       # (1.5v - v) / 1.5v
    assert np.allclose(skid.slip, -0.5, atol=0.01)              # (0.5v - v) / v
    assert np.allclose(spin.slip_side, spin.slip[:2], atol=1e-9)


def test_standstill_is_zero_slip():
    e = run_constant([0.0] * 4, 0.0)
    assert np.allclose(e.slip, 0.0, atol=1e-6)


def test_floor_bounds_slip_at_crawling_speed():
    # 5 mm/s rim speed, body still: without the 0.02 m/s floor this would read +1.
    e = run_constant([0.005 / WHEEL_RADIUS_M] * 4, 0.0)
    assert np.allclose(e.slip, 0.25, atol=0.01)


def test_full_spin_when_stuck_reads_plus_one():
    e = run_constant([0.3 / WHEEL_RADIUS_M] * 4, 0.0)
    assert np.allclose(e.slip, 1.0, atol=1e-3)


def test_turning_in_place_without_slip_reads_zero():
    yaw = 1.0
    v_side = yaw * TRACK_WIDTH_M / 2
    w = np.array([-v_side, v_side, -v_side, v_side]) / WHEEL_RADIUS_M
    e = run_constant(w, 0.0, yaw_rate=yaw)
    assert np.allclose(e.slip, 0.0, atol=0.01)
    assert e.body_yaw_rate == pytest.approx(yaw)


def test_turning_with_left_spin_only_flags_left():
    yaw, v = 0.5, 0.3
    v_l, v_r = v - yaw * TRACK_WIDTH_M / 2, v + yaw * TRACK_WIDTH_M / 2
    w = np.array([v_l / 0.6, v_r, v_l / 0.6, v_r]) / WHEEL_RADIUS_M    # left s = 0.4
    e = run_constant(w, v, yaw_rate=yaw)
    assert e.slip_side[0] == pytest.approx(0.4, abs=0.01)
    assert e.slip_side[1] == pytest.approx(0.0, abs=0.01)


def test_estimator_uses_obs_spec_formula(monkeypatch):
    calls = []
    real = obs_spec.wheel_slip

    def spy(*args):
        calls.append(args)
        return real(*args)

    monkeypatch.setattr('orion_terrain.estimator.wheel_slip', spy)
    run_constant([2.0] * 4, 0.2, seconds=0.1)
    assert calls, 'estimator did not call obs_spec.wheel_slip'


def test_low_pass_time_constant():
    lp = LowPass(0.25)
    lp.update(0.0, [0.0])
    for k in range(1, 26):                     # 0.25 s at 100 Hz after a unit step
        y = lp.update(k * 0.01, [1.0])
    assert y[0] == pytest.approx(1 - math.exp(-1), abs=1e-6)


def test_slip_is_low_passed_with_quarter_second_tau():
    est = TerrainEstimator()
    w0, w1 = [0.3 / WHEEL_RADIUS_M] * 4, [0.6 / WHEEL_RADIUS_M] * 4    # slip 0 -> 0.5
    for k in range(200):
        t = k * 0.01
        est.on_imu(t, LEVEL, (0, 0, 0), REST_F)
        est.on_lidar_odom(t, 0.3)
        est.on_joint_state(t, list(DEFAULT_JOINT_NAMES), w0 if t < 1.0 else w1, [0.0] * 4)
        if abs(t - 1.25) < 1e-9:
            mid = est.estimate(t).slip[0]
    assert mid == pytest.approx(0.5 * (1 - math.exp(-1)), abs=0.02)
