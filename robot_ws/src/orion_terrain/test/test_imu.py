"""Gravity compensation and tilt conventions (see orion_terrain/imu.py)."""

import math

from conftest import RecordingLogger
import numpy as np
from orion_terrain.estimator import TerrainEstimator
from orion_terrain.imu import body_acceleration, quat_from_rpy, roll_pitch, rotation_matrix
import pytest

G = 9.81


def specific_force(a_body, quat):
    """Return what an ideal IMU reads: f = a - g, in the body frame."""
    return np.asarray(a_body) + G * rotation_matrix(quat)[2, :]


def test_level_rest_reads_plus_g_and_compensates_to_zero():
    q = quat_from_rpy(0, 0, 0)
    assert np.allclose(specific_force([0, 0, 0], q), [0, 0, G])
    assert np.allclose(body_acceleration(q, [0, 0, G]), 0.0, atol=1e-12)


@pytest.mark.parametrize('roll,pitch,yaw', [(0.0, 0.3, 0.0), (0.2, 0.0, 1.0),
                                            (-0.25, -0.35, -2.5), (0.0, -0.4, 0.7)])
def test_tilted_imu_at_rest_has_zero_body_acceleration(roll, pitch, yaw):
    q = quat_from_rpy(roll, pitch, yaw)
    f = specific_force([0, 0, 0], q)
    assert abs(f[0]) > 0.5 or abs(f[1]) > 0.5     # naive use of f would be very wrong
    assert np.allclose(body_acceleration(q, f), 0.0, atol=1e-9)


def test_tilted_and_accelerating_recovers_a_x():
    q = quat_from_rpy(0.1, -0.3, 0.5)
    a = [0.7, -0.2, 0.0]
    assert np.allclose(body_acceleration(q, specific_force(a, q)), a, atol=1e-9)


def test_nose_down_pitch_is_positive_and_gravity_pulls_forward():
    q = quat_from_rpy(0.0, 0.2, 0.0)
    assert roll_pitch(q) == pytest.approx((0.0, 0.2))
    # Nose down at rest: gravity has a +x component, so the IMU reads f_x = -g sin(pitch).
    assert specific_force([0, 0, 0], q)[0] == pytest.approx(-G * math.sin(0.2))


def test_roll_pitch_round_trip():
    for r, p, y in [(0.1, 0.2, 0.3), (-0.3, 0.1, -3.0), (0.0, -0.45, 1.2)]:
        assert roll_pitch(quat_from_rpy(r, p, y)) == pytest.approx((r, p))


def test_zero_quaternion_rejected():
    with pytest.raises(ValueError):
        rotation_matrix((0, 0, 0, 0))


def test_estimator_on_slope_at_rest_does_not_drift():
    """Parked on a 20 deg slope for 10 s: IMU-only speed stays ~0."""
    q = quat_from_rpy(0.0, -math.radians(20), 0.4)
    f = specific_force([0, 0, 0], q)
    est = TerrainEstimator()
    for k in range(1000):
        est.on_imu(k * 0.01, q, (0, 0, 0), f)
    e = est.estimate(10.0)
    assert abs(e.body_speed) < 1e-6


def test_estimator_without_orientation_holds_speed_and_warns():
    log = RecordingLogger()
    est = TerrainEstimator(logger=log)
    for k in range(100):
        est.on_imu(k * 0.01, (0, 0, 0, 0), (0, 0, 0), (1.0, 0, G))
    assert est.estimate(1.0).body_speed == 0.0
    assert log.count('no orientation') == 1
