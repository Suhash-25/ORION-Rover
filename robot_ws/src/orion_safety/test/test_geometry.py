import math

from orion_safety.geometry import accel_to_roll_pitch
from orion_safety.geometry import point_in_polygon
from orion_safety.geometry import predict_pose
from orion_safety.geometry import quaternion_to_rpy
from orion_safety.geometry import signed_distance_inside
import pytest

SQUARE = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)]


def quat(roll, pitch, yaw):
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return (sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy, cr * cp * cy + sr * sp * sy)


@pytest.mark.parametrize('rpy', [(0.3, 0.0, 0.0), (0.0, -0.4, 0.0), (0.2, 0.1, 2.5)])
def test_quaternion_round_trip(rpy):
    assert quaternion_to_rpy(*quat(*rpy)) == pytest.approx(rpy)


def test_accel_tilt():
    g = 9.81
    roll, pitch = accel_to_roll_pitch(0.0, g * math.sin(0.3), g * math.cos(0.3))
    assert (roll, pitch) == pytest.approx((0.3, 0.0))
    # Nose up by 0.2 rad: gravity's specific force gains a -x component in the body frame.
    roll, pitch = accel_to_roll_pitch(-g * math.sin(0.2), 0.0, g * math.cos(0.2))
    assert (roll, pitch) == pytest.approx((0.0, 0.2))


def test_signed_distance():
    assert signed_distance_inside(0.0, 0.0, SQUARE) == pytest.approx(1.0)
    assert signed_distance_inside(0.9, 0.0, SQUARE) == pytest.approx(0.1)
    assert signed_distance_inside(1.5, 0.0, SQUARE) == pytest.approx(-0.5)
    assert signed_distance_inside(2.0, 2.0, SQUARE) == pytest.approx(-math.sqrt(2))
    assert point_in_polygon(0.0, 0.0, list(reversed(SQUARE)))


def test_predict_straight_and_arc():
    assert predict_pose(0.0, 0.0, math.pi / 2, 0.5, 0.0, 1.0) == pytest.approx(
        (0.0, 0.5, math.pi / 2), abs=1e-9)
    # Quarter circle of radius 1 to the left, starting along +x.
    x, y, yaw = predict_pose(0.0, 0.0, 0.0, math.pi / 2, math.pi / 2, 1.0)
    assert (x, y, yaw) == pytest.approx((1.0, 1.0, math.pi / 2))
    # Spin in place goes nowhere.
    assert predict_pose(0.3, 0.2, 0.0, 0.0, 1.0, 1.0)[:2] == pytest.approx((0.3, 0.2))
