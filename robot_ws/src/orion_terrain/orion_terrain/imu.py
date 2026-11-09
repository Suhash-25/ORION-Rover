"""
IMU helpers: gravity compensation and tilt.

Frame conventions (REP-103 / REP-145), relied on everywhere in this package:

- The IMU frame is aligned with base_link: x forward, y left, z up (imu_link has rpy 0 0 0
  in orion_description; the real mount must match or the URDF must say otherwise and the
  driver must rotate into imu_link before publishing).
- ``orientation`` is the rotation of the IMU frame relative to a gravity-aligned world frame
  (ENU, z up). Yaw is irrelevant here; only roll and pitch are used.
- ``linear_acceleration`` is the specific force f = a - g in the IMU frame. At rest on level
  ground the IMU reads f = (0, 0, +9.81) m/s^2.

So the gravity-free body acceleration is a_b = f_b + R^T g_w with g_w = (0, 0, -9.81):
a_b = f_b - R^T (0, 0, 9.81), where R maps body vectors to world vectors.

Euler angles are ZYX (yaw-pitch-roll). Pitch is positive nose DOWN (right-hand rotation about
y with z up); a nose-up climb has negative pitch.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def rotation_matrix(q: Sequence[float]) -> np.ndarray:
    """
    Body-to-world rotation from a quaternion (x, y, z, w); normalised first.

    Raises ValueError for a zero quaternion (an IMU without an orientation estimate).
    """
    x, y, z, w = (float(c) for c in q)
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n < 1e-9:
        raise ValueError('zero quaternion: the IMU provides no orientation')
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def body_acceleration(quat: Sequence[float], specific_force: Sequence[float],
                      gravity: float = 9.81) -> np.ndarray:
    """Gravity-free acceleration in the body frame, m/s^2, from IMU orientation and f_b."""
    r = rotation_matrix(quat)
    return np.asarray(specific_force, dtype=np.float64) - gravity * r[2, :]


def roll_pitch(quat: Sequence[float]) -> tuple[float, float]:
    """(roll, pitch) in rad, ZYX convention; pitch positive nose down."""
    r = rotation_matrix(quat)
    pitch = math.asin(max(-1.0, min(1.0, -r[2, 0])))
    roll = math.atan2(r[2, 1], r[2, 2])
    return roll, pitch


def quat_from_rpy(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    """Quaternion (x, y, z, w) from ZYX Euler angles in rad (used by tests and tools)."""
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return (sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy)
