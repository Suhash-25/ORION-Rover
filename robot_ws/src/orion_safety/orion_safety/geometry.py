"""
Small 2-D geometry and attitude helpers for the safety rules (pure Python, no ROS).

Units: metres, radians. Polygons are lists of (x, y) vertices in order (either winding),
implicitly closed.
"""

import math


def quaternion_to_rpy(x, y, z, w):
    """Return (roll, pitch, yaw) in rad from a unit quaternion (ZYX / aerospace convention)."""
    sinr = 2.0 * (w * x + y * z)
    cosr = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr, cosr)
    sinp = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    pitch = math.asin(sinp)
    siny = 2.0 * (w * z + x * y)
    cosy = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny, cosy)
    return roll, pitch, yaw


def accel_to_roll_pitch(ax, ay, az):
    """
    Return (roll, pitch) in rad from the gravity direction of a resting accelerometer.

    Specific force in m/s^2, body frame x forward, y left, z up (REP 103). Only valid while
    the body acceleration is small compared with g: a fallback for IMUs without orientation.
    """
    roll = math.atan2(ay, az)
    pitch = math.atan2(-ax, math.hypot(ay, az))
    return roll, pitch


def point_in_polygon(px, py, polygon):
    """Return True if (px, py) is strictly inside the polygon (even-odd rule)."""
    inside = False
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        if (y1 > py) != (y2 > py):
            x_cross = x1 + (py - y1) * (x2 - x1) / (y2 - y1)
            if px < x_cross:
                inside = not inside
    return inside


def _segment_distance(px, py, x1, y1, x2, y2):
    dx, dy = x2 - x1, y2 - y1
    length_sq = dx * dx + dy * dy
    t = 0.0 if length_sq == 0.0 else ((px - x1) * dx + (py - y1) * dy) / length_sq
    t = max(0.0, min(1.0, t))
    return math.hypot(px - (x1 + t * dx), py - (y1 + t * dy))


def signed_distance_inside(px, py, polygon):
    """
    Distance in m from (px, py) to the polygon boundary: positive inside, negative outside.

    A point is within the polygon eroded by margin m exactly when this is >= m.
    """
    n = len(polygon)
    dist = min(
        _segment_distance(px, py, *polygon[i], *polygon[(i + 1) % n]) for i in range(n))
    return dist if point_in_polygon(px, py, polygon) else -dist


def predict_pose(x, y, yaw, v, w, horizon):
    """
    Pose (x, y, yaw) after driving at constant v (m/s) and w (rad/s) for horizon (s).

    Exact unicycle arc; straight line when |w| is negligible.
    """
    if abs(w) < 1e-6:
        return x + v * horizon * math.cos(yaw), y + v * horizon * math.sin(yaw), yaw
    yaw_end = yaw + w * horizon
    radius = v / w
    return (x + radius * (math.sin(yaw_end) - math.sin(yaw)),
            y - radius * (math.cos(yaw_end) - math.cos(yaw)),
            yaw_end)
