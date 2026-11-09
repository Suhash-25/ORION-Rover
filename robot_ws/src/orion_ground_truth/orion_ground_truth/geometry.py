"""Rigid-transform helpers and the ground-truth pose solver (no ROS imports).

Frames:
- camera: the overhead camera's optical frame, as reported by the AprilTag detector.
- ref tag: a tag glued flat at a known spot of the sandbox; it defines the `sandbox` frame
  (x along the sandbox's long edge, z up). Its pose in `sandbox` is a config value.
- rover tag: a tag fixed flat on top of the rover. `tag_to_base` is the measured transform
  from that tag to base_link.

Pose of base_link in sandbox =
    T(sandbox<-ref) * T(camera<-ref)^-1 * T(camera<-rover) * T(tag<-base).
Because the reference tag is in the same image, camera motion and camera mounting error cancel.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional, Sequence

import numpy as np


def quat_to_matrix(q: Sequence[float]) -> np.ndarray:
    """(x, y, z, w) unit quaternion to a 3x3 rotation matrix."""
    x, y, z, w = (float(v) for v in q)
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n < 1e-9:
        raise ValueError('zero quaternion')
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def matrix_to_quat(r: np.ndarray) -> tuple[float, float, float, float]:
    """3x3 rotation matrix to (x, y, z, w), w >= 0."""
    t = np.trace(r)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        w, x = 0.25 * s, (r[2, 1] - r[1, 2]) / s
        y, z = (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2
        w, x = (r[2, 1] - r[1, 2]) / s, 0.25 * s
        y, z = (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s
    elif r[1, 1] > r[2, 2]:
        s = math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2
        w, x = (r[0, 2] - r[2, 0]) / s, (r[0, 1] + r[1, 0]) / s
        y, z = 0.25 * s, (r[1, 2] + r[2, 1]) / s
    else:
        s = math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2
        w, x = (r[1, 0] - r[0, 1]) / s, (r[0, 2] + r[2, 0]) / s
        y, z = (r[1, 2] + r[2, 1]) / s, 0.25 * s
    q = np.array([x, y, z, w])
    q = q / np.linalg.norm(q)
    if q[3] < 0:
        q = -q
    return tuple(float(v) for v in q)


def make_tf(translation: Sequence[float], quat: Sequence[float]) -> np.ndarray:
    t = np.eye(4)
    t[:3, :3] = quat_to_matrix(quat)
    t[:3, 3] = np.asarray(translation, dtype=float)
    return t


def invert_tf(t: np.ndarray) -> np.ndarray:
    inv = np.eye(4)
    inv[:3, :3] = t[:3, :3].T
    inv[:3, 3] = -t[:3, :3].T @ t[:3, 3]
    return inv


def yaw_of(t: np.ndarray) -> float:
    return math.atan2(t[1, 0], t[0, 0])


@dataclass
class Pose2D:
    x: float
    y: float
    yaw: float
    stamp: float


@dataclass
class SolverConfig:
    sandbox_from_ref: np.ndarray          # 4x4, pose of the reference tag in the sandbox frame
    tag_to_base: np.ndarray               # 4x4, base_link pose in the rover-tag frame
    max_speed_mps: float = 1.0            # faster apparent motion between frames = outlier
    max_yaw_rate_rps: float = 4.0
    max_tilt_rad: float = 0.6             # rover tag tilted more than this vs sandbox = outlier


class GroundTruthSolver:
    """Turns per-image tag detections into a filtered 2-D rover pose in the sandbox frame."""

    def __init__(self, cfg: SolverConfig):
        self.cfg = cfg
        self.last: Optional[Pose2D] = None
        self.rejected = 0

    def solve(self, stamp: float, camera_from_ref: np.ndarray,
              camera_from_rover_tag: np.ndarray) -> Optional[Pose2D]:
        """Return the rover pose for one image, or None if the detection is implausible."""
        sandbox_from_base = (self.cfg.sandbox_from_ref @ invert_tf(camera_from_ref)
                             @ camera_from_rover_tag @ self.cfg.tag_to_base)
        z_axis = sandbox_from_base[:3, 2]
        tilt = math.acos(max(-1.0, min(1.0, float(z_axis[2]))))
        pose = Pose2D(float(sandbox_from_base[0, 3]), float(sandbox_from_base[1, 3]),
                      yaw_of(sandbox_from_base), stamp)
        if tilt > self.cfg.max_tilt_rad or not self._plausible(pose):
            self.rejected += 1
            return None
        self.last = pose
        return pose

    def _plausible(self, pose: Pose2D) -> bool:
        if self.last is None:
            return True
        dt = pose.stamp - self.last.stamp
        if dt <= 0:
            return False
        dist = math.hypot(pose.x - self.last.x, pose.y - self.last.y)
        dyaw = abs(math.atan2(math.sin(pose.yaw - self.last.yaw),
                              math.cos(pose.yaw - self.last.yaw)))
        if dt > 1.0:          # long gap: accept and restart the motion check
            return True
        return dist <= self.cfg.max_speed_mps * dt and dyaw <= self.cfg.max_yaw_rate_rps * dt
