"""
Body-speed estimation: time handling of the shared 1-D Kalman filter, pose differentiation.

docs/SPEC.md section 4 (fix F4): wheel odometry is never a body-speed source. The filter is
predicted with the gravity-compensated IMU acceleration and corrected by velocity fixes from
the overhead camera (EXTERNAL) or lidar odometry (LIDAR_ODOM). Its maths and the speed
confidence are orion_learning.estimator's (the env's code).

Noise model and its known limit: Q = q^2 dt treats the acceleration error as white noise, so
while IMU-only the predicted std grows as q sqrt(t) (0.1 m/s^2 -> 0.2 m/s after 4 s). A
constant accelerometer bias b instead produces a deterministic error b t that the filter does
not model; with the env's bias scale (0.02 m/s^2) it stays below the 1-sigma band for
t < (q / b)^2 = 25 s, far beyond the ~1 s after which the confidence (zero at std 0.1 m/s)
stops trusting it, so the band is conservative for every dropout the confidence trusts.
"""

from __future__ import annotations

from collections import deque
from enum import IntEnum
import math
from typing import Optional, Sequence

import numpy as np
from orion_learning import estimator as shared

from orion_terrain.imu import rotation_matrix


class SpeedSource(IntEnum):
    """Mirror of orion_msgs/TerrainState SPEED_SOURCE_* (checked by a test)."""

    NONE = 0
    IMU_ONLY = 1
    LIDAR_ODOM = 2
    EXTERNAL = 3


class PoseDifferentiator:
    """
    Body-x velocity from timestamped world poses (overhead camera + AprilTag).

    A quadratic is least-squares fitted to the last ``window`` positions (on their real
    timestamps, so jitter is handled) and differentiated at the CENTRE sample: on uniform
    timestamps this is exactly the 5-point Savitzky-Golay first derivative. The centre value is
    ``window // 2`` samples old (0.1 s at 20 Hz); the Kalman filter moves it to the present
    with the IMU-integrated velocity change (see BodySpeedKalman.update). The centre estimate
    amplifies pose noise 3.5x less than an end-point derivative (6.3 vs 22.3 per sample period
    for 5 samples).

    The world velocity is projected on the body x axis of the centre sample's orientation.
    The buffer is cleared when consecutive poses are more than ``max_gap_s`` apart.
    """

    def __init__(self, window: int = 5, max_gap_s: float = 0.25):
        if window < 3 or window % 2 == 0:
            raise ValueError(f'window must be odd and >= 3, got {window}')
        self._window = window
        self._max_gap_s = max_gap_s
        self._buf: deque = deque(maxlen=window)

    def push(self, t: float, position: Sequence[float],
             quat: Sequence[float]) -> Optional[tuple[float, float]]:
        """Add a pose (t s, position m, quaternion x y z w); return (t_centre, v_x m/s)."""
        if self._buf and (t <= self._buf[-1][0] or t - self._buf[-1][0] > self._max_gap_s):
            self._buf.clear()
        self._buf.append((t, np.asarray(position, dtype=np.float64), tuple(quat)))
        if len(self._buf) < self._window:
            return None
        mid = self._window // 2
        t_c = self._buf[mid][0]
        ts = np.array([s[0] for s in self._buf]) - t_c
        pos = np.stack([s[1] for s in self._buf])
        design = np.vander(ts, 3, increasing=True)          # [1, t, t^2]
        coef, *_ = np.linalg.lstsq(design, pos, rcond=None)
        v_world = coef[1]                                     # d/dt at t = t_c
        x_axis = rotation_matrix(self._buf[mid][2])[:, 0]
        return t_c, float(x_axis @ v_world)


class BodySpeedKalman:
    """
    Scalar Kalman filter on the longitudinal body speed v_x (m/s).

    predict / update are orion_learning.estimator.kf_predict / kf_update (the env's filter):
    v += a_x dt, P += q^2 dt (a_x gravity-compensated, m/s^2); standard scalar update with
    measurement std r (m/s). This class adds the time handling: irregular sample times, and a
    fix that refers to an earlier time t_meas is moved to the present by adding the integrated
    acceleration since t_meas (forward propagation of a delayed measurement); fixes older than
    the 2 s history use its oldest entry.
    """

    HISTORY_S = 2.0

    def __init__(self, q_accel: float = shared.KF_ACCEL_SIGMA, v0: float = 0.0,
                 std0: float = 0.1):
        self.q = q_accel
        self.v = v0
        self.var = std0 ** 2
        self.t: Optional[float] = None
        self._dv = 0.0                       # running integral of a dt, m/s
        self._hist: deque = deque()          # (t, _dv)

    @property
    def std(self) -> float:
        return math.sqrt(self.var)

    def predict_to(self, t: float, accel: float) -> None:
        """Propagate to time t (s) holding acceleration ``accel`` (m/s^2) since self.t."""
        if self.t is None:
            self.t = t
            self._hist.append((t, self._dv))
            return
        dt = t - self.t
        if dt <= 0.0:
            return
        self.v, self.var = shared.kf_predict(self.v, self.var, accel, dt, self.q)
        self._dv += accel * dt
        self.t = t
        self._hist.append((t, self._dv))
        while len(self._hist) > 2 and self._hist[1][0] < t - self.HISTORY_S:
            self._hist.popleft()

    def _dv_at(self, t: float) -> float:
        ts = [h[0] for h in self._hist]
        return float(np.interp(t, ts, [h[1] for h in self._hist])) if ts else self._dv

    def update(self, z: float, r_std: float, t_meas: Optional[float] = None) -> None:
        """Fuse a velocity fix z (m/s) with std r_std (m/s), measured at t_meas (s)."""
        if t_meas is not None and self.t is not None and t_meas < self.t:
            z += self._dv - self._dv_at(t_meas)
        v, var = shared.kf_update(self.v, self.var, z, True, r_std)
        self.v, self.var = float(v), float(var)
