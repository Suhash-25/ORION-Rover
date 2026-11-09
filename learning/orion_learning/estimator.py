"""Terrain-estimator maths shared by the learning env and the rover runtime (SPEC section 4).

The policy is trained on the same degraded estimates the rover computes, so the env builds
its observations with these functions and orion_terrain should import them instead of
re-implementing them. Pure NumPy; every function broadcasts over a leading batch shape.

Contents: 1-D Kalman filter on body speed v_x, speed confidence, flat-ground effort model,
sinkage proxy and the first-order low-pass filter.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from orion_learning.obs_spec import WHEEL_RADIUS_M

KF_ACCEL_SIGMA = 0.1         # m/s^2, process noise: Q = KF_ACCEL_SIGMA^2 dt
FIX_SIGMA_EXTERNAL = 0.02    # m/s, measurement noise of an external (camera) speed fix
FIX_SIGMA_LIDAR = 0.05       # m/s, lidar-odometry speed fix
CONFIDENCE_TAU_S = 3.0       # s, confidence decay with time since the last fix
CONFIDENCE_DISAGREE_MPS = 0.5   # m/s of wheel-vs-estimate disagreement that zeroes confidence
LOWPASS_TAU_S = 0.25         # s, low-pass on slip and effort residual


def kf_predict(v, P, accel_x, dt):
    """Predict v_x (m/s) and its variance P with gravity-compensated body accel (m/s^2)."""
    return v + accel_x * dt, P + KF_ACCEL_SIGMA ** 2 * dt


def kf_update(v, P, fix_v, has_fix, sigma=FIX_SIGMA_EXTERNAL):
    """Fuse a speed fix (m/s) where has_fix is true; returns the new (v, P)."""
    gain = P / (P + sigma ** 2)
    v_new = np.where(has_fix, v + gain * (fix_v - v), v)
    P_new = np.where(has_fix, (1.0 - gain) * P, P)
    return v_new, P_new


def speed_confidence(t_since_fix, v_wheel, v_est):
    """Speed confidence c = exp(-t_since_fix / 3 s) (1 - min(1, |v_wheel - v_est| / 0.5)).

    In [0, 1].

    v_wheel is the wheel-odometry forward speed r mean(w) (m/s), v_est the filtered v_x.
    """
    agree = 1.0 - np.minimum(1.0, np.abs(np.asarray(v_wheel) - v_est) / CONFIDENCE_DISAGREE_MPS)
    return np.exp(-np.asarray(t_since_fix) / CONFIDENCE_TAU_S) * agree


def wheel_odometry_speed(wheel_speed):
    """Forward speed implied by the wheels, r * mean(w) (m/s); (..., 4) -> (...)."""
    return WHEEL_RADIUS_M * np.mean(np.asarray(wheel_speed, dtype=np.float64), axis=-1)


def lowpass(prev, x, dt, tau=LOWPASS_TAU_S):
    """One step of a first-order low-pass filter with time constant tau (s)."""
    alpha = dt / (tau + dt)
    return prev + alpha * (np.asarray(x) - prev)


@dataclass(frozen=True)
class EffortModel:
    """Flat-ground effort model |I_hat| = c0 + c1 |w| + c2 dw/dt sgn(w) + c3 sin(pitch) sgn(w).

    Calibrated by least squares on packed-light driving. Effort is in the unit of the
    /joint_states effort field (A on the rover, N.m in simulation); w in rad/s, dw/dt in
    rad/s^2, pitch in rad. ref = I_stall - I_free, the residual that maps to proxy 1.
    The sgn(w) on the inertia and slope terms makes the model predict effort magnitude.
    """

    c0: float
    c1: float
    c2: float
    c3: float
    ref: float

    def predict(self, w, dw_dt, pitch):
        """Return the expected |effort| per wheel on flat packed ground.

        w, dw_dt have shape (..., 4), pitch (...).
        """
        sg = np.sign(w)
        return (self.c0 + self.c1 * np.abs(w) + self.c2 * np.asarray(dw_dt) * sg
                + self.c3 * np.sin(np.asarray(pitch))[..., None] * sg)

    def residual(self, effort, w, dw_dt, pitch):
        """Mean over the 4 wheels of |effort| - I_hat (effort unit); (..., 4) -> (...)."""
        return np.mean(np.abs(effort) - self.predict(w, dw_dt, pitch), axis=-1)

    @staticmethod
    def features(w, dw_dt, pitch):
        """Regressor matrix rows [1, |w|, dw/dt sgn(w), sin(pitch) sgn(w)] for fitting."""
        w = np.asarray(w, dtype=np.float64)
        sg = np.sign(w)
        sp = np.broadcast_to(np.sin(np.asarray(pitch))[..., None], w.shape)
        return np.stack([np.ones_like(w), np.abs(w), np.asarray(dw_dt) * sg, sp * sg], -1)


def sinkage_proxy(residual_lowpassed, ref):
    """clip(mean residual / I_ref, 0, 1): the SPEC section 4 sinkage proxy."""
    return np.clip(np.asarray(residual_lowpassed) / ref, 0.0, 1.0)
