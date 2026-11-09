"""Terrain-estimator maths shared by the learning env and the rover runtime (SPEC section 4).

The ONE implementation of: the 1-D Kalman filter on body speed v_x, the speed confidence, the
first-order low-pass, the flat-ground effort model (predict, residual, fit, sinkage proxy) and
the informative-window mask. The env (orion_learning.env) builds its observations with these
functions and orion_terrain imports them; orion_terrain only adds message plumbing, time
handling, the Savitzky-Golay pose differentiator and YAML provenance. A change here changes
both sides together, and the constants that define what an observation means are part of
obs_spec's hash (obs_spec.SEMANTICS), so a policy trained before the change is refused.

Pure NumPy; every function broadcasts over a leading batch shape. Units: m/s, rad/s, s, and
the effort unit of /joint_states (A on the rover, N.m in simulation).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

KF_ACCEL_SIGMA = 0.1         # m/s^2, process noise: Q = KF_ACCEL_SIGMA^2 dt
FIX_SIGMA_EXTERNAL = 0.02    # m/s, measurement noise of an external (camera) speed fix
FIX_SIGMA_LIDAR = 0.05       # m/s, lidar-odometry speed fix
CONFIDENCE_TAU_S = 3.0       # s, confidence decay with time since the last fix
CONFIDENCE_STD_MPS = 0.1     # m/s, Kalman 1-sigma at which the confidence reaches 0
LOWPASS_TAU_S = 0.25         # s, low-pass on slip and effort residual
INFORMATIVE_MIN_WHEEL_SPEED = 0.5   # rad/s, |w| below this: effort residual held
INFORMATIVE_MAX_YAW_RATE = 0.2      # rad/s, |yaw rate| above this: effort residual held
MIN_SLOPE_EXCITATION = 0.01  # std of sin(pitch) below which the slope term is unidentifiable

# Formula ids hashed into obs_spec.SEMANTICS: change the id whenever the formula changes.
CONFIDENCE_FORMULA = 'c=exp(-t_since_fix/3s)*clip(1-kf_std/0.1mps,0,1)'
LOWPASS_FORMULA = 'y+=(1-exp(-dt/tau))*(x-y),start=first sample,masked entries held'
EFFORT_FORMULA = ('d=sgn(w)(+1 at 0); d*I=c0+c1|w|+c2*d*dw/dt+c3*d*sin(pitch) per wheel; '
                  'residual e=d*I-I_hat per wheel, low-passed only where |w|>=0.5 and '
                  '|yaw|<=0.2 (held elsewhere); proxy=clip(mean(e)/I_ref,0,1)')


# --- body speed ------------------------------------------------------------------------------
def kf_predict(v, P, accel_x, dt, q=KF_ACCEL_SIGMA):
    """Predict v_x (m/s) and its variance P (m^2/s^2) over dt (s) with body accel (m/s^2).

    accel_x must be gravity-compensated. Q = q^2 dt (white acceleration noise).
    """
    return v + accel_x * dt, P + q ** 2 * dt


def kf_update(v, P, fix_v, has_fix=True, sigma=FIX_SIGMA_EXTERNAL):
    """Fuse a speed fix fix_v (m/s, std sigma) where has_fix is true; returns the new (v, P)."""
    gain = P / (P + sigma ** 2)
    v_new = np.where(has_fix, v + gain * (fix_v - v), v)
    P_new = np.where(has_fix, (1.0 - gain) * P, P)
    return v_new, P_new


def speed_confidence(t_since_fix, kf_std):
    """Confidence in the body speed, in [0, 1].

    c = exp(-t_since_fix / 3 s) * clip(1 - kf_std / 0.1 m/s, 0, 1), t_since_fix in s (inf
    before the first fix gives 0), kf_std the Kalman 1-sigma in m/s. Wheel odometry plays no
    part: under slip it disagrees with the body speed exactly when the estimate is needed.
    """
    t = np.maximum(np.asarray(t_since_fix, dtype=np.float64), 0.0)
    certain = np.clip(1.0 - np.asarray(kf_std, dtype=np.float64) / CONFIDENCE_STD_MPS, 0.0, 1.0)
    return np.exp(-t / CONFIDENCE_TAU_S) * certain


# --- low-pass --------------------------------------------------------------------------------
def lowpass_alpha(dt, tau=LOWPASS_TAU_S):
    """Gain 1 - exp(-dt / tau) of one low-pass step over dt (s); 0 for dt <= 0, 1 for tau <= 0.

    The exact discretisation of a first-order lag, so irregular sample times (the runtime) and
    the env's fixed 0.05 s step filter alike.
    """
    dt = np.maximum(np.asarray(dt, dtype=np.float64), 0.0)
    return 1.0 - np.exp(-dt / tau) if tau > 0 else np.where(dt > 0, 1.0, 0.0)


def lowpass(prev, x, dt, tau=LOWPASS_TAU_S, mask=None):
    """One low-pass step from prev towards x over dt (s); entries where mask is False are held."""
    prev = np.asarray(prev, dtype=np.float64)
    out = prev + lowpass_alpha(dt, tau) * (np.asarray(x, dtype=np.float64) - prev)
    return out if mask is None else np.where(mask, out, prev)


# --- effort model ----------------------------------------------------------------------------
def wheel_direction(w):
    """Direction of rotation d in {-1, +1} (+1 at exactly zero), same shape as w (rad/s)."""
    return np.where(np.asarray(w, dtype=np.float64) < 0.0, -1.0, 1.0)


def effort_features(w, dw_dt, pitch):
    """Regressors [1, |w|, d dw/dt, d sin(pitch)], shape (..., wheels, 4).

    w, dw_dt (..., wheels) in rad/s and rad/s^2; pitch (...) in rad.
    """
    w = np.asarray(w, dtype=np.float64)
    d = wheel_direction(w)
    s = np.broadcast_to(np.sin(np.asarray(pitch, dtype=np.float64))[..., None], w.shape)
    return np.stack([np.ones_like(w), np.abs(w), d * np.asarray(dw_dt, dtype=np.float64), d * s],
                    axis=-1)


def informative_mask(w, yaw_rate):
    """Where the flat-ground effort model applies: wheel turning, rover not turning hard.

    |w| >= 0.5 rad/s (direction defined, past stiction) and |yaw rate| <= 0.2 rad/s
    (skid-steer scrub torque is not in the model and dominates the effort). w (..., wheels),
    yaw_rate (...); returns a bool (..., wheels) mask.
    """
    w = np.asarray(w, dtype=np.float64)
    yaw_ok = np.abs(np.asarray(yaw_rate, dtype=np.float64)) <= INFORMATIVE_MAX_YAW_RATE
    return (np.abs(w) >= INFORMATIVE_MIN_WHEEL_SPEED) & yaw_ok[..., None]


@dataclass(frozen=True, eq=False)
class EffortModel:
    """Flat-ground effort model, every quantity taken along the wheel's direction of rotation.

    d I_hat = c0 + c1 |w| + c2 d dw/dt + c3 d sin(pitch), d = sgn(w): Coulomb, viscous,
    inertia and slope load, so the residual e = d I - d I_hat is positive whenever a wheel
    needs MORE effort than on flat packed ground, driving forward or reversing out of sand.
    coefficients: (wheels, 4) rows FL FR RL RR, columns c0..c3 (a (4,) row is used for every
    wheel). ref = I_stall - I_free (effort unit), the residual that maps to sinkage proxy 1.
    """

    coefficients: np.ndarray
    ref: float

    def __post_init__(self):
        c = np.array(self.coefficients, dtype=np.float64)
        if c.shape[-1] != 4 or c.ndim not in (1, 2):
            raise ValueError(f'coefficients must be (4,) or (wheels, 4), got {c.shape}')
        c.setflags(write=False)
        object.__setattr__(self, 'coefficients', c)

    def predict(self, w, dw_dt, pitch):
        """Expected effort along the direction of rotation, per wheel (..., wheels)."""
        return np.sum(effort_features(w, dw_dt, pitch) * self.coefficients, axis=-1)

    def residual(self, effort, w, dw_dt, pitch):
        """Per-wheel residual e = d I - d I_hat (effort unit, + = extra load), (..., wheels)."""
        return wheel_direction(w) * np.asarray(effort, dtype=np.float64) - self.predict(
            w, dw_dt, pitch)

    def proxy(self, residual):
        """Sinkage proxy clip(mean over wheels of the (low-passed) residual / ref, 0, 1)."""
        return sinkage_proxy(np.mean(np.asarray(residual, dtype=np.float64), axis=-1), self.ref)


def sinkage_proxy(mean_residual, ref):
    """clip(mean residual / I_ref, 0, 1): the SPEC section 4 sinkage proxy."""
    return np.clip(np.asarray(mean_residual) / ref, 0.0, 1.0)


@dataclass(frozen=True, eq=False)
class EffortFit:
    coefficients: np.ndarray     # (wheels, 4)
    r2: np.ndarray               # (wheels,), on the direction-aligned effort d I
    n_samples: np.ndarray        # (wheels,) informative samples used
    slope_identified: bool       # False: c3 forced to 0


def fit_effort_model(w, dw_dt, pitch, effort, yaw_rate, fit_slope: bool | None = None,
                     min_samples: int = 16) -> EffortFit:
    """Least-squares fit of the effort model per wheel on informative samples only.

    w, dw_dt, effort (N, wheels); pitch, yaw_rate (N,). fit_slope None: fit c3 only when the
    data has slope (std of sin(pitch) >= MIN_SLOPE_EXCITATION on every wheel's samples),
    False: c3 = 0. Raises ValueError when a wheel has fewer than min_samples informative rows.
    """
    w, dw_dt, effort, pitch = (np.asarray(a, dtype=np.float64) for a in (w, dw_dt, effort, pitch))
    mask = informative_mask(w, yaw_rate)
    n_wheels = w.shape[1]
    used = mask.sum(axis=0)
    coefs, r2 = np.zeros((n_wheels, 4)), np.full(n_wheels, np.nan)
    x_all = effort_features(w, dw_dt, pitch)
    y_all = wheel_direction(w) * effort
    slope = fit_slope is not False
    for i in range(n_wheels):
        if used[i] < min_samples:
            raise ValueError(f'wheel {i}: only {used[i]} samples with |w| >= '
                             f'{INFORMATIVE_MIN_WHEEL_SPEED} rad/s and |yaw rate| <= '
                             f'{INFORMATIVE_MAX_YAW_RATE} rad/s')
        if fit_slope is None and np.std(np.sin(pitch[mask[:, i]])) < MIN_SLOPE_EXCITATION:
            slope = False
    cols = 4 if slope else 3
    for i in range(n_wheels):
        keep = mask[:, i]
        x, y = x_all[keep, i], y_all[keep, i]
        coefs[i, :cols] = np.linalg.lstsq(x[:, :cols], y, rcond=None)[0]
        ss_res = float(np.sum((y - x @ coefs[i]) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2[i] = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    return EffortFit(coefs, r2, used, slope)
