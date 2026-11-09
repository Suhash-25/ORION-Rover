"""
Flat-ground effort model and effort residual (docs/SPEC.md section 4, fix F5).

SPEC form: I_hat = c0 + c1 |w| + c2 dw/dt + c3 sin(pitch) sign(w).

Interpretation adopted (logged in docs/engineering-log/agents/wp3-terrain.md): every quantity
is taken ALONG THE DIRECTION OF ROTATION d = sign(w), so the model reads

    I_dir = d I = c0 + c1 |w| + c2 d dw/dt + c3 d sin(pitch)

which, multiplied by d, is the physical signed model I = d c0 + c1 w + c2 dw/dt + c3 sin(pitch)
(Coulomb + viscous + inertia + slope load). The residual e = d I - I_dir_hat is then positive
whenever the wheel needs MORE effort than on flat packed ground, in either driving direction.
The literal SPEC form with a signed I would report reversing out of sand as negative
sinkage, i.e. never during an escape manoeuvre.

The residual is only informative while the wheel turns (|w| >= deadband: sign(w) is defined)
and the rover is not turning hard (|yaw rate| <= max_yaw_rate: skid-steer scrub torque is not
in the model and dominates the effort, see the WP3 log). Outside that window the residual of
that wheel is HELD by the caller (`EffortModel.informative`); the fit uses the same window.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
import math
from typing import Sequence

import numpy as np
from orion_learning.obs_spec import WHEEL_ORDER
import yaml


class EffortSource(IntEnum):
    """Mirror of orion_msgs/TerrainState EFFORT_SOURCE_* (checked by a test)."""

    NONE = 0
    CURRENT = 1
    DUTY_ESTIMATE = 2
    SIM_TORQUE = 3


EFFORT_UNITS = {EffortSource.CURRENT: 'A', EffortSource.DUTY_ESTIMATE: 'A',
                EffortSource.SIM_TORQUE: 'N.m'}
N_COEF = 4
MODEL_FORM = ('d*I = c0 + c1*|w| + c2*d*dw/dt + c3*d*sin(pitch), d = sign(w); '
              'valid for |w| >= deadband_radps and |yaw rate| <= max_yaw_rate_radps')
# Below this std of sin(pitch) in the calibration data the slope term is not identifiable.
MIN_SLOPE_EXCITATION = 0.01


def direction(w: Sequence[float]) -> np.ndarray:
    """Direction of rotation d in {-1, +1} (+1 at exactly zero)."""
    return np.where(np.asarray(w, dtype=np.float64) < 0.0, -1.0, 1.0)


def design_matrix(w, dwdt, pitch, d) -> np.ndarray:
    """Regressors [1, |w|, d dw/dt, d sin(pitch)], shape (..., 4); w in rad/s, pitch in rad."""
    w, dwdt, d = (np.asarray(a, dtype=np.float64) for a in (w, dwdt, d))
    s = np.sin(np.asarray(pitch, dtype=np.float64)) * np.ones_like(w)
    return np.stack([np.ones_like(w), np.abs(w), d * dwdt, d * s], axis=-1)


def informative_mask(w, yaw_rate, deadband: float, max_yaw_rate: float) -> np.ndarray:
    """Mark where the flat-ground model applies: wheel turning, rover not turning hard."""
    w = np.asarray(w, dtype=np.float64)
    yaw_ok = np.abs(np.asarray(yaw_rate, dtype=np.float64)) <= max_yaw_rate
    return (np.abs(w) >= deadband) & (yaw_ok if w.ndim == 1 else yaw_ok[:, None])


@dataclass
class EffortModel:
    """Per-wheel coefficients (rows FL FR RL RR, columns c0..c3) plus provenance."""

    coefficients: np.ndarray
    i_ref: float                         # I_stall - I_free, in effort_unit
    effort_unit: str                     # 'A' (hardware) or 'N.m' (Gazebo)
    effort_source: EffortSource          # what the effort signal was during calibration
    calibrated: bool
    deadband_radps: float = 0.5
    max_yaw_rate_radps: float = 0.2
    slope_identified: bool = True        # False: c3 forced to 0 (no slope in the data)
    git_sha: str = 'unknown'
    date: str = 'unknown'
    source: str = 'unknown'
    r2: list = field(default_factory=lambda: [float('nan')] * 4)
    n_samples: list = field(default_factory=lambda: [0] * 4)

    def __post_init__(self):
        self.coefficients = np.asarray(self.coefficients, dtype=np.float64)
        if self.coefficients.shape != (len(WHEEL_ORDER), N_COEF):
            raise ValueError(f'coefficients must be 4x4, got {self.coefficients.shape}')
        self.effort_source = EffortSource(self.effort_source)
        if self.calibrated and not (self.i_ref > 0.0 and math.isfinite(self.i_ref)):
            raise ValueError(f'a calibrated model needs i_ref > 0, got {self.i_ref}')

    def residual(self, w, dwdt, pitch: float, effort) -> np.ndarray:
        """Effort residual per wheel, e = d I - I_dir_hat, in effort_unit (+ = extra load)."""
        d = direction(w)
        x = design_matrix(w, dwdt, pitch, d)
        predicted = np.einsum('ij,ij->i', x, self.coefficients)
        return d * np.asarray(effort, dtype=np.float64) - predicted

    def informative(self, w, yaw_rate: float) -> np.ndarray:
        """Per-wheel mask of samples whose residual means something (else: hold)."""
        return informative_mask(w, yaw_rate, self.deadband_radps, self.max_yaw_rate_radps)

    def sinkage_proxy(self, residual: Sequence[float]) -> float:
        """Return clip(mean(residual) / I_ref, 0, 1)."""
        return float(np.clip(np.mean(residual) / self.i_ref, 0.0, 1.0))

    # --- persistence ---------------------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            'calibrated': bool(self.calibrated),
            'model': MODEL_FORM,
            'coefficients': {name: [float(c) for c in row]
                             for name, row in zip(WHEEL_ORDER, self.coefficients)},
            'i_ref': float(self.i_ref),
            'effort_unit': self.effort_unit,
            'effort_source': self.effort_source.name,
            'deadband_radps': float(self.deadband_radps),
            'max_yaw_rate_radps': float(self.max_yaw_rate_radps),
            'slope_identified': bool(self.slope_identified),
            'r2': {name: float(r) for name, r in zip(WHEEL_ORDER, self.r2)},
            'n_samples': {name: int(n) for name, n in zip(WHEEL_ORDER, self.n_samples)},
            'git_sha': self.git_sha,
            'date': self.date,
            'source': self.source,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'EffortModel':
        try:
            return cls(
                coefficients=[d['coefficients'][name] for name in WHEEL_ORDER],
                i_ref=float(d['i_ref']),
                effort_unit=str(d['effort_unit']),
                effort_source=EffortSource[d['effort_source']],
                calibrated=bool(d['calibrated']),
                deadband_radps=float(d['deadband_radps']),
                max_yaw_rate_radps=float(d['max_yaw_rate_radps']),
                slope_identified=bool(d['slope_identified']),
                git_sha=str(d['git_sha']), date=str(d['date']), source=str(d['source']),
                r2=[float(d['r2'][n]) for n in WHEEL_ORDER],
                n_samples=[int(d['n_samples'][n]) for n in WHEEL_ORDER])
        except KeyError as e:
            raise ValueError(f'effort model is missing field {e}') from None

    def save(self, path: str, header: str = '') -> None:
        with open(path, 'w') as f:
            if header:
                f.write(''.join(f'# {line}\n' for line in header.splitlines()))
            yaml.safe_dump({'effort_model': self.to_dict()}, f, sort_keys=False)

    @classmethod
    def load(cls, path: str) -> 'EffortModel':
        with open(path) as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict) or 'effort_model' not in data:
            raise ValueError(f'{path}: no top-level effort_model key')
        return cls.from_dict(data['effort_model'])


@dataclass
class FitResult:
    coefficients: np.ndarray        # (4, 4)
    r2: np.ndarray                  # (4,), on the direction-aligned effort d I
    n_samples: np.ndarray           # (4,) samples used per wheel
    slope_identified: bool          # False: too little pitch variation, c3 forced to 0


def fit_coefficients(w, dwdt, pitch, effort, yaw_rate, deadband: float,
                     max_yaw_rate: float) -> FitResult:
    """
    Least-squares fit per wheel.

    Inputs: w, dw/dt, effort (N, 4) in rad/s, rad/s^2, effort unit; pitch and yaw_rate (N,) in
    rad and rad/s. Only informative samples (informative_mask) are used. When the data has no
    slope (std of sin(pitch) < MIN_SLOPE_EXCITATION) c3 is not identifiable and is set to 0.
    """
    w, dwdt, effort, pitch, yaw_rate = (np.asarray(a, dtype=np.float64)
                                        for a in (w, dwdt, effort, pitch, yaw_rate))
    mask = informative_mask(w, yaw_rate, deadband, max_yaw_rate)
    n_wheels = w.shape[1]
    coefs, r2 = np.zeros((n_wheels, N_COEF)), np.full(n_wheels, np.nan)
    used = mask.sum(axis=0)
    slope_ok = True
    for i in range(n_wheels):
        keep = mask[:, i]
        if used[i] < 4 * N_COEF:
            raise ValueError(f'wheel {WHEEL_ORDER[i]}: only {used[i]} samples with '
                             f'|w| >= {deadband} rad/s and |yaw rate| <= {max_yaw_rate} rad/s')
        d = direction(w[keep, i])
        x = design_matrix(w[keep, i], dwdt[keep, i], pitch[keep], d)
        y = d * effort[keep, i]
        cols = N_COEF
        if np.std(np.sin(pitch[keep])) < MIN_SLOPE_EXCITATION:
            slope_ok, cols = False, N_COEF - 1
        coefs[i, :cols], *_ = np.linalg.lstsq(x[:, :cols], y, rcond=None)
        ss_res = float(np.sum((y - x @ coefs[i]) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2[i] = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    return FitResult(coefs, r2, used, slope_ok)
