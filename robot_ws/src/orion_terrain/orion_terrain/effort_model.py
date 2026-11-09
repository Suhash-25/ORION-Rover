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

Below the speed deadband sign(w) is noise; there d = sign(I) (a stalled wheel pushing into
sand still reads as positive extra effort). The fit ignores samples inside the deadband.
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
MODEL_FORM = ('d*I = c0 + c1*|w| + c2*d*dw/dt + c3*d*sin(pitch), '
              'd = sign(w) (sign(I) if |w| < deadband)')


def rotation_direction(w: Sequence[float], effort: Sequence[float],
                       deadband: float) -> np.ndarray:
    """Per-wheel direction d in {-1, +1}: sign(w) outside the deadband, else sign(effort)."""
    w = np.asarray(w, dtype=np.float64)
    effort = np.asarray(effort, dtype=np.float64)
    d = np.where(np.abs(w) >= deadband, np.sign(w), np.sign(effort))
    return np.where(d == 0.0, 1.0, d)


def design_matrix(w, dwdt, pitch, direction) -> np.ndarray:
    """Regressors [1, |w|, d dw/dt, d sin(pitch)], shape (..., 4); w in rad/s, pitch in rad."""
    w, dwdt, d = (np.asarray(a, dtype=np.float64) for a in (w, dwdt, direction))
    s = np.sin(np.asarray(pitch, dtype=np.float64)) * np.ones_like(w)
    return np.stack([np.ones_like(w), np.abs(w), d * dwdt, d * s], axis=-1)


@dataclass
class EffortModel:
    """Per-wheel coefficients (rows FL FR RL RR, columns c0..c3) plus provenance."""

    coefficients: np.ndarray
    i_ref: float                         # I_stall - I_free, in effort_unit
    effort_unit: str                     # 'A' (hardware) or 'N.m' (Gazebo)
    effort_source: EffortSource          # what the effort signal was during calibration
    calibrated: bool
    deadband_radps: float = 0.5
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
        d = rotation_direction(w, effort, self.deadband_radps)
        x = design_matrix(w, dwdt, pitch, d)
        predicted = np.einsum('ij,ij->i', x, self.coefficients)
        return d * np.asarray(effort, dtype=np.float64) - predicted

    def sinkage_proxy(self, residual: Sequence[float]) -> float:
        """clip(mean(residual) / I_ref, 0, 1)."""
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


def fit_coefficients(w: np.ndarray, dwdt: np.ndarray, pitch: np.ndarray, effort: np.ndarray,
                     deadband: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Least-squares fit per wheel.

    Inputs are (N, 4) arrays (w rad/s, dw/dt rad/s^2, effort) and pitch (N,) rad. Samples with
    |w| < deadband are dropped. Returns (coefficients (4, 4), R^2 (4,), samples used (4,)).
    R^2 is computed on the direction-aligned effort d I that the model predicts.
    """
    w, dwdt, effort = (np.asarray(a, dtype=np.float64) for a in (w, dwdt, effort))
    pitch = np.asarray(pitch, dtype=np.float64)
    coefs = np.zeros((w.shape[1], N_COEF))
    r2 = np.full(w.shape[1], np.nan)
    used = np.zeros(w.shape[1], dtype=int)
    for i in range(w.shape[1]):
        keep = np.abs(w[:, i]) >= deadband
        used[i] = int(keep.sum())
        if used[i] < 4 * N_COEF:
            raise ValueError(f'wheel {WHEEL_ORDER[i]}: only {used[i]} samples above the '
                             f'{deadband} rad/s deadband; drive more')
        d = np.sign(w[keep, i])
        x = design_matrix(w[keep, i], dwdt[keep, i], pitch[keep], d)
        y = d * effort[keep, i]
        coefs[i], *_ = np.linalg.lstsq(x, y, rcond=None)
        ss_res = float(np.sum((y - x @ coefs[i]) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2[i] = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    return coefs, r2, used
