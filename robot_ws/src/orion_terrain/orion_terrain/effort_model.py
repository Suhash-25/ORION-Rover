"""
Effort model of the rover runtime: provenance, effort source and YAML around the shared maths.

The model itself (form, residual, sinkage proxy, informative window, least-squares fit) is
orion_learning.estimator, the code the policy was trained with (council M5). Convention (WP3,
docs/engineering-log/agents/wp3-terrain.md): every quantity is taken ALONG THE DIRECTION OF
ROTATION d = sign(w),

    d I_hat = c0 + c1 |w| + c2 d dw/dt + c3 d sin(pitch),   residual e = d I - d I_hat,

so e > 0 whenever a wheel needs more effort than on flat packed ground, driving forward or
reversing out of sand. The residual is informative only for |w| >= 0.5 rad/s and |yaw rate|
<= 0.2 rad/s (skid-steer scrub is not modelled); outside that window the estimator HOLDS it.

This module adds what only the rover needs: per-wheel coefficients with their effort source
(current sensor, duty estimate, Gazebo torque), calibration provenance and the YAML file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
import math
from typing import Sequence

import numpy as np
from orion_learning import estimator as shared
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


@dataclass
class EffortModel:
    """Per-wheel coefficients (rows FL FR RL RR, columns c0..c3) plus provenance."""

    coefficients: np.ndarray
    i_ref: float                         # I_stall - I_free, in effort_unit
    effort_unit: str                     # 'A' (hardware) or 'N.m' (Gazebo)
    effort_source: EffortSource          # what the effort signal was during calibration
    calibrated: bool
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

    @property
    def shared(self) -> shared.EffortModel:
        """Return the orion_learning model with these coefficients (the code the env runs)."""
        return shared.EffortModel(self.coefficients, self.i_ref)

    def residual(self, effort, w, dwdt, pitch: float) -> np.ndarray:
        """Per-wheel residual e = d I - d I_hat (effort_unit, + = extra load); w rad/s."""
        return self.shared.residual(effort, w, dwdt, pitch)

    @staticmethod
    def informative(w, yaw_rate: float) -> np.ndarray:
        """Per-wheel mask of samples whose residual means something (else: hold)."""
        return shared.informative_mask(w, yaw_rate)

    def sinkage_proxy(self, residual: Sequence[float]) -> float:
        """Return clip(mean(residual) / I_ref, 0, 1)."""
        return float(self.shared.proxy(residual))

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
            # The window is fixed by orion_learning.estimator (part of the obs_spec hash);
            # written for provenance and checked on load.
            'deadband_radps': shared.INFORMATIVE_MIN_WHEEL_SPEED,
            'max_yaw_rate_radps': shared.INFORMATIVE_MAX_YAW_RATE,
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
            window = (float(d['deadband_radps']), float(d['max_yaw_rate_radps']))
            model = cls(
                coefficients=[d['coefficients'][name] for name in WHEEL_ORDER],
                i_ref=float(d['i_ref']),
                effort_unit=str(d['effort_unit']),
                effort_source=EffortSource[d['effort_source']],
                calibrated=bool(d['calibrated']),
                slope_identified=bool(d['slope_identified']),
                git_sha=str(d['git_sha']), date=str(d['date']), source=str(d['source']),
                r2=[float(d['r2'][n]) for n in WHEEL_ORDER],
                n_samples=[int(d['n_samples'][n]) for n in WHEEL_ORDER])
        except KeyError as e:
            raise ValueError(f'effort model is missing field {e}') from None
        expected = (shared.INFORMATIVE_MIN_WHEEL_SPEED, shared.INFORMATIVE_MAX_YAW_RATE)
        if window != expected:
            raise ValueError(f'effort model was fitted on the window (|w| >= {window[0]}, '
                             f'|yaw| <= {window[1]}) but orion_learning uses {expected}: refit')
        return model

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
