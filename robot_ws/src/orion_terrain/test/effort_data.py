"""Synthetic flat-ground driving with a known effort model (tests only)."""

import numpy as np

from orion_terrain.effort_model import EffortModel, EffortSource

# Rows FL FR RL RR, columns c0 (Coulomb), c1 (viscous), c2 (inertia), c3 (slope).
TRUE_COEFS = np.array([
    [0.40, 0.080, 0.050, -2.0],
    [0.35, 0.090, 0.045, -1.8],
    [0.45, 0.070, 0.055, -2.2],
    [0.38, 0.085, 0.050, -1.9],
])


def signed_effort(coefs, w, dwdt, pitch):
    """Physical signed effort I = d c0 + c1 w + c2 dw/dt + c3 sin(pitch) for each wheel."""
    d = np.sign(w)
    return (d * coefs[:, 0] + coefs[:, 1] * w + coefs[:, 2] * dwdt
            + coefs[:, 3] * np.sin(np.asarray(pitch, dtype=float))[..., None])


def drive(duration_s=120.0, hz=100.0, seed=0):
    """Smooth forward/reverse driving with turns and slope. Returns t, w, dwdt, pitch (arrays)."""
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, duration_s, 1.0 / hz)
    phases = rng.uniform(0, 2 * np.pi, 4)
    base = 4.0 * np.sin(2 * np.pi * t / 9.0)[:, None]                # +-0.48 m/s fwd / rev
    turn = 1.2 * np.sin(2 * np.pi * t / 5.0 + phases[0])[:, None] * np.array([-1, 1, -1, 1])
    wob = 0.6 * np.sin(2 * np.pi * t[:, None] / 2.3 + phases)
    w = base + turn + wob
    dwdt = np.gradient(w, t, axis=0)
    pitch = 0.15 * np.sin(2 * np.pi * t / 23.0)
    return t, w, dwdt, pitch


def make_model(coefs=TRUE_COEFS, source=EffortSource.CURRENT, calibrated=True, i_ref=5.0):
    return EffortModel(coefficients=coefs, i_ref=i_ref,
                       effort_unit='N.m' if source == EffortSource.SIM_TORQUE else 'A',
                       effort_source=source, calibrated=calibrated)
