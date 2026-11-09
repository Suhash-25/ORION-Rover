"""Wheel-soil terramechanics for the ORION learning environment (docs/SPEC.md section 6).

Everything is NumPy and broadcasts over any array shape, so the same functions serve one
wheel, four wheels of one rover, or (N, 4) wheels of N rovers. All quantities are SI:
forces in N, lengths in m, k_c in N/m^(n+1), k_phi in N/m^(n+2), cohesion c in Pa,
friction angle phi in rad, shear modulus K in m.

The model:

* static sinkage (Bekker, rigid wheel)
  z0 = [3 W / ((3 - n) b (k_c / b + k_phi_eff) sqrt(D))]^(2 / (2 n + 1))
* gravity coupling (the disputed effect) k_phi_eff = k_phi (g / 9.81)^beta
* slip sinkage z_target = z0 (1 + a_s |s| / (1 - |s| + 0.05)); the environment adds the
  "sticky" relaxation z <- max(z_target, z - 0.3 |v| dt)
* contact length l = r acos(1 - z / r), contact area A = b l
* Janosi-Hanamoto thrust
  H = (A c + W tan phi)(1 - K / (|s| l) (1 - exp(-|s| l / K))) sign(s),
  replaced by its small-slip limit (A c + W tan phi) |s| l / (2 K) sign(s) for |s| < 1e-3
* compaction resistance Rc = b (k_c / b + k_phi_eff) z^(n + 1) / (n + 1)
* model-form error: net force F = m_fH H - m_fR Rc

``reference`` holds a scalar pure-Python transcription of the same formulas. It is only
used by the tests to check the vectorised code against an independent implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import math

import numpy as np

from orion_learning.obs_spec import WHEEL_RADIUS_M

G_EARTH = 9.81                 # m/s^2, gravity at which k_phi was measured
WHEEL_WIDTH_M = 0.06           # b
WHEEL_DIAMETER_M = 2.0 * WHEEL_RADIUS_M   # D
SMALL_SLIP = 1e-3              # below this |s| the Janosi-Hanamoto small-slip limit is used
SLIP_SINKAGE_EPS = 0.05        # keeps the slip-sinkage factor finite at |s| = 1
DP_SLIP = 0.3                  # slip at which the drawbar coefficient descriptor is taken
DP_TARGET = 0.1                # DP / W for the "slip at DP = 0.1 W" descriptor
DESCRIPTOR_SLIP_GRID = np.linspace(0.0, 1.0, 201)

# Order of the columns when soil parameters are stored as one stacked array (env state).
SOIL_KEYS = ('k_c', 'k_phi', 'n', 'c', 'phi', 'K', 'a_s')


@dataclass(frozen=True)
class Soil:
    """Soil parameters, each a scalar or an array (all broadcastable), SI units.

    k_c   cohesive modulus, N/m^(n+1)
    k_phi frictional modulus measured at g = 9.81, N/m^(n+2)
    n     sinkage exponent, -
    c     cohesion, Pa
    phi   internal friction angle, rad
    K     shear deformation modulus, m
    a_s   slip-sinkage coefficient, - (GUESS model)
    """

    k_c: np.ndarray
    k_phi: np.ndarray
    n: np.ndarray
    c: np.ndarray
    phi: np.ndarray
    K: np.ndarray
    a_s: np.ndarray

    @classmethod
    def from_stacked(cls, arr: np.ndarray) -> 'Soil':
        """Build from an array whose last axis is ordered as SOIL_KEYS."""
        arr = np.asarray(arr, dtype=np.float64)
        return cls(*(arr[..., i] for i in range(len(SOIL_KEYS))))

    def stacked(self) -> np.ndarray:
        """Inverse of from_stacked: last axis ordered as SOIL_KEYS."""
        parts = np.broadcast_arrays(*(np.asarray(getattr(self, k), dtype=np.float64)
                                      for k in SOIL_KEYS))
        return np.stack(parts, axis=-1)

    def expand(self, axis: int = -1) -> 'Soil':
        """Insert a new axis into every field (e.g. to broadcast against a slip grid)."""
        return Soil(*(np.expand_dims(np.asarray(getattr(self, f.name)), axis)
                      for f in fields(self)))


def k_phi_eff(k_phi, g, beta):
    """Gravity-coupled frictional modulus k_phi (g / 9.81)^beta, N/m^(n+2)."""
    return np.asarray(k_phi) * (np.asarray(g) / G_EARTH) ** np.asarray(beta)


def bekker_modulus(soil: Soil, g, beta):
    """k_c / b + k_phi_eff, N/m^(n+2)."""
    return soil.k_c / WHEEL_WIDTH_M + k_phi_eff(soil.k_phi, g, beta)


def static_sinkage(W, soil: Soil, g, beta):
    """Rigid-wheel static sinkage z0 (m) under wheel load W (N)."""
    W = np.maximum(np.asarray(W, dtype=np.float64), 0.0)
    n = soil.n
    base = 3.0 * W / ((3.0 - n) * WHEEL_WIDTH_M * bekker_modulus(soil, g, beta)
                      * math.sqrt(WHEEL_DIAMETER_M))
    return base ** (2.0 / (2.0 * n + 1.0))


def slip_sinkage_target(z0, s, a_s):
    """Sinkage (m) the wheel digs to at slip s: z0 (1 + a_s |s| / (1 - |s| + 0.05))."""
    a = np.minimum(np.abs(s), 1.0)
    return z0 * (1.0 + a_s * a / (1.0 - a + SLIP_SINKAGE_EPS))


def contact_length(z):
    """Wheel-soil contact length l = r acos(1 - z / r), m. z is clipped to [0, r]."""
    zr = np.clip(np.asarray(z, dtype=np.float64) / WHEEL_RADIUS_M, 0.0, 1.0)
    return WHEEL_RADIUS_M * np.arccos(1.0 - zr)


def max_thrust(z, W, soil: Soil):
    """Fully mobilised shear strength A c + W tan(phi), N."""
    return WHEEL_WIDTH_M * contact_length(z) * soil.c + np.asarray(W) * np.tan(soil.phi)


def thrust(s, z, W, soil: Soil):
    """Janosi-Hanamoto gross thrust H (N), signed like the slip s (positive = driving).

    Uses the small-slip limit (A c + W tan phi) |s| l / (2 K) for |s| < SMALL_SLIP, which is
    the first-order Taylor expansion and avoids the 0/0 in the exact form.
    """
    s = np.asarray(s, dtype=np.float64)
    a = np.minimum(np.abs(s), 1.0)
    length = contact_length(z)
    x = a * length / soil.K
    small = a < SMALL_SLIP
    x_safe = np.where(small, 1.0, x)          # avoid the division warning in the masked branch
    shape = np.where(small, 0.5 * x, 1.0 - (1.0 - np.exp(-x_safe)) / x_safe)
    return max_thrust(z, W, soil) * shape * np.sign(s)


def compaction_resistance(z, soil: Soil, g, beta):
    """Compaction resistance Rc = b (k_c / b + k_phi_eff) z^(n+1) / (n+1), N (magnitude)."""
    z = np.maximum(np.asarray(z, dtype=np.float64), 0.0)
    n = soil.n
    return WHEEL_WIDTH_M * bekker_modulus(soil, g, beta) * z ** (n + 1.0) / (n + 1.0)


def steady_drawbar_pull(s, W, soil: Soil, g, beta, m_fH=1.0, m_fR=1.0):
    """Drawbar pull (N) of a wheel driven at constant slip s >= 0 until sinkage settles.

    Steady state of the sticky sinkage model at constant slip is z = z_target(s), so
    DP(s) = m_fH H(s, z_target) - m_fR Rc(z_target).
    """
    z = slip_sinkage_target(static_sinkage(W, soil, g, beta), s, soil.a_s)
    return m_fH * thrust(s, z, W, soil) - m_fR * compaction_resistance(z, soil, g, beta)


def steady_drawbar_curve(W, soil: Soil, g, beta, m_fH=1.0, m_fR=1.0) -> np.ndarray:
    """Steady drawbar coefficient DP/W on DESCRIPTOR_SLIP_GRID, shape (..., 201).

    W, g, beta, m_fH, m_fR and the soil fields broadcast against each other; the slip grid
    is appended as the last axis.
    """
    ex = np.expand_dims
    W = np.asarray(W, dtype=np.float64)
    return steady_drawbar_pull(DESCRIPTOR_SLIP_GRID, ex(W, -1), soil.expand(),
                               ex(np.asarray(g), -1), ex(np.asarray(beta), -1),
                               ex(np.asarray(m_fH), -1), ex(np.asarray(m_fR), -1)) / ex(W, -1)


def peak_drawbar_coefficient(W, soil: Soil, g, beta, m_fH=1.0, m_fR=1.0) -> np.ndarray:
    """Peak over slip in [0, 1] of the steady drawbar coefficient DP/W (dimensionless).

    Negative when the wheel cannot pull itself at any steady slip (it digs in instead).
    """
    return steady_drawbar_curve(W, soil, g, beta, m_fH, m_fR).max(axis=-1)


def traction_descriptors(W, soil: Soil, g, beta, m_fH=1.0, m_fR=1.0) -> np.ndarray:
    """Functional traction descriptors of a wheel, shape (..., 4).

    Columns: [DP/W at s = 0.3, slip at DP = 0.1 W, z0 / r, Rc(z0) / W], where Rc includes
    the m_fR multiplier and DP both multipliers, because the teacher should see the
    traction the wheel actually has, not the nominal Bekker parameters. The slip at
    DP = 0.1 W is the smallest slip on a 201-point grid over [0, 1] where the steady
    drawbar coefficient reaches 0.1 (linear interpolation between grid points), or 1.0 if it
    never does.
    """
    W = np.asarray(W, dtype=np.float64)
    z0 = static_sinkage(W, soil, g, beta)
    dp30 = steady_drawbar_pull(DP_SLIP, W, soil, g, beta, m_fH, m_fR) / W
    rc = np.asarray(m_fR) * compaction_resistance(z0, soil, g, beta) / W

    grid = DESCRIPTOR_SLIP_GRID
    coeff = steady_drawbar_curve(W, soil, g, beta, m_fH, m_fR)
    reached = coeff >= DP_TARGET
    any_reached = reached.any(axis=-1)
    idx = np.argmax(reached, axis=-1)                     # first grid index reaching target
    idx_prev = np.maximum(idx - 1, 0)
    c_hi = np.take_along_axis(coeff, idx[..., None], -1)[..., 0]
    c_lo = np.take_along_axis(coeff, idx_prev[..., None], -1)[..., 0]
    s_hi, s_lo = grid[idx], grid[idx_prev]
    denom = np.where(c_hi > c_lo, c_hi - c_lo, 1.0)
    frac = np.clip((DP_TARGET - c_lo) / denom, 0.0, 1.0)
    s_at = np.where(idx == 0, 0.0, s_lo + frac * (s_hi - s_lo))
    s_at = np.where(any_reached, s_at, 1.0)
    return np.stack(np.broadcast_arrays(dp30, s_at, z0 / WHEEL_RADIUS_M, rc), axis=-1)


class reference:  # noqa: N801  (namespace of scalar functions, used by tests only)
    """Scalar, pure-Python transcription of the formulas above (independent check)."""

    @staticmethod
    def static_sinkage(W, k_c, k_phi, n, g, beta):
        kpe = k_phi * (g / 9.81) ** beta
        num = 3.0 * W
        den = (3.0 - n) * 0.06 * (k_c / 0.06 + kpe) * math.sqrt(0.24)
        return (num / den) ** (2.0 / (2.0 * n + 1.0))

    @staticmethod
    def thrust(s, z, W, c, phi, K):
        length = 0.12 * math.acos(1.0 - min(max(z / 0.12, 0.0), 1.0))
        hmax = 0.06 * length * c + W * math.tan(phi)
        a = min(abs(s), 1.0)
        if a < 1e-3:
            h = hmax * a * length / (2.0 * K)
        else:
            h = hmax * (1.0 - K / (a * length) * (1.0 - math.exp(-a * length / K)))
        return math.copysign(h, s) if s != 0 else 0.0

    @staticmethod
    def compaction_resistance(z, k_c, k_phi, n, g, beta):
        kpe = k_phi * (g / 9.81) ** beta
        return 0.06 * (k_c / 0.06 + kpe) * max(z, 0.0) ** (n + 1.0) / (n + 1.0)

    @staticmethod
    def slip_sinkage_target(z0, s, a_s):
        a = min(abs(s), 1.0)
        return z0 * (1.0 + a_s * a / (1.0 - a + 0.05))
