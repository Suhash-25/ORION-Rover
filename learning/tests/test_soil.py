"""Wheel-soil model properties (SPEC section 6) and vectorised == scalar reference."""

import math

import numpy as np
import pytest

from orion_learning import soil as S

SAND = S.Soil(k_c=2e3, k_phi=1.2e6, n=1.2, c=1e3, phi=math.radians(34), K=0.015, a_s=0.5)


def _random_soils(rng, n):
    return S.Soil(k_c=rng.uniform(0.5e3, 15e3, n), k_phi=rng.uniform(400e3, 2500e3, n),
                  n=rng.uniform(0.7, 1.3, n), c=rng.uniform(0, 2e3, n),
                  phi=np.radians(rng.uniform(25, 40, n)), K=rng.uniform(0.01, 0.04, n),
                  a_s=rng.uniform(0.3, 2.0, n))


def test_static_sinkage_grows_with_load():
    W = np.linspace(5, 120, 50)
    z = S.static_sinkage(W, SAND, 9.81, 0.5)
    assert np.all(np.diff(z) > 0)


def test_slip_sinkage_grows_with_abs_slip_and_is_symmetric():
    s = np.linspace(0, 1, 101)
    z = S.slip_sinkage_target(0.01, s, 1.0)
    assert z[0] == pytest.approx(0.01)
    assert np.all(np.diff(z) > 0)
    assert np.allclose(S.slip_sinkage_target(0.01, -s, 1.0), z)


def test_thrust_rises_to_shear_strength_as_slip_goes_to_one():
    z, W = 0.02, 40.0
    hmax = S.max_thrust(z, W, SAND)
    assert hmax == pytest.approx(0.06 * S.contact_length(z) * 1e3 + W * math.tan(SAND.phi))
    s = np.linspace(0, 1, 201)
    H = S.thrust(s, z, W, SAND)
    assert np.all(np.diff(H) > 0) and np.all(H <= hmax)
    # Shear fully mobilised (K << slip displacement l): H(s = 1) -> A c + W tan(phi).
    stiff = S.Soil(**{**SAND.__dict__, 'K': 1e-4})
    assert S.thrust(1.0, z, W, stiff) == pytest.approx(hmax, rel=2e-3)
    # Thrust is odd in slip (braking = negative thrust).
    assert S.thrust(-0.4, z, W, SAND) == pytest.approx(-S.thrust(0.4, z, W, SAND))


def test_small_slip_guard_is_continuous():
    z, W = 0.02, 40.0
    below = S.thrust(S.SMALL_SLIP * (1 - 1e-6), z, W, SAND)
    above = S.thrust(S.SMALL_SLIP * (1 + 1e-6), z, W, SAND)
    # The SPEC guard is the first-order Taylor term x/2 of 1 - (1 - e^-x)/x, x = |s| l / K,
    # so the jump at the switch is the second-order term, relative size x/3.
    x = S.SMALL_SLIP * S.contact_length(z) / SAND.K
    assert below == pytest.approx(above, rel=x / 3 * 1.1)
    assert S.thrust(0.0, z, W, SAND) == 0.0


def test_compaction_resistance_monotone_in_sinkage():
    z = np.linspace(0, 0.1, 100)
    rc = S.compaction_resistance(z, SAND, 9.81, 0.5)
    assert rc[0] == 0.0 and np.all(np.diff(rc) > 0)


def test_gravity_coupling_lowers_k_phi_only_with_positive_beta():
    assert S.k_phi_eff(1e6, 1.62, 0.5) < S.k_phi_eff(1e6, 9.81, 0.5)
    assert S.k_phi_eff(1e6, 1.62, 0.0) == pytest.approx(1e6)
    assert S.k_phi_eff(1e6, 9.81, 1.0) == pytest.approx(1e6)
    # Same load, softer soil at low g with beta > 0: deeper static sinkage.
    W = 30.0
    assert S.static_sinkage(W, SAND, 1.62, 1.0) > S.static_sinkage(W, SAND, 1.62, 0.0)


def test_vectorised_matches_scalar_reference():
    rng = np.random.default_rng(0)
    n = 200
    soil = _random_soils(rng, n)
    W, g, beta = rng.uniform(5, 80, n), rng.uniform(1.62, 9.81, n), rng.uniform(0, 1, n)
    s, z = rng.uniform(-1, 1, n), rng.uniform(0, 0.1, n)
    s[:10] = rng.uniform(-1e-3, 1e-3, 10)          # exercise the small-slip branch
    z0 = S.static_sinkage(W, soil, g, beta)
    H = S.thrust(s, z, W, soil)
    rc = S.compaction_resistance(z, soil, g, beta)
    zt = S.slip_sinkage_target(z0, s, soil.a_s)
    R = S.reference
    for i in range(n):
        args = (soil.k_c[i], soil.k_phi[i], soil.n[i], g[i], beta[i])
        assert z0[i] == pytest.approx(R.static_sinkage(W[i], *args), rel=1e-12)
        assert H[i] == pytest.approx(
            R.thrust(s[i], z[i], W[i], soil.c[i], soil.phi[i], soil.K[i]), rel=1e-10, abs=1e-12)
        assert rc[i] == pytest.approx(R.compaction_resistance(z[i], *args), rel=1e-12)
        assert zt[i] == pytest.approx(R.slip_sinkage_target(z0[i], s[i], soil.a_s[i]), rel=1e-12)


PACKED = S.Soil(k_c=12e3, k_phi=2.5e6, n=0.8, c=2e3, phi=math.radians(38), K=0.01, a_s=0.3)


def test_descriptors_are_consistent_with_the_steady_drawbar_pull():
    W, g, beta = 14 * 9.81 / 4, 9.81, 0.5
    d = S.traction_descriptors(W, PACKED, g, beta)
    dp30, s10, z0r, rcw = d
    assert dp30 == pytest.approx(S.steady_drawbar_pull(0.3, W, PACKED, g, beta) / W)
    assert 0.0 < s10 < 1.0
    assert S.steady_drawbar_pull(s10, W, PACKED, g, beta) / W == pytest.approx(0.1, abs=2e-3)
    assert z0r == pytest.approx(S.static_sinkage(W, PACKED, g, beta) / 0.12)
    assert rcw == pytest.approx(
        S.compaction_resistance(z0r * 0.12, PACKED, g, beta) / W)
    # Looser soil: less pull at 30 % slip, deeper sinkage, more resistance.
    loose = S.traction_descriptors(W, SAND, g, beta)
    assert loose[0] < dp30 and loose[1] > s10 and loose[2] > z0r and loose[3] > rcw
    # A soil that never reaches DP = 0.1 W reports slip 1.0.
    mud = S.Soil(k_c=0.5e3, k_phi=400e3, n=1.3, c=0.0, phi=math.radians(25), K=0.04, a_s=2.0)
    assert S.traction_descriptors(W, mud, g, beta)[1] == 1.0


def test_descriptors_broadcast_over_batches():
    rng = np.random.default_rng(1)
    soil = _random_soils(rng, 6)
    d = S.traction_descriptors(np.full(6, 40.0), soil, 9.81, 0.3)
    assert d.shape == (6, 4) and np.all(np.isfinite(d))
    single = S.traction_descriptors(40.0, S.Soil(*(np.asarray(getattr(soil, k))[2]
                                                   for k in S.SOIL_KEYS)), 9.81, 0.3)
    assert np.allclose(d[2], single)
