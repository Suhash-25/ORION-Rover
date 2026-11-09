"""Shared terrain-estimator maths (SPEC section 4): Kalman, confidence, low-pass, effort model."""

import math

import numpy as np
import pytest

from orion_learning import estimator as est

# Rows FL FR RL RR, columns c0 (Coulomb), c1 (viscous), c2 (inertia), c3 (slope).
COEFS = np.array([[0.40, 0.080, 0.050, -2.0], [0.35, 0.090, 0.045, -1.8],
                  [0.45, 0.070, 0.055, -2.2], [0.38, 0.085, 0.050, -1.9]])


def signed_effort(w, dwdt, pitch):
    """Physical signed effort I = d c0 + c1 w + c2 dw/dt + c3 sin(pitch) for each wheel."""
    d = np.sign(w)
    return (d * COEFS[:, 0] + COEFS[:, 1] * w + COEFS[:, 2] * dwdt
            + COEFS[:, 3] * np.sin(np.asarray(pitch))[..., None])


def drive(n=6000, seed=0):
    """Forward/reverse driving with turns and slope at 100 Hz: w, dw/dt (n, 4), pitch (n,)."""
    t = np.arange(n) / 100.0
    ph = np.random.default_rng(seed).uniform(0, 2 * np.pi, 4)
    w = (4.0 * np.sin(2 * np.pi * t / 9.0)[:, None]
         + 1.2 * np.sin(2 * np.pi * t / 5.0 + ph[0])[:, None] * np.array([-1, 1, -1, 1])
         + 0.6 * np.sin(2 * np.pi * t[:, None] / 2.3 + ph))
    return w, np.gradient(w, t, axis=0), 0.15 * np.sin(2 * np.pi * t / 23.0)


def test_kalman_predict_and_update():
    v, P = est.kf_predict(0.1, 0.0, 0.5, 0.2)
    assert v == pytest.approx(0.2) and P == pytest.approx(0.1 ** 2 * 0.2)
    v2, P2 = est.kf_update(v, P, 0.3, True)
    gain = P / (P + 0.02 ** 2)
    assert v2 == pytest.approx(0.2 + gain * 0.1) and P2 < 0.02 ** 2
    v3, P3 = est.kf_update(np.array([v, v]), np.array([P, P]), 0.3, np.array([False, True]),
                           sigma=est.FIX_SIGMA_LIDAR)
    assert v3[0] == v and P3[0] == P and P3[1] < P


def test_confidence_formula_m7():
    """c = exp(-t/3) * clip(1 - std/0.1, 0, 1): wheel odometry plays no part (council M7)."""
    assert est.speed_confidence(0.0, 0.0) == pytest.approx(1.0)
    assert est.speed_confidence(3.0, 0.0) == pytest.approx(math.exp(-1))
    assert est.speed_confidence(0.0, 0.05) == pytest.approx(0.5)
    assert est.speed_confidence(1.5, 0.02) == pytest.approx(math.exp(-0.5) * 0.8)
    assert est.speed_confidence(0.0, 0.1) == 0.0 and est.speed_confidence(0.0, 0.3) == 0.0
    assert est.speed_confidence(math.inf, 0.0) == 0.0
    assert est.speed_confidence(-1.0, 0.0) == pytest.approx(1.0)       # clock jitter


def test_confidence_falls_within_a_second_of_a_dropout_at_env_rates():
    """With external fixes every 0.05 s the steady std is ~0.016 m/s (c ~ 0.84); IMU-only it
    grows as 0.1 sqrt(t), so c < 0.3 after ~0.4 s and the speed rule then needs a fresh fix."""
    v, P = 0.0, 0.1
    for _ in range(200):
        v, P = est.kf_predict(v, P, 0.0, 0.05)
        v, P = est.kf_update(v, P, 0.0, True)
    assert est.speed_confidence(0.0, math.sqrt(P)) == pytest.approx(0.84, abs=0.02)
    t = 0.0
    while est.speed_confidence(t, math.sqrt(P)) >= 0.3:
        v, P = est.kf_predict(v, P, 0.0, 0.05)
        t += 0.05
    assert 0.3 <= t <= 0.5


def test_lowpass_is_exact_first_order_lag_and_holds_masked_entries():
    assert est.lowpass_alpha(0.05) == pytest.approx(1 - math.exp(-0.2))
    assert est.lowpass_alpha(0.0) == 0.0 and est.lowpass_alpha(-1.0) == 0.0
    # Two half steps equal one full step (irregular runtime sampling filters like the env).
    y1 = est.lowpass(est.lowpass(0.0, 1.0, 0.025), 1.0, 0.025)
    assert y1 == pytest.approx(est.lowpass(0.0, 1.0, 0.05))
    y = est.lowpass(np.zeros(3), np.ones(3), 0.05, mask=np.array([True, False, True]))
    assert y[1] == 0.0 and y[0] == pytest.approx(est.lowpass_alpha(0.05))


def test_informative_window():
    m = est.informative_mask([0.0, 0.4, 3.0, -3.0], 0.0)
    assert list(m) == [False, False, True, True]
    assert not est.informative_mask(np.full(4, 3.0), 0.5).any()           # turning hard
    batch = est.informative_mask(np.full((2, 4), 3.0), np.array([0.0, -0.21]))
    assert batch[0].all() and not batch[1].any()


def test_residual_zero_on_model_and_positive_for_extra_load_both_directions():
    """WP3 convention: along the direction of rotation, so reversing out of sand reads +."""
    m = est.EffortModel(COEFS, ref=5.0)
    for w in (5.0, -5.0):
        wv = np.full(4, w)
        nominal = signed_effort(wv, np.zeros(4), 0.1)
        assert np.allclose(m.residual(nominal, wv, np.zeros(4), 0.1), 0.0, atol=1e-12)
        loaded = nominal + np.sign(w) * 1.0
        assert np.allclose(m.residual(loaded, wv, np.zeros(4), 0.1), 1.0)
        assert m.proxy(m.residual(loaded, wv, np.zeros(4), 0.1)) == pytest.approx(0.2)


def test_one_row_model_applies_to_every_wheel_and_batches():
    m = est.EffortModel(COEFS[0], ref=2.0)
    w = np.full((3, 4), 2.0)
    r = m.residual(np.zeros((3, 4)), w, np.zeros((3, 4)), np.zeros(3))
    assert r.shape == (3, 4) and np.allclose(r, -(0.40 + 0.16))
    assert m.proxy([1.0, 1.0, 1.0, 1.0]) == pytest.approx(0.5)
    assert m.proxy([-3.0] * 4) == 0.0 and m.proxy([9.0] * 4) == 1.0
    with pytest.raises(ValueError):
        est.EffortModel(np.zeros(3), ref=1.0)


def test_fit_recovers_known_coefficients_and_ignores_turning():
    w, dwdt, pitch = drive()
    rng = np.random.default_rng(3)
    yaw = np.where(np.arange(len(w)) % 3 == 0, 0.8, 0.0)
    effort = (signed_effort(w, dwdt, pitch) + rng.normal(0, 0.01, w.shape)
              + np.where(yaw > 0, 4.0, 0.0)[:, None])            # scrub while turning hard
    fit = est.fit_effort_model(w, dwdt, pitch, effort, yaw)
    assert np.allclose(fit.coefficients, COEFS, rtol=0.03, atol=0.005)
    assert np.all(fit.r2 > 0.99) and fit.slope_identified


def test_fit_on_flat_ground_sets_slope_term_to_zero():
    w, dwdt, _ = drive()
    pitch = np.full(len(w), 0.001)
    fit = est.fit_effort_model(w, dwdt, pitch, signed_effort(w, dwdt, pitch), np.zeros(len(w)))
    assert not fit.slope_identified and np.all(fit.coefficients[:, 3] == 0.0)
    assert np.allclose(fit.coefficients[:, :3], COEFS[:, :3], rtol=0.03, atol=0.01)


def test_fit_needs_informative_data():
    w = np.full((100, 4), 0.1)
    with pytest.raises(ValueError, match='only 0 samples'):
        est.fit_effort_model(w, w * 0, np.zeros(100), w, np.zeros(100))
