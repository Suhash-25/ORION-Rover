"""Critic labels and calibration maths."""

from dataclasses import replace

import numpy as np
import pytest

from orion_learning import env as E
from orion_learning.risk import (branch_labels, brier, expected_calibration_error,
                                 fit_temperature)


def test_ece_and_brier():
    p = np.array([0.1, 0.1, 0.9, 0.9])
    assert expected_calibration_error(p, np.array([0, 0, 1, 1]), 10) == pytest.approx(0.1)
    assert expected_calibration_error(np.array([0.3, 0.7]), np.array([0.3, 0.7])) == 0.0
    assert brier(p, np.array([0, 0, 1, 1])) == pytest.approx(0.01)


def test_temperature_scaling_recovers_overconfidence():
    rng = np.random.default_rng(0)
    true_logit = rng.normal(0, 1.5, 4000)
    y = (rng.random(4000) < 1 / (1 + np.exp(-true_logit))).astype(float)
    T = fit_temperature(3.0 * true_logit, y)          # a 3x over-confident model
    assert T == pytest.approx(3.0, rel=0.15)


def _snapshot(cell, steps, action, cfg_mod=None):
    cfg = E.cell_config(cell)
    if cfg_mod:
        cfg = replace(cfg, **cfg_mod)
    env = E.OrionEnv(2, cfg, seed=0, auto_reset=False)
    env.reset()
    for _ in range(steps):
        env.step(np.tile(action, (2, 1)))
    return cfg, env.clone_state()


def test_branch_labels_zero_on_packed_ground():
    cfg, snap = _snapshot('light-packed', 20, [0.6, 0.0])
    y = branch_labels(cfg, snap, branches=4, horizon_s=10.0, seed=1)
    assert y.shape == (2,) and np.all(y == 0.0)


def test_branch_labels_one_when_already_digging_in():
    mud = E.SoilRanges(k_c_kn=(0.5, 0.5), k_phi_kn=(400, 400), n=(1.3, 1.3), c_kpa=(0, 0),
                       phi_deg=(25, 25), K_m=(0.04, 0.04), a_s=(2.0, 2.0))
    cfg, snap = _snapshot('heavy-loose', 20, [1.0, 0.0], {'soil': mud})
    y = branch_labels(cfg, snap, branches=4, horizon_s=10.0, seed=1)
    assert np.all(y == 1.0)
