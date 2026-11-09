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


QUIET = {'latency_steps': (1, 1), 'p_fix_dropout': 0.0, 'noise_speed_fix': 0.0,
         'noise_accel': 0.0, 'accel_bias_rw': 0.0, 'noise_wheel_speed': 0.0, 'noise_gyro': 0.0}


def _stalled_snapshot(steps):
    """Rover with dead motors on packed ground (noise-free) after `steps` of 0.3 m/s cmd."""
    cfg = replace(E.cell_config('light-packed'), **QUIET)
    env = E.OrionEnv(1, cfg, seed=0, auto_reset=False)
    env.reset()
    env.state['fault'][:] = 0.0
    for _ in range(steps):
        _, _, _, _, info = env.step(np.array([[0.6, 0.0]]))
    return cfg, env.clone_state(), info


def test_label_horizon_counts_from_the_stuck_onset_not_the_halt():
    """A stalled rover is ENTRAPPED >= 23 s after it stops, but dated at FLAG_STUCK onset.

    From a state 1 s in, the branches escape, cruise, get flagged a few seconds later and are
    ENTRAPPED 20 s after that: positive for a 10 s horizon, negative for a 1 s horizon.
    """
    cfg, snap, _ = _stalled_snapshot(20)
    assert branch_labels(cfg, snap, branches=2, horizon_s=10.0, seed=1)[0] == 1.0
    assert branch_labels(cfg, snap, branches=2, horizon_s=1.0, seed=1)[0] == 0.0


def test_state_inside_a_doomed_stuck_episode_is_positive_at_any_horizon():
    cfg, snap, info = _stalled_snapshot(200)       # flagged at 3.15 s, now 10 s in
    assert 0.0 <= info['stuck_onset'][0] < 10.0
    assert branch_labels(cfg, snap, branches=2, horizon_s=0.5, seed=1)[0] == 1.0


def test_critic_collection_runs_the_gate_with_escapes_probes_and_retreats():
    """Council M9: the visiting student acts through CriticGate (stuck -> escape + rho)."""
    import torch

    from orion_learning.nets import TCNAdapter
    from orion_learning.ppo import ActorCritic, PPOConfig
    from orion_learning.risk import collect, CriticConfig
    torch.manual_seed(0)
    teacher = ActorCritic(PPOConfig(hidden=[8], encoder_hidden=[8]), use_latent=True)
    # 20 s: E[0] (7 s) holds a rover in ESCAPE long enough that a 10 s rollout leaves the
    # retreat band under 1 % of the steps.
    cc = CriticConfig(num_envs=8, rollout_steps=400, sample_every=1000, gate_p_active=0.5)
    _, _, y, modes = collect(teacher, TCNAdapter(4), E.EnvConfig(), cc, seed=0)
    assert len(y) == 0                            # sample_every > rollout: no labels
    assert modes['frac_explore'] > 0.3
    for m in ('frac_probe', 'frac_retreat', 'frac_escape'):
        assert modes[m] > 0.02, modes
