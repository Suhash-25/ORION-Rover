"""Training vs evaluation distribution (council B1) and per-episode randomisation (W2, m4-m6)."""

from dataclasses import replace

import numpy as np
import pytest

from orion_learning import env as E
from orion_learning import soil as soil_mod
from orion_learning.baselines import B0Controller

K_PHI, A_S, K = (soil_mod.SOIL_KEYS.index(k) for k in ('k_phi', 'a_s', 'K'))


def _episodes(cfg, n=4000, seed=0):
    env = E.OrionEnv(n, cfg, seed=seed, auto_reset=False)
    env.reset()
    return env


def test_patch_frequency_is_one_half_and_spread_across_the_path():
    st = _episodes(E.EnvConfig()).state
    assert st['patch_on'].mean() == pytest.approx(0.5, abs=0.025)
    y = st['patch_xy'][:, 1]
    assert y.min() >= -0.9 and y.max() <= 0.9 and np.std(y) == pytest.approx(1.8 / 12 ** 0.5,
                                                                             rel=0.05)
    # The 2 x 0.26 m wheel track crosses a patch of radius r centred at y when |y| < r + 0.26.
    crosses = np.abs(y) < st['patch_r'] + E.TRACK / 2
    assert 0.35 < np.mean(crosses & st['patch_on']) < 0.5 * 0.85


def test_patch_soil_is_a_degraded_copy_of_the_main_soil():
    st = _episodes(E.EnvConfig(), n=2000).state
    main, patch = st['soil'][:, 0], st['soil'][:, 2]
    ratio = patch / main
    assert np.all((ratio[:, K_PHI] >= 0.4) & (ratio[:, K_PHI] <= 0.9))
    assert np.median(ratio[:, K_PHI]) == pytest.approx((0.4 * 0.9) ** 0.5, rel=0.05)  # log-U
    assert np.all((ratio[:, A_S] >= 1.0) & (ratio[:, A_S] <= 2.0))
    assert np.all((ratio[:, K] >= 1.0) & (ratio[:, K] <= 1.5))
    others = [i for i in range(len(soil_mod.SOIL_KEYS)) if i not in (K_PHI, A_S, K)]
    assert np.array_equal(patch[:, others], main[:, others])


def test_mixture_rejection_sampling_on_peak_drawbar_pull():
    cfg = replace(E.EnvConfig(), dp_threshold=(0.03, 0.03), p_dp_restricted=0.4)
    env = _episodes(cfg)
    st = env.state
    phys = {'left': st['soil'][:, 0], 'right': st['soil'][:, 1], 'beta': st['beta'],
            'g': st['g'], 'mass': st['mass'], 'm_fH': st['m_fH'], 'm_fR': st['m_fR']}
    peak = E.peak_dp(phys)
    restricted = st['dp_restricted']
    assert restricted.mean() == pytest.approx(0.4, abs=0.025)
    assert np.all(peak[restricted] >= 0.03)
    # The unrestricted 60 % follow the SPEC ranges: mostly impassable (council: ~23 %).
    assert 0.15 < np.mean(peak[~restricted] >= 0.03) < 0.35


def test_curriculum_anneals_the_threshold_over_the_given_steps():
    cfg = replace(E.EnvConfig(), dp_threshold=(0.08, 0.0), dp_curriculum_steps=100)
    assert [cfg.dp_threshold_at(s) for s in (0, 50, 100, 1000)] == pytest.approx(
        [0.08, 0.04, 0.0, 0.0])
    assert replace(cfg, dp_curriculum_steps=0).dp_threshold_at(0) == 0.0
    env = E.OrionEnv(2, cfg, seed=0)
    env.reset()
    assert env.dp_threshold() == pytest.approx(0.08)
    for _ in range(50):
        env.step(np.zeros((2, 2)))
    assert env.dp_threshold() == pytest.approx(0.04)


def test_spec_eval_config_keeps_the_spec_ranges():
    st = _episodes(E.spec_eval_config(), n=2000).state
    assert st['patch_on'].all() and not st['dp_restricted'].any()
    assert np.abs(st['patch_xy'][:, 1]).max() <= 0.4
    patch = st['soil'][:, 2]
    assert patch[:, K_PHI].max() <= 1000e3 and patch[:, K].min() >= 0.02


def test_per_episode_model_and_sensor_randomisation():
    st = _episodes(E.EnvConfig(), n=4000).state
    dig = st['dig_rate']
    assert dig.min() >= 0.03 and dig.max() <= 0.3
    assert np.median(dig) == pytest.approx((0.03 * 0.3) ** 0.5, rel=0.06)      # log-uniform
    assert 1.0 <= st['rear_k_phi'].min() and st['rear_k_phi'].max() <= 1.5
    bias = np.degrees(st['imu_bias'])
    assert np.abs(bias).max() <= 2.0 and np.std(bias) == pytest.approx(4 / 12 ** 0.5, rel=0.05)
    assert 0.7 <= st['i_ref_factor'].min() and st['i_ref_factor'].max() <= 1.3


def test_imu_bias_and_rear_compaction_reach_the_sensors_and_physics():
    cfg = replace(E.cell_config('light-loose'), imu_bias_deg=(2.0, 2.0),
                  rear_k_phi_factor=(1.5, 1.5))
    env = E.OrionEnv(1, cfg, seed=0, auto_reset=False)
    _, info = env.reset()
    roll, pitch = env.tilt()
    assert info['sensors']['roll'][0] == pytest.approx(roll[0] + np.radians(2.0))
    assert info['sensors']['pitch'][0] == pytest.approx(pitch[0] + np.radians(2.0))
    z = env.state['z'][0]
    assert z[2] < z[0] and z[3] < z[1]              # compacted rut: rear sinks less


def _b0_outcomes(cfg, n, seed):
    env = E.OrionEnv(n, cfg, seed=seed, auto_reset=False)
    obs, _ = env.reset()
    b0 = B0Controller(n)
    success = np.zeros(n, dtype=bool)
    done = np.zeros(n, dtype=bool)
    for _ in range(env.config.max_steps):
        obs, _, term, _, info = env.step(b0.act(obs))
        success |= info['success'] & ~done
        done |= term
        if done.all():
            break
    return success.mean()


def test_b0_completes_about_half_of_the_training_episodes():
    """Pre-registered band 40-60 % (README); measured 0.50 over 800 episodes at 0.03.

    The SPEC distribution, kept for evaluation, is the trap-everywhere one of council B1.
    """
    train = _b0_outcomes(replace(E.EnvConfig(), dp_threshold=(0.03, 0.03)), 240, seed=11)
    spec = _b0_outcomes(E.spec_eval_config(), 120, seed=11)
    print(f'B0 success: training sampler @0.03 {train:.3f}, SPEC distribution {spec:.3f}')
    assert 0.38 <= train <= 0.62
    assert spec < 0.2
