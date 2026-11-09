"""Environment: determinism, clone/restore, Gymnasium API, physics and task rules."""

from dataclasses import replace

import numpy as np
import pytest

from orion_learning import env as E
from orion_learning import estimator as est
from orion_learning import obs_spec as S
from orion_learning import soil as soil_mod

CRUISE = np.array([0.6, 0.0])      # 0.3 m/s straight


def _roll(env, steps, action=CRUISE):
    out = []
    for _ in range(steps):
        obs, rew, term, trunc, info = env.step(np.tile(action, (env.num_envs, 1)))
        out.append((obs.copy(), rew.copy(), term.copy(), info['privileged'].copy()))
    return out


def test_deterministic_given_seed():
    a, b, c = (E.OrionEnv(6, seed=s) for s in (3, 3, 4))
    oa, ob, oc = a.reset()[0], b.reset()[0], c.reset()[0]
    assert np.array_equal(oa, ob) and not np.array_equal(oa, oc)
    ra, rb = _roll(a, 60), _roll(b, 60)
    for x, y in zip(ra, rb):
        for u, v in zip(x, y):
            assert np.array_equal(u, v)


def test_clone_restore_round_trip_is_exact():
    env = E.OrionEnv(5, seed=0)
    env.reset()
    _roll(env, 30)
    snap = env.clone_state()
    first = _roll(env, 40, np.array([0.8, 0.3]))
    env.restore_state(snap)
    second = _roll(env, 40, np.array([0.8, 0.3]))
    for x, y in zip(first, second):
        for u, v in zip(x, y):
            assert np.array_equal(u, v)
    # The snapshot is a deep copy: stepping did not modify it.
    env.restore_state(snap)
    assert np.array_equal(env.state['pose'], snap['arrays']['pose'])


def test_tile_state_builds_branch_envs():
    env = E.OrionEnv(3, seed=0)
    env.reset()
    _roll(env, 20)
    tiled = E.OrionEnv.tile_state(env.clone_state(), 4, rows=np.array([0, 2]))
    branch = E.OrionEnv(8, seed=9, auto_reset=False)
    branch.restore_state(tiled)
    assert np.array_equal(branch.state['pose'][4:8], np.repeat(env.state['pose'][2:3], 4, 0))
    with pytest.raises(ValueError):
        E.OrionEnv(7, seed=0).restore_state(tiled)


def test_gymnasium_env_checker():
    from gymnasium.utils.env_checker import check_env

    from orion_learning.gym_env import OrionGymEnv
    check_env(OrionGymEnv(), skip_render_check=True)


def test_observation_is_built_by_obs_spec_from_the_sensors():
    env = E.OrionEnv(4, seed=1)
    env.reset()
    _roll(env, 10)
    obs, _, _, _, info = env.step(np.tile(CRUISE, (4, 1)))
    s = info['sensors']
    ref = np.stack([S.build_observation(
        prev_cmd=info['cmd_twist'][i], wheel_speed=s['wheel_speed'][i],
        wheel_effort=s['wheel_effort'][i], effort_max=E.sim_effort_max,
        body_accel_xy=s['accel'][i], yaw_rate=float(s['yaw_rate'][i]), roll=float(s['roll'][i]),
        pitch=float(s['pitch'][i]), body_speed=float(s['body_speed'][i]),
        speed_confidence=float(s['confidence'][i]), slip_side=s['slip_side'][i],
        sinkage_proxy=float(s['sinkage_proxy'][i])) for i in range(4)])
    assert np.array_equal(obs, ref)
    assert E.sim_effort_max == E.TAU_MAX


def test_batched_obs_spec_matches_scalar():
    rng = np.random.default_rng(0)
    w, v, r = rng.normal(0, 3, (7, 4)), rng.normal(0, 0.3, 7), rng.normal(0, 0.5, 7)
    batch = S.wheel_slip(w, v, r)
    for i in range(7):
        assert np.array_equal(batch[i], S.wheel_slip(list(w[i]), float(v[i]), float(r[i])))
    assert np.allclose(E.rate_limit_batch(np.array([[0.0, 0.0]]), np.array([[0.5, 1.0]])),
                       [S.rate_limit((0.0, 0.0), (0.5, 1.0), E.DT)])
    assert np.allclose(E.action_to_twist_batch(np.array([[2.0, -0.3]])),
                       [S.action_to_twist([2.0, -0.3])])


def test_privileged_vector_layout():
    env = E.OrionEnv(3, E.cell_config('heavy-loose', 1.62), seed=0)
    _, info = env.reset()
    p = info['privileged']
    assert p.shape == (3, S.PRIVILEGED_DIM) and p.dtype == np.float32
    assert np.allclose(p[:, 8], 22.0 / 14.0) and np.allclose(p[:, 9], 1.62 / 9.81)
    soil = soil_mod.Soil.from_stacked(env.state['soil'][0, 0])
    d = soil_mod.traction_descriptors(22.0 * 1.62 / 4, soil, 1.62, E.CELL_BETA)
    assert np.allclose(p[0, [0, 2, 4, 6]], d, rtol=1e-5)       # left side, both sides equal
    assert np.allclose(p[0, [1, 3, 5, 7]], d, rtol=1e-5)


def test_steady_cruise_slip_matches_soil_model_equilibrium():
    """Env physics and soil.py agree: steady slip is where steady drawbar pull is zero."""
    env = E.OrionEnv(2, E.cell_config('heavy-loose'), seed=0, auto_reset=False)
    env.reset()
    _roll(env, 120)
    slip = np.mean(E.physical_slip(env.state['w'], env.state['vel'][:, :1]))
    soil = soil_mod.Soil.from_stacked(env.state['soil'][0, 0])
    W = 22.0 * 9.81 / 4
    grid = np.linspace(0.01, 0.6, 600)
    dp = soil_mod.steady_drawbar_pull(grid, W, soil, 9.81, E.CELL_BETA)
    s_eq = grid[np.argmax(dp >= 0)]
    assert slip == pytest.approx(s_eq, abs=0.01)
    assert env.state['vel'][:, 0].mean() == pytest.approx(0.3 * (1 - s_eq), abs=0.01)


def test_action_latency_delays_the_applied_twist():
    cfg = replace(E.cell_config('light-packed'), latency_steps=(3, 3))
    env = E.OrionEnv(1, cfg, seed=0)
    env.reset()
    applied = []
    for _ in range(4):
        _, _, _, _, info = env.step(np.array([[1.0, 0.0]]))
        applied.append(info['applied_twist'][0, 0])
    # Rate limit 1 m/s^2 -> 0.05 m/s per step; latency 3 -> first two steps still zero.
    assert applied == pytest.approx([0.0, 0.0, 0.05, 0.10])


def test_spinning_in_mud_ends_in_belly_out_with_penalty():
    mud = E.SoilRanges(k_c_kn=(0.5, 0.5), k_phi_kn=(400, 400), n=(1.3, 1.3), c_kpa=(0, 0),
                       phi_deg=(25, 25), K_m=(0.04, 0.04), a_s=(2.0, 2.0))
    cfg = replace(E.cell_config('heavy-loose'), soil=mud)
    env = E.OrionEnv(1, cfg, seed=0, auto_reset=False)
    env.reset()
    for k in range(400):
        _, rew, term, _, info = env.step(np.array([[1.0, 0.0]]))
        if term[0]:
            break
    assert info['belly_out'][0] and not info['success'][0]
    assert env.state['z'].max() >= E.BELLY_OUT_Z
    remaining = cfg.max_steps - (k + 1)
    # Failure costs the belly-out penalty plus the step penalty of the time left.
    assert rew[0] < cfg.w_belly_out + cfg.w_step * remaining + 1.0


def test_hard_stuck_after_10_s_of_commanded_motion_without_progress():
    cfg = replace(E.cell_config('light-packed'), latency_steps=(1, 1))
    env = E.OrionEnv(1, cfg, seed=0, auto_reset=False)
    env.reset()
    env.state['fault'][:] = 0.0          # motors deliver no torque: commanding, not moving
    steps = 0
    while True:
        _, _, term, _, info = env.step(np.array([[0.6, 0.0]]))
        steps += 1
        if term[0]:
            break
    assert info['hard_stuck'][0]
    # Clock starts once |r w_cmd| > 0.1 m/s, i.e. after 3 steps of the 1 m/s^2 rate limit.
    assert steps == round(E.HARD_STUCK_TIME / E.DT) + 2


def test_goal_success_and_effective_speed_on_packed_ground():
    cfg = replace(E.cell_config('light-packed'), heading_deg=(0.0, 0.0))
    env = E.OrionEnv(1, cfg, seed=0)
    env.reset()
    for _ in range(400):
        _, rew, term, _, info = env.step(np.array([CRUISE]))
        if term[0]:
            break
    assert info['success'][0] and rew[0] > cfg.w_goal - 1.0
    assert info['effective_speed'][0] == pytest.approx(0.25, abs=0.03)
    assert 0 < info['episode_length'][0] < 400
    assert env.state['t'][0] == 0           # auto-reset started a new episode


def test_leaving_the_fence_terminates():
    cfg = replace(E.cell_config('light-packed'), heading_deg=(30.0, 30.0))
    env = E.OrionEnv(1, cfg, seed=0, auto_reset=False)
    env.reset()
    for _ in range(600):
        _, _, term, _, info = env.step(np.array([[0.6, 0.6]]))     # arcs left
        if term[0]:
            break
    assert info['out_of_fence'][0] and not info['success'][0]


def test_stored_effort_model_is_current():
    """A refit on other seeds predicts the same effort over the driving range (c0 and c1 trade
    off against each other inside the informative window, so coefficients are not compared)."""
    fresh = E.calibrate_effort_model(seed=3, num_envs=16, steps=200)
    stored = E.SIM_EFFORT_MODEL
    w, dw = np.meshgrid(np.linspace(0.5, 4.2, 12), np.linspace(-2.0, 2.0, 5))
    for sign in (1.0, -1.0):
        args = (sign * w[..., None], sign * dw[..., None], np.zeros(w.shape))
        diff = np.max(np.abs(fresh.predict(*args) - stored.predict(*args)))
        assert diff < 0.04 * stored.ref           # N.m: at most 0.04 in sinkage-proxy units
    assert fresh.coefficients[3] == 0.0 and stored.ref == pytest.approx(E.TAU_MAX - 0.30494)


def test_sinkage_proxy_separates_packed_from_loose():
    means = {}
    for cell in ('light-packed', 'heavy-loose'):
        env = E.OrionEnv(4, E.cell_config(cell), seed=0, auto_reset=False)
        env.reset()
        obs = [o for o, *_ in _roll(env, 120)][40:]
        means[cell] = np.mean([o[:, S.field_slice('sinkage_proxy')] for o in obs])
    assert means['light-packed'] < 0.05 < 0.3 < means['heavy-loose']


def test_speed_estimate_degrades_during_dropouts():
    cfg = replace(E.cell_config('light-packed'), p_fix_dropout=1.0, dropout_rate_hz=5.0,
                  dropout_duration_s=(5.0, 5.0))
    env = E.OrionEnv(8, cfg, seed=0, auto_reset=False)
    env.reset()
    sens = [info['sensors'] for *_, info in
            (env.step(np.tile(CRUISE, (8, 1))) for _ in range(60))]
    conf = np.array([s['confidence'] for s in sens])
    assert np.mean(conf[-1]) < est.speed_confidence(3.0, 0.0) + 0.05
    # The env reports the shared formula on its own Kalman state (same code as the rover).
    s = sens[-1]
    assert np.allclose(conf[-1], est.speed_confidence(s['time_since_fix'], s['speed_std']))
    assert np.all(s['time_since_fix'][~s['has_fix']] > 0.0)


def test_effort_residual_held_outside_informative_window():
    """Turning hard (|yaw| > 0.2 rad/s) or standing still freezes the residual (as on the rover)."""
    env = E.OrionEnv(2, E.cell_config('heavy-loose'), seed=0, auto_reset=False)
    env.reset()
    _roll(env, 40)
    before = env.state['resid_lp'].copy()
    for _ in range(10):
        _, _, _, _, info = env.step(np.zeros((2, 2)))        # wheels slow down and stop
    stopped = np.abs(info['sensors']['wheel_speed']) < est.INFORMATIVE_MIN_WHEEL_SPEED
    assert stopped.any()
    held = env.state['resid_lp'][stopped]
    for _ in range(5):
        env.step(np.zeros((2, 2)))
    assert np.array_equal(env.state['resid_lp'][stopped], held)
    assert not np.array_equal(before, env.state['resid_lp'])
