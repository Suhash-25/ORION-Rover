"""Env stuck / ENTRAPPED rule == orion_safety's rule (council M2), belly-out timing."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from orion_learning import env as E

SAFETY_YAML = (Path(__file__).resolve().parents[2] / 'robot_ws' / 'src' / 'orion_safety'
               / 'config' / 'safety.yaml')
# Sensors without noise so the estimated body speed is exactly the true one (0 when stuck).
QUIET = {'latency_steps': (1, 1), 'p_fix_dropout': 0.0, 'noise_speed_fix': 0.0,
         'noise_accel': 0.0, 'accel_bias_rw': 0.0, 'noise_wheel_speed': 0.0, 'noise_gyro': 0.0}


def test_thresholds_match_the_rover_safety_config():
    yaml = pytest.importorskip('yaml')
    if not SAFETY_YAML.exists():
        pytest.skip('robot_ws not checked out next to learning/')
    p = yaml.safe_load(SAFETY_YAML.read_text())['orion_safety']['ros__parameters']
    assert (p['stuck_cmd_linear'], p['stuck_body_speed'], p['stuck_min_confidence'],
            p['stuck_slip'], p['stuck_s'], p['hard_stuck_window_s'],
            p['hard_stuck_progress_m']) == (
        E.STUCK_CMD_LINEAR, E.STUCK_BODY_SPEED, E.STUCK_MIN_CONFIDENCE, E.STUCK_SLIP,
        E.STUCK_S, E.HARD_STUCK_WINDOW_S, E.HARD_STUCK_PROGRESS_M)


def _import_monitor():
    """orion_safety.monitor from the monorepo (pure Python, no ROS), or skip."""
    src = SAFETY_YAML.parents[1]
    if not (src / 'orion_safety' / 'monitor.py').exists():
        pytest.skip('robot_ws not checked out next to learning/')
    import sys
    sys.path.insert(0, str(src))
    try:
        from orion_safety import monitor
    finally:
        sys.path.remove(str(src))
    return monitor


@pytest.mark.parametrize('stalled', [True, False])
def test_env_rule_equals_orion_safety_monitor_step_by_step(stalled):
    """Golden test: the env's flags equal SafetyMonitor's on the same inputs every step."""
    from orion_learning.baselines import B0Controller
    mon_mod = _import_monitor()
    n = 6
    cfg = replace(E.EnvConfig(), episode_s=60.0) if not stalled else \
        replace(E.cell_config('light-packed'), **QUIET)
    env = E.OrionEnv(n, cfg, seed=3, auto_reset=False)
    obs, _ = env.reset()
    if stalled:
        env.state['fault'][:] = 0.0
    monitors = [mon_mod.SafetyMonitor(mon_mod.SafetyConfig(geofence_enabled=False))
                for _ in range(n)]
    b0 = B0Controller(n, escape_start_index=np.arange(n))
    alive = np.ones(n, dtype=bool)
    compared = {'stuck': 0, 'entrapped': 0}
    for k in range(1, env.config.max_steps + 1):
        obs, _, term, _, info = env.step(b0.act(obs))
        now = k * E.DT
        s = info['sensors']
        for i in np.flatnonzero(alive):
            m = monitors[i]
            m.on_terrain(now, float(s['body_speed'][i]), float(s['confidence'][i]),
                         tuple(float(x) for x in s['slip_side'][i]))
            m.on_command(now, float(info['cmd_twist'][i, 0]), float(info['cmd_twist'][i, 1]))
            pose = env.state['pose'][i]
            m.on_external_pose(now, float(pose[0]), float(pose[1]), float(pose[2]))
            m.on_joint_states(now, (0.0,) * 4)
            m.on_imu(now, 0.0, 0.0)
            d = m.step(now)
            assert bool(d.flags & mon_mod.FLAG_STUCK) == bool(info['soft_stuck'][i]), (i, k)
            assert bool(d.flags & mon_mod.FLAG_HARD_STUCK) == bool(info['hard_stuck'][i]), \
                (i, k)
            compared['stuck'] += bool(info['soft_stuck'][i])
            compared['entrapped'] += bool(info['hard_stuck'][i])
        alive &= ~term
        if not alive.any():
            break
    assert compared['stuck'] > 0           # the comparison exercised the rule
    if stalled:
        assert compared['entrapped'] == n


def _stalled_rover(steps_max=1200, teleport_at=None):
    """Light-packed rover whose motors give no torque, commanded at 0.3 m/s.

    Returns (first FLAG_STUCK step, terminal step, terminal info). Steps count from 1.
    """
    env = E.OrionEnv(1, replace(E.cell_config('light-packed'), **QUIET), seed=0,
                     auto_reset=False)
    env.reset()
    env.state['fault'][:] = 0.0
    first_flag = None
    for k in range(1, steps_max + 1):
        _, _, term, _, info = env.step(np.array([[0.6, 0.0]]))
        if first_flag is None and info['soft_stuck'][0]:
            first_flag = k
        if k == teleport_at:
            env.state['pose'][0, 0] += 0.15      # somebody pushed it 15 cm
        if term[0]:
            return first_flag, k, info
    raise AssertionError('never terminated')


def test_flag_stuck_after_3_s_and_entrapped_20_s_later():
    first_flag, end, info = _stalled_rover()
    # |cmd v| > 0.1 m/s from step 3 (rate limit 0.05 m/s per step). Like monitor._Held,
    # the flag needs 3 s between the first and the current sample: 61 samples -> step 63.
    assert first_flag == 3 + 60
    # ENTRAPPED once the stuck episode is 20 s old with < 0.1 m in the last 20 s.
    assert end == first_flag + 400
    assert info['hard_stuck'][0] and not info['belly_out'][0]
    # Entrapment is dated at the stuck onset (FLAG_STUCK), not at the HALT.
    assert info['entrapment_time'][0] == pytest.approx(first_flag * E.DT)
    assert info['stuck_onset'][0] == pytest.approx(first_flag * E.DT)


def test_progress_restarts_the_stuck_episode():
    _, end, info = _stalled_rover(teleport_at=200)
    # Until step 600 the window still starts before the 15 cm jump, so every step ends the
    # stuck episode and (still stuck) starts a new one, as monitor._rule_hard_stuck does.
    # The episode that began at step 600 is ENTRAPPED 20 s (400 steps) later.
    assert end == 600 + 400
    assert info['entrapment_time'][0] == pytest.approx(600 * E.DT)


def test_no_flag_without_confidence_or_command():
    cfg = replace(E.cell_config('light-packed'), **QUIET)
    env = E.OrionEnv(2, cfg, seed=0, auto_reset=False)
    env.reset()
    env.state['fault'][:] = 0.0
    for _ in range(200):
        # Rover 0 commands 0.05 m/s (below 0.1), rover 1 nothing.
        _, _, term, _, info = env.step(np.array([[0.1, 0.0], [0.0, 0.0]]))
        assert not info['soft_stuck'].any() and not term.any()


def test_belly_out_is_dated_when_it_happens():
    mud = E.SoilRanges(k_c_kn=(0.5, 0.5), k_phi_kn=(400, 400), n=(1.3, 1.3), c_kpa=(0, 0),
                       phi_deg=(25, 25), K_m=(0.04, 0.04), a_s=(2.0, 2.0))
    env = E.OrionEnv(1, replace(E.cell_config('heavy-loose'), soil=mud), seed=0,
                     auto_reset=False)
    env.reset()
    for k in range(1, 600):
        _, _, term, _, info = env.step(np.array([[1.0, 0.0]]))
        assert info['entrapment_time'][0] == E.NO_TIME or term[0]
        if term[0]:
            break
    assert info['belly_out'][0]
    assert info['entrapment_time'][0] == pytest.approx(k * E.DT)
