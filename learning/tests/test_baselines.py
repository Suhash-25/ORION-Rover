"""B0, stuck detector, escape library, critic gating, and the B0 premise check in sim."""

import numpy as np
import pytest

from orion_learning import env as E
from orion_learning import obs_spec as S
from orion_learning.baselines import (B0Controller, CriticGate, ESCAPE_LIBRARY, EscapeLibrary,
                                      MODE_ESCAPE, MODE_EXPLORE, MODE_PROBE, MODE_RETREAT,
                                      StuckDetector)


def _obs(n=1, cmd_v=0.3, speed=0.3, conf=1.0, slip=(0.0, 0.0)):
    o = np.zeros((n, S.OBS_DIM), np.float32)
    o[:, S.field_slice('prev_cmd')] = [cmd_v / 0.5, 0.0]
    o[:, S.field_slice('body_speed')] = speed / 0.5
    o[:, S.field_slice('speed_confidence')] = conf
    o[:, S.field_slice('slip_side')] = slip
    return o


def test_b0_cruises_at_0_3_mps():
    a = B0Controller(1).act(_obs())
    assert S.action_to_twist(a[0]) == pytest.approx((0.3, 0.0))


def test_slip_governor_cuts_20_percent_per_s_and_recovers_5():
    b0 = B0Controller(1)
    for _ in range(20):                                 # 1 s of slip 0.5
        a = b0.act(_obs(slip=(0.5, 0.1)))
    assert S.action_to_twist(a[0])[0] == pytest.approx(0.3 * 0.8, abs=1e-6)
    for _ in range(20):                                 # 1 s without slip
        a = b0.act(_obs())
    assert S.action_to_twist(a[0])[0] == pytest.approx(0.3 * 0.85, abs=1e-6)
    for _ in range(200):
        a = b0.act(_obs(slip=(0.0, 0.6)))
    assert S.action_to_twist(a[0])[0] == pytest.approx(0.3 * 0.4)  # floor keeps v > 0.1


def test_stuck_rule_needs_3_s_and_confidence():
    det = StuckDetector(2)
    obs = np.concatenate([_obs(speed=0.0), _obs(speed=0.0, conf=0.1)])
    flags = [det.update(obs) for _ in range(60)]
    assert not flags[58][0] and flags[59][0]           # 3.0 s at 20 Hz
    assert not any(f[1] for f in flags)                # low confidence: no speed-based stuck
    det = StuckDetector(1)
    flags = [det.update(_obs(conf=0.0, slip=(0.9, 0.2)))[0] for _ in range(60)]
    assert flags[-1] and not flags[-2]                 # slip rule ignores confidence


def test_escape_library_plays_segments_then_moves_on():
    lib = EscapeLibrary(1)
    lib.start(np.array([True]))
    twists = []
    while lib.active[0]:
        v, w = lib.act()
        twists.append((v[0], w[0]))
    assert len(twists) == round(2.0 / 0.05) and twists[0] == (-0.2, 0.0)
    assert lib.move[0] == 1
    lib.start(np.array([True]))
    assert lib.act()[0][0] == ESCAPE_LIBRARY[1][0][1]


def test_b0_escapes_when_stuck_then_resumes():
    b0 = B0Controller(1)
    stuck = _obs(speed=0.0)
    actions = [b0.act(stuck) for _ in range(61)]
    assert S.action_to_twist(actions[59][0])[0] == pytest.approx(-0.2)   # escape at 3 s
    for _ in range(39):                  # escape = acts 59..98 (2 s), act 98 resets scale
        b0.act(_obs())
    # acts 99 and 100 recover 5 %/s each from scale 0.5.
    assert S.action_to_twist(b0.act(_obs())[0])[0] == pytest.approx(0.3 * 0.505, abs=1e-6)


def test_critic_gate_thresholds():
    gate = CriticGate(4)
    obs = _obs(4)
    act = np.tile([0.9, 0.2], (4, 1))
    out, mode = gate.gate(obs, act, np.array([0.1, 0.4, 0.6, 0.9]))
    assert list(mode) == [MODE_EXPLORE, MODE_PROBE, MODE_RETREAT, MODE_ESCAPE]
    assert np.allclose(out[0], [0.9, 0.2])
    assert out[1, 0] == pytest.approx(0.4) and out[1, 1] == pytest.approx(0.2)
    assert S.action_to_twist(out[2]) == pytest.approx((-0.15, 0.0))
    assert S.action_to_twist(out[3]) == pytest.approx((-0.2, 0.0))
    # Escape is latched until the manoeuvre ends even if rho drops.
    out, mode = gate.gate(obs, act, np.zeros(4))
    assert mode[3] == MODE_ESCAPE and mode[0] == MODE_EXPLORE


def _b0_cell(cell, n=12, seed=7):
    env = E.OrionEnv(n, E.cell_config(cell), seed=seed, auto_reset=False)
    obs, _ = env.reset()
    b0 = B0Controller(n)
    done = np.zeros(n, bool)
    fail = np.zeros(n, bool)
    t_end = np.full(n, 60.0)
    for k in range(env.config.max_steps):
        obs, _, term, _, info = env.step(b0.act(obs))
        new = term & ~done
        fail |= new & (info['belly_out'] | info['hard_stuck'])
        t_end[new & info['success']] = (k + 1) * E.DT
        done |= term
        if done.all():
            break
    eff = np.maximum(info['progress'], 0) / t_end
    return fail.mean(), eff.mean()


def test_b0_premise_degrades_from_light_packed_to_heavy_loose():
    """Phase-0 premise in sim, reported honestly.

    Measured here (seed 7, 12 trials): light-packed 0 % failures, 0.263 m/s; heavy-loose 0 %
    failures, 0.225 m/s (-15 %). With 25 trials per cell (evaluate.py) it is 0.249 vs
    0.206 m/s (-17 %). Slower, but NOT the >= 30 % speed drop or >= 0.25 failure-rate rise
    the kill test asks for. Only the direction of the effect is asserted.
    """
    f_lp, v_lp = _b0_cell('light-packed')
    f_hl, v_hl = _b0_cell('heavy-loose')
    print(f'B0 light-packed fail={f_lp:.2f} v={v_lp:.3f}; heavy-loose fail={f_hl:.2f} '
          f'v={v_hl:.3f} ({100 * (v_hl / v_lp - 1):+.0f} %)')
    assert f_hl >= f_lp
    assert v_hl < 0.9 * v_lp
