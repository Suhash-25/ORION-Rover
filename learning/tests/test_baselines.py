"""B0, stuck detector, escape library, critic gating, and the B0 premise check in sim."""

import numpy as np
import pytest

from orion_learning import env as E
from orion_learning import escape_library as EL
from orion_learning import obs_spec as S
from orion_learning.baselines import (B0Controller, CriticGate, EscapeLibrary, MODE_ESCAPE,
                                      MODE_EXPLORE, MODE_PROBE, MODE_RETREAT, StuckDetector,
                                      StuckRule)

TICKS0 = EL.manoeuvre_ticks(EL.E[0], E.DT)


def _obs(n=1, cmd_v=0.3, speed=0.3, conf=1.0, slip=(0.0, 0.0)):
    o = np.zeros((n, S.OBS_DIM), np.float32)
    o[:, S.field_slice('prev_cmd')] = [cmd_v / 0.5, 0.0]
    o[:, S.field_slice('body_speed')] = speed / 0.5
    o[:, S.field_slice('speed_confidence')] = conf
    o[:, S.field_slice('slip_side')] = slip
    return o


def _v(action):
    return S.action_to_twist(action[0])[0]


def test_b0_cruises_at_0_3_mps():
    a = B0Controller(1).act(_obs())
    assert S.action_to_twist(a[0]) == pytest.approx((0.3, 0.0))


def test_slip_governor_cuts_20_percent_per_s_and_recovers_5():
    b0 = B0Controller(1)
    for _ in range(20):                                 # 1 s of slip 0.5
        a = b0.act(_obs(slip=(0.5, 0.1)))
    assert _v(a) == pytest.approx(0.3 * 0.8, abs=1e-6)
    for _ in range(20):                                 # 1 s without slip
        a = b0.act(_obs())
    assert _v(a) == pytest.approx(0.3 * 0.85, abs=1e-6)
    for _ in range(200):
        a = b0.act(_obs(slip=(0.0, 0.6), speed=0.12))
    assert _v(a) == pytest.approx(0.3 * 0.4)            # floor keeps v > 0.1


def test_stuck_rule_holds_3_s_like_the_safety_monitor():
    """First true sample = 0 s (orion_safety _Held), so the flag comes on sample 61 at 20 Hz."""
    det = StuckDetector(2)
    obs = np.concatenate([_obs(speed=0.0), _obs(speed=0.0, conf=0.1)])
    flags = [det.update(obs) for _ in range(61)]
    assert not flags[59][0] and flags[60][0]
    assert not any(f[1] for f in flags)                # low confidence: no speed-based stuck
    det = StuckDetector(1)
    flags = [det.update(_obs(conf=0.0, slip=(0.9, 0.2)))[0] for _ in range(61)]
    assert flags[-1] and not flags[-2]                 # slip rule ignores confidence
    flags = [det.update(_obs(speed=0.0))[0] for _ in range(2)]
    assert not any(flags), 'a change of condition restarts the hold'


def test_stuck_rule_fresh_fix_replaces_confidence():
    det = StuckDetector(2)
    obs = np.concatenate([_obs(speed=0.0, conf=0.1)] * 2)
    for _ in range(61):
        flags = det.update(obs, time_since_fix=np.array([0.2, 0.6]))
    assert list(flags) == [True, False]


def test_stuck_rule_follows_the_driven_direction():
    det = StuckDetector(3)
    obs = np.concatenate([_obs(cmd_v=-0.2, speed=0.0),              # reversing, not moving
                          _obs(cmd_v=-0.2, slip=(-0.9, -0.9)),      # reversing, spinning
                          _obs(cmd_v=-0.2, slip=(0.9, 0.9))])       # skidding, not spinning
    for _ in range(61):
        flags = det.update(obs)
    assert list(flags) == [True, True, False]
    rule = StuckRule()
    assert rule.cmd_v_min == 0.1 and rule.hold_s == 3.0 and rule.fresh_fix_s == 0.5


def test_escape_library_plays_E_tick_by_tick_then_moves_on():
    lib = EscapeLibrary(1)
    assert lib.library is EL.E
    lib.start(np.array([True]))
    twists = []
    while lib.active[0]:
        v, w = lib.act()
        twists.append((v[0], w[0]))
    assert tuple(twists) == TICKS0
    assert len(twists) == round(EL.E[0].duration_s / E.DT)
    assert lib.move[0] == 1
    lib.start(np.array([True]))
    assert lib.act()[0][0] == EL.E[1].segments[0][1]
    lib.reset()
    assert lib.move[0] == 0 and not lib.active[0]


def test_b0_escapes_when_stuck_resumes_at_half_speed():
    b0 = B0Controller(1)
    stuck = _obs(speed=0.0)
    actions = [b0.act(stuck) for _ in range(61)]
    assert _v(actions[59]) == pytest.approx(0.3)
    assert S.action_to_twist(actions[60][0]) == pytest.approx(TICKS0[0])    # escape at 3 s
    rest = [b0.act(_obs()) for _ in range(len(TICKS0) - 1 + 2)]
    played = [tuple(S.action_to_twist(a[0])) for a in [actions[60]] + rest[:len(TICKS0) - 1]]
    assert np.allclose(played, TICKS0)
    # Two ticks after the escape the governor has recovered 5 %/s twice from 0.5.
    assert _v(rest[-1]) == pytest.approx(0.3 * 0.505, abs=1e-6)


def test_b0_gives_up_after_three_escapes_without_a_clean_tick():
    """As orion_policy: still stuck after each escape -> next manoeuvre; the 4th trigger stops."""
    b0 = B0Controller(1)
    stuck = _obs(speed=0.0)
    seen = []
    for _ in range(61 + sum(round(m.duration_s / E.DT) for m in EL.E[:3]) + 5):
        seen.append(tuple(S.action_to_twist(b0.act(stuck)[0])))
    assert b0.arbiter.attempts[0] == 3 and b0.arbiter.gave_up[0]
    assert seen[-1] == (0.0, 0.0)
    assert b0.escape.move[0] == 3
    b0.reset()
    assert _v(b0.act(_obs())) == pytest.approx(0.3)


def test_b0_external_stuck_flag_overrides_the_detector():
    b0 = B0Controller(1)
    a = b0.act(_obs(), stuck=np.array([True]))
    assert S.action_to_twist(a[0]) == pytest.approx(TICKS0[0])


def test_critic_gate_thresholds():
    gate = CriticGate(4)
    obs = _obs(4)
    act = np.tile([0.9, 0.2], (4, 1))
    out, mode = gate.gate(obs, act, np.array([0.1, 0.4, 0.6, 0.9]))
    assert list(mode) == [MODE_EXPLORE, MODE_PROBE, MODE_RETREAT, MODE_ESCAPE]
    assert np.allclose(out[0], [0.9, 0.2])
    assert out[1, 0] == pytest.approx(0.4) and out[1, 1] == pytest.approx(0.2)
    assert S.action_to_twist(out[2]) == pytest.approx((-0.15, 0.0))
    assert S.action_to_twist(out[3]) == pytest.approx(TICKS0[0])
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
