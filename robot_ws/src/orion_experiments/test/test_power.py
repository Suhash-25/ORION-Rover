"""The simulated Phase-0 rule is phase0_decision, and the quoted power numbers reproduce."""

import time

import numpy as np
from orion_experiments.analysis import phase0_decision
from orion_experiments.analysis import STAGE_ALPHA
from orion_experiments.power import comparisons
from orion_experiments.power import decide
from orion_experiments.power import draw_sessions
from orion_experiments.power import fisher_power
from orion_experiments.power import main
from orion_experiments.power import RULES
from orion_experiments.power import SCENARIOS
from orion_experiments.power import session_frames
from orion_experiments.power import simulate_phase0
import pytest


def test_vectorised_rule_equals_phase0_decision_session_by_session():
    # Rates near the thresholds and a 15 % slowdown so every criterion fires sometimes.
    rates = {'light-packed': 0.15, 'light-loose': 0.40, 'heavy-loose': 0.30}
    sessions = draw_sessions(rates, sims=40, seed=7, slowdown=0.85)
    stats1, stats2 = comparisons(sessions)
    seen = set()
    for label, gate, use_speed, factor in RULES:
        alphas = (STAGE_ALPHA[1] * factor, STAGE_ALPHA[2] * factor)
        early, final = decide(stats1, stats2, gate, use_speed, alphas)
        for i in range(40):
            stage1, everything = session_frames(sessions, i)
            ref1 = phase0_decision(stage1, effect_size_gate=gate, use_speed=use_speed,
                                   alpha=alphas[0], speed_in_decision=use_speed)
            assert ref1['stage'] == 1 and early[i] == ref1['go'], (label, i)
            ref = ref1['go'] or phase0_decision(everything, effect_size_gate=gate,
                                                use_speed=use_speed, alpha=alphas[1],
                                                speed_in_decision=use_speed)['go']
            assert final[i] == ref, (label, i)
            seen.add((bool(early[i]), bool(final[i])))
    assert seen == {(False, False), (False, True), (True, True)}   # every path exercised


def test_plain_fisher_reference_matches_the_paper():
    # Paper, Sample Size: two-sided Fisher at 0.05, 10 % vs 40 %: 0.49 / 0.71 / 0.85.
    for n, quoted in ((20, 0.49), (30, 0.71), (40, 0.85)):
        power = fisher_power(0.40, 0.10, n, n, 0.05, 'two-sided')
        assert power == pytest.approx(quoted, abs=0.005)
    assert fisher_power(0.2, 0.2, 20, 20, 0.05) <= 0.05          # the exact test keeps its size


def test_quoted_numbers_reproduce_with_the_fixed_seed():
    # SPEC section 8 and RESEARCH.md Sample Size quote these (20000 sessions, default seed).
    h1 = simulate_phase0(dict(SCENARIOS[0][1]))
    assert h1[RULES[0][0]] == pytest.approx((0.510, 0.910), abs=5e-4)   # default rule
    assert h1[RULES[1][0]] == pytest.approx((0.577, 0.931), abs=5e-4)
    assert h1[RULES[3][0]] == pytest.approx((0.548, 0.814), abs=5e-4)
    h0 = simulate_phase0(dict(SCENARIOS[3][1]))
    assert h0[RULES[0][0]][1] == pytest.approx(0.029, abs=5e-4)
    assert h0[RULES[1][0]][1] == pytest.approx(0.063, abs=5e-4)


def test_cli_runs_within_the_time_budget(capsys):
    start = time.monotonic()
    assert main([]) == 0
    assert time.monotonic() - start < 20.0
    out = capsys.readouterr().out
    assert out.count('| H0: every cell') == 2 * len(RULES) and 'pooled loose cells' in out


def test_speed_model_puts_every_failure_below_every_success():
    fail, speed = draw_sessions(dict(SCENARIOS[0][1]), sims=200, seed=1)['light-loose']
    assert speed[fail].max() < speed[~fail].min()
    assert np.all(speed > 0)
