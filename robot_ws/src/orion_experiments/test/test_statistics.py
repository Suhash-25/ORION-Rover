"""Pre-registered statistics against hand computations and synthetic 80-trial sessions."""

from math import comb

import numpy as np
from orion_experiments import analysis
from orion_experiments.analysis import calibration
from orion_experiments.analysis import fisher_one_sided
from orion_experiments.analysis import holm
from orion_experiments.analysis import main_study
from orion_experiments.analysis import mannwhitney_less
from orion_experiments.analysis import newcombe_interval
from orion_experiments.analysis import phase0_decision
from orion_experiments.analysis import speed_drop_interval
from orion_experiments.analysis import wilson_interval
import pandas as pd
import pytest


def hand_fisher_greater(fa, na, fb, nb):
    """P(X >= fa), X hypergeometric: the definition of the one-sided Fisher exact test."""
    k_total, n_total = fa + fb, na + nb
    return sum(comb(k_total, k) * comb(n_total - k_total, na - k)
               for k in range(fa, min(k_total, na) + 1)) / comb(n_total, na)


@pytest.mark.parametrize('fa,fb,expected', [
    (8, 2, 0.032416580803677576),
    (6, 1, 0.04573804573804574),
    (4, 0, 0.05301455301455302)])
def test_fisher_matches_hand_computation(fa, fb, expected):
    assert hand_fisher_greater(fa, 20, fb, 20) == pytest.approx(expected, rel=1e-12)
    assert fisher_one_sided(fa, 20, fb, 20, 'greater') == pytest.approx(expected, rel=1e-9)
    # 'less' is the same test with the groups swapped
    assert fisher_one_sided(fb, 20, fa, 20, 'less') == pytest.approx(expected, rel=1e-9)


def test_mannwhitney_matches_hand_computation():
    # U = 0 for complete separation: p = 1 / C(6, 3)
    assert mannwhitney_less([1, 2, 3], [4, 5, 6]) == pytest.approx(1 / 20)
    # one inversion (4 > 3): P(U <= 1) = 2 / C(6, 3)
    assert mannwhitney_less([1, 2, 4], [3, 5, 6]) == pytest.approx(2 / 20)
    # U = 0 with sizes 2 and 3: 1 / C(5, 2)
    assert mannwhitney_less([1, 2], [3, 4, 5]) == pytest.approx(1 / 10)


def test_holm_step_down():
    np.testing.assert_allclose(holm([0.01, 0.04, 0.03, 0.2]), [0.04, 0.09, 0.09, 0.2])
    np.testing.assert_allclose(holm([0.5, 0.6]), [1.0, 1.0])


def session(failures: dict, speeds: dict, controller='B0', n=20, name='phase0'):
    rows = []
    for cell, f in failures.items():
        for i in range(n):
            rows.append({'trial_id': f'{name}_{cell}_{i}', 'cell': cell, 'controller': controller,
                         'valid': True, 'failure': i < f,
                         'effective_speed': speeds[cell][i % len(speeds[cell])]})
    return pd.DataFrame(rows)


BASE_SPEED = np.linspace(0.08, 0.12, 20)
SAME = {c: BASE_SPEED for c in ('light-packed', 'light-loose', 'heavy-packed', 'heavy-loose')}


STAGE1 = {'light-packed': 2, 'light-loose': 8, 'heavy-packed': 3, 'heavy-loose': 5}
LIGHT = {c: SAME[c] for c in ('light-packed', 'light-loose')}


def test_stage1_never_says_no_go():
    # 8/20 vs 2/20: p = 0.032 passes the old single-look 0.05 but not the stage-1 0.01.
    res = phase0_decision(session(STAGE1, SAME))
    assert res['stage'] == 1 and res['alpha'] == 0.01
    assert res['decision'] == 'CONTINUE' and not res['go']
    row = res['table'].set_index('cell').loc['light-loose']
    assert row.fisher_p == pytest.approx(hand_fisher_greater(8, 20, 2, 20))
    assert row.rate_rise == pytest.approx(0.3) and not row.failure_criterion


def test_stage1_early_go():
    res = phase0_decision(session(dict(STAGE1, **{'light-loose': 10, 'light-packed': 1}),
                                  SAME))
    row = res['table'].set_index('cell').loc['light-loose']
    assert row.fisher_p == pytest.approx(hand_fisher_greater(10, 20, 1, 20))
    assert row.fisher_p < 0.01
    assert res['decision'] == 'GO' and res['go']


def test_stage2_pools_both_stages_and_tests_light_loose_only():
    df = pd.concat([session(STAGE1, SAME),
                    session({'light-packed': 2, 'light-loose': 8}, LIGHT, name='phase0b')])
    res = phase0_decision(df)
    assert res['stage'] == 2 and res['alpha'] == 0.04
    t = res['table']
    assert list(t.cell) == ['light-loose'] and t.n.iloc[0] == 40 and t.ref_n.iloc[0] == 40
    assert t.fisher_p.iloc[0] == pytest.approx(hand_fisher_greater(16, 40, 4, 40))
    assert res['decision'] == 'GO'


def test_stage2_no_go():
    df = pd.concat([session(STAGE1, SAME),
                    session({'light-packed': 4, 'light-loose': 4}, LIGHT, name='phase0b')])
    res = phase0_decision(df)  # 12/40 vs 6/40: p = 0.09; speeds identical
    assert res['decision'] == 'NO-GO' and not res['go'] and res['go_cells'] == []
    assert phase0_decision(df, effect_size_gate=True)['decision'] == 'NO-GO'


def test_effect_size_gate_is_off_by_default_and_switchable():
    assert analysis.EFFECT_SIZE_GATE is False
    # Stage 2, 12/40 vs 3/40: p = 0.0099 < 0.04, but the rise 0.225 is below 0.25.
    rate = pd.concat([session(STAGE1, SAME),
                      session({'light-packed': 1, 'light-loose': 4}, LIGHT, name='phase0b')])
    res = phase0_decision(rate)
    row = res['table'].iloc[0]
    assert row.fisher_p == pytest.approx(hand_fisher_greater(12, 40, 3, 40))
    assert row.rate_rise == pytest.approx(0.225)
    assert res['decision'] == 'GO' and not res['effect_size_gate'] and row.failure_criterion
    gated = phase0_decision(rate, effect_size_gate=True)
    assert gated['decision'] == 'NO-GO' and not gated['table'].iloc[0].failure_criterion
    # A 10 % slower loose cell with the same failures: significant (p < 0.001) but < 30 %.
    speeds = dict(SAME, **{'light-loose': BASE_SPEED * 0.9})
    slow = pd.concat([session(STAGE1, speeds),
                      session({'light-packed': 4, 'light-loose': 4},
                              {c: speeds[c] for c in LIGHT}, name='phase0b')])
    row = phase0_decision(slow)['table'].iloc[0]
    assert row.speed_drop == pytest.approx(0.1) and row.mannwhitney_p < 0.001
    assert row.speed_criterion and not row.failure_criterion
    assert phase0_decision(slow)['go'] and not phase0_decision(slow, effect_size_gate=True)['go']


def test_effect_sizes_are_reported_with_intervals():
    res = phase0_decision(session(STAGE1, SAME))
    row = res['table'].set_index('cell').loc['light-loose']   # 8/20 vs 2/20
    assert (row.rate_rise_ci_low, row.rate_rise_ci_high) == pytest.approx(
        newcombe_interval(8, 20, 2, 20))
    assert row.rate_rise_ci_low < row.rate_rise < row.rate_rise_ci_high
    assert row.speed_drop == pytest.approx(0.0)
    assert row.speed_drop_ci_low <= 0.0 <= row.speed_drop_ci_high
    # Reproducible: the bootstrap seed is fixed.
    again = phase0_decision(session(STAGE1, SAME))['table'].set_index('cell').loc['light-loose']
    assert again.speed_drop_ci_low == row.speed_drop_ci_low


def test_wilson_and_newcombe_match_published_values():
    # Newcombe (1998), Stat. Med. 17: Wilson 81/263 and the hybrid interval of 56/70 - 48/80.
    assert wilson_interval(81, 263) == pytest.approx((0.2553, 0.3662), abs=1e-4)
    assert newcombe_interval(56, 70, 48, 80) == pytest.approx((0.0524, 0.3339), abs=1e-4)
    assert wilson_interval(0, 20)[0] == 0.0 and wilson_interval(20, 20)[1] == 1.0


def test_speed_drop_interval_covers_a_known_drop():
    rng = np.random.default_rng(1)
    y = rng.uniform(0.08, 0.12, 40)
    low, high = speed_drop_interval(y * 0.6, y)
    assert low < 0.4 < high and high - low < 0.2


def test_two_factor_go_and_safety_outcomes_are_flagged():
    speeds = dict(SAME, **{'heavy-loose': BASE_SPEED * 0.6})
    df = session(dict.fromkeys(SAME, 3), speeds)
    df['outcome'] = np.where(df.failure, 'TIMEOUT', 'SUCCESS')
    df.loc[(df.cell == 'heavy-loose') & df.failure, 'outcome'] = 'SAFETY'
    res = phase0_decision(df)
    assert res['go'] and res['go_cells'] == ['heavy-loose'] and res['two_factor_only']
    t = res['table'].set_index('cell')
    assert t.loc['heavy-loose'].factors_changed == 2 and t.loc['light-loose'].factors_changed == 1
    assert t.loc['heavy-loose'].safety_failures == 3 and t.loc['light-loose'].safety_failures == 0
    assert t.loc['heavy-loose'].ref_safety_failures == 0


def test_stage_override_and_bad_stage():
    forced = phase0_decision(session(STAGE1, SAME), stage=2)  # p = 0.032 < 0.04
    assert forced['alpha'] == 0.04 and forced['decision'] == 'GO'
    with pytest.raises(ValueError):
        phase0_decision(session(STAGE1, SAME), stage=3)


def test_go_on_speed_only_and_failure_only_mode():
    speeds = dict(SAME, **{'heavy-loose': BASE_SPEED * 0.6})
    df = session(dict.fromkeys(SAME, 3), speeds)
    res = phase0_decision(df)
    t = res['table'].set_index('cell')
    assert res['go'] and t.loc['heavy-loose'].speed_criterion
    assert t.loc['heavy-loose'].speed_drop == pytest.approx(0.4)
    assert not phase0_decision(df, use_speed=False)['go']


def test_rise_without_significance_is_not_go():
    # rise 0.25 but p >= 0.05 at n = 20 (12/20 vs 7/20)
    p = hand_fisher_greater(12, 20, 7, 20)
    assert p > 0.05
    df = session({'light-packed': 7, 'light-loose': 12}, SAME)
    res = phase0_decision(df)
    assert not res['go']
    assert res['table'].iloc[0].fisher_p == pytest.approx(p)


def test_invalid_trials_and_other_controllers_are_ignored():
    df = session({'light-packed': 2, 'light-loose': 8}, SAME)
    df.loc[df.cell == 'light-loose', 'valid'] = False
    other = session({'light-packed': 0, 'light-loose': 20}, SAME, controller='B2')
    with pytest.raises(ValueError):
        phase0_decision(df[df.cell == 'light-loose'])
    assert not phase0_decision(pd.concat([df, other]))['go']


def test_main_study_primary_pooled_and_holm_secondary_family():
    frames = []
    for ctrl, f in {'B0': 10, 'B2': 8, 'student': 2, 'student_critic': 1}.items():
        frames.append(session({'light-loose': f, 'heavy-loose': f}, SAME, controller=ctrl,
                              n=25))
    frames.append(session({'light-packed': 25}, SAME, controller='B0', n=25))  # not pooled
    table = main_study(pd.concat(frames))
    primary = table[table.role == 'primary']
    secondary = table[table.role == 'secondary']
    assert len(primary) == 1 and len(secondary) == 8   # 2 loose cells x 4 comparisons
    p = primary.iloc[0]
    assert (p.a, p.b, p.n_a, p.n_b) == ('student_critic', 'B0', 50, 50)
    assert p.fisher_p == pytest.approx(hand_fisher_greater(20, 50, 2, 50))
    assert p.holm_p == p.fisher_p and p.reject_h0               # unadjusted, alpha 0.05
    assert p.rate_reduction == pytest.approx(0.36)
    assert p.rate_reduction_ci_low < 0.36 < p.rate_reduction_ci_high
    row = secondary[(secondary.cell == 'light-loose') & (secondary.a == 'student')
                    & (secondary.b == 'B2')]
    raw = hand_fisher_greater(8, 25, 2, 25)  # B2 fails more than student
    assert row.fisher_p.iloc[0] == pytest.approx(raw)
    np.testing.assert_allclose(secondary.holm_p, holm(secondary.fisher_p))
    assert (secondary.holm_p >= secondary.fisher_p).all()
    # student 2/25 vs B0 10/25: significant alone (p = 0.009), not after Holm over 8 tests
    vs_b0 = secondary.set_index(['cell', 'a', 'b']).loc[('heavy-loose', 'student', 'B0')]
    assert vs_b0.fisher_p < 0.05 and not vs_b0.reject_h0
    frames[0] = session({'light-loose': 20, 'heavy-loose': 20}, SAME, controller='B0', n=25)
    strong = main_study(pd.concat(frames)).set_index(['cell', 'a', 'b'])
    assert strong.loc[('heavy-loose', 'student', 'B0')].reject_h0
    assert not strong.loc[('heavy-loose', 'student_critic', 'student')].reject_h0


def test_main_study_primary_needs_both_controllers_in_loose_cells():
    df = session({'light-loose': 5, 'light-packed': 1}, SAME, controller='student', n=25)
    df = pd.concat([df, session({'light-packed': 3}, SAME, controller='B0', n=25)])
    table = main_study(df)
    assert 'primary' not in set(table.role)
    assert list(table.cell) == ['light-packed']


def test_calibration_metrics():
    perfect = calibration([0.0, 0.0, 1.0, 1.0], [0, 0, 1, 1])
    assert perfect['ece'] == 0.0 and perfect['brier'] == 0.0
    cal = calibration([0.25] * 4 + [0.75] * 4, [0, 0, 0, 1, 1, 1, 1, 0])
    assert cal['ece'] == pytest.approx(0.0)
    assert cal['brier'] == pytest.approx(np.mean([0.25 ** 2] * 6 + [0.75 ** 2] * 2))
    over = calibration([0.9] * 10, [0] * 10)
    assert over['ece'] == pytest.approx(0.9)
    assert calibration([], [])['n'] == 0
