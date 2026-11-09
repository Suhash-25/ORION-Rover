"""Pre-registered statistics against hand computations and synthetic 80-trial sessions."""

from math import comb

import numpy as np
from orion_experiments.analysis import calibration
from orion_experiments.analysis import fisher_one_sided
from orion_experiments.analysis import holm
from orion_experiments.analysis import main_study
from orion_experiments.analysis import mannwhitney_less
from orion_experiments.analysis import phase0_decision
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
    speeds = dict(SAME, **{'light-loose': BASE_SPEED * 0.9, 'heavy-loose': BASE_SPEED * 0.9})
    df = pd.concat([session(STAGE1, speeds),
                    session({'light-packed': 4, 'light-loose': 4},
                            {c: speeds[c] for c in LIGHT}, name='phase0b')])
    res = phase0_decision(df)  # 12/40 vs 6/40: p = 0.09
    assert res['decision'] == 'NO-GO' and not res['go']
    row = res['table'].iloc[0]
    assert row.speed_drop == pytest.approx(0.1) and not row.speed_criterion


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


def test_main_study_holm_family():
    frames = []
    for ctrl, f in {'B0': 10, 'B2': 8, 'student': 2, 'student_critic': 1}.items():
        frames.append(session({'light-loose': f, 'heavy-loose': f}, SAME, controller=ctrl,
                              n=25))
    table = main_study(pd.concat(frames))
    assert len(table) == 8   # 2 cells x 4 comparisons
    row = table[(table.cell == 'light-loose') & (table.a == 'student') & (table.b == 'B2')]
    raw = hand_fisher_greater(8, 25, 2, 25)  # B2 fails more than student
    assert row.fisher_p.iloc[0] == pytest.approx(raw)
    np.testing.assert_allclose(table.holm_p, holm(table.fisher_p))
    assert (table.holm_p >= table.fisher_p).all()
    # student 2/25 vs B0 10/25: significant alone (p = 0.009), not after Holm over 8 tests
    vs_b0 = table.set_index(['cell', 'a', 'b']).loc[('heavy-loose', 'student', 'B0')]
    assert vs_b0.fisher_p < 0.05 and not vs_b0.reject_h0
    frames[0] = session({'light-loose': 20, 'heavy-loose': 20}, SAME, controller='B0', n=25)
    strong = main_study(pd.concat(frames)).set_index(['cell', 'a', 'b'])
    assert strong.loc[('heavy-loose', 'student', 'B0')].reject_h0
    assert not strong.loc[('heavy-loose', 'student_critic', 'student')].reject_h0


def test_calibration_metrics():
    perfect = calibration([0.0, 0.0, 1.0, 1.0], [0, 0, 1, 1])
    assert perfect['ece'] == 0.0 and perfect['brier'] == 0.0
    cal = calibration([0.25] * 4 + [0.75] * 4, [0, 0, 0, 1, 1, 1, 1, 0])
    assert cal['ece'] == pytest.approx(0.0)
    assert cal['brier'] == pytest.approx(np.mean([0.25 ** 2] * 6 + [0.75 ** 2] * 2))
    over = calibration([0.9] * 10, [0] * 10)
    assert over['ece'] == pytest.approx(0.9)
    assert calibration([], [])['n'] == 0
