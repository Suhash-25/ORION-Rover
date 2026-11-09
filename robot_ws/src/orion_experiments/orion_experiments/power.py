"""
Power and false-GO rate of the implemented Phase-0 kill-test rule, by simulation (council M3).

    ros2 run orion_experiments phase0_power [--sims 20000] [--seed 20261109]
    python3 -m orion_experiments.power

simulate_phase0() draws whole two-stage B0 sessions and applies the rules of
analysis.phase0_decision (RULES: ungated / gated, with and without the speed test), vectorised
over sessions:
- stage 1, 20 trials per cell: light-loose and heavy-loose each against light-packed, one-sided
  Fisher on failures and one-sided Mann-Whitney on effective speed, alpha 0.01; GO if any of
  the four tests passes (gated rule: and its effect size reaches 0.25 / 30 %);
- otherwise stage 2 adds 20 trials to light-packed and light-loose and tests the pooled 40 vs 40
  once more at alpha 0.04 (failure-rate test, speed test).
test_power.py checks this vectorised rule against phase0_decision session by session.
The 0.01 + 0.04 split is a Bonferroni bound over the two looks for ONE test per look; the rule
runs four tests at stage 1 and two at stage 2, so its false-GO rate is whatever the H0 rows say.

Trial model (assumptions, quoted in SPEC section 8):
- failure ~ Bernoulli(p_cell), independent trials;
- effective speed of a failure = U(0, goal) / duration (it stopped short of the goal line);
  of a success = goal / time-to-goal, log-normal around 0.15 m/s (sigma 0.25), at least
  goal / duration, times `slowdown` in the loose cells.
Every failure is slower than every success, and Mann-Whitney only uses ranks, so with
slowdown = 1 the speed test sees only the failure counts and the log-normal guess does not
matter. A real slowdown of successful runs on loose soil would add power to the speed test.

Also printed: the exact power of a plain two-sided Fisher test at 0.05 (the 0.49 / 0.71 / 0.85
quoted in the paper; a reference, not the implemented rule) and the main-study power of one
per-cell comparison at n = 25 (Holm's first step alpha / 16, and unadjusted) against the
pooled primary comparison (50 vs 50), council m7.
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

from .analysis import fisher_one_sided
from .analysis import MIN_RATE_RISE
from .analysis import MIN_SPEED_DROP
from .analysis import REFERENCE_CELL
from .analysis import STAGE2_CELL
from .analysis import STAGE2_SESSION
from .analysis import STAGE_ALPHA

GOAL_M = 1.5
DURATION_S = 60.0
SUCCESS_SPEED_MEDIAN = 0.15     # m/s, GUESS (irrelevant to the result when slowdown = 1)
SUCCESS_SPEED_SIGMA = 0.25      # log-normal sigma
STAGE1_CELLS = (REFERENCE_CELL, 'light-loose', 'heavy-loose')
SEED = 20261109


def draw_sessions(rates: dict, sims: int, seed: int = SEED, n1: int = 20, n2: int = 20,
                  slowdown: float = 1.0) -> dict:
    """
    Simulate sessions: {cell: (failure (sims, n) bool, effective speed (sims, n) m/s)}.

    rates gives the failure probability of light-packed, light-loose and heavy-loose. The
    stage-2 cells (light-packed, light-loose) get n1 + n2 trials, the first n1 being stage 1.
    """
    rng = np.random.default_rng(seed)
    out = {}
    for cell in STAGE1_CELLS:
        n = n1 + n2 if cell in (REFERENCE_CELL, STAGE2_CELL) else n1
        fail = rng.random((sims, n)) < rates[cell]
        success_speed = SUCCESS_SPEED_MEDIAN * np.exp(SUCCESS_SPEED_SIGMA
                                                      * rng.standard_normal((sims, n)))
        success_speed = np.maximum(success_speed, GOAL_M / DURATION_S)
        if cell != REFERENCE_CELL:
            success_speed = success_speed * slowdown
        fail_speed = rng.uniform(0.0, GOAL_M, (sims, n)) / DURATION_S
        out[cell] = (fail, np.where(fail, fail_speed, success_speed))
    return out


def _fisher_table(n_a: int, n_b: int) -> np.ndarray:
    """table[fa, fb] = one-sided Fisher p (failure rate a > b) from analysis.fisher_one_sided."""
    table = np.empty((n_a + 1, n_b + 1))
    for fa in range(n_a + 1):
        for fb in range(n_b + 1):
            table[fa, fb] = fisher_one_sided(fa, n_a, fb, n_b, 'greater')
    return table


def _comparison(fail_x, speed_x, fail_y, speed_y, fisher) -> dict:
    """Per-session statistics of one loose cell x against the reference y (phase0_decision)."""
    na, nb = fail_x.shape[1], fail_y.shape[1]
    fa, fb = fail_x.sum(1), fail_y.sum(1)
    med_x, med_y = np.median(speed_x, 1), np.median(speed_y, 1)
    return {'rise': fa / na - fb / nb, 'fisher_p': fisher[fa, fb],
            'speed_ok': med_y > 0, 'drop': 1.0 - med_x / med_y,
            'mw_p': stats.mannwhitneyu(speed_x, speed_y, alternative='less', method='auto',
                                       axis=1).pvalue}


def _go(c: dict, alpha: float, effect_size_gate: bool, use_speed: bool) -> np.ndarray:
    """Vectorised phase0_decision criterion (failure_criterion | speed_criterion)."""
    big_rise = (c['rise'] >= MIN_RATE_RISE) | (not effect_size_gate)
    go = (c['fisher_p'] < alpha) & big_rise
    if use_speed:
        big_drop = (c['drop'] >= MIN_SPEED_DROP) | (not effect_size_gate)
        go |= c['speed_ok'] & (c['mw_p'] < alpha) & big_drop
    return go


def comparisons(sessions: dict, n1: int = 20) -> tuple[list, dict]:
    """([stage-1 comparison per loose cell], stage-2 comparison) statistics per session."""
    ref_fail, ref_speed = sessions[REFERENCE_CELL]
    fisher1 = _fisher_table(n1, n1)
    stage1 = [_comparison(sessions[cell][0][:, :n1], sessions[cell][1][:, :n1],
                          ref_fail[:, :n1], ref_speed[:, :n1], fisher1)
              for cell in ('light-loose', 'heavy-loose')]
    fail, speed = sessions[STAGE2_CELL]
    stage2 = _comparison(fail, speed, ref_fail, ref_speed,
                         _fisher_table(fail.shape[1], ref_fail.shape[1]))
    return stage1, stage2


def decide(stats1: list, stats2: dict, effect_size_gate: bool, use_speed: bool = True,
           alphas: tuple = (STAGE_ALPHA[1], STAGE_ALPHA[2])) -> tuple[np.ndarray, np.ndarray]:
    """(GO at stage 1, GO after both stages) per session, from comparisons()."""
    early = np.logical_or.reduce([_go(c, alphas[0], effect_size_gate, use_speed)
                                  for c in stats1])
    return early, early | _go(stats2, alphas[1], effect_size_gate, use_speed)


def session_frames(sessions: dict, i: int, n1: int = 20) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(stage-1 trials, all trials) of simulated session i as analysis.phase0_decision input."""
    rows = []
    for cell, (fail, speed) in sessions.items():
        for k in range(fail.shape[1]):
            session = 'phase0' if k < n1 else STAGE2_SESSION
            rows.append({'trial_id': f'{session}_{cell}_{k}', 'cell': cell, 'controller': 'B0',
                         'valid': True, 'failure': bool(fail[i, k]),
                         'effective_speed': float(speed[i, k])})
    trials = pd.DataFrame(rows)
    stage1 = trials[~trials.trial_id.str.startswith(STAGE2_SESSION + '_')]
    return stage1, trials


# (label, effect_size_gate, use_speed, alpha factor per test). The first is the default rule.
RULES = (
    ('ungated, failures only (default; speed reported, not decisive)', False, False, 1.0),
    ('ungated, failures + speed (--speed-in-decision)', False, True, 1.0),
    ('ungated, failures + speed, each test at alpha / 2', False, True, 0.5),
    ('gated, failures + speed (--effect-size-gate)', True, True, 1.0),
    ('gated, failures only', True, False, 1.0),
)


def simulate_phase0(rates: dict, sims: int = 20000, seed: int = SEED,
                    slowdown: float = 1.0) -> dict:
    """{rule label: (P(early GO at stage 1), P(GO overall))} for one scenario (RULES)."""
    stats1, stats2 = comparisons(draw_sessions(rates, sims, seed, slowdown=slowdown))
    out = {}
    for label, gate, use_speed, factor in RULES:
        alphas = (STAGE_ALPHA[1] * factor, STAGE_ALPHA[2] * factor)
        early, final = decide(stats1, stats2, gate, use_speed, alphas)
        out[label] = (float(early.mean()), float(final.mean()))
    return out


def fisher_power(p_a: float, p_b: float, n_a: int, n_b: int, alpha: float,
                 alternative: str = 'greater') -> float:
    """Exact power of Fisher's test (a vs b, rates p_a and p_b) by summing over all tables."""
    pmf_a = stats.binom.pmf(np.arange(n_a + 1), n_a, p_a)
    pmf_b = stats.binom.pmf(np.arange(n_b + 1), n_b, p_b)
    power = 0.0
    for fa in range(n_a + 1):
        for fb in range(n_b + 1):
            table = [[fa, n_a - fa], [fb, n_b - fb]]
            if stats.fisher_exact(table, alternative=alternative).pvalue < alpha:
                power += pmf_a[fa] * pmf_b[fb]
    return float(power)


SCENARIOS = (
    ('H1: C 10 %, D 40 %, B 40 %', {'light-packed': 0.10, 'light-loose': 0.40,
                                    'heavy-loose': 0.40}),
    ('H1: C 10 %, D 40 %, B 10 %', {'light-packed': 0.10, 'light-loose': 0.40,
                                    'heavy-loose': 0.10}),
    ('H0: every cell 10 %', dict.fromkeys(STAGE1_CELLS, 0.10)),
    ('H0: every cell 30 %', dict.fromkeys(STAGE1_CELLS, 0.30)),
)


def report(sims: int = 20000, seed: int = SEED, slowdown: float = 1.0) -> str:
    """Markdown tables of every number quoted in SPEC section 8 and RESEARCH.md."""
    se = 0.5 / np.sqrt(sims)
    lines = [f'## Phase-0 rule, {sims} simulated sessions per scenario (seed {seed}, '
             f'Monte-Carlo SE <= {se:.3f}; success slowdown in loose cells x{slowdown:g})', '',
             'C light-packed, D light-loose, B heavy-loose. H1 rows are power, H0 rows the '
             'false-GO rate.', '',
             '| scenario | rule | GO at stage 1 | GO overall |', '|---|---|---|---|']
    for name, rates in SCENARIOS:
        for label, (early, final) in simulate_phase0(rates, sims, seed, slowdown).items():
            lines.append(f'| {name} | {label} | {early:.3f} | {final:.3f} |')
    lines += ['', '## Reference: plain two-sided Fisher at alpha 0.05, 10 % vs 40 % (exact)', '',
              '| runs per cell | power |', '|---|---|']
    for n in (20, 30, 40):
        lines.append(f'| {n} | {fisher_power(0.40, 0.10, n, n, 0.05, "two-sided"):.3f} |')
    lines += ['', '## Main study (m7): one-sided Fisher, controller fails less than B0 (exact)',
              '', '| B0 rate | controller rate | per cell n = 25, alpha 0.05/16 (Holm first step) '
              '| per cell n = 25, alpha 0.05 | pooled loose cells n = 50, alpha 0.05 |',
              '|---|---|---|---|---|']
    for p_b0, p_c in ((0.40, 0.20), (0.40, 0.10), (0.30, 0.10)):
        lines.append(f'| {p_b0:.2f} | {p_c:.2f} | '
                     f'{fisher_power(p_b0, p_c, 25, 25, 0.05 / 16):.3f} | '
                     f'{fisher_power(p_b0, p_c, 25, 25, 0.05):.3f} | '
                     f'{fisher_power(p_b0, p_c, 50, 50, 0.05):.3f} |')
    return '\n'.join(lines) + '\n'


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description='Power of the Phase-0 kill-test rule.')
    parser.add_argument('--sims', type=int, default=20000, help='sessions per scenario')
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--slowdown', type=float, default=1.0,
                        help='speed factor of successful runs in the loose cells')
    args = parser.parse_args(argv)
    start = time.monotonic()
    print(report(args.sims, args.seed, args.slowdown))
    print(f'({time.monotonic() - start:.1f} s)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
