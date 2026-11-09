"""Belly-out time distribution produced by the dig-rate prior LogU(0.03, 0.3) (council W2).

The dig rate (m of slip sinkage per m of slipped rim travel) is a model-form guess: with
instant slip sinkage (the SPEC model) 84 % of randomised rovers belly out within 1 s of a
full-throttle start. This reports, for rovers drawn from the SPEC evaluation distribution
(the harshest one) under two drivers, the time to belly-out per dig-rate bin and over the
whole prior:

* full throttle: action (1, 0) from rest, no governor, the worst case for the start-up
  slip transient;
* B0: cruise 0.3 m/s with its slip governor and escapes.

    python scripts/dig_rate_belly_out.py --episodes 1000 --out runs/dig_rate

Writes dig_rate_belly_out.csv (one row per episode) and dig_rate_belly_out.md.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from orion_learning import env as E
from orion_learning.baselines import B0Controller
from orion_learning.evaluate import write_csv

BINS = (0.03, 0.05, 0.08, 0.13, 0.2, 0.3)       # dig-rate bin edges (log-spaced)
WITHIN_S = (1.0, 3.0, 10.0)


def belly_out_times(driver: str, episodes: int, seed: int, cfg: E.EnvConfig | None = None):
    """Return (dig rate, belly-out time in s or inf) per episode."""
    env = E.OrionEnv(episodes, cfg or E.spec_eval_config(), seed=seed, auto_reset=False)
    obs, _ = env.reset()
    b0 = B0Controller(episodes)
    t_belly = np.full(episodes, np.inf)
    done = np.zeros(episodes, dtype=bool)
    full = np.tile([1.0, 0.0], (episodes, 1))
    for k in range(env.config.max_steps):
        obs, _, term, _, info = env.step(full if driver == 'full' else b0.act(obs))
        t_belly[info['belly_out'] & ~done] = (k + 1) * E.DT
        done |= term
        if done.all():
            break
    return env.state['dig_rate'].copy(), t_belly


def summary_row(label: str, t: np.ndarray) -> str:
    finite = t[np.isfinite(t)]
    med = f'{np.median(finite):.1f}' if len(finite) else '-'
    within = ' | '.join(f'{100 * np.mean(t <= s):.0f} %' for s in WITHIN_S)
    return f'| {label} | {len(t)} | {within} | {100 * np.mean(np.isfinite(t)):.0f} % | {med} |'


def markdown(results: dict) -> str:
    lines = []
    for driver, (dig, t) in results.items():
        lines += [f'**{driver}**', '',
                  '| dig rate | episodes | belly-out <= 1 s | <= 3 s | <= 10 s | ever (60 s) '
                  '| median time if it happens (s) |',
                  '| --- | --- | --- | --- | --- | --- | --- |']
        for lo, hi in zip(BINS[:-1], BINS[1:]):
            m = (dig >= lo) & (dig < hi) if hi < BINS[-1] else (dig >= lo) & (dig <= hi)
            lines.append(summary_row(f'{lo:g}-{hi:g}', t[m]))
        lines += [summary_row('**prior LogU(0.03, 0.3)**', t), '']
    return '\n'.join(lines)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--episodes', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', type=Path, default=Path('.'))
    args = ap.parse_args(argv)
    results = {name: belly_out_times(key, args.episodes, args.seed)
               for key, name in (('full', 'full throttle from rest'), ('b0', 'B0'))}
    rows = [{'driver': name, 'dig_rate': round(float(d), 5),
             'belly_out_s': float(t) if np.isfinite(t) else ''}
            for name, (dig, ts) in results.items() for d, t in zip(dig, ts)]
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / 'dig_rate_belly_out.csv', rows)
    table = markdown(results)
    (args.out / 'dig_rate_belly_out.md').write_text(table)
    print(table)


if __name__ == '__main__':
    main()
