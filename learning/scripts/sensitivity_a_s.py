"""B0 in the four sandbox cells vs the guessed loose-soil a_s and the dig rate (council M1).

The sim's Phase-0 premise (B0 slows down / gets trapped on heavy-loose) hinges on two
guessed parameters: the slip-sinkage coefficient a_s of the loose cell and the dig rate of
the sinkage model. This sweeps both and reports B0 success, entrapment and effective speed
per cell at g = 9.81 m/s^2, plus the heavy-loose speed change vs light-packed.

    python scripts/sensitivity_a_s.py --trials 25 --out runs/sensitivity

Writes sensitivity_a_s.csv (one row per a_s x dig rate x cell) and sensitivity_a_s.md (the
table in learning/README.md). Non-cell settings are the SPEC evaluation distribution
(env.spec_eval_config); every grid point uses the same seeds (common random numbers).
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np

from orion_learning import env as E
from orion_learning.evaluate import run_trials, write_csv

A_S = (0.3, 0.5, 0.75, 1.0, 1.5, 2.0)
DIG_RATES = (0.03, 0.1, 0.3)      # m of sinkage per m of slipped rim travel (prior LogU ends)


def sweep(trials: int, seed: int, a_s_values=A_S, dig_rates=DIG_RATES, gravity: float = 9.81):
    rows = []
    for dig in dig_rates:
        base = replace(E.spec_eval_config(), dig_rate=(dig, dig))
        for a_s in a_s_values:
            for c_i, cell in enumerate(E.CELLS):
                cfg = E.cell_config(cell, gravity, base)
                if cell.endswith('loose'):
                    cfg = replace(cfg, soil=replace(cfg.soil, a_s=(a_s, a_s)))
                trial_rows = run_trials('B0', cell, gravity, trials, seed + c_i, config=cfg)
                eff = np.array([r['effective_speed'] for r in trial_rows])
                rows.append({'loose_a_s': a_s, 'dig_rate': dig, 'cell': cell, 'trials': trials,
                             'success_rate': np.mean([r['success'] for r in trial_rows]),
                             'entrapment_rate': np.mean([r['entrapped'] for r in trial_rows]),
                             'effective_speed_mean': round(float(eff.mean()), 4)})
                print(rows[-1], flush=True)
    return rows


def markdown(rows: list[dict]) -> str:
    """One line per (dig rate, a_s): success % and effective speed per cell, HL vs LP."""
    head = ('| dig rate | loose a_s | ' + ' | '.join(E.CELLS)
            + ' | heavy-loose vs light-packed speed |')
    lines = [head, '|' + ' --- |' * (len(E.CELLS) + 3)]
    keys = sorted({(r['dig_rate'], r['loose_a_s']) for r in rows})
    for dig, a_s in keys:
        cells = {r['cell']: r for r in rows if (r['dig_rate'], r['loose_a_s']) == (dig, a_s)}
        parts = [f"{100 * float(cells[c]['success_rate']):.0f} / "
                 f"{100 * float(cells[c]['entrapment_rate']):.0f} / "
                 f"{float(cells[c]['effective_speed_mean']):.3f}" for c in E.CELLS]
        lp = cells['light-packed']['effective_speed_mean']
        hl = cells['heavy-loose']['effective_speed_mean']
        change = 100 * (hl / lp - 1) if lp else float('nan')
        lines.append(f'| {dig:g} | {a_s:g} | ' + ' | '.join(parts) + f' | {change:+.0f} % |')
    legend = ('Cells: B0 success % / entrapped % (belly-out or ENTRAPPED) / mean effective '
              'speed (m/s); the remaining trials are 60 s timeouts.')
    return '\n'.join(lines) + f'\n\n{legend}\n'


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--trials', type=int, default=25)
    ap.add_argument('--seed', type=int, default=12345)
    ap.add_argument('--out', type=Path, default=Path('.'))
    args = ap.parse_args(argv)
    rows = sweep(args.trials, args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / 'sensitivity_a_s.csv', rows)
    table = markdown(rows)
    (args.out / 'sensitivity_a_s.md').write_text(table)
    print(table)


if __name__ == '__main__':
    main()
