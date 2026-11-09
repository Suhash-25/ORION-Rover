"""
Session protocol files and the randomised trial order (SPEC section 8).

A protocol is YAML::

    session: 2026-11-20_phase0      # prefix of every trial_id
    seed: 20261120                  # the order is a pure function of the file
    cells: [light-packed, light-loose, heavy-packed, heavy-loose]
    controllers: [B0]
    n: 20                           # trials per cell x controller
    randomisation: blocked          # blocked (default) | full
    duration_s: 60
    cone_depth_band_mm: {light-packed: [5, 15], ...}        # optional, cell valid inside

"blocked" runs n blocks, each holding every cell x controller once in its own random order,
so drift over the day (battery, soil moisture, operator fatigue) is spread evenly over the
conditions. "full" shuffles all trials at once (stronger randomisation, unbalanced in time).

CLI: ``ros2 run orion_experiments plan_session protocol.yaml [--csv plan.csv]``
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from dataclasses import field
import sys
from typing import Optional

import numpy as np
import yaml

CELLS = ('light-packed', 'light-loose', 'heavy-packed', 'heavy-loose', 'sim')
CONTROLLERS = ('B0', 'B2', 'student', 'student_critic')


@dataclass
class Protocol:
    session: str
    seed: int
    cells: list
    controllers: list
    n: int
    randomisation: str = 'blocked'
    duration_s: float = 60.0
    cone_depth_band_mm: dict = field(default_factory=dict)


@dataclass(frozen=True)
class PlannedTrial:
    order: int        # 1-based position in the session
    trial_id: str
    cell: str
    controller: str
    rep: int          # 1-based repetition of this cell x controller
    duration_s: float


def load_protocol(path: str) -> Protocol:
    """Read and validate a protocol YAML; ValueError explains what is wrong."""
    with open(path) as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f'{path}: expected a mapping')
    unknown = set(raw) - set(Protocol.__dataclass_fields__)
    if unknown:
        raise ValueError(f'{path}: unknown keys {sorted(unknown)}')
    missing = [k for k in ('session', 'seed', 'cells', 'controllers', 'n') if k not in raw]
    if missing:
        raise ValueError(f'{path}: missing keys {missing}')
    p = Protocol(**raw)
    bad_cells = [c for c in p.cells if c not in CELLS]
    bad_ctrl = [c for c in p.controllers if c not in CONTROLLERS]
    if bad_cells or bad_ctrl:
        raise ValueError(f'{path}: unknown cells {bad_cells} / controllers {bad_ctrl}; '
                         f'allowed {CELLS} / {CONTROLLERS}')
    if len(set(p.cells)) != len(p.cells) or len(set(p.controllers)) != len(p.controllers):
        raise ValueError(f'{path}: duplicate cells or controllers')
    if not (isinstance(p.n, int) and p.n > 0):
        raise ValueError(f'{path}: n must be a positive integer')
    if p.randomisation not in ('blocked', 'full'):
        raise ValueError(f'{path}: randomisation must be blocked or full')
    for cell, band in p.cone_depth_band_mm.items():
        if cell not in p.cells or len(band) != 2 or band[0] > band[1]:
            raise ValueError(f'{path}: bad cone_depth_band_mm entry {cell}: {band}')
    return p


def plan_session(p: Protocol) -> list[PlannedTrial]:
    """Randomised trial order, deterministic for a given protocol (numpy PCG64 on seed)."""
    rng = np.random.default_rng(p.seed)
    conditions = [(c, k) for c in p.cells for k in p.controllers]
    if p.randomisation == 'blocked':
        order = []
        for _ in range(p.n):
            order += [conditions[i] for i in rng.permutation(len(conditions))]
    else:
        pool = conditions * p.n
        order = [pool[i] for i in rng.permutation(len(pool))]
    reps: dict = {}
    trials = []
    for i, (cell, ctrl) in enumerate(order, start=1):
        reps[(cell, ctrl)] = reps.get((cell, ctrl), 0) + 1
        rep = reps[(cell, ctrl)]
        trials.append(PlannedTrial(i, f'{p.session}_{cell}_{ctrl}_{rep:03d}', cell, ctrl,
                                   rep, float(p.duration_s)))
    return trials


def cone_depth_valid(p: Protocol, cell: str, depths_mm) -> Optional[bool]:
    """Whether every cone depth lies in the cell's band; None when no band is set."""
    band = p.cone_depth_band_mm.get(cell)
    if band is None:
        return None
    return bool(depths_mm) and all(band[0] <= d <= band[1] for d in depths_mm)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description='Print the randomised trial order.')
    parser.add_argument('protocol')
    parser.add_argument('--csv', help='also write the plan to this CSV file')
    args = parser.parse_args(argv)
    try:
        p = load_protocol(args.protocol)
    except (OSError, ValueError, TypeError) as e:
        print(f'error: {e}', file=sys.stderr)
        return 2
    trials = plan_session(p)
    print(f'# session {p.session}, seed {p.seed}, {p.randomisation} randomisation, '
          f'{len(trials)} trials')
    print(f'{"#":>4}  {"trial_id":<48} {"cell":<13} {"controller":<15} rep')
    for t in trials:
        print(f'{t.order:>4}  {t.trial_id:<48} {t.cell:<13} {t.controller:<15} {t.rep}')
    if args.csv:
        with open(args.csv, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['order', 'trial_id', 'cell', 'controller', 'rep', 'duration_s'])
            for t in trials:
                w.writerow([t.order, t.trial_id, t.cell, t.controller, t.rep, t.duration_s])
    return 0


if __name__ == '__main__':
    sys.exit(main())
