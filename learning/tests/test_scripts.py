"""scripts/sensitivity_a_s.py (council M1) and scripts/dig_rate_belly_out.py (W2) run."""

import importlib.util
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f'{name}.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_sensitivity_sweep_flips_the_loose_cells_with_a_s():
    mod = _load('sensitivity_a_s')
    rows = mod.sweep(trials=3, seed=0, a_s_values=(0.3, 2.0), dig_rates=(0.1,))
    assert len(rows) == 2 * 4
    by = {(r['loose_a_s'], r['cell']): r for r in rows}
    # a_s only changes the loose cells; at 2.0 heavy-loose is not traversable, at 0.3 it is.
    assert by[(0.3, 'light-packed')] == {**by[(2.0, 'light-packed')], 'loose_a_s': 0.3}
    assert by[(0.3, 'heavy-loose')]['success_rate'] == 1.0
    assert by[(2.0, 'heavy-loose')]['success_rate'] == 0.0
    table = mod.markdown(rows)
    assert table.count('\n| 0.1 |') == 2 and 'heavy-loose vs light-packed' in table


def test_dig_rate_script_reports_belly_out_times():
    mod = _load('dig_rate_belly_out')
    dig, t = mod.belly_out_times('full', episodes=24, seed=0)
    assert dig.shape == t.shape == (24,)
    assert np.all((dig >= 0.03) & (dig <= 0.3))
    finite = t[np.isfinite(t)]
    assert len(finite) > 0 and np.all((finite > 0) & (finite <= 60.0))
    table = mod.markdown({'full throttle': (dig, t)})
    assert 'prior LogU(0.03, 0.3)' in table and table.count('\n| 0.') == len(mod.BINS) - 1
