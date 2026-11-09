"""Protocol files and the randomised session plan."""

from collections import Counter
import csv
import os

from orion_experiments.protocol import cone_depth_valid
from orion_experiments.protocol import load_protocol
from orion_experiments.protocol import main
from orion_experiments.protocol import plan_session
import pytest

CONFIG = os.path.join(os.path.dirname(__file__), '..', 'config')


def test_phase0_protocol_plans_80_blocked_trials():
    p = load_protocol(os.path.join(CONFIG, 'phase0_protocol.yaml'))
    plan = plan_session(p)
    assert len(plan) == 80
    assert Counter(t.cell for t in plan) == dict.fromkeys(p.cells, 20)
    for block in range(20):   # every block holds each cell once
        assert {t.cell for t in plan[4 * block:4 * block + 4]} == set(p.cells)
    assert len({t.trial_id for t in plan}) == 80
    assert plan[0].trial_id == f'phase0_{plan[0].cell}_B0_001'
    assert [t.order for t in plan] == list(range(1, 81))


def test_phase0b_stage2_extends_light_cells_to_40_each():
    stage1 = load_protocol(os.path.join(CONFIG, 'phase0_protocol.yaml'))
    stage2 = load_protocol(os.path.join(CONFIG, 'phase0b_protocol.yaml'))
    plan = plan_session(stage2)
    assert Counter(t.cell for t in plan) == {'light-packed': 20, 'light-loose': 20}
    assert all(t.trial_id.startswith('phase0b_') for t in plan)
    ids1 = {t.trial_id for t in plan_session(stage1)}
    assert ids1.isdisjoint(t.trial_id for t in plan)
    for cell in stage2.cells:   # soil prep must not change between the two looks
        assert stage2.cone_depth_band_mm[cell] == stage1.cone_depth_band_mm[cell]


def test_plan_is_a_pure_function_of_the_protocol():
    p = load_protocol(os.path.join(CONFIG, 'main_study_protocol.yaml'))
    a, b = plan_session(p), plan_session(p)
    assert a == b and len(a) == 400
    p.seed += 1
    assert [t.cell for t in plan_session(p)] != [t.cell for t in a]


def test_full_randomisation_is_unblocked_but_complete():
    p = load_protocol(os.path.join(CONFIG, 'main_study_protocol.yaml'))
    p.randomisation = 'full'
    plan = plan_session(p)
    counts = Counter((t.cell, t.controller) for t in plan)
    assert set(counts.values()) == {25} and len(counts) == 16
    first16 = {(t.cell, t.controller) for t in plan[:16]}
    assert len(first16) < 16   # a full shuffle almost surely repeats within 16 (fixed seed)


@pytest.mark.parametrize('text,problem', [
    ('session: s\nseed: 1\ncells: [mud]\ncontrollers: [B0]\nn: 2\n', 'unknown cells'),
    ('session: s\nseed: 1\ncells: [sim]\ncontrollers: [B0]\nn: 0\n', 'positive integer'),
    ('session: s\ncells: [sim]\ncontrollers: [B0]\nn: 2\n', 'missing keys'),
    ('session: s\nseed: 1\ncells: [sim]\ncontrollers: [B0]\nn: 2\nextra: 1\n', 'unknown keys'),
    ('session: s\nseed: 1\ncells: [sim]\ncontrollers: [B0]\nn: 2\nrandomisation: x\n',
     'randomisation'),
])
def test_bad_protocols_are_refused(tmp_path, text, problem):
    path = tmp_path / 'p.yaml'
    path.write_text(text)
    with pytest.raises(ValueError, match=problem):
        load_protocol(str(path))


def test_cone_depth_band():
    p = load_protocol(os.path.join(CONFIG, 'phase0_protocol.yaml'))
    assert cone_depth_valid(p, 'light-packed', [6.0, 10.0, 14.9])
    assert not cone_depth_valid(p, 'light-packed', [6.0, 10.0, 16.0])
    assert not cone_depth_valid(p, 'light-loose', [])
    p.cone_depth_band_mm = {}
    assert cone_depth_valid(p, 'light-loose', [1.0]) is None


def test_cli_prints_and_writes_csv(tmp_path, capsys):
    out = tmp_path / 'plan.csv'
    assert main([os.path.join(CONFIG, 'phase0_protocol.yaml'), '--csv', str(out)]) == 0
    printed = capsys.readouterr().out
    assert 'seed 20261109' in printed and 'phase0_light-packed_B0_001' in printed
    rows = list(csv.DictReader(out.open()))
    assert len(rows) == 80 and rows[0]['order'] == '1'
    assert main([str(tmp_path / 'missing.yaml')]) == 2
