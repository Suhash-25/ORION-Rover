"""configs/smoke.yaml runs every stage end to end with finite metrics (what CI runs)."""

import csv
import json
import math
from pathlib import Path

from orion_learning.train import run

CONFIG = Path(__file__).resolve().parents[1] / 'configs' / 'smoke.yaml'


def test_smoke_training_all_stages(tmp_path):
    run_dir = run(CONFIG, runs_root=tmp_path)
    for f in ('config.yaml', 'run_info.json', 'metrics.csv', 'summary.json', 'teacher.pt',
              'student.pt', 'critic.pt', 'b2.pt', 'eval_summary.csv', 'eval_episodes.csv'):
        assert (run_dir / f).exists(), f
    info = json.loads((run_dir / 'run_info.json').read_text())
    assert info['seed'] == 0 and info['git_sha']
    with open(run_dir / 'metrics.csv') as f:
        rows = list(csv.DictReader(f))
    stages = {r['stage'] for r in rows}
    assert {'teacher', 'student', 'critic', 'b2', 'export'} <= stages
    bad = [r for r in rows if not math.isfinite(float(r['value']))]
    assert not bad, bad[:5]
    for controller, files in (('student', {'adapter.onnx', 'policy.onnx', 'critic.onnx'}),
                              ('B2', {'policy.onnx'})):
        d = run_dir / 'export' / controller
        assert {p.name for p in d.glob('*.onnx')} == files
        assert json.loads((d / 'manifest.json').read_text())['controller'] == controller
    with open(run_dir / 'eval_summary.csv') as f:
        ev = list(csv.DictReader(f))
    assert len(ev) == 4 * 4 * 2          # controllers x cells x gravities
    assert all(math.isfinite(float(r['effective_speed_mean'])) for r in ev)
