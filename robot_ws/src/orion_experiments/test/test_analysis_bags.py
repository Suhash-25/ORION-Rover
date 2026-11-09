"""SPEC section 8 metrics computed from real rosbag2 files + metadata, without ROS."""

import math
import os

from bag_fixture import write_trial
from orion_experiments.analysis import load_trials
from orion_experiments.analysis import main
from orion_experiments.protocol import load_protocol
import pandas as pd
import pytest

CONFIG = os.path.join(os.path.dirname(__file__), '..', 'config')


@pytest.fixture(scope='module')
def trials_root(tmp_path_factory):
    root = str(tmp_path_factory.mktemp('trials'))
    write_trial(root, 'ok', outcome='SUCCESS', elapsed=20.0, speed=0.075)
    write_trial(root, 'trapped', cell='light-loose', outcome='ENTRAPPED', elapsed=30.0,
                speed=0.01, stuck=(5.0, 31.0), escape=None, risk=lambda t: t / 30.0)
    write_trial(root, 'aborted', outcome='ABORT', elapsed=3.0)
    write_trial(root, 'bad_cone', cell='light-loose', outcome='TIMEOUT', elapsed=60.0,
                cone=(10.0, 10.0, 10.0))
    return root


def test_metrics_of_a_successful_trial(trials_root):
    df, _ = load_trials(trials_root)
    r = df.set_index('trial_id').loc['ok']
    assert r.pose_source == '/orion/ground_truth_pose'
    assert r.progress_m == pytest.approx(1.5, abs=1e-6)        # along the start heading
    assert r.distance_m == pytest.approx(1.5, abs=1e-6)
    assert r.effective_speed == pytest.approx(1.5 / 20.0, abs=1e-6)
    assert r.time_to_goal_s == 20.0 and r.success and not r.failure
    assert r.time_to_entrapment_s == 60.0 and not r.entrapment_observed   # censored
    assert r.mean_slip == pytest.approx(0.3, abs=1e-6)
    assert r.mean_sinkage_proxy == pytest.approx(0.1, abs=1e-6)
    assert r.energy_source == 'mechanical'
    assert r.energy_j == pytest.approx(4 * 2.0 * 1.0 * 20.0, rel=1e-3)
    assert r.stuck_events == 1 and r.recovered_stuck_events == 1 and r.recovery
    assert r.escapes == 1 and math.isnan(r.max_risk)


def test_metrics_of_an_entrapped_trial_and_calibration_labels(trials_root):
    df, samples = load_trials(trials_root)
    r = df.set_index('trial_id').loc['trapped']
    assert r.entrapped and r.failure and r.time_to_entrapment_s == 30.0
    assert r.effective_speed == pytest.approx(0.01 * 30.0 / 60.0, abs=1e-6)  # per 60 s
    assert r.stuck_events == 1 and r.recovered_stuck_events == 0 and not r.recovery
    assert r.max_risk == pytest.approx(1.0)
    labels = [y for _, y in samples['B0']]
    # 20 Hz over 30 s, label 1 in the last 10 s; the other trials have no critic (NaN)
    assert len(labels) == 601 and sum(labels) == 201


def test_invalid_trials_are_marked(trials_root):
    protocol = load_protocol(os.path.join(CONFIG, 'phase0_protocol.yaml'))
    df, _ = load_trials(trials_root, protocol=protocol)
    v = df.set_index('trial_id').valid
    assert v['ok']
    assert not v['aborted']        # ABORT
    assert not v['bad_cone']       # 10 mm is outside the loose band
    assert not v['trapped']        # cone depths 10-12 mm are outside the loose band too


def test_cli_writes_report_and_csvs(trials_root, tmp_path, capsys):
    assert main([trials_root, '--out', str(tmp_path)]) == 0
    for name in ('trials.csv', 'tests.csv', 'calibration.csv', 'report.md'):
        assert (tmp_path / name).exists(), name
    report = (tmp_path / 'report.md').read_text()
    assert 'Phase-0 kill test' in report and 'Critic calibration' in report
    assert len(pd.read_csv(tmp_path / 'trials.csv')) == 4
    assert 'Phase-0 stage 1:' in capsys.readouterr().out
