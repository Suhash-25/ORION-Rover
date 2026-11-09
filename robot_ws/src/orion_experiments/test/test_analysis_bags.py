"""SPEC section 8 metrics computed from real rosbag2 files + metadata, without ROS."""

import math
import os

from bag_fixture import write_trial
import numpy as np
from orion_experiments.analysis import entrapment_onset
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
    # Drives 0.4 m by 8 s, then stands still; a short spurious FLAG_STUCK at 2-3 s while
    # moving, the real stuck episode from 11 s, hard-stuck HALT at 34 s (11 + 3 + 20).
    write_trial(root, 'trapped', cell='light-loose', outcome='ENTRAPPED', elapsed=34.0,
                speed=0.05, stop_at=8.0, stuck=((2.0, 3.0), (11.0, 35.0)), hard_stuck_at=34.0,
                escape=None, risk=lambda t: t / 34.0)
    # Same stop at 8 s; two escapes toggle FLAG_STUCK (11-16 s, 19-24 s) without moving the
    # rover, then the policy gives up at 24.5 s (no hard-stuck HALT: the command is zero).
    write_trial(root, 'gave_up', cell='heavy-loose', outcome='ENTRAPPED', elapsed=24.6,
                speed=0.05, stop_at=8.0, stuck=((11.0, 16.0), (19.0, 24.0)), escape=None,
                gave_up_at=24.5)
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
    assert r.entrapped and r.failure and r.time_to_entrapment_s == 34.0
    assert r.effective_speed == pytest.approx(0.4 / 60.0, abs=1e-6)  # per 60 s
    assert r.stuck_events == 2 and r.recovered_stuck_events == 1 and not r.recovery
    assert r.max_risk == pytest.approx(1.0)
    # Council M2: labels count from the onset of the stuck episode that ended in the HALT
    # (11 s), not from the HALT (34 s) and not from the earlier episode at 2 s.
    assert r.entrapment_onset_s == pytest.approx(11.0) and r.onset_source == \
        '/orion/ground_truth_pose'
    assert r.entrapment_source == 'hard_stuck'
    labels = [y for _, y in samples['B0']]
    # 20 Hz over 34 s; label 1 from 11 - 10 = 1 s on; the other trials have no critic (NaN)
    assert len(labels) == 681 and sum(labels) == 661


def test_policy_give_up_is_dated_at_the_onset_of_its_stuck_episode(trials_root):
    df, _ = load_trials(trials_root)
    r = df.set_index('trial_id').loc['gave_up']
    assert r.entrapped and r.failure and r.entrapment_source == 'policy_gave_up'
    assert r.time_to_entrapment_s == pytest.approx(24.6)
    # Same M2 anchor as the hard-stuck HALT: the first FLAG_STUCK of the episode (11 s), not
    # the give-up (24.5 s) and not the second escape's flag (19 s).
    assert r.entrapment_onset_s == pytest.approx(11.0)


T = np.arange(0.0, 40.0, 0.05)


def flags(*runs):
    """FLAG_STUCK samples at 20 Hz, high inside each (start, end) run."""
    return [(s, any(a <= s < b for a, b in runs)) for s in T]


def test_entrapment_onset_episode_rules():
    t = T
    # Drives to x = 1 m by 10 s, then still: a run while driving is an earlier episode.
    pos = [(s, (min(s, 10.0) * 0.1, 0.0)) for s in t]
    assert entrapment_onset(flags((2, 4), (13, 40)), pos, 39.0) == pytest.approx(13.0)
    # Slip flag already high while creeping the last 0.1 m: the onset is that run's start.
    creep = [(s, (min(s, 20.0) * 0.02, 0.0)) for s in t]
    assert entrapment_onset(flags((12, 40)), creep, 39.0) == pytest.approx(12.0)
    # Escape toggles the flag without moving the rover: still one episode.
    assert entrapment_onset(flags((13, 18), (21, 40)), pos, 39.0) == pytest.approx(13.0)
    # 1-D odometer track works the same way.
    odo = [(s, min(s, 10.0) * 0.1) for s in t]
    assert entrapment_onset(flags((2, 4), (13, 40)), odo, 39.0) == pytest.approx(13.0)
    # Without a track the last run is the episode; without FLAG_STUCK there is no onset.
    assert entrapment_onset(flags((13, 18), (21, 40)), [], 39.0) == pytest.approx(21.0)
    assert entrapment_onset(flags(), pos, 39.0) is None
    assert entrapment_onset(flags((39.5, 40)), pos, 39.0) is None   # after the HALT


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
    assert len(pd.read_csv(tmp_path / 'trials.csv')) == 5
    assert 'Phase-0 stage 1:' in capsys.readouterr().out
