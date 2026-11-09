"""Synthetic rosbag2 round trip: readable with rosbags, counts and contents right."""

import csv

import numpy as np
import pytest
from rosbags.highlevel import AnyReader
from rosbags.typesys import get_typestore, Stores

from orion_learning import obs_spec as S
from orion_learning.synth_bag import JOINT_NAMES, TOPICS, write_bag

SECONDS = 4.0
STEPS = int(SECONDS * S.CONTROL_HZ)


@pytest.fixture(scope='module', params=['sqlite3', 'mcap'])
def bag(request, tmp_path_factory):
    out = tmp_path_factory.mktemp(request.param) / 'bag'
    res = write_bag(out, cell='heavy-loose', seconds=SECONDS, seed=3, storage=request.param,
                    force_dropouts=True)
    return out, res


def _read(path):
    msgs = {}
    with AnyReader([path], default_typestore=get_typestore(Stores.ROS2_JAZZY)) as r:
        for conn, t, raw in r.messages():
            msgs.setdefault(conn.topic, []).append((t, r.deserialize(raw, conn.msgtype)))
        types = {c.topic: c.msgtype for c in r.connections}
    return msgs, types


def test_topics_types_and_counts(bag):
    path, res = bag
    msgs, types = _read(path)
    assert types == dict(TOPICS.values())
    with open(path / 'truth.csv') as f:
        truth = list(csv.DictReader(f))
    assert len(truth) == STEPS
    n_fix = sum(int(r['has_fix']) for r in truth)
    assert 0 < n_fix < STEPS                       # dropouts happened and were respected
    expect = {'/joint_states': 4 * STEPS, '/imu/data': 4 * STEPS,
              '/diff_drive_controller/odom': STEPS, '/cmd_vel/policy': STEPS,
              '/orion/ground_truth_pose': n_fix}
    assert {k: len(v) for k, v in msgs.items()} == expect
    assert res['counts'] == {k: expect[TOPICS[k][0]] for k in TOPICS}


def test_message_contents(bag):
    path, _ = bag
    msgs, _ = _read(path)
    js = [m for _, m in msgs['/joint_states']]
    assert list(js[0].name) == JOINT_NAMES
    eff = np.array([m.effort for m in js])
    assert np.all(np.abs(eff) < 4.0) and eff[-40:].mean() > 1.0     # N.m, loaded in loose sand
    imu = [m for _, m in msgs['/imu/data']]
    az = np.array([m.linear_acceleration.z for m in imu])
    assert az.mean() == pytest.approx(9.81, abs=0.1)                 # gravity on +z
    assert imu[0].header.frame_id == 'imu_link'
    odom = msgs['/diff_drive_controller/odom'][-1][1]
    assert odom.header.frame_id == 'odom' and odom.child_frame_id == 'base_link'
    stamps = [t for t, _ in msgs['/joint_states']]
    assert np.all(np.diff(stamps) > 0)
    for _, m in msgs['/orion/ground_truth_pose'][:3]:
        assert m.header.frame_id == 'sandbox'


def test_wheel_odometry_overestimates_true_progress_under_slip(bag):
    """Odometry is the slip numerator, not truth: in loose sand it runs ahead of the pose."""
    path, _ = bag
    msgs, _ = _read(path)
    with open(path / 'truth.csv') as f:
        truth = list(csv.DictReader(f))
    odom_x = msgs['/diff_drive_controller/odom'][-1][1].pose.pose.position.x
    true_x = float(truth[-1]['x'])
    start = -0.85
    slip = np.mean([float(r['slip_left']) for r in truth[-20:]])
    assert slip > 0.1
    assert odom_x - start > (true_x - start) * 1.1
