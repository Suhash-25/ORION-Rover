"""
Replay a synthetic trajectory with known slip through estimate_from_messages.

The kinematic generator (synthetic.py) produces 100 Hz IMU with noise and bias, 100 Hz
joint states in shuffled order, 50 Hz wheel odometry and 20 Hz ground-truth poses with
dropouts, with a slip ramp that differs per side.
"""

import numpy as np
from orion_terrain.estimator import estimate_from_messages
from orion_terrain.speed import SpeedSource
import pytest
from synthetic import generate, in_dropout, Scenario


def replay(sc):
    msgs, truth = generate(sc)
    return estimate_from_messages(msgs), truth


def slip_errors(out, truth, t_min=1.0):
    return np.array([e.slip_side - truth(e.t)['slip_side'] for e in out if e.t >= t_min])


def test_slip_side_rms_below_0p05_with_typical_occlusions():
    """Short tag occlusions (1-1.5 s) and a 0.02 m/s^2 accelerometer bias."""
    sc = Scenario(dropouts=[(8.0, 9.5), (20.0, 21.0)], imu_bias0=0.02, bias_walk=0.005)
    out, truth = replay(sc)
    err = slip_errors(out, truth)
    rms = float(np.sqrt(np.mean(err ** 2)))
    assert rms < 0.05, f'slip_side RMS error {rms:.3f}'
    assert len(out) == pytest.approx(sc.duration_s * 20, abs=2)       # 20 Hz output
    fixed = [e for e in out if e.body_speed_source == SpeedSource.EXTERNAL and e.t > 1.0]
    v_err = np.array([e.body_speed - truth(e.t)['v'] for e in fixed])
    assert np.sqrt(np.mean(v_err ** 2)) < 0.01                         # m/s while fixed


def test_long_dropout_confidence_and_recovery():
    sc = Scenario(dropouts=[(10.0, 16.0)], imu_bias0=0.02, bias_walk=0.005, seed=2)
    out, truth = replay(sc)
    by_t = {round(e.t, 2): e for e in out}
    assert by_t[9.9].confidence > 0.5 and by_t[9.9].body_speed_source == SpeedSource.EXTERNAL
    low = [e for e in out if 10.0 <= e.t < 16.0 and e.confidence < 0.3]
    assert low and low[0].t <= 14.0, 'confidence must fall below 0.3 within 4 s of dropout'
    assert all(e.confidence < 0.3 for e in out if low[0].t <= e.t < 16.0)
    during = [e for e in out if 10.0 <= e.t < 16.0]
    assert all(e.body_speed_source == SpeedSource.IMU_ONLY for e in during if e.t > 10.3)
    # The filter's 1-sigma band covers the IMU-only drift (documented bias behaviour).
    for e in during:
        assert abs(e.body_speed - truth(e.t)['v']) < 3 * e.body_speed_std + 0.01
    # Fixes back: 5 poses for the derivative, then confidence and accuracy return fast.
    after = [e for e in out if 16.5 <= e.t < 18.0]
    assert all(e.body_speed_source == SpeedSource.EXTERNAL for e in after)
    assert all(abs(e.body_speed - truth(e.t)['v']) < 0.03 for e in after)
    assert all(e.confidence > 0.5 for e in after)


def test_replay_is_order_independent_and_ignores_unknown_topics():
    sc = Scenario(duration_s=5.0, dropouts=[])
    msgs, _ = generate(sc)
    a = estimate_from_messages(msgs)
    noisy = list(reversed(msgs)) + [(1.0, '/rosout', {'msg': 'hi'})]
    b = estimate_from_messages(noisy)
    assert [e.t for e in a] == [e.t for e in b]
    assert np.allclose([e.slip_side for e in a], [e.slip_side for e in b])


def test_generator_dropouts_are_real():
    sc = Scenario()
    msgs, _ = generate(sc)
    gt = [t for t, topic, _ in msgs if topic == '/orion/ground_truth_pose']
    assert not any(in_dropout(t, sc.dropouts) for t in gt)
    assert len(gt) > 0
