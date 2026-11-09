"""/joint_states is mapped to FL FR RL RR by joint NAME, never by index."""

from conftest import RecordingLogger
from effort_data import make_model, signed_effort, TRUE_COEFS
import numpy as np
from orion_terrain.effort_model import EffortSource
from orion_terrain.estimator import TerrainEstimator
from orion_terrain.params import DEFAULT_JOINT_NAMES

LEVEL = (0.0, 0.0, 0.0, 1.0)
W = np.array([3.0, 4.0, 5.0, 6.0])              # FL FR RL RR, all distinct


def run(order, extra_joints=()):
    est = TerrainEstimator(effort_model=make_model(source=EffortSource.SIM_TORQUE), sim=True)
    effort = signed_effort(TRUE_COEFS, W, np.zeros(4), 0.0) + np.array([0.1, 0.2, 0.3, 0.4])
    names = [DEFAULT_JOINT_NAMES[i] for i in order] + list(extra_joints)
    vel = [W[i] for i in order] + [9.9] * len(extra_joints)
    eff = [effort[i] for i in order] + [9.9] * len(extra_joints)
    for k in range(300):
        t = k * 0.01
        est.on_imu(t, LEVEL, (0, 0, 0), (0, 0, 9.81))
        est.on_lidar_odom(t, 0.5)
        est.on_joint_state(t, names, vel, eff)
    return est.estimate(3.0)


def test_shuffled_joint_order_gives_identical_output():
    ref = run([0, 1, 2, 3])
    for order in ([3, 2, 1, 0], [1, 3, 0, 2], [2, 0, 3, 1]):
        e = run(order, extra_joints=('lidar_mast_joint',))
        assert np.allclose(e.slip, ref.slip)
        assert np.allclose(e.effort_residual, ref.effort_residual)


def test_mapping_puts_each_wheel_in_its_slot():
    e = run([2, 0, 3, 1])
    assert np.allclose(e.effort_residual, [0.1, 0.2, 0.3, 0.4], atol=0.01)
    assert np.all(np.diff(e.slip) > 0)                # slip grows with W: FL < FR < RL < RR


def test_missing_joint_is_ignored_with_one_warning():
    log = RecordingLogger()
    est = TerrainEstimator(logger=log)
    for k in range(10):
        est.on_joint_state(k * 0.01, ['front_left_wheel_joint', 'a', 'b', 'c'],
                           [1.0] * 4, [0.0] * 4)
    assert np.all(est.estimate(0.1).slip == 0.0)
    assert log.count('lack one of') == 1
