"""The observation builder must reproduce obs_spec.build_observation exactly."""

import math

import numpy as np
from orion_learning import obs_spec
from orion_policy.observation import gravity_free_accel
from orion_policy.observation import InputNotReady
from orion_policy.observation import ObservationBuilder
from orion_policy.observation import quat_to_rpy
import pytest

JOINTS = ['front_left_wheel_joint', 'front_right_wheel_joint',
          'rear_left_wheel_joint', 'rear_right_wheel_joint']


def rpy_to_quat(roll, pitch, yaw):
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return (sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy, cr * cp * cy + sr * sp * sy)


def rot_body_to_world(roll, pitch, yaw):
    cr, sr, cp, sp = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return rz @ ry @ rx


def test_quat_round_trip():
    for rpy in [(0.0, 0.0, 0.0), (0.2, -0.1, 1.0), (-0.4, 0.3, -2.5)]:
        assert np.allclose(quat_to_rpy(*rpy_to_quat(*rpy)), rpy, atol=1e-12)


def test_gravity_removed_for_tilted_imu():
    roll, pitch, yaw = 0.25, -0.15, 0.7
    true_body_accel = np.array([0.4, -0.2, 0.1])
    # what the accelerometer reads: R^T (a_world + g_up)
    r = rot_body_to_world(roll, pitch, yaw)
    measured = true_body_accel + r.T @ np.array([0.0, 0.0, 9.81])
    assert np.allclose(gravity_free_accel(measured, roll, pitch, 9.81), true_body_accel,
                       atol=1e-12)


def physical_inputs():
    rng = np.random.default_rng(3)
    return {
        'prev_cmd': (0.21, -0.37),
        'wheel_speed': rng.uniform(-4, 4, 4), 'wheel_effort': rng.uniform(-3, 3, 4),
        'body_accel_xy': (0.31, -0.12), 'yaw_rate': 0.42, 'roll': 0.11, 'pitch': -0.07,
        'body_speed': 0.18, 'speed_confidence': 0.8, 'slip_side': (0.35, -0.1),
        'sinkage_proxy': 0.27}


def feed(builder, phys, t=10.0, names=JOINTS):
    order = [JOINTS.index(n) for n in names]
    r = rot_body_to_world(phys['roll'], phys['pitch'], 0.3)
    accel = np.array([*phys['body_accel_xy'], 0.05]) + r.T @ np.array([0, 0, 9.81])
    builder.update_joint_state(t, names, [phys['wheel_speed'][i] for i in order],
                               [phys['wheel_effort'][i] for i in order])
    builder.update_imu(t, rpy_to_quat(phys['roll'], phys['pitch'], 0.3),
                       (0.01, -0.02, phys['yaw_rate']), accel)
    builder.update_terrain(t, phys['body_speed'], phys['speed_confidence'],
                           phys['slip_side'], phys['sinkage_proxy'])


@pytest.mark.parametrize('names', [JOINTS, list(reversed(JOINTS)),
                                   [JOINTS[2], JOINTS[0], JOINTS[3], JOINTS[1]]])
def test_builder_matches_obs_spec(names):
    phys = physical_inputs()
    builder = ObservationBuilder(JOINTS, effort_max=3.0)
    feed(builder, phys, names=names)
    got = builder.build(10.05, phys['prev_cmd'])
    want = obs_spec.build_observation(effort_max=3.0, **phys)
    assert got.dtype == np.float32 and got.shape == (obs_spec.OBS_DIM,)
    np.testing.assert_allclose(got, want, atol=1e-6)


def test_builder_refuses_stale_or_missing_inputs():
    phys = physical_inputs()
    builder = ObservationBuilder(JOINTS, effort_max=3.0)
    with pytest.raises(InputNotReady, match='no joint_states'):
        builder.build(0.0, (0, 0))
    feed(builder, phys, t=10.0)
    builder.build(10.29, (0, 0))
    with pytest.raises(InputNotReady, match='joint_states is 0.31 s old'):
        builder.build(10.31, (0, 0))


def test_joint_state_without_wheel_or_effort_is_rejected():
    builder = ObservationBuilder(JOINTS, effort_max=3.0)
    with pytest.raises(ValueError, match='lacks wheel joints'):
        builder.update_joint_state(0.0, JOINTS[:3], [0] * 3, [0] * 3)
    with pytest.raises(ValueError, match='effort'):
        builder.update_joint_state(0.0, JOINTS, [0] * 4, [])
