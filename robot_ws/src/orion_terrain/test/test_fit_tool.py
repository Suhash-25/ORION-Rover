"""fit_effort_model end to end: synthetic rosbag2 (written with rosbags) -> effort_model.yaml."""

from effort_data import drive, signed_effort, TRUE_COEFS
import numpy as np
from orion_terrain.effort_model import EffortModel, EffortSource
from orion_terrain.fit_effort_model import extract_samples, fit_model, git_sha, main
from orion_terrain.imu import quat_from_rpy
from orion_terrain.params import DEFAULT_JOINT_NAMES
import pytest

rosbags = pytest.importorskip('rosbags')


def write_bag(path, current_sensed=True, duration_s=60.0):
    from rosbags.rosbag2 import Writer
    from rosbags.typesys import get_typestore, Stores

    ts = get_typestore(Stores.ROS2_JAZZY)
    T = ts.types
    t, w, dwdt, pitch = drive(duration_s)
    rng = np.random.default_rng(5)
    effort = signed_effort(TRUE_COEFS, w, dwdt, pitch) + rng.normal(0, 0.01, w.shape)
    order = [2, 0, 3, 1]                                       # not FL FR RL RR on purpose

    def header(sec):
        ns = int(round(sec * 1e9)) + 1_000_000_000
        return T['std_msgs/msg/Header'](
            stamp=T['builtin_interfaces/msg/Time'](sec=ns // 10**9, nanosec=ns % 10**9),
            frame_id='')

    def vec3(x=0.0, y=0.0, z=0.0):
        return T['geometry_msgs/msg/Vector3'](x=x, y=y, z=z)

    with Writer(path, version=8) as writer:
        js = writer.add_connection('/joint_states', 'sensor_msgs/msg/JointState', typestore=ts)
        imu = writer.add_connection('/imu/data', 'sensor_msgs/msg/Imu', typestore=ts)
        for k in range(len(t)):
            ns = int(round(t[k] * 1e9)) + 2_000_000_000              # bag time != stamps
            q = quat_from_rpy(0.0, pitch[k], 0.3)
            m = T['sensor_msgs/msg/Imu'](
                header=header(t[k]),
                orientation=T['geometry_msgs/msg/Quaternion'](x=q[0], y=q[1], z=q[2], w=q[3]),
                orientation_covariance=np.zeros(9), angular_velocity=vec3(),
                angular_velocity_covariance=np.zeros(9), linear_acceleration=vec3(z=9.81),
                linear_acceleration_covariance=np.zeros(9))
            writer.write(imu, ns, ts.serialize_cdr(m, 'sensor_msgs/msg/Imu'))
            m = T['sensor_msgs/msg/JointState'](
                header=header(t[k]), name=[DEFAULT_JOINT_NAMES[i] for i in order],
                position=np.zeros(4), velocity=w[k, order], effort=effort[k, order])
            writer.write(js, ns + 1, ts.serialize_cdr(m, 'sensor_msgs/msg/JointState'))


def test_cli_fits_bag_and_writes_yaml(tmp_path, capsys):
    bag = tmp_path / 'bag'
    write_bag(bag)
    out = tmp_path / 'effort_model.yaml'
    rc = main([str(bag), '-o', str(out), '--effort-source', 'CURRENT', '--i-stall', '9.0',
               '--git-sha', 'deadbeef'])
    assert rc == 0
    m = EffortModel.load(str(out))
    assert m.calibrated and m.effort_source == EffortSource.CURRENT and m.effort_unit == 'A'
    assert np.allclose(m.coefficients[:, :2], TRUE_COEFS[:, :2], rtol=0.05, atol=0.01)
    assert np.allclose(m.coefficients[:, 2:], TRUE_COEFS[:, 2:], rtol=0.1)
    assert min(m.r2) > 0.99
    i_free = np.mean(TRUE_COEFS[:, 0] + TRUE_COEFS[:, 1] * 0.3 / 0.12)
    assert m.i_ref == pytest.approx(9.0 - i_free, rel=0.02)
    assert m.git_sha == 'deadbeef' and m.date != 'unknown' and str(bag) in m.source
    assert 'R^2' in capsys.readouterr().out


def test_cli_refuses_without_i_ref(tmp_path, capsys):
    bag = tmp_path / 'bag'
    write_bag(bag, duration_s=20.0)
    assert main([str(bag), '--sim', '-o', str(tmp_path / 'x.yaml')]) == 2
    assert 'I_ref' in capsys.readouterr().err
    assert not (tmp_path / 'x.yaml').exists()


def test_effort_source_from_rover_health_and_sim():
    t, w, dwdt, pitch = drive(30.0)
    effort = signed_effort(TRUE_COEFS, w, dwdt, pitch)
    msgs = [(0.0, '/orion/rover_health', {'current_sensed': False})]
    for k in range(len(t)):
        msgs.append((t[k], '/imu/data', {
            'orientation': dict(zip('xyzw', quat_from_rpy(0, pitch[k], 0))),
            'angular_velocity': {'x': 0.0, 'y': 0.0, 'z': 0.0}}))
        msgs.append((t[k] + 1e-4, '/joint_states', {
            'name': list(DEFAULT_JOINT_NAMES), 'velocity': w[k], 'effort': effort[k]}))
    s = extract_samples(msgs)
    kw = {'i_ref': 3.0, 'i_stall': None, 'i_free': None, 'free_speed_mps': 0.3,
          'deadband_radps': 0.5, 'max_yaw_rate_radps': 0.2, 'source': 'test', 'sha': 'x'}
    assert fit_model(s, sim=False, effort_source=None, **kw).effort_source == \
        EffortSource.DUTY_ESTIMATE
    sim = fit_model(s, sim=True, effort_source=None, **kw)
    assert sim.effort_source == EffortSource.SIM_TORQUE and sim.effort_unit == 'N.m'


def test_git_sha_outside_repo(tmp_path):
    assert git_sha(str(tmp_path)) == 'unknown'
