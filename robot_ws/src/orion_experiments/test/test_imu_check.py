"""IMU-vs-camera bring-up check (BRINGUP step 10) on synthetic runs and a real bag file."""

import math
from pathlib import Path

import numpy as np
from orion_experiments.analysis import find_msg_dir
from orion_experiments.analysis import make_typestore
from orion_experiments.imu_check import imu_integral_check
from orion_experiments.imu_check import main
import pytest

HEADING = 0.4


def run(scale=1.0, bias=0.05, static_s=10.0, wobble=0.0, seed=0):
    """Stand still, reach 0.2 m/s at 0.4 m/s^2, cruise to 2.15 m. IMU 100 Hz, camera 30 Hz."""
    rng = np.random.default_rng(seed)
    ta = np.arange(0.0, static_s + 11.0, 0.01)
    acc = np.where((ta >= static_s) & (ta < static_s + 0.5), 0.4, 0.0)
    ax = scale * acc + bias + rng.normal(0.0, 0.02, len(ta))
    tp = np.arange(0.0, static_s + 11.0, 1.0 / 30.0)
    rel = np.clip(tp - static_s, 0.0, None)
    d = np.where(rel < 0.5, 0.2 * rel ** 2, 0.05 + 0.2 * (rel - 0.5))
    d = d + wobble * (tp < static_s) * np.sin(tp) + rng.normal(0.0, 0.001, len(tp))
    imu = list(zip(ta, ax))
    poses = list(zip(tp, d * math.cos(HEADING), d * math.sin(HEADING)))
    return imu, poses


def test_a_correct_imu_passes_and_the_bias_is_removed():
    r = imu_integral_check(*run())
    assert r['passed'] and r['error'] < 0.03
    assert r['bias'] == pytest.approx(0.05, abs=0.005)
    assert r['camera_speed'] == pytest.approx(0.2, abs=0.01)
    assert r['distance_m'] == pytest.approx(2.15, abs=0.02)      # 0.05 + 0.2 x 10.5


def test_a_23_percent_low_imu_fails():
    r = imu_integral_check(*run(scale=0.77))
    assert not r['passed'] and r['ratio'] == pytest.approx(0.77, abs=0.03)


def test_a_bad_recording_is_rejected():
    with pytest.raises(ValueError, match='stand still'):
        imu_integral_check(*run(wobble=0.05))
    imu, poses = run()
    with pytest.raises(ValueError, match='never saw'):
        imu_integral_check(imu, [(t, 0.0, 0.0) for t, _, _ in poses])
    with pytest.raises(ValueError, match='need'):
        imu_integral_check([], poses)


def test_cli_reads_a_bag(tmp_path, capsys):
    from rosbags.rosbag2 import StoragePlugin
    from rosbags.rosbag2 import Writer
    ts = make_typestore(find_msg_dir())
    T = ts.types
    imu, poses = run()

    def header(t):
        stamp = T['builtin_interfaces/msg/Time'](sec=int(t), nanosec=int((t % 1) * 1e9))
        return T['std_msgs/msg/Header'](stamp=stamp, frame_id='x')

    vec = T['geometry_msgs/msg/Vector3']
    quat = T['geometry_msgs/msg/Quaternion'](x=0.0, y=0.0, z=0.0, w=1.0)
    cov = np.zeros(9)
    with Writer(Path(tmp_path) / 'bag', version=9, storage_plugin=StoragePlugin.MCAP) as w:
        ci = w.add_connection('/imu/data', 'sensor_msgs/msg/Imu', typestore=ts)
        cp = w.add_connection('/orion/ground_truth_pose', 'geometry_msgs/msg/PoseStamped',
                              typestore=ts)
        events = [(t, ci, T['sensor_msgs/msg/Imu'](
            header=header(t), orientation=quat, orientation_covariance=cov,
            angular_velocity=vec(x=0.0, y=0.0, z=0.0), angular_velocity_covariance=cov,
            linear_acceleration=vec(x=a, y=0.0, z=9.81), linear_acceleration_covariance=cov))
            for t, a in imu]
        events += [(t, cp, T['geometry_msgs/msg/PoseStamped'](
            header=header(t), pose=T['geometry_msgs/msg/Pose'](
                position=T['geometry_msgs/msg/Point'](x=x, y=y, z=0.0), orientation=quat)))
            for t, x, y in poses]
        for t, conn, msg in sorted(events, key=lambda e: e[0]):
            w.write(conn, int(round((100.0 + t) * 1e9)), ts.serialize_cdr(msg, msg.__msgtype__))
    assert main([str(tmp_path / 'bag')]) == 0
    assert 'PASS' in capsys.readouterr().out
