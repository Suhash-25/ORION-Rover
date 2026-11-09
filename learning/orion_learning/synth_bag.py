"""Write a synthetic rosbag2 from an env rollout, for replay tests of the ROS packages (F7).

No ROS install needed (rosbags library). The bag is what the rover would record:

  /joint_states                 sensor_msgs/JointState   80 Hz  wheel angle (rad), measured
                                speed (rad/s), effort (N.m) for front_left/front_right/
                                rear_left/rear_right_wheel_joint
  /imu/data                     sensor_msgs/Imu          80 Hz  orientation (roll, pitch, yaw),
                                gyro, specific force with gravity on +z (m/s^2), accel bias
  /diff_drive_controller/odom   nav_msgs/Odometry        20 Hz  wheel odometry (NOT truth)
  /orion/ground_truth_pose      geometry_msgs/PoseStamped 20 Hz overhead-camera pose
                                (1 mm noise), absent during fix dropouts
  /cmd_vel/policy               geometry_msgs/TwistStamped 20 Hz controller output

and a sidecar truth.csv (one row per 20 Hz control step) with the true pose, body speed,
per-wheel and per-side slip (obs_spec.wheel_slip on true wheel speeds and body velocity),
per-wheel sinkage (m) and whether a ground-truth fix was published.

    python -m orion_learning.synth_bag --out /tmp/orion_bag --cell heavy-loose --seconds 20
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from pathlib import Path
import shutil

import numpy as np
from rosbags.rosbag2 import StoragePlugin, Writer
from rosbags.typesys import get_typestore, Stores

from orion_learning import env as E
from orion_learning import obs_spec as S
from orion_learning.baselines import B0Controller

JOINT_NAMES = ['front_left_wheel_joint', 'front_right_wheel_joint',
               'rear_left_wheel_joint', 'rear_right_wheel_joint']   # = obs_spec.WHEEL_ORDER
TOPICS = {
    'joint_states': ('/joint_states', 'sensor_msgs/msg/JointState'),
    'imu': ('/imu/data', 'sensor_msgs/msg/Imu'),
    'odom': ('/diff_drive_controller/odom', 'nav_msgs/msg/Odometry'),
    'truth_pose': ('/orion/ground_truth_pose', 'geometry_msgs/msg/PoseStamped'),
    'cmd': ('/cmd_vel/policy', 'geometry_msgs/msg/TwistStamped'),
}
T0_NS = 1_700_000_000 * 10 ** 9     # bag start time
POSE_NOISE_M = 0.001                # overhead camera position noise (1 sigma)
YAW_NOISE_RAD = 0.002
FRAME_SANDBOX = 'sandbox'           # ground-truth (camera) frame, origin = sandbox centre


def quat_from_rpy(roll, pitch, yaw):
    """(x, y, z, w) quaternion of ZYX intrinsic roll/pitch/yaw (rad)."""
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    return (sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy, cr * cp * cy + sr * sp * sy)


class _Msgs:
    def __init__(self):
        self.ts = get_typestore(Stores.ROS2_JAZZY)
        t = self.ts.types
        self.Time, self.Header = t['builtin_interfaces/msg/Time'], t['std_msgs/msg/Header']
        self.Vector3, self.Point = t['geometry_msgs/msg/Vector3'], t['geometry_msgs/msg/Point']
        self.Quaternion, self.Pose = t['geometry_msgs/msg/Quaternion'], t['geometry_msgs/msg/Pose']
        self.Twist = t['geometry_msgs/msg/Twist']
        self.PoseWC = t['geometry_msgs/msg/PoseWithCovariance']
        self.TwistWC = t['geometry_msgs/msg/TwistWithCovariance']
        self.types = t

    def header(self, ns: int, frame: str):
        return self.Header(stamp=self.Time(sec=ns // 10 ** 9, nanosec=ns % 10 ** 9),
                           frame_id=frame)

    def quat(self, roll, pitch, yaw):
        x, y, z, w = quat_from_rpy(roll, pitch, yaw)
        return self.Quaternion(x=float(x), y=float(y), z=float(z), w=float(w))

    def ser(self, msg, msgtype: str) -> bytes:
        return self.ts.serialize_cdr(msg, msgtype)


def write_bag(out: Path, cell: str | None = 'light-loose', seconds: float = 20.0, seed: int = 0,
              gravity: float = 9.81, storage: str = 'sqlite3', force_dropouts: bool = False,
              overwrite: bool = False) -> dict:
    """Roll out B0 in one env and write the bag at `out` plus `out`/truth.csv.

    cell=None samples from the training distribution instead of a sandbox cell.
    force_dropouts makes ground-truth dropouts certain (onset rate 0.5 /s) for tests.
    Returns message counts per topic and the truth rows.
    """
    out = Path(out)
    if out.exists():
        if not overwrite:
            raise FileExistsError(out)
        shutil.rmtree(out)
    cfg = E.cell_config(cell, gravity) if cell else E.EnvConfig()
    if force_dropouts:
        cfg = replace(cfg, p_fix_dropout=1.0, dropout_rate_hz=0.5, dropout_duration_s=(0.5, 2.0))
    env = E.OrionEnv(1, cfg, seed=seed, auto_reset=False, record_substeps=True)
    obs, _ = env.reset()
    ctl = B0Controller(1)
    rng = np.random.default_rng(seed + 1)
    m = _Msgs()
    T = m.types
    plugin = {'sqlite3': StoragePlugin.SQLITE3, 'mcap': StoragePlugin.MCAP}[storage]
    counts = dict.fromkeys(TOPICS, 0)
    rows = []
    bias = np.zeros(3)
    odom = np.array([E.START_XY[0], E.START_XY[1], float(env.state['pose'][0, 2])])
    odom_v = np.zeros(2)
    substep_ns = int(round(E.DT_SUB * 1e9))

    with Writer(out, version=9, storage_plugin=plugin) as w:
        conns = {k: w.add_connection(topic, mt, typestore=m.ts)
                 for k, (topic, mt) in TOPICS.items()}
        steps = int(round(seconds / E.DT))
        for k in range(steps):
            action = ctl.act(obs)
            obs, _, term, trunc, info = env.step(action)
            step_ns = T0_NS + (k + 1) * E.SUBSTEPS * substep_ns
            # 80 Hz sensors from the physics substeps.
            for j, sub in enumerate(info['substeps']):
                ns = T0_NS + (k * E.SUBSTEPS + j + 1) * substep_ns
                w_meas = sub['w'][0] + rng.normal(0, cfg.noise_wheel_speed, 4)
                eff = sub['tau_cmd'][0] + rng.normal(0, cfg.noise_effort_frac * E.TAU_MAX, 4)
                js = T['sensor_msgs/msg/JointState'](
                    header=m.header(ns, ''), name=JOINT_NAMES,
                    position=sub['wheel_angle'][0].astype(np.float64),
                    velocity=w_meas.astype(np.float64), effort=eff.astype(np.float64))
                w.write(conns['joint_states'], ns, m.ser(js, TOPICS['joint_states'][1]))
                counts['joint_states'] += 1

                roll, pitch = float(sub['roll'][0]), float(sub['pitch'][0])
                yaw = float(sub['pose'][0, 2])
                g = float(env.state['g'][0])
                bias += rng.normal(0, cfg.accel_bias_rw * np.sqrt(E.DT_SUB), 3)
                g_body = g * np.array([-np.sin(pitch), np.sin(roll) * np.cos(pitch),
                                       np.cos(roll) * np.cos(pitch)])
                f = np.array([sub['acc'][0, 0], sub['acc'][0, 1], 0.0]) + g_body + bias \
                    + rng.normal(0, cfg.noise_accel, 3)
                gyro = np.array([0.0, 0.0, sub['vel'][0, 2]]) + rng.normal(0, cfg.noise_gyro, 3)
                imu = T['sensor_msgs/msg/Imu'](
                    header=m.header(ns, 'imu_link'), orientation=m.quat(roll, pitch, yaw),
                    orientation_covariance=np.diag([1e-4, 1e-4, 1e-3]).ravel(),
                    angular_velocity=m.Vector3(x=float(gyro[0]), y=float(gyro[1]),
                                               z=float(gyro[2])),
                    angular_velocity_covariance=np.eye(3).ravel() * cfg.noise_gyro ** 2,
                    linear_acceleration=m.Vector3(x=float(f[0]), y=float(f[1]), z=float(f[2])),
                    linear_acceleration_covariance=np.eye(3).ravel() * cfg.noise_accel ** 2)
                w.write(conns['imu'], ns, m.ser(imu, TOPICS['imu'][1]))
                counts['imu'] += 1

                # Wheel odometry integrated at 80 Hz from measured wheel speeds.
                wl = (w_meas[0] + w_meas[2]) / 2
                wr = (w_meas[1] + w_meas[3]) / 2
                odom_v = np.array([S.WHEEL_RADIUS_M * (wl + wr) / 2,
                                   S.WHEEL_RADIUS_M * (wr - wl) / S.TRACK_WIDTH_M])
                odom += E.DT_SUB * np.array([odom_v[0] * np.cos(odom[2]),
                                             odom_v[0] * np.sin(odom[2]), odom_v[1]])

            od = T['nav_msgs/msg/Odometry'](
                header=m.header(step_ns, 'odom'), child_frame_id='base_link',
                pose=m.PoseWC(pose=m.Pose(position=m.Point(x=float(odom[0]), y=float(odom[1]),
                                                           z=0.0),
                                          orientation=m.quat(0.0, 0.0, float(odom[2]))),
                              covariance=np.zeros(36)),
                twist=m.TwistWC(twist=m.Twist(linear=m.Vector3(x=float(odom_v[0]), y=0.0, z=0.0),
                                              angular=m.Vector3(x=0.0, y=0.0,
                                                                z=float(odom_v[1]))),
                                covariance=np.zeros(36)))
            w.write(conns['odom'], step_ns, m.ser(od, TOPICS['odom'][1]))
            counts['odom'] += 1

            has_fix = bool(info['sensors']['has_fix'][0])
            pose = env.state['pose'][0]
            if has_fix:
                p = pose[:2] + rng.normal(0, POSE_NOISE_M, 2)
                ps = T['geometry_msgs/msg/PoseStamped'](
                    header=m.header(step_ns, FRAME_SANDBOX),
                    pose=m.Pose(position=m.Point(x=float(p[0]), y=float(p[1]), z=0.0),
                                orientation=m.quat(0.0, 0.0, float(pose[2] + rng.normal(
                                    0, YAW_NOISE_RAD)))))
                w.write(conns['truth_pose'], step_ns, m.ser(ps, TOPICS['truth_pose'][1]))
                counts['truth_pose'] += 1

            v_cmd, w_cmd = info['cmd_twist'][0]
            tw = T['geometry_msgs/msg/TwistStamped'](
                header=m.header(step_ns, 'base_link'),
                twist=m.Twist(linear=m.Vector3(x=float(v_cmd), y=0.0, z=0.0),
                              angular=m.Vector3(x=0.0, y=0.0, z=float(w_cmd))))
            w.write(conns['cmd'], step_ns, m.ser(tw, TOPICS['cmd'][1]))
            counts['cmd'] += 1

            vel = env.state['vel'][0]
            slip = S.wheel_slip(env.state['w'][0], vel[0], vel[2])
            side = S.side_mean(slip)
            z = env.state['z'][0]
            rows.append({
                't': round((k + 1) * E.DT, 4), 'stamp_ns': step_ns,
                'x': pose[0], 'y': pose[1], 'yaw': pose[2], 'body_speed': vel[0],
                'lateral_speed': vel[1], 'yaw_rate': vel[2],
                **{f'slip_{n}': slip[i] for i, n in enumerate(S.WHEEL_ORDER)},
                'slip_left': side[0], 'slip_right': side[1],
                **{f'sinkage_{n}': z[i] for i, n in enumerate(S.WHEEL_ORDER)},
                'sinkage_mean': z.mean(), 'has_fix': int(has_fix),
                'cmd_v': v_cmd, 'cmd_w': w_cmd, 'gravity': float(env.state['g'][0]),
            })
            if term[0] or trunc[0]:
                break

    with open(out / 'truth.csv', 'w', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        for r in rows:
            wr.writerow({k: (f'{v:.6g}' if isinstance(v, float) else v) for k, v in r.items()})
    return {'counts': counts, 'truth': rows}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--cell', default='light-loose', choices=list(E.CELLS) + ['random'])
    ap.add_argument('--gravity', type=float, default=9.81)
    ap.add_argument('--seconds', type=float, default=20.0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--storage', default='sqlite3', choices=['sqlite3', 'mcap'])
    ap.add_argument('--force-dropouts', action='store_true')
    ap.add_argument('--overwrite', action='store_true')
    a = ap.parse_args(argv)
    res = write_bag(a.out, None if a.cell == 'random' else a.cell, a.seconds, a.seed, a.gravity,
                    a.storage, a.force_dropouts, a.overwrite)
    print(res['counts'])


if __name__ == '__main__':
    main()
