"""
IMU acceleration check against camera truth (docs/BRINGUP.md step 10, council m9).

    ros2 run orion_experiments imu_check <bag> [--static-s 10] [--window-s 2]

The bag holds /imu/data and /orion/ground_truth_pose for: the rover standing still for at
least static_s, then driving straight (about 2 m at 0.2 m/s). The terrain estimator predicts
body speed with the IMU's forward acceleration between camera fixes, so a scaled or biased a_x
shows up as speed drift in every fix dropout (Gazebo's IMU integrates 23 % low).

- static bias b = mean a_x over the standing window (s 0 .. static_s of the bag), which must
  show < 5 mm of camera motion;
- start of motion t0 = first camera speed > 0.02 m/s;
- integral ratio = integral of (a_x - b) dt over [t0 - 0.5 s, t0 + window_s] divided by the
  camera speed at t0 + window_s (central difference over +-0.25 s). The window covers the
  acceleration to cruise speed; it is short so a bias error of 0.01 m/s^2 costs < 0.03 m/s.
Pass: |ratio - 1| < 0.05.
"""

from __future__ import annotations

import argparse
import math
import sys

import numpy as np

from .analysis import find_msg_dir
from .analysis import make_typestore
from .analysis import read_bag

PASS_TOLERANCE = 0.05


def _speed_at(t, s, t_query, half=0.25):
    """Camera speed (m/s) at t_query by central difference over +-half s of track s(t)."""
    return (np.interp(t_query + half, t, s) - np.interp(t_query - half, t, s)) / (2 * half)


def imu_integral_check(imu, poses, static_s: float = 10.0, window_s: float = 2.0) -> dict:
    """
    Compare integrated IMU forward acceleration with camera speed.

    imu: [(t, a_x m/s^2)]; poses: [(t, x, y)] in m. Returns bias, its std, ratio, error and
    pass; raises ValueError when the bag does not contain the manoeuvre.
    """
    if len(imu) < 2 or len(poses) < 2:
        raise ValueError('need /imu/data and /orion/ground_truth_pose')
    ta, ax = np.array(imu, float).T
    tp, x, y = np.array(poses, float).T
    t_first = min(ta[0], tp[0])
    static_imu = ta <= t_first + static_s
    static_pose = tp <= t_first + static_s
    drift = np.hypot(x[static_pose] - x[0], y[static_pose] - y[0]).max()
    if drift > 0.005:
        raise ValueError(f'rover moved {drift * 1000:.0f} mm during the first {static_s:g} s; '
                         'it must stand still for the bias')
    moving = np.hypot(x - x[0], y - y[0]) > 0.05
    if not moving.any():
        raise ValueError('the camera never saw the rover drive')
    i = int(np.argmax(moving))
    heading = math.atan2(y[i] - y[0], x[i] - x[0])
    s = (x - x[0]) * math.cos(heading) + (y - y[0]) * math.sin(heading)
    v_cam = _speed_at(tp, s, tp)
    t0 = float(tp[np.argmax((v_cam > 0.02) & (tp > t_first + static_s))])
    t1 = t0 + window_s
    if t1 + 0.25 > tp[-1]:
        raise ValueError('the run ends before the integration window does')
    bias = float(ax[static_imu].mean())
    sel = (ta >= t0 - 0.5) & (ta <= t1)
    a, t = ax[sel] - bias, ta[sel]
    dv_imu = float(np.sum(0.5 * (a[1:] + a[:-1]) * np.diff(t)))
    v_end = float(_speed_at(tp, s, t1))
    ratio = dv_imu / v_end
    return {'bias': bias, 'bias_std': float(ax[static_imu].std()), 'motion_start': t0 - t_first,
            'camera_speed': v_end, 'imu_speed': dv_imu, 'ratio': ratio,
            'error': abs(ratio - 1.0), 'distance_m': float(s[-1]),
            'passed': abs(ratio - 1.0) < PASS_TOLERANCE}


def read_imu_and_poses(bag_path: str):
    """([(t, a_x)], [(t, x, y)]) from a bag with /imu/data and /orion/ground_truth_pose."""
    msgs = read_bag(bag_path, make_typestore(find_msg_dir()),
                    ('/imu/data', '/orion/ground_truth_pose'))
    imu = [(t, m.linear_acceleration.x) for t, m in msgs['/imu/data']]
    poses = [(t, m.pose.position.x, m.pose.position.y)
             for t, m in msgs['/orion/ground_truth_pose']]
    return imu, poses


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description='IMU forward acceleration vs camera truth.')
    parser.add_argument('bag')
    parser.add_argument('--static-s', type=float, default=10.0,
                        help='standing-still time at the start of the bag (s)')
    parser.add_argument('--window-s', type=float, default=2.0,
                        help='integration window after the start of motion (s)')
    args = parser.parse_args(argv)
    r = imu_integral_check(*read_imu_and_poses(args.bag), args.static_s, args.window_s)
    print(f'static a_x bias {r["bias"]:+.3f} m/s^2 (std {r["bias_std"]:.3f}); motion at '
          f'{r["motion_start"]:.1f} s; camera {r["camera_speed"]:.3f} m/s, integrated IMU '
          f'{r["imu_speed"]:.3f} m/s: ratio {r["ratio"]:.3f}, error {r["error"]:.1%} '
          f'(pass < {PASS_TOLERANCE:.0%}); run length {r["distance_m"]:.2f} m')
    print('PASS' if r['passed'] else 'FAIL')
    return 0 if r['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
