"""
Fit the flat-ground effort model from a rosbag2 of packed-light driving.

    ros2 run orion_terrain fit_effort_model <bag_dir> --i-stall 9.5 [-o effort_model.yaml]
    ros2 run orion_terrain fit_effort_model <bag_dir> --sim --i-ref 2.0

Reads /joint_states and /imu/data (and /orion/rover_health for the effort source) with the
pure-Python `rosbags` library, so it also runs without ROS. Drive about 5 min on packed sand
with the light rover: varied speeds both directions, accelerations, turns and, if the
sandbox allows, some slope (docs/SPEC.md section 4).

I_ref = I_stall - I_free. I_stall comes from the stall test (wheel blocked, full command);
I_free defaults to the fitted flat-ground effort at --free-speed.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys
from typing import Iterable, Iterator, Optional, Sequence

import numpy as np
from orion_learning.obs_spec import WHEEL_ORDER, WHEEL_RADIUS_M

from orion_terrain.effort_model import (EFFORT_UNITS, EffortModel, EffortSource,
                                        fit_coefficients)
from orion_terrain.estimator import msg_get, msg_quat, stamp_seconds
from orion_terrain.filters import WheelAccel
from orion_terrain.imu import roll_pitch
from orion_terrain.params import DEFAULT_JOINT_NAMES, DEFAULT_TOPICS, TerrainParams


def read_bag(bag_dir: str, topics: Sequence[str]) -> Iterator[tuple[float, str, object]]:
    """Yield (t s, topic, msg) from a rosbag2 directory; t = header stamp, else bag time."""
    from rosbags.highlevel import AnyReader, AnyReaderError
    from rosbags.typesys import get_typestore, Stores

    try:
        with AnyReader([Path(bag_dir)],
                       default_typestore=get_typestore(Stores.ROS2_JAZZY)) as reader:
            conns = [c for c in reader.connections if c.topic in topics]
            for conn, t_ns, raw in reader.messages(connections=conns):
                msg = reader.deserialize(raw, conn.msgtype)
                yield stamp_seconds(msg, t_ns * 1e-9), conn.topic, msg
    except AnyReaderError as e:
        raise ValueError(f'cannot read bag {bag_dir}: {e} (a recorder killed with SIGKILL '
                         'leaves an unreadable file: `ros2 bag convert` it first)') from e


def extract_samples(messages: Iterable[tuple[float, str, object]],
                    joint_names: Sequence[str] = DEFAULT_JOINT_NAMES,
                    accel_tau_s: float = TerrainParams.wheel_accel_tau_s,
                    topics=DEFAULT_TOPICS) -> dict:
    """
    Regression data in WHEEL_ORDER, built with the runtime's own dw/dt estimator.

    Returns arrays w, dwdt, effort (N, 4), pitch and yaw_rate (N,) (latest IMU sample) and
    the set of current_sensed values seen in RoverHealth. Joint samples before the first IMU
    orientation are skipped.
    """
    wheel_accel = WheelAccel(accel_tau_s)
    pitch: Optional[float] = None
    yaw_rate = 0.0
    rows = {'w': [], 'dwdt': [], 'effort': [], 'pitch': [], 'yaw_rate': []}
    current_sensed = set()
    for t, topic, msg in sorted(messages, key=lambda m: m[0]):
        if topic == topics['imu']:
            try:
                pitch = roll_pitch(msg_quat(msg, 'orientation'))[1]
            except ValueError:
                continue
            yaw_rate = float(msg_get(msg, 'angular_velocity.z'))
        elif topic == topics['rover_health']:
            current_sensed.add(bool(msg_get(msg, 'current_sensed')))
        elif topic == topics['joint_states']:
            names = list(msg_get(msg, 'name'))
            effort = msg_get(msg, 'effort')
            if any(j not in names for j in joint_names) or len(effort) != len(names):
                continue
            idx = [names.index(j) for j in joint_names]
            w = np.asarray(msg_get(msg, 'velocity'), dtype=np.float64)[idx]
            dwdt = wheel_accel.update(t, w).copy()
            if pitch is None:
                continue
            rows['w'].append(w)
            rows['dwdt'].append(dwdt)
            rows['effort'].append(np.asarray(effort, dtype=np.float64)[idx])
            rows['pitch'].append(pitch)
            rows['yaw_rate'].append(yaw_rate)
    if not rows['w']:
        raise ValueError('no usable /joint_states (with effort) after the first /imu/data')
    out = {k: np.asarray(v) for k, v in rows.items()}
    out['current_sensed'] = current_sensed
    return out


def git_sha(cwd: Optional[str] = None) -> str:
    """
    Short SHA of HEAD in cwd (+ '-dirty' with uncommitted changes); 'unknown' outside git.

    Run the tool from the repository checkout that is deployed on the rover.
    """
    def run(*args):
        return subprocess.run(['git', *args], cwd=cwd, capture_output=True, text=True,
                              check=False)
    head = run('rev-parse', '--short=12', 'HEAD')
    if head.returncode != 0:
        return 'unknown'
    dirty = run('diff', '--quiet', 'HEAD').returncode != 0
    return head.stdout.strip() + ('-dirty' if dirty else '')


def fit_model(samples: dict, *, sim: bool, effort_source: Optional[EffortSource],
              i_ref: Optional[float], i_stall: Optional[float], i_free: Optional[float],
              free_speed_mps: float, deadband_radps: float, max_yaw_rate_radps: float,
              source: str, sha: str) -> EffortModel:
    """Least-squares fit plus I_ref and provenance; raises ValueError on unusable input."""
    if sim:
        effort_source = EffortSource.SIM_TORQUE
    elif effort_source is None:
        seen = samples['current_sensed']
        if len(seen) != 1:
            raise ValueError('cannot tell the effort source: pass --effort-source '
                             f'(RoverHealth.current_sensed values in the bag: {sorted(seen)})')
        effort_source = EffortSource.CURRENT if seen.pop() else EffortSource.DUTY_ESTIMATE
    fit = fit_coefficients(samples['w'], samples['dwdt'], samples['pitch'], samples['effort'],
                           samples['yaw_rate'], deadband_radps, max_yaw_rate_radps)
    coefs = fit.coefficients
    if i_ref is None:
        if i_stall is None:
            raise ValueError('I_ref is needed: pass --i-ref, or --i-stall from the stall test')
        if i_free is None:
            w_free = free_speed_mps / WHEEL_RADIUS_M
            i_free = float(np.mean(coefs[:, 0] + coefs[:, 1] * w_free))
        i_ref = i_stall - i_free
    return EffortModel(
        coefficients=coefs, i_ref=float(i_ref), effort_unit=EFFORT_UNITS[effort_source],
        effort_source=effort_source, calibrated=True, deadband_radps=deadband_radps,
        max_yaw_rate_radps=max_yaw_rate_radps, slope_identified=fit.slope_identified,
        git_sha=sha, date=datetime.now(timezone.utc).isoformat(timespec='seconds'),
        source=source, r2=[float(r) for r in fit.r2],
        n_samples=[int(n) for n in fit.n_samples])


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('bag_dir')
    ap.add_argument('-o', '--output', help='default: effort_model.yaml '
                    '(effort_model_sim.yaml with --sim) in the current directory')
    ap.add_argument('--sim', action='store_true', help='Gazebo bag: effort is joint torque')
    ap.add_argument('--effort-source', choices=['CURRENT', 'DUTY_ESTIMATE'],
                    help='default: from RoverHealth.current_sensed in the bag')
    ap.add_argument('--i-ref', type=float, help='I_ref directly (effort unit)')
    ap.add_argument('--i-stall', type=float, help='stall-test effort; I_ref = I_stall - I_free')
    ap.add_argument('--i-free', type=float, help='free-running effort (default: fitted)')
    ap.add_argument('--free-speed', type=float, default=0.3, help='m/s for the fitted I_free')
    ap.add_argument('--deadband', type=float, default=0.5, help='rad/s; slower samples dropped')
    ap.add_argument('--max-yaw-rate', type=float, default=0.2,
                    help='rad/s; samples turning faster are dropped (scrub is not modelled)')
    ap.add_argument('--joint-names', nargs=4, default=list(DEFAULT_JOINT_NAMES),
                    metavar=WHEEL_ORDER)
    ap.add_argument('--git-sha', help='override the detected git SHA')
    args = ap.parse_args(argv)

    topics = [DEFAULT_TOPICS[k] for k in ('joint_states', 'imu', 'rover_health')]
    try:
        samples = extract_samples(read_bag(args.bag_dir, topics), args.joint_names)
        n = len(samples['w'])
        model = fit_model(
            samples, sim=args.sim,
            effort_source=EffortSource[args.effort_source] if args.effort_source else None,
            i_ref=args.i_ref, i_stall=args.i_stall, i_free=args.i_free,
            free_speed_mps=args.free_speed, deadband_radps=args.deadband,
            max_yaw_rate_radps=args.max_yaw_rate,
            source=f'{os.path.abspath(args.bag_dir)} ({n} joint_states samples)',
            sha=args.git_sha or git_sha())
    except ValueError as e:
        print(f'fit_effort_model: {e}', file=sys.stderr)
        return 2
    out = args.output or ('effort_model_sim.yaml' if args.sim else 'effort_model.yaml')
    model.save(out, header='Written by `ros2 run orion_terrain fit_effort_model`; '
               'do not edit by hand.')
    print(f'{"wheel":5} {"c0":>9} {"c1":>9} {"c2":>9} {"c3":>9} {"R^2":>6} {"n":>7}')
    for name, c, r, n in zip(WHEEL_ORDER, model.coefficients, model.r2, model.n_samples):
        print(f'{name:5} {c[0]:9.4f} {c[1]:9.4f} {c[2]:9.4f} {c[3]:9.4f} {r:6.3f} {n:7d}')
    print(f'I_ref = {model.i_ref:.4g} {model.effort_unit}, source {model.effort_source.name}, '
          f'git {model.git_sha} -> {out}')
    if not model.slope_identified:
        print('warning: no slope in the bag, c3 (slope term) set to 0; on slopes the residual '
              'will include the gravity load', file=sys.stderr)
    if min(model.r2) < 0.5:
        print('warning: R^2 < 0.5 on some wheel; the residual will be noisy. Check the bag.',
              file=sys.stderr)
    return 0


if __name__ == '__main__':
    sys.exit(main())
