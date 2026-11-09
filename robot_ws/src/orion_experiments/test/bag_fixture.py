"""Write synthetic trial bags (rosbag2, mcap) + metadata.json with known metrics, ROS-free."""

import json
import math
import os
from pathlib import Path

from orion_experiments.analysis import find_msg_dir
from orion_experiments.analysis import make_typestore

T0 = 1000.0           # s, trial start on the recording clock
YAW0 = 0.5            # rad, heading at the start pose
RATE = 20.0           # Hz for every synthetic topic


def _stamp(ts, t):
    sec = int(t)
    return ts.types['builtin_interfaces/msg/Time'](sec=sec, nanosec=int(round((t - sec) * 1e9)))


def _header(ts, t, frame=''):
    return ts.types['std_msgs/msg/Header'](stamp=_stamp(ts, t), frame_id=frame)


def write_trial(root, trial_id, *, cell='light-packed', controller='B0', outcome='SUCCESS',
                elapsed=20.0, speed=0.075, stuck=((5.0, 8.0),), escape=(5.5, 6.5),
                risk=None, cone=(10.0, 11.0, 12.0), stop_at=None, hard_stuck_at=None):
    """
    Write one trial with known metrics.

    Straight drive at speed along YAW0 (standing still from stop_at on), slip_side (0.2, 0.4),
    sinkage 0.1, 4 wheels at 2 rad/s and 1 N.m, FLAG_STUCK in each (start, end) of stuck,
    one escape, and HALT with FLAG_HARD_STUCK from hard_stuck_at (times after start, s).
    risk(t_rel) gives PolicyStatus.entrapment_risk (NaN when None).
    """
    import numpy as np
    from rosbags.rosbag2 import StoragePlugin
    from rosbags.rosbag2 import Writer

    ts = make_typestore(find_msg_dir())
    T = ts.types
    trial_dir = os.path.join(root, trial_id)
    os.makedirs(trial_dir)
    t1 = T0 + elapsed
    with Writer(Path(trial_dir) / 'bag', version=9, storage_plugin=StoragePlugin.MCAP) as w:
        conns = {
            topic: w.add_connection(topic, msgtype, typestore=ts)
            for topic, msgtype in [
                ('/orion/ground_truth_pose', 'geometry_msgs/msg/PoseStamped'),
                ('/orion/terrain_state', 'orion_msgs/msg/TerrainState'),
                ('/orion/safety_status', 'orion_msgs/msg/SafetyStatus'),
                ('/orion/policy_status', 'orion_msgs/msg/PolicyStatus'),
                ('/joint_states', 'sensor_msgs/msg/JointState')]}

        def write(topic, t, msg):
            w.write(conns[topic], int(round(t * 1e9)), ts.serialize_cdr(msg, msg.__msgtype__))

        # 1 s before the start and 1 s after the end: must be cut by the analysis window
        for k in range(int((elapsed + 2.0) * RATE) + 1):
            t = T0 - 1.0 + k / RATE
            rel = min(max(t - T0, 0.0), elapsed, stop_at if stop_at is not None else elapsed)
            d = speed * rel + (5.0 if t < T0 or t > t1 else 0.0)   # junk outside the window
            pose = T['geometry_msgs/msg/Pose'](
                position=T['geometry_msgs/msg/Point'](x=d * math.cos(YAW0),
                                                      y=d * math.sin(YAW0), z=0.0),
                orientation=T['geometry_msgs/msg/Quaternion'](
                    x=0.0, y=0.0, z=math.sin(YAW0 / 2), w=math.cos(YAW0 / 2)))
            write('/orion/ground_truth_pose', t,
                  T['geometry_msgs/msg/PoseStamped'](header=_header(ts, t, 'sandbox'),
                                                     pose=pose))
            slip = (0.2, 0.4) if T0 <= t <= t1 else (1.0, 1.0)
            write('/orion/terrain_state', t, T['orion_msgs/msg/TerrainState'](
                header=_header(ts, t), slip=np.zeros(4, np.float32),
                slip_side=np.array(slip, np.float32), effort_residual=np.zeros(4, np.float32),
                sinkage_proxy=0.1, body_yaw_rate=0.0, body_speed_source=3,
                body_speed=speed if stop_at is None or t - T0 < stop_at else 0.0,
                effort_source=3, confidence=0.9, time_since_fix=0.0))
            is_stuck = any(a <= t - T0 < b for a, b in stuck or ())
            halted = hard_stuck_at is not None and t - T0 >= hard_stuck_at
            write('/orion/safety_status', t, T['orion_msgs/msg/SafetyStatus'](
                header=_header(ts, t), state=2 if halted else 0,
                flags=(16 if is_stuck else 0) | (64 if halted else 0),
                max_linear_speed=0.5, max_angular_speed=1.0, max_current=6.0, reason=''))
            in_escape = escape is not None and escape[0] <= t - T0 < escape[1]
            write('/orion/policy_status', t, T['orion_msgs/msg/PolicyStatus'](
                header=_header(ts, t), mode=4 if in_escape else 1,
                entrapment_risk=float('nan') if risk is None else risk(t - T0),
                latent=np.zeros(8, np.float32), policy_sha='', obs_spec_hash='',
                critic_gating=risk is not None))
            write('/joint_states', t, T['sensor_msgs/msg/JointState'](
                header=_header(ts, t), name=['fl', 'fr', 'rl', 'rr'],
                position=np.zeros(4), velocity=np.full(4, 2.0), effort=np.full(4, 1.0)))
    meta = {'trial_id': trial_id, 'cell': cell, 'controller': controller,
            'duration_s': 60.0, 'outcome': outcome, 'start_ros_s': T0, 'end_ros_s': t1,
            'elapsed_s': elapsed, 'cone_depths_mm': list(cone), 'ballast_kg': 0.0,
            'operator': 'test'}
    with open(os.path.join(trial_dir, 'metadata.json'), 'w') as f:
        json.dump(meta, f)
    return trial_dir
