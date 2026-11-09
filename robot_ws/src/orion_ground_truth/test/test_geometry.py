import math

import numpy as np
from orion_ground_truth.geometry import GroundTruthSolver, invert_tf, make_tf
from orion_ground_truth.geometry import matrix_to_quat, quat_to_matrix, SolverConfig
import pytest


def yaw_quat(yaw):
    return (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))


def rot_x_quat(a):
    return (math.sin(a / 2), 0.0, 0.0, math.cos(a / 2))


def test_quaternion_round_trip():
    rng = np.random.default_rng(0)
    for _ in range(200):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        if q[3] < 0:
            q = -q
        assert np.allclose(matrix_to_quat(quat_to_matrix(q)), q, atol=1e-9)


def test_invert():
    t = make_tf((1, 2, 3), yaw_quat(0.7))
    assert np.allclose(t @ invert_tf(t), np.eye(4))


def _camera_looking_down(height=2.0, cam_yaw=0.3, cam_xy=(0.4, -0.2)):
    # Camera z points down at the floor: rotate pi about x, then yaw.
    sandbox_from_cam = make_tf((cam_xy[0], cam_xy[1], height), rot_x_quat(math.pi))
    sandbox_from_cam = sandbox_from_cam @ make_tf((0, 0, 0), yaw_quat(cam_yaw))
    return sandbox_from_cam


def _detections(sandbox_from_cam, sandbox_from_ref, sandbox_from_rover_tag):
    cam_from_sandbox = invert_tf(sandbox_from_cam)
    return cam_from_sandbox @ sandbox_from_ref, cam_from_sandbox @ sandbox_from_rover_tag


def _solver():
    return GroundTruthSolver(SolverConfig(
        sandbox_from_ref=make_tf((0, 0, 0), (0, 0, 0, 1)),
        tag_to_base=make_tf((0, 0, -0.3), (0, 0, 0, 1))))


def test_recovers_rover_pose_independent_of_camera_placement():
    truth_x, truth_y, truth_yaw = 1.2, 0.7, 2.5
    sandbox_from_tag = make_tf((truth_x, truth_y, 0.3 + 0.12), yaw_quat(truth_yaw))
    for cam in [_camera_looking_down(), _camera_looking_down(2.5, -1.0, (1.0, 1.0))]:
        ref, rover = _detections(cam, make_tf((0, 0, 0), (0, 0, 0, 1)), sandbox_from_tag)
        pose = _solver().solve(1.0, ref, rover)
        assert pose.x == pytest.approx(truth_x, abs=1e-9)
        assert pose.y == pytest.approx(truth_y, abs=1e-9)
        assert pose.yaw == pytest.approx(truth_yaw, abs=1e-9)


def test_rejects_teleport_and_tilt():
    s = _solver()
    cam = _camera_looking_down()
    ident = make_tf((0, 0, 0), (0, 0, 0, 1))
    ref, rover = _detections(cam, ident, make_tf((0.5, 0.5, 0.3), yaw_quat(0)))
    assert s.solve(1.00, ref, rover) is not None
    # 0.5 m in 1/30 s is 15 m/s: a misdetection.
    _, jump = _detections(cam, ident, make_tf((1.0, 0.5, 0.3), yaw_quat(0)))
    assert s.solve(1.0 + 1 / 30, ref, jump) is None
    # Normal motion is accepted: 1 cm in 1/30 s = 0.3 m/s.
    _, ok = _detections(cam, ident, make_tf((0.51, 0.5, 0.3), yaw_quat(0)))
    assert s.solve(1.0 + 2 / 30, ref, ok) is not None
    # Tag seen tilted 60 deg: a flipped / ambiguous detection.
    _, tilted = _detections(cam, ident, make_tf((0.52, 0.5, 0.3), rot_x_quat(1.05)))
    assert s.solve(1.0 + 3 / 30, ref, tilted) is None
    assert s.rejected == 2


def test_long_gap_restarts_motion_check():
    s = _solver()
    cam = _camera_looking_down()
    ident = make_tf((0, 0, 0), (0, 0, 0, 1))
    ref, a = _detections(cam, ident, make_tf((0.0, 0.0, 0.3), yaw_quat(0)))
    _, b = _detections(cam, ident, make_tf((1.5, 0.0, 0.3), yaw_quat(0)))
    assert s.solve(1.0, ref, a) is not None
    assert s.solve(3.0, ref, b) is not None   # tag hidden for 2 s, then reappears far away
