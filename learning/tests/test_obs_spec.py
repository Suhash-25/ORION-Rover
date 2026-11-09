import numpy as np
import pytest

from orion_learning import obs_spec as S


def _obs(**kw):
    base = dict(prev_cmd=(0.0, 0.0), wheel_speed=(0, 0, 0, 0), wheel_effort=(0, 0, 0, 0),
                effort_max=3.0, body_accel_xy=(0, 0), yaw_rate=0.0, roll=0.0, pitch=0.0,
                body_speed=0.0, speed_confidence=1.0, slip_side=(0, 0), sinkage_proxy=0.0)
    base.update(kw)
    return S.build_observation(**base)


def test_dimensions():
    assert S.OBS_DIM == 20
    assert S.PRIVILEGED_DIM == 10
    assert _obs().shape == (20,) and _obs().dtype == np.float32


def test_fields_land_where_the_slices_say():
    o = _obs(body_speed=0.25, slip_side=(0.3, -0.2), sinkage_proxy=0.7)
    assert o[S.field_slice('body_speed')][0] == pytest.approx(0.5)
    assert list(o[S.field_slice('slip_side')]) == pytest.approx([0.3, -0.2])
    assert o[S.field_slice('sinkage_proxy')][0] == pytest.approx(0.7)


def test_clipping():
    assert _obs(wheel_speed=(100, 0, 0, 0)).max() == pytest.approx(S.OBS_CLIP)


def test_slip_signs_and_floor():
    # Driving straight at 0.3 m/s with wheels turning for 0.3 m/s: no slip.
    w = 0.3 / S.WHEEL_RADIUS_M
    assert np.allclose(S.wheel_slip([w] * 4, 0.3, 0.0), 0.0)
    # Wheels spin, body stopped: full positive slip.
    assert np.allclose(S.wheel_slip([w] * 4, 0.0, 0.0), 1.0)
    # Body moves, wheels locked: full negative slip (skid).
    assert np.allclose(S.wheel_slip([0] * 4, 0.3, 0.0), -1.0)
    # Standstill noise is not amplified past the 0.02 m/s floor.
    tiny = 0.001 / S.WHEEL_RADIUS_M
    assert np.all(np.abs(S.wheel_slip([tiny] * 4, 0.0, 0.0)) <= 0.05 + 1e-9)


def test_turning_uses_side_speeds():
    # Pure rotation at 1 rad/s: left side moves backward, right forward, no slip.
    v_side = S.TRACK_WIDTH_M / 2
    w = v_side / S.WHEEL_RADIUS_M
    assert np.allclose(S.wheel_slip([-w, w, -w, w], 0.0, 1.0), 0.0)


def test_action_mapping_and_rate_limit():
    assert S.action_to_twist([1, -1]) == (0.5, -1.0)
    assert S.action_to_twist([5, 0]) == (0.5, 0.0)
    v, w = S.rate_limit((0.0, 0.0), (0.5, 1.0), dt=0.05)
    assert v == pytest.approx(0.05) and w == pytest.approx(0.1)


def test_hash_is_stable_and_short():
    assert len(S.OBS_SPEC_HASH) == 16
    assert S.describe()['obs_spec_hash'] == S.OBS_SPEC_HASH
