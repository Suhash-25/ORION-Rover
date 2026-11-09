"""History convention (council B2): row k = [obs_k, executed a_(k-1)], pushed before inference."""

import numpy as np
import pytest

from orion_learning import env as E
from orion_learning import obs_spec as S
from orion_learning.history import executed_action, History


def test_rows_are_obs_and_executed_command_oldest_first_zero_padded():
    h = History(2)
    obs = np.zeros((2, S.OBS_DIM), np.float32)
    obs[:, S.field_slice('prev_cmd')] = [[0.4, -0.2], [0.0, 0.1]]
    obs[:, 5] = [7.0, 8.0]
    buf = h.push(obs)
    assert buf.shape == (2, S.HISTORY_LEN, S.OBS_DIM + S.ACTION_DIM)
    assert not buf[:, :-1].any()
    assert np.array_equal(buf[:, -1, :S.OBS_DIM], obs)
    assert np.array_equal(buf[:, -1, S.OBS_DIM:], obs[:, S.field_slice('prev_cmd')])
    obs2 = obs + 1.0
    h.push(obs2)
    assert np.array_equal(h.buffer[:, -2, :S.OBS_DIM], obs)
    h.reset(np.array([True, False]))
    assert not h.buffer[0].any() and h.buffer[1].any()


def test_push_takes_no_action_argument():
    """The executed command comes from obs only; a raw policy action cannot be pushed."""
    h = History(1)
    with pytest.raises(TypeError):
        h.push(np.zeros(S.OBS_DIM, np.float32), np.array([0.9, -0.9]))


def test_env_history_action_is_the_rate_limited_command_not_the_policy_output():
    """In the env 'executed' = st['cmd'] in action units, which lags a step input."""
    env = E.OrionEnv(1, E.cell_config('light-packed'), seed=0, auto_reset=False)
    obs, _ = env.reset()
    h = History(1)
    h.push(obs)
    raw = np.array([[1.0, 0.5]], np.float32)
    obs, *_, info = env.step(raw)
    row = h.push(obs)[0, -1]
    expect = info['cmd_twist'][0] / [S.MAX_LINEAR_SPEED, S.MAX_ANGULAR_SPEED]
    assert np.allclose(row[S.OBS_DIM:], expect)
    assert np.allclose(row[S.OBS_DIM:], [0.1, 0.1])        # 1 m/s^2, 2 rad/s^2 over 0.05 s
    assert not np.allclose(row[S.OBS_DIM:], raw)
    assert np.array_equal(executed_action(obs), row[None, S.OBS_DIM:])


def test_wrong_obs_shape_is_rejected():
    with pytest.raises(ValueError):
        History(2).push(np.zeros(S.OBS_DIM))
