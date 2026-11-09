"""Rolling observation-action history fed to the adaptation module (pure NumPy).

The ONE implementation: training (env rollouts, DAgger, critic labels) and the rover runtime
(orion_policy) both use this class, so the adapter sees the same input in both.

Convention (obs_spec.HISTORY_CONVENTION, part of OBS_SPEC_HASH): HISTORY_LEN rows, oldest
first; row k = [obs_k (OBS_DIM floats from build_observation), a_(k-1) (ACTION_DIM floats)],
where a_(k-1) is the command actually EXECUTED on the previous control step in action units
(v / 0.5 m/s, w / 1.0 rad/s): rate-limited, and on the rover also scaled to the SafetyStatus
limits. That is exactly the prev_cmd field of obs_k, so push() takes it from there and no
caller can put the raw policy output in its place. Rows before the first observation of an
episode are zero. Each control step: push(obs_t) BEFORE inference, run the adapter on the
buffer (which therefore includes obs_t), then the policy on (obs_t, latent).
"""

from __future__ import annotations

import warnings

import numpy as np

from orion_learning import obs_spec as S

WIDTH = S.OBS_DIM + S.ACTION_DIM
_PREV_CMD = S.field_slice('prev_cmd')


def executed_action(obs) -> np.ndarray:
    """The previous step's executed command in action units, read from obs prev_cmd (float32)."""
    return np.clip(np.asarray(obs, dtype=np.float32)[..., _PREV_CMD], -1.0, 1.0)


class History:
    """History buffers of N rovers, float32 array (N, HISTORY_LEN, OBS_DIM + ACTION_DIM)."""

    _warned = False      # the deprecated prev_action argument is reported once per process

    def __init__(self, num_envs: int = 1):
        self.buffer = np.zeros((num_envs, S.HISTORY_LEN, WIDTH), dtype=np.float32)

    def reset(self, mask=None) -> None:
        """Zero the history of the rovers in mask (all if None), e.g. at episode start."""
        self.buffer[slice(None) if mask is None else mask] = 0.0

    def push(self, obs, prev_action=None) -> np.ndarray:
        """Append [obs, executed_action(obs)] per rover (dropping the oldest); return the buffer.

        prev_action is accepted only for callers written before the convention was fixed and is
        IGNORED (with a DeprecationWarning): the executed command comes from obs. TEMPORARY until
        student.py, risk.py and evaluate.py call push(obs).
        """
        if prev_action is not None and not History._warned:
            History._warned = True
            warnings.warn('History.push(obs, prev_action): prev_action is ignored, the row holds '
                          'the executed command obs[prev_cmd]; call push(obs)',
                          DeprecationWarning, stacklevel=2)
        obs = np.atleast_2d(np.asarray(obs, dtype=np.float32))
        if obs.shape != (len(self.buffer), S.OBS_DIM):
            raise ValueError(f'obs has shape {obs.shape}, expected ({len(self.buffer)}, '
                             f'{S.OBS_DIM})')
        self.buffer[:, :-1] = self.buffer[:, 1:]
        self.buffer[:, -1, :S.OBS_DIM] = obs
        self.buffer[:, -1, S.OBS_DIM:] = executed_action(obs)
        return self.buffer
