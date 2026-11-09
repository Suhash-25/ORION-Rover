"""Rolling observation-action history fed to the adaptation module (pure NumPy).

Layout (shared with orion_policy, see the ONNX contract): HISTORY_LEN rows, oldest first;
row k = [obs_k (OBS_DIM floats from build_observation), a_(k-1) (ACTION_DIM floats in
[-1, 1], the action the policy output on the previous control step)]. Rows before the first
observation of an episode are zero. Each control step: push(obs_t, a_(t-1)), run the adapter
on .buffer, run the policy on (obs_t, latent), remember a_t for the next push.
"""

from __future__ import annotations

import numpy as np

from orion_learning import obs_spec as S

WIDTH = S.OBS_DIM + S.ACTION_DIM


class History:
    """History buffers of N rovers, float32 array (N, HISTORY_LEN, OBS_DIM + ACTION_DIM)."""

    def __init__(self, num_envs: int = 1):
        self.buffer = np.zeros((num_envs, S.HISTORY_LEN, WIDTH), dtype=np.float32)

    def reset(self, mask=None) -> None:
        """Zero the history of the rovers in mask (all if None), e.g. at episode start."""
        self.buffer[slice(None) if mask is None else mask] = 0.0

    def push(self, obs, prev_action) -> np.ndarray:
        """Append one row per rover (dropping the oldest) and return the buffer."""
        row = np.concatenate([np.atleast_2d(obs), np.atleast_2d(prev_action)], axis=-1)
        self.buffer[:, :-1] = self.buffer[:, 1:]
        self.buffer[:, -1] = row
        return self.buffer
