"""Gymnasium single-rover wrapper around the vectorised OrionEnv."""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from orion_learning import obs_spec as S
from orion_learning.env import EnvConfig, OrionEnv


class OrionGymEnv(gym.Env):
    """One rover, standard Gymnasium API. info['privileged'] carries the (10,) teacher input.

    Episodes end with terminated (goal, belly-out, hard-stuck, fence) or truncated (60 s).
    """

    metadata = {'render_modes': []}

    def __init__(self, config: EnvConfig | None = None):
        super().__init__()
        self.observation_space = gym.spaces.Box(-S.OBS_CLIP, S.OBS_CLIP, (S.OBS_DIM,),
                                                dtype=np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (S.ACTION_DIM,), dtype=np.float32)
        self._env = OrionEnv(1, config, seed=0, auto_reset=False)
        self._needs_reset = True

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is None:
            # Derive the next episode's seed from gymnasium's np_random (itself seeded once).
            seed = int(self.np_random.integers(0, 2 ** 31 - 1))
        obs, info = self._env.reset(seed=seed)
        self._needs_reset = False
        return obs[0], self._unbatch(info)

    def step(self, action):
        if self._needs_reset:
            raise RuntimeError('call reset() before step()')
        obs, reward, terminated, truncated, info = self._env.step(np.asarray(action)[None])
        self._needs_reset = bool(terminated[0] or truncated[0])
        return obs[0], float(reward[0]), bool(terminated[0]), bool(truncated[0]), \
            self._unbatch(info)

    @staticmethod
    def _unbatch(info: dict) -> dict:
        out = {}
        for k, v in info.items():
            if isinstance(v, dict):
                out[k] = OrionGymEnv._unbatch(v)
            elif isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == 1:
                out[k] = v[0]
            elif isinstance(v, list):
                out[k] = [OrionGymEnv._unbatch(x) for x in v]
            else:
                out[k] = v
        return out
