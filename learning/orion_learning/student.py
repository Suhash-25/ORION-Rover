"""Student adaptation module: DAgger-style distillation of the teacher's latent.

The student acts in the env (teacher policy on the student's latent z-hat, with a decaying
probability beta of acting on the teacher's latent instead), the visited histories are
labelled with the teacher's z = mu(privileged) and aggregated, and the TCN adapter is
regressed onto z with MSE. The policy itself is the frozen teacher policy.

The actions sent to the env go through baselines.CriticGate exactly as on the rover (council
M9): the stuck rule triggers the escape library, and a surrogate risk (SurrogateRisk; no
critic exists yet at this stage) drives the probe / retreat / escape modes, so the adapter
sees histories that contain the gated behaviour it will meet at deployment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch

from orion_learning import obs_spec as S
from orion_learning.baselines import CriticGate, GateThresholds
from orion_learning.env import DT, EnvConfig, OrionEnv
from orion_learning.history import History
from orion_learning.nets import TCNAdapter
from orion_learning.ppo import ActorCritic, EpisodeStats


@dataclass
class StudentConfig:
    num_envs: int = 64
    iterations: int = 20
    steps_per_iter: int = 400
    epochs: int = 4
    batch_size: int = 512
    lr: float = 1e-3
    channels: int = 32
    beta0: float = 1.0          # probability of acting on the teacher latent at iteration 0
    max_dataset: int = 200_000  # aggregated samples kept (oldest dropped first)
    gate_p_active: float = 0.3  # SurrogateRisk: probability a rho draw is >= the probe level
    gate_hold_s: tuple[float, float] = (0.5, 3.0)   # SurrogateRisk hold time range, s


class SurrogateRisk:
    """Stand-in risk rho for CriticGate while collecting data before the critic exists.

    Each rover holds a rho for U(hold_s) seconds, then redraws it: 0 (explore) with
    probability 1 - p_active, else U(probe threshold, 1), which spreads the time over the
    probe, retreat and critic-escape bands. Rejected alternative: gating with the critic
    being trained (iterative), which is circular for the student (the critic needs z-hat).
    """

    def __init__(self, num_envs: int, p_active: float, hold_s: tuple[float, float],
                 rng: np.random.Generator, thresholds: GateThresholds = GateThresholds()):
        self.p_active, self.hold_s, self.rng = p_active, tuple(hold_s), rng
        self.lo = thresholds.probe
        self.rho = np.zeros(num_envs)
        self.left = np.zeros(num_envs)       # s until the next redraw

    def reset(self, mask=None) -> None:
        self.left[slice(None) if mask is None else mask] = 0.0

    def __call__(self) -> np.ndarray:
        """Advance one control period and return rho per rover."""
        redraw = self.left <= 0.0
        k = int(redraw.sum())
        if k:
            active = self.rng.random(k) < self.p_active
            self.rho[redraw] = np.where(active, self.rng.uniform(self.lo, 1.0, k), 0.0)
            self.left[redraw] = self.rng.uniform(*self.hold_s, k)
        self.left -= DT
        return self.rho.copy()


def mode_fractions(modes: list[np.ndarray]) -> dict[str, float]:
    """Fraction of rover-steps in each CriticGate mode."""
    m = np.concatenate(modes) if modes else np.zeros(0, np.int64)
    names = ('explore', 'probe', 'retreat', 'escape')
    return {f'frac_{name}': float(np.mean(m == i)) if len(m) else 0.0
            for i, name in enumerate(names)}


def make_adapter(teacher: ActorCritic, channels: int) -> TCNAdapter:
    """Adapter whose obs normaliser is a frozen copy of the teacher policy's."""
    adapter = TCNAdapter(channels)
    adapter.obs_norm.load_state_dict(teacher.policy.obs_norm.state_dict())
    return adapter


def train_student(teacher: ActorCritic, env_cfg: EnvConfig, cfg: StudentConfig, seed: int,
                  log: Callable[[int, dict], None]) -> TCNAdapter:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    adapter = make_adapter(teacher, cfg.channels)
    opt = torch.optim.Adam(adapter.parameters(), lr=cfg.lr)
    env = OrionEnv(cfg.num_envs, env_cfg, seed=seed)
    obs, info = env.reset()
    priv = info['privileged']
    hist = History(cfg.num_envs)
    gate = CriticGate(cfg.num_envs)
    risk = SurrogateRisk(cfg.num_envs, cfg.gate_p_active, cfg.gate_hold_s, rng)
    prev_action = np.zeros((cfg.num_envs, S.ACTION_DIM), np.float32)
    data_h: list[np.ndarray] = []
    data_z: list[np.ndarray] = []

    for it in range(cfg.iterations):
        beta = cfg.beta0 * max(0.0, 1.0 - it / max(1, cfg.iterations - 1))
        stats = EpisodeStats()
        sq_err, modes = [], []
        for _ in range(cfg.steps_per_iter):
            h = hist.push(obs, prev_action)
            with torch.no_grad():
                z_t = teacher.latent(torch.as_tensor(priv))
                z_s = adapter(torch.as_tensor(h))
                use_teacher = torch.as_tensor(rng.random(cfg.num_envs) < beta)[:, None]
                z_act = torch.where(use_teacher, z_t, z_s)
                action = teacher.policy(torch.as_tensor(obs), z_act).numpy()
            action, mode = gate.gate(obs, action, risk())
            modes.append(mode)
            sq_err.append(((z_s - z_t) ** 2).mean(-1).numpy())
            data_h.append(h.copy())
            data_z.append(z_t.numpy())
            obs, _, term, trunc, info = env.step(action)
            done = term | trunc
            stats.add(info, done)
            priv = info['privileged']
            prev_action = action.astype(np.float32)     # the gated action actually sent
            if done.any():
                hist.reset(done)
                gate.reset(done)
                risk.reset(done)
                prev_action[done] = 0.0

        # Keep the most recent max_dataset samples (DAgger aggregation).
        H = np.concatenate(data_h)[-cfg.max_dataset:]
        Z = np.concatenate(data_z)[-cfg.max_dataset:]
        data_h, data_z = [H], [Z]
        Ht, Zt = torch.as_tensor(H), torch.as_tensor(Z)
        losses = []
        for _ in range(cfg.epochs):
            perm = torch.randperm(len(Ht))
            for i in range(0, len(Ht), cfg.batch_size):
                idx = perm[i:i + cfg.batch_size]
                loss = ((adapter(Ht[idx]) - Zt[idx]) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                losses.append(loss.item())
        metrics = {'beta': beta, 'onpolicy_latent_mse': float(np.mean(sq_err)),
                   'train_latent_mse': float(np.mean(losses[-max(1, len(losses) // 4):])),
                   'dataset': float(len(Ht))}
        metrics.update(mode_fractions(modes))
        metrics.update(stats.summary())
        log(it, metrics)
    return adapter
