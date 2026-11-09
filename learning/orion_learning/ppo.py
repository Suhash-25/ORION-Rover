"""Compact PPO (clipped surrogate, GAE, value loss, entropy bonus) for the teacher and B2.

The teacher's policy gets latent z = mu(privileged); B2's policy gets no latent (its graph
multiplies the latent input by 0). Both use an asymmetric value function V(obs, privileged),
which is training-only and never exported.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Callable

import numpy as np
import torch
from torch import nn

from orion_learning import obs_spec as S
from orion_learning.env import EnvConfig, OrionEnv
from orion_learning.nets import Policy, PrivEncoder, Value


@dataclass
class PPOConfig:
    num_envs: int = 64
    rollout_steps: int = 128
    iterations: int = 100
    epochs: int = 4
    minibatches: int = 4
    lr: float = 3e-4
    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    hidden: list[int] = field(default_factory=lambda: [128, 128])
    encoder_hidden: list[int] = field(default_factory=lambda: [64])
    init_log_std: float = -0.5


class ActorCritic(nn.Module):
    """Policy + (optional) privileged encoder + asymmetric value function."""

    def __init__(self, cfg: PPOConfig, use_latent: bool):
        super().__init__()
        self.use_latent = use_latent
        self.encoder = PrivEncoder(cfg.encoder_hidden) if use_latent else None
        self.policy = Policy(cfg.hidden, cfg.init_log_std, use_latent=use_latent)
        self.value = Value(cfg.hidden)

    def latent(self, priv: torch.Tensor) -> torch.Tensor:
        if self.encoder is None:
            return torch.zeros(priv.shape[:-1] + (S.LATENT_DIM,), dtype=priv.dtype)
        return self.encoder(priv)

    @torch.no_grad()
    def update_normalisers(self, obs: torch.Tensor, priv: torch.Tensor) -> None:
        self.policy.obs_norm.update(obs)
        self.value.obs_norm.update(obs)
        self.value.priv_norm.update(priv)
        if self.encoder is not None:
            self.encoder.norm.update(priv)


class EpisodeStats:
    """Accumulates outcome counts of finished episodes."""

    KEYS = ('success', 'belly_out', 'hard_stuck', 'out_of_fence')

    def __init__(self):
        self.returns, self.lengths, self.eff_speed = [], [], []
        self.counts = dict.fromkeys(self.KEYS, 0)

    def add(self, info: dict, done: np.ndarray) -> None:
        if not done.any():
            return
        self.returns += list(info['episode_return'][done])
        self.lengths += list(info['episode_length'][done])
        self.eff_speed += list(info['effective_speed'][done])
        for k in self.KEYS:
            self.counts[k] += int(np.sum(info[k][done]))

    def summary(self) -> dict[str, float]:
        n = len(self.returns)
        out = {'episodes': float(n)}
        if n:
            out['episode_return'] = float(np.mean(self.returns))
            out['episode_length'] = float(np.mean(self.lengths))
            out['effective_speed'] = float(np.mean(self.eff_speed))
            for k in self.KEYS:
                out[f'rate_{k}'] = self.counts[k] / n
        return out


def _t(x) -> torch.Tensor:
    return torch.as_tensor(np.asarray(x), dtype=torch.float32)


def train_ppo(env_cfg: EnvConfig, cfg: PPOConfig, seed: int, use_latent: bool,
              log: Callable[[int, dict], None]) -> ActorCritic:
    """Train a PPO agent; calls log(iteration, metrics) once per iteration."""
    torch.manual_seed(seed)
    ac = ActorCritic(cfg, use_latent)
    opt = torch.optim.Adam(ac.parameters(), lr=cfg.lr)
    env = OrionEnv(cfg.num_envs, env_cfg, seed=seed)
    obs, info = env.reset()
    priv = info['privileged']
    ac.update_normalisers(_t(obs), _t(priv))   # first statistics from the initial states
    T, N = cfg.rollout_steps, cfg.num_envs

    for it in range(cfg.iterations):
        t0 = time.time()
        b_obs = np.zeros((T, N, S.OBS_DIM), np.float32)
        b_priv = np.zeros((T, N, S.PRIVILEGED_DIM), np.float32)
        b_act = np.zeros((T, N, S.ACTION_DIM), np.float32)
        b_logp = np.zeros((T, N), np.float32)
        b_val = np.zeros((T + 1, N), np.float32)
        b_rew = np.zeros((T, N), np.float32)
        b_done = np.zeros((T, N), np.float32)
        stats = EpisodeStats()
        for t in range(T):
            with torch.no_grad():
                o, p = _t(obs), _t(priv)
                dist = ac.policy.dist(o, ac.latent(p))
                a = dist.sample()
                b_logp[t] = dist.log_prob(a).sum(-1).numpy()
                b_val[t] = ac.value(o, p).numpy()
            b_obs[t], b_priv[t], b_act[t] = obs, priv, a.numpy()
            obs, rew, term, trunc, info = env.step(a.numpy())
            rew = rew.astype(np.float32)
            if trunc.any():   # time-limit: bootstrap from the value of the final state
                with torch.no_grad():
                    v_final = ac.value(_t(info['final_obs']), _t(info['final_privileged']))
                rew[trunc] += cfg.gamma * v_final.numpy()[trunc]
            done = term | trunc
            stats.add(info, done)
            b_rew[t], b_done[t] = rew, done
            priv = info['privileged']
        with torch.no_grad():
            b_val[T] = ac.value(_t(obs), _t(priv)).numpy()

        # Generalised advantage estimation.
        adv = np.zeros((T, N), np.float32)
        last = np.zeros(N, np.float32)
        for t in reversed(range(T)):
            nonterm = 1.0 - b_done[t]
            delta = b_rew[t] + cfg.gamma * b_val[t + 1] * nonterm - b_val[t]
            last = delta + cfg.gamma * cfg.lam * nonterm * last
            adv[t] = last
        ret = adv + b_val[:T]

        flat = lambda x: _t(x.reshape(T * N, *x.shape[2:]))  # noqa: E731
        f_obs, f_priv, f_act = flat(b_obs), flat(b_priv), flat(b_act)
        f_logp, f_adv, f_ret = flat(b_logp), flat(adv), flat(ret)
        mb = max(1, (T * N) // cfg.minibatches)
        kls, pls, vls, ents, clipfrac = [], [], [], [], []
        for _ in range(cfg.epochs):
            perm = torch.randperm(T * N)
            for i in range(0, T * N, mb):
                idx = perm[i:i + mb]
                dist = ac.policy.dist(f_obs[idx], ac.latent(f_priv[idx]))
                logp = dist.log_prob(f_act[idx]).sum(-1)
                ratio = torch.exp(logp - f_logp[idx])
                a_mb = f_adv[idx]
                a_mb = (a_mb - a_mb.mean()) / (a_mb.std() + 1e-8)
                pg = -torch.min(ratio * a_mb,
                                torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip) * a_mb).mean()
                v_loss = 0.5 * ((ac.value(f_obs[idx], f_priv[idx]) - f_ret[idx]) ** 2).mean()
                ent = dist.entropy().sum(-1).mean()
                loss = pg + cfg.vf_coef * v_loss - cfg.ent_coef * ent
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(ac.parameters(), cfg.max_grad_norm)
                opt.step()
                with torch.no_grad():
                    kls.append(float(((ratio - 1) - torch.log(ratio)).mean()))
                    clipfrac.append(float(((ratio - 1).abs() > cfg.clip).float().mean()))
                pls.append(pg.item())
                vls.append(v_loss.item())
                ents.append(ent.item())
        ac.update_normalisers(f_obs, f_priv)

        metrics = {'policy_loss': np.mean(pls), 'value_loss': np.mean(vls),
                   'entropy': np.mean(ents), 'approx_kl': np.mean(kls),
                   'clip_frac': np.mean(clipfrac), 'mean_reward': float(b_rew.mean()),
                   'steps_per_s': T * N / (time.time() - t0)}
        metrics.update(stats.summary())
        log(it, metrics)
    return ac
