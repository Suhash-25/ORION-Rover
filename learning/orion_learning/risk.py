"""Entrapment-risk critic (SPEC F1): branched-rollout labels, BCE training, calibration.

rho(t) = P(an entrapment begins within T_h = 10 s of t | obs, history, escape now). An
entrapment is a belly-out (at the time it happens) or an orion_safety ENTRAPPED verdict,
dated at the onset of the stuck episode that ended in it (council M2: the rover's analysis
uses the same anchor, never the HALT time ~23 s after the wheels stop).

Label of a visited state: clone the env state into K = 4 branches; branch b runs escape
manoeuvre b of the library E at once, then B0 (cruise, slip governor, stuck -> next escape),
with its own sensor-noise stream and no time limit, until its outcome for the deadline
t + T_h is decided (at most T_h + 20 s ENTRAPPED window + LABEL_SLACK_S). Label = fraction
of branches with an entrapment dated <= t + T_h. E-then-B0 rather than E back to back: the
stuck rule only fires while the rover drives, so a branch that only plays E could never be
ENTRAPPED.

States are visited by the student acting through baselines.CriticGate (stuck -> escape,
plus the SurrogateRisk probe / retreat / escape modes, council M9). The critic sees
(obs, z-hat) only, is trained with BCE on these soft labels, then temperature-scaled on a
held-out calibration split; ECE and Brier score are reported on a separate held-out test
split.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable

import numpy as np
import torch

from orion_learning import obs_spec as S
from orion_learning.baselines import B0Controller, CriticGate
from orion_learning.escape_library import E
from orion_learning.env import DT, EnvConfig, HARD_STUCK_WINDOW_S, OrionEnv
from orion_learning.history import History
from orion_learning.nets import RiskCritic, TCNAdapter
from orion_learning.ppo import ActorCritic
from orion_learning.student import mode_fractions, SurrogateRisk

LABEL_SLACK_S = 15.0    # s past onset + 20 s window for the ENTRAPPED verdict to come in


@dataclass
class CriticConfig:
    num_envs: int = 64            # rovers rolled out by the student to visit states
    rollout_steps: int = 600      # steps per collection round
    sample_every: int = 20        # snapshot all rovers every this many steps
    warmup_steps: int = 40        # don't sample before the history is full
    rounds: int = 4               # collection rounds on training seeds
    holdout_rounds: int = 2       # rounds on held-out seeds (half calibration, half test)
    branches: int = len(E)        # K: one branch per escape manoeuvre in E
    horizon_s: float = 10.0       # T_h
    epochs: int = 30
    batch_size: int = 512
    lr: float = 1e-3
    hidden: list[int] = field(default_factory=lambda: [64, 64])
    ece_bins: int = 10
    gate_p_active: float = 0.3    # SurrogateRisk of the visiting student (student.py)
    gate_hold_s: tuple[float, float] = (0.5, 3.0)


def branch_labels(env_cfg: EnvConfig, snapshot: dict, branches: int, horizon_s: float,
                  seed: int) -> np.ndarray:
    """Fraction of K branches with an entrapment dated within horizon_s, per snapshot rover.

    Branch b of a rover starts escape manoeuvre b now and then runs B0 (module docstring).
    """
    n = next(iter(snapshot['arrays'].values())).shape[0]
    m = n * branches
    # No time limit in the branches: the label is about the terrain, not the trial clock.
    env = OrionEnv(m, replace(env_cfg, episode_s=1e6), seed=seed, auto_reset=False)
    env.restore_state(OrionEnv.tile_state(snapshot, branches))
    env.seed(seed)    # fresh noise streams: branches differ in sensor noise and dropouts
    deadline = env.state['t'] * DT + horizon_s + 1e-9
    ctl = B0Controller(m, escape_start_index=np.tile(np.arange(branches), n))
    ctl.escape.start(np.ones(m, dtype=bool))
    # The first observation only feeds B0's governor and stuck timers, which the running
    # escape overrides, so zeros stand in for the (unstored) snapshot observation.
    obs = np.zeros((m, S.OBS_DIM), np.float32)
    stuck = None      # env's monitor-rule FLAG_STUCK, fed back like orion_safety on the rover
    failed = np.zeros(m, dtype=bool)
    ended = np.zeros(m, dtype=bool)
    for _ in range(int(round((horizon_s + HARD_STUCK_WINDOW_S + LABEL_SLACK_S) / DT))):
        obs, _, term, _, info = env.step(ctl.act(obs, stuck))
        stuck = info['soft_stuck']
        event = (info['belly_out'] | info['hard_stuck']) & ~ended
        failed |= event & (info['entrapment_time'] <= deadline)
        ended |= term
        # Past the deadline only a stuck episode that began by then can still turn positive.
        onset = info['stuck_onset']
        pending = (onset >= 0.0) & (onset <= deadline)
        if np.all(ended | ((info['episode_time'] > deadline) & ~pending)):
            break
    return failed.reshape(n, branches).mean(axis=1)


def collect(teacher: ActorCritic, adapter: TCNAdapter, env_cfg: EnvConfig, cfg: CriticConfig,
            seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Run the gated student, snapshot visited states and label them.

    Returns (obs, z_hat, label, CriticGate mode fractions of the visiting rollout).
    """
    env = OrionEnv(cfg.num_envs, env_cfg, seed=seed)
    obs, _ = env.reset()
    hist = History(cfg.num_envs)
    gate = CriticGate(cfg.num_envs)
    risk = SurrogateRisk(cfg.num_envs, cfg.gate_p_active, cfg.gate_hold_s,
                         np.random.default_rng(seed))
    stuck = None
    since_reset = np.zeros(cfg.num_envs, dtype=np.int64)
    X, Z, Y, modes = [], [], [], []
    for t in range(cfg.rollout_steps):
        h = hist.push(obs)
        with torch.no_grad():
            z_hat = adapter(torch.as_tensor(h))
            action = teacher.policy(torch.as_tensor(obs), z_hat).numpy()
        action, mode = gate.gate(obs, action, risk(), stuck)
        modes.append(mode)
        if t % cfg.sample_every == 0:
            ok = since_reset >= cfg.warmup_steps
            if ok.any():
                rows = np.flatnonzero(ok)
                snap = env.clone_state()
                sub = OrionEnv.tile_state(snap, 1, rows)
                y = branch_labels(env_cfg, sub, cfg.branches, cfg.horizon_s,
                                  seed=seed * 7919 + t)
                X.append(obs[rows])
                Z.append(z_hat.numpy()[rows])
                Y.append(y)
        obs, _, term, trunc, info = env.step(action)
        done = term | trunc
        stuck = info['soft_stuck'] & ~done      # a reset rover starts unstuck
        since_reset = np.where(done, 0, since_reset + 1)
        if done.any():
            hist.reset(done)
            gate.reset(done)
            risk.reset(done)
    cat = lambda xs, d: np.concatenate(xs) if xs else np.zeros((0,) + d, np.float32)  # noqa
    return (cat(X, (S.OBS_DIM,)), cat(Z, (S.LATENT_DIM,)), cat(Y, ()),
            mode_fractions(modes))


def bce_soft(p, y, eps=1e-6) -> float:
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def expected_calibration_error(p, y, bins: int = 10) -> float:
    """ECE with equal-width bins: sum_b (n_b / n) |mean p_b - mean y_b| (y may be soft)."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    if len(p) == 0:
        return float('nan')
    idx = np.minimum((p * bins).astype(int), bins - 1)
    ece = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(ece)


def brier(p, y) -> float:
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2)) if len(p) else float('nan')


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """Temperature T > 0 minimising soft-label BCE of sigmoid(logits / T) (1-D search)."""
    if len(logits) == 0:
        return 1.0
    grid = np.exp(np.linspace(np.log(0.05), np.log(20.0), 400))
    losses = [bce_soft(1 / (1 + np.exp(-logits / T)), y) for T in grid]
    return float(grid[int(np.argmin(losses))])


def train_critic(teacher: ActorCritic, adapter: TCNAdapter, env_cfg: EnvConfig,
                 cfg: CriticConfig, seed: int,
                 log: Callable[[int, dict], None]) -> tuple[RiskCritic, dict]:
    torch.manual_seed(seed)
    parts = [collect(teacher, adapter, env_cfg, cfg, seed=seed + 1000 + r)
             for r in range(cfg.rounds)]
    held = [collect(teacher, adapter, env_cfg, cfg, seed=seed + 2000 + r)
            for r in range(cfg.holdout_rounds)]
    X, Z, Y = (np.concatenate([p[i] for p in parts]) for i in range(3))
    Xh, Zh, Yh = (np.concatenate([p[i] for p in held]) for i in range(3))
    visit_modes = {k: float(np.mean([p[3][k] for p in parts])) for k in parts[0][3]}
    half = len(Yh) // 2      # held-out seeds: first half calibrates T, second half is the test
    critic = RiskCritic(cfg.hidden)
    critic.obs_norm.load_state_dict(adapter.obs_norm.state_dict())
    opt = torch.optim.Adam(critic.parameters(), lr=cfg.lr)
    Xt, Zt, Yt = (torch.as_tensor(a, dtype=torch.float32) for a in (X, Z, Y))
    lossf = torch.nn.BCEWithLogitsLoss()
    for ep in range(cfg.epochs):
        perm = torch.randperm(len(Yt))
        losses = []
        for i in range(0, len(Yt), cfg.batch_size):
            idx = perm[i:i + cfg.batch_size]
            loss = lossf(critic.logits(Xt[idx], Zt[idx])[:, 0], Yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        log(ep, {'bce_train': float(np.mean(losses)) if losses else float('nan')})

    with torch.no_grad():
        logit_h = critic.logits(torch.as_tensor(Xh), torch.as_tensor(Zh))[:, 0].numpy()
    T = fit_temperature(logit_h[:half], Yh[:half])
    critic.temperature.fill_(T)
    sig = lambda x: 1 / (1 + np.exp(-x))  # noqa: E731
    p_raw, p_cal = sig(logit_h[half:]), sig(logit_h[half:] / T)
    y_test = Yh[half:]
    report = {
        'n_train': float(len(Y)), 'n_calib': float(half), 'n_test': float(len(y_test)),
        'label_mean_train': float(Y.mean()) if len(Y) else float('nan'),
        'label_mean_test': float(y_test.mean()) if len(y_test) else float('nan'),
        'temperature': T,
        'ece_raw': expected_calibration_error(p_raw, y_test, cfg.ece_bins),
        'ece': expected_calibration_error(p_cal, y_test, cfg.ece_bins),
        'brier_raw': brier(p_raw, y_test), 'brier': brier(p_cal, y_test),
        'brier_base_rate': brier(np.full(len(y_test), Y.mean() if len(Y) else 0.0), y_test),
        **{f'visit_{k}': v for k, v in visit_modes.items()},
    }
    log(cfg.epochs, report)
    return critic, report
