"""Saving and loading stage checkpoints inside a run directory (runs/<timestamp>/)."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch

from orion_learning.nets import RiskCritic, TCNAdapter
from orion_learning.ppo import ActorCritic, PPOConfig

FILES = {'teacher': 'teacher.pt', 'B2': 'b2.pt', 'student': 'student.pt', 'critic': 'critic.pt'}


def save_actor_critic(path: Path, ac: ActorCritic, cfg: PPOConfig) -> None:
    torch.save({'config': asdict(cfg), 'use_latent': ac.use_latent,
                'state_dict': ac.state_dict()}, path)


def load_actor_critic(path: Path) -> ActorCritic:
    ck = torch.load(path, weights_only=False)
    ac = ActorCritic(PPOConfig(**ck['config']), ck['use_latent'])
    ac.load_state_dict(ck['state_dict'])
    return ac.eval()


def save_module(path: Path, module: torch.nn.Module, **kwargs) -> None:
    torch.save({'kwargs': kwargs, 'state_dict': module.state_dict()}, path)


def load_adapter(path: Path) -> TCNAdapter:
    ck = torch.load(path, weights_only=False)
    m = TCNAdapter(**ck['kwargs'])
    m.load_state_dict(ck['state_dict'])
    return m.eval()


def load_critic(path: Path) -> RiskCritic:
    ck = torch.load(path, weights_only=False)
    m = RiskCritic(**ck['kwargs'])
    m.load_state_dict(ck['state_dict'])
    return m.eval()


def load_controller(run_dir: Path, controller: str):
    """(policy, adapter | None, critic | None) of a controller trained in run_dir."""
    run_dir = Path(run_dir)
    if controller == 'B2':
        return load_actor_critic(run_dir / FILES['B2']).policy, None, None
    teacher = load_actor_critic(run_dir / FILES['teacher'])
    if controller == 'teacher':
        return teacher.policy, None, None
    if controller == 'student':
        adapter = load_adapter(run_dir / FILES['student'])
        critic_path = run_dir / FILES['critic']
        critic = load_critic(critic_path) if critic_path.exists() else None
        return teacher.policy, adapter, critic
    raise ValueError(f'unknown controller {controller!r}')
