"""Training entry point: python -m orion_learning.train --config configs/<name>.yaml.

Stages (run in this order, or a subset with --stages):
  teacher  PPO with the privileged encoder (10 -> 8 latent)
  student  TCN adapter over the [40, 22] history -> latent, DAgger with the student acting
  critic   entrapment-risk critic on branched-rollout labels, temperature-scaled
  b2       robust PPO without latent (baseline B2)
  export   ONNX artefacts for student (+critic) and B2, parity-checked against PyTorch
  evaluate controllers over the sandbox cells at g = 9.81 and 1.62

Every run writes runs/<timestamp>/ with config.yaml, run_info.json (seed, git SHA, stages),
metrics.csv (stage, iteration, metric, value), checkpoints and the outputs of export and
evaluate. --run-dir continues an existing run (e.g. run only `critic` on a trained student).

Training distribution (council B1, env.EnvConfig): the PPO stages anneal the rejection
threshold env.dp_threshold from its start to its end value over the first
dp_curriculum_fraction of their iterations; student and critic use the end value. Every PPO
iteration also logs b0_completable, the fraction of episodes of the current training
distribution that B0 completes (shadow B0 rollouts, b0_monitor), and summary.json compares
its mean with the pre-registered band b0_completable_band (40-60 %). Evaluation uses the SPEC
section 6 distribution (env.spec_eval_config) as the base of the sandbox cells.
"""

from __future__ import annotations

import argparse
from collections import deque
import csv
from dataclasses import fields, replace
import datetime as dt
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch
import yaml

from orion_learning import env as E
from orion_learning.baselines import B0Controller
from orion_learning.checkpoints import (FILES, load_actor_critic, load_adapter,
                                        save_actor_critic, save_module)
from orion_learning.evaluate import evaluate
from orion_learning.export_onnx import export_from_run, git_sha
from orion_learning.ppo import PPOConfig, train_ppo
from orion_learning.risk import CriticConfig, train_critic
from orion_learning.student import StudentConfig, train_student

STAGES = ('teacher', 'student', 'critic', 'b2', 'export', 'evaluate')
DEFAULT_CURRICULUM_FRACTION = 1.0 / 3.0
DEFAULT_B0_BAND = (0.4, 0.6)       # pre-registered B0-completable band (README)


def _tuplify(v):
    return tuple(v) if isinstance(v, list) else v


def env_config_from(d: dict | None) -> E.EnvConfig:
    """Build an EnvConfig with overrides from a YAML mapping (nested for soil / patch_soil)."""
    cfg = E.EnvConfig()
    d = dict(d or {})
    for key in ('soil', 'patch_soil'):
        if d.get(key) is not None:
            base = getattr(cfg, key) or E.SoilRanges()
            d[key] = replace(base, **{k: _tuplify(v) for k, v in d[key].items()})
    unknown = set(d) - {f.name for f in fields(E.EnvConfig)}
    if unknown:
        raise KeyError(f'unknown env config keys: {sorted(unknown)}')
    return replace(cfg, **{k: _tuplify(v) for k, v in d.items()})


def dataclass_from(cls, d: dict | None):
    d = dict(d or {})
    unknown = set(d) - {f.name for f in fields(cls)}
    if unknown:
        raise KeyError(f'unknown {cls.__name__} keys: {sorted(unknown)}')
    return cls(**d)


class MetricsLog:
    """Long-format metrics CSV: stage, iteration, metric, value, wall_s."""

    def __init__(self, path: Path, echo: bool = True):
        self.path, self.echo, self.t0 = path, echo, time.time()
        new = not path.exists()
        self.f = open(path, 'a', newline='')
        self.w = csv.writer(self.f)
        if new:
            self.w.writerow(['stage', 'iteration', 'metric', 'value', 'wall_s'])

    def logger(self, stage: str):
        def log(it: int, metrics: dict) -> None:
            wall = round(time.time() - self.t0, 2)
            for k, v in metrics.items():
                self.w.writerow([stage, it, k, float(v), wall])
            self.f.flush()
            if self.echo:
                short = ' '.join(f'{k}={float(v):.4g}' for k, v in list(metrics.items())[:8])
                print(f'[{stage} {it}] {short}', flush=True)
        return log

    def close(self) -> None:
        self.f.close()


class B0CompletableMonitor:
    """Fraction of training episodes that B0 completes, estimated alongside PPO (council B1).

    A shadow env with the training config, held at the trainer's point of the curriculum,
    is driven by B0 for steps_per_iter steps per training iteration (auto-reset); the
    estimate is the success rate over the last `window` finished shadow episodes.
    """

    def __init__(self, env_cfg: E.EnvConfig, num_envs: int = 32, steps_per_iter: int = 256,
                 window: int = 256, seed: int = 0):
        self.env = E.OrionEnv(num_envs, env_cfg, seed=seed)
        self.obs, _ = self.env.reset()
        self.b0 = B0Controller(num_envs)
        self.steps = steps_per_iter
        self.outcomes: deque = deque(maxlen=window)
        self.history: list[float] = []

    def update(self, trainer_env_steps: int) -> dict:
        """Advance the shadow rollouts; return {'dp_threshold', 'b0_completable'?}."""
        for _ in range(self.steps):
            self.env.env_steps = trainer_env_steps      # sample at the trainer's curriculum
            self.obs, _, term, trunc, info = self.env.step(self.b0.act(self.obs))
            done = term | trunc
            if done.any():
                self.outcomes.extend(info['success'][done].tolist())
                self.b0.reset(done)
        out = {'dp_threshold': self.env.config.dp_threshold_at(trainer_env_steps)}
        if self.outcomes:
            out['b0_completable'] = float(np.mean(self.outcomes))
            self.history.append(out['b0_completable'])
        return out


def ppo_stage(cfg: dict, env_cfg: E.EnvConfig, pc: PPOConfig, seed: int, use_latent: bool,
              log, summary: dict, name: str):
    """Train one PPO stage with the DP/W curriculum and the B0-completable monitor."""
    frac = float(cfg.get('dp_curriculum_fraction', DEFAULT_CURRICULUM_FRACTION))
    stage_env = replace(env_cfg, dp_curriculum_steps=int(round(
        frac * pc.iterations * pc.rollout_steps)))
    monitor = B0CompletableMonitor(stage_env, seed=seed + 7, **(cfg.get('b0_monitor') or {}))

    def log_with_monitor(it: int, metrics: dict) -> None:
        # Threshold in effect while this iteration collected its rollout (it started then).
        log(it, {**monitor.update(it * pc.rollout_steps), **metrics})

    ac = train_ppo(stage_env, pc, seed, use_latent=use_latent, log=log_with_monitor)
    band = tuple(float(b) for b in cfg.get('b0_completable_band', DEFAULT_B0_BAND))
    mean = float(np.mean(monitor.history)) if monitor.history else float('nan')
    in_band = bool(band[0] <= mean <= band[1])
    summary[f'b0_completable_{name}'] = {'mean': mean, 'band': list(band), 'in_band': in_band}
    if not in_band:
        print(f'warning: {name} B0-completable fraction {mean:.2f} is outside the '
              f'pre-registered band {band}')
    return ac


def new_run_dir(root: Path) -> Path:
    stamp = dt.datetime.now().strftime('%Y%m%d-%H%M%S')
    path = root / stamp
    k = 1
    while path.exists():
        path = root / f'{stamp}-{k}'
        k += 1
    path.mkdir(parents=True)
    return path


def run(config_path: Path, stages=STAGES, run_dir: Path | None = None,
        runs_root: Path = Path('runs')) -> Path:
    cfg = yaml.safe_load(Path(config_path).read_text())
    seed = int(cfg.get('seed', 0))
    torch.set_num_threads(int(cfg.get('torch_threads', 1)))
    np.random.seed(seed)
    torch.manual_seed(seed)
    env_cfg = env_config_from(cfg.get('env'))
    final_env_cfg = replace(env_cfg, dp_curriculum_steps=0)   # end of the curriculum
    if run_dir is None:
        run_dir = new_run_dir(Path(runs_root))
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / 'config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
    info = {'config': str(config_path), 'seed': seed, 'git_sha': git_sha(Path(__file__).parent),
            'started': dt.datetime.now().isoformat(timespec='seconds'), 'stages': list(stages),
            'python': sys.version.split()[0], 'torch': torch.__version__}
    (run_dir / 'run_info.json').write_text(json.dumps(info, indent=2) + '\n')
    log = MetricsLog(run_dir / 'metrics.csv')
    summary: dict = {}
    t_start = time.time()
    try:
        if 'teacher' in stages:
            pc = dataclass_from(PPOConfig, cfg.get('teacher'))
            ac = ppo_stage(cfg, env_cfg, pc, seed, True, log.logger('teacher'), summary,
                           'teacher')
            save_actor_critic(run_dir / FILES['teacher'], ac, pc)
        if 'student' in stages:
            sc = dataclass_from(StudentConfig, cfg.get('student'))
            teacher = load_actor_critic(run_dir / FILES['teacher'])
            adapter = train_student(teacher, final_env_cfg, sc, seed + 1,
                                    log.logger('student'))
            save_module(run_dir / FILES['student'], adapter, channels=sc.channels)
        if 'critic' in stages:
            cc = dataclass_from(CriticConfig, cfg.get('critic'))
            teacher = load_actor_critic(run_dir / FILES['teacher'])
            adapter = load_adapter(run_dir / FILES['student'])
            critic, report = train_critic(teacher, adapter, final_env_cfg, cc, seed + 2,
                                          log.logger('critic'))
            save_module(run_dir / FILES['critic'], critic, hidden=list(cc.hidden))
            summary['critic'] = report
        if 'b2' in stages:
            pc = dataclass_from(PPOConfig, cfg.get('b2'))
            ac = ppo_stage(cfg, env_cfg, pc, seed + 3, False, log.logger('b2'), summary,
                           'b2')
            save_actor_critic(run_dir / FILES['B2'], ac, pc)
        if 'export' in stages:
            for controller in ('student', 'B2'):
                m = export_from_run(run_dir, controller)
                summary[f'export_{controller}'] = m['parity_max_abs_err']
                log.logger('export')(0, {f'{controller}_{k}_parity': v
                                         for k, v in m['parity_max_abs_err'].items()})
        if 'evaluate' in stages:
            ev = dict(cfg.get('evaluate') or {})
            rows = evaluate(ev.get('controllers', ['B0', 'B2', 'student', 'student+critic']),
                            int(ev.get('trials', 25)), run_dir, run_dir,
                            seed=int(ev.get('seed', 12345)),
                            base=E.spec_eval_config(final_env_cfg))
            summary['evaluate'] = rows
    finally:
        log.close()
    summary['wall_s'] = round(time.time() - t_start, 1)
    (run_dir / 'summary.json').write_text(json.dumps(summary, indent=2, default=float) + '\n')
    bad = [k for k, v in _flatten(summary).items()
           if isinstance(v, float) and not math.isfinite(v)]
    print(f'run finished in {summary["wall_s"]} s: {run_dir}')
    if bad:
        print(f'warning: non-finite summary values: {bad}')
    return run_dir


def _flatten(d, prefix=''):
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(_flatten(v, f'{prefix}{k}.'))
    elif isinstance(d, list):
        for i, v in enumerate(d):
            out.update(_flatten(v, f'{prefix}{i}.'))
    else:
        out[prefix[:-1]] = d
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description='Train ORION controllers (see module docstring).')
    ap.add_argument('--config', required=True, type=Path)
    ap.add_argument('--stages', nargs='+', default=list(STAGES), choices=STAGES)
    ap.add_argument('--run-dir', type=Path, default=None, help='continue this run directory')
    ap.add_argument('--runs-root', type=Path, default=Path('runs'))
    args = ap.parse_args(argv)
    run(args.config, args.stages, args.run_dir, args.runs_root)


if __name__ == '__main__':
    main()
