"""Evaluate controllers over the sandbox cells in simulation (SPEC section 8, sim analogue).

Cells: {light, heavy} x {packed, loose} (env.CELL_* parameter sets) at g = 9.81 and again at
g = 1.62. Every controller sees the same seeds per cell (common random numbers). Writes
<out>/eval_episodes.csv (one row per trial) and <out>/eval_summary.csv with success rate,
entrapment rate (belly-out or hard-stuck), and effective speed.

Effective speed of a trial = forward progress / trial time, where trial time is the
time-to-goal for a success and the full 60 s otherwise (a trapped rover keeps the clock
running, as in the sandbox protocol).

    python -m orion_learning.evaluate --run runs/<timestamp> --trials 25
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from orion_learning import env as E
from orion_learning.baselines import B0Controller, CriticGate
from orion_learning.checkpoints import FILES, load_actor_critic, load_controller
from orion_learning.history import History

GRAVITIES = (9.81, 1.62)
ALL_CONTROLLERS = ('B0', 'B2', 'student', 'student+critic', 'teacher')


class _B0:
    def __init__(self, n):
        self.c = B0Controller(n)

    def act(self, obs, info):
        return self.c.act(obs)


class _Learned:
    """Student / B2 / teacher (B1 oracle, sim only), optionally with critic gating."""

    def __init__(self, n, policy, adapter=None, critic=None, encoder=None):
        self.policy, self.adapter, self.critic, self.encoder = policy, adapter, critic, encoder
        self.hist = History(n)
        self.prev = np.zeros((n, 2), np.float32)
        self.gate = CriticGate(n) if critic is not None else None
        self.modes = []

    @torch.no_grad()
    def act(self, obs, info):
        h = self.hist.push(obs, self.prev)
        o = torch.as_tensor(obs)
        if self.adapter is not None:
            z = self.adapter(torch.as_tensor(h))
        elif self.encoder is not None:
            z = self.encoder(torch.as_tensor(info['privileged']))
        else:
            z = torch.zeros(len(obs), 8)
        a = self.policy(o, z).numpy()
        if self.gate is not None:
            rho = self.critic(o, z).numpy()[:, 0]
            a, mode = self.gate.gate(obs, a, rho)
            self.modes.append(mode)
        self.prev = a.astype(np.float32)   # history holds the action actually sent
        return a


def make_controller(name: str, n: int, run_dir: Path | None):
    if name == 'B0':
        return _B0(n)
    if run_dir is None:
        raise ValueError(f'controller {name} needs --run')
    if name == 'B2':
        return _Learned(n, load_controller(run_dir, 'B2')[0])
    if name == 'teacher':
        ac = load_actor_critic(run_dir / FILES['teacher'])
        return _Learned(n, ac.policy, encoder=ac.encoder)
    policy, adapter, critic = load_controller(run_dir, 'student')
    if name == 'student':
        return _Learned(n, policy, adapter)
    if name == 'student+critic':
        if critic is None:
            raise FileNotFoundError(f'no critic checkpoint in {run_dir}')
        return _Learned(n, policy, adapter, critic)
    raise ValueError(name)


def run_trials(controller_name: str, cell: str, gravity: float, trials: int, seed: int,
               run_dir: Path | None = None, base: E.EnvConfig | None = None) -> list[dict]:
    env = E.OrionEnv(trials, E.cell_config(cell, gravity, base), seed=seed, auto_reset=False)
    obs, info = env.reset()
    ctl = make_controller(controller_name, trials, run_dir)
    outcome = np.full(trials, 'timeout', dtype=object)
    t_end = np.full(trials, env.config.episode_s)
    prog = np.zeros(trials)
    done = np.zeros(trials, dtype=bool)
    for k in range(env.config.max_steps):
        obs, _, term, trunc, info = env.step(ctl.act(obs, info))
        prog = np.where(done, prog, info['progress'])
        for key in ('success', 'belly_out', 'hard_stuck', 'out_of_fence'):
            new = info[key] & ~done
            outcome[new] = key
            if key == 'success':
                t_end[new] = (k + 1) * E.DT
            done |= new
        if done.all():
            break
    rows = []
    for i in range(trials):
        rows.append({'controller': controller_name, 'cell': cell, 'gravity': gravity,
                     'trial': i, 'outcome': outcome[i],
                     'success': int(outcome[i] == 'success'),
                     'entrapped': int(outcome[i] in ('belly_out', 'hard_stuck')),
                     'time_s': round(float(t_end[i]), 3), 'progress_m': round(float(prog[i]), 4),
                     'effective_speed': round(max(float(prog[i]), 0.0) / float(t_end[i]), 5)})
    return rows


def summarise(rows: list[dict]) -> list[dict]:
    keys = sorted({(r['controller'], r['cell'], r['gravity']) for r in rows},
                  key=lambda k: (k[0], k[2], k[1]))
    out = []
    for c, cell, g in keys:
        rs = [r for r in rows if (r['controller'], r['cell'], r['gravity']) == (c, cell, g)]
        eff = np.array([r['effective_speed'] for r in rs])
        out.append({'controller': c, 'cell': cell, 'gravity': g, 'trials': len(rs),
                    'success_rate': round(float(np.mean([r['success'] for r in rs])), 4),
                    'entrapment_rate': round(float(np.mean([r['entrapped'] for r in rs])), 4),
                    'effective_speed_mean': round(float(eff.mean()), 4),
                    'effective_speed_median': round(float(np.median(eff)), 4)})
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def evaluate(controllers, trials: int, out_dir: Path, run_dir: Path | None = None,
             seed: int = 12345, gravities=GRAVITIES, base: E.EnvConfig | None = None
             ) -> list[dict]:
    """Run every controller over CELLS x gravities and write the CSVs.

    base supplies the non-cell settings (episode length, noise, latency) and defaults to the
    training EnvConfig.
    """
    rows = []
    for g_i, g in enumerate(gravities):
        for c_i, cell in enumerate(E.CELLS):
            cell_seed = seed + 100 * g_i + c_i      # same seed for every controller
            for name in controllers:
                rows += run_trials(name, cell, g, trials, cell_seed, run_dir, base)
    summary = summarise(rows)
    write_csv(Path(out_dir) / 'eval_episodes.csv', rows)
    write_csv(Path(out_dir) / 'eval_summary.csv', summary)
    return summary


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--run', type=Path, default=None, help='runs/<timestamp> with checkpoints')
    ap.add_argument('--controllers', nargs='+', default=['B0', 'B2', 'student', 'student+critic'],
                    choices=ALL_CONTROLLERS)
    ap.add_argument('--trials', type=int, default=25)
    ap.add_argument('--seed', type=int, default=12345)
    ap.add_argument('--out', type=Path, default=None)
    args = ap.parse_args(argv)
    out = args.out or (args.run if args.run else Path('.'))
    for row in evaluate(args.controllers, args.trials, out, args.run, args.seed):
        print(row)


if __name__ == '__main__':
    main()
