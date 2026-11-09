"""Export a trained controller to the ONNX artefact contract read by orion_policy.

Writes adapter.onnx (student only), policy.onnx, critic.onnx (optional) and manifest.json
into one directory. Every graph has batch size 1 and carries metadata_props obs_spec_hash,
control_hz, history, latent_dim, obs_dim, action_dim, model_kind, controller, git_sha.
After writing, each file is run in onnxruntime against the PyTorch module on random inputs
and the export fails unless they agree to atol 1e-5.

    python -m orion_learning.export_onnx --run runs/<timestamp> --controller student
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import warnings

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn

from orion_learning import obs_spec as S
from orion_learning.checkpoints import load_controller

OPSET = 17
PARITY_ATOL = 1e-5
CONTROLLERS = ('student', 'B2', 'teacher')


def git_sha(cwd: Path | None = None) -> str:
    """Short SHA of HEAD (with '-dirty' if the tree has changes), or 'unknown'."""
    try:
        sha = subprocess.run(['git', 'rev-parse', '--short=12', 'HEAD'], cwd=cwd, check=True,
                             capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(['git', 'status', '--porcelain'], cwd=cwd, check=True,
                               capture_output=True, text=True).stdout.strip()
        return sha + ('-dirty' if dirty else '')
    except (OSError, subprocess.CalledProcessError):
        return 'unknown'


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class _RiskOut(nn.Module):
    def __init__(self, critic):
        super().__init__()
        self.critic = critic

    def forward(self, obs, latent):
        return self.critic(obs, latent)


def _export(module: nn.Module, inputs: dict[str, torch.Tensor], output: str, path: Path,
            meta: dict[str, str]) -> None:
    module = module.eval()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')   # legacy exporter deprecation notice
        torch.onnx.export(module, tuple(inputs.values()), str(path),
                          input_names=list(inputs), output_names=[output],
                          opset_version=OPSET, dynamo=False)
    model = onnx.load(str(path))
    for k, v in meta.items():
        entry = model.metadata_props.add()
        entry.key, entry.value = k, str(v)
    onnx.checker.check_model(model)
    onnx.save(model, str(path))


def check_parity(module: nn.Module, inputs: dict[str, torch.Tensor], path: Path,
                 n: int = 32, seed: int = 0) -> float:
    """Max |onnxruntime - torch| over n random input draws; raises if above PARITY_ATOL."""
    sess = ort.InferenceSession(str(path), providers=['CPUExecutionProvider'])
    gen = torch.Generator().manual_seed(seed)
    worst = 0.0
    for _ in range(n):
        feed = {k: (torch.rand(v.shape, generator=gen) * 3.0 - 1.5) for k, v in inputs.items()}
        with torch.no_grad():
            ref = module(*feed.values()).numpy()
        out = sess.run(None, {k: v.numpy() for k, v in feed.items()})[0]
        worst = max(worst, float(np.max(np.abs(out - ref))))
    if worst > PARITY_ATOL:
        raise RuntimeError(f'{path.name}: onnxruntime differs from torch by {worst:.2e}')
    return worst


def export_controller(out_dir: Path, controller: str, policy: nn.Module,
                      adapter: nn.Module | None = None, critic: nn.Module | None = None,
                      sha: str | None = None) -> dict:
    """Write the ONNX files + manifest.json for one controller; returns the manifest."""
    if controller not in CONTROLLERS:
        raise ValueError(f'controller must be one of {CONTROLLERS}')
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sha = sha or git_sha()
    base = {**{k: str(v) for k, v in S.describe().items()}, 'controller': controller,
            'git_sha': sha}
    obs = torch.zeros(1, S.OBS_DIM)
    latent = torch.zeros(1, S.LATENT_DIM)
    history = torch.zeros(1, S.HISTORY_LEN, S.OBS_DIM + S.ACTION_DIM)
    jobs = [('policy', policy, {'obs': obs, 'latent': latent}, 'action')]
    if adapter is not None:
        jobs.append(('adapter', adapter, {'history': history}, 'latent'))
    if critic is not None:
        jobs.append(('critic', _RiskOut(critic), {'obs': obs, 'latent': latent}, 'risk'))
    files: dict[str, str | None] = {'adapter': None, 'policy': None, 'critic': None}
    parity = {}
    for kind, module, inputs, output in jobs:
        path = out_dir / f'{kind}.onnx'
        _export(module, inputs, output, path, {**base, 'model_kind': kind})
        parity[kind] = check_parity(module, inputs, path)
        files[kind] = path.name
    manifest = {'controller': controller, 'obs_spec_hash': S.OBS_SPEC_HASH, 'files': files,
                'sha256': {f: sha256_file(out_dir / f) for f in files.values() if f}}
    (out_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    manifest['parity_max_abs_err'] = parity
    return manifest


def export_from_run(run_dir: Path, controller: str, out_dir: Path | None = None) -> dict:
    """Load checkpoints of a training run and export one controller."""
    run_dir = Path(run_dir)
    policy, adapter, critic = load_controller(run_dir, controller)
    sha = json.loads((run_dir / 'run_info.json').read_text()).get('git_sha')
    return export_controller(out_dir or run_dir / 'export' / controller, controller, policy,
                             adapter, critic, sha=sha)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--run', required=True, type=Path, help='runs/<timestamp> directory')
    ap.add_argument('--controller', default='student', choices=CONTROLLERS)
    ap.add_argument('--out', type=Path, default=None, help='output directory')
    args = ap.parse_args(argv)
    manifest = export_from_run(args.run, args.controller, args.out)
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
