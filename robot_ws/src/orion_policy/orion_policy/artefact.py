"""
Load and verify a trained controller exported per the ONNX artefact contract.

A directory holds manifest.json plus policy.onnx and optionally adapter.onnx and critic.onnx.
Loading refuses (ArtefactError) unless every check passes:

- the manifest's and every file's obs_spec_hash equal the installed obs_spec.OBS_SPEC_HASH,
- every file's sha256 equals the manifest entry,
- every file's metadata (model_kind, obs_dim, history, latent_dim, action_dim, control_hz)
  matches its role and obs_spec, and its graph has the contract's input/output names.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from typing import Optional

import numpy as np
from orion_learning import obs_spec

ROLES = ('adapter', 'policy', 'critic')
IO_NAMES = {
    'adapter': (('history',), ('latent',)),
    'policy': (('obs', 'latent'), ('action',)),
    'critic': (('obs', 'latent'), ('risk',)),
}
REQUIRED_METADATA = ('obs_spec_hash', 'control_hz', 'history', 'latent_dim', 'obs_dim',
                     'action_dim', 'model_kind', 'controller', 'git_sha')


class ArtefactError(RuntimeError):
    """The artefact is missing, tampered with, or built for another obs_spec."""


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class Artefact:
    """Verified onnxruntime sessions for one controller."""

    directory: str
    controller: str
    policy: object
    adapter: Optional[object]
    critic: Optional[object]
    sha256: dict
    git_sha: str

    @property
    def policy_sha(self) -> str:
        """Short (16 hex) sha256 of policy.onnx, as reported in PolicyStatus.policy_sha."""
        return self.sha256['policy'][:16]

    def latent(self, history: np.ndarray) -> np.ndarray:
        """
        z-hat float32 [LATENT_DIM] from a [HISTORY_LEN, OBS_DIM + ACTION_DIM] history.

        Zeros when the artefact has no adapter (non-adaptive baseline B2).
        """
        if self.adapter is None:
            return np.zeros(obs_spec.LATENT_DIM, dtype=np.float32)
        out = self.adapter.run(['latent'], {'history': history[None].astype(np.float32)})[0]
        return np.asarray(out, dtype=np.float32).reshape(obs_spec.LATENT_DIM)

    def action(self, obs: np.ndarray, latent: np.ndarray) -> np.ndarray:
        """Deterministic action float32 [2], clipped to [-1, 1]."""
        out = self.policy.run(['action'], self._obs_latent(obs, latent))[0]
        return np.clip(np.asarray(out, dtype=np.float32).reshape(obs_spec.ACTION_DIM), -1, 1)

    def risk(self, obs: np.ndarray, latent: np.ndarray) -> float:
        """Entrapment risk rho in [0, 1], NaN when there is no critic."""
        if self.critic is None:
            return float('nan')
        out = self.critic.run(['risk'], self._obs_latent(obs, latent))[0]
        return float(np.clip(np.asarray(out).reshape(-1)[0], 0.0, 1.0))

    @staticmethod
    def _obs_latent(obs, latent) -> dict:
        return {'obs': np.asarray(obs, dtype=np.float32).reshape(1, obs_spec.OBS_DIM),
                'latent': np.asarray(latent, dtype=np.float32).reshape(1, obs_spec.LATENT_DIM)}


def _expected_metadata(role: str) -> dict:
    d = obs_spec.describe()
    return {'obs_spec_hash': d['obs_spec_hash'], 'model_kind': role,
            'control_hz': str(d['control_hz']), 'history': str(d['history']),
            'latent_dim': str(d['latent_dim']), 'obs_dim': str(d['obs_dim']),
            'action_dim': str(d['action_dim'])}


def _check_model(session, name: str, role: str, controller: str) -> dict:
    """Check metadata_props and graph I/O names of one loaded model; return its metadata."""
    meta = dict(session.get_modelmeta().custom_metadata_map)
    missing = [k for k in REQUIRED_METADATA if k not in meta]
    if missing:
        raise ArtefactError(f'{name}: metadata_props missing {missing}')
    for key, want in _expected_metadata(role).items():
        if meta[key] != want:
            raise ArtefactError(f'{name}: metadata {key}={meta[key]!r}, runtime expects {want!r}')
    if meta['controller'] != controller:
        raise ArtefactError(f'{name}: controller {meta["controller"]!r} != manifest '
                            f'{controller!r}')
    inputs = tuple(sorted(i.name for i in session.get_inputs()))
    outputs = tuple(o.name for o in session.get_outputs())
    want_in, want_out = IO_NAMES[role]
    if inputs != tuple(sorted(want_in)) or outputs != want_out:
        raise ArtefactError(f'{name}: graph I/O {inputs} -> {outputs}, contract needs '
                            f'{want_in} -> {want_out}')
    return meta


def load_artefact(directory: str) -> Artefact:
    """Verify and load the controller in directory; raises ArtefactError on any mismatch."""
    manifest_path = os.path.join(directory, 'manifest.json')
    try:
        with open(manifest_path) as f:
            manifest = json.load(f)
    except (OSError, ValueError) as e:
        raise ArtefactError(f'cannot read {manifest_path}: {e}') from e
    for key in ('controller', 'obs_spec_hash', 'files', 'sha256'):
        if key not in manifest:
            raise ArtefactError(f'manifest.json has no {key!r}')
    if manifest['obs_spec_hash'] != obs_spec.OBS_SPEC_HASH:
        raise ArtefactError(f'manifest obs_spec_hash {manifest["obs_spec_hash"]} != installed '
                            f'{obs_spec.OBS_SPEC_HASH}: re-export the model with this obs_spec')
    files = manifest['files']
    if not files.get('policy'):
        raise ArtefactError('manifest has no policy file')

    import onnxruntime as ort
    sessions, digests, git_sha = {}, {}, ''
    for role in ROLES:
        name = files.get(role)
        if not name:
            sessions[role] = None
            continue
        if os.path.basename(name) != name:
            raise ArtefactError(f'{role} file {name!r} must be a plain file name in {directory}')
        path = os.path.join(directory, name)
        if not os.path.isfile(path):
            raise ArtefactError(f'{role} file {path} does not exist')
        want = manifest['sha256'].get(name)
        if want is None:
            raise ArtefactError(f'manifest sha256 has no entry for {name}')
        got = sha256_file(path)
        if got != want:
            raise ArtefactError(f'{name}: sha256 {got} does not match manifest {want}')
        digests[role] = got
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1   # tiny models; do not compete with the rest of ROS
        options.inter_op_num_threads = 1
        try:
            sessions[role] = ort.InferenceSession(path, options,
                                                  providers=['CPUExecutionProvider'])
        except Exception as e:  # onnxruntime raises its own exception types
            raise ArtefactError(f'{name}: onnxruntime cannot load it: {e}') from e
        meta = _check_model(sessions[role], name, role, manifest['controller'])
        if role == 'policy':
            git_sha = meta['git_sha']
    return Artefact(directory=directory, controller=manifest['controller'],
                    policy=sessions['policy'], adapter=sessions['adapter'],
                    critic=sessions['critic'], sha256=digests, git_sha=git_sha)
