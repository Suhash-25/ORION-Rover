"""
Tiny controllers exported exactly per the ONNX artefact contract, for tests only.

The real models come from learning/ (WP4). These have fixed random weights so the runtime's
outputs can be compared against the torch modules that produced them.
"""

import hashlib
import json
import os

from orion_learning import obs_spec
import torch
from torch import nn

ROW = obs_spec.OBS_DIM + obs_spec.ACTION_DIM


class TinyAdapter(nn.Module):
    """history [1, H, 22] -> latent [1, 8]."""

    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(ROW, obs_spec.LATENT_DIM)

    def forward(self, history):
        return torch.tanh(self.lin(history.mean(dim=1)))


class TinyPolicy(nn.Module):
    """obs [1, 20], latent [1, 8] -> action [1, 2] in [-1, 1]."""

    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(obs_spec.OBS_DIM + obs_spec.LATENT_DIM, obs_spec.ACTION_DIM)

    def forward(self, obs, latent):
        return torch.tanh(self.lin(torch.cat([obs, latent], dim=1)))


class TinyCritic(nn.Module):
    """obs, latent -> risk [1, 1] in [0, 1]; bias shifts the risk level."""

    def __init__(self, bias=0.0):
        super().__init__()
        self.lin = nn.Linear(obs_spec.OBS_DIM + obs_spec.LATENT_DIM, 1)
        self.bias = bias

    def forward(self, obs, latent):
        return torch.sigmoid(self.lin(torch.cat([obs, latent], dim=1)) + self.bias)


def _export(module, args, path, input_names, output_names, metadata):
    import onnx
    torch.onnx.export(module, args, path, input_names=input_names, output_names=output_names,
                      dynamo=False)
    model = onnx.load(path)
    for key, value in metadata.items():
        entry = model.metadata_props.add()
        entry.key, entry.value = key, str(value)
    onnx.save(model, path)


def export_artefact(directory, controller='student', adapter=True, critic=True,
                    critic_bias=0.0, file_hash=None, manifest_hash=None,
                    input_names=None, seed=0, policy_action=None):
    """
    Write adapter/policy/critic ONNX + manifest.json to directory; return the modules.

    file_hash / manifest_hash override obs_spec_hash (to test refusal); input_names
    overrides the policy's input names (to test the I/O check). policy_action (a0, a1) in
    (-1, 1) makes the policy output that constant action (zero weights, bias atanh(a)), e.g.
    (0.6, 0.0) = 0.3 m/s straight, for a predictable drive in simulation.
    """
    torch.manual_seed(seed)
    os.makedirs(directory, exist_ok=True)
    modules = {'adapter': TinyAdapter() if adapter else None, 'policy': TinyPolicy(),
               'critic': TinyCritic(critic_bias) if critic else None}
    if policy_action is not None:
        with torch.no_grad():
            modules['policy'].lin.weight.zero_()
            modules['policy'].lin.bias.copy_(torch.atanh(torch.tensor(policy_action)))
    obs = torch.zeros(1, obs_spec.OBS_DIM)
    latent = torch.zeros(1, obs_spec.LATENT_DIM)
    history = torch.zeros(1, obs_spec.HISTORY_LEN, ROW)
    specs = {
        'adapter': ((history,), ['history'], ['latent']),
        'policy': ((obs, latent), input_names or ['obs', 'latent'], ['action']),
        'critic': ((obs, latent), ['obs', 'latent'], ['risk']),
    }
    files, digests = {}, {}
    for role, module in modules.items():
        if module is None:
            files[role] = None
            continue
        name = f'{role}.onnx'
        meta = dict(obs_spec.describe())
        meta.update(obs_spec_hash=file_hash or obs_spec.OBS_SPEC_HASH, model_kind=role,
                    controller=controller, git_sha='0123456789abcdef')
        args, ins, outs = specs[role]
        path = os.path.join(directory, name)
        _export(module.eval(), args, path, ins, outs, meta)
        files[role] = name
        with open(path, 'rb') as f:
            digests[name] = hashlib.sha256(f.read()).hexdigest()
    manifest = {'controller': controller,
                'obs_spec_hash': manifest_hash or obs_spec.OBS_SPEC_HASH,
                'files': files, 'sha256': digests}
    with open(os.path.join(directory, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)
    return modules
