"""ONNX export follows the artefact contract and matches PyTorch."""

import hashlib
import json

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch

from orion_learning import obs_spec as S
from orion_learning.export_onnx import export_controller
from orion_learning.nets import Policy, RiskCritic, TCNAdapter


def _trained_like(module):
    """Randomise biases and normaliser stats (weights keep PyTorch's fan-in scaled init)."""
    gen = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for name, p in module.named_parameters():
            if name.endswith('bias') or name == 'log_std':
                p.copy_(torch.randn(p.shape, generator=gen) * 0.3)
        for name, b in module.named_buffers():
            if name.endswith('mean'):
                b.copy_(torch.randn(b.shape, generator=gen))
            elif name.endswith('var'):
                b.copy_(torch.rand(b.shape, generator=gen) + 0.1)
    return module.eval()


@pytest.fixture(scope='module')
def student_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp('student')
    policy = _trained_like(Policy([16, 16]))
    adapter = _trained_like(TCNAdapter(8))
    critic = _trained_like(RiskCritic([8]))
    critic.temperature.fill_(1.7)
    manifest = export_controller(out, 'student', policy, adapter, critic, sha='abc123')
    return out, manifest, policy, adapter, critic


def test_manifest_and_hashes(student_dir):
    out, manifest, *_ = student_dir
    disk = json.loads((out / 'manifest.json').read_text())
    assert disk['controller'] == 'student' and disk['obs_spec_hash'] == S.OBS_SPEC_HASH
    assert disk['files'] == {'adapter': 'adapter.onnx', 'policy': 'policy.onnx',
                             'critic': 'critic.onnx'}
    for f, h in disk['sha256'].items():
        assert hashlib.sha256((out / f).read_bytes()).hexdigest() == h
    assert max(manifest['parity_max_abs_err'].values()) <= 1e-5


def test_io_names_shapes_and_metadata(student_dir):
    out, *_ = student_dir
    expect = {'adapter': ({'history': [1, 40, 22]}, {'latent': [1, 8]}),
              'policy': ({'obs': [1, 20], 'latent': [1, 8]}, {'action': [1, 2]}),
              'critic': ({'obs': [1, 20], 'latent': [1, 8]}, {'risk': [1, 1]})}
    for kind, (ins, outs) in expect.items():
        m = onnx.load(str(out / f'{kind}.onnx'))
        dims = lambda vi: [d.dim_value for d in vi.type.tensor_type.shape.dim]  # noqa: E731
        assert {i.name: dims(i) for i in m.graph.input} == ins
        assert {o.name: dims(o) for o in m.graph.output} == outs
        for vi in list(m.graph.input) + list(m.graph.output):
            assert vi.type.tensor_type.elem_type == onnx.TensorProto.FLOAT
        meta = {p.key: p.value for p in m.metadata_props}
        assert meta == {'obs_spec_hash': S.OBS_SPEC_HASH, 'control_hz': '20', 'history': '40',
                        'latent_dim': '8', 'obs_dim': '20', 'action_dim': '2',
                        'model_kind': kind, 'controller': 'student', 'git_sha': 'abc123'}


def test_onnxruntime_matches_torch_and_outputs_are_in_range(student_dir):
    out, _, policy, adapter, critic = student_dir
    rng = np.random.default_rng(1)
    h = rng.uniform(-1.5, 1.5, (1, 40, 22)).astype(np.float32)
    o = rng.uniform(-1.5, 1.5, (1, 20)).astype(np.float32)
    run = lambda k, feed: ort.InferenceSession(str(out / f'{k}.onnx')).run(None, feed)[0]  # noqa
    z = run('adapter', {'history': h})
    a = run('policy', {'obs': o, 'latent': z})
    r = run('critic', {'obs': o, 'latent': z})
    with torch.no_grad():
        zt = adapter(torch.as_tensor(h))
        np.testing.assert_allclose(z, zt.numpy(), atol=1e-5)
        np.testing.assert_allclose(a, policy(torch.as_tensor(o), zt).numpy(), atol=1e-5)
        np.testing.assert_allclose(r, critic(torch.as_tensor(o), zt).numpy(), atol=1e-5)
    assert np.all(np.abs(a) <= 1.0) and 0.0 <= r[0, 0] <= 1.0
    # Temperature is applied inside the graph.
    with torch.no_grad():
        raw = torch.sigmoid(critic.logits(torch.as_tensor(o), zt)).numpy()
    assert not np.allclose(raw, r)


def test_b2_export_has_no_adapter_and_ignores_latent(tmp_path):
    policy = _trained_like(Policy([16], use_latent=False))
    policy.latent_gain.fill_(0.0)       # _trained_like does not touch it; keep B2 semantics
    m = export_controller(tmp_path, 'B2', policy, sha='x')
    assert m['files'] == {'adapter': None, 'policy': 'policy.onnx', 'critic': None}
    sess = ort.InferenceSession(str(tmp_path / 'policy.onnx'))
    o = np.full((1, 20), 0.3, np.float32)
    a0 = sess.run(None, {'obs': o, 'latent': np.zeros((1, 8), np.float32)})[0]
    a1 = sess.run(None, {'obs': o, 'latent': np.ones((1, 8), np.float32)})[0]
    assert np.array_equal(a0, a1)


def test_parity_check_can_fail(tmp_path, monkeypatch):
    """The parity gate is real: a torch module that disagrees with its export is rejected."""
    from orion_learning import export_onnx as X
    policy = _trained_like(Policy([8]))
    real = X.check_parity

    def tampered(module, inputs, path, **kw):
        with torch.no_grad():
            module.net[-1].bias.add_(1e-3)       # torch drifts after export
        return real(module, inputs, path, **kw)
    monkeypatch.setattr(X, 'check_parity', tampered)
    with pytest.raises(RuntimeError, match='differs'):
        X.export_controller(tmp_path, 'B2', policy, sha='x')
