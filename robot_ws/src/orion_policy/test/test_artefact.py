"""ONNX loader: accepts a contract-conforming artefact, refuses everything else."""

import json
import os

import numpy as np
from onnx_fixture import export_artefact
from orion_learning import obs_spec
from orion_policy.artefact import ArtefactError
from orion_policy.artefact import load_artefact
from orion_policy.controllers import ControlInput
from orion_policy.controllers import OnnxController
import pytest
import torch


@pytest.fixture(scope='module')
def good(tmp_path_factory):
    d = str(tmp_path_factory.mktemp('student'))
    modules = export_artefact(d, critic_bias=0.5)
    return d, modules


def random_inputs(seed=1):
    rng = np.random.default_rng(seed)
    obs = rng.uniform(-1, 1, obs_spec.OBS_DIM).astype(np.float32)
    history = rng.uniform(-1, 1, (obs_spec.HISTORY_LEN, 22)).astype(np.float32)
    return obs, history


def test_accepts_good_artefact_and_matches_torch(good):
    d, modules = good
    art = load_artefact(d)
    assert art.controller == 'student' and art.git_sha == '0123456789abcdef'
    assert len(art.policy_sha) == 16
    obs, history = random_inputs()
    with torch.no_grad():
        z = modules['adapter'](torch.from_numpy(history)[None])
        a = modules['policy'](torch.from_numpy(obs)[None], z)
        rho = modules['critic'](torch.from_numpy(obs)[None], z)
    latent = art.latent(history)
    np.testing.assert_allclose(latent, z.numpy()[0], atol=1e-5)
    np.testing.assert_allclose(art.action(obs, latent), a.numpy()[0], atol=1e-5)
    assert art.risk(obs, latent) == pytest.approx(float(rho), abs=1e-5)

    out = OnnxController(art).step(ControlInput(obs, history, 0.05))
    v, w = obs_spec.action_to_twist(a.numpy()[0])
    assert (out.v, out.w) == pytest.approx((v, w), abs=1e-5)


def test_no_adapter_feeds_zero_latent_and_no_critic_gives_nan(tmp_path):
    modules = export_artefact(str(tmp_path), controller='B2', adapter=False, critic=False)
    art = load_artefact(str(tmp_path))
    obs, history = random_inputs()
    assert not art.latent(history).any()
    with torch.no_grad():
        a = modules['policy'](torch.from_numpy(obs)[None], torch.zeros(1, 8))
    np.testing.assert_allclose(art.action(obs, art.latent(history)), a.numpy()[0], atol=1e-5)
    assert np.isnan(art.risk(obs, np.zeros(8, np.float32)))


def test_refuses_wrong_obs_spec_hash_in_manifest(tmp_path):
    export_artefact(str(tmp_path), manifest_hash='deadbeefdeadbeef')
    with pytest.raises(ArtefactError, match='obs_spec_hash'):
        load_artefact(str(tmp_path))


def test_refuses_wrong_obs_spec_hash_in_file_metadata(tmp_path):
    export_artefact(str(tmp_path), file_hash='deadbeefdeadbeef')
    with pytest.raises(ArtefactError, match='obs_spec_hash'):
        load_artefact(str(tmp_path))


def test_refuses_tampered_file(tmp_path):
    export_artefact(str(tmp_path))
    with open(tmp_path / 'policy.onnx', 'ab') as f:
        f.write(b'\0')
    with pytest.raises(ArtefactError, match='sha256'):
        load_artefact(str(tmp_path))


def test_refuses_tampered_manifest_sha(tmp_path):
    export_artefact(str(tmp_path))
    manifest = json.loads((tmp_path / 'manifest.json').read_text())
    manifest['sha256']['critic.onnx'] = '0' * 64
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ArtefactError, match='critic.onnx: sha256'):
        load_artefact(str(tmp_path))


def test_refuses_missing_sha_entry_and_path_escape(tmp_path):
    export_artefact(str(tmp_path))
    manifest = json.loads((tmp_path / 'manifest.json').read_text())
    adapter_sha = manifest['sha256'].pop('adapter.onnx')
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ArtefactError, match='no entry for adapter.onnx'):
        load_artefact(str(tmp_path))
    manifest['sha256']['adapter.onnx'] = adapter_sha
    manifest['files']['policy'] = '../policy.onnx'
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ArtefactError, match='plain file name'):
        load_artefact(str(tmp_path))


def test_refuses_wrong_io_names(tmp_path):
    export_artefact(str(tmp_path), input_names=['x', 'latent'])
    with pytest.raises(ArtefactError, match='graph I/O'):
        load_artefact(str(tmp_path))


def test_refuses_missing_manifest(tmp_path):
    with pytest.raises(ArtefactError, match='manifest.json'):
        load_artefact(str(tmp_path))
    assert not os.listdir(tmp_path)
