"""config/terrain.yaml mirrors TerrainParams; enum values mirror orion_msgs/TerrainState."""

from dataclasses import asdict
import os

from orion_terrain.effort_model import EffortSource
from orion_terrain.params import DEFAULT_JOINT_NAMES, DEFAULT_TOPICS, TerrainParams
from orion_terrain.speed import SpeedSource
import pytest
import yaml

CONFIG = os.path.join(os.path.dirname(__file__), '..', 'config', 'terrain.yaml')


def load_yaml():
    with open(CONFIG) as f:
        return yaml.safe_load(f)['terrain_node']['ros__parameters']


def test_yaml_matches_dataclass_defaults():
    p = load_yaml()
    values = {k: p[k] for k in asdict(TerrainParams())}
    assert TerrainParams.from_mapping(values) == TerrainParams()
    assert p['topics'] == DEFAULT_TOPICS
    assert tuple(p['joint_names']) == DEFAULT_JOINT_NAMES
    node_only = {'topics', 'joint_names', 'sim', 'effort_model_path'}
    extra = set(p) - set(asdict(TerrainParams())) - node_only
    assert not extra, f'parameters in YAML the node never declares: {extra}'


def test_unknown_parameter_rejected():
    with pytest.raises(KeyError):
        TerrainParams.from_mapping({'q_acel_mps2': 0.2})


def test_enums_match_message_constants():
    msg = pytest.importorskip('orion_msgs.msg').TerrainState
    for s in SpeedSource:
        assert getattr(msg, f'SPEED_SOURCE_{s.name}') == s.value
    for s in EffortSource:
        assert getattr(msg, f'EFFORT_SOURCE_{s.name}') == s.value
