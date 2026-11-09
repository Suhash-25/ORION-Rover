"""The YAML files must name exactly the SafetyConfig fields, so a typo cannot disable a rule."""

from dataclasses import fields
import os

from orion_safety.monitor import SafetyConfig
import yaml

CONFIG_DIR = os.path.join(os.path.dirname(__file__), '..', 'config')
NODE_ONLY = {'rate_hz', 'wheel_joints', 'geofence_frame'}
FIELDS = {f.name for f in fields(SafetyConfig)}


def load(name):
    with open(os.path.join(CONFIG_DIR, name)) as f:
        return yaml.safe_load(f)['orion_safety']['ros__parameters']


def test_rover_config_is_complete_and_valid():
    params = load('safety.yaml')
    assert set(params) - NODE_ONLY == FIELDS
    cfg = SafetyConfig(**{k: v for k, v in params.items() if k in FIELDS})
    assert cfg == SafetyConfig(), 'safety.yaml and SafetyConfig defaults disagree'


def test_sim_overrides_are_known_keys_and_valid():
    sim = load('safety_sim.yaml')
    assert set(sim) <= FIELDS | NODE_ONLY
    merged = {**load('safety.yaml'), **sim}
    cfg = SafetyConfig(**{k: v for k, v in merged.items() if k in FIELDS})
    assert not cfg.geofence_enabled


def test_float_parameters_are_written_as_floats():
    # ROS refuses an integer for a double parameter, which would crash the node at startup.
    for name in ('safety.yaml', 'safety_sim.yaml'):
        for key, value in load(name).items():
            if key in FIELDS and isinstance(getattr(SafetyConfig(), key), float):
                assert isinstance(value, float), f'{name}: {key} must be written as a float'
