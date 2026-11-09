"""robot.launch.py forwards the drive-train launch arguments to the URDF xacro."""

import importlib.util
import os
import re
import xml.etree.ElementTree as ET

from ament_index_python.packages import get_package_share_directory
from launch import LaunchContext
from launch.actions import DeclareLaunchArgument
import pytest

LAUNCH_FILE = os.path.join(os.path.dirname(__file__), '..', 'launch', 'robot.launch.py')
XACRO_DIR = os.path.join(get_package_share_directory('orion_description'), 'urdf')


@pytest.fixture(scope='module')
def launch_module():
    spec = importlib.util.spec_from_file_location('orion_robot_launch', LAUNCH_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def render(module, **overrides):
    """Hardware params of the URDF that robot.launch.py builds with the given launch args."""
    context = LaunchContext()
    for entity in module.generate_launch_description().entities:
        if isinstance(entity, DeclareLaunchArgument):
            entity.visit(context)   # sets the declared default
    for name, value in overrides.items():
        context.launch_configurations[name] = value
    xacro_file = os.path.join(XACRO_DIR, 'robot', 'orion.urdf.xacro')
    urdf = module.robot_description_command(xacro_file).perform(context)
    rc = ET.fromstring(urdf).find('ros2_control')
    assert rc.find('hardware/plugin').text.strip() == 'orion_hardware/OrionSystem'
    return {e.get('name'): e.text for e in rc.findall('hardware/param')}


def test_non_default_port_and_encoder_cpr_reach_the_hardware_params(launch_module):
    params = render(launch_module, port='/dev/orion_fw', encoder_cpr='4096',
                    brake_on_idle='true')
    assert params['port'] == '/dev/orion_fw'
    assert float(params['encoder_cpr']) == 4096
    assert params['brake_on_idle'] == 'true'


def test_every_drive_arg_is_declared_and_forwarded(launch_module):
    declared = {e.name for e in launch_module.generate_launch_description().entities
                if isinstance(e, DeclareLaunchArgument)}
    assert set(launch_module.DRIVE_ARGS) <= declared
    values = {'port': '/tmp/orion_host', 'baud': '115200', 'encoder_cpr': '1234',
              'gear_ratio': '17', 'kt': '0.03', 'motor_resistance': '0.9',
              'brake_on_idle': 'true'}
    assert set(values) == set(launch_module.DRIVE_ARGS)
    params = render(launch_module, **values)
    assert {k: params[k] for k in values} == values


def test_launch_defaults_equal_the_xacro_defaults(launch_module):
    with open(os.path.join(XACRO_DIR, 'control', 'ros2_control.xacro')) as f:
        xacro_args = dict(re.findall(r'<xacro:arg name="(\w+)" default="([^"]*)"/>', f.read()))
    for name, (default, _) in launch_module.DRIVE_ARGS.items():
        assert xacro_args[name] == default, name
