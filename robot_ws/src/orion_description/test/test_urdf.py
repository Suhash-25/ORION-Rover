"""The ORION xacro expands to a valid URDF with the joints the controllers expect."""

import os
import shutil
import subprocess
import xml.etree.ElementTree as ET

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
XACRO = os.path.join(HERE, '..', 'urdf', 'robot', 'orion.urdf.xacro')
WHEEL_JOINTS = {
    'front_left_wheel_joint', 'front_right_wheel_joint',
    'rear_left_wheel_joint', 'rear_right_wheel_joint',
}


@pytest.fixture(scope='module')
def urdf(tmp_path_factory):
    out = subprocess.run(['xacro', XACRO], capture_output=True, text=True, check=True).stdout
    path = tmp_path_factory.mktemp('urdf') / 'orion.urdf'
    path.write_text(out)
    return path


def test_check_urdf_accepts_it(urdf):
    if shutil.which('check_urdf') is None:
        pytest.skip('check_urdf (liburdfdom-tools) not installed')
    result = subprocess.run(['check_urdf', str(urdf)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_wheel_joints_exist_and_are_continuous(urdf):
    root = ET.parse(urdf).getroot()
    joints = {j.get('name'): j.get('type') for j in root.findall('joint')}
    for name in WHEEL_JOINTS:
        assert joints.get(name) == 'continuous', f'{name}: {joints.get(name)}'


def test_ros2_control_exposes_velocity_command_and_effort_state(urdf):
    root = ET.parse(urdf).getroot()
    rc = root.find('ros2_control')
    assert rc is not None, 'no <ros2_control> block'
    for joint in rc.findall('joint'):
        assert joint.get('name') in WHEEL_JOINTS
        cmds = {c.get('name') for c in joint.findall('command_interface')}
        states = {s.get('name') for s in joint.findall('state_interface')}
        assert cmds == {'velocity'}
        assert {'position', 'velocity', 'effort'} <= states


def test_lidar_frame_exists(urdf):
    root = ET.parse(urdf).getroot()
    assert 'lidar_link' in {link.get('name') for link in root.findall('link')}


def test_imu_frame_exists(urdf):
    root = ET.parse(urdf).getroot()
    assert 'imu_link' in {link.get('name') for link in root.findall('link')}
