"""orion_description's xacro selects OrionSystem with hardware:=serial and Gazebo by default."""

import os
import subprocess
import xml.etree.ElementTree as ET

from ament_index_python.packages import get_package_share_directory
import pytest


def expand(*args):
    xacro_file = os.path.join(
        get_package_share_directory('orion_description'), 'urdf', 'robot', 'orion.urdf.xacro')
    out = subprocess.run(['xacro', xacro_file, *args], capture_output=True, text=True)
    return out


def hardware_block(urdf_text):
    rc = ET.fromstring(urdf_text).find('ros2_control')
    assert rc is not None
    return rc


def test_default_is_gazebo():
    out = expand()
    assert out.returncode == 0, out.stderr
    rc = hardware_block(out.stdout)
    assert rc.find('hardware/plugin').text.strip() == 'gz_ros2_control/GazeboSimSystem'
    assert rc.findall('hardware/param') == []


def test_serial_selects_orion_system_with_params():
    out = expand('hardware:=serial', 'port:=/dev/ttyORION', 'encoder_cpr:=4096',
                 'gear_ratio:=51', 'kt:=0.02', 'motor_resistance:=0.8')
    assert out.returncode == 0, out.stderr
    rc = hardware_block(out.stdout)
    assert rc.find('hardware/plugin').text.strip() == 'orion_hardware/OrionSystem'
    params = {e.get('name'): e.text for e in rc.findall('hardware/param')}
    assert params['port'] == '/dev/ttyORION'
    assert params['baud'] == '921600'
    assert float(params['encoder_cpr']) == 4096
    assert float(params['gear_ratio']) == 51
    assert float(params['kt']) == pytest.approx(0.02)
    assert float(params['motor_resistance']) == pytest.approx(0.8)
    indices = sorted(int(j.find("param[@name='motor_index']").text)
                     for j in rc.findall('joint'))
    assert indices == [0, 1, 2, 3]


def test_unknown_hardware_value_fails_loudly():
    out = expand('hardware:=bogus')
    assert out.returncode != 0
    assert "hardware must be 'gazebo' or 'serial'" in out.stderr
