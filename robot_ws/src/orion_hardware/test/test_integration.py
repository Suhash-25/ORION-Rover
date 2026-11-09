"""
Full-stack test on a socat PTY pair, no hardware and no Gazebo.

controller_manager loads OrionSystem from orion_description's xacro (hardware:=serial), the
fake firmware answers on the other PTY end, diff_drive_controller turns a TwistStamped into
wheel targets. Checks: CMD frames carry the right targets, STATE moves /joint_states,
RoverHealth is published, /orion/estop reaches the firmware, and a silent firmware makes
read() fail so the hardware component leaves the active state.
Run with a dedicated ROS_DOMAIN_ID (CMake sets 72).
"""

import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time

from ament_index_python.packages import get_package_prefix, get_package_share_directory
from controller_manager_msgs.srv import ListHardwareComponents
from geometry_msgs.msg import TwistStamped
from orion_hardware_py import protocol as p
from orion_hardware_py.fake_firmware import FakeFirmware
import pytest
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool
import yaml

WHEEL_SEPARATION = 0.52  # orion_control/config/controllers.yaml
WHEEL_RADIUS = 0.12
STATE_TIMEOUT_S = 0.5
LEFT = ('front_left_wheel_joint', 'rear_left_wheel_joint')
RIGHT = ('front_right_wheel_joint', 'rear_right_wheel_joint')


def wait_for(predicate, timeout, node=None, period=0.02):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if node is not None:
            rclpy.spin_once(node, timeout_sec=period)
        else:
            time.sleep(period)
        if predicate():
            return True
    return False


def stop_process(proc):
    if proc.poll() is None:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


@pytest.fixture(scope='module')
def stack():
    if shutil.which('socat') is None:
        pytest.fail('socat not installed (test_depend socat)')
    tmp = tempfile.mkdtemp(prefix='orion_hw_it_')
    fw_port, host_port = os.path.join(tmp, 'fw'), os.path.join(tmp, 'host')
    procs = []
    log_path = os.path.join(tmp, 'controller_manager.log')
    cm_log = open(log_path, 'w')
    fw = None
    try:
        procs.append(subprocess.Popen(
            ['socat', f'pty,raw,echo=0,link={fw_port}', f'pty,raw,echo=0,link={host_port}'],
            stderr=subprocess.DEVNULL))
        assert wait_for(lambda: os.path.exists(fw_port) and os.path.exists(host_port), 10)
        fw = FakeFirmware(fw_port).start()

        xacro_file = os.path.join(get_package_share_directory('orion_description'),
                                  'urdf', 'robot', 'orion.urdf.xacro')
        urdf = subprocess.run(['xacro', xacro_file, 'hardware:=serial', f'port:={host_port}'],
                              check=True, capture_output=True, text=True).stdout
        # The 100 ms staleness limit is unit-tested (test_orion_system). Here the fake firmware
        # is a Python thread on a shared, loaded machine, so allow scheduling hiccups.
        port_param = f'<param name="port">{host_port}</param>'
        assert port_param in urdf
        urdf = urdf.replace(
            port_param, port_param + f'<param name="state_timeout_s">{STATE_TIMEOUT_S}</param>')
        rsp_params = os.path.join(tmp, 'rsp.yaml')
        with open(rsp_params, 'w') as f:
            yaml.safe_dump({'robot_state_publisher': {'ros__parameters': {
                'robot_description': urdf}}}, f)
        controllers = os.path.join(get_package_share_directory('orion_control'),
                                   'config', 'controllers.yaml')
        rsp = os.path.join(get_package_prefix('robot_state_publisher'),
                           'lib', 'robot_state_publisher', 'robot_state_publisher')
        cm_dir = os.path.join(get_package_prefix('controller_manager'),
                              'lib', 'controller_manager')
        procs.append(subprocess.Popen([rsp, '--ros-args', '--params-file', rsp_params],
                                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        procs.append(subprocess.Popen(
            [os.path.join(cm_dir, 'ros2_control_node'), '--ros-args', '--params-file',
             controllers], stdout=cm_log, stderr=subprocess.STDOUT))
        spawner = subprocess.run(
            [sys.executable, os.path.join(cm_dir, 'spawner'), 'joint_state_broadcaster',
             'diff_drive_controller', '--param-file', controllers,
             '--controller-manager-timeout', '60'],
            capture_output=True, text=True, timeout=120)
        assert spawner.returncode == 0, spawner.stdout + spawner.stderr + open(log_path).read()

        rclpy.init()
        node = Node('orion_hardware_integration_test')
        yield {'fw': fw, 'node': node, 'cm_log': log_path,
               'log_tail': lambda: open(log_path).read()[-3000:]}
        node.destroy_node()
        rclpy.shutdown()
    finally:
        if fw is not None:
            fw.close()
        for proc in reversed(procs):
            stop_process(proc)
        cm_log.close()
        shutil.rmtree(tmp, ignore_errors=True)


def test_cmd_vel_produces_wheel_targets(stack):
    fw, node = stack['fw'], stack['node']
    pub = node.create_publisher(TwistStamped, '/diff_drive_controller/cmd_vel', 10)
    v, w = 0.24, 0.5

    def publish():
        msg = TwistStamped()
        msg.header.stamp = node.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.twist.linear.x, msg.twist.angular.z = v, w
        pub.publish(msg)

    # Keep commanding at 20 Hz for the following tests too (cmd_vel_timeout is 0.5 s).
    stack['cmd_timer'] = node.create_timer(0.05, publish)
    expected_left = round((v - w * WHEEL_SEPARATION / 2) / WHEEL_RADIUS * 1000)
    expected_right = round((v + w * WHEEL_SEPARATION / 2) / WHEEL_RADIUS * 1000)

    def latest():
        cmds = fw.received_cmds()
        return cmds[-1].cmd if cmds else None

    assert wait_for(lambda: latest() is not None and latest().w_target_mrad_s[0] != 0, 15,
                    node), 'no non-zero CMD reached the firmware\n' + stack['log_tail']()
    wait_for(lambda: False, 0.5, node)  # let the controller settle on the command
    cmd = latest()
    fl, fr, rl, rr = cmd.w_target_mrad_s
    assert abs(fl - expected_left) <= 1 and abs(rl - expected_left) <= 1, cmd
    assert abs(fr - expected_right) <= 1 and abs(rr - expected_right) <= 1, cmd
    assert cmd.flags & p.CMD_FLAG_ENABLE

    # The CMD stream runs at the controller rate (100 Hz) with no sequence gaps.
    recent = [c for c in fw.received_cmds() if c.t > time.monotonic() - 1.0]
    assert 70 <= len(recent) <= 120, len(recent)
    assert fw.stats.cmd_seq_gaps == 0


def test_state_moves_joint_states(stack):
    node = stack['node']
    latest = {}

    def on_js(msg):
        latest.update({n: (pos, vel) for n, pos, vel in zip(msg.name, msg.position,
                                                            msg.velocity)})

    sub = node.create_subscription(JointState, '/joint_states', on_js, 10)
    assert wait_for(lambda: set(LEFT + RIGHT) <= set(latest), 10, node)
    start = dict(latest)
    wait_for(lambda: False, 1.0, node)
    node.destroy_subscription(sub)
    left_w = (0.24 - 0.5 * WHEEL_SEPARATION / 2) / WHEEL_RADIUS
    right_w = (0.24 + 0.5 * WHEEL_SEPARATION / 2) / WHEEL_RADIUS
    for joint, w in [(j, left_w) for j in LEFT] + [(j, right_w) for j in RIGHT]:
        assert latest[joint][1] == pytest.approx(w, abs=0.05), joint
        assert latest[joint][0] > start[joint][0] + 0.5 * w, joint


def test_rover_health_is_published(stack):
    from orion_msgs.msg import RoverHealth
    node = stack['node']
    msgs = []
    sub = node.create_subscription(RoverHealth, '/orion/rover_health', msgs.append, 10)
    assert wait_for(lambda: len(msgs) >= 5, 5, node), 'no RoverHealth'
    t0 = time.monotonic()
    n0 = len(msgs)
    wait_for(lambda: False, 1.0, node)
    rate = (len(msgs) - n0) / (time.monotonic() - t0)
    node.destroy_subscription(sub)
    assert 6 <= rate <= 14, rate
    h = msgs[-1]
    assert h.battery_v == pytest.approx(12.0)
    assert h.current_sensed
    assert h.fw_fault_flags & p.FAULT_WATCHDOG == 0
    assert h.duty[0] > 0


def test_estop_topic_reaches_firmware(stack):
    fw, node = stack['fw'], stack['node']
    pub = node.create_publisher(Bool, '/orion/estop', 10)

    def publish_until(value, predicate):
        def step():
            pub.publish(Bool(data=value))
            return predicate()
        return wait_for(step, 5, node, period=0.05)

    assert publish_until(True, lambda: fw.estop_latched), 'ESTOP did not reach the firmware'
    assert publish_until(False, lambda: not fw.estop_latched), 'ESTOP_CLEAR did not arrive'
    # Wheel commands resume after the clear.
    n = len(fw.received_cmds())
    assert wait_for(lambda: len(fw.received_cmds()) > n + 10, 3, node)


def test_silent_firmware_deactivates_hardware(stack):
    fw, node = stack['fw'], stack['node']
    client = node.create_client(ListHardwareComponents,
                                '/controller_manager/list_hardware_components')
    assert client.wait_for_service(timeout_sec=10)

    def hardware_state():
        future = client.call_async(ListHardwareComponents.Request())
        rclpy.spin_until_future_complete(node, future, timeout_sec=5)
        comps = {c.name: c.state.label for c in future.result().component}
        return comps.get('orion_system')

    assert hardware_state() == 'active'
    fw.stop()  # the board stops talking; the PTY stays open
    assert wait_for(lambda: hardware_state() != 'active', 10, node), hardware_state()
    with open(stack['cm_log']) as f:
        assert 'STATE is stale' in f.read()
