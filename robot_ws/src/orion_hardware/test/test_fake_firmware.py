"""The fake firmware honours the protocol contract the integration test relies on."""

import os
import pty
import select
import time

from orion_hardware_py import protocol as p
from orion_hardware_py.fake_firmware import FakeFirmware
import pytest


class Host:
    """Host end of a PTY pair talking to a FakeFirmware on the other end."""

    def __init__(self, fd):
        self.fd = fd
        self.decoder = p.FrameDecoder()
        self.encoder = p.Encoder(lambda b: os.write(fd, b))
        self.frames = []

    def pump(self, seconds):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            ready, _, _ = select.select([self.fd], [], [], 0.005)
            if ready:
                self.frames += self.decoder.feed(os.read(self.fd, 4096))

    def of_type(self, msg_type):
        return [p.decode_message(f) for f in self.frames if f.msg_type == msg_type]

    def drive(self, seconds, mrad_s, flags=p.CMD_FLAG_ENABLE):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            self.encoder.send_msg(p.Cmd([mrad_s] * 4, flags))
            self.pump(0.01)


@pytest.fixture
def link():
    host_fd, fw_fd = pty.openpty()
    fw = FakeFirmware(os.ttyname(fw_fd), current_sensed=False).start()
    host = Host(host_fd)
    yield host, fw
    fw.close()
    os.close(fw_fd)
    os.close(host_fd)


def test_state_rate_ping_and_params(link):
    host, fw = link
    host.encoder.send(p.MSG_PING)
    host.encoder.send_msg(p.SetParam(p.PARAM_GEAR_RATIO, 51.0))
    host.encoder.send_msg(p.SetParam(p.PARAM_MAX_DUTY, 3.0))
    host.pump(1.0)
    states = host.of_type(p.MSG_STATE)
    assert 80 <= len(states) <= 110
    assert states[-1].current_ma == [p.CURRENT_NOT_AVAILABLE] * 4
    assert host.of_type(p.MSG_PONG)[0].proto_ver == p.PROTOCOL_VERSION
    acks = host.of_type(p.MSG_PARAM_ACK)
    assert acks[0] == p.ParamAck(p.PARAM_GEAR_RATIO, 51.0, p.PARAM_OK)
    assert acks[1].status == p.PARAM_OUT_OF_RANGE


def test_tracks_command_then_watchdog_stops_it(link):
    host, fw = link
    host.drive(0.5, 2000)
    assert fw.wheel_speeds()[0] == pytest.approx(2.0, abs=0.05)
    assert host.of_type(p.MSG_STATE)[-1].status & p.STATUS_ENABLED
    host.pump(0.35)  # > 200 ms without CMD
    state = host.of_type(p.MSG_STATE)[-1]
    assert state.faults & p.FAULT_WATCHDOG
    assert fw.stats.watchdog_trips == 1
    assert abs(fw.wheel_speeds()[0]) < 0.2


def test_estop_latch_and_magic_clear(link):
    host, fw = link
    host.drive(0.2, 1000)
    host.encoder.send(p.MSG_ESTOP)
    host.drive(0.3, 1000)
    assert fw.estop_latched
    assert host.of_type(p.MSG_STATE)[-1].faults & p.FAULT_ESTOP_LATCHED
    host.encoder.send_msg(p.EstopClear(0x12345678))
    host.pump(0.05)
    assert fw.estop_latched
    host.encoder.send_msg(p.EstopClear())
    host.pump(0.05)
    assert not fw.estop_latched


def test_counts_command_sequence_gaps(link):
    host, fw = link
    host.encoder.send_msg(p.Cmd())
    host.encoder.seq += 3
    host.encoder.send_msg(p.Cmd())
    host.pump(0.05)
    assert fw.stats.cmd_seq_gaps == 3
