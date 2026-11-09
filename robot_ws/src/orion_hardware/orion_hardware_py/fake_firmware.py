"""
Fake ORION firmware for tests and bench work without a microcontroller.

Speaks the real protocol on a serial device (typically one end of a `socat` PTY pair). The
wheel model is deliberately simple - each wheel tracks its target speed with a first-order lag -
because this is a test double for the link, not a copy of the firmware control law (that is
unit-tested in firmware/test). It does implement the externally visible contract: STATE at
100 Hz, 200 ms watchdog, e-stop latch with magic clear, SET_PARAM/PARAM_ACK, PING/PONG, STATS.

    ros2 run orion_hardware fake_firmware --port /tmp/orion_fw
"""

import argparse
from dataclasses import dataclass
import math
import os
import select
import struct
import sys
import termios
import threading
import time
import tty
from typing import List, Optional

from . import protocol as p

# Defaults mirror firmware/lib/orion_core/params.hpp.
DEFAULT_PARAMS = [0.10, 1.0, 0.0, 0.15, 1.0, 14.4, 0.1, 2048.0, 30.0, 10.5, 0.01]
PARAM_RANGES = [(0, 10), (0, 100), (0, 1), (0, 10), (0, 1), (0.5, 40), (0, 1), (1, 1e6),
                (0.01, 1000), (0, 60), (0, 0.5)]


@dataclass
class ReceivedCmd:
    t: float          # time.monotonic() at reception, s
    seq: int
    cmd: p.Cmd


class FakeFirmware:
    """Threaded fake firmware bound to a serial device path."""

    WATCHDOG_S = 0.2
    STATE_PERIOD_S = 0.01
    STATS_PERIOD_S = 1.0

    def __init__(self, port: str, current_sensed: bool = True, vbat_v: float = 12.0,
                 tau_s: float = 0.05, fw_git_hash: int = 0x0FA4E000):
        """`tau_s`: wheel speed lag time constant in s; `vbat_v`: reported battery voltage."""
        self.port = port
        self.current_sensed = current_sensed
        self.vbat_v = vbat_v
        self.tau_s = tau_s
        self.fw_git_hash = fw_git_hash
        self.params: List[float] = list(DEFAULT_PARAMS)
        self.cmds: List[ReceivedCmd] = []
        self.logs_sent: List[str] = []
        self.estop_frames = 0
        self.estop_latched = False
        self.watchdog_tripped = True
        self.stats = p.Stats()
        self._speed = [0.0] * 4       # rad/s
        self._position = [0.0] * 4    # rad
        self._target = [0.0] * 4      # rad/s
        self._enabled = False
        self._last_cmd_seq: Optional[int] = None
        self._last_cmd_t = 0.0
        self._decoder = p.FrameDecoder()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._fd: Optional[int] = None
        self._encoder: Optional[p.Encoder] = None
        self._t0 = time.monotonic()

    # ---------------------------------------------------------------- lifecycle
    def start(self) -> 'FakeFirmware':
        self._fd = os.open(self.port, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
        tty.setraw(self._fd, termios.TCSANOW)
        self._encoder = p.Encoder(self._write)
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name='fake_firmware', daemon=True)
        self._thread.start()
        return self

    def stop(self):
        """Stop answering, as if the board lost power. The device stays open until close()."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def close(self):
        self.stop()
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None

    # ---------------------------------------------------------------- inspection
    def received_cmds(self) -> List[ReceivedCmd]:
        with self._lock:
            return list(self.cmds)

    def wheel_speeds(self) -> List[float]:
        with self._lock:
            return list(self._speed)

    # ---------------------------------------------------------------- internals
    def _write(self, data: bytes):
        try:
            os.write(self._fd, data)
        except BlockingIOError:
            pass  # host is not reading; drop like a full UART FIFO would
        except OSError:
            self._stop.set()  # the device went away (PTY closed): behave like a dead board

    def _send_log(self, text: str):
        self.logs_sent.append(text)
        self._encoder.send(p.MSG_LOG, text.encode('ascii')[:p.MAX_PAYLOAD])

    def _run(self):
        last = time.monotonic()
        next_state = last + self.STATE_PERIOD_S
        next_stats = last + self.STATS_PERIOD_S
        self._last_cmd_t = last
        self._send_log('ORION fake firmware up')
        while not self._stop.is_set():
            ready, _, _ = select.select([self._fd], [], [], 0.001)
            if ready:
                try:
                    data = os.read(self._fd, 4096)
                except (BlockingIOError, OSError):
                    data = b''
                for frame in self._decoder.feed(data):
                    self._handle(frame)
            now = time.monotonic()
            self._step(now - last, now)
            last = now
            if now >= next_state:
                next_state += self.STATE_PERIOD_S
                if now > next_state:
                    next_state = now + self.STATE_PERIOD_S
                self._send_state(now)
            if now >= next_stats:
                next_stats = now + self.STATS_PERIOD_S
                self.stats.rx_frames_ok = self._decoder.frames_ok
                self.stats.rx_crc_errors = self._decoder.crc_errors
                self.stats.rx_framing_errors = self._decoder.framing_errors
                self._encoder.send_msg(self.stats)

    def _step(self, dt: float, now: float):
        with self._lock:
            if (not self.watchdog_tripped and not self.estop_latched
                    and now - self._last_cmd_t >= self.WATCHDOG_S):
                self.watchdog_tripped = True
                self.stats.watchdog_trips += 1
                self._send_log('watchdog: no CMD for 200 ms')
            active = self._enabled and not self.watchdog_tripped and not self.estop_latched
            alpha = min(1.0, dt / self.tau_s)
            for i in range(4):
                target = self._target[i] if active else 0.0
                self._speed[i] += alpha * (target - self._speed[i])
                self._position[i] += self._speed[i] * dt

    def _send_state(self, now: float):
        with self._lock:
            s = p.State()
            s.t_us = int((now - self._t0) * 1e6) & 0xFFFFFFFF
            s.position_mrad = [_wrap_i32(round(x * 1000)) for x in self._position]
            s.velocity_mrad_s = [_sat_i16(round(w * 1000)) for w in self._speed]
            active = self._enabled and not self.watchdog_tripped and not self.estop_latched
            s.duty = [_sat_i16(round(p.DUTY_FULL_SCALE * min(1.0, abs(w) / 6.0)
                                     * math.copysign(1.0, w))) if active else 0
                      for w in self._speed]
            s.current_ma = ([_sat_i16(round(0.5 * w * 1000)) for w in self._speed]
                            if self.current_sensed else [p.CURRENT_NOT_AVAILABLE] * 4)
            s.vbat_mv = int(self.vbat_v * 1000)
            s.faults = ((p.FAULT_WATCHDOG if self.watchdog_tripped else 0)
                        | (p.FAULT_ESTOP_LATCHED if self.estop_latched else 0))
            s.status = ((p.STATUS_ENABLED if active else 0)
                        | (p.STATUS_CURRENT_SENSED if self.current_sensed else 0)
                        | p.STATUS_VBAT_SENSED)
        self._encoder.send_msg(s)

    def _handle(self, frame: p.Frame):
        msg = p.decode_message(frame)
        if msg is None:
            self._decoder.framing_errors += 1
            return
        if frame.msg_type == p.MSG_CMD:
            self._handle_cmd(frame.seq, msg)
        elif frame.msg_type == p.MSG_PING:
            crc = 0xFFFF
            for v in self.params:
                crc = p.crc16_ccitt_false(struct.pack('<f', v), crc)
            self._encoder.send_msg(p.Pong(p.PROTOCOL_VERSION, self.fw_git_hash, crc))
        elif frame.msg_type == p.MSG_SET_PARAM:
            self._encoder.send_msg(self._set_param(msg))
        elif frame.msg_type == p.MSG_ESTOP:
            with self._lock:
                self.estop_frames += 1
                if not self.estop_latched:
                    self.estop_latched = True
                    self._send_log('ESTOP latched')
        elif frame.msg_type == p.MSG_ESTOP_CLEAR:
            with self._lock:
                if msg.magic == p.ESTOP_CLEAR_MAGIC and self.estop_latched:
                    self.estop_latched = False
                    self.watchdog_tripped = True
                    self._enabled = False
                    self._send_log('ESTOP cleared')

    def _handle_cmd(self, seq: int, cmd: p.Cmd):
        with self._lock:
            self.cmds.append(ReceivedCmd(time.monotonic(), seq, cmd))
            if self._last_cmd_seq is not None:
                self.stats.cmd_seq_gaps += p.seq_gap(self._last_cmd_seq, seq)
            self._last_cmd_seq = seq
            if self.estop_latched:
                return
            self._last_cmd_t = time.monotonic()
            self.watchdog_tripped = False
            self._enabled = bool(cmd.flags & p.CMD_FLAG_ENABLE)
            self._target = [w * 1e-3 if self._enabled else 0.0 for w in cmd.w_target_mrad_s]

    def _set_param(self, req: p.SetParam) -> p.ParamAck:
        if req.param_id >= len(self.params):
            return p.ParamAck(req.param_id, float('nan'), p.PARAM_UNKNOWN_ID)
        lo, hi = PARAM_RANGES[req.param_id]
        value = struct.unpack('<f', struct.pack('<f', req.value))[0]
        if not math.isfinite(value) or not lo <= value <= hi:
            return p.ParamAck(req.param_id, self.params[req.param_id], p.PARAM_OUT_OF_RANGE)
        self.params[req.param_id] = value
        return p.ParamAck(req.param_id, value, p.PARAM_OK)


def _sat_i16(v: int) -> int:
    return max(-32768, min(32767, int(v)))


def _wrap_i32(v: int) -> int:
    return ((int(v) + 2**31) % 2**32) - 2**31


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--port', required=True, help='serial device, e.g. a socat PTY link')
    parser.add_argument('--no-current', action='store_true',
                        help='report current as not available (0x7FFF)')
    args = parser.parse_args(argv)
    fw = FakeFirmware(args.port, current_sensed=not args.no_current).start()
    print(f'fake firmware on {args.port}; Ctrl-C to stop', file=sys.stderr)
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        fw.close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
