"""
Python reference implementation of the ORION serial protocol (docs/PROTOCOL.md).

Mirrors firmware/lib/orion_proto/orion_proto.hpp byte for byte; both are checked against the
same golden vectors (firmware/lib/orion_proto/test_vectors.json). Used by the fake firmware
and by tests, not on the rover's control path.
"""

from dataclasses import dataclass, field
import struct
from typing import Callable, List, Optional, Tuple

PROTOCOL_VERSION = 1
MAX_PAYLOAD = 64
NUM_WHEELS = 4

MSG_CMD = 0x01
MSG_STATE = 0x02
MSG_SET_PARAM = 0x03
MSG_PARAM_ACK = 0x04
MSG_PING = 0x05
MSG_PONG = 0x06
MSG_ESTOP = 0x07
MSG_ESTOP_CLEAR = 0x08
MSG_LOG = 0x09
MSG_STATS = 0x0A

ESTOP_CLEAR_MAGIC = 0x0DE5C0DE
CURRENT_NOT_AVAILABLE = 0x7FFF
DUTY_FULL_SCALE = 10000

FAULT_WATCHDOG = 1
FAULT_OVERCURRENT = 2
FAULT_ENCODER = 4
FAULT_UNDERVOLTAGE = 8
FAULT_ESTOP_LATCHED = 16

STATUS_ENABLED = 1
STATUS_CURRENT_SENSED = 2
STATUS_VBAT_SENSED = 4
STATUS_BRAKE_ON_IDLE = 8

CMD_FLAG_ENABLE = 1
CMD_FLAG_BRAKE_ON_IDLE = 2

PARAM_OK = 0
PARAM_UNKNOWN_ID = 1
PARAM_OUT_OF_RANGE = 2

# SET_PARAM ids (units in PROTOCOL.md).
PARAM_KP = 0
PARAM_KI = 1
PARAM_KD = 2
PARAM_KFF = 3
PARAM_MAX_DUTY = 4
PARAM_CURRENT_LIMIT_A = 5
PARAM_CURRENT_LIMIT_TIME_S = 6
PARAM_ENCODER_CPR = 7
PARAM_GEAR_RATIO = 8
PARAM_VBAT_MIN_V = 9
PARAM_VEL_FILTER_TAU_S = 10
PARAM_COUNT = 11


def crc16_ccitt_false(data: bytes, crc: int = 0xFFFF) -> int:
    """CRC-16/CCITT-FALSE (poly 0x1021, init 0xFFFF, no reflection, no xor-out)."""
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) if crc & 0x8000 else (crc << 1)
            crc &= 0xFFFF
    return crc


def cobs_encode(data: bytes) -> bytes:
    """COBS-encode `data` (canonical form: no empty block after a full 254-byte block)."""
    out = bytearray([0])
    code_idx = 0
    code = 1
    for i, byte in enumerate(data):
        if byte == 0:
            out[code_idx] = code
            code = 1
            code_idx = len(out)
            out.append(0)
        else:
            out.append(byte)
            code += 1
            if code == 0xFF and i + 1 < len(data):
                out[code_idx] = code
                code = 1
                code_idx = len(out)
                out.append(0)
    out[code_idx] = code
    return bytes(out)


def cobs_decode(data: bytes) -> Optional[bytes]:
    """Decode one COBS block (no delimiter). Returns None if it is malformed."""
    out = bytearray()
    i = 0
    while i < len(data):
        code = data[i]
        i += 1
        if code == 0:
            return None
        for _ in range(code - 1):
            if i >= len(data) or data[i] == 0:
                return None
            out.append(data[i])
            i += 1
        if code != 0xFF and i < len(data):
            out.append(0)
    return bytes(out)


def encode_frame(msg_type: int, seq: int, payload: bytes = b'') -> bytes:
    """Return the wire bytes of one frame: COBS(type seq len payload crc16le) + 0x00."""
    if len(payload) > MAX_PAYLOAD:
        raise ValueError(f'payload of {len(payload)} B exceeds {MAX_PAYLOAD} B')
    raw = bytes([msg_type & 0xFF, seq & 0xFF, len(payload)]) + bytes(payload)
    raw += struct.pack('<H', crc16_ccitt_false(raw))
    return cobs_encode(raw) + b'\x00'


@dataclass
class Frame:
    msg_type: int
    seq: int
    payload: bytes


def parse_raw_frame(raw: bytes) -> Tuple[Optional[Frame], str]:
    """Validate an un-COBSed block. Returns (frame, '') or (None, 'framing'|'crc')."""
    if len(raw) < 5:
        return None, 'framing'
    length = raw[2]
    if length > MAX_PAYLOAD or len(raw) != 3 + length + 2:
        return None, 'framing'
    (crc,) = struct.unpack_from('<H', raw, 3 + length)
    if crc16_ccitt_false(raw[:3 + length]) != crc:
        return None, 'crc'
    return Frame(raw[0], raw[1], bytes(raw[3:3 + length])), ''


class FrameDecoder:
    """Streaming decoder with the same counters and rules as the C++ FrameDecoder."""

    MAX_ENCODED = 70

    def __init__(self):
        self._buf = bytearray()
        self._overflow = False
        self.frames_ok = 0
        self.crc_errors = 0
        self.framing_errors = 0

    def feed(self, data: bytes) -> List[Frame]:
        """Consume bytes; return every valid frame they completed."""
        frames = []
        for byte in data:
            if byte != 0:
                if len(self._buf) < self.MAX_ENCODED:
                    self._buf.append(byte)
                else:
                    self._overflow = True
                continue
            block, overflow = bytes(self._buf), self._overflow
            self._buf.clear()
            self._overflow = False
            if not block and not overflow:
                continue
            if overflow:
                self.framing_errors += 1
                continue
            raw = cobs_decode(block)
            if not raw:
                self.framing_errors += 1
                continue
            frame, error = parse_raw_frame(raw)
            if frame is None:
                if error == 'crc':
                    self.crc_errors += 1
                else:
                    self.framing_errors += 1
                continue
            self.frames_ok += 1
            frames.append(frame)
        return frames


# --------------------------------------------------------------------------------------------
# Payloads. Each dataclass has TYPE, FORMAT (struct, little-endian), pack() and unpack().
# --------------------------------------------------------------------------------------------

@dataclass
class Cmd:
    TYPE = MSG_CMD
    FORMAT = '<4hB'
    w_target_mrad_s: List[int] = field(default_factory=lambda: [0, 0, 0, 0])
    flags: int = 0

    def pack(self) -> bytes:
        return struct.pack(self.FORMAT, *self.w_target_mrad_s, self.flags)

    @classmethod
    def unpack(cls, payload: bytes) -> 'Cmd':
        v = struct.unpack(cls.FORMAT, payload)
        return cls(list(v[0:4]), v[4])


@dataclass
class State:
    TYPE = MSG_STATE
    FORMAT = '<I4i4h4h4hHBB'
    t_us: int = 0
    position_mrad: List[int] = field(default_factory=lambda: [0, 0, 0, 0])
    velocity_mrad_s: List[int] = field(default_factory=lambda: [0, 0, 0, 0])
    current_ma: List[int] = field(default_factory=lambda: [CURRENT_NOT_AVAILABLE] * 4)
    duty: List[int] = field(default_factory=lambda: [0, 0, 0, 0])
    vbat_mv: int = 0
    faults: int = 0
    status: int = 0

    def pack(self) -> bytes:
        return struct.pack(
            self.FORMAT, self.t_us & 0xFFFFFFFF, *self.position_mrad, *self.velocity_mrad_s,
            *self.current_ma, *self.duty, self.vbat_mv, self.faults, self.status)

    @classmethod
    def unpack(cls, payload: bytes) -> 'State':
        v = struct.unpack(cls.FORMAT, payload)
        return cls(v[0], list(v[1:5]), list(v[5:9]), list(v[9:13]), list(v[13:17]),
                   v[17], v[18], v[19])


@dataclass
class SetParam:
    TYPE = MSG_SET_PARAM
    FORMAT = '<Bf'
    param_id: int = 0
    value: float = 0.0

    def pack(self) -> bytes:
        return struct.pack(self.FORMAT, self.param_id, self.value)

    @classmethod
    def unpack(cls, payload: bytes) -> 'SetParam':
        return cls(*struct.unpack(cls.FORMAT, payload))


@dataclass
class ParamAck:
    TYPE = MSG_PARAM_ACK
    FORMAT = '<BfB'
    param_id: int = 0
    value: float = 0.0
    status: int = PARAM_OK

    def pack(self) -> bytes:
        return struct.pack(self.FORMAT, self.param_id, self.value, self.status)

    @classmethod
    def unpack(cls, payload: bytes) -> 'ParamAck':
        return cls(*struct.unpack(cls.FORMAT, payload))


@dataclass
class Pong:
    TYPE = MSG_PONG
    FORMAT = '<BIH'
    proto_ver: int = PROTOCOL_VERSION
    fw_git_hash: int = 0
    param_crc: int = 0

    def pack(self) -> bytes:
        return struct.pack(self.FORMAT, self.proto_ver, self.fw_git_hash, self.param_crc)

    @classmethod
    def unpack(cls, payload: bytes) -> 'Pong':
        return cls(*struct.unpack(cls.FORMAT, payload))


@dataclass
class EstopClear:
    TYPE = MSG_ESTOP_CLEAR
    FORMAT = '<I'
    magic: int = ESTOP_CLEAR_MAGIC

    def pack(self) -> bytes:
        return struct.pack(self.FORMAT, self.magic)

    @classmethod
    def unpack(cls, payload: bytes) -> 'EstopClear':
        return cls(*struct.unpack(cls.FORMAT, payload))


@dataclass
class Stats:
    TYPE = MSG_STATS
    FORMAT = '<6I'
    rx_frames_ok: int = 0
    rx_crc_errors: int = 0
    rx_framing_errors: int = 0
    cmd_seq_gaps: int = 0
    watchdog_trips: int = 0
    overcurrent_trips: int = 0

    def pack(self) -> bytes:
        return struct.pack(
            self.FORMAT, self.rx_frames_ok, self.rx_crc_errors, self.rx_framing_errors,
            self.cmd_seq_gaps, self.watchdog_trips, self.overcurrent_trips)

    @classmethod
    def unpack(cls, payload: bytes) -> 'Stats':
        return cls(*struct.unpack(cls.FORMAT, payload))


MESSAGE_CLASSES = {c.TYPE: c for c in (Cmd, State, SetParam, ParamAck, Pong, EstopClear, Stats)}


def decode_message(frame: Frame):
    """
    Return the payload object for `frame`, or None if its size is wrong.

    PING and ESTOP carry no payload and decode to the empty tuple; LOG decodes to str.
    """
    if frame.msg_type in (MSG_PING, MSG_ESTOP):
        return () if not frame.payload else None
    if frame.msg_type == MSG_LOG:
        return frame.payload.decode('ascii', errors='replace')
    cls = MESSAGE_CLASSES.get(frame.msg_type)
    if cls is None or len(frame.payload) != struct.calcsize(cls.FORMAT):
        return None
    return cls.unpack(frame.payload)


class Encoder:
    """Frames messages with an auto-incrementing 8-bit sequence number."""

    def __init__(self, write: Callable[[bytes], None]):
        self._write = write
        self.seq = 0

    def send(self, msg_type: int, payload: bytes = b'') -> bytes:
        wire = encode_frame(msg_type, self.seq, payload)
        self.seq = (self.seq + 1) & 0xFF
        self._write(wire)
        return wire

    def send_msg(self, msg) -> bytes:
        return self.send(msg.TYPE, msg.pack())


def seq_gap(previous: int, current: int) -> int:
    """Frames missing between two consecutive 8-bit sequence numbers."""
    return (current - previous - 1) & 0xFF
