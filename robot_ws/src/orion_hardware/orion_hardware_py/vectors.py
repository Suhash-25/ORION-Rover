"""
Golden test vectors shared by the C++ (firmware + orion_hardware) and Python protocol tests.

Regenerate after a deliberate protocol change (and bump PROTOCOL_VERSION):

    python3 -m orion_hardware_py.vectors firmware/lib/orion_proto/test_vectors.json

The CRC and COBS entries are independent anchors from the published algorithm definitions
(CRC catalogue check value, the COBS paper / Wikipedia examples), not outputs of this code.
"""

import json
import sys

from . import protocol as p

# Published known answers.
CRC_ANCHORS = [
    {'hex': '313233343536373839', 'crc': 0x29B1},  # "123456789", CRC catalogue check value
    {'hex': '', 'crc': 0xFFFF},
]
COBS_ANCHORS = [
    {'decoded': '00', 'encoded': '0101'},
    {'decoded': '0000', 'encoded': '010101'},
    {'decoded': '001100', 'encoded': '01021101'},
    {'decoded': '11220033', 'encoded': '0311220233'},
    {'decoded': '11223344', 'encoded': '0511223344'},
    {'decoded': '11000000', 'encoded': '0211010101'},
    {'decoded': ''.join(f'{i:02x}' for i in range(1, 255)),
     'encoded': 'ff' + ''.join(f'{i:02x}' for i in range(1, 255))},
    {'decoded': ''.join(f'{i:02x}' for i in range(0, 255)),
     'encoded': '01ff' + ''.join(f'{i:02x}' for i in range(1, 255))},
]


def _frame(name, msg, seq, fields):
    payload = msg.pack()
    return {
        'name': name, 'type': msg.TYPE, 'seq': seq, 'payload': payload.hex(),
        'wire': p.encode_frame(msg.TYPE, seq, payload).hex(), 'fields': fields,
    }


def _empty(name, msg_type, seq):
    return {'name': name, 'type': msg_type, 'seq': seq, 'payload': '',
            'wire': p.encode_frame(msg_type, seq).hex(), 'fields': {}}


def build_vectors() -> dict:
    """Return the full vector set as a JSON-serialisable dict."""
    cmd = p.Cmd([1000, -1000, 32767, -32768], p.CMD_FLAG_ENABLE | p.CMD_FLAG_BRAKE_ON_IDLE)
    cmd_zero = p.Cmd([0, 0, 0, 0], 0)  # payload of zeros: exercises COBS zero runs
    state = p.State(
        t_us=0xFEDCBA98, position_mrad=[123456, -654321, 2147483647, -2147483648],
        velocity_mrad_s=[2000, -2000, 0, 1], current_ma=[1500, -250, p.CURRENT_NOT_AVAILABLE, 0],
        duty=[10000, -10000, 5000, 0], vbat_mv=12345,
        faults=p.FAULT_WATCHDOG | p.FAULT_ESTOP_LATCHED,
        status=p.STATUS_CURRENT_SENSED | p.STATUS_VBAT_SENSED)
    set_param = p.SetParam(p.PARAM_ENCODER_CPR, 4096.0)
    ack = p.ParamAck(p.PARAM_CURRENT_LIMIT_A, 0.25, p.PARAM_OUT_OF_RANGE)
    pong = p.Pong(p.PROTOCOL_VERSION, 0x9E2754A0, 0xBEEF)
    clear = p.EstopClear(p.ESTOP_CLEAR_MAGIC)
    stats = p.Stats(1, 2, 3, 4, 5, 0xFFFFFFFF)
    frames = [
        _frame('cmd', cmd, 7, {'w_target_mrad_s': cmd.w_target_mrad_s, 'flags': cmd.flags}),
        _frame('cmd_zero', cmd_zero, 0, {'w_target_mrad_s': [0, 0, 0, 0], 'flags': 0}),
        _frame('state', state, 255, {
            't_us': state.t_us, 'position_mrad': state.position_mrad,
            'velocity_mrad_s': state.velocity_mrad_s, 'current_ma': state.current_ma,
            'duty': state.duty, 'vbat_mv': state.vbat_mv, 'faults': state.faults,
            'status': state.status}),
        _frame('set_param', set_param, 1,
               {'param_id': set_param.param_id, 'value': set_param.value}),
        _frame('param_ack', ack, 2,
               {'param_id': ack.param_id, 'value': ack.value, 'status': ack.status}),
        _empty('ping', p.MSG_PING, 3),
        _frame('pong', pong, 4, {'proto_ver': pong.proto_ver, 'fw_git_hash': pong.fw_git_hash,
                                 'param_crc': pong.param_crc}),
        _empty('estop', p.MSG_ESTOP, 5),
        _frame('estop_clear', clear, 6, {'magic': clear.magic}),
        {'name': 'log', 'type': p.MSG_LOG, 'seq': 8, 'payload': b'watchdog'.hex(),
         'wire': p.encode_frame(p.MSG_LOG, 8, b'watchdog').hex(), 'fields': {}},
        _frame('stats', stats, 9, {
            'rx_frames_ok': 1, 'rx_crc_errors': 2, 'rx_framing_errors': 3, 'cmd_seq_gaps': 4,
            'watchdog_trips': 5, 'overcurrent_trips': 0xFFFFFFFF}),
    ]

    good = bytes.fromhex(frames[0]['wire'])
    raw = bytearray(p.cobs_decode(good[:-1]))
    bad_crc = bytearray(raw)
    bad_crc[-1] ^= 0x01
    bad_len = bytearray(raw)
    bad_len[2] += 1
    invalid = [
        {'name': 'bad_crc', 'wire': (p.cobs_encode(bytes(bad_crc)) + b'\x00').hex(),
         'error': 'crc'},
        {'name': 'len_mismatch', 'wire': (p.cobs_encode(bytes(bad_len)) + b'\x00').hex(),
         'error': 'framing'},
        {'name': 'truncated', 'wire': (good[:5] + b'\x00').hex(), 'error': 'framing'},
        {'name': 'too_short', 'wire': (p.cobs_encode(b'\x01\x02') + b'\x00').hex(),
         'error': 'framing'},
        {'name': 'bad_cobs_code', 'wire': '0511223300', 'error': 'framing'},
        {'name': 'oversize', 'wire': '01' * 80 + '00', 'error': 'framing'},
    ]
    return {
        'protocol_version': p.PROTOCOL_VERSION,
        'crc16': CRC_ANCHORS,
        'cobs': COBS_ANCHORS,
        'frames': frames,
        'invalid': invalid,
    }


def dumps(vectors: dict) -> str:
    return json.dumps(vectors, indent=1, sort_keys=True) + '\n'


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print('usage: python3 -m orion_hardware_py.vectors OUT.json', file=sys.stderr)
        return 2
    with open(argv[0], 'w') as f:
        f.write(dumps(build_vectors()))
    return 0


if __name__ == '__main__':
    sys.exit(main())
