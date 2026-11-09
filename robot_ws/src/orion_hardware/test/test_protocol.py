"""Python protocol reference against the golden vectors shared with the C++ implementation."""

import json
import os
import random
import struct

from orion_hardware_py import protocol as p
from orion_hardware_py import vectors
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
VECTORS_PATH = os.path.join(
    HERE, '..', '..', '..', '..', 'firmware', 'lib', 'orion_proto', 'test_vectors.json')


@pytest.fixture(scope='module')
def golden():
    with open(VECTORS_PATH) as f:
        return json.load(f)


def test_committed_vectors_match_the_generator(golden):
    # Fails if someone edits the protocol without regenerating (or edits the JSON by hand).
    assert vectors.dumps(vectors.build_vectors()) == vectors.dumps(golden)


def test_protocol_version(golden):
    assert golden['protocol_version'] == p.PROTOCOL_VERSION


def test_crc_known_answers(golden):
    for v in golden['crc16']:
        assert p.crc16_ccitt_false(bytes.fromhex(v['hex'])) == v['crc']


def test_cobs_known_answers(golden):
    for v in golden['cobs']:
        dec, enc = bytes.fromhex(v['decoded']), bytes.fromhex(v['encoded'])
        assert p.cobs_encode(dec) == enc
        assert p.cobs_decode(enc) == dec


def test_frames_encode_and_decode(golden):
    for v in golden['frames']:
        payload = bytes.fromhex(v['payload'])
        wire = bytes.fromhex(v['wire'])
        assert p.encode_frame(v['type'], v['seq'], payload) == wire, v['name']
        dec = p.FrameDecoder()
        frames = dec.feed(wire)
        assert len(frames) == 1, v['name']
        assert (frames[0].msg_type, frames[0].seq, frames[0].payload) == (
            v['type'], v['seq'], payload)
        msg = p.decode_message(frames[0])
        assert msg is not None, v['name']
        for key, expected in v['fields'].items():
            # Floats in the vectors are exactly representable in f32, so == is right.
            assert getattr(msg, key) == expected, (v['name'], key)
        if hasattr(msg, 'pack'):
            assert msg.pack() == payload


def test_invalid_frames_are_rejected_and_counted(golden):
    for v in golden['invalid']:
        dec = p.FrameDecoder()
        assert dec.feed(bytes.fromhex(v['wire'])) == [], v['name']
        expected = (1, 0) if v['error'] == 'crc' else (0, 1)
        assert (dec.crc_errors, dec.framing_errors) == expected, v['name']


def test_every_truncation_is_rejected():
    wire = p.encode_frame(p.MSG_CMD, 3, p.Cmd([1, 2, 3, 4], 1).pack())
    for cut in range(1, len(wire) - 1):
        dec = p.FrameDecoder()
        assert dec.feed(wire[:cut] + b'\x00') == []
        assert dec.crc_errors + dec.framing_errors == 1


def test_random_garbage_never_raises_and_decoder_resyncs():
    rng = random.Random(5)
    dec = p.FrameDecoder()
    junk = bytes(0 if rng.random() < 0.12 else rng.randrange(256) for _ in range(50000))
    dec.feed(junk)
    assert dec.crc_errors + dec.framing_errors > 100
    frames = dec.feed(b'\x00' + p.encode_frame(p.MSG_PONG, 1, p.Pong(1, 7, 9).pack()))
    assert frames and p.decode_message(frames[-1]) == p.Pong(1, 7, 9)


def test_random_cobs_round_trip():
    rng = random.Random(1)
    for n in range(0, 600, 7):
        data = bytes(rng.choice([0, rng.randrange(256)]) for _ in range(n))
        enc = p.cobs_encode(data)
        assert 0 not in enc
        assert p.cobs_decode(enc) == data


def test_payload_sizes_match_the_spec():
    # docs/SPEC.md section 7 and docs/PROTOCOL.md.
    assert struct.calcsize(p.Cmd.FORMAT) == 9
    assert struct.calcsize(p.State.FORMAT) == 48
    assert struct.calcsize(p.SetParam.FORMAT) == 5
    assert struct.calcsize(p.ParamAck.FORMAT) == 6
    assert struct.calcsize(p.Pong.FORMAT) == 7
    assert struct.calcsize(p.EstopClear.FORMAT) == 4
    assert struct.calcsize(p.Stats.FORMAT) == 24


def test_oversize_payload_is_refused():
    with pytest.raises(ValueError):
        p.encode_frame(p.MSG_LOG, 0, bytes(p.MAX_PAYLOAD + 1))


def test_wrong_payload_size_decodes_to_none():
    assert p.decode_message(p.Frame(p.MSG_CMD, 0, b'\x00' * 8)) is None
    assert p.decode_message(p.Frame(p.MSG_PING, 0, b'\x01')) is None
    assert p.decode_message(p.Frame(0x7E, 0, b'')) is None


def test_seq_gap_wraps():
    assert p.seq_gap(255, 0) == 0
    assert p.seq_gap(254, 1) == 2
