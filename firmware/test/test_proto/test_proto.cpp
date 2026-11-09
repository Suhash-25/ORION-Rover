// Codec tests: golden vectors (shared with the Python reference), round trips, and hostile input.
#include <gtest/gtest.h>

#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include "mini_json.hpp"
#include "orion_proto.hpp"

#ifndef ORION_PROTO_VECTORS
#define ORION_PROTO_VECTORS "lib/orion_proto/test_vectors.json"
#endif

namespace op = orion_proto;
using Bytes = std::vector<uint8_t>;

namespace
{

const mini_json::Value & vectors()
{
  static const mini_json::Value v = mini_json::parse_file(ORION_PROTO_VECTORS);
  return v;
}

Bytes wire_of(uint8_t type, uint8_t seq, const Bytes & payload)
{
  Bytes out(op::kMaxWireFrame);
  const size_t n = op::encode_frame(
    type, seq, payload.data(), payload.size(), out.data(),
    out.size());
  out.resize(n);
  return out;
}

struct FeedResult
{
  std::vector<op::Frame> frames;
  op::DecoderCounters counters;
};

FeedResult feed(const Bytes & bytes)
{
  op::FrameDecoder d;
  FeedResult r;
  for (uint8_t b : bytes) {
    if (d.push(b)) {
      r.frames.push_back(d.frame());
    }
  }
  r.counters = d.counters();
  return r;
}

template<size_t N, typename T>
void expect_array(const mini_json::Value & arr, const T (& actual)[N])
{
  ASSERT_EQ(arr.array.size(), N);
  for (size_t i = 0; i < N; ++i) {
    EXPECT_EQ(static_cast<long long>(actual[i]), arr[i].as_int()) << "index " << i;
  }
}

// Unpacks `frame` according to its type, checks every field against `f`, and re-packs.
void check_fields(const op::Frame & frame, const mini_json::Value & f, const Bytes & payload)
{
  Bytes repacked(frame.len);
  switch (frame.type) {
    case op::MSG_CMD: {
        op::CmdMsg m;
        ASSERT_TRUE(op::decode_msg(frame, m));
        expect_array(f["w_target_mrad_s"], m.w_target_mrad_s);
        EXPECT_EQ(m.flags, f["flags"].as_int());
        m.pack(repacked.data());
        break;
      }
    case op::MSG_STATE: {
        op::StateMsg m;
        ASSERT_TRUE(op::decode_msg(frame, m));
        EXPECT_EQ(m.t_us, f["t_us"].as_int());
        expect_array(f["position_mrad"], m.position_mrad);
        expect_array(f["velocity_mrad_s"], m.velocity_mrad_s);
        expect_array(f["current_ma"], m.current_ma);
        expect_array(f["duty"], m.duty);
        EXPECT_EQ(m.vbat_mv, f["vbat_mv"].as_int());
        EXPECT_EQ(m.faults, f["faults"].as_int());
        EXPECT_EQ(m.status, f["status"].as_int());
        m.pack(repacked.data());
        break;
      }
    case op::MSG_SET_PARAM: {
        op::SetParamMsg m;
        ASSERT_TRUE(op::decode_msg(frame, m));
        EXPECT_EQ(m.id, f["param_id"].as_int());
        EXPECT_EQ(m.value, static_cast<float>(f["value"].number));
        m.pack(repacked.data());
        break;
      }
    case op::MSG_PARAM_ACK: {
        op::ParamAckMsg m;
        ASSERT_TRUE(op::decode_msg(frame, m));
        EXPECT_EQ(m.id, f["param_id"].as_int());
        EXPECT_EQ(m.value, static_cast<float>(f["value"].number));
        EXPECT_EQ(m.status, f["status"].as_int());
        m.pack(repacked.data());
        break;
      }
    case op::MSG_PONG: {
        op::PongMsg m;
        ASSERT_TRUE(op::decode_msg(frame, m));
        EXPECT_EQ(m.proto_ver, f["proto_ver"].as_int());
        EXPECT_EQ(m.fw_git_hash, f["fw_git_hash"].as_int());
        EXPECT_EQ(m.param_crc, f["param_crc"].as_int());
        m.pack(repacked.data());
        break;
      }
    case op::MSG_ESTOP_CLEAR: {
        op::EstopClearMsg m;
        ASSERT_TRUE(op::decode_msg(frame, m));
        EXPECT_EQ(m.magic, f["magic"].as_int());
        m.pack(repacked.data());
        break;
      }
    case op::MSG_STATS: {
        op::StatsMsg m;
        ASSERT_TRUE(op::decode_msg(frame, m));
        EXPECT_EQ(m.rx_frames_ok, f["rx_frames_ok"].as_int());
        EXPECT_EQ(m.rx_crc_errors, f["rx_crc_errors"].as_int());
        EXPECT_EQ(m.rx_framing_errors, f["rx_framing_errors"].as_int());
        EXPECT_EQ(m.cmd_seq_gaps, f["cmd_seq_gaps"].as_int());
        EXPECT_EQ(m.watchdog_trips, f["watchdog_trips"].as_int());
        EXPECT_EQ(m.overcurrent_trips, f["overcurrent_trips"].as_int());
        m.pack(repacked.data());
        break;
      }
    default:  // PING, ESTOP (empty) and LOG (ASCII): the payload bytes are the fields
      repacked.assign(frame.payload, frame.payload + frame.len);
      break;
  }
  EXPECT_EQ(repacked, payload);
}

}  // namespace

TEST(Crc16, MatchesPublishedCheckValues)
{
  for (const auto & v : vectors()["crc16"].array) {
    const Bytes data = mini_json::from_hex(v["hex"].string);
    EXPECT_EQ(op::crc16_ccitt_false(data.data(), data.size()), v["crc"].as_int());
  }
}

TEST(Cobs, MatchesPublishedExamples)
{
  for (const auto & v : vectors()["cobs"].array) {
    const Bytes dec = mini_json::from_hex(v["decoded"].string);
    const Bytes enc = mini_json::from_hex(v["encoded"].string);
    Bytes out(dec.size() + dec.size() / 254 + 2);
    const size_t n = op::cobs_encode(dec.data(), dec.size(), out.data(), out.size());
    out.resize(n);
    EXPECT_EQ(out, enc) << v["decoded"].string;
    Bytes back(dec.size() + 1);
    const size_t m = op::cobs_decode(enc.data(), enc.size(), back.data(), back.size());
    back.resize(m);
    EXPECT_EQ(back, dec);
  }
}

TEST(Cobs, RandomRoundTripNeverEmitsZero)
{
  std::mt19937 rng(42);
  for (size_t len = 1; len < 600; ++len) {
    Bytes in(len);
    for (auto & b : in) {
      b = static_cast<uint8_t>(rng() % 4 == 0 ? 0 : rng());
    }
    Bytes enc(len + len / 254 + 2);
    const size_t n = op::cobs_encode(in.data(), in.size(), enc.data(), enc.size());
    ASSERT_GT(n, 0u);
    for (size_t i = 0; i < n; ++i) {
      ASSERT_NE(enc[i], 0) << "len " << len;
    }
    Bytes dec(len);
    ASSERT_EQ(op::cobs_decode(enc.data(), n, dec.data(), dec.size()), len);
    EXPECT_EQ(dec, in);
  }
}

TEST(Cobs, EncodeRefusesSmallBuffer)
{
  const Bytes in(10, 0x11);
  Bytes out(5);
  EXPECT_EQ(op::cobs_encode(in.data(), in.size(), out.data(), out.size()), 0u);
}

TEST(Frames, EncodeMatchesGoldenVectors)
{
  EXPECT_EQ(vectors()["protocol_version"].as_int(), op::kProtocolVersion);
  for (const auto & v : vectors()["frames"].array) {
    SCOPED_TRACE(v["name"].string);
    const Bytes payload = mini_json::from_hex(v["payload"].string);
    const Bytes wire = mini_json::from_hex(v["wire"].string);
    EXPECT_EQ(
      wire_of(
        static_cast<uint8_t>(v["type"].as_int()), static_cast<uint8_t>(v["seq"].as_int()),
        payload), wire);
  }
}

TEST(Frames, DecodeMatchesGoldenVectors)
{
  for (const auto & v : vectors()["frames"].array) {
    SCOPED_TRACE(v["name"].string);
    const FeedResult r = feed(mini_json::from_hex(v["wire"].string));
    ASSERT_EQ(r.frames.size(), 1u);
    const op::Frame & f = r.frames[0];
    EXPECT_EQ(f.type, v["type"].as_int());
    EXPECT_EQ(f.seq, v["seq"].as_int());
    check_fields(f, v["fields"], mini_json::from_hex(v["payload"].string));
  }
}

TEST(Frames, InvalidGoldenVectorsAreRejectedAndCounted)
{
  for (const auto & v : vectors()["invalid"].array) {
    SCOPED_TRACE(v["name"].string);
    const FeedResult r = feed(mini_json::from_hex(v["wire"].string));
    EXPECT_TRUE(r.frames.empty());
    const bool crc = v["error"].string == "crc";
    EXPECT_EQ(r.counters.crc_errors, crc ? 1u : 0u);
    EXPECT_EQ(r.counters.framing_errors, crc ? 0u : 1u);
  }
}

TEST(Frames, MaximumPayloadFitsWireBuffer)
{
  Bytes payload(op::kMaxPayload, 0xAB);
  Bytes wire = wire_of(op::MSG_LOG, 1, payload);
  ASSERT_FALSE(wire.empty());
  EXPECT_LE(wire.size(), op::kMaxWireFrame);
  const FeedResult r = feed(wire);
  ASSERT_EQ(r.frames.size(), 1u);
  EXPECT_EQ(r.frames[0].len, op::kMaxPayload);
  // One byte more must be refused by the encoder.
  payload.push_back(0);
  Bytes out(op::kMaxWireFrame * 2);
  EXPECT_EQ(
    op::encode_frame(
      op::MSG_LOG, 0, payload.data(), payload.size(), out.data(),
      out.size()), 0u);
}

TEST(Frames, EveryTruncationIsRejectedAndCounted)
{
  op::CmdMsg cmd;
  cmd.w_target_mrad_s[0] = 1234;
  Bytes wire(op::kMaxWireFrame);
  wire.resize(op::encode_msg(cmd, 9, wire.data(), wire.size()));
  for (size_t cut = 1; cut + 1 < wire.size(); ++cut) {
    Bytes partial(wire.begin(), wire.begin() + static_cast<std::ptrdiff_t>(cut));
    partial.push_back(0x00);
    const FeedResult r = feed(partial);
    EXPECT_TRUE(r.frames.empty()) << "cut " << cut;
    EXPECT_EQ(r.counters.crc_errors + r.counters.framing_errors, 1u) << "cut " << cut;
  }
}

TEST(Frames, EverySingleBitFlipIsRejected)
{
  op::StateMsg s;
  s.t_us = 123456;
  s.vbat_mv = 12000;
  Bytes wire(op::kMaxWireFrame);
  wire.resize(op::encode_msg(s, 3, wire.data(), wire.size()));
  for (size_t i = 0; i + 1 < wire.size(); ++i) {
    for (int bit = 0; bit < 8; ++bit) {
      Bytes bad = wire;
      bad[i] ^= static_cast<uint8_t>(1u << bit);
      const FeedResult r = feed(bad);
      EXPECT_TRUE(r.frames.empty()) << "byte " << i << " bit " << bit;
      EXPECT_GE(r.counters.crc_errors + r.counters.framing_errors, 1u);
    }
  }
}

TEST(Frames, RandomGarbageNeverCrashesAndDecoderResyncs)
{
  std::mt19937 rng(7);
  op::FrameDecoder d;
  size_t accepted = 0;
  for (int i = 0; i < 200000; ++i) {
    // Bias towards zeros so many short blocks are tried, not only overflows.
    const uint8_t b = rng() % 8 == 0 ? 0 : static_cast<uint8_t>(rng());
    accepted += d.push(b) ? 1 : 0;
  }
  const op::DecoderCounters before = d.counters();
  EXPECT_EQ(before.frames_ok, accepted);
  EXPECT_GT(before.framing_errors + before.crc_errors, 1000u);
  // After garbage, one delimiter and a valid frame decode normally.
  d.push(0x00);
  op::PongMsg pong;
  pong.fw_git_hash = 0xCAFEF00D;
  Bytes wire(op::kMaxWireFrame);
  wire.resize(op::encode_msg(pong, 77, wire.data(), wire.size()));
  bool got = false;
  for (uint8_t b : wire) {
    got = d.push(b) || got;
  }
  ASSERT_TRUE(got);
  op::PongMsg back;
  ASSERT_TRUE(op::decode_msg(d.frame(), back));
  EXPECT_EQ(back.fw_git_hash, 0xCAFEF00Du);
}

TEST(Frames, OversizeBlockIsOneFramingErrorThenRecovers)
{
  Bytes junk(500, 0x55);
  junk.push_back(0x00);
  op::PongMsg pong;
  Bytes wire(op::kMaxWireFrame);
  wire.resize(op::encode_msg(pong, 1, wire.data(), wire.size()));
  junk.insert(junk.end(), wire.begin(), wire.end());
  const FeedResult r = feed(junk);
  EXPECT_EQ(r.counters.framing_errors, 1u);
  EXPECT_EQ(r.frames.size(), 1u);
}

TEST(Messages, UnpackRejectsWrongLength)
{
  op::Frame f;
  f.type = op::MSG_CMD;
  f.len = op::CmdMsg::kSize - 1;
  op::CmdMsg cmd;
  EXPECT_FALSE(op::decode_msg(f, cmd));
  f.len = op::CmdMsg::kSize;
  EXPECT_TRUE(op::decode_msg(f, cmd));
  f.type = op::MSG_STATE;  // right size, wrong type
  EXPECT_FALSE(op::decode_msg(f, cmd));
}

TEST(Messages, SeqGapHandlesWrap)
{
  EXPECT_EQ(op::seq_gap(4, 5), 0);
  EXPECT_EQ(op::seq_gap(4, 8), 3);
  EXPECT_EQ(op::seq_gap(255, 0), 0);
  EXPECT_EQ(op::seq_gap(254, 1), 2);
}
