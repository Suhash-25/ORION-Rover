// FirmwareLink against a mock transport: decoding, staleness with an injected clock, handshake
// helpers, failure detection and hostile input.
#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "orion_hardware/conversions.hpp"
#include "orion_hardware/firmware_link.hpp"
#include "mock_transport.hpp"

using orion_hardware::FirmwareLink;
using orion_hardware_test::MockTransport;
namespace op = orion_proto;
using namespace std::chrono_literals;

namespace
{

std::vector<op::Frame> decode_all(const std::vector<uint8_t> & bytes)
{
  op::FrameDecoder d;
  std::vector<op::Frame> out;
  for (uint8_t b : bytes) {
    if (d.push(b)) {
      out.push_back(d.frame());
    }
  }
  return out;
}

struct Fixture
{
  std::shared_ptr<MockTransport::Shared> shared = std::make_shared<MockTransport::Shared>();
  FirmwareLink::Clock::time_point fake_now = FirmwareLink::Clock::time_point{} + 100s;
  std::unique_ptr<FirmwareLink> link;

  explicit Fixture(bool start = true)
  {
    link = std::make_unique<FirmwareLink>(
      std::make_unique<MockTransport>(shared), [this] {return fake_now;});
    if (start) {
      std::string err;
      EXPECT_TRUE(link->start(err));
    }
  }
};

}  // namespace

TEST(FirmwareLink, LatestStateAndStalenessFollowTheInjectedClock)
{
  Fixture f(false);
  op::StateMsg s;
  s.t_us = 1000;
  s.position_mrad[0] = 42;
  uint8_t wire[op::kMaxWireFrame];
  size_t n = op::encode_msg(s, 0, wire, sizeof(wire));
  f.link->ingest(wire, n);

  auto sample = f.link->latest_state();
  ASSERT_TRUE(sample);
  EXPECT_EQ(sample->msg.position_mrad[0], 42);
  EXPECT_EQ(sample->count, 1u);
  EXPECT_EQ(sample->rx_time, f.fake_now);

  const auto max_age = std::chrono::duration_cast<FirmwareLink::Clock::duration>(100ms);
  EXPECT_TRUE(orion_hardware::is_fresh(f.fake_now + 99ms, sample->rx_time, max_age));
  EXPECT_FALSE(orion_hardware::is_fresh(f.fake_now + 101ms, sample->rx_time, max_age));

  // A new frame refreshes it; firmware time is unwrapped across the 32-bit boundary.
  f.fake_now += 150ms;
  s.t_us = 0xFFFFFFF0u;
  n = op::encode_msg(s, 1, wire, sizeof(wire));
  f.link->ingest(wire, n);
  s.t_us = 0x10u;
  n = op::encode_msg(s, 2, wire, sizeof(wire));
  f.link->ingest(wire, n);
  sample = f.link->latest_state();
  EXPECT_EQ(sample->rx_time, f.fake_now);
  EXPECT_EQ(sample->fw_time_us, 0x100000010ull);
  EXPECT_EQ(sample->count, 3u);
}

TEST(FirmwareLink, NoStateMeansNothingToRead)
{
  Fixture f(false);
  EXPECT_FALSE(f.link->latest_state());
}

TEST(FirmwareLink, ReaderThreadDecodesStreamWithGarbageAndCountsErrors)
{
  Fixture f;
  {
    std::lock_guard<std::mutex> lock(f.shared->m);
    const uint8_t junk[] = {0x12, 0x34, 0x00, 0x05, 0x11, 0x00};  // two bad blocks
    f.shared->to_host.insert(f.shared->to_host.end(), std::begin(junk), std::end(junk));
    op::StatsMsg st;
    st.watchdog_trips = 3;
    f.shared->push(st);
    op::StateMsg s;
    s.velocity_mrad_s[3] = -777;
    f.shared->push(s);
  }
  ASSERT_TRUE(f.link->wait_for_state(500ms) || f.link->latest_state());
  for (int i = 0; i < 100 && !f.link->latest_stats(); ++i) {
    std::this_thread::sleep_for(5ms);
  }
  EXPECT_EQ(f.link->latest_state()->msg.velocity_mrad_s[3], -777);
  ASSERT_TRUE(f.link->latest_stats());
  EXPECT_EQ(f.link->latest_stats()->watchdog_trips, 3u);
  const auto c = f.link->counters();
  EXPECT_EQ(c.rx_frames_ok, 2u);
  EXPECT_EQ(c.rx_crc_errors + c.rx_framing_errors, 2u);
}

TEST(FirmwareLink, SendCmdFramesWithIncrementingSeq)
{
  Fixture f(false);
  op::CmdMsg cmd;
  cmd.w_target_mrad_s[1] = -1234;
  cmd.flags = op::cmd_flag::kEnable;
  ASSERT_TRUE(f.link->send_cmd(cmd));
  ASSERT_TRUE(f.link->send_cmd(cmd));
  ASSERT_TRUE(f.link->send_estop());
  ASSERT_TRUE(f.link->send_estop_clear());
  const auto frames = decode_all(f.shared->from_host);
  ASSERT_EQ(frames.size(), 4u);
  op::CmdMsg back;
  ASSERT_TRUE(op::decode_msg(frames[0], back));
  EXPECT_EQ(back.w_target_mrad_s[1], -1234);
  EXPECT_EQ(back.flags, op::cmd_flag::kEnable);
  EXPECT_EQ(op::seq_gap(frames[0].seq, frames[1].seq), 0);
  EXPECT_EQ(frames[2].type, op::MSG_ESTOP);
  EXPECT_EQ(frames[2].len, 0);
  op::EstopClearMsg clear;
  ASSERT_TRUE(op::decode_msg(frames[3], clear));
  EXPECT_EQ(clear.magic, op::kEstopClearMagic);
  EXPECT_EQ(f.link->counters().tx_frames, 4u);
}

TEST(FirmwareLink, DroppedWritesAreCountedNotBlocking)
{
  Fixture f(false);
  f.shared->accept_writes = false;
  EXPECT_FALSE(f.link->send_cmd(op::CmdMsg{}));
  EXPECT_EQ(f.link->counters().tx_dropped, 1u);
}

TEST(FirmwareLink, PingAndSetParamWaitForMatchingAnswers)
{
  Fixture f;
  f.shared->respond = [](const op::Frame & fr, MockTransport::Shared & s) {
    if (fr.type == op::MSG_PING) {
      op::PongMsg pong;
      pong.fw_git_hash = 0xABCD0123;
      s.push(pong);
    } else if (fr.type == op::MSG_SET_PARAM) {
      op::SetParamMsg req;
      ASSERT_TRUE(op::decode_msg(fr, req));
      op::ParamAckMsg other;    // an unrelated ack first: must be skipped
      other.id = 99;
      s.push(other);
      op::ParamAckMsg ack;
      ack.id = req.id;
      ack.value = req.value;
      s.push(ack);
    }
  };
  const auto pong = f.link->ping(500ms);
  ASSERT_TRUE(pong);
  EXPECT_EQ(pong->fw_git_hash, 0xABCD0123u);
  const auto ack = f.link->set_param(op::param_id::kGearRatio, 51.0f, 500ms);
  ASSERT_TRUE(ack);
  EXPECT_EQ(ack->id, op::param_id::kGearRatio);
  EXPECT_EQ(ack->value, 51.0f);
}

TEST(FirmwareLink, PingTimesOutWhenFirmwareIsSilent)
{
  Fixture f;
  const auto t0 = std::chrono::steady_clock::now();
  EXPECT_FALSE(f.link->ping(100ms));
  EXPECT_GE(std::chrono::steady_clock::now() - t0, 95ms);
  EXPECT_FALSE(f.link->wait_for_state(50ms));
}

TEST(FirmwareLink, TransportFailureIsReported)
{
  Fixture f;
  EXPECT_FALSE(f.link->failed());
  f.shared->fail = true;
  for (int i = 0; i < 100 && !f.link->failed(); ++i) {
    std::this_thread::sleep_for(5ms);
  }
  EXPECT_TRUE(f.link->failed());
}

TEST(FirmwareLink, StopIsIdempotentAndRestartable)
{
  Fixture f;
  f.link->stop();
  f.link->stop();
  std::string err;
  EXPECT_TRUE(f.link->start(err));
  f.link->stop();
}
