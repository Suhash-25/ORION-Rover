// Behaviour of the portable firmware core against a fake board and a fake clock.
#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include "fake_hal.hpp"
#include "pid.hpp"

using orion_test::Bench;
namespace op = orion_proto;

// ------------------------------------------------------------------------------------------
// PID
// ------------------------------------------------------------------------------------------

TEST(Pid, OutputIsClampedToLimit)
{
  orion_fw::Pid pid;
  const orion_fw::PidGains g{5.0f, 50.0f, 0.0f, 1.0f};
  for (int i = 0; i < 1000; ++i) {
    const float u = pid.update(g, 100.0f, 0.0f, 0.005f, 0.8f);
    ASSERT_LE(std::fabs(u), 0.8f + 1e-6f);
  }
}

TEST(Pid, IntegratorDoesNotWindUpWhileSaturated)
{
  orion_fw::Pid pid;
  const orion_fw::PidGains g{0.0f, 10.0f, 0.0f, 0.0f};
  // 5 s of an unreachable target: a naive integrator would reach 10 * 10 * 5 = 500.
  for (int i = 0; i < 1000; ++i) {
    pid.update(g, 10.0f, 0.0f, 0.005f, 1.0f);
  }
  EXPECT_LE(std::fabs(pid.integral()), 1.0f + 1e-6f);
  // As soon as the error reverses, the output must leave saturation (no unwinding delay).
  const float u = pid.update(g, 0.0f, 0.5f, 0.005f, 1.0f);
  EXPECT_LT(u, 1.0f);
}

TEST(Pid, IntegratorHoldsWhenOtherTermsAlreadySaturate)
{
  // Feed-forward alone saturates the output; integrating anyway (even with the integrator
  // clamped to the limit) would keep the output pinned after the target becomes reachable.
  orion_fw::Pid pid;
  const orion_fw::PidGains g{0.0f, 10.0f, 0.0f, 0.1f};
  for (int i = 0; i < 1000; ++i) {
    pid.update(g, 10.0f, 0.0f, 0.005f, 1.0f);
  }
  EXPECT_NEAR(pid.integral(), 0.0f, 1e-6f);
  const float u = pid.update(g, 5.0f, 4.5f, 0.005f, 1.0f);
  EXPECT_NEAR(u, 0.5f + 10.0f * 0.5f * 0.005f, 1e-4f);
}

TEST(Pid, ResetClearsState)
{
  orion_fw::Pid pid;
  const orion_fw::PidGains g{0.0f, 10.0f, 0.0f, 0.0f};
  pid.update(g, 1.0f, 0.0f, 0.1f, 1.0f);
  ASSERT_NE(pid.integral(), 0.0f);
  pid.reset();
  EXPECT_EQ(pid.integral(), 0.0f);
}

TEST(ControlCore, StepResponseSettlesWithoutLargeOvershoot)
{
  Bench b;
  const float target = 3.0f;  // rad/s, ~half of the plant's free speed
  float peak = 0.0f;
  for (int k = 0; k < 100; ++k) {  // 1 s
    b.drive_ms(10.0, target);
    peak = std::max(peak, b.core.speed_rad_s(0));
    if (k >= 40) {  // after 400 ms, within 3 % and staying there
      for (size_t i = 0; i < 4; ++i) {
        ASSERT_NEAR(b.core.speed_rad_s(i), target, 0.03f * target) << "t=" << k * 10 << " ms";
      }
    }
  }
  EXPECT_LT(peak, 1.2f * target);
  op::StateMsg s;
  ASSERT_TRUE(b.last(s));
  EXPECT_NEAR(s.velocity_mrad_s[0], 3000, 100);
  EXPECT_TRUE(s.status & op::status::kEnabled);
  EXPECT_EQ(s.faults, 0);
}

TEST(ControlCore, ReverseTargetsAndPerWheelCommands)
{
  Bench b;
  op::CmdMsg cmd;
  cmd.flags = op::cmd_flag::kEnable;
  cmd.w_target_mrad_s[0] = 2000;
  cmd.w_target_mrad_s[1] = -2000;
  cmd.w_target_mrad_s[2] = 1000;
  cmd.w_target_mrad_s[3] = 0;
  for (int k = 0; k < 60; ++k) {
    b.send(cmd);
    b.run_ms(10.0);
  }
  EXPECT_NEAR(b.core.speed_rad_s(0), 2.0f, 0.06f);
  EXPECT_NEAR(b.core.speed_rad_s(1), -2.0f, 0.06f);
  EXPECT_NEAR(b.core.speed_rad_s(2), 1.0f, 0.03f);
  EXPECT_NEAR(b.core.speed_rad_s(3), 0.0f, 0.03f);
}

// ------------------------------------------------------------------------------------------
// Watchdog
// ------------------------------------------------------------------------------------------

namespace
{
// Sends one CMD, then polls every 1 ms; returns ms from the CMD to the watchdog trip.
double time_to_watchdog_trip(Bench & b, uint8_t flags)
{
  b.drive_ms(100.0, 2.0f, flags);  // running and enabled
  EXPECT_FALSE(b.core.watchdog_tripped());
  b.send_cmd(2.0f, flags);
  for (int ms = 1; ms <= 400; ++ms) {
    b.run_ms(1.0);
    if (b.core.watchdog_tripped()) {
      return ms;
    }
  }
  return -1;
}
}  // namespace

TEST(Watchdog, TripsAt200msAndCoasts)
{
  Bench b;
  const double t = time_to_watchdog_trip(b, op::cmd_flag::kEnable);
  EXPECT_GE(t, 190.0);
  EXPECT_LE(t, 210.0);
  EXPECT_EQ(b.core.stats().watchdog_trips, 1u);
  EXPECT_FALSE(b.core.control_active());
  for (size_t i = 0; i < 4; ++i) {
    EXPECT_EQ(b.hal.cmd_duty[i], 0.0f);
    EXPECT_FALSE(b.hal.cmd_brake[i]) << "must coast unless brake_on_idle";
    EXPECT_EQ(b.core.target_rad_s(i), 0.0f);
  }
  b.run_ms(20.0);
  op::StateMsg s;
  ASSERT_TRUE(b.last(s));
  EXPECT_TRUE(s.faults & op::fault::kWatchdog);
  EXPECT_FALSE(s.status & op::status::kEnabled);
}

TEST(Watchdog, BrakesWhenBrakeOnIdleIsSet)
{
  Bench b;
  const double t = time_to_watchdog_trip(b, op::cmd_flag::kEnable | op::cmd_flag::kBrakeOnIdle);
  EXPECT_NEAR(t, 200.0, 10.0);
  for (size_t i = 0; i < 4; ++i) {
    EXPECT_EQ(b.hal.cmd_duty[i], 0.0f);
    EXPECT_TRUE(b.hal.cmd_brake[i]);
  }
}

TEST(Watchdog, WorksAcrossMicrosWraparound)
{
  Bench b(0xFFFFFFFFu - 150000u);  // micros() wraps 150 ms into the test
  const double t = time_to_watchdog_trip(b, op::cmd_flag::kEnable);
  EXPECT_NEAR(t, 200.0, 10.0);
}

TEST(Watchdog, DoesNotTripWhileCommandsArriveAt100Hz)
{
  Bench b;
  b.drive_ms(2000.0, 1.0f);
  EXPECT_FALSE(b.core.watchdog_tripped());
  EXPECT_EQ(b.core.stats().watchdog_trips, 0u);
}

TEST(Watchdog, BootStateIsIdleButNotCountedAsTrip)
{
  Bench b;
  b.run_ms(500.0);
  EXPECT_TRUE(b.core.watchdog_tripped());
  EXPECT_EQ(b.core.stats().watchdog_trips, 0u);
  EXPECT_EQ(b.hal.cmd_duty[0], 0.0f);
}

TEST(Watchdog, RecoversOnNextCommandWithFreshPid)
{
  Bench b;
  time_to_watchdog_trip(b, op::cmd_flag::kEnable);
  b.drive_ms(500.0, 2.0f);
  EXPECT_FALSE(b.core.watchdog_tripped());
  EXPECT_NEAR(b.core.speed_rad_s(0), 2.0f, 0.06f);
}

TEST(Command, EnableFlagClearMeansIdle)
{
  Bench b;
  b.drive_ms(200.0, 2.0f, 0);  // targets present but enable bit clear
  EXPECT_FALSE(b.core.control_active());
  EXPECT_FALSE(b.core.watchdog_tripped());
  EXPECT_EQ(b.hal.cmd_duty[0], 0.0f);
}

// ------------------------------------------------------------------------------------------
// Current limit
// ------------------------------------------------------------------------------------------

TEST(CurrentLimit, StalledWheelIsClampedAfterLimitTime)
{
  Bench b;
  b.hal.wheel[2].locked = true;  // ~24 A at full duty and 12 V through 0.5 ohm
  const float limit = b.core.params().get(op::param_id::kCurrentLimitA);
  double over_since_ms = -1.0;
  double fault_at_ms = -1.0;
  for (int ms = 0; ms < 1000; ++ms) {
    if (ms % 10 == 0) {
      b.send_cmd(5.0f);
    }
    b.run_ms(1.0);
    if (over_since_ms < 0 && std::fabs(b.hal.wheel[2].current) > limit) {
      over_since_ms = ms;
    }
    if (fault_at_ms < 0 && (b.core.faults() & op::fault::kOvercurrent)) {
      fault_at_ms = ms;
    }
  }
  ASSERT_GE(over_since_ms, 0.0) << "plant never exceeded the limit; test is not exercising it";
  ASSERT_GE(fault_at_ms, 0.0);
  const double delay = fault_at_ms - over_since_ms;
  EXPECT_GE(delay, 95.0);   // 100 ms over the limit before clamping (SPEC section 5)
  EXPECT_LE(delay, 110.0);
  EXPECT_EQ(b.core.stats().overcurrent_trips, 1u);
  EXPECT_LE(std::fabs(b.hal.wheel[2].current), limit * 1.05);  // clamped by now
  EXPECT_LT(b.core.duty_cap(2), 1.0f);
  // Other wheels are unaffected.
  EXPECT_EQ(b.core.duty_cap(0), 1.0f);
  EXPECT_NEAR(b.core.speed_rad_s(0), 5.0f, 0.2f);
}

TEST(CurrentLimit, ShortSpikeBelowLimitTimeIsTolerated)
{
  Bench b;
  b.hal.wheel[1].locked = true;
  b.drive_ms(60.0, 5.0f);  // 60 ms over the limit, less than the 100 ms window
  b.hal.wheel[1].locked = false;
  b.drive_ms(300.0, 1.0f);
  EXPECT_EQ(b.core.stats().overcurrent_trips, 0u);
  EXPECT_FALSE(b.core.faults() & op::fault::kOvercurrent);
}

TEST(CurrentLimit, RecoversWhenLoadIsRemoved)
{
  Bench b;
  b.hal.wheel[0].locked = true;
  b.drive_ms(400.0, 5.0f);
  ASSERT_TRUE(b.core.faults() & op::fault::kOvercurrent);
  b.hal.wheel[0].locked = false;
  b.drive_ms(3000.0, 1.0f);
  EXPECT_FALSE(b.core.faults() & op::fault::kOvercurrent);
  EXPECT_EQ(b.core.duty_cap(0), 1.0f);
}

TEST(CurrentLimit, NotSensedReportsUnavailableAndNeverClamps)
{
  Bench b;
  b.hal.current_sensed = false;
  b.hal.wheel[0].locked = true;
  b.drive_ms(500.0, 5.0f);
  EXPECT_FALSE(b.core.faults() & op::fault::kOvercurrent);
  op::StateMsg s;
  ASSERT_TRUE(b.last(s));
  for (auto ma : s.current_ma) {
    EXPECT_EQ(ma, op::kCurrentNotAvailable);
  }
  EXPECT_FALSE(s.status & op::status::kCurrentSensed);
}

// ------------------------------------------------------------------------------------------
// E-stop
// ------------------------------------------------------------------------------------------

TEST(Estop, LatchesBrakesAndIgnoresCommands)
{
  Bench b;
  b.drive_ms(300.0, 3.0f);
  b.send_raw(op::MSG_ESTOP, {});
  EXPECT_TRUE(b.core.estop_latched());
  for (size_t i = 0; i < 4; ++i) {
    EXPECT_EQ(b.hal.cmd_duty[i], 0.0f);
    EXPECT_TRUE(b.hal.cmd_brake[i]);
  }
  b.drive_ms(500.0, 3.0f);  // commands keep coming; they must be ignored
  EXPECT_TRUE(b.core.estop_latched());
  EXPECT_EQ(b.hal.cmd_duty[0], 0.0f);
  EXPECT_NEAR(b.core.speed_rad_s(0), 0.0f, 0.05f);
  op::StateMsg s;
  ASSERT_TRUE(b.last(s));
  EXPECT_TRUE(s.faults & op::fault::kEstopLatched);
}

TEST(Estop, ClearNeedsMagicThenAFreshCommand)
{
  Bench b;
  b.drive_ms(100.0, 3.0f);
  b.send_raw(op::MSG_ESTOP, {});
  op::EstopClearMsg wrong;
  wrong.magic = 0xDEADBEEF;
  b.send(wrong);
  EXPECT_TRUE(b.core.estop_latched());
  b.send_raw(op::MSG_ESTOP_CLEAR, {0xDE, 0xC0});  // truncated payload
  EXPECT_TRUE(b.core.estop_latched());

  op::EstopClearMsg right;
  b.send(right);
  EXPECT_FALSE(b.core.estop_latched());
  b.run_ms(50.0);
  EXPECT_FALSE(b.core.control_active()) << "must not resume the pre-e-stop command";
  b.drive_ms(400.0, 2.0f);
  EXPECT_TRUE(b.core.control_active());
  EXPECT_NEAR(b.core.speed_rad_s(0), 2.0f, 0.06f);
}

TEST(Estop, WatchdogDoesNotFireWhileLatched)
{
  Bench b;
  b.drive_ms(100.0, 1.0f);
  b.send_raw(op::MSG_ESTOP, {});
  b.run_ms(1000.0);
  EXPECT_EQ(b.core.stats().watchdog_trips, 0u);
}

// ------------------------------------------------------------------------------------------
// Link: rates, sequence gaps, params, ping, hostile input
// ------------------------------------------------------------------------------------------

TEST(Link, StateAt100HzAndStatsAt1Hz)
{
  Bench b;
  b.rx.clear();
  b.drive_ms(2000.0, 1.0f);
  EXPECT_NEAR(static_cast<double>(b.count(op::MSG_STATE)), 200.0, 1.0);
  EXPECT_NEAR(static_cast<double>(b.count(op::MSG_STATS)), 2.0, 1.0);
}

TEST(Link, CountsCmdSequenceGapsAcrossWrap)
{
  Bench b;
  b.host_seq = 250;
  b.send_cmd(0.0f);   // 250
  b.send_cmd(0.0f);   // 251
  b.host_seq = 254;   // 252, 253 lost
  b.send_cmd(0.0f);   // 254
  b.send_cmd(0.0f);   // 255
  b.send_cmd(0.0f);   // 0 (wrap, no gap)
  b.host_seq = 5;     // 1..4 lost
  b.send_cmd(0.0f);
  EXPECT_EQ(b.core.stats().cmd_seq_gaps, 6u);
  b.run_ms(1000.0);
  op::StatsMsg st;
  ASSERT_TRUE(b.last(st));
  EXPECT_EQ(st.cmd_seq_gaps, 6u);
}

TEST(Link, PingReturnsVersionHashAndParamCrc)
{
  Bench b;
  b.send_raw(op::MSG_PING, {});
  op::PongMsg pong;
  ASSERT_TRUE(b.last(pong));
  EXPECT_EQ(pong.proto_ver, op::kProtocolVersion);
  EXPECT_EQ(pong.fw_git_hash, 0x12345678u);
  EXPECT_EQ(pong.param_crc, b.core.params().crc());
}

TEST(Link, SetParamAppliesValidatesAndAcks)
{
  Bench b;
  const uint16_t crc_before = b.core.params().crc();
  op::SetParamMsg req;
  req.id = op::param_id::kEncoderCpr;
  req.value = 4096.0f;
  b.send(req);
  op::ParamAckMsg ack;
  ASSERT_TRUE(b.last(ack));
  EXPECT_EQ(ack.id, op::param_id::kEncoderCpr);
  EXPECT_EQ(ack.status, op::param_status::kOk);
  EXPECT_EQ(ack.value, 4096.0f);
  EXPECT_NE(b.core.params().crc(), crc_before);

  req.id = op::param_id::kMaxDuty;
  req.value = 1.5f;  // out of range
  b.send(req);
  ASSERT_TRUE(b.last(ack));
  EXPECT_EQ(ack.status, op::param_status::kOutOfRange);
  EXPECT_EQ(ack.value, 1.0f);

  req.value = NAN;
  b.send(req);
  ASSERT_TRUE(b.last(ack));
  EXPECT_EQ(ack.status, op::param_status::kOutOfRange);

  req.id = 200;
  req.value = 1.0f;
  b.send(req);
  ASSERT_TRUE(b.last(ack));
  EXPECT_EQ(ack.status, op::param_status::kUnknownId);
}

TEST(Link, PositionUsesEncoderCprAndGearRatio)
{
  Bench b;
  b.hal.wheel[0].angle = 1.0;  // rad, with the default 2048 CPR x 30:1
  b.run_ms(20.0);
  op::StateMsg s;
  ASSERT_TRUE(b.last(s));
  EXPECT_NEAR(s.position_mrad[0], 1000, 1);
  EXPECT_EQ(s.position_mrad[1], 0);
}

TEST(Link, EncoderWrapDoesNotJumpPosition)
{
  Bench b;
  b.hal.count_offset = INT32_MAX - 500;  // decoder sits just below the wrap
  b.core.begin();
  b.drive_ms(1000.0, 3.0f);  // ~29000 counts: wraps the int32 decoder
  op::StateMsg s;
  ASSERT_TRUE(b.last(s));
  EXPECT_NEAR(s.position_mrad[0], b.hal.wheel[0].angle * 1000.0, 2.0);
  EXPECT_GT(s.position_mrad[0], 2000);
}

TEST(Link, CorruptAndUnknownFramesAreCountedNeverCrash)
{
  Bench b;
  std::mt19937 rng(3);
  for (int i = 0; i < 50000; ++i) {
    b.core.on_rx_byte(rng() % 6 == 0 ? 0 : static_cast<uint8_t>(rng()));
  }
  b.core.on_rx_byte(0);
  b.send_raw(0x7E, {1, 2, 3});              // unknown type, valid CRC
  b.send_raw(op::MSG_CMD, {1, 2, 3});       // CMD with the wrong size
  b.send_raw(op::MSG_STATE, std::vector<uint8_t>(op::StateMsg::kSize));  // host-bound type
  b.run_ms(1100.0);
  op::StatsMsg st;
  ASSERT_TRUE(b.last(st));
  EXPECT_GT(st.rx_framing_errors + st.rx_crc_errors, 100u);
  EXPECT_GE(st.rx_framing_errors, 3u);
  // Still alive and controllable afterwards.
  b.drive_ms(300.0, 1.0f);
  EXPECT_TRUE(b.core.control_active());
}

// ------------------------------------------------------------------------------------------
// Encoder and battery faults
// ------------------------------------------------------------------------------------------

TEST(Faults, FrozenEncoderUnderDutyRaisesEncoderFault)
{
  Bench b;
  b.drive_ms(200.0, 3.0f);
  b.hal.encoder_frozen[3] = true;  // e.g. encoder cable unplugged
  b.drive_ms(400.0, 3.0f);
  EXPECT_FALSE(b.core.faults() & op::fault::kEncoder) << "needs 0.5 s of stall";
  b.drive_ms(300.0, 3.0f);
  EXPECT_TRUE(b.core.faults() & op::fault::kEncoder);
  b.hal.encoder_frozen[3] = false;
  b.drive_ms(50.0, 3.0f);
  EXPECT_FALSE(b.core.faults() & op::fault::kEncoder);
}

TEST(Faults, UndervoltageWithHysteresis)
{
  Bench b;
  b.hal.vbat = 10.0;  // below the 10.5 V default
  b.run_ms(20.0);
  EXPECT_TRUE(b.core.faults() & op::fault::kUndervoltage);
  b.hal.vbat = 10.6;  // inside the 0.2 V hysteresis band
  b.run_ms(20.0);
  EXPECT_TRUE(b.core.faults() & op::fault::kUndervoltage);
  b.hal.vbat = 10.8;
  b.run_ms(20.0);
  EXPECT_FALSE(b.core.faults() & op::fault::kUndervoltage);
  op::StateMsg s;
  ASSERT_TRUE(b.last(s));
  EXPECT_EQ(s.vbat_mv, 10800);
  b.hal.vbat_sensed = false;
  b.run_ms(20.0);
  ASSERT_TRUE(b.last(s));
  EXPECT_EQ(s.vbat_mv, 0);
  EXPECT_FALSE(s.status & op::status::kVbatSensed);
}
