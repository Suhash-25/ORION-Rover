// EstopLatch (council m1) and PerturbationGate (WheelPerturbation hook), without ROS.
#include <gtest/gtest.h>

#include <chrono>
#include <cmath>

#include "orion_hardware/safety_gates.hpp"
#include "orion_msgs/msg/safety_status.hpp"

using orion_hardware::EstopLatch;
using orion_hardware::PerturbationGate;
namespace ss = orion_hardware::safety_state;
using SafetyStatus = orion_msgs::msg::SafetyStatus;
using namespace std::chrono_literals;

static_assert(ss::kNominal == SafetyStatus::STATE_NOMINAL, "state values must match orion_msgs");
static_assert(ss::kLimited == SafetyStatus::STATE_LIMITED, "state values must match orion_msgs");
static_assert(ss::kHalt == SafetyStatus::STATE_HALT, "state values must match orion_msgs");
static_assert(ss::kEstop == SafetyStatus::STATE_ESTOP, "state values must match orion_msgs");

using Action = EstopLatch::Action;

class EstopLatchTest : public ::testing::Test
{
protected:
  /// Feeds `n` SafetyStatus messages at 50 Hz; returns the last action.
  Action statuses(uint8_t state, int n = 1)
  {
    Action a = Action::kNone;
    for (int i = 0; i < n; ++i) {
      t += 20ms;
      a = e.on_safety_state(state, t);
    }
    return a;
  }

  EstopLatch e{0.5};
  EstopLatch::Clock::time_point t = EstopLatch::Clock::now();
};

TEST_F(EstopLatchTest, ReleasingTheTopicAloneNeverClears)
{
  EXPECT_EQ(e.on_estop_topic(true), Action::kSendEstop);
  EXPECT_EQ(e.on_estop_topic(true), Action::kNone) << "ESTOP is sent once per latch";
  EXPECT_EQ(statuses(ss::kEstop), Action::kNone);
  EXPECT_EQ(e.on_estop_topic(false), Action::kNone);
  EXPECT_EQ(statuses(ss::kEstop, 100), Action::kNone) << "orion_safety still latched: no reset";
  EXPECT_TRUE(e.latched());
}

TEST_F(EstopLatchTest, ClearsOnlyAfterOrionSafetyLeavesEstop)
{
  e.on_estop_topic(true);
  statuses(ss::kEstop, 5);
  e.on_estop_topic(false);
  EXPECT_EQ(statuses(ss::kHalt), Action::kSendClear) << "reset accepted (HALT from another rule)";
  EXPECT_FALSE(e.latched());
  EXPECT_EQ(statuses(ss::kNominal), Action::kNone) << "ESTOP_CLEAR is sent once";
}

TEST_F(EstopLatchTest, NeedsOrionSafetyToHaveLatchedThisEstop)
{
  // A SafetyStatus that never showed ESTOP (orion_safety dead or not yet aware of the e-stop)
  // is no evidence of a reset.
  e.on_estop_topic(true);
  e.on_estop_topic(false);
  EXPECT_EQ(statuses(ss::kNominal, 10), Action::kNone);
  EXPECT_TRUE(e.latched());
}

TEST_F(EstopLatchTest, GapInTheStatusStreamIsNoEvidenceOfAReset)
{
  // orion_safety died after latching and a restarted one (which forgot the e-stop) reports
  // NOMINAL: no clear, and the old ESTOP status no longer counts.
  e.on_estop_topic(true);
  statuses(ss::kEstop, 5);
  e.on_estop_topic(false);
  t += 2s;
  EXPECT_EQ(statuses(ss::kNominal, 10), Action::kNone);
  EXPECT_TRUE(e.latched());
}

TEST_F(EstopLatchTest, ResetSeenBeforeOurTopicReleaseClearsOnTheRelease)
{
  // orion_safety's /orion/estop subscription can see the release earlier than ours.
  e.on_estop_topic(true);
  statuses(ss::kEstop, 5);
  EXPECT_EQ(statuses(ss::kNominal, 50), Action::kNone) << "topic still true here";
  EXPECT_EQ(e.on_estop_topic(false), Action::kSendClear);
  EXPECT_FALSE(e.latched());
}

TEST_F(EstopLatchTest, PressedAgainBeforeOurReleaseNeedsANewReset)
{
  e.on_estop_topic(true);
  statuses(ss::kEstop, 5);
  statuses(ss::kNominal);           // reset seen, our release still pending
  e.on_estop_topic(true);           // pressed again
  EXPECT_EQ(e.on_estop_topic(false), Action::kNone);
  EXPECT_TRUE(e.latched());
}

TEST_F(EstopLatchTest, SecondEstopNeedsItsOwnReset)
{
  e.on_estop_topic(true);
  statuses(ss::kEstop, 5);
  e.on_estop_topic(false);
  ASSERT_EQ(statuses(ss::kNominal), Action::kSendClear);
  EXPECT_EQ(e.on_estop_topic(true), Action::kSendEstop);
  e.on_estop_topic(false);
  EXPECT_EQ(statuses(ss::kNominal, 10), Action::kNone) << "the old ESTOP status must not count";
}

class PerturbationGateTest : public ::testing::Test
{
protected:
  PerturbationGate::Clock::time_point t0 = PerturbationGate::Clock::now();
  PerturbationGate gate{1.0};
  PerturbationGate::Offsets offsets{0.5, -1.0, 0.0, 0.25};
};

TEST_F(PerturbationGateTest, DisabledByDefault)
{
  PerturbationGate off;
  EXPECT_NE(off.request(offsets, 1.0, true, t0), "");
  EXPECT_EQ(off.offsets(t0)[0], 0.0);
}

TEST_F(PerturbationGateTest, AppliesForTheDurationThenStops)
{
  ASSERT_EQ(gate.request(offsets, 0.5, true, t0), "");
  EXPECT_EQ(gate.offsets(t0 + 100ms), offsets);
  EXPECT_EQ(gate.offsets(t0 + 499ms), offsets);
  EXPECT_EQ(gate.offsets(t0 + 500ms), PerturbationGate::Offsets{});
  EXPECT_FALSE(gate.active());
}

TEST_F(PerturbationGateTest, RejectsOffsetsOutsideTheBoundAndNaN)
{
  EXPECT_NE(gate.request({1.01, 0, 0, 0}, 1.0, true, t0), "");
  EXPECT_NE(gate.request({0, 0, 0, -1.5}, 1.0, true, t0), "");
  EXPECT_NE(gate.request({0, NAN, 0, 0}, 1.0, true, t0), "");
  EXPECT_FALSE(gate.active());
}

TEST_F(PerturbationGateTest, RejectedUnlessSafetyIsOk)
{
  EXPECT_NE(gate.request(offsets, 1.0, false, t0), "");
  EXPECT_FALSE(gate.active());
}

TEST_F(PerturbationGateTest, ZeroDurationCancelsAndNewMessageReplaces)
{
  ASSERT_EQ(gate.request(offsets, 5.0, true, t0), "");
  const PerturbationGate::Offsets other{0.1, 0.1, 0.1, 0.1};
  ASSERT_EQ(gate.request(other, 5.0, true, t0), "");
  EXPECT_EQ(gate.offsets(t0), other);
  EXPECT_EQ(gate.request(offsets, 0.0, false, t0), "") << "a cancel is always accepted";
  EXPECT_EQ(gate.offsets(t0), PerturbationGate::Offsets{});
}
