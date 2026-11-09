// Pure, ROS-free decisions of OrionSystem that depend on orion_safety (unit-tested):
// when the firmware e-stop latch may be cleared, and the wheel-perturbation hook.
#ifndef ORION_HARDWARE__SAFETY_GATES_HPP_
#define ORION_HARDWARE__SAFETY_GATES_HPP_

#include <array>
#include <chrono>
#include <cstdint>
#include <string>

namespace orion_hardware
{

/// orion_msgs/SafetyStatus.STATE_* (same values; checked by test_safety_gates).
namespace safety_state
{
constexpr uint8_t kNominal = 0;
constexpr uint8_t kLimited = 1;
constexpr uint8_t kHalt = 2;
constexpr uint8_t kEstop = 3;
}  // namespace safety_state

/// When the firmware's ESTOP latch may be released (council m1, docs/SAFETY.md decision 14).
///
/// /orion/estop true latches it. Releasing the topic is not enough: ESTOP_CLEAR is sent only
/// while /orion/estop is false and after orion_safety has shown the reset: a SafetyStatus in
/// STATE_ESTOP received after the latch, then one that left STATE_ESTOP at most max_gap later
/// (orion_safety publishes at 50 Hz, so a longer gap means it died or restarted, and a
/// restarted orion_safety has forgotten the e-stop). Only an accepted /orion/estop_reset
/// (confirm == "CLEAR ESTOP") makes orion_safety leave STATE_ESTOP. Without a running
/// orion_safety the firmware stays latched (fail-safe).
class EstopLatch
{
public:
  using Clock = std::chrono::steady_clock;
  enum class Action {kNone, kSendEstop, kSendClear};

  explicit EstopLatch(double max_gap_s = 0.5)
  : max_gap_(std::chrono::duration_cast<Clock::duration>(
        std::chrono::duration<double>(max_gap_s))) {}

  /// /orion/estop. kSendEstop when the latch was just set; kSendClear when a reset already
  /// seen only waited for the topic release.
  Action on_estop_topic(bool engaged)
  {
    topic_engaged_ = engaged;
    if (engaged) {
      reset_seen_ = false;  // pressed again: this press needs its own reset
      if (!latched_) {
        latched_ = true;
        safety_latched_ = false;
        return Action::kSendEstop;
      }
      return Action::kNone;
    }
    return try_clear();
  }

  /// A SafetyStatus state received at `now`. kSendClear when the latch was just released.
  Action on_safety_state(uint8_t state, Clock::time_point now)
  {
    if (!latched_) {
      return Action::kNone;
    }
    if (state == safety_state::kEstop) {
      safety_latched_ = true;
      reset_seen_ = false;
      last_estop_status_ = now;
      return Action::kNone;
    }
    if (safety_latched_ && !reset_seen_) {
      if (now - last_estop_status_ <= max_gap_) {
        reset_seen_ = true;
      } else {
        safety_latched_ = false;  // a gap in the stream is no evidence of a reset
      }
    }
    return try_clear();
  }

  bool latched() const {return latched_;}
  bool topic_engaged() const {return topic_engaged_;}

private:
  Action try_clear()
  {
    if (latched_ && reset_seen_ && !topic_engaged_) {
      latched_ = false;
      safety_latched_ = false;
      reset_seen_ = false;
      return Action::kSendClear;
    }
    return Action::kNone;
  }

  Clock::duration max_gap_;
  bool latched_ = false;
  bool topic_engaged_ = false;
  bool safety_latched_ = false;  // orion_safety reported STATE_ESTOP since the latch was set
  bool reset_seen_ = false;      // ... and then left it without a gap in the stream
  Clock::time_point last_estop_status_{};
};

/// Bounded wheel-velocity offsets for fault-injection experiments (orion_msgs/WheelPerturbation,
/// reserved for Round 7). Disabled when max_rad_s == 0.
class PerturbationGate
{
public:
  using Clock = std::chrono::steady_clock;
  using Offsets = std::array<double, 4>;  // rad/s per joint, URDF order FL FR RL RR

  explicit PerturbationGate(double max_rad_s = 0.0)
  : max_rad_s_(max_rad_s) {}

  /// Starts (or replaces) a perturbation, or cancels the active one when duration_s <= 0.
  /// `safety_ok`: the latest SafetyStatus is fresh and NOMINAL or LIMITED, and no e-stop.
  /// Returns "" when accepted, else why it was rejected (nothing changes then).
  std::string request(
    const Offsets & offsets, double duration_s, bool safety_ok, Clock::time_point now)
  {
    if (max_rad_s_ <= 0.0) {
      return "disabled (hardware parameter perturbation_max_rad_s is 0)";
    }
    if (!(duration_s > 0.0)) {
      cancel();
      return "";
    }
    if (!safety_ok) {
      return "safety state is not NOMINAL/LIMITED (or SafetyStatus is stale, or e-stopped)";
    }
    for (double o : offsets) {
      if (!(o >= -max_rad_s_ && o <= max_rad_s_)) {  // also rejects NaN
        return "offset " + std::to_string(o) + " rad/s outside +-" + std::to_string(max_rad_s_);
      }
    }
    offsets_ = offsets;
    end_ = now + std::chrono::duration_cast<Clock::duration>(
      std::chrono::duration<double>(duration_s));
    active_ = true;
    return "";
  }

  void cancel() {active_ = false;}

  /// The offsets to add at `now`; zeros when none is active (an expired one is dropped).
  Offsets offsets(Clock::time_point now)
  {
    if (active_ && now >= end_) {
      active_ = false;
    }
    return active_ ? offsets_ : Offsets{};
  }

  bool active() const {return active_;}
  double max_rad_s() const {return max_rad_s_;}

private:
  double max_rad_s_;
  bool active_ = false;
  Offsets offsets_{};
  Clock::time_point end_{};
};

}  // namespace orion_hardware

#endif  // ORION_HARDWARE__SAFETY_GATES_HPP_
