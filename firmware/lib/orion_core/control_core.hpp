// Portable firmware core: protocol handling, 200 Hz wheel PID, 200 ms command watchdog,
// per-wheel current clamp, e-stop latch, STATE at 100 Hz and STATS at 1 Hz.
//
// Board code only has to implement orion_fw::Hal, then call poll() from loop() and
// on_rx_byte() for every received byte. All timing comes from Hal::micros().
#ifndef ORION_CORE_CONTROL_CORE_HPP_
#define ORION_CORE_CONTROL_CORE_HPP_

#include <cstddef>
#include <cstdint>

#include "hal.hpp"
#include "orion_proto.hpp"
#include "params.hpp"
#include "pid.hpp"

namespace orion_fw
{

class ControlCore
{
public:
  static constexpr size_t kWheels = orion_proto::kNumWheels;
  static constexpr uint32_t kPidPeriodUs = 5000;       // 200 Hz
  static constexpr uint32_t kStatePeriodUs = 10000;    // 100 Hz
  static constexpr uint32_t kStatsPeriodUs = 1000000;  // 1 Hz
  static constexpr uint32_t kWatchdogUs = 200000;      // 200 ms without a CMD frame
  static constexpr float kFoldbackPerStep = 0.9f;      // duty cap factor per step over the limit
  static constexpr float kCapRecoveryPerS = 0.5f;      // duty cap recovery rate once under it
  static constexpr float kStallDuty = 0.3f;            // encoder check: |duty| at least this ...
  static constexpr float kStallTimeS = 0.5f;           // ... with no count change for this long
  static constexpr float kVbatHysteresisV = 0.2f;

  /// `fw_git_hash`: first 8 hex digits of the firmware commit, reported in PONG.
  ControlCore(Hal & hal, uint32_t fw_git_hash);

  /// Call once after the HAL is ready. Motors start idle; the watchdog fault is set until the
  /// first CMD arrives (that initial state is not counted as a trip).
  void begin();

  /// Feed every byte received from the host.
  void on_rx_byte(uint8_t byte);

  /// Call as often as possible (at least every 1 ms). Runs whatever is due.
  void poll();

  // ---- Introspection (tests, board diagnostics) ----
  bool estop_latched() const {return estop_latched_;}
  bool watchdog_tripped() const {return watchdog_tripped_;}
  bool control_active() const;
  uint8_t faults() const;
  uint8_t status() const;
  float duty(size_t wheel) const {return wheels_[wheel].duty;}
  float duty_cap(size_t wheel) const {return wheels_[wheel].duty_cap;}
  float speed_rad_s(size_t wheel) const {return wheels_[wheel].speed_rad_s;}
  float target_rad_s(size_t wheel) const {return wheels_[wheel].target_rad_s;}
  const orion_proto::StatsMsg & stats() const {return stats_;}
  const Params & params() const {return params_;}

private:
  struct Wheel
  {
    Pid pid;
    int64_t position_counts = 0;
    int32_t last_count = 0;
    float speed_rad_s = 0.0f;
    float target_rad_s = 0.0f;
    float duty = 0.0f;
    float current_a = 0.0f;
    bool current_valid = false;
    float duty_cap = 1.0f;
    float over_limit_s = 0.0f;
    bool limiting = false;
    float stall_s = 0.0f;
    bool encoder_fault = false;
  };

  void handle_frame(const orion_proto::Frame & frame);
  void handle_cmd(const orion_proto::Frame & frame);
  void control_step(float dt);
  void update_current_limit(Wheel & w, float dt);
  void update_encoder_check(Wheel & w, int32_t delta, float dt);
  void apply_idle();
  void send_state();
  void send_log(const char * text);
  template<typename Msg>
  void send(const Msg & msg);
  void send_raw(uint8_t type, const uint8_t * payload, size_t len);
  double rad_per_count() const;

  Hal & hal_;
  const uint32_t fw_git_hash_;
  Params params_;
  orion_proto::FrameDecoder decoder_;
  orion_proto::StatsMsg stats_;
  Wheel wheels_[kWheels];

  uint32_t last_pid_us_ = 0;
  uint32_t last_state_us_ = 0;
  uint32_t last_stats_us_ = 0;
  uint32_t last_cmd_us_ = 0;
  uint8_t tx_seq_ = 0;
  uint8_t last_cmd_seq_ = 0;
  bool have_cmd_seq_ = false;
  bool cmd_enable_ = false;
  bool brake_on_idle_ = false;
  bool watchdog_tripped_ = true;
  bool estop_latched_ = false;
  bool was_active_ = false;
  bool vbat_valid_ = false;
  float vbat_v_ = 0.0f;
  bool undervoltage_ = false;
  uint32_t rx_bad_payload_ = 0;
};

}  // namespace orion_fw

#endif  // ORION_CORE_CONTROL_CORE_HPP_
