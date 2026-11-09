#include "control_core.hpp"

#include <cmath>
#include <cstring>

namespace orion_fw
{

namespace op = orion_proto;

namespace
{
constexpr double kTwoPi = 6.283185307179586;

int16_t saturate_i16(double v)
{
  if (!(v == v)) {  // NaN
    return 0;
  }
  if (v > 32767.0) {
    return 32767;
  }
  if (v < -32768.0) {
    return -32768;
  }
  return static_cast<int16_t>(std::lround(v));
}

/// Measured current in mA, kept clear of the 0x7FFF "not available" marker.
int16_t current_to_ma(float amps)
{
  const int16_t ma = saturate_i16(amps * 1000.0);
  return ma == op::kCurrentNotAvailable ? static_cast<int16_t>(op::kCurrentNotAvailable - 1) : ma;
}

float abs_f(float v) {return v < 0.0f ? -v : v;}
}  // namespace

ControlCore::ControlCore(Hal & hal, uint32_t fw_git_hash)
: hal_(hal), fw_git_hash_(fw_git_hash) {}

void ControlCore::begin()
{
  const uint32_t now = hal_.micros();
  last_pid_us_ = now;
  last_state_us_ = now;
  last_stats_us_ = now;
  last_cmd_us_ = now;
  for (size_t i = 0; i < kWheels; ++i) {
    wheels_[i].last_count = hal_.encoder_count(i);
    wheels_[i].duty_cap = params_.get(op::param_id::kMaxDuty);
  }
  apply_idle();
  send_log("ORION firmware up");
}

bool ControlCore::control_active() const
{
  return cmd_enable_ && !watchdog_tripped_ && !estop_latched_;
}

uint8_t ControlCore::faults() const
{
  uint8_t f = 0;
  if (watchdog_tripped_) {
    f |= op::fault::kWatchdog;
  }
  if (estop_latched_) {
    f |= op::fault::kEstopLatched;
  }
  if (undervoltage_) {
    f |= op::fault::kUndervoltage;
  }
  for (const Wheel & w : wheels_) {
    if (w.limiting) {
      f |= op::fault::kOvercurrent;
    }
    if (w.encoder_fault) {
      f |= op::fault::kEncoder;
    }
  }
  return f;
}

uint8_t ControlCore::status() const
{
  uint8_t s = 0;
  if (control_active()) {
    s |= op::status::kEnabled;
  }
  bool all_sensed = true;
  for (const Wheel & w : wheels_) {
    all_sensed = all_sensed && w.current_valid;
  }
  if (all_sensed) {
    s |= op::status::kCurrentSensed;
  }
  if (vbat_valid_) {
    s |= op::status::kVbatSensed;
  }
  if (brake_on_idle_) {
    s |= op::status::kBrakeOnIdle;
  }
  return s;
}

void ControlCore::on_rx_byte(uint8_t byte)
{
  if (decoder_.push(byte)) {
    handle_frame(decoder_.frame());
  }
}

void ControlCore::handle_frame(const op::Frame & frame)
{
  switch (frame.type) {
    case op::MSG_CMD:
      handle_cmd(frame);
      return;
    case op::MSG_PING:
      if (frame.len != 0) {
        break;
      }
      {
        op::PongMsg pong;
        pong.fw_git_hash = fw_git_hash_;
        pong.param_crc = params_.crc();
        send(pong);
      }
      return;
    case op::MSG_SET_PARAM: {
        op::SetParamMsg req;
        if (!req.unpack(frame.payload, frame.len)) {
          break;
        }
        op::ParamAckMsg ack;
        ack.id = req.id;
        ack.status = params_.set(req.id, req.value);
        ack.value = params_.get(req.id);
        if (req.id == op::param_id::kMaxDuty && ack.status == op::param_status::kOk) {
          for (Wheel & w : wheels_) {
            w.duty_cap = w.limiting && w.duty_cap < ack.value ? w.duty_cap : ack.value;
          }
        }
        send(ack);
        return;
      }
    case op::MSG_ESTOP:
      if (frame.len != 0) {
        break;
      }
      if (!estop_latched_) {
        estop_latched_ = true;
        apply_idle();
        send_log("ESTOP latched");
      }
      return;
    case op::MSG_ESTOP_CLEAR: {
        op::EstopClearMsg clear;
        if (!clear.unpack(frame.payload, frame.len)) {
          break;
        }
        if (clear.magic != op::kEstopClearMagic) {
          send_log("ESTOP_CLEAR rejected: bad magic");
          return;
        }
        if (estop_latched_) {
          estop_latched_ = false;
          // Require a fresh CMD before moving again; this is not a watchdog trip.
          watchdog_tripped_ = true;
          cmd_enable_ = false;
          send_log("ESTOP cleared");
        }
        return;
      }
    default:
      break;
  }
  // Unknown type, a host-bound type, or a payload of the wrong size.
  ++rx_bad_payload_;
}

void ControlCore::handle_cmd(const op::Frame & frame)
{
  op::CmdMsg cmd;
  if (!cmd.unpack(frame.payload, frame.len)) {
    ++rx_bad_payload_;
    return;
  }
  if (have_cmd_seq_) {
    stats_.cmd_seq_gaps += op::seq_gap(last_cmd_seq_, frame.seq);
  }
  last_cmd_seq_ = frame.seq;
  have_cmd_seq_ = true;
  if (estop_latched_) {
    return;  // ignored until ESTOP_CLEAR; does not feed the watchdog
  }
  last_cmd_us_ = hal_.micros();
  watchdog_tripped_ = false;
  cmd_enable_ = (cmd.flags & op::cmd_flag::kEnable) != 0;
  brake_on_idle_ = (cmd.flags & op::cmd_flag::kBrakeOnIdle) != 0;
  for (size_t i = 0; i < kWheels; ++i) {
    wheels_[i].target_rad_s = cmd_enable_ ? cmd.w_target_mrad_s[i] * 1.0e-3f : 0.0f;
  }
}

void ControlCore::poll()
{
  const uint32_t now = hal_.micros();

  if (!watchdog_tripped_ && !estop_latched_ && now - last_cmd_us_ >= kWatchdogUs) {
    watchdog_tripped_ = true;
    ++stats_.watchdog_trips;
    apply_idle();
    send_log("watchdog: no CMD for 200 ms");
  }

  const uint32_t since_pid = now - last_pid_us_;
  if (since_pid >= kPidPeriodUs) {
    // Keep a fixed 200 Hz grid; if we fell behind by more than a period, resynchronise.
    last_pid_us_ = since_pid >= 2 * kPidPeriodUs ? now : last_pid_us_ + kPidPeriodUs;
    control_step(static_cast<float>(since_pid) * 1.0e-6f);
  }

  const uint32_t since_state = now - last_state_us_;
  if (since_state >= kStatePeriodUs) {
    last_state_us_ = since_state >= 2 * kStatePeriodUs ? now : last_state_us_ + kStatePeriodUs;
    send_state();
  }

  if (now - last_stats_us_ >= kStatsPeriodUs) {
    last_stats_us_ = now;
    const op::DecoderCounters & c = decoder_.counters();
    stats_.rx_frames_ok = c.frames_ok;
    stats_.rx_crc_errors = c.crc_errors;
    stats_.rx_framing_errors = c.framing_errors + rx_bad_payload_;
    send(stats_);
  }
}

double ControlCore::rad_per_count() const
{
  return kTwoPi /
         (static_cast<double>(params_.get(op::param_id::kEncoderCpr)) *
         params_.get(op::param_id::kGearRatio));
}

void ControlCore::control_step(float dt)
{
  const bool active = control_active();
  if (active && !was_active_) {
    for (Wheel & w : wheels_) {
      w.pid.reset();
    }
  }
  was_active_ = active;

  const PidGains gains{
    params_.get(op::param_id::kKp), params_.get(op::param_id::kKi),
    params_.get(op::param_id::kKd), params_.get(op::param_id::kKff)};
  const float tau = params_.get(op::param_id::kVelFilterTauS);
  const float alpha = dt / (tau + dt);
  const double k = rad_per_count();

  for (size_t i = 0; i < kWheels; ++i) {
    Wheel & w = wheels_[i];
    const int32_t count = hal_.encoder_count(i);
    // Wrap-safe: unsigned subtraction, then reinterpret as signed.
    const int32_t delta = static_cast<int32_t>(
      static_cast<uint32_t>(count) - static_cast<uint32_t>(w.last_count));
    w.last_count = count;
    w.position_counts += delta;
    const float raw_speed = static_cast<float>(delta * k) / dt;
    w.speed_rad_s += alpha * (raw_speed - w.speed_rad_s);

    float amps = 0.0f;
    w.current_valid = hal_.read_current(i, amps);
    w.current_a = w.current_valid ? amps : 0.0f;
    update_current_limit(w, dt);
    update_encoder_check(w, delta, dt);

    if (active) {
      w.duty = w.pid.update(gains, w.target_rad_s, w.speed_rad_s, dt, w.duty_cap);
      hal_.set_motor(i, w.duty, false);
    } else {
      w.duty = 0.0f;
      w.pid.reset();
      hal_.set_motor(i, 0.0f, estop_latched_ || brake_on_idle_);
    }
  }

  float volts = 0.0f;
  vbat_valid_ = hal_.read_battery_voltage(volts);
  vbat_v_ = vbat_valid_ ? volts : 0.0f;
  const float vmin = params_.get(op::param_id::kVbatMinV);
  if (!vbat_valid_) {
    undervoltage_ = false;
  } else if (vbat_v_ < vmin) {
    undervoltage_ = true;
  } else if (vbat_v_ > vmin + kVbatHysteresisV) {
    undervoltage_ = false;
  }
}

void ControlCore::update_current_limit(Wheel & w, float dt)
{
  const float max_duty = params_.get(op::param_id::kMaxDuty);
  if (!w.current_valid) {
    // No sensing: the firmware hard limit cannot be enforced (see docs/HARDWARE.md).
    w.limiting = false;
    w.over_limit_s = 0.0f;
    w.duty_cap = max_duty;
    return;
  }
  const float limit = params_.get(op::param_id::kCurrentLimitA);
  const bool over = abs_f(w.current_a) > limit;
  w.over_limit_s = over ? w.over_limit_s + dt : 0.0f;
  if (!w.limiting && w.over_limit_s >= params_.get(op::param_id::kCurrentLimitTimeS)) {
    w.limiting = true;
    ++stats_.overcurrent_trips;
    w.duty_cap = abs_f(w.duty) < max_duty ? abs_f(w.duty) : max_duty;
  }
  if (!w.limiting) {
    w.duty_cap = max_duty;
    return;
  }
  if (over) {
    w.duty_cap *= kFoldbackPerStep;
  } else {
    w.duty_cap += kCapRecoveryPerS * dt;
    if (w.duty_cap >= max_duty) {
      w.duty_cap = max_duty;
      w.limiting = false;
    }
  }
}

void ControlCore::update_encoder_check(Wheel & w, int32_t delta, float dt)
{
  if (delta != 0) {
    w.stall_s = 0.0f;
    w.encoder_fault = false;
    return;
  }
  w.stall_s = abs_f(w.duty) >= kStallDuty ? w.stall_s + dt : 0.0f;
  if (w.stall_s >= kStallTimeS) {
    w.encoder_fault = true;
  }
}

void ControlCore::apply_idle()
{
  for (size_t i = 0; i < kWheels; ++i) {
    wheels_[i].duty = 0.0f;
    wheels_[i].target_rad_s = 0.0f;
    wheels_[i].pid.reset();
    hal_.set_motor(i, 0.0f, estop_latched_ || brake_on_idle_);
  }
}

void ControlCore::send_state()
{
  op::StateMsg s;
  s.t_us = hal_.micros();
  const double mrad_per_count = rad_per_count() * 1000.0;
  for (size_t i = 0; i < kWheels; ++i) {
    const Wheel & w = wheels_[i];
    const int64_t mrad = std::llround(static_cast<double>(w.position_counts) * mrad_per_count);
    s.position_mrad[i] = static_cast<int32_t>(static_cast<uint32_t>(mrad));  // wraps at int32
    s.velocity_mrad_s[i] = saturate_i16(w.speed_rad_s * 1000.0);
    s.current_ma[i] = w.current_valid ? current_to_ma(w.current_a) : op::kCurrentNotAvailable;
    s.duty[i] = saturate_i16(w.duty * op::kDutyFullScale);
  }
  const double mv = vbat_v_ * 1000.0;
  s.vbat_mv = static_cast<uint16_t>(mv < 0.0 ? 0.0 : (mv > 65535.0 ? 65535.0 : mv));
  s.faults = faults();
  s.status = status();
  send(s);
}

void ControlCore::send_log(const char * text)
{
  const size_t n = std::strlen(text);
  send_raw(
    op::MSG_LOG, reinterpret_cast<const uint8_t *>(text),
    n > op::kMaxPayload ? op::kMaxPayload : n);
}

template<typename Msg>
void ControlCore::send(const Msg & msg)
{
  uint8_t payload[Msg::kSize];
  msg.pack(payload);
  send_raw(Msg::kType, payload, Msg::kSize);
}

void ControlCore::send_raw(uint8_t type, const uint8_t * payload, size_t len)
{
  uint8_t wire[op::kMaxWireFrame];
  const size_t n = op::encode_frame(type, tx_seq_++, payload, len, wire, sizeof(wire));
  if (n > 0) {
    hal_.serial_write(wire, n);
  }
}

}  // namespace orion_fw
