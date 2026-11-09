// Fake board for host tests: settable clock, simple DC-motor wheel plant, captured serial output.
#ifndef FAKE_HAL_HPP_
#define FAKE_HAL_HPP_

#include <cmath>
#include <cstdint>
#include <vector>

#include "control_core.hpp"
#include "orion_proto.hpp"

namespace orion_test
{

namespace op = orion_proto;

/// Brushed DC motor + gearbox seen from the wheel: I = (duty V - ke w) / R, J dw/dt = kt I - b w.
struct WheelPlant
{
  double ke = 2.0;        // V per wheel rad/s (=> ~6 rad/s free speed at 12 V)
  double kt = 2.0;        // N m per A at the wheel
  double r_ohm = 0.5;     // winding resistance
  double j = 0.05;        // kg m^2 reflected inertia
  double b = 0.01;        // N m s viscous friction
  double load_nm = 0.0;   // external load torque opposing motion
  bool locked = false;    // stalled wheel (rock, entrapped)
  double w = 0.0;         // rad/s
  double angle = 0.0;     // rad
  double current = 0.0;   // A
};

class FakeHal : public orion_fw::Hal
{
public:
  static constexpr double kCountsPerRad = 2048.0 * 30.0 / 6.283185307179586;  // default params

  uint32_t now_us = 0;
  double vbat = 12.0;
  bool current_sensed = true;
  bool vbat_sensed = true;
  bool encoder_frozen[4] = {false, false, false, false};
  WheelPlant wheel[4];
  float cmd_duty[4] = {0, 0, 0, 0};
  bool cmd_brake[4] = {false, false, false, false};
  std::vector<uint8_t> tx;

  void set_motor(size_t i, float duty, bool brake) override
  {
    cmd_duty[i] = duty;
    cmd_brake[i] = brake;
  }
  int32_t encoder_count(size_t i) override
  {
    if (!encoder_frozen[i]) {
      frozen_count_[i] = static_cast<int32_t>(
        static_cast<uint32_t>(static_cast<int64_t>(std::llround(wheel[i].angle * kCountsPerRad))));
    }
    // Unsigned add: the decoder wraps like real hardware (signed overflow would be UB).
    return static_cast<int32_t>(
      static_cast<uint32_t>(frozen_count_[i]) + static_cast<uint32_t>(count_offset));
  }
  bool read_current(size_t i, float & amps) override
  {
    amps = static_cast<float>(wheel[i].current);
    return current_sensed;
  }
  bool read_battery_voltage(float & volts) override
  {
    volts = static_cast<float>(vbat);
    return vbat_sensed;
  }
  uint32_t micros() override {return now_us;}
  void serial_write(const uint8_t * data, size_t len) override
  {
    tx.insert(tx.end(), data, data + len);
  }

  /// Advances the plant by dt seconds using the last commanded duty.
  void step_plant(double dt)
  {
    for (size_t i = 0; i < 4; ++i) {
      WheelPlant & p = wheel[i];
      const double v = cmd_duty[i] * vbat;
      if (p.locked) {
        p.w = 0.0;
        p.current = v / p.r_ohm;
        continue;
      }
      p.current = (v - p.ke * p.w) / p.r_ohm;
      if (cmd_duty[i] == 0.0f && cmd_brake[i]) {
        p.current = -p.ke * p.w / p.r_ohm;  // shorted terminals
      } else if (cmd_duty[i] == 0.0f) {
        p.current = 0.0;  // coasting: open circuit
      }
      const double torque = p.kt * p.current - p.b * p.w - std::copysign(p.load_nm, p.w);
      p.w += torque / p.j * dt;
      p.angle += p.w * dt;
    }
  }

  int32_t count_offset = 0;  // lets tests start the decoder near the int32 wrap

private:
  int32_t frozen_count_[4] = {0, 0, 0, 0};
};

/// Firmware core + fake board + a host-side decoder for what the firmware sent.
class Bench
{
public:
  explicit Bench(uint32_t start_us = 1000)
  : core(hal, 0x12345678u)
  {
    hal.now_us = start_us;
    core.begin();
  }

  /// Runs for `ms` milliseconds in 100 us steps, polling the core every step.
  void run_ms(double ms)
  {
    const int steps = static_cast<int>(std::lround(ms * 10.0));
    for (int k = 0; k < steps; ++k) {
      hal.now_us += 100;
      hal.step_plant(100e-6);
      core.poll();
      drain();
    }
  }

  template<typename Msg>
  void send(const Msg & msg)
  {
    send_raw(Msg::kType, msg_bytes(msg));
  }

  template<typename Msg>
  static std::vector<uint8_t> msg_bytes(const Msg & msg)
  {
    std::vector<uint8_t> payload(Msg::kSize);
    msg.pack(payload.data());
    return payload;
  }

  void send_raw(uint8_t type, const std::vector<uint8_t> & payload)
  {
    uint8_t wire[op::kMaxWireFrame];
    const size_t n = op::encode_frame(type, host_seq++, payload.data(), payload.size(), wire,
        sizeof(wire));
    for (size_t i = 0; i < n; ++i) {
      core.on_rx_byte(wire[i]);
    }
    drain();
  }

  void send_cmd(float w_rad_s, uint8_t flags = op::cmd_flag::kEnable)
  {
    op::CmdMsg cmd;
    for (auto & w : cmd.w_target_mrad_s) {
      w = static_cast<int16_t>(std::lround(w_rad_s * 1000.0f));
    }
    cmd.flags = flags;
    send(cmd);
  }

  /// Runs `ms` while sending a CMD every 10 ms (the host's 100 Hz stream).
  void drive_ms(double ms, float w_rad_s, uint8_t flags = op::cmd_flag::kEnable)
  {
    for (double t = 0; t < ms; t += 10.0) {
      send_cmd(w_rad_s, flags);
      run_ms(10.0);
    }
  }

  void drain()
  {
    for (uint8_t b : hal.tx) {
      if (host_decoder.push(b)) {
        rx.push_back(host_decoder.frame());
      }
    }
    hal.tx.clear();
  }

  size_t count(uint8_t type) const
  {
    size_t n = 0;
    for (const auto & f : rx) {
      n += f.type == type ? 1 : 0;
    }
    return n;
  }

  template<typename Msg>
  bool last(Msg & out) const
  {
    for (auto it = rx.rbegin(); it != rx.rend(); ++it) {
      if (op::decode_msg(*it, out)) {
        return true;
      }
    }
    return false;
  }

  FakeHal hal;
  orion_fw::ControlCore core;
  op::FrameDecoder host_decoder;
  std::vector<op::Frame> rx;
  uint8_t host_seq = 0;
};

}  // namespace orion_test

#endif  // FAKE_HAL_HPP_
