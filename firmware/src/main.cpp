// Teensy 4.1 entry point: the HAL for this board plus Arduino setup()/loop().
// UNVERIFIED on hardware and not compiled in this repo's CI sandbox (no PlatformIO toolchain
// there); the portable core it drives is unit-tested on the host (firmware/test).
#include <Arduino.h>
#include <Encoder.h>  // PJRC Encoder library, bundled with the Teensy Arduino core

#include "board_config.hpp"
#include "control_core.hpp"

#ifndef ORION_FW_GIT_HASH
#define ORION_FW_GIT_HASH 0x00000000u  // set by tools/git_hash.py at build time
#endif

namespace
{

Stream & host_link()
{
  if (board::kUseUart) {
    return Serial1;
  }
  return Serial;
}

class Teensy41Hal : public orion_fw::Hal
{
public:
  Teensy41Hal()
  : enc_{
      Encoder(board::kMotors[0].enc_a, board::kMotors[0].enc_b),
      Encoder(board::kMotors[1].enc_a, board::kMotors[1].enc_b),
      Encoder(board::kMotors[2].enc_a, board::kMotors[2].enc_b),
      Encoder(board::kMotors[3].enc_a, board::kMotors[3].enc_b)} {}

  void begin()
  {
    analogWriteResolution(board::kPwmResolutionBits);
    analogReadResolution(board::kAdcResolutionBits);
    for (const auto & m : board::kMotors) {
      pinMode(m.dir, OUTPUT);
      pinMode(m.pwm, OUTPUT);
      analogWriteFrequency(m.pwm, board::kPwmFrequencyHz);
      analogWrite(m.pwm, 0);
    }
    if (board::kUseUart) {
      Serial1.begin(board::kUartBaud);
    } else {
      Serial.begin(921600);  // ignored by USB CDC
    }
  }

  void set_motor(size_t wheel, float duty, bool brake) override
  {
    const board::MotorPins & m = board::kMotors[wheel];
    if (m.invert_motor) {
      duty = -duty;
    }
    const uint32_t full = (1u << board::kPwmResolutionBits) - 1u;
    const float mag = duty < 0.0f ? -duty : duty;
    // PWM+DIR drivers of the MDD10A class brake when PWM is low; they cannot coast, so `brake`
    // makes no difference here. Drivers with a separate enable/coast input must use it.
    (void)brake;
    digitalWrite(m.dir, duty >= 0.0f ? HIGH : LOW);
    analogWrite(m.pwm, static_cast<int>(mag * static_cast<float>(full) + 0.5f));
  }

  int32_t encoder_count(size_t wheel) override
  {
    const int32_t c = enc_[wheel].read();
    return board::kMotors[wheel].invert_encoder ? -c : c;
  }

  bool read_current(size_t wheel, float & amps) override
  {
    const int8_t pin = board::kMotors[wheel].current_pin;
    if (pin < 0) {
      return false;
    }
    amps = (adc_volts(pin) - board::kCurrentOffsetV) * board::kCurrentAmpsPerV;
    if (board::kMotors[wheel].invert_motor) {
      amps = -amps;
    }
    return true;
  }

  bool read_battery_voltage(float & volts) override
  {
    if (board::kVbatPin < 0) {
      return false;
    }
    volts = adc_volts(board::kVbatPin) * board::kVbatDividerRatio;
    return true;
  }

  uint32_t micros() override {return ::micros();}

  void serial_write(const uint8_t * data, size_t len) override
  {
    Stream & s = host_link();
    // Never block the control loop: drop the frame if the host is not draining the link.
    if (s.availableForWrite() >= static_cast<int>(len)) {
      s.write(data, len);
    }
  }

private:
  static float adc_volts(uint8_t pin)
  {
    const float full = static_cast<float>((1u << board::kAdcResolutionBits) - 1u);
    return static_cast<float>(analogRead(pin)) * board::kAdcRefV / full;
  }

  Encoder enc_[4];
};

Teensy41Hal hal;
orion_fw::ControlCore core(hal, ORION_FW_GIT_HASH);

}  // namespace

void setup()
{
  hal.begin();
  core.begin();
}

void loop()
{
  Stream & link = host_link();
  int budget = 256;  // bound the time spent parsing before the next poll()
  while (budget-- > 0 && link.available() > 0) {
    core.on_rx_byte(static_cast<uint8_t>(link.read()));
  }
  core.poll();
}
