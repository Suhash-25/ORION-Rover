// Teensy 4.1 example board configuration.
//
// !!! UNCONFIRMED PIN MAPPING !!! The ORION hardware list is not known yet (docs/HARDWARE.md).
// Every pin below is a placeholder chosen to be electrically plausible on a Teensy 4.1 with a
// PWM+DIR motor driver (e.g. Cytron MDD10A class). Confirm each one against the real wiring
// before powering motors. The firmware refuses nothing here; wrong pins mean wrong motors.
#ifndef ORION_BOARD_CONFIG_HPP_
#define ORION_BOARD_CONFIG_HPP_

#include <cstdint>

namespace board
{

struct MotorPins
{
  uint8_t pwm;          // PWM-capable pin
  uint8_t dir;          // direction (HIGH = forward before inversion)
  bool invert_motor;    // flip so positive duty turns the wheel URDF-positive
  uint8_t enc_a;        // quadrature A
  uint8_t enc_b;        // quadrature B
  bool invert_encoder;  // flip so URDF-positive rotation counts up
  int8_t current_pin;   // analog pin of a current-sense output, -1 = not fitted
};

// Order is the protocol order: FL, FR, RL, RR. Right-side motors are usually mirrored, hence
// the GUESSED invert flags; verify with the bring-up checklist in docs/HARDWARE.md.
constexpr MotorPins kMotors[4] = {
  {2, 3, false, 30, 31, false, -1},   // front left   CONFIRM
  {4, 5, true, 32, 33, true, -1},     // front right  CONFIRM
  {6, 7, false, 34, 35, false, -1},   // rear left    CONFIRM
  {8, 9, true, 36, 37, true, -1},     // rear right   CONFIRM
};

// Current sense scaling (only used if current_pin >= 0): amps = (volts - offset) * gain.
// Example: ACS712-20A on a 3.3 V divider. CONFIRM with the real sensor.
constexpr float kCurrentOffsetV = 1.65f;
constexpr float kCurrentAmpsPerV = 10.0f;

// Battery voltage divider (-1 = not fitted). Volts at the pin * ratio = battery volts.
constexpr int8_t kVbatPin = 41;            // A17, CONFIRM
constexpr float kVbatDividerRatio = 11.0f; // e.g. 100k / 10k, CONFIRM

constexpr uint32_t kPwmFrequencyHz = 20000;  // inaudible, within MDD10A-class limits
constexpr uint8_t kPwmResolutionBits = 12;
constexpr uint8_t kAdcResolutionBits = 12;
constexpr float kAdcRefV = 3.3f;

// Host link. On Teensy, `Serial` is native USB (baud ignored, ~12 Mbit/s). Set kUseUart to
// talk to an onboard computer over a hardware UART instead.
constexpr bool kUseUart = false;
constexpr uint32_t kUartBaud = 921600;

}  // namespace board

#endif  // ORION_BOARD_CONFIG_HPP_
