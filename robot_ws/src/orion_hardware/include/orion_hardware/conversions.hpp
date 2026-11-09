// Pure, ROS-free unit conversions and configuration parsing for OrionSystem (unit-tested).
#ifndef ORION_HARDWARE__CONVERSIONS_HPP_
#define ORION_HARDWARE__CONVERSIONS_HPP_

#include <array>
#include <chrono>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "orion_proto.hpp"

namespace orion_hardware
{

/// Hardware-level parameters from the URDF <hardware><param> block.
struct DriveConfig
{
  std::string port;                    // serial device, e.g. /dev/ttyACM0
  int baud = 921600;                   // ignored by USB CDC devices
  double encoder_cpr = 0.0;            // decoder counts per motor revolution (after x4)
  double gear_ratio = 0.0;             // motor revolutions per wheel revolution
  double kt = 0.0;                     // N m/A at the motor shaft; 0 = no back-EMF correction
  double motor_resistance = 0.0;       // ohm, terminal resistance (effort estimate fallback)
  double nominal_battery_v = 12.0;     // V, used only when the board does not measure vbat
  double state_timeout_s = 0.1;        // STATE older than this -> read() returns ERROR
  double health_rate_hz = 10.0;        // orion_msgs/RoverHealth publish rate
  bool brake_on_idle = false;          // short the motors when idle instead of coasting
};

/// Per-joint parameters from <joint><param>.
struct JointConfig
{
  std::string name;
  size_t motor_index = 0;  // firmware wheel slot: 0 FL, 1 FR, 2 RL, 3 RR
  double direction = 1.0;  // +1, or -1 if firmware-positive is URDF-negative for this joint
};

/// Parses and validates hardware parameters. Throws std::invalid_argument with a message
/// naming the bad parameter.
DriveConfig parse_drive_config(const std::map<std::string, std::string> & params);

/// Parses per-joint params. `joint_params[i]` belongs to `names[i]`; motor_index defaults to
/// the joint's position in the URDF. Throws std::invalid_argument unless the four joints map to
/// four distinct motors.
std::vector<JointConfig> parse_joint_configs(
  const std::vector<std::string> & names,
  const std::vector<std::map<std::string, std::string>> & joint_params);

/// rad/s -> mrad/s, rounded and saturated to int16. NaN/inf (an unset command) map to 0.
int16_t rad_s_to_mrad_s(double rad_s);

/// Effort (A) estimated without a current sensor (SPEC F5):
///   I = (duty * V_bat - kt * gear_ratio * w_wheel) / R
/// With kt = 0 this is the SPEC's duty * V_bat / R. `duty` in [-1, 1], `w_wheel` in rad/s.
double estimate_current_a(
  double duty, double vbat_v, double w_wheel_rad_s, double kt, double gear_ratio,
  double motor_resistance);

/// One joint's state in URDF units.
struct JointReading
{
  double position_rad = 0.0;
  double velocity_rad_s = 0.0;
  double effort_a = 0.0;
  bool effort_measured = false;  // false: estimated from duty
};

/// Converts the firmware STATE for one joint into URDF units and sign convention.
JointReading to_joint_reading(
  const orion_proto::StateMsg & state, const JointConfig & joint, const DriveConfig & drive);

/// True if a STATE received at `rx_time` is still usable at `now` (age <= max_age).
inline bool is_fresh(
  std::chrono::steady_clock::time_point now, std::chrono::steady_clock::time_point rx_time,
  std::chrono::steady_clock::duration max_age)
{
  return now - rx_time <= max_age;
}

/// Extends the firmware's wrapping 32-bit microsecond clock to 64 bits.
class MicrosUnwrapper
{
public:
  uint64_t update(uint32_t t_us)
  {
    if (has_last_ && t_us < last_) {
      high_ += uint64_t{1} << 32;
    }
    has_last_ = true;
    last_ = t_us;
    return high_ + t_us;
  }

private:
  bool has_last_ = false;
  uint32_t last_ = 0;
  uint64_t high_ = 0;
};

}  // namespace orion_hardware

#endif  // ORION_HARDWARE__CONVERSIONS_HPP_
