#include "orion_hardware/conversions.hpp"

#include <cmath>
#include <set>
#include <stdexcept>

namespace orion_hardware
{

namespace
{

const std::string * find(const std::map<std::string, std::string> & m, const std::string & key)
{
  auto it = m.find(key);
  return it == m.end() ? nullptr : &it->second;
}

double to_double(const std::string & key, const std::string & text)
{
  try {
    size_t used = 0;
    const double v = std::stod(text, &used);
    if (used != text.size() || !std::isfinite(v)) {
      throw std::invalid_argument("");
    }
    return v;
  } catch (const std::exception &) {
    throw std::invalid_argument("parameter '" + key + "': not a number: '" + text + "'");
  }
}

double get_double(
  const std::map<std::string, std::string> & m, const std::string & key, double fallback,
  double min_value, double max_value)
{
  const std::string * text = find(m, key);
  const double v = text ? to_double(key, *text) : fallback;
  if (v < min_value || v > max_value) {
    throw std::invalid_argument(
            "parameter '" + key + "' = " + std::to_string(v) + " outside [" +
            std::to_string(min_value) + ", " + std::to_string(max_value) + "]");
  }
  return v;
}

double require_double(
  const std::map<std::string, std::string> & m, const std::string & key, double min_value,
  double max_value)
{
  if (!find(m, key)) {
    throw std::invalid_argument("missing required parameter '" + key + "'");
  }
  return get_double(m, key, 0.0, min_value, max_value);
}

bool get_bool(const std::map<std::string, std::string> & m, const std::string & key, bool fb)
{
  const std::string * text = find(m, key);
  if (!text) {
    return fb;
  }
  if (*text == "true" || *text == "True" || *text == "1") {
    return true;
  }
  if (*text == "false" || *text == "False" || *text == "0") {
    return false;
  }
  throw std::invalid_argument("parameter '" + key + "': not a bool: '" + *text + "'");
}

}  // namespace

DriveConfig parse_drive_config(const std::map<std::string, std::string> & params)
{
  DriveConfig c;
  const std::string * port = find(params, "port");
  if (!port || port->empty()) {
    throw std::invalid_argument("missing required parameter 'port'");
  }
  c.port = *port;
  c.baud = static_cast<int>(get_double(params, "baud", 921600, 1200, 4000000));
  c.encoder_cpr = require_double(params, "encoder_cpr", 1.0, 1.0e6);
  c.gear_ratio = require_double(params, "gear_ratio", 0.01, 1000.0);
  c.kt = get_double(params, "kt", 0.0, 0.0, 10.0);
  c.motor_resistance = require_double(params, "motor_resistance", 1.0e-3, 100.0);
  c.nominal_battery_v = get_double(params, "nominal_battery_v", 12.0, 1.0, 60.0);
  c.state_timeout_s = get_double(params, "state_timeout_s", 0.1, 0.02, 1.0);
  c.health_rate_hz = get_double(params, "health_rate_hz", 10.0, 0.1, 100.0);
  c.brake_on_idle = get_bool(params, "brake_on_idle", false);
  return c;
}

std::vector<JointConfig> parse_joint_configs(
  const std::vector<std::string> & names,
  const std::vector<std::map<std::string, std::string>> & joint_params)
{
  if (names.size() != orion_proto::kNumWheels || joint_params.size() != names.size()) {
    throw std::invalid_argument(
            "expected exactly " + std::to_string(orion_proto::kNumWheels) + " wheel joints, got " +
            std::to_string(names.size()));
  }
  std::vector<JointConfig> joints;
  std::set<size_t> used;
  for (size_t i = 0; i < names.size(); ++i) {
    JointConfig j;
    j.name = names[i];
    const auto & p = joint_params[i];
    const double index = get_double(
      p, "motor_index", static_cast<double>(i), 0.0, orion_proto::kNumWheels - 1.0);
    if (index != std::floor(index)) {
      throw std::invalid_argument(names[i] + ": motor_index must be an integer 0..3");
    }
    j.motor_index = static_cast<size_t>(index);
    j.direction = get_double(p, "direction", 1.0, -1.0, 1.0);
    if (j.direction != 1.0 && j.direction != -1.0) {
      throw std::invalid_argument(names[i] + ": direction must be 1 or -1");
    }
    if (!used.insert(j.motor_index).second) {
      throw std::invalid_argument(names[i] + ": motor_index used twice");
    }
    joints.push_back(j);
  }
  return joints;
}

int16_t rad_s_to_mrad_s(double rad_s)
{
  if (!std::isfinite(rad_s)) {
    return 0;
  }
  const double mrad = std::round(rad_s * 1000.0);
  if (mrad > 32767.0) {
    return 32767;
  }
  if (mrad < -32768.0) {
    return -32768;
  }
  return static_cast<int16_t>(mrad);
}

double estimate_current_a(
  double duty, double vbat_v, double w_wheel_rad_s, double kt, double gear_ratio,
  double motor_resistance)
{
  const double back_emf_v = kt * gear_ratio * w_wheel_rad_s;  // ke = kt in SI units
  return (duty * vbat_v - back_emf_v) / motor_resistance;
}

JointReading to_joint_reading(
  const orion_proto::StateMsg & s, const JointConfig & joint, const DriveConfig & drive)
{
  const size_t m = joint.motor_index;
  const double d = joint.direction;
  JointReading r;
  r.position_rad = d * s.position_mrad[m] * 1.0e-3;
  const double w_fw = s.velocity_mrad_s[m] * 1.0e-3;
  r.velocity_rad_s = d * w_fw;
  if (s.current_ma[m] != orion_proto::kCurrentNotAvailable) {
    r.effort_a = d * s.current_ma[m] * 1.0e-3;
    r.effort_measured = true;
  } else {
    const double vbat = (s.status & orion_proto::status::kVbatSensed) ?
      s.vbat_mv * 1.0e-3 : drive.nominal_battery_v;
    const double duty = static_cast<double>(s.duty[m]) / orion_proto::kDutyFullScale;
    r.effort_a = d * estimate_current_a(
      duty, vbat, w_fw, drive.kt, drive.gear_ratio, drive.motor_resistance);
    r.effort_measured = false;
  }
  return r;
}

}  // namespace orion_hardware
