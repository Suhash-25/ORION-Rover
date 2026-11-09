// Unit conversions, effort fallback, staleness and configuration parsing.
#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "orion_hardware/conversions.hpp"

using orion_hardware::DriveConfig;
using orion_hardware::JointConfig;
namespace op = orion_proto;

namespace
{

std::map<std::string, std::string> valid_params()
{
  return {{"port", "/dev/ttyACM0"}, {"encoder_cpr", "2048"}, {"gear_ratio", "30"},
    {"motor_resistance", "0.5"}};
}

DriveConfig drive()
{
  return orion_hardware::parse_drive_config(valid_params());
}

}  // namespace

TEST(Conversions, RadPerSecondToMilliradSaturatesAndRejectsNaN)
{
  EXPECT_EQ(orion_hardware::rad_s_to_mrad_s(2.0), 2000);
  EXPECT_EQ(orion_hardware::rad_s_to_mrad_s(-0.0004), 0);
  EXPECT_EQ(orion_hardware::rad_s_to_mrad_s(0.0016), 2);
  EXPECT_EQ(orion_hardware::rad_s_to_mrad_s(-1.2345), -1235);  // rounds half away from zero
  EXPECT_EQ(orion_hardware::rad_s_to_mrad_s(1.0e6), 32767);
  EXPECT_EQ(orion_hardware::rad_s_to_mrad_s(-1.0e6), -32768);
  EXPECT_EQ(orion_hardware::rad_s_to_mrad_s(std::numeric_limits<double>::quiet_NaN()), 0);
  EXPECT_EQ(orion_hardware::rad_s_to_mrad_s(std::numeric_limits<double>::infinity()), 0);
}

TEST(Conversions, MeasuredStateToJointUnits)
{
  op::StateMsg s;
  s.position_mrad[2] = -12345;
  s.velocity_mrad_s[2] = 1500;
  s.current_ma[2] = -2500;
  JointConfig j{"rear_left_wheel_joint", 2, 1.0};
  auto r = orion_hardware::to_joint_reading(s, j, drive());
  EXPECT_DOUBLE_EQ(r.position_rad, -12.345);
  EXPECT_DOUBLE_EQ(r.velocity_rad_s, 1.5);
  EXPECT_DOUBLE_EQ(r.effort_a, -2.5);
  EXPECT_TRUE(r.effort_measured);

  j.direction = -1.0;  // mirrored motor: every quantity flips
  r = orion_hardware::to_joint_reading(s, j, drive());
  EXPECT_DOUBLE_EQ(r.position_rad, 12.345);
  EXPECT_DOUBLE_EQ(r.velocity_rad_s, -1.5);
  EXPECT_DOUBLE_EQ(r.effort_a, 2.5);
}

TEST(Conversions, EffortFallsBackToDutyTimesVbatOverR)
{
  op::StateMsg s;
  s.current_ma[0] = op::kCurrentNotAvailable;
  s.duty[0] = 5000;  // 50 %
  s.velocity_mrad_s[0] = 3000;
  s.vbat_mv = 11000;
  s.status = op::status::kVbatSensed;
  const JointConfig j{"front_left_wheel_joint", 0, 1.0};
  DriveConfig d = drive();  // kt = 0: exactly SPEC F5, duty * V_bat / R
  auto r = orion_hardware::to_joint_reading(s, j, d);
  EXPECT_FALSE(r.effort_measured);
  EXPECT_NEAR(r.effort_a, 0.5 * 11.0 / 0.5, 1e-9);

  s.status = 0;  // no battery measurement: nominal voltage
  r = orion_hardware::to_joint_reading(s, j, d);
  EXPECT_NEAR(r.effort_a, 0.5 * 12.0 / 0.5, 1e-9);

  d.kt = 0.02;  // with a torque constant, subtract back-EMF kt * gear * w
  s.status = op::status::kVbatSensed;
  r = orion_hardware::to_joint_reading(s, j, d);
  EXPECT_NEAR(r.effort_a, (0.5 * 11.0 - 0.02 * 30.0 * 3.0) / 0.5, 1e-9);
}

TEST(Conversions, StalenessBoundaryIsInclusive)
{
  using namespace std::chrono_literals;
  const auto t0 = std::chrono::steady_clock::time_point{} + 10s;
  EXPECT_TRUE(orion_hardware::is_fresh(t0 + 100ms, t0, 100ms));
  EXPECT_FALSE(orion_hardware::is_fresh(t0 + 101ms, t0, 100ms));
  EXPECT_TRUE(orion_hardware::is_fresh(t0, t0, 100ms));
}

TEST(Conversions, MicrosUnwrapperExtendsAcrossWrap)
{
  orion_hardware::MicrosUnwrapper u;
  EXPECT_EQ(u.update(0xFFFFFF00u), 0xFFFFFF00ull);
  EXPECT_EQ(u.update(0x00000010u), 0x100000010ull);
  EXPECT_EQ(u.update(0x00000020u), 0x100000020ull);
}

TEST(Config, ParsesDefaultsAndRequiredValues)
{
  const DriveConfig d = drive();
  EXPECT_EQ(d.port, "/dev/ttyACM0");
  EXPECT_EQ(d.baud, 921600);
  EXPECT_DOUBLE_EQ(d.encoder_cpr, 2048.0);
  EXPECT_DOUBLE_EQ(d.gear_ratio, 30.0);
  EXPECT_DOUBLE_EQ(d.kt, 0.0);
  EXPECT_DOUBLE_EQ(d.state_timeout_s, 0.1);
  EXPECT_FALSE(d.brake_on_idle);
}

TEST(Config, RejectsMissingOrInvalidValues)
{
  for (const char * key : {"port", "encoder_cpr", "gear_ratio", "motor_resistance"}) {
    auto p = valid_params();
    p.erase(key);
    EXPECT_THROW(orion_hardware::parse_drive_config(p), std::invalid_argument) << key;
  }
  auto p = valid_params();
  p["encoder_cpr"] = "12abc";
  EXPECT_THROW(orion_hardware::parse_drive_config(p), std::invalid_argument);
  p = valid_params();
  p["motor_resistance"] = "0";
  EXPECT_THROW(orion_hardware::parse_drive_config(p), std::invalid_argument);
  p = valid_params();
  p["brake_on_idle"] = "maybe";
  EXPECT_THROW(orion_hardware::parse_drive_config(p), std::invalid_argument);
}

TEST(Config, JointMappingDefaultsToUrdfOrderAndMustBeAPermutation)
{
  const std::vector<std::string> names{"fl", "fr", "rl", "rr"};
  std::vector<std::map<std::string, std::string>> params(4);
  auto joints = orion_hardware::parse_joint_configs(names, params);
  for (size_t i = 0; i < 4; ++i) {
    EXPECT_EQ(joints[i].motor_index, i);
    EXPECT_EQ(joints[i].direction, 1.0);
  }
  params[1]["direction"] = "-1";
  params[0]["motor_index"] = "3";
  params[3]["motor_index"] = "0";
  joints = orion_hardware::parse_joint_configs(names, params);
  EXPECT_EQ(joints[0].motor_index, 3u);
  EXPECT_EQ(joints[3].motor_index, 0u);
  EXPECT_EQ(joints[1].direction, -1.0);

  params[2]["motor_index"] = "0";  // duplicate
  EXPECT_THROW(orion_hardware::parse_joint_configs(names, params), std::invalid_argument);
  params[2]["motor_index"] = "1.5";
  EXPECT_THROW(orion_hardware::parse_joint_configs(names, params), std::invalid_argument);
  params[2].erase("motor_index");
  params[2]["direction"] = "0.5";
  EXPECT_THROW(orion_hardware::parse_joint_configs(names, params), std::invalid_argument);
  EXPECT_THROW(
    orion_hardware::parse_joint_configs({"a", "b", "c"}, {{}, {}, {}}), std::invalid_argument);
}
