// OrionSystem lifecycle, read()/write() and e-stop forwarding against a mock transport that
// plays firmware. The real controller_manager path is covered by test_integration.py.
#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "hardware_interface/component_parser.hpp"
#include "hardware_interface/types/hardware_component_params.hpp"
#include "mock_transport.hpp"
#include "orion_hardware/orion_system.hpp"
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/bool.hpp"

using orion_hardware::OrionSystem;
using orion_hardware_test::MockTransport;
using hardware_interface::return_type;
using CallbackReturn = OrionSystem::CallbackReturn;
namespace op = orion_proto;
using namespace std::chrono_literals;

namespace
{

std::string urdf(const std::string & extra_hw_params = "", bool with_cpr = true)
{
  std::string joints;
  // FR is mirrored (direction -1); the rear wheels are swapped on the firmware side.
  const char * names[] = {"fl", "fr", "rl", "rr"};
  const char * index[] = {"0", "1", "3", "2"};
  const char * dir[] = {"1", "-1", "1", "1"};
  std::string robot_joints;
  for (int i = 0; i < 4; ++i) {
    robot_joints += std::string("<link name=\"") + names[i] + "\"/>" +
      "<joint name=\"" + names[i] + "_joint\" type=\"continuous\"><parent link=\"base\"/>" +
      "<child link=\"" + names[i] + "\"/><axis xyz=\"0 1 0\"/></joint>";
    joints += std::string("<joint name=\"") + names[i] + "_joint\">" +
      "<param name=\"motor_index\">" + index[i] + "</param>" +
      "<param name=\"direction\">" + dir[i] + "</param>" +
      "<command_interface name=\"velocity\"/><state_interface name=\"position\"/>" +
      "<state_interface name=\"velocity\"/><state_interface name=\"effort\"/></joint>";
  }
  return std::string("<?xml version=\"1.0\"?><robot name=\"t\"><link name=\"base\"/>") +
         robot_joints + "<ros2_control name=\"orion_system\" type=\"system\"><hardware>" +
         "<plugin>orion_hardware/OrionSystem</plugin><param name=\"port\">mock</param>" +
         (with_cpr ? "<param name=\"encoder_cpr\">2048</param>" : "") +
         "<param name=\"gear_ratio\">30</param><param name=\"motor_resistance\">0.5</param>" +
         extra_hw_params + "</hardware>" + joints + "</ros2_control></robot>";
}

std::vector<op::Frame> frames_from_host(MockTransport::Shared & s)
{
  std::lock_guard<std::mutex> lock(s.m);
  op::FrameDecoder d;
  std::vector<op::Frame> out;
  for (uint8_t b : s.from_host) {
    if (d.push(b)) {
      out.push_back(d.frame());
    }
  }
  return out;
}

void play_firmware(const op::Frame & f, MockTransport::Shared & s)
{
  if (f.type == op::MSG_PING) {
    s.push(op::PongMsg{});
  } else if (f.type == op::MSG_SET_PARAM) {
    op::SetParamMsg req;
    if (op::decode_msg(f, req)) {
      op::ParamAckMsg ack;
      ack.id = req.id;
      ack.value = req.value;
      s.push(ack);
    }
  }
}

}  // namespace

class OrionSystemTest : public ::testing::Test
{
protected:
  static void SetUpTestSuite() {rclcpp::init(0, nullptr);}
  static void TearDownTestSuite() {rclcpp::shutdown();}

  void SetUp() override
  {
    shared = std::make_shared<MockTransport::Shared>();
    shared->respond = play_firmware;
    shared->stream_state = true;
    for (int i = 0; i < 4; ++i) {
      shared->state.position_mrad[i] = 1000 * (i + 1);
      shared->state.velocity_mrad_s[i] = 100 * (i + 1);
      shared->state.current_ma[i] = 500;
    }
    executor = std::make_shared<rclcpp::executors::SingleThreadedExecutor>();
  }

  CallbackReturn init(const std::string & description)
  {
    auto sh = shared;
    system = std::make_unique<OrionSystem>(
      [sh](const orion_hardware::DriveConfig &) {return std::make_unique<MockTransport>(sh);});
    hardware_interface::HardwareComponentParams p;
    p.hardware_info = hardware_interface::parse_control_resources_from_urdf(description).at(0);
    p.clock = std::make_shared<rclcpp::Clock>(RCL_STEADY_TIME);
    p.executor = executor;
    const CallbackReturn r = system->init(p);
    if (r == CallbackReturn::SUCCESS) {
      states = system->on_export_state_interfaces();
      commands = system->on_export_command_interfaces();
    }
    return r;
  }

  double state(const std::string & name)
  {
    for (const auto & s : states) {
      if (s->get_name() == name) {
        return s->get_optional<double>().value();
      }
    }
    ADD_FAILURE() << "no state interface " << name;
    return NAN;
  }

  void command(const std::string & joint, double value)
  {
    for (const auto & c : commands) {
      if (c->get_name() == joint + "/velocity") {
        ASSERT_TRUE(c->set_value(value));
        return;
      }
    }
    ADD_FAILURE() << "no command interface for " << joint;
  }

  return_type read() {return system->read(rclcpp::Time(0), rclcpp::Duration(10ms));}
  return_type write() {return system->write(rclcpp::Time(0), rclcpp::Duration(10ms));}

  std::shared_ptr<MockTransport::Shared> shared;
  std::shared_ptr<rclcpp::executors::SingleThreadedExecutor> executor;
  std::unique_ptr<OrionSystem> system;
  std::vector<hardware_interface::StateInterface::ConstSharedPtr> states;
  std::vector<hardware_interface::CommandInterface::SharedPtr> commands;
  rclcpp_lifecycle::State lifecycle_state;
};

TEST_F(OrionSystemTest, InitRejectsMissingRequiredParameter)
{
  EXPECT_EQ(init(urdf("", false)), CallbackReturn::ERROR);
}

TEST_F(OrionSystemTest, ConfigurePushesDrivetrainParamsAndReadsState)
{
  ASSERT_EQ(init(urdf()), CallbackReturn::SUCCESS);
  ASSERT_EQ(system->on_configure(lifecycle_state), CallbackReturn::SUCCESS);
  bool cpr = false;
  bool gear = false;
  for (const auto & f : frames_from_host(*shared)) {
    op::SetParamMsg p;
    if (op::decode_msg(f, p)) {
      cpr = cpr || (p.id == op::param_id::kEncoderCpr && p.value == 2048.0f);
      gear = gear || (p.id == op::param_id::kGearRatio && p.value == 30.0f);
    }
  }
  EXPECT_TRUE(cpr);
  EXPECT_TRUE(gear);

  ASSERT_EQ(read(), return_type::OK);
  EXPECT_DOUBLE_EQ(state("fl_joint/position"), 1.0);
  EXPECT_DOUBLE_EQ(state("fr_joint/position"), -2.0);  // direction -1
  EXPECT_DOUBLE_EQ(state("rl_joint/position"), 4.0);   // motor_index 3
  EXPECT_DOUBLE_EQ(state("rr_joint/position"), 3.0);   // motor_index 2
  EXPECT_DOUBLE_EQ(state("fr_joint/velocity"), -0.2);
  EXPECT_DOUBLE_EQ(state("fl_joint/effort"), 0.5);
}

TEST_F(OrionSystemTest, ReadReturnsErrorWhenStateIsOlderThan100ms)
{
  ASSERT_EQ(init(urdf()), CallbackReturn::SUCCESS);
  ASSERT_EQ(system->on_configure(lifecycle_state), CallbackReturn::SUCCESS);
  ASSERT_EQ(read(), return_type::OK);
  shared->stream_state = false;  // firmware goes silent
  std::this_thread::sleep_for(60ms);
  EXPECT_EQ(read(), return_type::OK) << "60 ms is within the 100 ms limit";
  std::this_thread::sleep_for(80ms);
  EXPECT_EQ(read(), return_type::ERROR);
  shared->stream_state = true;
  std::this_thread::sleep_for(50ms);
  EXPECT_EQ(read(), return_type::OK);
}

TEST_F(OrionSystemTest, ConfigureFailsWhenFirmwareIsSilentOrIncompatible)
{
  shared->respond = nullptr;
  ASSERT_EQ(init(urdf()), CallbackReturn::SUCCESS);
  EXPECT_EQ(system->on_configure(lifecycle_state), CallbackReturn::FAILURE);

  shared->respond = [](const op::Frame & f, MockTransport::Shared & s) {
    if (f.type == op::MSG_PING) {
      op::PongMsg pong;
      pong.proto_ver = op::kProtocolVersion + 1;
      s.push(pong);
    }
  };
  EXPECT_EQ(system->on_configure(lifecycle_state), CallbackReturn::FAILURE);
}

TEST_F(OrionSystemTest, WriteSendsMappedTargetsOnlyWhileActive)
{
  ASSERT_EQ(init(urdf("<param name=\"brake_on_idle\">true</param>")), CallbackReturn::SUCCESS);
  ASSERT_EQ(system->on_configure(lifecycle_state), CallbackReturn::SUCCESS);
  const size_t before = frames_from_host(*shared).size();
  ASSERT_EQ(write(), return_type::OK);
  EXPECT_EQ(frames_from_host(*shared).size(), before) << "no CMD while inactive";

  ASSERT_EQ(system->on_activate(lifecycle_state), CallbackReturn::SUCCESS);
  command("fl_joint", 1.0);
  command("fr_joint", 2.0);
  command("rl_joint", NAN);  // unset command must not move the wheel
  command("rr_joint", -0.5);
  ASSERT_EQ(write(), return_type::OK);
  auto frames = frames_from_host(*shared);
  op::CmdMsg cmd;
  ASSERT_TRUE(op::decode_msg(frames.back(), cmd));
  EXPECT_EQ(cmd.w_target_mrad_s[0], 1000);
  EXPECT_EQ(cmd.w_target_mrad_s[1], -2000);  // direction -1
  EXPECT_EQ(cmd.w_target_mrad_s[3], 0);      // rl -> motor 3, NaN -> 0
  EXPECT_EQ(cmd.w_target_mrad_s[2], -500);   // rr -> motor 2
  EXPECT_EQ(cmd.flags, op::cmd_flag::kEnable | op::cmd_flag::kBrakeOnIdle);

  ASSERT_EQ(system->on_deactivate(lifecycle_state), CallbackReturn::SUCCESS);
  frames = frames_from_host(*shared);
  ASSERT_TRUE(op::decode_msg(frames.back(), cmd));
  EXPECT_EQ(cmd.flags & op::cmd_flag::kEnable, 0) << "deactivate must disable the firmware";
  EXPECT_EQ(cmd.w_target_mrad_s[0], 0);
}

TEST_F(OrionSystemTest, EstopTopicForwardsEstopAndClear)
{
  ASSERT_EQ(init(urdf()), CallbackReturn::SUCCESS);
  ASSERT_EQ(system->on_configure(lifecycle_state), CallbackReturn::SUCCESS);
  ASSERT_EQ(system->on_activate(lifecycle_state), CallbackReturn::SUCCESS);
  auto node = rclcpp::Node::make_shared("estop_test_publisher");
  auto pub = node->create_publisher<std_msgs::msg::Bool>(OrionSystem::kEstopTopic, 10);

  auto count = [&](uint8_t type) {
    size_t n = 0;
    for (const auto & f : frames_from_host(*shared)) {
      n += f.type == type ? 1 : 0;
    }
    return n;
  };
  auto publish_until = [&](bool value, uint8_t type) {
    for (int i = 0; i < 200 && count(type) == 0; ++i) {
      std_msgs::msg::Bool m;
      m.data = value;
      pub->publish(m);
      executor->spin_some(10ms);
    }
    return count(type) > 0;
  };
  ASSERT_TRUE(publish_until(true, op::MSG_ESTOP)) << "ESTOP never reached the firmware";
  const size_t cmds = count(op::MSG_CMD);
  const size_t estops = count(op::MSG_ESTOP);
  ASSERT_EQ(write(), return_type::OK);
  EXPECT_EQ(count(op::MSG_CMD), cmds) << "no wheel commands while e-stopped";
  EXPECT_EQ(count(op::MSG_ESTOP), estops + 1) << "ESTOP repeated every cycle";

  ASSERT_TRUE(publish_until(false, op::MSG_ESTOP_CLEAR));
  ASSERT_EQ(write(), return_type::OK);
  EXPECT_EQ(count(op::MSG_CMD), cmds + 1);
}
