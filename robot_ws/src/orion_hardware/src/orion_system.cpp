#include "orion_hardware/orion_system.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "hardware_interface/types/hardware_interface_type_values.hpp"
#include "pluginlib/class_list_macros.hpp"

namespace orion_hardware
{

namespace op = orion_proto;
using hardware_interface::return_type;
using namespace std::chrono_literals;

namespace
{

std::map<std::string, std::string> to_map(const std::unordered_map<std::string, std::string> & m)
{
  return {m.begin(), m.end()};
}

bool has_interface(
  const std::vector<hardware_interface::InterfaceInfo> & list, const std::string & name)
{
  for (const auto & i : list) {
    if (i.name == name) {
      return true;
    }
  }
  return false;
}

}  // namespace

OrionSystem::OrionSystem()
: OrionSystem([](const DriveConfig & c) {
      return std::make_unique<SerialTransport>(c.port, c.baud);
    }) {}

OrionSystem::OrionSystem(TransportFactory factory)
: factory_(std::move(factory)) {}

OrionSystem::~OrionSystem()
{
  release_link();
}

OrionSystem::CallbackReturn OrionSystem::on_init(
  const hardware_interface::HardwareComponentInterfaceParams & params)
{
  if (hardware_interface::SystemInterface::on_init(params) != CallbackReturn::SUCCESS) {
    return CallbackReturn::ERROR;
  }
  try {
    drive_ = parse_drive_config(to_map(info_.hardware_parameters));
    std::vector<std::string> names;
    std::vector<std::map<std::string, std::string>> joint_params;
    for (const auto & j : info_.joints) {
      names.push_back(j.name);
      joint_params.push_back(to_map(j.parameters));
    }
    joints_ = parse_joint_configs(names, joint_params);
  } catch (const std::invalid_argument & e) {
    RCLCPP_ERROR(get_logger(), "Invalid OrionSystem configuration: %s", e.what());
    return CallbackReturn::ERROR;
  }

  namespace hi = hardware_interface;
  names_.clear();
  for (const auto & j : info_.joints) {
    if (j.command_interfaces.size() != 1 || j.command_interfaces[0].name != hi::HW_IF_VELOCITY) {
      RCLCPP_ERROR(
        get_logger(), "Joint '%s' needs exactly one 'velocity' command interface",
        j.name.c_str());
      return CallbackReturn::ERROR;
    }
    if (!has_interface(j.state_interfaces, hi::HW_IF_POSITION) ||
      !has_interface(j.state_interfaces, hi::HW_IF_VELOCITY))
    {
      RCLCPP_ERROR(
        get_logger(), "Joint '%s' needs 'position' and 'velocity' state interfaces",
        j.name.c_str());
      return CallbackReturn::ERROR;
    }
    JointInterfaces n;
    n.position = j.name + "/" + hi::HW_IF_POSITION;
    n.velocity = j.name + "/" + hi::HW_IF_VELOCITY;
    n.effort = has_interface(j.state_interfaces, hi::HW_IF_EFFORT) ?
      j.name + "/" + hi::HW_IF_EFFORT : "";
    n.command = j.name + "/" + hi::HW_IF_VELOCITY;
    names_.push_back(n);
  }
  RCLCPP_INFO(
    get_logger(),
    "OrionSystem: port %s @ %d, encoder_cpr %.0f, gear_ratio %.3f, R %.3f ohm, kt %.4f N m/A",
    drive_.port.c_str(), drive_.baud, drive_.encoder_cpr, drive_.gear_ratio,
    drive_.motor_resistance, drive_.kt);
  return CallbackReturn::SUCCESS;
}

std::shared_ptr<FirmwareLink> OrionSystem::link() const
{
  std::lock_guard<std::mutex> lock(link_mutex_);
  return link_;
}

bool OrionSystem::handshake()
{
  auto l = link();
  std::optional<op::PongMsg> pong;
  for (int attempt = 0; attempt < 5 && !pong; ++attempt) {
    pong = l->ping(200ms);
  }
  if (!pong) {
    RCLCPP_ERROR(
      get_logger(), "No PONG from firmware on %s (is it powered and flashed?)",
      drive_.port.c_str());
    return false;
  }
  if (pong->proto_ver != op::kProtocolVersion) {
    RCLCPP_ERROR(
      get_logger(), "Firmware speaks protocol v%u, host v%u: reflash the firmware",
      pong->proto_ver, op::kProtocolVersion);
    return false;
  }
  RCLCPP_INFO(
    get_logger(), "Firmware %08x, protocol v%u, param crc %04x",
    pong->fw_git_hash, pong->proto_ver, pong->param_crc);
  {
    char hash[9];
    std::snprintf(hash, sizeof(hash), "%08x", static_cast<unsigned>(pong->fw_git_hash));
    std::lock_guard<std::mutex> lock(link_mutex_);
    fw_git_hash_ = hash;
  }

  const std::pair<uint8_t, double> push[] = {
    {op::param_id::kEncoderCpr, drive_.encoder_cpr},
    {op::param_id::kGearRatio, drive_.gear_ratio},
  };
  for (const auto & [id, value] : push) {
    const auto ack = l->set_param(id, static_cast<float>(value), 300ms);
    if (!ack || ack->status != op::param_status::kOk || ack->value != static_cast<float>(value)) {
      const std::string why = !ack ? "no PARAM_ACK" :
        ack->status != op::param_status::kOk ? "status " + std::to_string(ack->status) :
        "firmware kept " + std::to_string(ack->value);
      RCLCPP_ERROR(
        get_logger(), "Firmware rejected parameter %u = %f (%s)", id, value,
        why.c_str());
      return false;
    }
  }
  if (!l->wait_for_state(500ms)) {
    RCLCPP_ERROR(get_logger(), "Firmware answers PING but sends no STATE");
    return false;
  }
  return true;
}

OrionSystem::CallbackReturn OrionSystem::on_configure(const rclcpp_lifecycle::State &)
{
  release_link();
  auto logger = get_logger();
  auto new_link = std::make_shared<FirmwareLink>(
    factory_(drive_), FirmwareLink::Clock::now,
    [logger](const std::string & text) {RCLCPP_INFO(logger, "[firmware] %s", text.c_str());});
  std::string error;
  if (!new_link->start(error)) {
    RCLCPP_ERROR(get_logger(), "Cannot open firmware link: %s", error.c_str());
    return CallbackReturn::FAILURE;
  }
  {
    std::lock_guard<std::mutex> lock(link_mutex_);
    link_ = new_link;
  }
  if (!handshake()) {
    release_link();
    return CallbackReturn::FAILURE;
  }

  auto node = get_node();
  if (!node) {
    RCLCPP_ERROR(get_logger(), "No hardware component node; cannot publish RoverHealth");
    release_link();
    return CallbackReturn::ERROR;
  }
  {
    std::lock_guard<std::mutex> lock(link_mutex_);
    health_pub_ = node->create_publisher<orion_msgs::msg::RoverHealth>(kHealthTopic, 10);
    health_timer_ = node->create_wall_timer(
      std::chrono::duration<double>(1.0 / drive_.health_rate_hz), [this] {publish_health();});
    estop_sub_ = node->create_subscription<std_msgs::msg::Bool>(
      kEstopTopic, rclcpp::QoS(10),
      [this](std_msgs::msg::Bool::ConstSharedPtr msg) {on_estop(msg->data);});
  }

  // Fill the state interfaces once so controllers start from the real wheel positions.
  if (read(rclcpp::Time(0, 0, RCL_STEADY_TIME), rclcpp::Duration(0, 0)) != return_type::OK) {
    release_link();
    return CallbackReturn::FAILURE;
  }
  return CallbackReturn::SUCCESS;
}

void OrionSystem::release_link()
{
  active_ = false;
  std::shared_ptr<FirmwareLink> old;
  {
    std::lock_guard<std::mutex> lock(link_mutex_);
    estop_sub_.reset();
    health_timer_.reset();
    health_pub_.reset();
    old.swap(link_);
  }
  if (old) {
    old->stop();
  }
}

void OrionSystem::send_idle()
{
  if (auto l = link()) {
    op::CmdMsg idle;  // zero targets, enable bit clear: firmware stops driving immediately
    idle.flags = drive_.brake_on_idle ? op::cmd_flag::kBrakeOnIdle : 0;
    l->send_cmd(idle);
  }
}

OrionSystem::CallbackReturn OrionSystem::on_cleanup(const rclcpp_lifecycle::State &)
{
  send_idle();
  release_link();
  return CallbackReturn::SUCCESS;
}

OrionSystem::CallbackReturn OrionSystem::on_shutdown(const rclcpp_lifecycle::State &)
{
  send_idle();
  release_link();
  return CallbackReturn::SUCCESS;
}

OrionSystem::CallbackReturn OrionSystem::on_activate(const rclcpp_lifecycle::State &)
{
  auto l = link();
  if (!l || !l->wait_for_state(
      std::chrono::milliseconds(
        static_cast<int>(drive_.state_timeout_s * 1000.0) + 50)))
  {
    RCLCPP_ERROR(get_logger(), "Cannot activate: no fresh STATE from firmware");
    return CallbackReturn::ERROR;
  }
  for (const auto & n : names_) {
    set_command(n.command, 0.0);
  }
  active_ = true;
  RCLCPP_INFO(get_logger(), "OrionSystem active");
  return CallbackReturn::SUCCESS;
}

OrionSystem::CallbackReturn OrionSystem::on_deactivate(const rclcpp_lifecycle::State &)
{
  active_ = false;
  send_idle();
  return CallbackReturn::SUCCESS;
}

OrionSystem::CallbackReturn OrionSystem::on_error(const rclcpp_lifecycle::State &)
{
  RCLCPP_ERROR(get_logger(), "OrionSystem error: stopping wheels and closing the link");
  send_idle();  // best effort; the firmware watchdog stops the wheels within 200 ms anyway
  release_link();
  return CallbackReturn::SUCCESS;
}

return_type OrionSystem::read(const rclcpp::Time &, const rclcpp::Duration &)
{
  auto l = link();
  if (!l) {
    return return_type::ERROR;
  }
  if (l->failed()) {
    RCLCPP_ERROR(get_logger(), "Serial device %s failed", drive_.port.c_str());
    return return_type::ERROR;
  }
  const auto sample = l->latest_state();
  const auto now = FirmwareLink::Clock::now();
  const auto max_age = std::chrono::duration_cast<FirmwareLink::Clock::duration>(
    std::chrono::duration<double>(drive_.state_timeout_s));
  if (!sample || !is_fresh(now, sample->rx_time, max_age)) {
    const double age_ms = sample ?
      std::chrono::duration<double, std::milli>(now - sample->rx_time).count() : INFINITY;
    RCLCPP_ERROR(
      get_logger(), "Firmware STATE is stale (%.0f ms old, limit %.0f ms)", age_ms,
      drive_.state_timeout_s * 1000.0);
    return return_type::ERROR;
  }
  for (size_t i = 0; i < joints_.size(); ++i) {
    const JointReading r = to_joint_reading(sample->msg, joints_[i], drive_);
    set_state(names_[i].position, r.position_rad);
    set_state(names_[i].velocity, r.velocity_rad_s);
    if (!names_[i].effort.empty()) {
      set_state(names_[i].effort, r.effort_a);
    }
  }
  return return_type::OK;
}

return_type OrionSystem::write(const rclcpp::Time &, const rclcpp::Duration &)
{
  auto l = link();
  if (!l || l->failed()) {
    return return_type::ERROR;
  }
  if (estop_requested_) {
    l->send_estop();  // repeated every cycle: the latch is idempotent, a lost frame is not
    return return_type::OK;
  }
  if (!active_) {
    return return_type::OK;
  }
  op::CmdMsg cmd;
  cmd.flags = op::cmd_flag::kEnable | (drive_.brake_on_idle ? op::cmd_flag::kBrakeOnIdle : 0);
  for (size_t i = 0; i < joints_.size(); ++i) {
    const double w = get_command<double>(names_[i].command);
    cmd.w_target_mrad_s[joints_[i].motor_index] = rad_s_to_mrad_s(joints_[i].direction * w);
  }
  l->send_cmd(cmd);  // a dropped frame is counted; the next one follows in one cycle
  return return_type::OK;
}

void OrionSystem::on_estop(bool engaged)
{
  auto l = link();
  if (!l) {
    return;
  }
  if (engaged && !estop_requested_.exchange(true)) {
    l->send_estop();
    RCLCPP_WARN(get_logger(), "E-stop engaged: ESTOP sent to firmware");
  } else if (!engaged && estop_requested_.exchange(false)) {
    l->send_estop_clear();
    RCLCPP_WARN(get_logger(), "E-stop released: ESTOP_CLEAR sent to firmware");
  }
}

void OrionSystem::publish_health()
{
  std::shared_ptr<FirmwareLink> l;
  rclcpp::Publisher<orion_msgs::msg::RoverHealth>::SharedPtr pub;
  std::string fw_hash;
  {
    std::lock_guard<std::mutex> lock(link_mutex_);
    l = link_;
    pub = health_pub_;
    fw_hash = fw_git_hash_;
  }
  if (!l || !pub) {
    return;
  }
  const auto sample = l->latest_state();
  if (!sample) {
    return;
  }
  const op::StateMsg & s = sample->msg;
  const auto host = l->counters();
  const auto fw = l->latest_stats();

  orion_msgs::msg::RoverHealth h;
  h.header.stamp = get_clock()->now();
  h.battery_v = (s.status & op::status::kVbatSensed) ? s.vbat_mv * 1.0e-3f : NAN;
  h.fw_fault_flags = s.faults;
  h.fw_uptime_ms = static_cast<uint32_t>(sample->fw_time_us / 1000u);
  h.crc_errors = host.rx_crc_errors + host.rx_framing_errors +
    (fw ? fw->rx_crc_errors + fw->rx_framing_errors : 0u);
  h.watchdog_trips = fw ? fw->watchdog_trips : 0u;
  h.fw_git_hash = fw_hash;
  h.current_sensed = true;
  for (const auto & j : joints_) {
    h.duty[j.motor_index] = static_cast<float>(
      j.direction * s.duty[j.motor_index] / static_cast<double>(op::kDutyFullScale));
    h.current_sensed = h.current_sensed && s.current_ma[j.motor_index] != op::kCurrentNotAvailable;
  }
  pub->publish(h);
}

}  // namespace orion_hardware

PLUGINLIB_EXPORT_CLASS(orion_hardware::OrionSystem, hardware_interface::SystemInterface)
