// ros2_control SystemInterface for the ORION rover's wheel firmware (docs/PROTOCOL.md).
//
// Four velocity-commanded wheel joints. read() takes the newest STATE frame from a background
// reader thread and never blocks on the serial port; a STATE older than state_timeout_s
// (default 100 ms) makes read() return ERROR so the controller manager stops the hardware.
#ifndef ORION_HARDWARE__ORION_SYSTEM_HPP_
#define ORION_HARDWARE__ORION_SYSTEM_HPP_

#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "hardware_interface/system_interface.hpp"
#include "hardware_interface/types/hardware_component_interface_params.hpp"
#include "orion_hardware/conversions.hpp"
#include "orion_hardware/firmware_link.hpp"
#include "orion_msgs/msg/rover_health.hpp"
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/bool.hpp"

namespace orion_hardware
{

class OrionSystem : public hardware_interface::SystemInterface
{
public:
  using CallbackReturn = rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;
  using TransportFactory = std::function<std::unique_ptr<Transport>(const DriveConfig &)>;

  static constexpr char kHealthTopic[] = "/orion/rover_health";
  static constexpr char kEstopTopic[] = "/orion/estop";

  /// Uses a POSIX SerialTransport on the configured port.
  OrionSystem();
  /// Test seam: build the transport some other way (e.g. a mock).
  explicit OrionSystem(TransportFactory factory);
  ~OrionSystem() override;

  CallbackReturn on_init(
    const hardware_interface::HardwareComponentInterfaceParams & params) override;
  CallbackReturn on_configure(const rclcpp_lifecycle::State & previous_state) override;
  CallbackReturn on_cleanup(const rclcpp_lifecycle::State & previous_state) override;
  CallbackReturn on_shutdown(const rclcpp_lifecycle::State & previous_state) override;
  CallbackReturn on_activate(const rclcpp_lifecycle::State & previous_state) override;
  CallbackReturn on_deactivate(const rclcpp_lifecycle::State & previous_state) override;
  CallbackReturn on_error(const rclcpp_lifecycle::State & previous_state) override;

  hardware_interface::return_type read(
    const rclcpp::Time & time, const rclcpp::Duration & period) override;
  hardware_interface::return_type write(
    const rclcpp::Time & time, const rclcpp::Duration & period) override;

  const DriveConfig & drive_config() const {return drive_;}
  const std::vector<JointConfig> & joint_configs() const {return joints_;}

private:
  struct JointInterfaces
  {
    std::string position;
    std::string velocity;
    std::string effort;  // empty if the URDF does not export effort
    std::string command;
  };

  bool handshake();
  void release_link();
  void send_idle();
  void publish_health();
  void on_estop(bool engaged);
  std::shared_ptr<FirmwareLink> link() const;

  TransportFactory factory_;
  DriveConfig drive_;
  std::vector<JointConfig> joints_;
  std::vector<JointInterfaces> names_;

  mutable std::mutex link_mutex_;  // guards link_ against the executor thread (callbacks)
  std::shared_ptr<FirmwareLink> link_;
  std::atomic<bool> active_{false};
  std::atomic<bool> estop_requested_{false};

  rclcpp::Publisher<orion_msgs::msg::RoverHealth>::SharedPtr health_pub_;
  rclcpp::TimerBase::SharedPtr health_timer_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr estop_sub_;
};

}  // namespace orion_hardware

#endif  // ORION_HARDWARE__ORION_SYSTEM_HPP_
