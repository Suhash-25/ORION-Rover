// ros2_control SystemInterface for the ORION rover's wheel firmware (docs/PROTOCOL.md).
//
// Four velocity-commanded wheel joints. read() takes the newest STATE frame from a background
// reader thread and never blocks on the serial port; a STATE older than state_timeout_s
// (default 100 ms) makes read() return ERROR so the controller manager stops the hardware.
//
// /orion/estop true sends ESTOP to the firmware (repeated every write()); ESTOP_CLEAR follows
// only after orion_safety accepted /orion/estop_reset (EstopLatch, council m1). A
// WheelPerturbation on /orion/wheel_perturbation adds bounded offsets to the commanded wheel
// velocities (PerturbationGate; off unless perturbation_max_rad_s > 0).
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
#include "orion_hardware/safety_gates.hpp"
#include "orion_msgs/msg/rover_health.hpp"
#include "orion_msgs/msg/safety_status.hpp"
#include "orion_msgs/msg/wheel_perturbation.hpp"
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
  static constexpr char kSafetyTopic[] = "/orion/safety_status";
  static constexpr char kPerturbationTopic[] = "/orion/wheel_perturbation";

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
  void on_safety_status(uint8_t state);
  void on_perturbation(const orion_msgs::msg::WheelPerturbation & msg);
  /// Fresh SafetyStatus in NOMINAL/LIMITED and no e-stop latch. Needs gate_mutex_ held.
  bool safety_ok_locked(FirmwareLink::Clock::time_point now) const;
  std::shared_ptr<FirmwareLink> link() const;

  TransportFactory factory_;
  DriveConfig drive_;
  std::vector<JointConfig> joints_;
  std::vector<JointInterfaces> names_;

  mutable std::mutex link_mutex_;  // guards link_ against the executor thread (callbacks)
  std::shared_ptr<FirmwareLink> link_;
  std::string fw_git_hash_;  // from the PONG handshake, guarded by link_mutex_
  std::atomic<bool> active_{false};

  mutable std::mutex gate_mutex_;  // guards the members below (callbacks vs. write())
  EstopLatch estop_;
  PerturbationGate perturbation_;
  bool have_safety_ = false;
  uint8_t safety_state_ = safety_state::kNominal;
  FirmwareLink::Clock::time_point safety_rx_{};

  rclcpp::Publisher<orion_msgs::msg::RoverHealth>::SharedPtr health_pub_;
  rclcpp::TimerBase::SharedPtr health_timer_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr estop_sub_;
  rclcpp::Subscription<orion_msgs::msg::SafetyStatus>::SharedPtr safety_sub_;
  rclcpp::Subscription<orion_msgs::msg::WheelPerturbation>::SharedPtr perturbation_sub_;
};

}  // namespace orion_hardware

#endif  // ORION_HARDWARE__ORION_SYSTEM_HPP_
