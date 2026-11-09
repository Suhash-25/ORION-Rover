"""
Bring up the REAL ORION rover (no Gazebo). Checklist before first drive: docs/BRINGUP.md.

Starts robot_state_publisher (URDF with hardware:=serial), ros2_control_node with
orion_control/config/controllers.yaml, the controller spawners, twist_mux and orion_safety
(rover thresholds, system time). The lidar and IMU drivers depend on the hardware list,
which is not decided yet (docs/HARDWARE.md): pass their launch files as imu_launch:=... and
lidar_launch:=... . Without an IMU driver orion_safety HALTs after its startup grace,
by design: the rover must not drive without tilt protection.

The drive-train values (port, baud, encoder_cpr, gear_ratio, kt, motor_resistance,
brake_on_idle) and perturbation_max_rad_s are launch arguments forwarded to the xacro; their
defaults are the values (GUESSES) of orion_description/urdf/control/ros2_control.xacro
(test_robot_launch checks they agree).
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import IncludeLaunchDescription
from launch.actions import LogInfo
from launch.conditions import IfCondition
from launch.conditions import UnlessCondition
from launch.launch_description_sources import AnyLaunchDescriptionSource
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command
from launch.substitutions import LaunchConfiguration
from launch.substitutions import PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


# xacro argument -> (default, description). Defaults equal ros2_control.xacro.
DRIVE_ARGS = {
    "port": ("/dev/ttyACM0", "Serial device of the wheel firmware, e.g. /dev/orion_fw "
                             "(udev rule: orion_bringup/config/99-orion.rules)"),
    "baud": ("921600", "Serial baud rate (ignored by USB CDC devices)"),
    "encoder_cpr": ("2048", "Decoder counts per motor revolution after x4 decoding (GUESS)"),
    "gear_ratio": ("30", "Motor revolutions per wheel revolution (GUESS)"),
    "kt": ("0.0", "Motor torque constant N m/A; 0 disables back-EMF correction"),
    "motor_resistance": ("0.5", "Motor terminal resistance, ohm (GUESS)"),
    "brake_on_idle": ("false", "Short the motors when idle instead of coasting"),
    "perturbation_max_rad_s": ("0.0", "Bound of /orion/wheel_perturbation offsets, rad/s; "
                                      "0 disables the fault-injection hook"),
}


def robot_description_command(xacro_file):
    """Build the xacro Command with hardware:= and every DRIVE_ARGS launch argument forwarded."""
    parts = ["xacro ", xacro_file, " hardware:=", LaunchConfiguration("hardware")]
    for name in DRIVE_ARGS:
        parts += [f" {name}:=", LaunchConfiguration(name)]
    return Command(parts)


def _given(arg):
    """Condition: launch argument `arg` is a non-empty string."""
    return PythonExpression(["'", LaunchConfiguration(arg), "' != ''"])


def _driver(arg, topic, frame):
    """Include the driver launch file named by `arg`, or warn loudly that there is none."""
    return [
        IncludeLaunchDescription(
            AnyLaunchDescriptionSource(LaunchConfiguration(arg)),
            condition=IfCondition(_given(arg)),
        ),
        LogInfo(
            condition=UnlessCondition(_given(arg)),
            msg=f"WARNING: no {arg} given: nothing publishes {topic} (frame {frame}). "
                "The driver depends on the hardware list (docs/HARDWARE.md); pass "
                f"{arg}:=/path/to/driver.launch.py once it is known.",
        ),
    ]


def generate_launch_description():
    description_pkg = get_package_share_directory("orion_description")
    control_pkg = get_package_share_directory("orion_control")
    safety_pkg = get_package_share_directory("orion_safety")

    xacro_file = os.path.join(description_pkg, "urdf", "robot", "orion.urdf.xacro")
    robot_description = ParameterValue(
        robot_description_command(xacro_file), value_type=str,
    )

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="screen",
        parameters=[{"robot_description": robot_description, "use_sim_time": False}],
    )

    # Jazzy: the controller manager takes the URDF from robot_state_publisher's topic.
    ros2_control_node = Node(
        package="controller_manager",
        executable="ros2_control_node",
        output="screen",
        parameters=[os.path.join(control_pkg, "config", "controllers.yaml")],
        remappings=[("~/robot_description", "/robot_description")],
    )

    controllers = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(control_pkg, "launch", "controllers.launch.py")),
        launch_arguments={
            "teleop": LaunchConfiguration("teleop"),
            "use_sim_time": "false",
        }.items(),
    )

    twist_mux = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(control_pkg, "launch", "twist_mux.launch.py")),
        launch_arguments={"use_sim_time": "false"}.items(),
    )

    safety = Node(
        package="orion_safety",
        executable="safety_node",
        name="orion_safety",
        output="screen",
        parameters=[os.path.join(safety_pkg, "config", "safety.yaml"), {"use_sim_time": False}],
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "hardware",
            default_value="serial",
            choices=["serial", "gazebo"],
            description="ros2_control hardware passed to the URDF xacro (serial = orion_hardware)",
        ),
        *[DeclareLaunchArgument(name, default_value=default, description=text)
          for name, (default, text) in DRIVE_ARGS.items()],
        DeclareLaunchArgument(
            "teleop",
            default_value="false",
            description="Open the keyboard teleop terminal (needs a desktop session)",
        ),
        DeclareLaunchArgument(
            "imu_launch",
            default_value="",
            description="Launch file of the IMU driver (publishes /imu/data, frame imu_link)",
        ),
        DeclareLaunchArgument(
            "lidar_launch",
            default_value="",
            description="Launch file of the lidar driver (publishes /scan, frame lidar_link)",
        ),
        robot_state_publisher,
        ros2_control_node,
        controllers,
        twist_mux,
        safety,
        *_driver("imu_launch", "/imu/data", "imu_link"),
        *_driver("lidar_launch", "/scan", "lidar_link"),
    ])
