"""Spawn ORION controllers, then optionally keyboard teleop (Gazebo or the real rover)."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import IncludeLaunchDescription
from launch.actions import RegisterEventHandler
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Spawn controllers in dependency order for gz_ros2_control."""
    control_share = get_package_share_directory("orion_control")

    joint_state_broadcaster = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "joint_state_broadcaster",
            "--controller-manager",
            "/controller_manager",
            "--controller-manager-timeout",
            "60",
            "--switch-asap",
            "--switch-timeout",
            "60",
        ],
        output="screen",
    )

    diff_drive_controller = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "diff_drive_controller",
            "--controller-manager",
            "/controller_manager",
            "--controller-manager-timeout",
            "60",
            "--switch-asap",
            "--switch-timeout",
            "60",
        ],
        output="screen",
    )

    teleop = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(control_share, "launch", "teleop.launch.py")
        ),
        condition=IfCondition(LaunchConfiguration("teleop")),
        launch_arguments={"use_sim_time": LaunchConfiguration("use_sim_time")}.items(),
    )

    def start_teleop_after_success(event, _context):
        """Open the keyboard terminal only after drive-controller activation."""
        return [teleop] if event.returncode == 0 else []

    return LaunchDescription([
        DeclareLaunchArgument(
            "teleop",
            default_value="true",
            description="Open the keyboard teleop terminal (needs a desktop)",
        ),
        DeclareLaunchArgument(
            "use_sim_time",
            default_value="true",
            description="Passed to teleop; false on the real rover",
        ),
        joint_state_broadcaster,
        RegisterEventHandler(
            OnProcessExit(
                target_action=joint_state_broadcaster,
                on_exit=[diff_drive_controller],
            )
        ),
        RegisterEventHandler(
            OnProcessExit(
                target_action=diff_drive_controller,
                on_exit=start_teleop_after_success,
            )
        ),
    ])
