"""Start twist_mux: every command source -> /diff_drive_controller/cmd_vel (TwistStamped)."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    config = os.path.join(
        get_package_share_directory("orion_control"), "config", "twist_mux.yaml")
    return LaunchDescription([
        DeclareLaunchArgument(
            "use_sim_time",
            default_value="true",
            description="Use /clock (Gazebo); false on the real rover",
        ),
        Node(
            package="twist_mux",
            executable="twist_mux",
            name="twist_mux",
            output="screen",
            parameters=[
                config,
                {"use_sim_time": ParameterValue(
                    LaunchConfiguration("use_sim_time"), value_type=bool)},
            ],
            remappings=[("cmd_vel_out", "/diff_drive_controller/cmd_vel")],
        ),
    ])
