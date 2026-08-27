from launch import LaunchDescription

from launch.actions import DeclareLaunchArgument

from launch.substitutions import Command, LaunchConfiguration

from launch_ros.actions import Node

from launch_ros.parameter_descriptions import ParameterValue

from ament_index_python.packages import get_package_share_directory

import os


def generate_launch_description():

    # ----------------------------------------------------------
    # Get ORION Description Package
    # ----------------------------------------------------------

    pkg_description = get_package_share_directory(
        "orion_description"
    )

    xacro_file = os.path.join(
        pkg_description,
        "urdf",
        "robot",
        "orion.urdf.xacro"
    )

    robot_description = ParameterValue(
        Command(["xacro ", xacro_file]),
        value_type=str
    )

    # ----------------------------------------------------------
    # RViz Configuration
    # ----------------------------------------------------------

    rviz_config = os.path.join(
        get_package_share_directory("orion_bringup"),
        "rviz",
        "display.rviz"
    )

    # ----------------------------------------------------------
    # Launch Description
    # ----------------------------------------------------------

    return LaunchDescription([

        DeclareLaunchArgument(
            "use_sim_time",
            default_value="false"
        ),

        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            parameters=[
                {
                    "robot_description": robot_description,
                    "use_sim_time": LaunchConfiguration("use_sim_time")
                }
            ],
            output="screen"
        ),

        Node(
            package="joint_state_publisher_gui",
            executable="joint_state_publisher_gui",
            output="screen"
        ),

        Node(
            package="rviz2",
            executable="rviz2",
            arguments=["-d", rviz_config],
            output="screen"
        ),

    ])
