from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import (
    AnyLaunchDescriptionSource,
    PythonLaunchDescriptionSource,
)

from launch.substitutions import Command
from launch_ros.parameter_descriptions import ParameterValue

from launch_ros.actions import Node

from ament_index_python.packages import get_package_share_directory

import os


def generate_launch_description():

    bringup_pkg = get_package_share_directory("orion_bringup")
    description_pkg = get_package_share_directory("orion_description")
    gazebo_pkg = get_package_share_directory("ros_gz_sim")
    bridge_pkg = get_package_share_directory("ros_gz_bridge")

    world = os.path.join(
        bringup_pkg,
        "worlds",
        "empty.sdf"
    )

    xacro_file = os.path.join(
        description_pkg,
        "urdf",
        "robot",
        "orion.urdf.xacro"
    )

    robot_description = ParameterValue(
        Command(["xacro ", xacro_file]),
        value_type=str
    )

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                gazebo_pkg,
                "launch",
                "gz_sim.launch.py"
            )
        ),
        launch_arguments={
            "gz_args": world
        }.items()
    )

    # gz_ros2_control uses simulated time. Bridge Gazebo's /clock before the
    # robot is spawned so controller activation receives simulation updates.
    clock_bridge = IncludeLaunchDescription(
        AnyLaunchDescriptionSource(
            os.path.join(bridge_pkg, "launch", "clock_bridge.launch")
        ),
        launch_arguments={"bridge_name": "orion_clock_bridge"}.items(),
    )

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[
            {
                "robot_description": robot_description,
                "use_sim_time": True,
            }
        ],
        output="screen"
    )

    spawn = Node(
        package="ros_gz_sim",
        executable="create",
        arguments=[
            "-topic",
            "robot_description",
            "-name",
            "orion"
        ],
        output="screen"
    )

    controllers = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("orion_control"),
                "launch",
                "controllers.launch.py",
            )
        )
    )

    return LaunchDescription([
        gazebo,
        clock_bridge,
        robot_state_publisher,
        RegisterEventHandler(
            OnProcessExit(
                target_action=spawn,
                on_exit=[controllers],
            )
        ),
        spawn,
    ])
