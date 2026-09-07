from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource

from launch_ros.actions import Node

from ament_index_python.packages import get_package_share_directory

import os


def generate_launch_description():

    ###########################################################
    # Package Paths
    ###########################################################

    bringup_pkg = get_package_share_directory("orion_bringup")

    ###########################################################
    # Start Localization
    ###########################################################

    localization = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                bringup_pkg,
                "launch",
                "localization.launch.py",
            )
        )
    )

    ###########################################################
    # Navigation Parameters
    ###########################################################

    nav2_params = os.path.join(
        bringup_pkg,
        "config",
        "nav2_params.yaml",
    )

    ###########################################################
    # Planner Server
    ###########################################################

    planner_server = Node(
        package="nav2_planner",
        executable="planner_server",
        name="planner_server",
        output="screen",
        parameters=[nav2_params],
    )

    ###########################################################
    # Controller Server
    ###########################################################

    controller_server = Node(
        package="nav2_controller",
        executable="controller_server",
        name="controller_server",
        output="screen",
        parameters=[nav2_params],
    )

    ###########################################################
    # Behavior Server
    ###########################################################

    behavior_server = Node(
        package="nav2_behaviors",
        executable="behavior_server",
        name="behavior_server",
        output="screen",
        parameters=[nav2_params],
    )

    ###########################################################
    # BT Navigator
    ###########################################################

    bt_navigator = Node(
        package="nav2_bt_navigator",
        executable="bt_navigator",
        name="bt_navigator",
        output="screen",
        parameters=[nav2_params],
    )

    ###########################################################
    # Navigation Lifecycle Manager
    ###########################################################

    lifecycle_manager_navigation = Node(
        package="nav2_lifecycle_manager",
        executable="lifecycle_manager",
        name="lifecycle_manager_navigation",
        output="screen",
        parameters=[nav2_params],
    )

    ###########################################################
    # Launch Everything
    ###########################################################

    return LaunchDescription([
        localization,
        planner_server,
        controller_server,
        behavior_server,
        bt_navigator,
        lifecycle_manager_navigation,
    ])
