from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource

from launch_ros.actions import Node

from ament_index_python.packages import get_package_share_directory

import os


def generate_launch_description():

    bringup_pkg = get_package_share_directory("orion_bringup")

    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                bringup_pkg,
                "launch",
                "gazebo.launch.py",
            )
        ),
        launch_arguments={
            "world": os.path.join(
                bringup_pkg,
                "worlds",
                "mapping_world.sdf",
            )
        }.items(),
    )

    nav2_params = os.path.join(
        bringup_pkg,
        "config",
        "nav2_params.yaml",
    )

    map_file = os.path.join(
        bringup_pkg,
        "maps",
        "my_first_map.yaml",
    )

    map_server = Node(
        package="nav2_map_server",
        executable="map_server",
        name="map_server",
        output="screen",
        parameters=[
            nav2_params,
            {
                "yaml_filename": map_file,
            },
        ],
    )

    amcl = Node(
        package="nav2_amcl",
        executable="amcl",
        name="amcl",
        output="screen",
        parameters=[
            nav2_params,
        ],
    )

    lifecycle_manager = Node(
        package="nav2_lifecycle_manager",
        executable="lifecycle_manager",
        name="lifecycle_manager_localization",
        output="screen",
        parameters=[
            nav2_params,
        ],
    )

    return LaunchDescription([
        gazebo_launch,
        map_server,
        amcl,
        lifecycle_manager,
    ])
