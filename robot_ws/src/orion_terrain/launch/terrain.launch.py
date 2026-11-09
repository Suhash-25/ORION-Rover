"""
Start terrain_node with config/terrain.yaml.

    ros2 launch orion_terrain terrain.launch.py sim:=true     # Gazebo (sim time, SIM_TORQUE)
    ros2 launch orion_terrain terrain.launch.py               # rover
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    config = os.path.join(get_package_share_directory('orion_terrain'), 'config',
                          'terrain.yaml')
    sim = LaunchConfiguration('sim')
    return LaunchDescription([
        DeclareLaunchArgument('sim', default_value='false',
                              description='Gazebo: sim time and joint-torque effort'),
        DeclareLaunchArgument('effort_model_path', default_value='',
                              description="'' = the default model for sim / hardware"),
        Node(
            package='orion_terrain',
            executable='terrain_node',
            name='terrain_node',
            output='screen',
            parameters=[config, {
                'sim': ParameterValue(sim, value_type=bool),
                'use_sim_time': ParameterValue(sim, value_type=bool),
                'effort_model_path': LaunchConfiguration('effort_model_path'),
            }],
        ),
    ])
