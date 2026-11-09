"""
Start orion_policy with config/policy.yaml.

ros2 launch orion_policy policy.launch.py initial_mode:=BASELINE model_dir:=/path/to/student
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    config = os.path.join(get_package_share_directory('orion_policy'), 'config', 'policy.yaml')
    args = [
        DeclareLaunchArgument('initial_mode', default_value='IDLE'),
        DeclareLaunchArgument('model_dir', default_value=''),
        DeclareLaunchArgument('use_sim_time', default_value='false'),
    ]
    node = Node(
        package='orion_policy',
        executable='policy_node',
        name='orion_policy',
        output='screen',
        parameters=[config, {
            'initial_mode': ParameterValue(LaunchConfiguration('initial_mode'), value_type=str),
            'model_dir': ParameterValue(LaunchConfiguration('model_dir'), value_type=str),
            'use_sim_time': ParameterValue(LaunchConfiguration('use_sim_time'), value_type=bool),
        }],
    )
    return LaunchDescription(args + [node])
