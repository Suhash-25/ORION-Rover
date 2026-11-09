"""
Start the trial runner.

ros2 launch orion_experiments trial_runner.launch.py operator:=suhas bag_root:=/data/trials
Per-trial values (ballast_kg, cone_depths_mm) are set with `ros2 param set` before each trial.
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    args = {
        'bag_root': os.path.expanduser('~/orion_trials'),
        'operator': '',
        'model_dir': '',
        'firmware_hash': '',
    }
    declared = [DeclareLaunchArgument(k, default_value=v) for k, v in args.items()]
    declared.append(DeclareLaunchArgument('use_sim_time', default_value='false'))
    params = {k: ParameterValue(LaunchConfiguration(k), value_type=str) for k in args}
    params['use_sim_time'] = ParameterValue(LaunchConfiguration('use_sim_time'), value_type=bool)
    node = Node(package='orion_experiments', executable='trial_runner',
                name='orion_trial_runner', output='screen', parameters=[params])
    return LaunchDescription(declared + [node])
