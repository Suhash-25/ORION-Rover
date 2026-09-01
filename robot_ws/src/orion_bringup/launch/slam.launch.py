from launch import LaunchDescription

from launch.actions import IncludeLaunchDescription

from launch.launch_description_sources import PythonLaunchDescriptionSource

from launch_ros.actions import Node

from ament_index_python.packages import get_package_share_directory

import os


def generate_launch_description():

    bringup_pkg = get_package_share_directory("orion_bringup")

    slam_pkg = get_package_share_directory("slam_toolbox")

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

    slam_params = os.path.join(
    bringup_pkg,
    "config",
    "slam_params.yaml",
    )

    slam = Node(
       package="slam_toolbox",
       executable="async_slam_toolbox_node",
       name="slam_toolbox",
       output="screen",
       parameters=[
           slam_params,
       ],
    )    


    return LaunchDescription([
        gazebo_launch,
        slam,
    ])
