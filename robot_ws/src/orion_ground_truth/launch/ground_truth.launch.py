"""Overhead camera -> AprilTag detector -> /orion/ground_truth_pose.

Needs a calibrated camera (camera_info_url) for metric tag poses. See docs/BRINGUP.md.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory('orion_ground_truth')
    device = LaunchConfiguration('video_device')
    calib = LaunchConfiguration('camera_info_url')
    return LaunchDescription([
        DeclareLaunchArgument('video_device', default_value='/dev/video0'),
        DeclareLaunchArgument('camera_info_url', default_value='',
                              description='file:///path/to/overhead_camera.yaml'),
        Node(package='v4l2_camera', executable='v4l2_camera_node', name='overhead_camera',
             namespace='overhead', parameters=[{
                 'video_device': device, 'camera_info_url': calib,
                 'camera_frame_id': 'overhead_camera', 'image_size': [1280, 720]}]),
        Node(package='image_proc', executable='rectify_node', name='rectify',
             namespace='overhead',
             remappings=[('image', 'image_raw'), ('image_rect', 'image_rect')]),
        Node(package='apriltag_ros', executable='apriltag_node', name='apriltag',
             parameters=[os.path.join(share, 'config', 'apriltag.yaml')],
             remappings=[('image_rect', '/overhead/image_rect'),
                         ('camera_info', '/overhead/camera_info')]),
        Node(package='orion_ground_truth', executable='ground_truth_node',
             name='orion_ground_truth',
             parameters=[os.path.join(share, 'config', 'ground_truth.yaml')]),
    ])
