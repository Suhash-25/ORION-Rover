"""Launch keyboard teleoperation for ORION's Jazzy diff-drive controller."""

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    """Run teleop with the topic and message type expected by the controller."""
    return LaunchDescription([
        Node(
            package="teleop_twist_keyboard",
            executable="teleop_twist_keyboard",
            name="orion_keyboard_teleop",
            parameters=[{
                "stamped": True,
                "use_sim_time": True,
            }],
            remappings=[
                ("cmd_vel", "/diff_drive_controller/cmd_vel"),
            ],
            # Keyboard input requires a real terminal; ROS launch itself does
            # not forward stdin to child nodes.
            prefix="gnome-terminal --",
        ),
    ])
