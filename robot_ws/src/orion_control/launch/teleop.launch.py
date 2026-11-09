"""Launch keyboard teleoperation into twist_mux's teleop input (/cmd_vel/teleop)."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    """Run teleop with the topic and message type expected by twist_mux."""
    return LaunchDescription([
        DeclareLaunchArgument(
            "use_sim_time",
            default_value="true",
            description="Stamp commands with /clock (Gazebo); false on the real rover",
        ),
        Node(
            package="teleop_twist_keyboard",
            executable="teleop_twist_keyboard",
            name="orion_keyboard_teleop",
            parameters=[{
                "stamped": True,
                "use_sim_time": ParameterValue(
                    LaunchConfiguration("use_sim_time"), value_type=bool),
            }],
            # Priority 200 in twist_mux: overrides policy, Nav2 and a safety HALT,
            # but not a latched e-stop (docs/SAFETY.md).
            remappings=[
                ("cmd_vel", "/cmd_vel/teleop"),
            ],
            # Keyboard input requires a real terminal; ROS launch itself does
            # not forward stdin to child nodes.
            prefix="gnome-terminal --",
        ),
    ])
