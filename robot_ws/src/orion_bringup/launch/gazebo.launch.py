import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import IncludeLaunchDescription
from launch.actions import RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import AnyLaunchDescriptionSource
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command
from launch.substitutions import LaunchConfiguration
from launch.substitutions import PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():

    bringup_pkg = get_package_share_directory("orion_bringup")
    description_pkg = get_package_share_directory("orion_description")
    gazebo_pkg = get_package_share_directory("ros_gz_sim")
    bridge_pkg = get_package_share_directory("ros_gz_bridge")
    control_pkg = get_package_share_directory("orion_control")
    safety_pkg = get_package_share_directory("orion_safety")

    declare_world = DeclareLaunchArgument(
        "world",
        default_value=os.path.join(
            bringup_pkg,
            "worlds",
            "empty.sdf",
        ),
        description="Gazebo world file",
    )

    world = LaunchConfiguration("world")

    # headless:=true runs the Gazebo server only (no GUI) with offscreen
    # rendering, so the lidar still works on CI machines without a display.
    declare_headless = DeclareLaunchArgument(
        "headless",
        default_value="false",
        description="Run Gazebo without a GUI (CI and remote machines)",
    )
    declare_teleop = DeclareLaunchArgument(
        "teleop",
        default_value="true",
        description="Open the keyboard teleop terminal once controllers are up",
    )
    # Passed to xacro as hardware:=... ; orion_description selects the ros2_control plugin
    # with it (gazebo = gz_ros2_control). xacro ignores the argument until that exists.
    declare_hardware = DeclareLaunchArgument(
        "hardware",
        default_value="gazebo",
        choices=["gazebo", "serial"],
        description="ros2_control hardware passed to the URDF xacro",
    )
    headless = LaunchConfiguration("headless")
    gz_args = PythonExpression([
        "'-r -s --headless-rendering ' if '", headless,
        "' == 'true' else '-r '",
    ])

    xacro_file = os.path.join(
        description_pkg,
        "urdf",
        "robot",
        "orion.urdf.xacro",
    )

    robot_description = ParameterValue(
        Command(["xacro ", xacro_file, " hardware:=", LaunchConfiguration("hardware")]),
        value_type=str,
    )

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                gazebo_pkg,
                "launch",
                "gz_sim.launch.py",
            )
        ),
        launch_arguments={
            "gz_args": [gz_args, world],
            "on_exit_shutdown": "true",
        }.items(),
    )

    clock_bridge = IncludeLaunchDescription(
        AnyLaunchDescriptionSource(
            os.path.join(
                bridge_pkg,
                "launch",
                "clock_bridge.launch",
            )
        ),
        launch_arguments={
            "bridge_name": "orion_clock_bridge",
        }.items(),
    )

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="screen",
        parameters=[
            {
                "robot_description": robot_description,
                "use_sim_time": True,
            }
        ],
    )

    spawn = Node(
        package="ros_gz_sim",
        executable="create",
        arguments=[
            "-topic",
            "robot_description",
            "-name",
            "orion",
        ],
        output="screen",
    )

    bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=[
            "/scan@sensor_msgs/msg/LaserScan@gz.msgs.LaserScan",
            "/imu/data@sensor_msgs/msg/Imu@gz.msgs.IMU",
        ],
        output="screen",
    )

    controllers = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                control_pkg,
                "launch",
                "controllers.launch.py",
            )
        ),
        launch_arguments={"teleop": LaunchConfiguration("teleop")}.items(),
    )

    # Every command source goes through twist_mux (docs/INTERFACES.md); orion_safety owns
    # the safety/estop inputs. Both start with the simulator: orion_safety's startup grace
    # covers the time until /joint_states and /imu/data first arrive.
    twist_mux = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(control_pkg, "launch", "twist_mux.launch.py")
        ),
        launch_arguments={"use_sim_time": "true"}.items(),
    )

    safety = Node(
        package="orion_safety",
        executable="safety_node",
        name="orion_safety",
        output="screen",
        parameters=[
            os.path.join(safety_pkg, "config", "safety.yaml"),
            os.path.join(safety_pkg, "config", "safety_sim.yaml"),
            {"use_sim_time": True},
        ],
    )

    return LaunchDescription(
        [
            declare_world,
            declare_headless,
            declare_teleop,
            declare_hardware,
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
            bridge,
            twist_mux,
            safety,
        ]
    )
