from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    moveit_pkg = get_package_share_directory("ur5_dual_robot_moveit_config")

    # MoveIt + RViz + ros2_control
    demo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(moveit_pkg, "launch", "demo.launch.py")
        )
    )

    # Servo nodes — delayed 5s to let demo fully start
    servo = TimerAction(
        period=5.0,
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(moveit_pkg, "launch", "servo.launch.py")
                )
            )
        ]
    )

    # Hand tracker node — delayed 6s to let servo start first
    hand_tracker = TimerAction(
        period=6.0,
        actions=[
            Node(
                package="ur5_dual_robot_teleop",
                executable="hand_tracker_node",
                name="hand_tracker_node",
                output="screen",
            )
        ]
    )

    # Teleop node with hand tracking input — delayed 7s
    teleop = TimerAction(
        period=7.0,
        actions=[
            Node(
                package="ur5_dual_robot_teleop",
                executable="dual_arm_teleop_node",
                name="teleop_node",
                parameters=[{"input": "hand_tracking"}],
                output="screen",
            )
        ]
    )

    return LaunchDescription([demo, servo, hand_tracker, teleop])