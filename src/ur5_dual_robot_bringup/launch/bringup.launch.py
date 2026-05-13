from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    moveit_pkg = get_package_share_directory("ur5_dual_robot_moveit_config")

    demo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(moveit_pkg, "launch", "demo.launch.py")
        )
    )

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

    return LaunchDescription([demo, servo])