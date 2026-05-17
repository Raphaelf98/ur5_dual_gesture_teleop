from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    moveit_pkg = get_package_share_directory("ur5_dual_robot_moveit_config")

    kinematics_solver_arg = DeclareLaunchArgument(
        "kinematics_solver",
        default_value="analytical",
        description="Kinematics solver: 'analytical' (IKFast) or 'numerical' (KDL)",
        choices=["analytical", "numerical"],
    )

    demo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(moveit_pkg, "launch", "demo.launch.py")
        ),
        launch_arguments={"kinematics_solver": LaunchConfiguration("kinematics_solver")}.items(),
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

    return LaunchDescription([kinematics_solver_arg, demo, servo])
