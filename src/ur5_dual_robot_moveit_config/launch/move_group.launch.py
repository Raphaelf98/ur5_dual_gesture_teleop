import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from moveit_configs_utils import MoveItConfigsBuilder
from moveit_configs_utils.launches import generate_move_group_launch
from ament_index_python.packages import get_package_share_directory


def configure_moveit(context, *args, **kwargs):
    solver = context.perform_substitution(LaunchConfiguration("kinematics_solver"))
    pkg_share = get_package_share_directory("ur5_dual_robot_moveit_config")
    kinematics_file = os.path.join(pkg_share, "config", f"kinematics_{solver}.yaml")
    moveit_config = (
        MoveItConfigsBuilder("dual_ur5", package_name="ur5_dual_robot_moveit_config")
        .robot_description_kinematics(file_path=kinematics_file)
        .to_moveit_configs()
    )
    return generate_move_group_launch(moveit_config).entities


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            "kinematics_solver",
            default_value="analytical",
            description="Kinematics solver: 'analytical' (IKFast) or 'numerical' (KDL)",
            choices=["analytical", "numerical"],
        ),
        OpaqueFunction(function=configure_moveit),
    ])
