from launch import LaunchDescription
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder
from ament_index_python.packages import get_package_share_directory
import os
import yaml

def load_yaml(package_name, file_path):
    pkg = get_package_share_directory(package_name)
    with open(os.path.join(pkg, file_path)) as f:
        return yaml.safe_load(f)

def generate_launch_description():
    moveit_config = MoveItConfigsBuilder(
        "dual_ur5",
        package_name="ur5_dual_robot_moveit_config"
    ).to_moveit_configs()

    # wrap in "moveit_servo" namespace — this is the key!
    left_servo_params  = {"moveit_servo": load_yaml("ur5_dual_robot_moveit_config", "config/left_servo_config.yaml")}
    right_servo_params = {"moveit_servo": load_yaml("ur5_dual_robot_moveit_config", "config/right_servo_config.yaml")}

    left_servo_node = Node(
        package="moveit_servo",
        executable="servo_node_main",
        name="left_servo_node",
        parameters=[
            left_servo_params,
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
        ],
        output="screen",
    )

    right_servo_node = Node(
        package="moveit_servo",
        executable="servo_node_main",
        name="right_servo_node",
        parameters=[
            right_servo_params,
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
        ],
        output="screen",
    )
    # add this node to the launch description
    joint_state_publisher = Node(
        package="joint_state_publisher",
        executable="joint_state_publisher",
        name="joint_state_publisher",
        parameters=[{"source_list": ["/joint_states"]}],
    )

    return LaunchDescription([left_servo_node, right_servo_node, joint_state_publisher])