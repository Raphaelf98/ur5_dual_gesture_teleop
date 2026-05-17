import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from moveit_configs_utils import MoveItConfigsBuilder


def launch_setup(context, *args, **kwargs):
    moveit_pkg  = get_package_share_directory('ur5_dual_robot_moveit_config')
    bringup_pkg = get_package_share_directory('ur5_dual_robot_bringup')

    solver = context.perform_substitution(LaunchConfiguration("kinematics_solver"))
    kinematics_file = os.path.join(moveit_pkg, "config", f"kinematics_{solver}.yaml")

    rviz_config = os.path.join(bringup_pkg, 'rviz', 'dual_robot.rviz')

    moveit_config = (
        MoveItConfigsBuilder('dual_ur5', package_name='ur5_dual_robot_moveit_config')
        .robot_description_kinematics(file_path=kinematics_file)
        .to_moveit_configs()
    )

    # ── Demo (rsp, move_group, ros2_control, spawn_controllers) — no RViz ─────
    demo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(moveit_pkg, 'launch', 'demo.launch.py')
        ),
        launch_arguments={
            'use_rviz': 'false',
            'kinematics_solver': solver,
        }.items(),
    )

    # ── RViz with our saved config ────────────────────────────────────────────
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config],
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
        ],
        output='log',
    )

    # ── Gripper controllers — spawned after ros2_control is ready ─────────────
    left_gripper_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['left_gripper_controller'],
        output='screen',
    )
    right_gripper_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['right_gripper_controller'],
        output='screen',
    )

    # ── Gripper driver node ───────────────────────────────────────────────────
    gripper_node = Node(
        package='ur5_dual_robot_teleop',
        executable='gripper_node',
        name='gripper_node',
        output='screen',
    )

    # ── Servo — delayed 5 s ───────────────────────────────────────────────────
    servo = TimerAction(
        period=5.0,
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(moveit_pkg, 'launch', 'servo.launch.py')
                )
            )
        ]
    )

    # ── Workspace visualizer — delayed 6 s ───────────────────────────────────
    workspace_visualizer = TimerAction(
        period=6.0,
        actions=[
            Node(
                package='ur5_dual_robot_teleop',
                executable='workspace_visualizer',
                name='workspace_visualizer',
                output='screen',
            )
        ]
    )

    # ── Keyboard teleop — delayed 7 s ─────────────────────────────────────────
    teleop = TimerAction(
        period=7.0,
        actions=[
            Node(
                package='ur5_dual_robot_teleop',
                executable='dual_arm_teleop_node',
                name='teleop_node',
                parameters=[{'input': 'keyboard'}],
                output='screen',
            )
        ]
    )

    return [
        demo, rviz,
        left_gripper_spawner, right_gripper_spawner, gripper_node,
        servo, workspace_visualizer, teleop,
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            "kinematics_solver",
            default_value="analytical",
            description="Kinematics solver: 'analytical' (IKFast) or 'numerical' (KDL)",
            choices=["analytical", "numerical"],
        ),
        OpaqueFunction(function=launch_setup),
    ])
