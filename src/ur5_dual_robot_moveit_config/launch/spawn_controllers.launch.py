from moveit_configs_utils import MoveItConfigsBuilder
from moveit_configs_utils.launches import generate_spawn_controllers_launch


def generate_launch_description():
    moveit_config = MoveItConfigsBuilder("dual_ur5", package_name="ur5_dual_robot_moveit_config").to_moveit_configs()
    return generate_spawn_controllers_launch(moveit_config)
