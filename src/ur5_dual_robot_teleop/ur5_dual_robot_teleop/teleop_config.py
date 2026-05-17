import os
import yaml
from ament_index_python.packages import get_package_share_directory


def _load() -> dict:
    pkg = get_package_share_directory('ur5_dual_robot_teleop')
    path = os.path.join(pkg, 'config', 'teleop_params.yaml')
    with open(path) as f:
        return yaml.safe_load(f)


CONFIG = _load()
