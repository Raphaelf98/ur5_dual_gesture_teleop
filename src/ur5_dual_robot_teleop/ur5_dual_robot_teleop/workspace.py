"""
Workspace Bounds
=================
Defines the rectangular XY plane in which both robot EEFs will operate.
Shared between the visualizer and the control layer.

Parameters are loaded from config/teleop_params.yaml.
All coordinates are in the 'world' frame.
"""

from dataclasses import dataclass
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA

from ur5_dual_robot_teleop.teleop_config import CONFIG


@dataclass
class WorkspaceBounds:
    """Rectangular workspace plane in the world frame."""
    frame_id: str   = 'world'
    center_x: float = 0.0
    center_y: float = 0.0
    center_z: float = 0.3
    width:    float = 1.0
    depth:    float = 0.6

    @property
    def x_min(self) -> float: return self.center_x - self.width / 2

    @property
    def x_max(self) -> float: return self.center_x + self.width / 2

    @property
    def y_min(self) -> float: return self.center_y - self.depth / 2

    @property
    def y_max(self) -> float: return self.center_y + self.depth / 2

    def contains(self, x: float, y: float) -> bool:
        return self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max

    def clamp(self, x: float, y: float) -> tuple:
        return (
            max(self.x_min, min(self.x_max, x)),
            max(self.y_min, min(self.y_max, y)),
        )


def _build_workspace() -> WorkspaceBounds:
    cfg = CONFIG['workspace']
    return WorkspaceBounds(
        frame_id=cfg['frame_id'],
        center_x=cfg['center_x'],
        center_y=cfg['center_y'],
        center_z=cfg['center_z'],
        width=cfg['width'],
        depth=cfg['depth'],
    )


WORKSPACE = _build_workspace()


def build_markers(bounds: WorkspaceBounds, stamp) -> MarkerArray:
    """Build a MarkerArray visualizing the workspace plane in RViz."""
    array = MarkerArray()

    fill = Marker()
    fill.header.frame_id = bounds.frame_id
    fill.header.stamp    = stamp
    fill.ns              = 'workspace'
    fill.id              = 0
    fill.type            = Marker.CUBE
    fill.action          = Marker.ADD
    fill.pose.position.x = bounds.center_x
    fill.pose.position.y = bounds.center_y
    fill.pose.position.z = bounds.center_z
    fill.pose.orientation.w = 1.0
    fill.scale.x         = bounds.width
    fill.scale.y         = bounds.depth
    fill.scale.z         = 0.003
    fill.color           = ColorRGBA(r=0.2, g=0.6, b=1.0, a=0.25)
    array.markers.append(fill)

    border = Marker()
    border.header.frame_id = bounds.frame_id
    border.header.stamp    = stamp
    border.ns              = 'workspace'
    border.id              = 1
    border.type            = Marker.LINE_STRIP
    border.action          = Marker.ADD
    border.pose.orientation.w = 1.0
    border.scale.x         = 0.01
    border.color           = ColorRGBA(r=0.2, g=0.6, b=1.0, a=0.9)
    z = bounds.center_z
    for x, y in [
        (bounds.x_min, bounds.y_min),
        (bounds.x_max, bounds.y_min),
        (bounds.x_max, bounds.y_max),
        (bounds.x_min, bounds.y_max),
        (bounds.x_min, bounds.y_min),
    ]:
        border.points.append(Point(x=x, y=y, z=z))
    array.markers.append(border)

    return array
