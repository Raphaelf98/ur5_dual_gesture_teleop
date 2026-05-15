"""
Workspace Bounds
=================
Defines the rectangular XY plane in which both robot EEFs will operate.
Shared between the visualizer and the control layer.

Tune the constants at the top of this file to match your robot setup.
All coordinates are in the 'world' frame.
"""

from dataclasses import dataclass
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA


# ─── Workspace definition — tune these to match your setup ───────────────────
FRAME_ID  = 'world'
CENTER_X  = -0.1    # m — workspace center in X
CENTER_Y  = 0.5    # m — workspace center in Y
CENTER_Z  = 0.60   # m — TCP operating height
WIDTH     = 1.5    # m — extent in X direction
DEPTH     = 0.6    # m — extent in Y direction


@dataclass
class WorkspaceBounds:
    """Rectangular workspace plane in the world frame."""
    frame_id: str   = FRAME_ID
    center_x: float = CENTER_X
    center_y: float = CENTER_Y
    center_z: float = CENTER_Z
    width:    float = WIDTH
    depth:    float = DEPTH

    # ── Derived bounds ────────────────────────────────────────────────────

    @property
    def x_min(self) -> float: return self.center_x - self.width / 2

    @property
    def x_max(self) -> float: return self.center_x + self.width / 2

    @property
    def y_min(self) -> float: return self.center_y - self.depth / 2

    @property
    def y_max(self) -> float: return self.center_y + self.depth / 2

    # ── Utility ───────────────────────────────────────────────────────────

    def contains(self, x: float, y: float) -> bool:
        """True if (x, y) is inside the workspace."""
        return self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max

    def clamp(self, x: float, y: float) -> tuple:
        """Clamp (x, y) to the workspace boundary."""
        return (
            max(self.x_min, min(self.x_max, x)),
            max(self.y_min, min(self.y_max, y)),
        )


# ─── Shared instance used by both visualizer and controller ──────────────────
WORKSPACE = WorkspaceBounds()


# ─── RViz marker builder ──────────────────────────────────────────────────────

def build_markers(bounds: WorkspaceBounds, stamp) -> MarkerArray:
    """
    Build a MarkerArray visualizing the workspace plane in RViz.

    Produces two markers:
      id=0  semi-transparent filled rectangle (the plane itself)
      id=1  solid border (LINE_STRIP around the perimeter)
    """
    array = MarkerArray()

    # ── Fill: semi-transparent blue plane ─────────────────────────────────
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
    fill.scale.z         = 0.003       # thin slab — appears as a plane
    fill.color           = ColorRGBA(r=0.2, g=0.6, b=1.0, a=0.25)
    array.markers.append(fill)

    # ── Border: solid outline around the rectangle ─────────────────────────
    border = Marker()
    border.header.frame_id = bounds.frame_id
    border.header.stamp    = stamp
    border.ns              = 'workspace'
    border.id              = 1
    border.type            = Marker.LINE_STRIP
    border.action          = Marker.ADD
    border.pose.orientation.w = 1.0
    border.scale.x         = 0.01     # line width in meters
    border.color           = ColorRGBA(r=0.2, g=0.6, b=1.0, a=0.9)
    z = bounds.center_z
    for x, y in [
        (bounds.x_min, bounds.y_min),
        (bounds.x_max, bounds.y_min),
        (bounds.x_max, bounds.y_max),
        (bounds.x_min, bounds.y_max),
        (bounds.x_min, bounds.y_min),  # close the loop
    ]:
        p = Point(x=x, y=y, z=z)
        border.points.append(p)
    array.markers.append(border)

    return array
