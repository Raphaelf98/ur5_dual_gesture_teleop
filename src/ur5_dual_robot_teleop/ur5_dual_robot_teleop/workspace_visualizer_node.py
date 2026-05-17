#!/usr/bin/env python3
"""
Workspace Visualizer Node
==========================
Publishes the workspace plane + live EEF position markers at 10 Hz.

Add to RViz:
  Topic:  /workspace/markers
  Type:   visualization_msgs/MarkerArray

EEF dots turn red when outside the workspace boundary.
Tune workspace bounds in workspace.py.
"""

import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
import tf2_ros

from ur5_dual_robot_teleop.workspace import WORKSPACE, build_markers
from ur5_dual_robot_teleop.teleop_config import CONFIG

_EEF_SCALE   = CONFIG['visualization']['eef_sphere_scale']
_VIS_RATE    = CONFIG['visualization']['publish_rate']


class WorkspaceVisualizerNode(Node):

    def __init__(self):
        super().__init__('workspace_visualizer')

        self._pub = self.create_publisher(MarkerArray, '/workspace/markers', 10)

        # TF listener to read live EEF positions
        self._tf_buffer   = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        # Frame names — must match teleop node parameters
        self.declare_parameter('left_frame',  'left_tool0')
        self.declare_parameter('right_frame', 'right_tool0')
        self._left_frame  = self.get_parameter('left_frame').value
        self._right_frame = self.get_parameter('right_frame').value

        self._eef_logged = False          # print EEF positions once on startup
        self.create_timer(1.0 / _VIS_RATE, self._publish)

        self.get_logger().info(
            f'Workspace  frame: {WORKSPACE.frame_id}\n'
            f'  X: [{WORKSPACE.x_min:.3f}, {WORKSPACE.x_max:.3f}] m\n'
            f'  Y: [{WORKSPACE.y_min:.3f}, {WORKSPACE.y_max:.3f}] m\n'
            f'  Z:  {WORKSPACE.center_z:.3f} m')

    # ── Publishing ────────────────────────────────────────────────────────────

    def _publish(self):
        stamp = self.get_clock().now().to_msg()
        array = build_markers(WORKSPACE, stamp)

        self._add_eef_marker(array, self._left_frame,  marker_id=2, stamp=stamp)
        self._add_eef_marker(array, self._right_frame, marker_id=3, stamp=stamp)

        if not self._eef_logged:
            self._log_eef_positions()

        self._pub.publish(array)

    def _log_eef_positions(self):
        """Log actual EEF XY positions once when TF first becomes available."""
        try:
            lt = self._tf_buffer.lookup_transform(
                WORKSPACE.frame_id, self._left_frame,  rclpy.time.Time())
            rt = self._tf_buffer.lookup_transform(
                WORKSPACE.frame_id, self._right_frame, rclpy.time.Time())
        except Exception:
            return   # not ready yet
        lx, ly = lt.transform.translation.x, lt.transform.translation.y
        rx, ry = rt.transform.translation.x, rt.transform.translation.y
        mid_x = (lx + rx) / 2
        mid_y = (ly + ry) / 2
        self.get_logger().info(
            f'\n--- EEF initial positions (world frame) ---\n'
            f'  left_tool0  x={lx:.3f}  y={ly:.3f}\n'
            f'  right_tool0 x={rx:.3f}  y={ry:.3f}\n'
            f'  midpoint    x={mid_x:.3f}  y={mid_y:.3f}  ← suggested CENTER_X / CENTER_Y\n'
            f'-------------------------------------------'
        )
        self._eef_logged = True

    def _add_eef_marker(self, array: MarkerArray, frame: str, marker_id: int, stamp):
        """
        Add a sphere at the EEF position projected onto the workspace plane.
        Green = inside workspace, red = outside.
        """
        try:
            tf = self._tf_buffer.lookup_transform(
                WORKSPACE.frame_id, frame, rclpy.time.Time())
        except Exception:
            return   # TF not yet available — skip silently

        x = tf.transform.translation.x
        y = tf.transform.translation.y
        inside = WORKSPACE.contains(x, y)

        m = Marker()
        m.header.frame_id = WORKSPACE.frame_id
        m.header.stamp    = stamp
        m.ns              = 'workspace'
        m.id              = marker_id
        m.type            = Marker.SPHERE
        m.action          = Marker.ADD
        m.pose.position.x = x
        m.pose.position.y = y
        m.pose.position.z = WORKSPACE.center_z   # projected onto the plane
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = _EEF_SCALE
        m.color = (ColorRGBA(r=0.1, g=0.9, b=0.1, a=1.0) if inside
                   else ColorRGBA(r=0.9, g=0.1, b=0.1, a=1.0))
        array.markers.append(m)


def main(args=None):
    rclpy.init(args=args)
    node = WorkspaceVisualizerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
