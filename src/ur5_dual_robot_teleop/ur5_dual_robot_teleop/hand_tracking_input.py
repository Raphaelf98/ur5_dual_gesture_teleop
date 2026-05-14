#!/usr/bin/env python3
"""
Hand Tracking Input — Delta Tracking
======================================
Close fist to start tracking.  The hand position at fist-close is the origin.
Moving the hand from that origin drives a proportional delta on the robot EEF.

Scale: full image width  = full workspace X range
       full image height = full workspace Y range

Control model:
  Fist closes  →  snapshot ref_hand (origin in camera space)
  Each frame   →  delta = (current_hand - ref_hand) × workspace_scale
  Fist opens   →  delta = (0, 0) → zero velocity, robot holds position

Tune INVERT_X / INVERT_Y if the robot moves in the wrong direction.
"""

import threading
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool, Float64

from ur5_dual_robot_teleop.controllers.base_controller import Pose2D
from ur5_dual_robot_teleop.workspace import WORKSPACE


# ─── Tuning ───────────────────────────────────────────────────────────────────
INVERT_X  = True  # mirror left/right  (set True if left/right is inverted)
INVERT_Y  = False  # flip up/down       (set True if vertical axis moves wrong way)
DEAD_ZONE = 0.02   # minimum hand displacement (fraction of image) before tracking


class HandTrackingInput:
    """
    Converts hand tracker topics into position offsets for the P controller.

    Interface contract with TeleopNode:
      is_position_mode  — True: get_input() returns a position offset, not velocity
      is_active         — True while fist is closed
      get_input()       — Pose2D offset from the EEF position at fist-close (meters)
      get_gripper_command() — gripper value 0.0 (closed) → 1.0 (open)
    """

    is_position_mode = True

    def __init__(self, node: Node):
        self._node    = node
        self._lock    = threading.Lock()

        # ── Sensor state (written by ROS callbacks) ────────────────────────
        self._pose    = Pose2D()
        self._active  = False
        self._gripper = 0.0

        # ── Tracking state (single control thread) ─────────────────────────
        self._ref_hand   = None   # hand pose at moment of fist-close
        self._was_active = False

        # ── Subscriptions ─────────────────────────────────────────────────
        node.create_subscription(
            PoseStamped, '/hand_pose/right',      self._on_pose,    10)
        node.create_subscription(
            Bool,        '/hand_tracker/active',  self._on_active,  10)
        node.create_subscription(
            Float64,     '/hand_tracker/gripper', self._on_gripper, 10)

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self):
        scale_x = WORKSPACE.x_max - WORKSPACE.x_min
        scale_y = WORKSPACE.y_max - WORKSPACE.y_min
        self._node.get_logger().info(
            f'Hand tracking ready | delta mode | '
            f'scale: {scale_x:.2f} m/image  ×  {scale_y:.2f} m/image  '
            f'dead zone: {DEAD_ZONE:.2f}')

    def stop(self):
        pass

    @property
    def should_quit(self) -> bool:
        return False

    # ── Control interface ─────────────────────────────────────────────────────

    @property
    def is_active(self) -> bool:
        with self._lock:
            return self._active

    def get_inputs(self) -> tuple:
        """
        Returns (left_offset, right_offset) in meters.

        Currently both arms are driven by the right hand (same offset).
        When dual hand tracking is added, left and right will diverge.
        """
        offset = self._compute_offset()
        return offset, offset

    def _compute_offset(self) -> Pose2D:
        """Compute position offset from right hand for one arm."""
        with self._lock:
            active = self._active
            pose   = self._pose

        if not active:
            self._ref_hand   = None
            self._was_active = False
            return Pose2D()

        if not self._was_active:              # rising edge: fist just closed
            self._ref_hand   = pose
            self._was_active = True
            return Pose2D()

        dx_cam = pose.x - self._ref_hand.x
        dy_cam = pose.y - self._ref_hand.y

        # Dead zone in camera space — suppresses micro-drift at rest
        if abs(dx_cam) < DEAD_ZONE: dx_cam = 0.0
        if abs(dy_cam) < DEAD_ZONE: dy_cam = 0.0

        # Scale: 1 full image width = full workspace range
        dx_world = dx_cam * (WORKSPACE.x_max - WORKSPACE.x_min)
        dy_world = dy_cam * (WORKSPACE.y_max - WORKSPACE.y_min)

        if INVERT_X: dx_world = -dx_world
        if INVERT_Y: dy_world = -dy_world

        return Pose2D(x=dx_world, y=dy_world, yaw=0.0)

    def get_gripper_command(self) -> float:
        """Returns gripper position: 0.0 = closed, 1.0 = open."""
        with self._lock:
            return self._gripper

    # ── ROS callbacks ─────────────────────────────────────────────────────────

    def _on_pose(self, msg: PoseStamped):
        with self._lock:
            self._pose = Pose2D(
                x   = msg.pose.position.x,
                y   = msg.pose.position.y,
                yaw = msg.pose.position.z,  # yaw packed into z by hand_tracker_node
            )

    def _on_active(self, msg: Bool):
        with self._lock:
            self._active = msg.data

    def _on_gripper(self, msg: Float64):
        with self._lock:
            self._gripper = msg.data
