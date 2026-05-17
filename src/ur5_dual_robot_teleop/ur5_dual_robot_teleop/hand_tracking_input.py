#!/usr/bin/env python3
"""
Hand Tracking Input — Dual Hand Delta Tracking
===============================================
Close right fist → right arm starts tracking from its current position.
Close left fist  → left arm starts tracking from its current position.
Each hand operates completely independently.

Scale: full image width  = full workspace X range
       full image height = full workspace Y range
"""

import math
import threading
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool, Float64

from ur5_dual_robot_teleop.controllers.base_controller import Pose2D
from ur5_dual_robot_teleop.workspace import WORKSPACE
from ur5_dual_robot_teleop.teleop_config import CONFIG

# ─── Load parameters from config ─────────────────────────────────────────────
_cfg = CONFIG['hand_tracking_input']
INVERT_X      = _cfg['invert_x']
INVERT_Y      = _cfg['invert_y']
DEAD_ZONE     = _cfg['dead_zone']
YAW_SCALE     = _cfg['yaw_scale']
YAW_DEAD_ZONE = _cfg['yaw_dead_zone']


class HandTrackingInput:
    """
    Converts dual hand tracker topics into per-arm position offsets.

    Interface contract with TeleopNode:
      is_position_mode  — True
      is_active         — True while either fist is closed
      left_active       — True while left fist is closed
      right_active      — True while right fist is closed
      get_inputs()      — (left_offset, right_offset) Pose2D in meters
      get_gripper_command() — right gripper 0.0 (closed) → 1.0 (open)
    """

    is_position_mode = True

    def __init__(self, node: Node):
        self._node = node
        self._lock = threading.Lock()

        # ── Right hand sensor state ────────────────────────────────────────
        self._right_pose    = Pose2D()
        self._right_active  = False
        self._right_gripper = 0.0

        # ── Right hand tracking state ──────────────────────────────────────
        self._right_ref_hand   = None
        self._right_was_active = False
        self._right_prev_yaw   = None
        self._right_acc_dyaw   = 0.0

        # ── Left hand sensor state ─────────────────────────────────────────
        self._left_pose    = Pose2D()
        self._left_active  = False
        self._left_gripper = 0.0

        # ── Left hand tracking state ───────────────────────────────────────
        self._left_ref_hand   = None
        self._left_was_active = False
        self._left_prev_yaw   = None
        self._left_acc_dyaw   = 0.0

        # ── Subscriptions — right hand ─────────────────────────────────────
        node.create_subscription(
            PoseStamped, '/hand_pose/right',      self._on_right_pose,    10)
        node.create_subscription(
            Bool,        '/hand_tracker/active',  self._on_right_active,  10)
        node.create_subscription(
            Float64,     '/hand_tracker/gripper', self._on_right_gripper, 10)

        # ── Subscriptions — left hand ──────────────────────────────────────
        node.create_subscription(
            PoseStamped, '/hand_pose/left',            self._on_left_pose,    10)
        node.create_subscription(
            Bool,        '/hand_tracker/left/active',  self._on_left_active,  10)
        node.create_subscription(
            Float64,     '/hand_tracker/left/gripper', self._on_left_gripper, 10)

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self):
        scale_x = WORKSPACE.x_max - WORKSPACE.x_min
        scale_y = WORKSPACE.y_max - WORKSPACE.y_min
        self._node.get_logger().info(
            f'Hand tracking ready | dual hand | delta mode | '
            f'scale: {scale_x:.2f} m/image × {scale_y:.2f} m/image | '
            f'dead zone: {DEAD_ZONE:.2f}')

    def stop(self):
        pass

    @property
    def should_quit(self) -> bool:
        return False

    # ── Control interface ─────────────────────────────────────────────────────

    @property
    def right_active(self) -> bool:
        with self._lock:
            return self._right_active

    @property
    def left_active(self) -> bool:
        with self._lock:
            return self._left_active

    @property
    def is_active(self) -> bool:
        with self._lock:
            return self._left_active or self._right_active

    def get_inputs(self) -> tuple:
        """Returns (left_offset, right_offset) in meters."""
        return self._compute_left_offset(), self._compute_right_offset()

    def _compute_right_offset(self) -> Pose2D:
        with self._lock:
            active = self._right_active
            pose   = self._right_pose

        if not active:
            self._right_ref_hand   = None
            self._right_was_active = False
            self._right_prev_yaw   = None
            self._right_acc_dyaw   = 0.0
            return Pose2D()

        if not self._right_was_active:
            self._right_ref_hand   = pose
            self._right_was_active = True
            self._right_prev_yaw   = pose.yaw
            self._right_acc_dyaw   = 0.0
            return Pose2D()

        dx_cam = pose.x - self._right_ref_hand.x
        dy_cam = pose.y - self._right_ref_hand.y

        # Incremental yaw: accumulate per-frame steps normalized to [-π, π]
        # so the total never jumps when crossing the ±π boundary.
        step = pose.yaw - self._right_prev_yaw
        step = (step + math.pi) % (2 * math.pi) - math.pi
        self._right_prev_yaw = pose.yaw
        if abs(step) >= YAW_DEAD_ZONE:
            self._right_acc_dyaw += step

        if abs(dx_cam) < DEAD_ZONE: dx_cam = 0.0
        if abs(dy_cam) < DEAD_ZONE: dy_cam = 0.0

        dx_world   = dx_cam * (WORKSPACE.x_max - WORKSPACE.x_min)
        dy_world   = dy_cam * (WORKSPACE.y_max - WORKSPACE.y_min)
        dyaw_world = -self._right_acc_dyaw * YAW_SCALE

        if INVERT_X: dx_world = -dx_world
        if INVERT_Y: dy_world = -dy_world

        return Pose2D(x=dx_world, y=dy_world, yaw=dyaw_world)

    def _compute_left_offset(self) -> Pose2D:
        with self._lock:
            active = self._left_active
            pose   = self._left_pose

        if not active:
            self._left_ref_hand   = None
            self._left_was_active = False
            self._left_prev_yaw   = None
            self._left_acc_dyaw   = 0.0
            return Pose2D()

        if not self._left_was_active:
            self._left_ref_hand   = pose
            self._left_was_active = True
            self._left_prev_yaw   = pose.yaw
            self._left_acc_dyaw   = 0.0
            return Pose2D()

        dx_cam = pose.x - self._left_ref_hand.x
        dy_cam = pose.y - self._left_ref_hand.y

        # Incremental yaw: accumulate per-frame steps normalized to [-π, π]
        # so the total never jumps when crossing the ±π boundary.
        step = pose.yaw - self._left_prev_yaw
        step = (step + math.pi) % (2 * math.pi) - math.pi
        self._left_prev_yaw = pose.yaw
        if abs(step) >= YAW_DEAD_ZONE:
            self._left_acc_dyaw += step

        if abs(dx_cam) < DEAD_ZONE: dx_cam = 0.0
        if abs(dy_cam) < DEAD_ZONE: dy_cam = 0.0

        dx_world   = dx_cam * (WORKSPACE.x_max - WORKSPACE.x_min)
        dy_world   = dy_cam * (WORKSPACE.y_max - WORKSPACE.y_min)
        dyaw_world = self._left_acc_dyaw * YAW_SCALE

        if INVERT_X: dx_world = -dx_world
        if INVERT_Y: dy_world = -dy_world

        return Pose2D(x=dx_world, y=dy_world, yaw=dyaw_world)

    def get_gripper_command(self) -> float:
        """Returns right gripper position: 0.0 = closed, 1.0 = open."""
        with self._lock:
            return self._right_gripper

    def get_left_gripper_command(self) -> float:
        """Returns left gripper position: 0.0 = closed, 1.0 = open."""
        with self._lock:
            return self._left_gripper

    # ── ROS callbacks — right hand ────────────────────────────────────────────

    def _on_right_pose(self, msg: PoseStamped):
        q = msg.pose.orientation
        yaw = math.atan2(2.0 * q.w * q.z, 1.0 - 2.0 * q.z * q.z)
        with self._lock:
            self._right_pose = Pose2D(
                x=msg.pose.position.x,
                y=msg.pose.position.y,
                yaw=yaw,
            )

    def _on_right_active(self, msg: Bool):
        with self._lock:
            self._right_active = msg.data

    def _on_right_gripper(self, msg: Float64):
        with self._lock:
            self._right_gripper = msg.data

    # ── ROS callbacks — left hand ─────────────────────────────────────────────

    def _on_left_pose(self, msg: PoseStamped):
        q = msg.pose.orientation
        yaw = math.atan2(2.0 * q.w * q.z, 1.0 - 2.0 * q.z * q.z)
        with self._lock:
            self._left_pose = Pose2D(
                x=msg.pose.position.x,
                y=msg.pose.position.y,
                yaw=yaw,
            )

    def _on_left_active(self, msg: Bool):
        with self._lock:
            self._left_active = msg.data

    def _on_left_gripper(self, msg: Float64):
        with self._lock:
            self._left_gripper = msg.data
