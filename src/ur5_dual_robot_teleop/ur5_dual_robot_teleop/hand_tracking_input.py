#!/usr/bin/env python3
"""
Hand Tracking Input Handler
=============================
Subscribes to /hand_pose/right and converts normalized hand position
to robot velocity commands via workspace mapping.

Replaces keyboard_input.py — same get_input() -> Pose2D interface.
Drop-in replacement, no changes needed in teleop_node.py.

Workspace mapping:
    hand x (normalized 0-1) → robot velocity vx
    hand y (normalized 0-1) → robot velocity vy

The hand position is mapped relative to a "neutral zone" at center (0.5, 0.5).
Moving hand away from center produces proportional velocity.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool
import threading

from ur5_dual_robot_teleop.controllers.base_controller import Pose2D


# ─── Workspace mapping parameters ───────────────────────────────────────────
DEAD_ZONE     = 0.08    # normalized — hand movement inside this radius = no motion
MAX_ZONE      = 0.35    # normalized — hand movement beyond this = max velocity
MAX_VELOCITY  = 0.3     # m/s — maximum robot velocity


class HandTrackingInput:
    """
    Hand tracking input handler.

    Converts normalized hand pose from /hand_pose/right
    to Pose2D velocity setpoints for the direct velocity controller.

    The hand position relative to center (0.5, 0.5) is mapped to velocity:
    - Inside dead zone: zero velocity (stable holding position)
    - Outside dead zone: proportional velocity
    - Beyond max zone: maximum velocity (clamped)

    Usage:
        input_handler = HandTrackingInput(node)
        input_handler.start()
        pose = input_handler.get_input()  # same interface as KeyboardInput
    """

    def __init__(self, node: Node):
        self._node   = node
        self._lock   = threading.Lock()
        self._active = False
        self._pose   = Pose2D()   # current smoothed hand pose (normalized)
        self._running = True

        # Subscribers
        self._pose_sub = node.create_subscription(
            PoseStamped, '/hand_pose/right',
            self._pose_callback, 10)
        self._active_sub = node.create_subscription(
            Bool, '/hand_tracker/active',
            self._active_callback, 10)

    def start(self):
        """No background thread needed — uses ROS callbacks."""
        self._node.get_logger().info('Hand tracking input ready.')
        self._node.get_logger().info(
            f'Dead zone: {DEAD_ZONE:.2f}  Max zone: {MAX_ZONE:.2f}  '
            f'Max vel: {MAX_VELOCITY:.2f} m/s')

    def stop(self):
        self._running = False

    @property
    def should_quit(self) -> bool:
        return not self._running

    def get_input(self) -> Pose2D:
        """
        Convert hand position to velocity setpoint.
        Returns Pose2D where x/y are velocity commands [-MAX_VELOCITY, MAX_VELOCITY].
        """
        with self._lock:
            if not self._active:
                return Pose2D()  # zero velocity when hand not tracked

            # Hand position relative to center
            dx = self._pose.x - 0.5   # range [-0.5, 0.5]
            dy = self._pose.y - 0.5

        # Apply dead zone
        vx = self._apply_dead_zone(dx)
        vy = self._apply_dead_zone(dy)

        # Scale to velocity
        scale = MAX_VELOCITY / (MAX_ZONE - DEAD_ZONE)
        vx = max(-MAX_VELOCITY, min(MAX_VELOCITY, vx * scale))
        vy = max(-MAX_VELOCITY, min(MAX_VELOCITY, vy * scale))

        return Pose2D(x=vx, y=vy, yaw=0.0)

    def _apply_dead_zone(self, value: float) -> float:
        """Apply dead zone to a single axis value."""
        if abs(value) < DEAD_ZONE:
            return 0.0
        sign = 1.0 if value > 0 else -1.0
        return sign * (abs(value) - DEAD_ZONE)

    def _pose_callback(self, msg: PoseStamped):
        with self._lock:
            self._pose = Pose2D(
                x=msg.pose.position.x,
                y=msg.pose.position.y,
                yaw=0.0
            )

    def _active_callback(self, msg: Bool):
        with self._lock:
            self._active = msg.data
            if not msg.data:
                self._pose = Pose2D()  # reset when hand lost