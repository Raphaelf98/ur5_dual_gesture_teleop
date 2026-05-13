#!/usr/bin/env python3
"""
Hand Tracking Input Handler — Phase 2
========================================
Subscribes to hand tracker topics and converts to robot commands.

Gesture → Robot behavior:
  OPEN   → zero velocity (robot stops)
  FOLLOW → velocity from hand position (robot follows)
  GRIP   → velocity from hand position + gripper open command

Dead zone around neutral position prevents drift when hand is still.
"""

import threading
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool, Float64, String

from ur5_dual_robot_teleop.controllers.base_controller import Pose2D


# ─── Workspace mapping ───────────────────────────────────────────────────────
NEUTRAL_X    = 0.5     # normalized center X
NEUTRAL_Y    = 0.5     # normalized center Y
DEAD_ZONE    = 0.08    # hand movement inside this radius = no motion
MAX_ZONE     = 0.35    # hand movement beyond this = max velocity
MAX_VELOCITY = 0.25    # m/s — maximum robot velocity


class HandTrackingInput:
    """
    Hand tracking input handler with gesture support.

    Converts /hand_pose/right + /hand_tracker/gesture to velocity commands.
    Only outputs velocity when gesture is FOLLOW or GRIP.
    """

    def __init__(self, node: Node):
        self._node    = node
        self._lock    = threading.Lock()
        self._running = True

        # Current state
        self._pose    = Pose2D()
        self._gesture = 'OPEN'
        self._gripper = 0.0   # 0.0 = closed, 1.0 = open

        # Subscribers
        node.create_subscription(
            PoseStamped, '/hand_pose/right',
            self._pose_callback, 10)
        node.create_subscription(
            String, '/hand_tracker/gesture',
            self._gesture_callback, 10)
        node.create_subscription(
            Float64, '/hand_tracker/gripper',
            self._gripper_callback, 10)

    def start(self):
        self._node.get_logger().info('Hand tracking input ready.')
        self._node.get_logger().info(
            f'OPEN=stop | FOLLOW=track | GRIP=track+gripper')
        self._node.get_logger().info(
            f'Dead zone: {DEAD_ZONE:.2f}  Max vel: {MAX_VELOCITY:.2f} m/s')

    def stop(self):
        self._running = False

    @property
    def should_quit(self) -> bool:
        return not self._running

    def get_input(self) -> Pose2D:
        """
        Returns velocity setpoint based on hand position and gesture.
        Returns zero if gesture is OPEN or hand not detected.
        """
        with self._lock:
            gesture = self._gesture
            pose    = self._pose

        # Only move if fist is closed
        if gesture == 'OPEN':
            return Pose2D()

        # Hand displacement from neutral
        dx = pose.x - NEUTRAL_X
        dy = pose.y - NEUTRAL_Y

        # Apply dead zone and scale
        vx = self._scale(dx)
        vy = self._scale(dy)

        return Pose2D(x=vx, y=vy, yaw=0.0)

    def get_gripper_command(self) -> float:
        """Returns gripper command: 0.0=closed, 1.0=open."""
        with self._lock:
            return self._gripper

    def get_gesture(self) -> str:
        """Returns current gesture: OPEN, FOLLOW, GRIP."""
        with self._lock:
            return self._gesture

    def _scale(self, value: float) -> float:
        """Apply dead zone and scale to velocity."""
        if abs(value) < DEAD_ZONE:
            return 0.0
        sign  = 1.0 if value > 0 else -1.0
        scale = MAX_VELOCITY / (MAX_ZONE - DEAD_ZONE)
        return max(-MAX_VELOCITY,
               min(MAX_VELOCITY,
                   sign * (abs(value) - DEAD_ZONE) * scale))

    def _pose_callback(self, msg: PoseStamped):
        with self._lock:
            self._pose = Pose2D(
                x=msg.pose.position.x,
                y=msg.pose.position.y,
                yaw=0.0
            )

    def _gesture_callback(self, msg: String):
        with self._lock:
            self._gesture = msg.data

    def _gripper_callback(self, msg: Float64):
        with self._lock:
            self._gripper = msg.data