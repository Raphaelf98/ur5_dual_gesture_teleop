#!/usr/bin/env python3
"""
Direct Velocity Controller
============================
Passes target velocity directly as output — no feedback, no error computation.
The 'target' pose is treated as a velocity setpoint, not a position goal.

Best for: keyboard teleoperation, joystick control.
Worst for: precise positioning, hand tracking (drifts over time).
"""

from .base_controller import BaseController, Pose2D, Twist2D


class DirectVelocityController(BaseController):
    """
    Direct pass-through controller.

    The input 'target' is interpreted as a velocity command directly:
        target.x   → vx
        target.y   → vy
        target.yaw → wz

    No position feedback is used. Whatever velocity the input
    device provides is sent straight to Servo.
    """

    def __init__(
        self,
        max_linear:  float = 0.3,   # m/s   — clamp linear velocity
        max_angular: float = 0.8,   # rad/s — clamp angular velocity
    ):
        self.max_linear  = max_linear
        self.max_angular = max_angular

    def compute_velocity(
        self,
        current: Pose2D,
        target: Pose2D,
        dt: float
    ) -> Twist2D:
        """
        current is ignored — target IS the velocity command.
        """
        vx = float(max(-self.max_linear,  min(self.max_linear,  target.x)))
        vy = float(max(-self.max_linear,  min(self.max_linear,  target.y)))
        wz = float(max(-self.max_angular, min(self.max_angular, target.yaw)))
        return Twist2D(vx=vx, vy=vy, wz=wz)