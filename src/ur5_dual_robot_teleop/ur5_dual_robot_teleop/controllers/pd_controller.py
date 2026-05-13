#!/usr/bin/env python3
"""
PD Controller
==============
Proportional-Derivative controller for smooth pose following.
Uses actual EEF pose from TF as feedback.

Best for: hand tracking, following a moving target smoothly.
Worst for: precise point-to-point (use position controller instead).

Tuning guide:
    Kp too low  → robot is sluggish, lags behind target
    Kp too high → robot oscillates around target
    Kd too low  → oscillations not damped
    Kd too high → robot feels "sticky", resists motion
"""

import math
from .base_controller import BaseController, Pose2D, Twist2D


class PDController(BaseController):
    """
    PD controller with independent gains per axis.

    velocity = Kp * error + Kd * (error - prev_error) / dt

    The P term attracts the robot toward the target (spring).
    The D term damps oscillations (damper).
    Together they create smooth, natural following behavior.
    """

    def __init__(
        self,
        kp_linear:   float = 2.0,   # proportional gain for X/Y
        kd_linear:   float = 0.3,   # derivative gain for X/Y
        kp_angular:  float = 1.5,   # proportional gain for yaw
        kd_angular:  float = 0.2,   # derivative gain for yaw
        max_linear:  float = 0.4,   # m/s   — output clamp
        max_angular: float = 0.8,   # rad/s — output clamp
    ):
        self.kp_linear   = kp_linear
        self.kd_linear   = kd_linear
        self.kp_angular  = kp_angular
        self.kd_angular  = kd_angular
        self.max_linear  = max_linear
        self.max_angular = max_angular

        # Previous errors for derivative term
        self._prev_ex  = 0.0
        self._prev_ey  = 0.0
        self._prev_eyaw = 0.0

    def reset(self):
        """Reset derivative state — call when switching targets."""
        self._prev_ex   = 0.0
        self._prev_ey   = 0.0
        self._prev_eyaw = 0.0

    def _angle_diff(self, target: float, current: float) -> float:
        """Shortest angular distance, wrapped to [-π, π]."""
        diff = target - current
        while diff >  math.pi: diff -= 2 * math.pi
        while diff < -math.pi: diff += 2 * math.pi
        return diff

    def compute_velocity(
        self,
        current: Pose2D,
        target: Pose2D,
        dt: float
    ) -> Twist2D:
        """
        Compute PD velocity command.

        Args:
            current: Actual EEF pose from TF lookup
            target:  Desired EEF pose (from hand tracker, waypoint, etc.)
            dt:      Time since last call [seconds]
        """
        if dt <= 0.0:
            return Twist2D()

        # Position errors
        ex  = target.x - current.x
        ey  = target.y - current.y
        eyaw = self._angle_diff(target.yaw, current.yaw)

        # Derivative terms
        dex   = (ex   - self._prev_ex)   / dt
        dey   = (ey   - self._prev_ey)   / dt
        deyaw = (eyaw - self._prev_eyaw) / dt

        # PD output
        vx  = self.kp_linear  * ex   + self.kd_linear  * dex
        vy  = self.kp_linear  * ey   + self.kd_linear  * dey
        wz  = self.kp_angular * eyaw + self.kd_angular * deyaw

        # Clamp outputs
        vx  = max(-self.max_linear,  min(self.max_linear,  vx))
        vy  = max(-self.max_linear,  min(self.max_linear,  vy))
        wz  = max(-self.max_angular, min(self.max_angular, wz))

        # Store for next derivative computation
        self._prev_ex   = ex
        self._prev_ey   = ey
        self._prev_eyaw = eyaw

        return Twist2D(vx=float(vx), vy=float(vy), wz=float(wz))