#!/usr/bin/env python3
"""
Position Controller
====================
Waypoint-based controller using MoveIt MoveGroup Cartesian path planning.
Computes a full collision-free trajectory to each target pose.

This controller works differently from the others — instead of returning
a continuous velocity, it returns zero velocity and triggers a MoveGroup
plan+execute via a callback. The node handles the actual execution.

Best for: precise point-to-point motion, pick and place, known waypoints.
Worst for: real-time following, hand tracking (too slow, ~100ms latency).

Note: Requires MoveGroup to be running. The node must provide
      the execute_callback to trigger planning+execution.
"""

import math
from .base_controller import BaseController, Pose2D, Twist2D
from typing import Callable, Optional


class PositionController(BaseController):
    """
    Position-based controller — delegates to MoveGroup for planning.

    When a new target is set and it differs from the current target
    by more than the threshold, it triggers plan+execute via callback.

    The controller returns zero velocity — motion is handled entirely
    by the JointTrajectoryController through MoveGroup execution.
    """

    def __init__(
        self,
        position_threshold: float = 0.01,   # m   — min change to trigger replan
        angle_threshold:    float = 0.05,   # rad — min angle change to trigger replan
        execute_callback: Optional[Callable[[Pose2D], bool]] = None,
    ):
        """
        Args:
            position_threshold: Minimum XY displacement to trigger a new plan
            angle_threshold:    Minimum yaw change to trigger a new plan
            execute_callback:   Function called with target Pose2D to plan+execute.
                                Should return True if execution succeeded.
                                Provided by the ROS node.
        """
        self.position_threshold = position_threshold
        self.angle_threshold    = angle_threshold
        self.execute_callback   = execute_callback

        self._last_target: Optional[Pose2D] = None
        self._is_executing = False

    def set_execute_callback(self, callback: Callable[[Pose2D], bool]):
        """Set the MoveGroup execute callback from the ROS node."""
        self.execute_callback = callback

    def reset(self):
        self._last_target  = None
        self._is_executing = False

    def _target_changed(self, target: Pose2D) -> bool:
        """Check if target has changed enough to warrant replanning."""
        if self._last_target is None:
            return True
        dx = abs(target.x - self._last_target.x)
        dy = abs(target.y - self._last_target.y)
        dyaw = abs(target.yaw - self._last_target.yaw)
        return (dx > self.position_threshold or
                dy > self.position_threshold or
                dyaw > self.angle_threshold)

    def compute_velocity(
        self,
        current: Pose2D,
        target: Pose2D,
        dt: float
    ) -> Twist2D:
        """
        Trigger MoveGroup plan+execute if target changed.
        Always returns zero velocity — motion handled by MoveGroup.
        """
        if (not self._is_executing and
                self._target_changed(target) and
                self.execute_callback is not None):

            self._is_executing = True
            self._last_target  = Pose2D(x=target.x, y=target.y, yaw=target.yaw)

            # Trigger planning in a non-blocking way
            # The callback should run in a separate thread
            success = self.execute_callback(target)
            self._is_executing = False

        # Position controller doesn't use Servo — return zero velocity
        return Twist2D()