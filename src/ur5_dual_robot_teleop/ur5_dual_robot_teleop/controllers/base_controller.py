#!/usr/bin/env python3
"""
Base Controller Interface
==========================
All controllers must implement this interface.
Controllers are pure Python — no ROS dependencies.
Input:  current_pose, target_pose, dt
Output: Twist (linear x/y/z, angular x/y/z)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class Pose2D:
    """Simplified pose for XY plane control."""
    x:   float = 0.0
    y:   float = 0.0
    yaw: float = 0.0   # wrist rotation around Z


@dataclass
class Twist2D:
    """Velocity command for XY plane + wrist."""
    vx:  float = 0.0   # linear X velocity  [m/s]
    vy:  float = 0.0   # linear Y velocity  [m/s]
    wz:  float = 0.0   # angular Z velocity [rad/s]


class BaseController(ABC):
    """
    Abstract base class for all robot controllers.

    Subclasses implement compute_velocity() which takes the current
    and target pose and returns a velocity command.

    The ROS node calls this every control cycle and handles
    all publishing/subscribing — controllers contain only math.
    """

    @abstractmethod
    def compute_velocity(
        self,
        current: Pose2D,
        target: Pose2D,
        dt: float
    ) -> Twist2D:
        """
        Compute velocity command given current and target pose.

        Args:
            current: Current EEF pose (from TF or dead reckoning)
            target:  Desired EEF pose (from keyboard, hand tracker, etc.)
            dt:      Time since last call [seconds]

        Returns:
            Twist2D: Velocity command to publish to Servo
        """
        raise NotImplementedError

    def reset(self):
        """Reset internal state (e.g. integral, previous error)."""
        pass

    @property
    def name(self) -> str:
        return self.__class__.__name__