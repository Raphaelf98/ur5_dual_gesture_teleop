#!/usr/bin/env python3
"""
Keyboard Input Handler
=======================
Reads keyboard input and produces target poses or velocity setpoints.
Completely separated from the controller — swap this for hand_tracking_input.py
when ready for hand tracking.

In DirectVelocity mode: target.x/y/yaw = velocity setpoints
In PD/Position mode:    target.x/y/yaw = incremental position updates
"""

import threading
from ur5_dual_robot_teleop.controllers.base_controller import Pose2D
try:
    from pynput import keyboard
except ImportError:
    raise ImportError("Install pynput: pip install pynput")


class KeyboardInput:
    """
    Keyboard input handler.

    Produces a Pose2D that represents either:
    - A velocity setpoint (for DirectVelocityController)
    - A position delta to apply to current target (for PD/Position controllers)

    Keys:
        ↑ / ↓   — X axis
        ← / →   — Y axis
        Q / E   — Wrist yaw
        SPACE   — Stop / zero all
        ESC     — Signal quit
    """

    def __init__(
        self,
        linear_speed:  float = 0.3,    # m/s or m/step
        angular_speed: float = 0.5,    # rad/s or rad/step
    ):
        self.linear_speed  = linear_speed
        self.angular_speed = angular_speed

        self._active_keys: set = set()
        self._lock    = threading.Lock()
        self._running = True
        self._listener = None

    def start(self):
        """Start keyboard listener in background thread."""
        self._listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release
        )
        self._listener.start()

    def stop(self):
        """Stop keyboard listener."""
        self._running = False
        if self._listener:
            self._listener.stop()

    @property
    def should_quit(self) -> bool:
        return not self._running

    def get_input(self) -> Pose2D:
        """
        Get current input as a Pose2D.
        Returns zero if no keys pressed.
        """
        with self._lock:
            keys = set(self._active_keys)

        x   = 0.0
        y   = 0.0
        yaw = 0.0

        for key in keys:
            if key == keyboard.Key.up:
                x += self.linear_speed
            elif key == keyboard.Key.down:
                x -= self.linear_speed
            elif key == keyboard.Key.left:
                y += self.linear_speed
            elif key == keyboard.Key.right:
                y -= self.linear_speed
            elif hasattr(key, 'char') and key.char in ('q', 'Q'):
                yaw += self.angular_speed
            elif hasattr(key, 'char') and key.char in ('e', 'E'):
                yaw -= self.angular_speed

        return Pose2D(x=x, y=y, yaw=yaw)

    def _on_press(self, key):
        with self._lock:
            self._active_keys.add(key)
        if key == keyboard.Key.esc:
            self._running = False
            return False

    def _on_release(self, key):
        with self._lock:
            self._active_keys.discard(key)