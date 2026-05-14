#!/usr/bin/env python3
"""
Keyboard Input Handler
=======================
Keyboard teleoperation with continuous gripper control.
Designed to match the hand-tracking interface so both inputs
can be swapped without changing the teleop node.

Keys:
    ↑ / ↓       — Move EEF in +X / -X
    ← / →       — Move EEF in +Y / -Y
    A / D       — Rotate wrist CCW / CW
    J / L       — Open / close gripper (hold to ramp continuously)
    ESC         — Quit
"""

import threading
from ur5_dual_robot_teleop.controllers.base_controller import Pose2D

try:
    from pynput import keyboard
except ImportError:
    raise ImportError("Install pynput: pip install pynput")


# ─── Gripper ramping ─────────────────────────────────────────────────────────
GRIPPER_SPEED = 0.02   # gripper units per control tick (0→1 in ~1 s at 50 Hz)


class KeyboardInput:
    """
    Keyboard input handler.

    Returns Pose2D velocity setpoints from arrow keys and A/D,
    and a continuous gripper value [0.0, 1.0] ramped by W/S.
    """

    def __init__(
        self,
        linear_speed:  float = 0.3,   # m/s
        angular_speed: float = 0.5,   # rad/s
    ):
        self.linear_speed  = linear_speed
        self.angular_speed = angular_speed

        self._active_keys: set = set()
        self._lock    = threading.Lock()
        self._running = True
        self._listener = None

        self._gripper = 0.0   # current gripper position [0.0=closed, 1.0=open]

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self):
        self._listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
        )
        self._listener.start()

    def stop(self):
        self._running = False
        if self._listener:
            self._listener.stop()

    @property
    def should_quit(self) -> bool:
        return not self._running

    # ── Control interface ─────────────────────────────────────────────────────

    def get_input(self) -> Pose2D:
        """Returns velocity setpoint from currently held movement keys."""
        with self._lock:
            keys = set(self._active_keys)

        x = y = yaw = 0.0
        for key in keys:
            if   key == keyboard.Key.up:                          x   += self.linear_speed
            elif key == keyboard.Key.down:                        x   -= self.linear_speed
            elif key == keyboard.Key.left:                        y   += self.linear_speed
            elif key == keyboard.Key.right:                       y   -= self.linear_speed
            elif hasattr(key, 'char') and key.char in ('a', 'A'): yaw += self.angular_speed
            elif hasattr(key, 'char') and key.char in ('d', 'D'): yaw -= self.angular_speed

        return Pose2D(x=x, y=y, yaw=yaw)

    def get_gripper_command(self) -> float:
        """
        Returns current gripper value [0.0=closed, 1.0=open].
        Ramps toward open while W is held, toward closed while S is held.
        Called once per control tick by the teleop node.
        """
        with self._lock:
            keys = set(self._active_keys)

        w_held = any(hasattr(k, 'char') and k.char in ('j', 'J') for k in keys)
        s_held = any(hasattr(k, 'char') and k.char in ('l', 'L') for k in keys)

        if w_held:
            self._gripper = min(1.0, self._gripper + GRIPPER_SPEED)
        elif s_held:
            self._gripper = max(0.0, self._gripper - GRIPPER_SPEED)

        return self._gripper

    # ── Keyboard callbacks ────────────────────────────────────────────────────

    def _on_press(self, key):
        with self._lock:
            self._active_keys.add(key)
        if key == keyboard.Key.esc:
            self._running = False
            return False   # stop listener

    def _on_release(self, key):
        with self._lock:
            self._active_keys.discard(key)
