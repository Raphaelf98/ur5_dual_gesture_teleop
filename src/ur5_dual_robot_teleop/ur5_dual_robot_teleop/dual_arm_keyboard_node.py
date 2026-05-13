#!/usr/bin/env python3
"""
Keyboard Teleop Node — Direct Velocity Control in XY Plane
============================================================
Controls both UR5 arms simultaneously in the XY plane using arrow keys.
Uses MoveIt Servo for real-time Cartesian velocity streaming.

Controls:
    ↑  (UP)     — move both arms +X
    ↓  (DOWN)   — move both arms -X
    ←  (LEFT)   — move both arms +Y
    →  (RIGHT)  — move both arms -Y
    Q           — rotate wrist CCW (+Z)
    E           — rotate wrist CW  (-Z)
    SPACE       — stop
    ESC / CTRL+C — quit

Requirements:
    pip install pynput
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped
from std_srvs.srv import Trigger

try:
    from pynput import keyboard
except ImportError:
    print("ERROR: pynput not installed. Run: pip install pynput")
    exit(1)

import time
import threading


# ─── Velocity settings ───────────────────────────────────────────────────────
LINEAR_SPEED  = 0.15   # m/s  — XY translation speed
ANGULAR_SPEED = 0.5    # rad/s — wrist rotation speed
PUBLISH_RATE  = 50     # Hz


class KeyboardTeleopNode(Node):

    def __init__(self):
        super().__init__('keyboard_teleop_node')

        # Publishers
        self.left_pub = self.create_publisher(
            TwistStamped, '/left_servo_node/delta_twist_cmds', 10)
        self.right_pub = self.create_publisher(
            TwistStamped, '/right_servo_node/delta_twist_cmds', 10)

        # Servo service clients
        self.left_start  = self.create_client(Trigger, '/left_servo_node/start_servo')
        self.right_start = self.create_client(Trigger, '/right_servo_node/start_servo')

        # Current velocity state
        self.vx  = 0.0   # linear X
        self.vy  = 0.0   # linear Y
        self.wz  = 0.0   # angular Z (wrist)
        self._lock = threading.Lock()

        # Active keys set
        self._active_keys = set()

        self.running = True

        self.get_logger().info('Keyboard teleop node started.')
        self.get_logger().info('Starting servo nodes...')

        # Start servo after short delay
        self.startup_timer = self.create_timer(2.0, self.startup)

    def startup(self):
        self.startup_timer.cancel()
        self._call_service(self.left_start,  '/left_servo_node/start_servo')
        self._call_service(self.right_start, '/right_servo_node/start_servo')

        self.get_logger().info('')
        self.get_logger().info('═══════════════════════════════════════')
        self.get_logger().info('  KEYBOARD TELEOP — XY Plane Control  ')
        self.get_logger().info('═══════════════════════════════════════')
        self.get_logger().info('  ↑        Move +X (forward)          ')
        self.get_logger().info('  ↓        Move -X (backward)         ')
        self.get_logger().info('  ←        Move +Y (left)             ')
        self.get_logger().info('  →        Move -Y (right)            ')
        self.get_logger().info('  Q        Rotate wrist CCW (+Z)      ')
        self.get_logger().info('  E        Rotate wrist CW  (-Z)      ')
        self.get_logger().info('  SPACE    Stop                        ')
        self.get_logger().info('  ESC      Quit                        ')
        self.get_logger().info('═══════════════════════════════════════')
        self.get_logger().info('')

        # Start keyboard listener in background thread
        self._kb_thread = threading.Thread(target=self._start_keyboard, daemon=True)
        self._kb_thread.start()

        # Start publish timer
        self._pub_timer = self.create_timer(1.0 / PUBLISH_RATE, self._publish)

    def _call_service(self, client, name):
        if not client.wait_for_service(timeout_sec=3.0):
            self.get_logger().warn(f'Service {name} not available')
            return
        future = client.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=3.0)
        if future.result() and future.result().success:
            self.get_logger().info(f'{name}: OK')

    def _start_keyboard(self):
        """Run keyboard listener in background thread."""
        with keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release
        ) as listener:
            listener.join()

    def _on_press(self, key):
        with self._lock:
            self._active_keys.add(key)
            self._update_velocity()

    def _on_release(self, key):
        with self._lock:
            self._active_keys.discard(key)
            self._update_velocity()

        # Quit on ESC
        if key == keyboard.Key.esc:
            self.running = False
            return False

    def _update_velocity(self):
        """Recompute velocity based on currently pressed keys."""
        vx = 0.0
        vy = 0.0
        wz = 0.0

        for key in self._active_keys:
            if key == keyboard.Key.up:
                vx += LINEAR_SPEED
            elif key == keyboard.Key.down:
                vx -= LINEAR_SPEED
            elif key == keyboard.Key.left:
                vy += LINEAR_SPEED
            elif key == keyboard.Key.right:
                vy -= LINEAR_SPEED
            elif hasattr(key, 'char'):
                if key.char == 'q' or key.char == 'Q':
                    wz += ANGULAR_SPEED
                elif key.char == 'e' or key.char == 'E':
                    wz -= ANGULAR_SPEED
                elif key.char == ' ':
                    vx = 0.0
                    vy = 0.0
                    wz = 0.0

        self.vx = vx
        self.vy = vy
        self.wz = wz

    def _make_twist(self, vx, vy, wz):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'world'
        msg.twist.linear.x  = float(vx)
        msg.twist.linear.y  = float(vy)
        msg.twist.linear.z  = 0.0
        msg.twist.angular.x = 0.0
        msg.twist.angular.y = 0.0
        msg.twist.angular.z = float(wz)
        return msg

    def _publish(self):
        with self._lock:
            vx = self.vx
            vy = self.vy
            wz = self.wz

        left_msg  = self._make_twist(vx, vy, wz)
        right_msg = self._make_twist(-vx, -vy, wz)  # flip for right arm

        self.left_pub.publish(left_msg)
        self.right_pub.publish(right_msg)


def main(args=None):
    rclpy.init(args=args)
    node = KeyboardTeleopNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Send stop command before shutting down
        stop = node._make_twist(0.0, 0.0, 0.0)
        node.left_pub.publish(stop)
        node.right_pub.publish(stop)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()