#!/usr/bin/env python3
"""
Dual Arm Teleop Node — hardcoded command sequences for both UR5 arms via MoveIt Servo.

Sends a predefined sequence of Cartesian twist commands to both arms simultaneously.
Each command runs for a set duration before moving to the next.

Usage:
    ros2 run your_teleop_package dual_arm_teleop_node
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped
from std_srvs.srv import Trigger
import time


# ─── Command sequence definition ────────────────────────────────────────────
# Each entry: (duration_sec, left_twist, right_twist)
# Twist format: (lin_x, lin_y, lin_z, ang_x, ang_y, ang_z)
# Units: unitless in [-1, 1] range (scaled by servo config: linear=0.4 m/s, rot=0.8 rad/s)

COMMAND_SEQUENCE = [
    # (duration, left_twist,               right_twist)
    (2.0,  (0.3,  0.0,  0.0,  0.0, 0.0, 0.0),   (0.3,  0.0,  0.0,  0.0, 0.0, 0.0)),   # both move +X
    (2.0,  (-0.3, 0.0,  0.0,  0.0, 0.0, 0.0),   (-0.3, 0.0,  0.0,  0.0, 0.0, 0.0)),  # both move -X
    (2.0,  (0.0,  0.3,  0.0,  0.0, 0.0, 0.0),   (0.0, -0.3,  0.0,  0.0, 0.0, 0.0)),  # left +Y, right -Y
    (2.0,  (0.0, -0.3,  0.0,  0.0, 0.0, 0.0),   (0.0,  0.3,  0.0,  0.0, 0.0, 0.0)),  # left -Y, right +Y
    (2.0,  (0.0,  0.0,  0.3,  0.0, 0.0, 0.0),   (0.0,  0.0,  0.3,  0.0, 0.0, 0.0)),  # both move +Z
    (2.0,  (0.0,  0.0, -0.3,  0.0, 0.0, 0.0),   (0.0,  0.0, -0.3,  0.0, 0.0, 0.0)),  # both move -Z
    (2.0,  (0.0,  0.0,  0.0,  0.0, 0.3, 0.0),   (0.0,  0.0,  0.0,  0.0, 0.3, 0.0)),  # both rotate pitch
    (2.0,  (0.0,  0.0,  0.0,  0.0,-0.3, 0.0),   (0.0,  0.0,  0.0,  0.0,-0.3, 0.0)),  # both rotate -pitch
    (1.0,  (0.0,  0.0,  0.0,  0.0, 0.0, 0.0),   (0.0,  0.0,  0.0,  0.0, 0.0, 0.0)),  # stop
]


class DualArmTeleopNode(Node):

    def __init__(self):
        super().__init__('dual_arm_teleop_node')

        # Publishers for twist commands
        self.left_pub = self.create_publisher(
            TwistStamped,
            '/left_servo_node/delta_twist_cmds',
            10
        )
        self.right_pub = self.create_publisher(
            TwistStamped,
            '/right_servo_node/delta_twist_cmds',
            10
        )

        # Service clients to start servo
        self.left_start  = self.create_client(Trigger, '/left_servo_node/start_servo')
        self.right_start = self.create_client(Trigger, '/right_servo_node/start_servo')

        # Publish rate — servo expects ~50Hz
        self.publish_rate = 50  # Hz
        self.timer = None

        self.get_logger().info('Dual arm teleop node started.')
        self.get_logger().info('Waiting for servo nodes...')

        # Start sequence after a short delay
        self.startup_timer = self.create_timer(2.0, self.startup)

    def startup(self):
        """Start servo nodes and begin command sequence."""
        self.startup_timer.cancel()

        # Start both servo nodes
        self._call_service(self.left_start,  'left_servo_node/start_servo')
        self._call_service(self.right_start, 'right_servo_node/start_servo')

        self.get_logger().info('Servo nodes started. Beginning command sequence...')
        self._run_sequence()

    def _call_service(self, client, name):
        """Call a trigger service synchronously."""
        if not client.wait_for_service(timeout_sec=3.0):
            self.get_logger().warn(f'Service {name} not available')
            return
        req = Trigger.Request()
        future = client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=3.0)
        if future.result():
            self.get_logger().info(f'{name}: {future.result().message or "OK"}')

    def _make_twist(self, frame_id, lx, ly, lz, ax, ay, az):
        """Build a TwistStamped message."""
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = frame_id
        msg.twist.linear.x  = lx
        msg.twist.linear.y  = ly
        msg.twist.linear.z  = lz
        msg.twist.angular.x = ax
        msg.twist.angular.y = ay
        msg.twist.angular.z = az
        return msg

    def _run_sequence(self):
        """Execute the command sequence."""
        for i, (duration, left_twist, right_twist) in enumerate(COMMAND_SEQUENCE):
            self.get_logger().info(
                f'Step {i+1}/{len(COMMAND_SEQUENCE)}: '
                f'duration={duration}s  '
                f'left={left_twist[:3]}  right={right_twist[:3]}'
            )

            # Publish at rate for duration
            steps = int(duration * self.publish_rate)
            sleep_time = 1.0 / self.publish_rate

            for _ in range(steps):
                left_msg  = self._make_twist('world', *left_twist)
                right_msg = self._make_twist('world', *right_twist)
                self.left_pub.publish(left_msg)
                self.right_pub.publish(right_msg)
                time.sleep(sleep_time)

        # Send stop command
        self.get_logger().info('Sequence complete. Sending stop command.')
        stop = self._make_twist('world', 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)        
        for _ in range(10):
            self.left_pub.publish(stop)
            self.right_pub.publish(stop)
            time.sleep(0.02)

        self.get_logger().info('Done.')


def main(args=None):
    rclpy.init(args=args)
    node = DualArmTeleopNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()