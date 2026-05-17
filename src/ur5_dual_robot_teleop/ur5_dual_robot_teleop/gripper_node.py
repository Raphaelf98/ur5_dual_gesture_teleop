#!/usr/bin/env python3
"""
Gripper Driver Node
====================
Subscribes to /left_gripper/command and /right_gripper/command (Float64,
0.0=open 1.0=closed) and forwards each to its respective ForwardCommandController
as a Float64MultiArray (direct position, no trajectory interpolation).

Joint mapping (Robotiq 2F-85):
  left_robotiq_85_left_knuckle_joint  : 0.0 (open) → 0.7929 (closed)
  right_robotiq_85_left_knuckle_joint : 0.0 (open) → 0.7929 (closed)
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64, Float64MultiArray


KNUCKLE_CLOSED = 0.7929  # rad — Robotiq 2F-85 fully closed


class GripperNode(Node):

    def __init__(self):
        super().__init__('gripper_node')

        self._left_pub = self.create_publisher(
            Float64MultiArray, '/left_gripper_controller/commands', 10)
        self._right_pub = self.create_publisher(
            Float64MultiArray, '/right_gripper_controller/commands', 10)

        self.create_subscription(
            Float64, '/left_gripper/command', self._on_left_command, 10)
        self.create_subscription(
            Float64, '/right_gripper/command', self._on_right_command, 10)

        self.get_logger().info(
            'Gripper node ready — listening on /left_gripper/command and /right_gripper/command')

    def _on_left_command(self, msg: Float64):
        knuckle_pos = KNUCKLE_CLOSED * max(0.0, min(1.0, msg.data))
        out = Float64MultiArray()
        out.data = [knuckle_pos]
        self._left_pub.publish(out)

    def _on_right_command(self, msg: Float64):
        knuckle_pos = KNUCKLE_CLOSED * max(0.0, min(1.0, msg.data))
        out = Float64MultiArray()
        out.data = [knuckle_pos]
        self._right_pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = GripperNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
