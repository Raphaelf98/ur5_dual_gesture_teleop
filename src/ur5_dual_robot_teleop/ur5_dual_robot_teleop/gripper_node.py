#!/usr/bin/env python3
"""
Gripper Driver Node
====================
Subscribes to /left_gripper/command and /right_gripper/command (Float64,
0.0=open 1.0=closed) and forwards each to its respective gripper controller
as a JointTrajectory message.

Joint mapping (Robotiq 2F-85):
  left_robotiq_85_left_knuckle_joint  : 0.0 (open) → 0.7929 (closed)
  right_robotiq_85_left_knuckle_joint : 0.0 (open) → 0.7929 (closed)
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration


KNUCKLE_CLOSED = 0.7929  # rad — default gripper_closed_position from macro
MOVE_TIME      = 0.3    # seconds — 2 control periods at 100 Hz


class GripperNode(Node):

    def __init__(self):
        super().__init__('gripper_node')

        self._left_pub = self.create_publisher(
            JointTrajectory, '/left_gripper_controller/joint_trajectory', 10)
        self._right_pub = self.create_publisher(
            JointTrajectory, '/right_gripper_controller/joint_trajectory', 10)

        self.create_subscription(
            Float64, '/left_gripper/command', self._on_left_command, 10)
        self.create_subscription(
            Float64, '/right_gripper/command', self._on_right_command, 10)

        self.get_logger().info(
            'Gripper node ready — listening on /left_gripper/command and /right_gripper/command')

    def _on_left_command(self, msg: Float64):
        knuckle_pos = KNUCKLE_CLOSED * max(0.0, min(1.0, msg.data))
        self._publish(self._left_pub,
                      ['left_robotiq_85_left_knuckle_joint'],
                      [knuckle_pos])

    def _on_right_command(self, msg: Float64):
        knuckle_pos = KNUCKLE_CLOSED * max(0.0, min(1.0, msg.data))
        self._publish(self._right_pub,
                      ['right_robotiq_85_left_knuckle_joint'],
                      [knuckle_pos])

    def _publish(self, pub, joints, positions):
        traj = JointTrajectory()
        traj.header.stamp = self.get_clock().now().to_msg()
        traj.joint_names  = joints

        pt = JointTrajectoryPoint()
        pt.positions      = positions
        pt.velocities     = [0.0] * len(positions)
        pt.time_from_start = Duration(sec=0, nanosec=int(MOVE_TIME * 1e9))

        traj.points = [pt]
        pub.publish(traj)


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
