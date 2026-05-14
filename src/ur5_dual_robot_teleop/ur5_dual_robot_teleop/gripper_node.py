#!/usr/bin/env python3
"""
Gripper Driver Node
====================
Subscribes to /gripper/command (Float64, 0.0=closed 1.0=open) and
forwards the command to both left and right gripper controllers as
JointTrajectory messages.

Joint mapping:
  left_finger_left_joint  : 0.0 (closed) → -0.025 (open)
  left_finger_right_joint : 0.0 (closed) →  0.025 (open)
  (same pattern for right_)
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration


FINGER_OPEN  = 0.025   # metres — max travel per finger
MOVE_TIME    = 0.3     # seconds — trajectory duration


class GripperNode(Node):

    def __init__(self):
        super().__init__('gripper_node')

        self._left_pub = self.create_publisher(
            JointTrajectory, '/left_gripper_controller/joint_trajectory', 10)
        self._right_pub = self.create_publisher(
            JointTrajectory, '/right_gripper_controller/joint_trajectory', 10)

        self.create_subscription(Float64, '/gripper/command', self._on_command, 10)

        self.get_logger().info('Gripper node ready — listening on /gripper/command')

    def _on_command(self, msg: Float64):
        val = max(0.0, min(1.0, msg.data))

        left_fl = -FINGER_OPEN * val   # finger_left moves negative to open
        left_fr =  FINGER_OPEN * val   # finger_right moves positive to open

        self._publish(self._left_pub,
                      ['left_finger_left_joint', 'left_finger_right_joint'],
                      [left_fl, left_fr])
        self._publish(self._right_pub,
                      ['right_finger_left_joint', 'right_finger_right_joint'],
                      [left_fl, left_fr])

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
