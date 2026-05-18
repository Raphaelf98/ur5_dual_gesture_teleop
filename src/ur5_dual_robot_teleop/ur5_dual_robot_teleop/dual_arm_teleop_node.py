#!/usr/bin/env python3
"""
Dual Arm Hand-Tracking Teleop Node
====================================
PD-controlled hand tracking for dual UR5 arms via MoveIt Servo.
Close left fist  → left arm tracks hand position.
Close right fist → right arm tracks hand position.
Each arm is controlled independently.
"""

import time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped
from std_msgs.msg import Float64
from visualization_msgs.msg import MarkerArray
from std_srvs.srv import Trigger
import tf2_ros

from ur5_dual_robot_teleop.controllers import PDController
from ur5_dual_robot_teleop.hand_tracking_input import HandTrackingInput
from ur5_dual_robot_teleop.utils import Pose2D, eef_pose, make_twist, build_target_markers
from ur5_dual_robot_teleop.workspace import WORKSPACE
from ur5_dual_robot_teleop.teleop_config import CONFIG

_cc = CONFIG['controllers']
_vc = CONFIG['visualization']

PUBLISH_RATE  = _cc['publish_rate']
ARROW_LENGTH  = _vc['target_arrow_length']
SPHERE_SCALE  = _vc['target_sphere_scale']
ARROW_SHAFT_D = _vc['arrow_shaft_diameter']
ARROW_HEAD_D  = _vc['arrow_head_diameter']


def _make_pd() -> PDController:
    c = _cc['hand_tracking']
    return PDController(
        kp_linear=c['kp_linear'],   kd_linear=c['kd_linear'],
        kp_angular=c['kp_angular'], kd_angular=c['kd_angular'],
        max_linear=c['max_linear'], max_angular=c['max_angular'],
    )


class TeleopNode(Node):

    def __init__(self):
        super().__init__('teleop_node')

        self.declare_parameter('left_frame',  'left_tool0')
        self.declare_parameter('right_frame', 'right_tool0')
        self.declare_parameter('world_frame', 'world')

        self.left_frame  = self.get_parameter('left_frame').value
        self.right_frame = self.get_parameter('right_frame').value
        self.world_frame = self.get_parameter('world_frame').value

        # ── Publishers ────────────────────────────────────────────────────
        self.left_pub  = self.create_publisher(
            TwistStamped, '/left_servo_node/delta_twist_cmds', 10)
        self.right_pub = self.create_publisher(
            TwistStamped, '/right_servo_node/delta_twist_cmds', 10)
        self.left_gripper_pub  = self.create_publisher(
            Float64, '/left_gripper/command', 10)
        self.right_gripper_pub = self.create_publisher(
            Float64, '/right_gripper/command', 10)
        self._marker_pub = self.create_publisher(
            MarkerArray, '/teleop/target_markers', 10)

        # ── Servo service clients ─────────────────────────────────────────
        self._left_start  = self.create_client(Trigger, '/left_servo_node/start_servo')
        self._right_start = self.create_client(Trigger, '/right_servo_node/start_servo')

        # ── TF ────────────────────────────────────────────────────────────
        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ── PD controllers — one per arm ──────────────────────────────────
        self.left_controller  = _make_pd()
        self.right_controller = _make_pd()

        # ── Hand tracking input ───────────────────────────────────────────
        self.input = HandTrackingInput(self)

        # ── Per-arm tracking state ────────────────────────────────────────
        self._left_ref_eef     = Pose2D()
        self._right_ref_eef    = Pose2D()
        self._left_was_active  = False
        self._right_was_active = False
        self._last_time        = self.get_clock().now()

        self._startup_timer = self.create_timer(2.0, self._startup)

    # ── Startup sequence ──────────────────────────────────────────────────

    def _startup(self):
        """Start servo nodes then wait for TF to become available."""
        self._startup_timer.cancel()
        self._call_service(self._left_start,  '/left_servo_node/start_servo')
        self._call_service(self._right_start, '/right_servo_node/start_servo')
        self.get_logger().info('Servo started — waiting for TF...')
        self._tf_poll = self.create_timer(0.1, self._poll_tf_ready)

    def _poll_tf_ready(self):
        """Fire every 100 ms until both EEF frames appear in TF, then start the loop."""
        try:
            self.tf_buffer.lookup_transform(
                self.world_frame, self.left_frame,  rclpy.time.Time())
            self.tf_buffer.lookup_transform(
                self.world_frame, self.right_frame, rclpy.time.Time())
        except Exception:
            return

        self._tf_poll.cancel()
        self._left_ref_eef  = eef_pose(self.tf_buffer, self.world_frame, self.left_frame)
        self._right_ref_eef = eef_pose(self.tf_buffer, self.world_frame, self.right_frame)

        self.get_logger().info('Ready — close fist to activate arm tracking')
        self.input.start()
        self._pub_timer = self.create_timer(1.0 / PUBLISH_RATE, self._control_loop)

    def _call_service(self, client, name: str):
        if not client.wait_for_service(timeout_sec=3.0):
            self.get_logger().warn(f'{name} not available')
            return
        future = client.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=3.0)

    # ── Control loop ──────────────────────────────────────────────────────

    def _control_loop(self):
        now = self.get_clock().now()
        dt  = (now - self._last_time).nanoseconds / 1e9
        self._last_time = now
        stamp = now.to_msg()

        left_offset, right_offset = self.input.get_inputs()
        left_target = right_target = None

        # ── Left arm ──────────────────────────────────────────────────────
        if not self.input.left_active:
            if self._left_was_active:
                self.left_controller.reset()
            self._left_was_active = False
            self.left_pub.publish(TwistStamped())
        else:
            if not self._left_was_active:
                self.left_controller.reset()
                raw = eef_pose(self.tf_buffer, self.world_frame, self.left_frame)
                lx, ly = WORKSPACE.clamp(raw.x, raw.y)
                self._left_ref_eef = Pose2D(x=lx, y=ly, yaw=raw.yaw)
            self._left_was_active = True

            lx, ly = WORKSPACE.clamp(
                self._left_ref_eef.x + left_offset.x,
                self._left_ref_eef.y + left_offset.y)
            left_current = eef_pose(self.tf_buffer, self.world_frame, self.left_frame)
            left_target  = Pose2D(x=lx, y=ly, yaw=self._left_ref_eef.yaw - left_offset.yaw)
            left_vel     = self.left_controller.compute_velocity(left_current, left_target, dt)
            self.left_pub.publish(
                make_twist(left_vel, self.world_frame, stamp, invert=True, swap_xy=True))

        # ── Right arm ─────────────────────────────────────────────────────
        if not self.input.right_active:
            if self._right_was_active:
                self.right_controller.reset()
            self._right_was_active = False
            self.right_pub.publish(TwistStamped())
        else:
            if not self._right_was_active:
                self.right_controller.reset()
                raw = eef_pose(self.tf_buffer, self.world_frame, self.right_frame)
                rx, ry = WORKSPACE.clamp(raw.x, raw.y)
                self._right_ref_eef = Pose2D(x=rx, y=ry, yaw=raw.yaw)
            self._right_was_active = True

            rx, ry = WORKSPACE.clamp(
                self._right_ref_eef.x + right_offset.x,
                self._right_ref_eef.y + right_offset.y)
            right_current = eef_pose(self.tf_buffer, self.world_frame, self.right_frame)
            right_target  = Pose2D(x=rx, y=ry, yaw=self._right_ref_eef.yaw + right_offset.yaw)
            right_vel     = self.right_controller.compute_velocity(right_current, right_target, dt)
            self.right_pub.publish(
                make_twist(right_vel, self.world_frame, stamp, invert=False, swap_xy=True))

        # ── Grippers ──────────────────────────────────────────────────────
        rg = Float64(); rg.data = self.input.get_gripper_command()
        lg = Float64(); lg.data = self.input.get_left_gripper_command()
        self.right_gripper_pub.publish(rg)
        self.left_gripper_pub.publish(lg)

        # ── Target markers ────────────────────────────────────────────────
        if left_target is not None or right_target is not None:
            self._marker_pub.publish(build_target_markers(
                left_target  or Pose2D(x=self._left_ref_eef.x,  y=self._left_ref_eef.y),
                right_target or Pose2D(x=self._right_ref_eef.x, y=self._right_ref_eef.y),
                self.world_frame, stamp, WORKSPACE.center_z,
                ARROW_LENGTH, SPHERE_SCALE, ARROW_SHAFT_D, ARROW_HEAD_D,
            ))

    # ── Shutdown ──────────────────────────────────────────────────────────

    def _stop(self):
        stop = TwistStamped()
        stop.header.stamp    = self.get_clock().now().to_msg()
        stop.header.frame_id = self.world_frame
        for _ in range(10):
            self.left_pub.publish(stop)
            self.right_pub.publish(stop)
            time.sleep(0.02)
        self.input.stop()
        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = TeleopNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._stop()
        node.destroy_node()


if __name__ == '__main__':
    main()
