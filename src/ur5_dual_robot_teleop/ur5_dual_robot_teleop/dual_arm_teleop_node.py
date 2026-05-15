#!/usr/bin/env python3
"""
Unified Teleop Node
====================
Single ROS2 node that works with any controller via the strategy pattern.
Select controller at launch via ROS parameter.

Usage:
    # Direct velocity (default) — keyboard maps to velocity
    ros2 run ur5_dual_robot_teleop teleop_node

    # PD controller — keyboard sets target, PD follows smoothly
    ros2 run ur5_dual_robot_teleop teleop_node --ros-args -p controller:=pd

    # Position controller — keyboard triggers MoveGroup plans
    ros2 run ur5_dual_robot_teleop teleop_node --ros-args -p controller:=position

Switch at runtime (direct_velocity and pd only):
    ros2 param set /teleop_node controller pd
"""

import math
import time
import threading

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped
from std_msgs.msg import Float64
from visualization_msgs.msg import Marker
from std_srvs.srv import Trigger
import tf2_ros

from ur5_dual_robot_teleop.controllers import BaseController, Pose2D, Twist2D
from ur5_dual_robot_teleop.controllers import DirectVelocityController
from ur5_dual_robot_teleop.controllers import PDController
from ur5_dual_robot_teleop.controllers import PositionController
from ur5_dual_robot_teleop.workspace import WORKSPACE


# ─── Controller registry — add new controllers here ─────────────────────────
CONTROLLERS = {
    'direct_velocity': lambda: DirectVelocityController(
        max_linear=0.4,
        max_angular=0.8
    ),
    'p': lambda: PDController(
        kp_linear=5.0, kd_linear=0.0,
        kp_angular=2.5, kd_angular=0.0,
        max_linear=0.3, max_angular=0.5
    ),
    'pd': lambda: PDController(
        kp_linear=2.0, kd_linear=0.3,
        kp_angular=1.5, kd_angular=0.2,
        max_linear=0.4, max_angular=0.8
    ),
    'position': lambda: PositionController(
        position_threshold=0.01,
        angle_threshold=0.05
    ),
}

PUBLISH_RATE = 50  # Hz


class TeleopNode(Node):

    def __init__(self):
        super().__init__('teleop_node')

        # ── ROS parameters ────────────────────────────────────────────────
        self.declare_parameter('controller', 'direct_velocity')
        self.declare_parameter('left_frame',  'left_tool0')
        self.declare_parameter('right_frame', 'right_tool0')
        self.declare_parameter('world_frame', 'world')

        controller_name = self.get_parameter('controller').value
        self.left_frame  = self.get_parameter('left_frame').value
        self.right_frame = self.get_parameter('right_frame').value
        self.world_frame = self.get_parameter('world_frame').value

        # ── Publishers ────────────────────────────────────────────────────
        self.left_pub = self.create_publisher(
            TwistStamped, '/left_servo_node/delta_twist_cmds', 10)
        self.right_pub = self.create_publisher(
            TwistStamped, '/right_servo_node/delta_twist_cmds', 10)
        self.gripper_pub = self.create_publisher(
            Float64, '/gripper/command', 10)
        self._target_marker_pub = self.create_publisher(
            Marker, '/teleop/target_markers', 10)

        # ── Servo service clients ─────────────────────────────────────────
        self.left_start  = self.create_client(Trigger, '/left_servo_node/start_servo')
        self.right_start = self.create_client(Trigger, '/right_servo_node/start_servo')

        # ── TF listener for EEF feedback ──────────────────────────────────
        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ── Controllers (one per arm) ─────────────────────────────────────
        self.left_controller  = self._make_controller(controller_name)
        self.right_controller = self._make_controller(controller_name)
        self.controller_name  = controller_name
        self.get_logger().info(f'Using controller: {controller_name}')

        # ── Input handler ─────────────────────────────────────────────────
        self.declare_parameter('input', 'keyboard')
        input_type = self.get_parameter('input').value

        if input_type == 'keyboard':
            from ur5_dual_robot_teleop.dual_arm_keyboard_node import KeyboardInput
            self.input = KeyboardInput(linear_speed=0.3, angular_speed=0.5)
        elif input_type == 'hand_tracking':
            from ur5_dual_robot_teleop.hand_tracking_input import HandTrackingInput
            self.input = HandTrackingInput(self)
            # Hand tracking starts with P controller — switch to pd once tuned
            self.left_controller  = self._make_controller('p')
            self.right_controller = self._make_controller('p')
            self.controller_name  = 'p'
            self.get_logger().info('Hand tracking: using P controller (Kp=1.5, Kd=0)')
        else:
            self.get_logger().warn(f'Unknown input type "{input_type}", falling back to keyboard')
            from ur5_dual_robot_teleop.dual_arm_keyboard_node import KeyboardInput
            self.input = KeyboardInput(linear_speed=0.3, angular_speed=0.5)

        # ── State ─────────────────────────────────────────────────────────
        self.left_current  = Pose2D()
        self.right_current = Pose2D()
        self.left_target   = Pose2D()
        self.right_target  = Pose2D()
        self._last_time    = self.get_clock().now()
        self._lock         = threading.Lock()

        # ── Position tracking state (hand tracking only) ───────────────────
        self._left_ref_eef      = Pose2D()   # EEF pose at left fist-close
        self._right_ref_eef     = Pose2D()   # EEF pose at right fist-close
        self._left_was_active   = False      # rising-edge detect per arm
        self._right_was_active  = False

        # ── Parameter change callback (runtime controller switching) ──────
        self.add_on_set_parameters_callback(self._on_param_change)

        # ── Startup ───────────────────────────────────────────────────────
        self.startup_timer = self.create_timer(2.0, self._startup)

    def _make_controller(self, name: str) -> BaseController:
        if name not in CONTROLLERS:
            self.get_logger().warn(
                f'Unknown controller "{name}", falling back to direct_velocity')
            name = 'direct_velocity'
        return CONTROLLERS[name]()

    def _on_param_change(self, params):
        """Handle runtime parameter changes."""
        from rcl_interfaces.msg import SetParametersResult
        for param in params:
            if param.name == 'controller':
                new_name = param.value
                if new_name in CONTROLLERS:
                    with self._lock:
                        self.left_controller  = self._make_controller(new_name)
                        self.right_controller = self._make_controller(new_name)
                        self.controller_name  = new_name
                    self.get_logger().info(f'Switched to controller: {new_name}')
                else:
                    self.get_logger().warn(f'Unknown controller: {new_name}')
        return SetParametersResult(successful=True)

    def _startup(self):
        """Phase 1 — call start_servo, then poll until TF is ready."""
        self.startup_timer.cancel()
        self._call_service(self.left_start,  '/left_servo_node/start_servo')
        self._call_service(self.right_start, '/right_servo_node/start_servo')
        self.get_logger().info('Servo started — waiting for TF...')
        self._tf_poll = self.create_timer(0.1, self._poll_tf_ready)

    def _poll_tf_ready(self):
        """Phase 2 — fires every 100 ms until both EEF frames appear in TF."""
        try:
            self.tf_buffer.lookup_transform(
                self.world_frame, self.left_frame,  rclpy.time.Time())
            self.tf_buffer.lookup_transform(
                self.world_frame, self.right_frame, rclpy.time.Time())
        except Exception:
            return   # not ready yet — try again next tick

        self._tf_poll.cancel()

        # Seed targets from the actual EEF positions so the controller
        # starts with zero error instead of driving toward the world origin.
        self.left_target    = self._get_eef_pose(self.left_frame)
        self.right_target   = self._get_eef_pose(self.right_frame)
        self._left_ref_eef  = self.left_target
        self._right_ref_eef = self.right_target

        self.get_logger().info('')
        self.get_logger().info('═══════════════════════════════════════')
        self.get_logger().info(f'  Controller: {self.controller_name}  ')
        self.get_logger().info('  ↑↓←→  Move XY   A/D  Wrist  ESC Quit')
        self.get_logger().info('═══════════════════════════════════════')

        self.input.start()
        self._pub_timer = self.create_timer(1.0 / PUBLISH_RATE, self._control_loop)

    def _call_service(self, client, name):
        if not client.wait_for_service(timeout_sec=3.0):
            self.get_logger().warn(f'{name} not available')
            return
        future = client.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=3.0)

    def _get_eef_pose(self, frame: str) -> Pose2D:
        """Look up actual EEF pose from TF. Returns zeros if unavailable."""
        try:
            tf = self.tf_buffer.lookup_transform(
                self.world_frame, frame, rclpy.time.Time())
            # Extract yaw from quaternion
            q = tf.transform.rotation
            yaw = math.atan2(
                2.0 * (q.w * q.z + q.x * q.y),
                1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            )
            return Pose2D(
                x=tf.transform.translation.x,
                y=tf.transform.translation.y,
                yaw=yaw
            )
        except Exception:
            return Pose2D()

    def _update_target(self, inp: Pose2D, current_target: Pose2D) -> Pose2D:
        """
        Update target pose based on controller type.
        - DirectVelocity: target IS the velocity (inp passed directly)
        - PD/Position:    target is incrementally updated by input
        """
        if self.controller_name == 'direct_velocity':
            return inp  # target = velocity setpoint
        else:
            # Increment target position by input * small step
            step = 1.0 / PUBLISH_RATE
            return Pose2D(
                x=current_target.x   + inp.x   * step,
                y=current_target.y   + inp.y   * step,
                yaw=current_target.yaw + inp.yaw * step
            )

    def _control_loop(self):
        """Main control loop — runs at PUBLISH_RATE Hz."""
        if self.input.should_quit:
            self._stop()
            return

        now = self.get_clock().now()
        dt  = (now - self._last_time).nanoseconds / 1e9
        self._last_time = now

        if getattr(self.input, 'is_position_mode', False):
            if hasattr(self.input, 'get_inputs'):
                left_offset, right_offset = self.input.get_inputs()
            else:
                left_offset = right_offset = self.input.get_input()
            self._step_position_control(left_offset, right_offset, dt)
        else:
            self._step_velocity_control(self.input.get_input(), dt)

        if hasattr(self.input, 'get_gripper_command'):
            g = Float64()
            g.data = self.input.get_gripper_command()
            self.gripper_pub.publish(g)

    def _step_position_control(self, left_offset: Pose2D, right_offset: Pose2D, dt: float):
        """
        Position tracking step for hand tracking mode.

        Each arm is handled independently:
          Inactive (fist open)   → zero velocity, reset controller.
          Rising edge (fist just closed) → snapshot EEF as reference origin.
          Active (fist held)     → target = ref_eef + offset, P drives EEF.
        """
        left_active  = getattr(self.input, 'left_active',  self.input.is_active)
        right_active = getattr(self.input, 'right_active', self.input.is_active)
        stamp = self.get_clock().now().to_msg()

        left_target = right_target = None

        # ── Left arm ──────────────────────────────────────────────────────
        if not left_active:
            if self._left_was_active:
                self.left_controller.reset()
            self._left_was_active = False
            stop = TwistStamped()
            stop.header.stamp    = stamp
            stop.header.frame_id = self.world_frame
            self.left_pub.publish(stop)
        else:
            if not self._left_was_active:
                self.left_controller.reset()
                left_raw = self._get_eef_pose(self.left_frame)
                lx, ly   = WORKSPACE.clamp(left_raw.x, left_raw.y)
                self._left_ref_eef = Pose2D(x=lx, y=ly, yaw=left_raw.yaw)
            self._left_was_active = True
            lx, ly = WORKSPACE.clamp(
                self._left_ref_eef.x + left_offset.x,
                self._left_ref_eef.y + left_offset.y)
            left_target  = Pose2D(x=lx, y=ly, yaw=self._left_ref_eef.yaw + left_offset.yaw)
            left_current = self._get_eef_pose(self.left_frame)
            left_vel     = self.left_controller.compute_velocity(left_current, left_target, dt)
            self.left_pub.publish(self._to_twist(left_vel, invert=True, swap_xy=True))

        # ── Right arm ─────────────────────────────────────────────────────
        if not right_active:
            if self._right_was_active:
                self.right_controller.reset()
            self._right_was_active = False
            stop = TwistStamped()
            stop.header.stamp    = stamp
            stop.header.frame_id = self.world_frame
            self.right_pub.publish(stop)
        else:
            if not self._right_was_active:
                self.right_controller.reset()
                right_raw = self._get_eef_pose(self.right_frame)
                rx, ry    = WORKSPACE.clamp(right_raw.x, right_raw.y)
                self._right_ref_eef = Pose2D(x=rx, y=ry, yaw=right_raw.yaw)
            self._right_was_active = True
            rx, ry = WORKSPACE.clamp(
                self._right_ref_eef.x + right_offset.x,
                self._right_ref_eef.y + right_offset.y)
            right_target  = Pose2D(x=rx, y=ry, yaw=self._right_ref_eef.yaw + right_offset.yaw)
            right_current = self._get_eef_pose(self.right_frame)
            right_vel     = self.right_controller.compute_velocity(right_current, right_target, dt)
            self.right_pub.publish(self._to_twist(right_vel, invert=False, swap_xy=True))

        # Publish markers for any arm that is active
        if left_target is not None or right_target is not None:
            self._publish_target_markers(
                left_target  or Pose2D(x=self._left_ref_eef.x,  y=self._left_ref_eef.y),
                right_target or Pose2D(x=self._right_ref_eef.x, y=self._right_ref_eef.y))

    def _step_velocity_control(self, inp: Pose2D, dt: float):
        """
        Velocity control step for keyboard mode.

        DirectVelocity: inp is sent straight to servo.
        PD: inp increments a target position, PD drives EEF toward it.
        Velocity components that would push the EEF past the workspace
        boundary are zeroed out before publishing.
        """
        with self._lock:
            self.left_current  = self._get_eef_pose(self.left_frame)
            self.right_current = self._get_eef_pose(self.right_frame)

            inp = self._clamp_input_at_boundary(inp)

            self.left_target  = self._update_target(inp, self.left_target)
            self.right_target = self._update_target(inp, self.right_target)

            left_vel  = self.left_controller.compute_velocity(
                self.left_current, self.left_target, dt)
            right_vel = self.right_controller.compute_velocity(
                self.right_current, self.right_target, dt)

        self.left_pub.publish(self._to_twist(left_vel,  invert=False))
        self.right_pub.publish(self._to_twist(right_vel, invert=True))

    def _clamp_input_at_boundary(self, inp: Pose2D) -> Pose2D:
        """
        Zero out input components when either arm is near the workspace boundary.

        A 3 cm margin is applied so the stop triggers before the exact edge,
        compensating for servo command latency and TF reporting lag.
        Both arms share the same input — if one arm hits a boundary the velocity
        is zeroed for both. The right arm's world velocity is -inp (invert=True),
        so its sign checks are mirrored.
        """
        MARGIN = 0.03
        lp = self.left_current
        rp = self.right_current
        vx, vy = inp.x, inp.y

        # Left arm  (world vel = +inp)
        if lp.x <= WORKSPACE.x_min + MARGIN and vx < 0: vx = 0.0
        if lp.x >= WORKSPACE.x_max - MARGIN and vx > 0: vx = 0.0
        if lp.y <= WORKSPACE.y_min + MARGIN and vy < 0: vy = 0.0
        if lp.y >= WORKSPACE.y_max - MARGIN and vy > 0: vy = 0.0

        # Right arm (world vel = -inp, so sign checks are reversed)
        if rp.x <= WORKSPACE.x_min + MARGIN and vx > 0: vx = 0.0
        if rp.x >= WORKSPACE.x_max - MARGIN and vx < 0: vx = 0.0
        if rp.y <= WORKSPACE.y_min + MARGIN and vy > 0: vy = 0.0
        if rp.y >= WORKSPACE.y_max - MARGIN and vy < 0: vy = 0.0

        return Pose2D(x=vx, y=vy, yaw=inp.yaw)

    def _publish_target_markers(self, left: Pose2D, right: Pose2D):
        """Publish sphere markers at left/right target positions on the workspace plane."""
        stamp = self.get_clock().now().to_msg()
        for marker_id, pose, r, g, b in [
            (0, left,  0.2, 1.0, 0.2),   # green — left arm target
            (1, right, 0.2, 0.4, 1.0),   # blue  — right arm target
        ]:
            m = Marker()
            m.header.frame_id    = self.world_frame
            m.header.stamp       = stamp
            m.ns                 = 'teleop_targets'
            m.id                 = marker_id
            m.type               = Marker.SPHERE
            m.action             = Marker.ADD
            m.pose.position.x    = pose.x
            m.pose.position.y    = pose.y
            m.pose.position.z    = WORKSPACE.center_z
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 0.06
            m.color.r = r
            m.color.g = g
            m.color.b = b
            m.color.a = 0.85
            self._target_marker_pub.publish(m)

    def _to_twist(self, vel: Twist2D, invert: bool = False, swap_xy: bool = False) -> TwistStamped:
        sign = -1.0 if invert else 1.0
        msg = TwistStamped()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = self.world_frame
        vx, vy = (-vel.vy, vel.vx) if swap_xy else (vel.vx, vel.vy)
        msg.twist.linear.x  = float(sign * vx)
        msg.twist.linear.y  = float(sign * vy)
        msg.twist.linear.z  = 0.0
        msg.twist.angular.z = float(sign * vel.wz)
        return msg

    def _stop(self):
        """Publish zero velocity and shut down."""
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