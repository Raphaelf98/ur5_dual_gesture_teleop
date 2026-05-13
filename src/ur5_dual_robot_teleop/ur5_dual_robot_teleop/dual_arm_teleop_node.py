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

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped
from std_srvs.srv import Trigger
import tf2_ros
import time
import threading

from ur5_dual_robot_teleop.controllers import BaseController, Pose2D, Twist2D
from ur5_dual_robot_teleop.controllers import DirectVelocityController
from ur5_dual_robot_teleop.controllers import PDController
from ur5_dual_robot_teleop.controllers import PositionController


# ─── Controller registry — add new controllers here ─────────────────────────
CONTROLLERS = {
    'direct_velocity': lambda: DirectVelocityController(
        max_linear=0.3,
        max_angular=0.5
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
        self.startup_timer.cancel()
        self._call_service(self.left_start,  '/left_servo_node/start_servo')
        self._call_service(self.right_start, '/right_servo_node/start_servo')

        self.get_logger().info('')
        self.get_logger().info('═══════════════════════════════════════')
        self.get_logger().info(f'  Controller: {self.controller_name}  ')
        self.get_logger().info('  ↑↓←→  Move XY   Q/E  Wrist  ESC Quit')
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
            import math
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

        # Compute dt
        now = self.get_clock().now()
        dt  = (now - self._last_time).nanoseconds / 1e9
        self._last_time = now

        # Get input
        inp = self.input.get_input()

        with self._lock:
            # Get actual EEF poses from TF
            self.left_current  = self._get_eef_pose(self.left_frame)
            self.right_current = self._get_eef_pose(self.right_frame)

            # Update targets
            self.left_target  = self._update_target(inp, self.left_target)
            self.right_target = self._update_target(inp, self.right_target)

            # Compute velocities
            left_vel  = self.left_controller.compute_velocity(
                self.left_current, self.left_target, dt)
            right_vel = self.right_controller.compute_velocity(
                self.right_current, self.right_target, dt)

        # Publish — invert right arm for same-direction motion
        self.left_pub.publish(self._to_twist(left_vel,  invert=False))
        self.right_pub.publish(self._to_twist(right_vel, invert=True))

    def _to_twist(self, vel: Twist2D, invert: bool = False) -> TwistStamped:
        sign = -1.0 if invert else 1.0
        msg = TwistStamped()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = self.world_frame
        msg.twist.linear.x  = float(sign * vel.vx)
        msg.twist.linear.y  = float(sign * vel.vy)
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