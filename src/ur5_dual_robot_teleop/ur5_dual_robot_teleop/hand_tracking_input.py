#!/usr/bin/env python3
"""
Hand Tracking Input — Dual Hand Delta Tracking
===============================================
Close right fist → right arm starts tracking from its current position.
Close left fist  → left arm starts tracking from its current position.
Each hand operates completely independently.

Scale: full image width  = full workspace X range
       full image height = full workspace Y range
"""

import math
import threading
import time
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool, Float64

from ur5_dual_robot_teleop.utils import Pose2D
from ur5_dual_robot_teleop.workspace import WORKSPACE
from ur5_dual_robot_teleop.teleop_config import CONFIG
from ur5_dual_robot_teleop.kalman_filter import KF1D

# ─── Load parameters from config ─────────────────────────────────────────────
_cfg = CONFIG['hand_tracking_input']
INVERT_X      = _cfg['invert_x']
INVERT_Y      = _cfg['invert_y']
DEAD_ZONE     = _cfg['dead_zone']
YAW_SCALE     = _cfg['yaw_scale']
YAW_DEAD_ZONE = _cfg['yaw_dead_zone']

FILTER_MODE      = _cfg.get('filter_mode',    'none')
EMA_ALPHA        = _cfg.get('ema_alpha',       0.7)
ACTIVE_HOLD_SEC  = _cfg.get('active_hold_sec', 0.15)

_kf            = _cfg.get('kalman', {})
KF_SIGMA_A     = _kf.get('sigma_a',     20.0)
KF_SIGMA_A_YAW = _kf.get('sigma_a_yaw', 15.0)
KF_SIGMA_R_POS = _kf.get('sigma_r_pos', 0.01)
KF_SIGMA_R_YAW = _kf.get('sigma_r_yaw', 0.05)


class HandTrackingInput:
    """
    Converts dual hand tracker topics into per-arm position offsets.

    Interface contract with TeleopNode:
      is_position_mode  — True
      is_active         — True while either fist is closed
      left_active       — True while left fist is closed
      right_active      — True while right fist is closed
      get_inputs()      — (left_offset, right_offset) Pose2D in meters
      get_gripper_command() — right gripper 0.0 (closed) → 1.0 (open)
    """

    is_position_mode = True

    def __init__(self, node: Node):
        self._node = node
        self._lock = threading.Lock()

        # ── Right hand sensor state ────────────────────────────────────────
        self._right_pose         = Pose2D()
        self._right_active       = False
        self._right_active_until = 0.0   # monotonic time — hold active until this
        self._right_gripper      = 0.0

        # ── Right hand tracking state ──────────────────────────────────────
        self._right_ref_hand   = None
        self._right_was_active = False
        self._right_prev_yaw   = None
        self._right_acc_dyaw   = 0.0

        # ── Right hand Kalman filters (x, y, yaw) ─────────────────────────
        self._kf_rx   = KF1D(KF_SIGMA_A,     KF_SIGMA_R_POS)
        self._kf_ry   = KF1D(KF_SIGMA_A,     KF_SIGMA_R_POS)
        self._kf_ryaw = KF1D(KF_SIGMA_A_YAW, KF_SIGMA_R_YAW)
        self._kf_rt   = None          # monotonic time of last KF step
        self._right_meas_pending = False
        self._right_meas_xyz     = (0.5, 0.5, 0.0)

        # ── Right hand EMA state ───────────────────────────────────────────
        self._ema_rx   = 0.5
        self._ema_ry   = 0.5
        self._ema_ryaw = 0.0

        # ── Left hand sensor state ─────────────────────────────────────────
        self._left_pose         = Pose2D()
        self._left_active       = False
        self._left_active_until = 0.0
        self._left_gripper      = 0.0

        # ── Left hand tracking state ───────────────────────────────────────
        self._left_ref_hand   = None
        self._left_was_active = False
        self._left_prev_yaw   = None
        self._left_acc_dyaw   = 0.0

        # ── Left hand Kalman filters (x, y, yaw) ──────────────────────────
        self._kf_lx   = KF1D(KF_SIGMA_A,     KF_SIGMA_R_POS)
        self._kf_ly   = KF1D(KF_SIGMA_A,     KF_SIGMA_R_POS)
        self._kf_lyaw = KF1D(KF_SIGMA_A_YAW, KF_SIGMA_R_YAW)
        self._kf_lt   = None
        self._left_meas_pending = False
        self._left_meas_xyz     = (0.5, 0.5, 0.0)

        # ── Left hand EMA state ────────────────────────────────────────────
        self._ema_lx   = 0.5
        self._ema_ly   = 0.5
        self._ema_lyaw = 0.0

        # ── Subscriptions — right hand ─────────────────────────────────────
        node.create_subscription(
            PoseStamped, '/hand_pose/right',      self._on_right_pose,    10)
        node.create_subscription(
            Bool,        '/hand_tracker/active',  self._on_right_active,  10)
        node.create_subscription(
            Float64,     '/hand_tracker/gripper', self._on_right_gripper, 10)

        # ── Subscriptions — left hand ──────────────────────────────────────
        node.create_subscription(
            PoseStamped, '/hand_pose/left',            self._on_left_pose,    10)
        node.create_subscription(
            Bool,        '/hand_tracker/left/active',  self._on_left_active,  10)
        node.create_subscription(
            Float64,     '/hand_tracker/left/gripper', self._on_left_gripper, 10)

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self):
        scale_x = WORKSPACE.x_max - WORKSPACE.x_min
        scale_y = WORKSPACE.y_max - WORKSPACE.y_min
        self._node.get_logger().info(
            f'Hand tracking ready | dual hand | delta mode | '
            f'scale: {scale_x:.2f} m/image × {scale_y:.2f} m/image | '
            f'dead zone: {DEAD_ZONE:.2f}')

    def stop(self):
        pass

    @property
    def should_quit(self) -> bool:
        return False

    # ── Control interface ─────────────────────────────────────────────────────

    @property
    def right_active(self) -> bool:
        with self._lock:
            return self._right_active or time.monotonic() < self._right_active_until

    @property
    def left_active(self) -> bool:
        with self._lock:
            return self._left_active or time.monotonic() < self._left_active_until

    @property
    def is_active(self) -> bool:
        with self._lock:
            return self._left_active or self._right_active

    def get_inputs(self) -> tuple:
        """Returns (left_offset, right_offset) in meters."""
        return self._compute_left_offset(), self._compute_right_offset()

    def _kf_step_right(self) -> Pose2D:
        with self._lock:
            pending = self._right_meas_pending
            if pending:
                mx, my, myaw             = self._right_meas_xyz
                self._right_meas_pending = False
            raw = self._right_pose

        if FILTER_MODE == 'none':
            return Pose2D(x=raw.x, y=raw.y, yaw=raw.yaw)

        if FILTER_MODE == 'ema':
            if pending:
                self._ema_rx  += EMA_ALPHA * (mx - self._ema_rx)
                self._ema_ry  += EMA_ALPHA * (my - self._ema_ry)
                step = (myaw - self._ema_ryaw + math.pi) % (2 * math.pi) - math.pi
                self._ema_ryaw += EMA_ALPHA * step
            return Pose2D(x=self._ema_rx, y=self._ema_ry, yaw=self._ema_ryaw)

        # kalman
        t = time.monotonic()
        if self._kf_rt is None:
            self._kf_rt = t
        dt = min(t - self._kf_rt, 0.1)
        self._kf_rt = t
        self._kf_rx.predict(dt)
        self._kf_ry.predict(dt)
        self._kf_ryaw.predict(dt)
        if pending:
            self._kf_rx.update(mx)
            self._kf_ry.update(my)
            self._kf_ryaw.update(myaw, wrap=True)
        return Pose2D(x=self._kf_rx.position,
                      y=self._kf_ry.position,
                      yaw=self._kf_ryaw.position)

    def _kf_step_left(self) -> Pose2D:
        with self._lock:
            pending = self._left_meas_pending
            if pending:
                mx, my, myaw            = self._left_meas_xyz
                self._left_meas_pending = False
            raw = self._left_pose

        if FILTER_MODE == 'none':
            return Pose2D(x=raw.x, y=raw.y, yaw=raw.yaw)

        if FILTER_MODE == 'ema':
            if pending:
                self._ema_lx  += EMA_ALPHA * (mx - self._ema_lx)
                self._ema_ly  += EMA_ALPHA * (my - self._ema_ly)
                step = (myaw - self._ema_lyaw + math.pi) % (2 * math.pi) - math.pi
                self._ema_lyaw += EMA_ALPHA * step
            return Pose2D(x=self._ema_lx, y=self._ema_ly, yaw=self._ema_lyaw)

        # kalman
        t = time.monotonic()
        if self._kf_lt is None:
            self._kf_lt = t
        dt = min(t - self._kf_lt, 0.1)
        self._kf_lt = t
        self._kf_lx.predict(dt)
        self._kf_ly.predict(dt)
        self._kf_lyaw.predict(dt)
        if pending:
            self._kf_lx.update(mx)
            self._kf_ly.update(my)
            self._kf_lyaw.update(myaw, wrap=True)
        return Pose2D(x=self._kf_lx.position,
                      y=self._kf_ly.position,
                      yaw=self._kf_lyaw.position)

    def _compute_right_offset(self) -> Pose2D:
        pose = self._kf_step_right()

        with self._lock:
            active = self._right_active

        if not active:
            self._right_ref_hand   = None
            self._right_was_active = False
            self._right_prev_yaw   = None
            self._right_acc_dyaw   = 0.0
            return Pose2D()

        if not self._right_was_active:
            self._right_ref_hand   = pose
            self._right_was_active = True
            self._right_prev_yaw   = pose.yaw
            self._right_acc_dyaw   = 0.0
            return Pose2D()

        dx_cam = pose.x - self._right_ref_hand.x
        dy_cam = pose.y - self._right_ref_hand.y

        step = pose.yaw - self._right_prev_yaw
        step = (step + math.pi) % (2 * math.pi) - math.pi
        self._right_prev_yaw = pose.yaw
        if abs(step) >= YAW_DEAD_ZONE:
            self._right_acc_dyaw += step

        if abs(dx_cam) < DEAD_ZONE: dx_cam = 0.0
        if abs(dy_cam) < DEAD_ZONE: dy_cam = 0.0

        dx_world   = dx_cam * (WORKSPACE.x_max - WORKSPACE.x_min)
        dy_world   = dy_cam * (WORKSPACE.y_max - WORKSPACE.y_min)
        dyaw_world = -self._right_acc_dyaw * YAW_SCALE

        if INVERT_X: dx_world = -dx_world
        if INVERT_Y: dy_world = -dy_world

        return Pose2D(x=dx_world, y=dy_world, yaw=dyaw_world)

    def _compute_left_offset(self) -> Pose2D:
        pose = self._kf_step_left()

        with self._lock:
            active = self._left_active

        if not active:
            self._left_ref_hand   = None
            self._left_was_active = False
            self._left_prev_yaw   = None
            self._left_acc_dyaw   = 0.0
            return Pose2D()

        if not self._left_was_active:
            self._left_ref_hand   = pose
            self._left_was_active = True
            self._left_prev_yaw   = pose.yaw
            self._left_acc_dyaw   = 0.0
            return Pose2D()

        dx_cam = pose.x - self._left_ref_hand.x
        dy_cam = pose.y - self._left_ref_hand.y

        step = pose.yaw - self._left_prev_yaw
        step = (step + math.pi) % (2 * math.pi) - math.pi
        self._left_prev_yaw = pose.yaw
        if abs(step) >= YAW_DEAD_ZONE:
            self._left_acc_dyaw += step

        if abs(dx_cam) < DEAD_ZONE: dx_cam = 0.0
        if abs(dy_cam) < DEAD_ZONE: dy_cam = 0.0

        dx_world   = dx_cam * (WORKSPACE.x_max - WORKSPACE.x_min)
        dy_world   = dy_cam * (WORKSPACE.y_max - WORKSPACE.y_min)
        dyaw_world = self._left_acc_dyaw * YAW_SCALE

        if INVERT_X: dx_world = -dx_world
        if INVERT_Y: dy_world = -dy_world

        return Pose2D(x=dx_world, y=dy_world, yaw=dyaw_world)

    def get_gripper_command(self) -> float:
        """Returns right gripper position: 0.0 = closed, 1.0 = open."""
        with self._lock:
            return self._right_gripper

    def get_left_gripper_command(self) -> float:
        """Returns left gripper position: 0.0 = closed, 1.0 = open."""
        with self._lock:
            return self._left_gripper

    # ── ROS callbacks — right hand ────────────────────────────────────────────

    def _on_right_pose(self, msg: PoseStamped):
        q = msg.pose.orientation
        yaw = math.atan2(2.0 * q.w * q.z, 1.0 - 2.0 * q.z * q.z)
        with self._lock:
            self._right_pose         = Pose2D(x=msg.pose.position.x,
                                              y=msg.pose.position.y, yaw=yaw)
            self._right_meas_pending = True
            self._right_meas_xyz     = (msg.pose.position.x, msg.pose.position.y, yaw)

    def _on_right_active(self, msg: Bool):
        with self._lock:
            self._right_active = msg.data
            if msg.data:
                self._right_active_until = time.monotonic() + ACTIVE_HOLD_SEC

    def _on_right_gripper(self, msg: Float64):
        with self._lock:
            self._right_gripper = msg.data

    # ── ROS callbacks — left hand ─────────────────────────────────────────────

    def _on_left_pose(self, msg: PoseStamped):
        q = msg.pose.orientation
        yaw = math.atan2(2.0 * q.w * q.z, 1.0 - 2.0 * q.z * q.z)
        with self._lock:
            self._left_pose         = Pose2D(x=msg.pose.position.x,
                                             y=msg.pose.position.y, yaw=yaw)
            self._left_meas_pending = True
            self._left_meas_xyz     = (msg.pose.position.x, msg.pose.position.y, yaw)

    def _on_left_active(self, msg: Bool):
        with self._lock:
            self._left_active = msg.data
            if msg.data:
                self._left_active_until = time.monotonic() + ACTIVE_HOLD_SEC

    def _on_left_gripper(self, msg: Float64):
        with self._lock:
            self._left_gripper = msg.data
