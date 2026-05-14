#!/usr/bin/env python3
"""
Hand Tracker Node — Dual Hand + Wrist Orientation
===================================================
Tracks both hands independently with:
  - XY position (wrist landmark 0)
  - Wrist orientation/yaw (wrist→middle_mcp angle)
  - Fist detection (avg fingertip distance)
  - Continuous gripper control (thumb angle)

Published topics (right hand):
  /hand_pose/right          (geometry_msgs/PoseStamped)
  /hand_tracker/active      (std_msgs/Bool)
  /hand_tracker/gripper     (std_msgs/Float64)

Published topics (left hand):
  /hand_pose/left           (geometry_msgs/PoseStamped)
  /hand_tracker/left/active (std_msgs/Bool)
  /hand_tracker/left/gripper(std_msgs/Float64)

Shared:
  /hand_tracker/image       (sensor_msgs/Image)
  /hand_tracker/camera_info (sensor_msgs/CameraInfo)
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool, Float64
from sensor_msgs.msg import Image, CameraInfo

import cv2
import mediapipe as mp
import math
import threading

from cv_bridge import CvBridge


# ─── ROI ─────────────────────────────────────────────────────────────────────
ROI_X1, ROI_Y1 = 0.15, 0.05
ROI_X2, ROI_Y2 = 0.85, 0.95

# ─── Safety ───────────────────────────────────────────────────────────────────
MAX_JUMP         = 0.15
SMOOTH_ALPHA     = 0.85
SMOOTH_ALPHA_YAW = 0.75
LOST_TIMEOUT     = 0.5

# ─── Gesture thresholds ───────────────────────────────────────────────────────
FIST_THRESHOLD  = 0.30
THUMB_ANGLE_MIN = 5.0
THUMB_ANGLE_MAX = 38.0

# ─── Wrist yaw mapping ───────────────────────────────────────────────────────
YAW_MIN_DEG     = -100.0
YAW_MAX_DEG     =   10.0
YAW_ROBOT_SCALE =    1.0

# ─── Colors (BGR) ─────────────────────────────────────────────────────────────
C_ROI      = (0, 255, 0)
C_ACTIVE   = (0, 255, 0)
C_INACTIVE = (0, 0, 255)
C_WARN     = (0, 165, 255)
C_SMOOTHED_R = (255, 255, 0)    # right hand — yellow
C_SMOOTHED_L = (0, 200, 255)    # left hand  — cyan
C_WRIST    = (0, 255, 255)
C_LM       = (255, 100, 100)
C_THUMB    = (0, 200, 255)
C_MIDDLE   = (255, 0, 255)
C_CONN     = (200, 200, 200)
C_YAW      = (255, 180, 0)


def dist2d(a, b):
    return math.sqrt((a.x - b.x)**2 + (a.y - b.y)**2)


class HandTrackerNode(Node):

    def __init__(self):
        super().__init__('hand_tracker_node')

        # ── Publishers — shared ────────────────────────────────────────────
        self.image_pub   = self.create_publisher(Image,       '/hand_tracker/image',       10)
        self.caminfo_pub = self.create_publisher(CameraInfo,  '/hand_tracker/camera_info', 10)

        # ── Publishers — right hand ────────────────────────────────────────
        self.pose_pub    = self.create_publisher(PoseStamped, '/hand_pose/right',          10)
        self.active_pub  = self.create_publisher(Bool,        '/hand_tracker/active',      10)
        self.gripper_pub = self.create_publisher(Float64,     '/hand_tracker/gripper',     10)

        # ── Publishers — left hand ─────────────────────────────────────────
        self.pose_left_pub    = self.create_publisher(PoseStamped, '/hand_pose/left',             10)
        self.active_left_pub  = self.create_publisher(Bool,        '/hand_tracker/left/active',   10)
        self.gripper_left_pub = self.create_publisher(Float64,     '/hand_tracker/left/gripper',  10)

        self.bridge = CvBridge()

        # ── Right hand state ───────────────────────────────────────────────
        self._smooth_x    = 0.5
        self._smooth_y    = 0.5
        self._smooth_yaw  = 0.0
        self._prev_x      = None
        self._prev_y      = None
        self._last_seen   = None
        self._fist        = False
        self._gripper     = 0.0

        # ── Left hand state ────────────────────────────────────────────────
        self._smooth_x_L   = 0.5
        self._smooth_y_L   = 0.5
        self._smooth_yaw_L = 0.0
        self._prev_x_L     = None
        self._prev_y_L     = None
        self._last_seen_L  = None
        self._fist_L       = False
        self._gripper_L    = 0.0

        self._cam_thread = threading.Thread(target=self._camera_loop, daemon=True)
        self._cam_thread.start()

        self.get_logger().info('Hand tracker started.')
        self.get_logger().info(
            f'Fist threshold: {FIST_THRESHOLD} | '
            f'Yaw range: [{YAW_MIN_DEG}°, {YAW_MAX_DEG}°] | '
            f'Thumb: [{THUMB_ANGLE_MIN}°, {THUMB_ANGLE_MAX}°]')

    # ── Gesture & orientation detection ──────────────────────────────────────

    def _is_fist(self, lms) -> bool:
        wrist      = lms[0]
        fingertips = [lms[i] for i in [8, 12, 16, 20]]
        avg_dist   = sum(dist2d(t, wrist) for t in fingertips) / 4
        return avg_dist < FIST_THRESHOLD

    def _thumb_angle(self, lms) -> float:
        wrist     = lms[0]
        index_mcp = lms[5]
        thumb_tip = lms[4]
        palm_vec  = (index_mcp.x - wrist.x, index_mcp.y - wrist.y)
        thumb_vec = (thumb_tip.x  - wrist.x, thumb_tip.y  - wrist.y)
        mag_p = math.sqrt(palm_vec[0]**2 + palm_vec[1]**2)
        mag_t = math.sqrt(thumb_vec[0]**2 + thumb_vec[1]**2)
        if mag_p < 1e-6 or mag_t < 1e-6:
            return 0.0
        dot   = palm_vec[0]*thumb_vec[0] + palm_vec[1]*thumb_vec[1]
        cos_a = max(-1.0, min(1.0, dot / (mag_p * mag_t)))
        return math.degrees(math.acos(cos_a))

    def _gripper_from_angle(self, angle: float) -> float:
        g = (angle - THUMB_ANGLE_MIN) / (THUMB_ANGLE_MAX - THUMB_ANGLE_MIN)
        return max(0.0, min(1.0, g))

    def _hand_yaw_deg(self, lms) -> float:
        wrist      = lms[0]
        middle_mcp = lms[9]
        return math.degrees(math.atan2(
            middle_mcp.y - wrist.y,
            middle_mcp.x - wrist.x
        ))

    def _normalize_yaw(self, yaw_deg: float) -> float:
        normalized = (yaw_deg - YAW_MIN_DEG) / (YAW_MAX_DEG - YAW_MIN_DEG)
        normalized = max(0.0, min(1.0, normalized))
        normalized = (normalized * 2.0) - 1.0
        return normalized * YAW_ROBOT_SCALE

    # ── Safety filters ────────────────────────────────────────────────────────

    def _in_roi(self, x, y):
        return ROI_X1 < x < ROI_X2 and ROI_Y1 < y < ROI_Y2

    def _jump_ok(self, x, y, prev_x, prev_y):
        if prev_x is None:
            return True
        return math.sqrt((x - prev_x)**2 + (y - prev_y)**2) < MAX_JUMP

    # ── Drawing ───────────────────────────────────────────────────────────────

    def _draw_landmarks(self, frame, hand_landmarks, w, h):
        lms        = hand_landmarks.landmark
        fingertips = {4, 8, 12, 16, 20}
        for conn in mp.solutions.hands.HAND_CONNECTIONS:
            p1, p2 = lms[conn[0]], lms[conn[1]]
            cv2.line(frame,
                     (int(p1.x * w), int(p1.y * h)),
                     (int(p2.x * w), int(p2.y * h)),
                     C_CONN, 1)
        for i, lm in enumerate(lms):
            x, y = int(lm.x * w), int(lm.y * h)
            if i == 0:   color, size = C_WRIST,  10
            elif i == 4: color, size = C_THUMB,   8
            elif i == 9: color, size = C_MIDDLE,  8
            elif i in fingertips: color, size = C_LM, 7
            else:        color, size = C_LM,       4
            cv2.circle(frame, (x, y), size, color, -1)
            cv2.circle(frame, (x, y), size, (0, 0, 0), 1)

    def _draw_smoothed(self, frame, w, h, sx, sy, syaw, color):
        px, py = int(sx * w), int(sy * h)
        cv2.line(frame, (px - 20, py), (px + 20, py), color, 2)
        cv2.line(frame, (px, py - 20), (px, py + 20), color, 2)
        cv2.circle(frame, (px, py), 5, color, -1)
        cv2.putText(frame,
            f'xy:({sx:.2f},{sy:.2f}) yaw:{syaw:.2f}',
            (px + 12, py - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    def _draw_roi(self, frame, w, h):
        pt1 = (int(ROI_X1 * w), int(ROI_Y1 * h))
        pt2 = (int(ROI_X2 * w), int(ROI_Y2 * h))
        cv2.rectangle(frame, pt1, pt2, C_ROI, 2)
        cv2.putText(frame, 'ROI', (pt1[0] + 4, pt1[1] + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, C_ROI, 2)

    def _draw_info_bar(self, frame, w, h, fist_r, fist_l, gripper_r, gripper_l, yaw_deg):
        cv2.rectangle(frame, (0, 0), (w, 55), (30, 30, 30), -1)

        # Right hand fist state
        if fist_r:
            cv2.putText(frame, 'R:✊ MOVING', (10, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, C_ACTIVE, 2)
        else:
            cv2.putText(frame, 'R:🖐 STOP', (10, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, C_INACTIVE, 2)

        # Left hand fist state
        if fist_l:
            cv2.putText(frame, 'L:✊ MOVING', (10, 46),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, C_ACTIVE, 2)
        else:
            cv2.putText(frame, 'L:🖐 STOP', (10, 46),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, C_INACTIVE, 2)

        # Right gripper bar
        bx, by, bw, bh = w - 230, 5, 80, 12
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (80, 80, 80), -1)
        fill  = int(gripper_r * bw)
        color = (0, int(255 * gripper_r), int(255 * (1 - gripper_r)))
        cv2.rectangle(frame, (bx, by), (bx + fill, by + bh), color, -1)
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (200, 200, 200), 1)
        cv2.putText(frame, f'Rgrip:{gripper_r:.2f}', (bx, by + bh + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

        # Left gripper bar
        bx2 = w - 140
        cv2.rectangle(frame, (bx2, by), (bx2 + bw, by + bh), (80, 80, 80), -1)
        fill2  = int(gripper_l * bw)
        color2 = (int(255 * (1 - gripper_l)), int(255 * gripper_l), 0)
        cv2.rectangle(frame, (bx2, by), (bx2 + fill2, by + bh), color2, -1)
        cv2.rectangle(frame, (bx2, by), (bx2 + bw, by + bh), (200, 200, 200), 1)
        cv2.putText(frame, f'Lgrip:{gripper_l:.2f}', (bx2, by + bh + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

    def _draw_status(self, frame, w, h, r_detected, l_detected, warning=''):
        cv2.rectangle(frame, (0, h - 36), (w, h), (30, 30, 30), -1)
        status = f'R:{"OK" if r_detected else "--"}  L:{"OK" if l_detected else "--"}'
        cv2.putText(frame, status, (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, C_ACTIVE, 2)
        if warning:
            cv2.putText(frame, warning, (w // 2 - 60, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, C_WARN, 1)

    # ── Publishing ────────────────────────────────────────────────────────────

    def _publish_hand(self, x, y, yaw, active, gripper, right: bool):
        stamp = self.get_clock().now().to_msg()
        pose = PoseStamped()
        pose.header.stamp    = stamp
        pose.header.frame_id = 'camera'
        pose.pose.position.x = float(1.0 - x)
        pose.pose.position.y = float(1.0 - y)
        pose.pose.position.z = float(yaw)
        pose.pose.orientation.w = 1.0
        a = Bool();   a.data = active
        g = Float64(); g.data = float(gripper)
        if right:
            self.pose_pub.publish(pose)
            self.active_pub.publish(a)
            self.gripper_pub.publish(g)
        else:
            self.pose_left_pub.publish(pose)
            self.active_left_pub.publish(a)
            self.gripper_left_pub.publish(g)

    def _publish_image(self, frame):
        try:
            stamp = self.get_clock().now().to_msg()
            img = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            img.header.stamp    = stamp
            img.header.frame_id = 'camera'
            self.image_pub.publish(img)
            info = CameraInfo()
            info.header.stamp    = stamp
            info.header.frame_id = 'camera'
            info.width  = frame.shape[1]
            info.height = frame.shape[0]
            self.caminfo_pub.publish(info)
        except Exception as e:
            self.get_logger().warn(str(e), throttle_duration_sec=5.0)

    # ── Camera loop ───────────────────────────────────────────────────────────

    def _camera_loop(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.get_logger().error('Could not open webcam!')
            return

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.get_logger().info(f'Camera: {w}x{h}')

        with mp.solutions.hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as hands:

            while rclpy.ok():
                ret, frame = cap.read()
                if not ret:
                    break

                frame   = cv2.flip(frame, 1)
                rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)

                self._draw_roi(frame, w, h)

                # ── Separate detected hands by label ──────────────────────
                right_hand = left_hand = None
                if results.multi_hand_landmarks and results.multi_handedness:
                    for hand_lms, handedness in zip(
                            results.multi_hand_landmarks,
                            results.multi_handedness):
                        label = handedness.classification[0].label
                        if label == 'Right':
                            right_hand = hand_lms
                        elif label == 'Left':
                            left_hand = hand_lms

                warning       = ''
                r_detected    = False
                l_detected    = False

                # ── Process right hand ────────────────────────────────────
                if right_hand:
                    lms  = right_hand.landmark
                    x, y = lms[0].x, lms[0].y
                    in_roi   = self._in_roi(x, y)
                    jump_ok  = self._jump_ok(x, y, self._prev_x, self._prev_y)
                    self._draw_landmarks(frame, right_hand, w, h)
                    if in_roi and jump_ok:
                        self._prev_x    = x
                        self._prev_y    = y
                        self._last_seen = self.get_clock().now()
                        r_detected      = True
                        fist     = self._is_fist(lms)
                        yaw_deg  = self._hand_yaw_deg(lms)
                        yaw_norm = self._normalize_yaw(yaw_deg)
                        gripper  = self._gripper_from_angle(
                            self._thumb_angle(lms)) if fist else 0.0
                        self._smooth_x   = SMOOTH_ALPHA * x + (1-SMOOTH_ALPHA) * self._smooth_x
                        self._smooth_y   = SMOOTH_ALPHA * y + (1-SMOOTH_ALPHA) * self._smooth_y
                        self._smooth_yaw = SMOOTH_ALPHA_YAW * yaw_norm + (1-SMOOTH_ALPHA_YAW) * self._smooth_yaw
                        self._fist    = fist
                        self._gripper = gripper
                        self._draw_smoothed(frame, w, h,
                            self._smooth_x, self._smooth_y, self._smooth_yaw, C_SMOOTHED_R)
                        self._publish_hand(
                            self._smooth_x, self._smooth_y, self._smooth_yaw,
                            active=fist, gripper=gripper, right=True)
                    elif not in_roi:
                        warning = 'R:OUTSIDE ROI'
                        self._publish_hand(0.5, 0.5, 0.0, False, 0.0, right=True)
                    else:
                        warning = 'R:JUMP'
                        self._publish_hand(
                            self._smooth_x, self._smooth_y, self._smooth_yaw,
                            active=self._fist, gripper=self._gripper, right=True)
                else:
                    if self._last_seen is not None:
                        elapsed = (self.get_clock().now() - self._last_seen).nanoseconds / 1e9
                        if elapsed > LOST_TIMEOUT:
                            self._publish_hand(0.5, 0.5, 0.0, False, 0.0, right=True)
                    else:
                        self._publish_hand(0.5, 0.5, 0.0, False, 0.0, right=True)

                # ── Process left hand ─────────────────────────────────────
                if left_hand:
                    lms  = left_hand.landmark
                    x, y = lms[0].x, lms[0].y
                    in_roi   = self._in_roi(x, y)
                    jump_ok  = self._jump_ok(x, y, self._prev_x_L, self._prev_y_L)
                    self._draw_landmarks(frame, left_hand, w, h)
                    if in_roi and jump_ok:
                        self._prev_x_L    = x
                        self._prev_y_L    = y
                        self._last_seen_L = self.get_clock().now()
                        l_detected        = True
                        fist_l   = self._is_fist(lms)
                        yaw_deg  = self._hand_yaw_deg(lms)
                        yaw_norm = self._normalize_yaw(yaw_deg)
                        gripper_l = self._gripper_from_angle(
                            self._thumb_angle(lms)) if fist_l else 0.0
                        self._smooth_x_L   = SMOOTH_ALPHA * x + (1-SMOOTH_ALPHA) * self._smooth_x_L
                        self._smooth_y_L   = SMOOTH_ALPHA * y + (1-SMOOTH_ALPHA) * self._smooth_y_L
                        self._smooth_yaw_L = SMOOTH_ALPHA_YAW * yaw_norm + (1-SMOOTH_ALPHA_YAW) * self._smooth_yaw_L
                        self._fist_L    = fist_l
                        self._gripper_L = gripper_l
                        self._draw_smoothed(frame, w, h,
                            self._smooth_x_L, self._smooth_y_L, self._smooth_yaw_L, C_SMOOTHED_L)
                        self._publish_hand(
                            self._smooth_x_L, self._smooth_y_L, self._smooth_yaw_L,
                            active=fist_l, gripper=gripper_l, right=False)
                    elif not in_roi:
                        self._publish_hand(0.5, 0.5, 0.0, False, 0.0, right=False)
                    else:
                        self._publish_hand(
                            self._smooth_x_L, self._smooth_y_L, self._smooth_yaw_L,
                            active=self._fist_L, gripper=self._gripper_L, right=False)
                else:
                    if self._last_seen_L is not None:
                        elapsed = (self.get_clock().now() - self._last_seen_L).nanoseconds / 1e9
                        if elapsed > LOST_TIMEOUT:
                            self._publish_hand(0.5, 0.5, 0.0, False, 0.0, right=False)
                    else:
                        self._publish_hand(0.5, 0.5, 0.0, False, 0.0, right=False)

                self._draw_info_bar(frame, w, h,
                    self._fist, self._fist_L,
                    self._gripper, self._gripper_L, 0.0)
                self._draw_status(frame, w, h, r_detected, l_detected, warning)
                self._publish_image(frame)

                cv2.imshow('Hand Tracker', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap.release()
        cv2.destroyAllWindows()


def main(args=None):
    rclpy.init(args=args)
    node = HandTrackerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
