#!/usr/bin/env python3
"""
Hand Tracker Node — Phase 2 + Wrist Orientation
=================================================
Tracks right hand with:
  - XY position (wrist landmark 0)
  - Wrist orientation/yaw (wrist→middle_mcp angle)
  - Fist detection (avg fingertip distance)
  - Continuous gripper control (thumb angle)

Gesture logic:
  Hand open  (avg_dist > 0.30) → robot STOPS
  Fist closed (avg_dist < 0.30) → robot MOVES
    └── hand yaw    → robot wrist rotation
    └── thumb angle → gripper opening [0.0 → 1.0]

Published topics:
  /hand_tracker/image       (sensor_msgs/Image)        — annotated feed for RViz
  /hand_tracker/camera_info (sensor_msgs/CameraInfo)   — required by RViz
  /hand_pose/right          (geometry_msgs/PoseStamped)— x, y, yaw
  /hand_tracker/active      (std_msgs/Bool)            — True when fist closed
  /hand_tracker/gripper     (std_msgs/Float64)         — gripper 0.0=closed 1.0=open

Wrist yaw mapping:
  Raw hand yaw: [-100°, 10°] → normalized to [-1, 1] → scaled to robot wrist range
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
MAX_JUMP        = 0.15
SMOOTH_ALPHA    = 0.85
SMOOTH_ALPHA_YAW = 0.75
LOST_TIMEOUT    = 0.5

# ─── Gesture thresholds ───────────────────────────────────────────────────────
FIST_THRESHOLD  = 0.30
THUMB_ANGLE_MIN = 5.0    # degrees — thumb tucked → gripper closed
THUMB_ANGLE_MAX = 38.0   # degrees — thumb extended → gripper open

# ─── Wrist yaw mapping ───────────────────────────────────────────────────────
YAW_MIN_DEG = -100.0     # hand yaw at one extreme
YAW_MAX_DEG =   10.0     # hand yaw at other extreme
YAW_ROBOT_SCALE = 1.0    # scale factor for robot wrist — tune if needed

# ─── Colors (BGR) ─────────────────────────────────────────────────────────────
C_ROI      = (0, 255, 0)
C_ACTIVE   = (0, 255, 0)
C_INACTIVE = (0, 0, 255)
C_WARN     = (0, 165, 255)
C_SMOOTHED = (255, 255, 0)
C_WRIST    = (0, 255, 255)
C_LM       = (255, 100, 100)
C_THUMB    = (0, 200, 255)
C_MIDDLE   = (255, 0, 255)   # middle MCP — used for yaw visualization
C_CONN     = (200, 200, 200)
C_YAW      = (255, 180, 0)


def dist2d(a, b):
    return math.sqrt((a.x - b.x)**2 + (a.y - b.y)**2)


class HandTrackerNode(Node):

    def __init__(self):
        super().__init__('hand_tracker_node')

        # ── Publishers ────────────────────────────────────────────────────
        self.image_pub   = self.create_publisher(Image,       '/hand_tracker/image',       10)
        self.caminfo_pub = self.create_publisher(CameraInfo,  '/hand_tracker/camera_info', 10)
        self.pose_pub    = self.create_publisher(PoseStamped, '/hand_pose/right',          10)
        self.active_pub  = self.create_publisher(Bool,        '/hand_tracker/active',      10)
        self.gripper_pub = self.create_publisher(Float64,     '/hand_tracker/gripper',     10)

        self.bridge = CvBridge()

        # ── State ─────────────────────────────────────────────────────────
        self._smooth_x   = 0.5
        self._smooth_y   = 0.5
        self._smooth_yaw = 0.0   # normalized [-1, 1]
        self._prev_x     = None
        self._prev_y     = None
        self._last_seen  = None
        self._fist       = False
        self._gripper    = 0.0

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
        """
        Compute hand yaw from wrist(0) → middle_mcp(9) vector.
        Returns angle in degrees.
        """
        wrist      = lms[0]
        middle_mcp = lms[9]
        return math.degrees(math.atan2(
            middle_mcp.y - wrist.y,
            middle_mcp.x - wrist.x
        ))

    def _normalize_yaw(self, yaw_deg: float) -> float:
        """
        Map hand yaw [YAW_MIN_DEG, YAW_MAX_DEG] → [-1.0, 1.0]
        then scale by YAW_ROBOT_SCALE for robot wrist velocity.
        """
        normalized = (yaw_deg - YAW_MIN_DEG) / (YAW_MAX_DEG - YAW_MIN_DEG)
        normalized = max(0.0, min(1.0, normalized))  # clamp [0, 1]
        normalized = (normalized * 2.0) - 1.0        # shift to [-1, 1]
        return normalized * YAW_ROBOT_SCALE

    # ── Safety filters ────────────────────────────────────────────────────────

    def _in_roi(self, x, y):
        return ROI_X1 < x < ROI_X2 and ROI_Y1 < y < ROI_Y2

    def _jump_ok(self, x, y):
        if self._prev_x is None:
            return True
        return math.sqrt((x - self._prev_x)**2 + (y - self._prev_y)**2) < MAX_JUMP

    def _smooth(self, x, y, yaw):
        self._smooth_x   = SMOOTH_ALPHA     * x   + (1 - SMOOTH_ALPHA)     * self._smooth_x
        self._smooth_y   = SMOOTH_ALPHA     * y   + (1 - SMOOTH_ALPHA)     * self._smooth_y
        self._smooth_yaw = SMOOTH_ALPHA_YAW * yaw + (1 - SMOOTH_ALPHA_YAW) * self._smooth_yaw

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
            if i == 0:
                color, size = C_WRIST, 10
            elif i == 4:
                color, size = C_THUMB, 8
            elif i == 9:
                color, size = C_MIDDLE, 8   # middle MCP — yaw reference
            elif i in fingertips:
                color, size = C_LM, 7
            else:
                color, size = C_LM, 4
            cv2.circle(frame, (x, y), size, color, -1)
            cv2.circle(frame, (x, y), size, (0, 0, 0), 1)
            cv2.putText(frame, str(i), (x + 5, y - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, (255, 255, 255), 1)

    def _draw_yaw_arrow(self, frame, lms, w, h, yaw_deg):
        """Draw arrow showing hand yaw direction."""
        wrist = lms[0]
        wx, wy = int(wrist.x * w), int(wrist.y * h)
        length = 60
        end_x = int(wx + length * math.cos(math.radians(yaw_deg)))
        end_y = int(wy + length * math.sin(math.radians(yaw_deg)))
        cv2.arrowedLine(frame, (wx, wy), (end_x, end_y), C_YAW, 3, tipLength=0.3)
        cv2.putText(frame, f'{yaw_deg:.0f}°',
                    (wx + 15, wy - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_YAW, 1)

    def _draw_smoothed(self, frame, w, h):
        sx, sy = int(self._smooth_x * w), int(self._smooth_y * h)
        cv2.line(frame, (sx - 20, sy), (sx + 20, sy), C_SMOOTHED, 2)
        cv2.line(frame, (sx, sy - 20), (sx, sy + 20), C_SMOOTHED, 2)
        cv2.circle(frame, (sx, sy), 5, C_SMOOTHED, -1)
        cv2.putText(frame,
            f'xy:({self._smooth_x:.2f},{self._smooth_y:.2f}) '
            f'yaw:{self._smooth_yaw:.2f}',
            (sx + 12, sy - 8), cv2.FONT_HERSHEY_SIMPLEX,
            0.4, C_SMOOTHED, 1)

    def _draw_roi(self, frame, w, h):
        pt1 = (int(ROI_X1 * w), int(ROI_Y1 * h))
        pt2 = (int(ROI_X2 * w), int(ROI_Y2 * h))
        cv2.rectangle(frame, pt1, pt2, C_ROI, 2)
        cv2.putText(frame, 'ROI', (pt1[0] + 4, pt1[1] + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, C_ROI, 2)

    def _draw_info_bar(self, frame, w, h, fist, yaw_deg, gripper):
        """Top bar: fist state + gripper bar + yaw indicator."""
        cv2.rectangle(frame, (0, 0), (w, 55), (30, 30, 30), -1)

        # Fist state
        if fist:
            cv2.putText(frame, '✊ FIST — MOVING', (10, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, C_ACTIVE, 2)
        else:
            cv2.putText(frame, '🖐 OPEN — STOPPED', (10, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, C_INACTIVE, 2)

        # Gripper bar
        bx, by, bw, bh = w - 230, 8, 100, 16
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (80, 80, 80), -1)
        fill  = int(gripper * bw)
        color = (0, int(255 * gripper), int(255 * (1 - gripper)))
        cv2.rectangle(frame, (bx, by), (bx + fill, by + bh), color, -1)
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (200, 200, 200), 1)
        cv2.putText(frame, f'grip:{gripper:.2f}', (bx, by + bh + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        # Yaw bar
        yx, yy, yw, yh = w - 120, 8, 100, 16
        cv2.rectangle(frame, (yx, yy), (yx + yw, yy + yh), (80, 80, 80), -1)
        # Center marker
        center_x = yx + yw // 2
        norm = (yaw_deg - YAW_MIN_DEG) / (YAW_MAX_DEG - YAW_MIN_DEG)
        norm = max(0.0, min(1.0, norm))
        marker_x = int(yx + norm * yw)
        cv2.rectangle(frame, (yx, yy), (yx + yw, yy + yh), (80, 80, 80), -1)
        cv2.line(frame, (center_x, yy), (center_x, yy + yh), (100, 100, 100), 1)
        cv2.circle(frame, (marker_x, yy + yh // 2), 6, C_YAW, -1)
        cv2.rectangle(frame, (yx, yy), (yx + yw, yy + yh), (200, 200, 200), 1)
        cv2.putText(frame, f'yaw:{yaw_deg:.0f}°', (yx, yy + yh + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, C_YAW, 1)

    def _draw_status(self, frame, w, h, tracking, warning=''):
        cv2.rectangle(frame, (0, h - 36), (w, h), (30, 30, 30), -1)
        status = 'HAND DETECTED' if tracking else 'WAITING FOR HAND'
        cv2.putText(frame, status, (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    C_ACTIVE if tracking else C_INACTIVE, 2)
        if warning:
            cv2.putText(frame, warning, (w // 2 - 60, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, C_WARN, 1)

    # ── Publishing ────────────────────────────────────────────────────────────

    def _publish_all(self, x, y, yaw, active, gripper):
        stamp = self.get_clock().now().to_msg()

        # Hand pose — x, y position + yaw in orientation z
        pose = PoseStamped()
        pose.header.stamp    = stamp
        pose.header.frame_id = 'camera'
        pose.pose.position.x = float(1.0 - x)    # flip horizontal
        pose.pose.position.y = float(1.0 - y)    # flip vertical
        pose.pose.position.z = float(yaw)         # pack yaw into z for simplicity
        pose.pose.orientation.w = 1.0
        self.pose_pub.publish(pose)

        # Active
        a = Bool(); a.data = active
        self.active_pub.publish(a)

        # Gripper
        g = Float64(); g.data = float(gripper)
        self.gripper_pub.publish(g)

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
            max_num_hands=1,
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

                hand_detected = False
                warning       = ''
                fist          = False
                yaw_deg       = 0.0
                gripper       = 0.0

                # Extract right hand from multi-hand results
                right_hand = None
                if results.multi_hand_landmarks and results.multi_handedness:
                    for hand_lms, handedness in zip(
                            results.multi_hand_landmarks,
                            results.multi_handedness):
                        if handedness.classification[0].label == 'Right':
                            right_hand = hand_lms
                            break

                if right_hand:
                    lms  = right_hand.landmark
                    x, y = lms[0].x, lms[0].y

                    in_roi  = self._in_roi(x, y)
                    jump_ok = self._jump_ok(x, y)

                    self._draw_landmarks(frame, right_hand, w, h)

                    if in_roi and jump_ok:
                        self._prev_x    = x
                        self._prev_y    = y
                        self._last_seen = self.get_clock().now()
                        hand_detected   = True

                        # Compute all features
                        fist    = self._is_fist(lms)
                        yaw_deg = self._hand_yaw_deg(lms)
                        yaw_norm = self._normalize_yaw(yaw_deg)
                        gripper = self._gripper_from_angle(
                            self._thumb_angle(lms)) if fist else 0.0

                        # Smooth all
                        self._smooth(x, y, yaw_norm)
                        self._fist    = fist
                        self._gripper = gripper

                        self._draw_smoothed(frame, w, h)
                        self._draw_yaw_arrow(frame, lms, w, h, yaw_deg)

                        self._publish_all(
                            self._smooth_x,
                            self._smooth_y,
                            self._smooth_yaw,
                            active=fist,
                            gripper=gripper
                        )

                    elif not in_roi:
                        warning = 'OUTSIDE ROI'
                        self._publish_all(0.5, 0.5, 0.0, False, 0.0)
                    else:
                        warning = 'JUMP'
                        self._publish_all(
                            self._smooth_x, self._smooth_y,
                            self._smooth_yaw,
                            active=self._fist, gripper=self._gripper
                        )
                else:
                    if self._last_seen is not None:
                        elapsed = (self.get_clock().now() - self._last_seen
                                   ).nanoseconds / 1e9
                        if elapsed > LOST_TIMEOUT:
                            self._publish_all(0.5, 0.5, 0.0, False, 0.0)
                    else:
                        self._publish_all(0.5, 0.5, 0.0, False, 0.0)

                self._draw_info_bar(frame, w, h, self._fist, yaw_deg, self._gripper)
                self._draw_status(frame, w, h, hand_detected, warning)
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