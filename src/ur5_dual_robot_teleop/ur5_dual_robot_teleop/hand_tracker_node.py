#!/usr/bin/env python3
"""
Hand Tracker Node — Phase 1 with RViz visualization
=====================================================
Tracks right hand using MediaPipe Holistic, publishes:
  - /hand_tracker/image       (sensor_msgs/Image)      — annotated camera feed for RViz
  - /hand_tracker/camera_info (sensor_msgs/CameraInfo) — required by RViz image display
  - /hand_pose/right          (geometry_msgs/PoseStamped) — hand pose for robot control
  - /hand_tracker/active      (std_msgs/Bool)          — tracking status

Visualized features:
  - All 21 hand landmarks with connections
  - ROI box (green)
  - Wrist position (yellow dot)
  - Smoothed position crosshair (cyan)
  - Jump/ROI rejection warnings
  - Status bar

Safety features:
  - ROI filter: only accept hand inside defined screen zone
  - Jump filter: reject sudden position jumps
  - Temporal smoothing: exponential moving average

Requirements:
    pip install mediapipe opencv-python
    sudo apt install ros-humble-cv-bridge ros-humble-vision-opencv
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool
from sensor_msgs.msg import Image, CameraInfo

import cv2
import mediapipe as mp
import numpy as np
import threading
import math

try:
    from cv_bridge import CvBridge
except ImportError:
    raise ImportError("Install cv_bridge: sudo apt install ros-humble-cv-bridge")


# ─── ROI definition (normalized 0-1 screen coordinates) ─────────────────────
ROI_X1, ROI_Y1 = 0.15, 0.05
ROI_X2, ROI_Y2 = 0.85, 0.95

# ─── Safety thresholds ───────────────────────────────────────────────────────
MAX_JUMP      = 0.15
SMOOTH_ALPHA  = 0.4
LOST_TIMEOUT  = 0.5

# ─── Visualization colors (BGR) ──────────────────────────────────────────────
COLOR_ROI         = (0, 255, 0)
COLOR_ACTIVE      = (0, 255, 0)
COLOR_INACTIVE    = (0, 0, 255)
COLOR_WARNING     = (0, 165, 255)
COLOR_SMOOTHED    = (255, 255, 0)
COLOR_WRIST       = (0, 255, 255)
COLOR_LANDMARKS   = (255, 100, 100)
COLOR_CONNECTIONS = (200, 200, 200)


class HandTrackerNode(Node):

    def __init__(self):
        super().__init__('hand_tracker_node')

        # ── Publishers ────────────────────────────────────────────────────
        self.image_pub   = self.create_publisher(Image,       '/hand_tracker/image',       10)
        self.caminfo_pub = self.create_publisher(CameraInfo,  '/hand_tracker/camera_info', 10)
        self.pose_pub    = self.create_publisher(PoseStamped, '/hand_pose/right',          10)
        self.active_pub  = self.create_publisher(Bool,        '/hand_tracker/active',      10)

        self.bridge = CvBridge()

        # ── State ─────────────────────────────────────────────────────────
        self._smooth_x  = 0.5
        self._smooth_y  = 0.5
        self._prev_x    = None
        self._prev_y    = None
        self._last_seen = None

        # ── Start camera thread ───────────────────────────────────────────
        self._cam_thread = threading.Thread(target=self._camera_loop, daemon=True)
        self._cam_thread.start()

        self.get_logger().info('Hand tracker node started.')
        self.get_logger().info('In RViz: Add → By topic → /hand_tracker/image → Image')

    # ── Safety filters ────────────────────────────────────────────────────────

    def _in_roi(self, x, y):
        return ROI_X1 < x < ROI_X2 and ROI_Y1 < y < ROI_Y2

    def _jump_ok(self, x, y):
        if self._prev_x is None:
            return True
        return math.sqrt((x - self._prev_x)**2 + (y - self._prev_y)**2) < MAX_JUMP

    def _smooth(self, x, y):
        self._smooth_x = SMOOTH_ALPHA * x + (1 - SMOOTH_ALPHA) * self._smooth_x
        self._smooth_y = SMOOTH_ALPHA * y + (1 - SMOOTH_ALPHA) * self._smooth_y

    # ── Drawing ───────────────────────────────────────────────────────────────

    def _draw_landmarks(self, frame, hand_landmarks, w, h):
        lms = hand_landmarks.landmark
        fingertips = {4, 8, 12, 16, 20}

        # Connections
        for conn in mp.solutions.holistic.HAND_CONNECTIONS:
            p1 = lms[conn[0]]
            p2 = lms[conn[1]]
            cv2.line(frame,
                     (int(p1.x * w), int(p1.y * h)),
                     (int(p2.x * w), int(p2.y * h)),
                     COLOR_CONNECTIONS, 1)

        # Landmarks
        for i, lm in enumerate(lms):
            x, y = int(lm.x * w), int(lm.y * h)
            if i == 0:
                cv2.circle(frame, (x, y), 10, COLOR_WRIST, -1)
                cv2.circle(frame, (x, y), 10, (0, 0, 0), 1)
            elif i in fingertips:
                cv2.circle(frame, (x, y), 7, COLOR_LANDMARKS, -1)
                cv2.circle(frame, (x, y), 7, (0, 0, 0), 1)
            else:
                cv2.circle(frame, (x, y), 4, COLOR_LANDMARKS, -1)
            cv2.putText(frame, str(i), (x + 6, y - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, (255, 255, 255), 1)

    def _draw_smoothed(self, frame, w, h):
        sx = int(self._smooth_x * w)
        sy = int(self._smooth_y * h)
        cv2.line(frame, (sx - 20, sy), (sx + 20, sy), COLOR_SMOOTHED, 2)
        cv2.line(frame, (sx, sy - 20), (sx, sy + 20), COLOR_SMOOTHED, 2)
        cv2.circle(frame, (sx, sy), 5, COLOR_SMOOTHED, -1)
        cv2.putText(frame, f'({self._smooth_x:.2f}, {self._smooth_y:.2f})',
                    (sx + 15, sy - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, COLOR_SMOOTHED, 1)

    def _draw_roi(self, frame, w, h):
        pt1 = (int(ROI_X1 * w), int(ROI_Y1 * h))
        pt2 = (int(ROI_X2 * w), int(ROI_Y2 * h))
        cv2.rectangle(frame, pt1, pt2, COLOR_ROI, 2)
        cv2.putText(frame, 'ROI', (pt1[0] + 4, pt1[1] + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_ROI, 2)

    def _draw_status(self, frame, w, h, tracking, msg=''):
        cv2.rectangle(frame, (0, h - 38), (w, h), (30, 30, 30), -1)
        status = 'TRACKING' if tracking else 'WAITING FOR HAND'
        cv2.putText(frame, status, (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    COLOR_ACTIVE if tracking else COLOR_INACTIVE, 2)
        if msg:
            cv2.putText(frame, msg, (w // 2 - 80, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WARNING, 2)
        cv2.putText(frame,
            f'pos: ({self._smooth_x:.3f}, {self._smooth_y:.3f})',
            (w - 220, h - 10), cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (200, 200, 200), 1)

    # ── Publishing ────────────────────────────────────────────────────────────

    def _publish_image(self, frame):
        try:
            stamp = self.get_clock().now().to_msg()
            img_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            img_msg.header.stamp    = stamp
            img_msg.header.frame_id = 'camera'
            self.image_pub.publish(img_msg)

            info = CameraInfo()
            info.header.stamp    = stamp
            info.header.frame_id = 'camera'
            info.width  = frame.shape[1]
            info.height = frame.shape[0]
            self.caminfo_pub.publish(info)
        except Exception as e:
            self.get_logger().warn(f'Image publish error: {e}', throttle_duration_sec=5.0)

    def _publish_pose(self, x, y, active):
        pose = PoseStamped()
        pose.header.stamp    = self.get_clock().now().to_msg()
        pose.header.frame_id = 'camera'
        pose.pose.position.x = float(1.0 - x)
        pose.pose.position.y = float(1.0 - y)
        pose.pose.position.z = 0.0
        pose.pose.orientation.w = 1.0
        self.pose_pub.publish(pose)

        msg = Bool()
        msg.data = active
        self.active_pub.publish(msg)

    # ── Camera loop ───────────────────────────────────────────────────────────

    def _camera_loop(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.get_logger().error('Could not open webcam!')
            return

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.get_logger().info(f'Camera: {w}x{h}')

        with mp.solutions.holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as holistic:

            while rclpy.ok():
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)
                rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(rgb)

                self._draw_roi(frame, w, h)

                hand_detected = False
                warning       = ''

                if results.right_hand_landmarks:
                    lm = results.right_hand_landmarks.landmark[0]
                    x, y = lm.x, lm.y

                    in_roi  = self._in_roi(x, y)
                    jump_ok = self._jump_ok(x, y)

                    # Always draw landmarks so user can see hand
                    self._draw_landmarks(frame, results.right_hand_landmarks, w, h)

                    if in_roi and jump_ok:
                        self._smooth(x, y)
                        self._prev_x    = x
                        self._prev_y    = y
                        self._last_seen = self.get_clock().now()
                        hand_detected   = True
                        self._draw_smoothed(frame, w, h)
                        self._publish_pose(self._smooth_x, self._smooth_y, True)
                    elif not in_roi:
                        warning = 'OUTSIDE ROI'
                        self._publish_pose(0.5, 0.5, False)
                    else:
                        warning = 'JUMP DETECTED'
                        self._publish_pose(self._smooth_x, self._smooth_y, True)
                else:
                    if self._last_seen is not None:
                        elapsed = (self.get_clock().now() - self._last_seen).nanoseconds / 1e9
                        if elapsed > LOST_TIMEOUT:
                            self._publish_pose(0.5, 0.5, False)
                    else:
                        self._publish_pose(0.5, 0.5, False)

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