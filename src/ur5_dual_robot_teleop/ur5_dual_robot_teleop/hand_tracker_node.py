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

Threading model:
  Camera loop runs in a daemon thread — does all CV work, writes snapshots
  to a size-1 queue. A ROS2 timer on the executor thread drains the queue
  and publishes. No ROS APIs are called from the camera thread.
"""

import math
import queue
import threading
import time

import cv2
import mediapipe as mp
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool, Float64
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge

from ur5_dual_robot_teleop.teleop_config import CONFIG

# ─── Load parameters from config ─────────────────────────────────────────────
_ht  = CONFIG['hand_tracker']
_roi = _ht['roi']
_smo = _ht['smoothing']
_saf = _ht['safety']
_ges = _ht['gestures']

ROI_X1, ROI_Y1   = _roi['x1'], _roi['y1']
ROI_X2, ROI_Y2   = _roi['x2'], _roi['y2']
MAX_JUMP         = _saf['max_jump']
SMOOTH_ALPHA     = _smo['alpha_xy']
SMOOTH_ALPHA_YAW = _smo['alpha_yaw']
LOST_TIMEOUT     = _saf['lost_timeout']
FIST_THRESHOLD   = _ges['fist_threshold']
THUMB_ANGLE_MIN  = _ges['thumb_angle_min']
THUMB_ANGLE_MAX  = _ges['thumb_angle_max']
SHOW_WINDOW      = _ht.get('show_window', True)
FPS_CAP          = _ht.get('fps_cap', 20)
MODEL_COMPLEXITY = _ht.get('model_complexity', 0)
CAM_WIDTH        = _ht.get('cam_width', 640)
CAM_HEIGHT       = _ht.get('cam_height', 480)

# ─── Colors (BGR) ─────────────────────────────────────────────────────────────
C_ROI        = (0, 255, 0)
C_ACTIVE     = (0, 255, 0)
C_INACTIVE   = (0, 0, 255)
C_WARN       = (0, 165, 255)
C_SMOOTHED_R = (255, 255, 0)
C_SMOOTHED_L = (0, 200, 255)
C_WRIST      = (0, 255, 255)
C_LM         = (255, 100, 100)
C_THUMB      = (0, 200, 255)
C_MIDDLE     = (255, 0, 255)
C_CONN       = (200, 200, 200)


def dist2d(a, b):
    return math.sqrt((a.x - b.x)**2 + (a.y - b.y)**2)


class _Snapshot:
    """Data produced by the camera thread, consumed by the publish timer."""
    __slots__ = (
        'rx', 'ry', 'ryaw', 'r_active', 'r_gripper',
        'lx', 'ly', 'lyaw', 'l_active', 'l_gripper',
        'frame',
    )

    def __init__(self):
        self.rx = self.ry = 0.5; self.ryaw = 0.0
        self.r_active = False;   self.r_gripper = 0.0
        self.lx = self.ly = 0.5; self.lyaw = 0.0
        self.l_active = False;   self.l_gripper = 0.0
        self.frame = None


class HandTrackerNode(Node):

    def __init__(self):
        super().__init__('hand_tracker_node')

        # ── Publishers — shared ────────────────────────────────────────────
        self._image_pub   = self.create_publisher(Image,      '/hand_tracker/image',       10)
        self._caminfo_pub = self.create_publisher(CameraInfo, '/hand_tracker/camera_info', 10)

        # ── Publishers — right hand ────────────────────────────────────────
        self._pose_pub    = self.create_publisher(PoseStamped, '/hand_pose/right',     10)
        self._active_pub  = self.create_publisher(Bool,        '/hand_tracker/active', 10)
        self._gripper_pub = self.create_publisher(Float64,     '/hand_tracker/gripper',10)

        # ── Publishers — left hand ─────────────────────────────────────────
        self._pose_left_pub    = self.create_publisher(PoseStamped, '/hand_pose/left',              10)
        self._active_left_pub  = self.create_publisher(Bool,        '/hand_tracker/left/active',    10)
        self._gripper_left_pub = self.create_publisher(Float64,     '/hand_tracker/left/gripper',   10)

        self._bridge = CvBridge()

        # ── Thread-safe snapshot queue (size 1 = always latest frame) ─────
        self._pub_queue: queue.Queue = queue.Queue(maxsize=1)

        # ── Publish timer — runs on the ROS2 executor thread ──────────────
        self.create_timer(0.016, self._publish_from_queue)   # ~60 Hz drain

        # ── Camera thread ─────────────────────────────────────────────────
        self._cam_thread = threading.Thread(target=self._camera_loop, daemon=True)
        self._cam_thread.start()

        self.get_logger().info(
            f'Hand tracker started | '
            f'fist thr: {FIST_THRESHOLD} | '
            f'thumb: [{THUMB_ANGLE_MIN}, {THUMB_ANGLE_MAX}] deg | '
            f'window: {SHOW_WINDOW}')

    # ─── Publish timer callback (executor thread) ─────────────────────────────

    def _publish_from_queue(self):
        try:
            snap = self._pub_queue.get_nowait()
        except queue.Empty:
            return

        stamp = self.get_clock().now().to_msg()
        self._publish_hand(stamp, snap.rx,  snap.ry,  snap.ryaw, snap.r_active, snap.r_gripper, right=True)
        self._publish_hand(stamp, snap.lx,  snap.ly,  snap.lyaw, snap.l_active, snap.l_gripper, right=False)
        if snap.frame is not None:
            self._publish_image(stamp, snap.frame)

    def _publish_hand(self, stamp, x, y, yaw, active, gripper, right: bool):
        pose = PoseStamped()
        pose.header.stamp       = stamp
        pose.header.frame_id    = 'world'
        pose.pose.position.x    = float(1.0 - x)
        pose.pose.position.y    = float(1.0 - y)
        pose.pose.position.z    = 0.0
        pose.pose.orientation.w = math.cos(yaw / 2.0)
        pose.pose.orientation.z = math.sin(yaw / 2.0)
        a = Bool();    a.data = active
        g = Float64(); g.data = float(gripper)
        if right:
            self._pose_pub.publish(pose)
            self._active_pub.publish(a)
            self._gripper_pub.publish(g)
        else:
            self._pose_left_pub.publish(pose)
            self._active_left_pub.publish(a)
            self._gripper_left_pub.publish(g)

    def _publish_image(self, stamp, frame):
        try:
            img = self._bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            img.header.stamp    = stamp
            img.header.frame_id = 'world'
            self._image_pub.publish(img)
            info = CameraInfo()
            info.header.stamp    = stamp
            info.header.frame_id = 'world'
            info.width  = frame.shape[1]
            info.height = frame.shape[0]
            self._caminfo_pub.publish(info)
        except Exception as e:
            self.get_logger().warn(str(e), throttle_duration_sec=5.0)

    # ─── Camera loop (daemon thread — no ROS API calls) ───────────────────────

    def _enqueue(self, snap: _Snapshot):
        """Replace any stale snapshot with the latest one."""
        try:
            self._pub_queue.get_nowait()
        except queue.Empty:
            pass
        self._pub_queue.put_nowait(snap)

    def _camera_loop(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.get_logger().error('Could not open webcam!')
            return

        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        _frame_interval = 1.0 / FPS_CAP
        self.get_logger().info(
            f'Camera: {w}x{h} @ {FPS_CAP} fps cap | model_complexity={MODEL_COMPLEXITY}')

        # Per-hand tracking state (all plain Python — no ROS)
        smooth_x   = smooth_y   = 0.5; smooth_yaw   = 0.0
        smooth_x_L = smooth_y_L = 0.5; smooth_yaw_L = 0.0
        prev_x = prev_y = prev_x_L = prev_y_L = None
        last_seen = last_seen_L = None
        fist = fist_L = False
        gripper = gripper_L = 0.0

        with mp.solutions.hands.Hands(
            max_num_hands=2,
            model_complexity=MODEL_COMPLEXITY,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as hands:

            while rclpy.ok():
                _t0 = time.monotonic()

                ret, frame = cap.read()
                if not ret:
                    break

                frame   = cv2.flip(frame, 1)
                rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)

                self._draw_roi(frame, w, h)

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

                snap      = _Snapshot()
                warning   = ''
                r_detected = l_detected = False

                # ── Right hand ────────────────────────────────────────────
                if right_hand:
                    lms  = right_hand.landmark
                    x, y = lms[0].x, lms[0].y
                    in_roi  = self._in_roi(x, y)
                    jump_ok = self._jump_ok(x, y, prev_x, prev_y)
                    self._draw_landmarks(frame, right_hand, w, h)
                    if in_roi and jump_ok:
                        prev_x = x; prev_y = y
                        last_seen  = time.monotonic()
                        r_detected = True
                        fist    = self._is_fist(lms)
                        yaw_rad = self._normalize_yaw(self._hand_yaw_deg(lms))
                        gripper = self._gripper_from_angle(self._thumb_angle(lms)) if fist else 0.0
                        smooth_x   = SMOOTH_ALPHA * x + (1 - SMOOTH_ALPHA) * smooth_x
                        smooth_y   = SMOOTH_ALPHA * y + (1 - SMOOTH_ALPHA) * smooth_y
                        smooth_yaw = SMOOTH_ALPHA_YAW * yaw_rad + (1 - SMOOTH_ALPHA_YAW) * smooth_yaw
                        self._draw_smoothed(frame, w, h, smooth_x, smooth_y, smooth_yaw, C_SMOOTHED_R)
                        snap.rx = smooth_x; snap.ry = smooth_y; snap.ryaw = smooth_yaw
                        snap.r_active = fist; snap.r_gripper = gripper
                    elif not in_roi:
                        warning = 'R:OUTSIDE ROI'
                        snap.rx = 0.5; snap.ry = 0.5; snap.ryaw = 0.0
                    else:
                        warning = 'R:JUMP'
                        snap.rx = smooth_x; snap.ry = smooth_y; snap.ryaw = smooth_yaw
                        snap.r_active = fist; snap.r_gripper = gripper
                else:
                    if last_seen is None or (time.monotonic() - last_seen) > LOST_TIMEOUT:
                        snap.rx = 0.5; snap.ry = 0.5; snap.ryaw = 0.0
                    else:
                        snap.rx = smooth_x; snap.ry = smooth_y; snap.ryaw = smooth_yaw
                        snap.r_active = fist; snap.r_gripper = gripper

                # ── Left hand ─────────────────────────────────────────────
                if left_hand:
                    lms  = left_hand.landmark
                    x, y = lms[0].x, lms[0].y
                    in_roi  = self._in_roi(x, y)
                    jump_ok = self._jump_ok(x, y, prev_x_L, prev_y_L)
                    self._draw_landmarks(frame, left_hand, w, h)
                    if in_roi and jump_ok:
                        prev_x_L = x; prev_y_L = y
                        last_seen_L = time.monotonic()
                        l_detected  = True
                        fist_L    = self._is_fist(lms)
                        yaw_rad   = self._normalize_yaw(self._hand_yaw_deg(lms))
                        gripper_L = self._gripper_from_angle(self._thumb_angle(lms)) if fist_L else 0.0
                        smooth_x_L   = SMOOTH_ALPHA * x + (1 - SMOOTH_ALPHA) * smooth_x_L
                        smooth_y_L   = SMOOTH_ALPHA * y + (1 - SMOOTH_ALPHA) * smooth_y_L
                        smooth_yaw_L = SMOOTH_ALPHA_YAW * yaw_rad + (1 - SMOOTH_ALPHA_YAW) * smooth_yaw_L
                        self._draw_smoothed(frame, w, h, smooth_x_L, smooth_y_L, smooth_yaw_L, C_SMOOTHED_L)
                        snap.lx = smooth_x_L; snap.ly = smooth_y_L; snap.lyaw = smooth_yaw_L
                        snap.l_active = fist_L; snap.l_gripper = gripper_L
                    elif not in_roi:
                        snap.lx = 0.5; snap.ly = 0.5; snap.lyaw = 0.0
                    else:
                        snap.lx = smooth_x_L; snap.ly = smooth_y_L; snap.lyaw = smooth_yaw_L
                        snap.l_active = fist_L; snap.l_gripper = gripper_L
                else:
                    if last_seen_L is None or (time.monotonic() - last_seen_L) > LOST_TIMEOUT:
                        snap.lx = 0.5; snap.ly = 0.5; snap.lyaw = 0.0
                    else:
                        snap.lx = smooth_x_L; snap.ly = smooth_y_L; snap.lyaw = smooth_yaw_L
                        snap.l_active = fist_L; snap.l_gripper = gripper_L

                self._draw_info_bar(frame, w, h, fist, fist_L, gripper, gripper_L)
                self._draw_status(frame, w, h, r_detected, l_detected, warning)
                snap.frame = frame

                self._enqueue(snap)

                if SHOW_WINDOW:
                    cv2.imshow('Hand Tracker', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    cv2.waitKey(1)

                elapsed = time.monotonic() - _t0
                if elapsed < _frame_interval:
                    time.sleep(_frame_interval - elapsed)

        cap.release()
        if SHOW_WINDOW:
            cv2.destroyAllWindows()

    # ─── Gesture & orientation helpers ───────────────────────────────────────

    def _is_fist(self, lms) -> bool:
        wrist = lms[0]
        return sum(dist2d(lms[i], wrist) for i in [8, 12, 16, 20]) / 4 < FIST_THRESHOLD

    def _thumb_angle(self, lms) -> float:
        wrist = lms[0]; index_mcp = lms[5]; thumb_tip = lms[4]
        pv = (index_mcp.x - wrist.x, index_mcp.y - wrist.y)
        tv = (thumb_tip.x  - wrist.x, thumb_tip.y  - wrist.y)
        mp_ = math.sqrt(pv[0]**2 + pv[1]**2)
        mt  = math.sqrt(tv[0]**2 + tv[1]**2)
        if mp_ < 1e-6 or mt < 1e-6:
            return 0.0
        return math.degrees(math.acos(max(-1.0, min(1.0, (pv[0]*tv[0] + pv[1]*tv[1]) / (mp_ * mt)))))

    def _gripper_from_angle(self, angle: float) -> float:
        return max(0.0, min(1.0, (angle - THUMB_ANGLE_MIN) / (THUMB_ANGLE_MAX - THUMB_ANGLE_MIN)))

    def _hand_yaw_deg(self, lms) -> float:
        w = lms[0]; m = lms[9]
        return math.degrees(math.atan2(m.y - w.y, m.x - w.x))

    def _normalize_yaw(self, yaw_deg: float) -> float:
        return math.radians(yaw_deg)

    def _in_roi(self, x, y) -> bool:
        return ROI_X1 < x < ROI_X2 and ROI_Y1 < y < ROI_Y2

    def _jump_ok(self, x, y, prev_x, prev_y) -> bool:
        if prev_x is None:
            return True
        return math.sqrt((x - prev_x)**2 + (y - prev_y)**2) < MAX_JUMP

    # ─── Drawing helpers ──────────────────────────────────────────────────────

    def _draw_landmarks(self, frame, hand_landmarks, w, h):
        lms = hand_landmarks.landmark
        tips = {4, 8, 12, 16, 20}
        for c in mp.solutions.hands.HAND_CONNECTIONS:
            p1, p2 = lms[c[0]], lms[c[1]]
            cv2.line(frame, (int(p1.x*w), int(p1.y*h)), (int(p2.x*w), int(p2.y*h)), C_CONN, 1)
        for i, lm in enumerate(lms):
            x, y = int(lm.x*w), int(lm.y*h)
            color, size = (C_WRIST, 10) if i == 0 else (C_THUMB, 8) if i == 4 else \
                          (C_MIDDLE, 8) if i == 9 else (C_LM, 7) if i in tips else (C_LM, 4)
            cv2.circle(frame, (x, y), size, color, -1)
            cv2.circle(frame, (x, y), size, (0, 0, 0), 1)

    def _draw_smoothed(self, frame, w, h, sx, sy, syaw, color):
        px, py = int(sx*w), int(sy*h)
        cv2.line(frame, (px-20, py), (px+20, py), color, 2)
        cv2.line(frame, (px, py-20), (px, py+20), color, 2)
        cv2.circle(frame, (px, py), 5, color, -1)
        cv2.putText(frame, f'xy:({sx:.2f},{sy:.2f}) yaw:{syaw:.2f}',
                    (px+12, py-8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    def _draw_roi(self, frame, w, h):
        pt1 = (int(ROI_X1*w), int(ROI_Y1*h))
        pt2 = (int(ROI_X2*w), int(ROI_Y2*h))
        cv2.rectangle(frame, pt1, pt2, C_ROI, 2)
        cv2.putText(frame, 'ROI', (pt1[0]+4, pt1[1]+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, C_ROI, 2)

    def _draw_info_bar(self, frame, w, _h, fist_r, fist_l, gripper_r, gripper_l):
        cv2.rectangle(frame, (0, 0), (w, 55), (30, 30, 30), -1)
        cv2.putText(frame, 'R:MOVING' if fist_r else 'R:STOP', (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, C_ACTIVE if fist_r else C_INACTIVE, 2)
        cv2.putText(frame, 'L:MOVING' if fist_l else 'L:STOP', (10, 46),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, C_ACTIVE if fist_l else C_INACTIVE, 2)
        for bx, gripper in [(w-230, gripper_r), (w-140, gripper_l)]:
            bw, bh = 80, 12
            cv2.rectangle(frame, (bx, 5), (bx+bw, 5+bh), (80, 80, 80), -1)
            fill = int(gripper * bw)
            cv2.rectangle(frame, (bx, 5), (bx+fill, 5+bh),
                          (0, int(255*gripper), int(255*(1-gripper))), -1)
            cv2.rectangle(frame, (bx, 5), (bx+bw, 5+bh), (200, 200, 200), 1)

    def _draw_status(self, frame, w, h, r_ok, l_ok, warning=''):
        cv2.rectangle(frame, (0, h-36), (w, h), (30, 30, 30), -1)
        cv2.putText(frame, f'R:{"OK" if r_ok else "--"}  L:{"OK" if l_ok else "--"}',
                    (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.65, C_ACTIVE, 2)
        if warning:
            cv2.putText(frame, warning, (w//2-60, h-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, C_WARN, 1)


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
