#!/usr/bin/env python3
"""
kf_tuner.py  —  Kalman filter parameter measurement tool.

Subscribes to /hand_pose/right while you perform guided gestures and
computes sigma_r (measurement noise) and sigma_a (process noise) for the
KF1D filter used in hand_tracking_input.py.

Prerequisites: hand_tracker_node must be running and publishing
  /hand_pose/right before each collection phase.

Usage:
  ros2 run ur5_dual_robot_teleop kf_tuner
  python3 kf_tuner.py
"""

import math
import threading
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped

STILL_DURATION  = 5.0   # seconds
MOTION_DURATION = 10.0  # seconds
MAX_DT_GAP      = 0.3   # s — skip velocity across detection gaps longer than this


class _Collector(Node):

    def __init__(self):
        super().__init__('kf_tuner')
        self._lock      = threading.Lock()
        self._recording = False
        self._samples   = []  # list of (monotonic_t, x, y, yaw)
        self.create_subscription(PoseStamped, '/hand_pose/right', self._cb, 10)

    def start(self):
        with self._lock:
            self._samples   = []
            self._recording = True

    def stop(self) -> list:
        with self._lock:
            self._recording = False
            return list(self._samples)

    def _cb(self, msg: PoseStamped):
        q   = msg.pose.orientation
        yaw = math.atan2(2.0 * q.w * q.z, 1.0 - 2.0 * q.z * q.z)
        with self._lock:
            if self._recording:
                self._samples.append((
                    time.monotonic(),
                    msg.pose.position.x,
                    msg.pose.position.y,
                    yaw,
                ))


def _collect(node: _Collector, duration: float) -> list:
    node.start()
    t0        = time.monotonic()
    bar_width = 38
    while True:
        elapsed = time.monotonic() - t0
        frac    = min(elapsed / duration, 1.0)
        filled  = int(bar_width * frac)
        bar     = '█' * filled + '░' * (bar_width - filled)
        with node._lock:
            n = len(node._samples)
        print(f'\r  [{bar}] {elapsed:4.1f}/{duration:.0f}s  {n:4d} samples',
              end='', flush=True)
        if elapsed >= duration:
            break
        time.sleep(0.1)
    print()
    return node.stop()


# ── Statistics helpers ────────────────────────────────────────────────────────

def _mean(v): return sum(v) / len(v) if v else 0.0


def _std(v):
    if len(v) < 2:
        return 0.0
    m = _mean(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))


def _pct(v, p):
    if not v:
        return 0.0
    s   = sorted(v)
    idx = p / 100.0 * (len(s) - 1)
    lo  = int(idx)
    hi  = min(lo + 1, len(s) - 1)
    return s[lo] + (idx - lo) * (s[hi] - s[lo])


# ── Phase analysis ────────────────────────────────────────────────────────────

def _analyse_still(samples):
    """sigma_r from position variance while hand is held stationary."""
    ts   = [s[0] for s in samples]
    xs   = [s[1] for s in samples]
    ys   = [s[2] for s in samples]
    yaws = [s[3] for s in samples]

    sx  = _std(xs)
    sy  = _std(ys)

    # Yaw std from wrapped frame-to-frame diffs to avoid 2π discontinuities.
    # std(diff of two independent N(0,σ²) variables) = σ*√2, so divide back.
    yaw_diffs = []
    for i in range(1, len(yaws)):
        d = yaws[i] - yaws[i - 1]
        yaw_diffs.append((d + math.pi) % (2 * math.pi) - math.pi)
    syaw = _std(yaw_diffs) / math.sqrt(2) if yaw_diffs else 0.0

    hz          = (len(ts) - 1) / (ts[-1] - ts[0]) if ts[-1] > ts[0] else 0.0
    sigma_r_pos = (sx + sy) / 2.0
    sigma_r_yaw = syaw

    print(f'\n  Still-phase  ({len(samples)} samples @ {hz:.1f} Hz):')
    print(f'    std_x   = {sx:.5f} img_frac')
    print(f'    std_y   = {sy:.5f} img_frac')
    print(f'    std_yaw = {syaw:.5f} rad')
    print(f'\n    → sigma_r_pos : {sigma_r_pos:.4f}')
    print(f'    → sigma_r_yaw : {sigma_r_yaw:.4f}')

    return sigma_r_pos, sigma_r_yaw


def _analyse_motion(samples):
    """sigma_a from acceleration distribution during natural hand movement."""
    if len(samples) < 5:
        print('  Not enough samples for motion analysis.')
        return 20.0, 15.0

    ts   = [s[0] for s in samples]
    xs   = [s[1] for s in samples]
    ys   = [s[2] for s in samples]
    yaws = [s[3] for s in samples]

    # ── Velocities (forward difference, skip detection gaps) ──────────────
    vt, vx_l, vy_l, vyw_l = [], [], [], []
    for i in range(len(ts) - 1):
        dt = ts[i + 1] - ts[i]
        if dt <= 1e-6 or dt > MAX_DT_GAP:
            continue
        dy = yaws[i + 1] - yaws[i]
        dy = (dy + math.pi) % (2 * math.pi) - math.pi
        vt.append((ts[i] + ts[i + 1]) / 2.0)
        vx_l.append( (xs[i + 1]   - xs[i])   / dt)
        vy_l.append( (ys[i + 1]   - ys[i])   / dt)
        vyw_l.append(dy / dt)

    # ── Accelerations (forward difference on velocities) ──────────────────
    ax_l, ay_l, ayw_l = [], [], []
    for i in range(len(vt) - 1):
        dt_v = vt[i + 1] - vt[i]
        if dt_v <= 1e-6 or dt_v > MAX_DT_GAP:
            continue
        ax_l.append( abs((vx_l[i + 1]  - vx_l[i])  / dt_v))
        ay_l.append( abs((vy_l[i + 1]  - vy_l[i])  / dt_v))
        ayw_l.append(abs((vyw_l[i + 1] - vyw_l[i]) / dt_v))

    if not ax_l:
        print('  Could not compute accelerations — check /hand_pose/right is active.')
        return 20.0, 15.0

    hz      = (len(ts) - 1) / (ts[-1] - ts[0]) if ts[-1] > ts[0] else 0.0
    p95_ax  = _pct(ax_l,  95)
    p95_ay  = _pct(ay_l,  95)
    p95_ayw = _pct(ayw_l, 95)

    sigma_a     = max(p95_ax, p95_ay)
    sigma_a_yaw = p95_ayw

    print(f'\n  Motion-phase  ({len(samples)} samples @ {hz:.1f} Hz):')
    print(f'    |accel_x|   mean={_mean(ax_l):.1f}  95th={p95_ax:.1f} img_frac/s²')
    print(f'    |accel_y|   mean={_mean(ay_l):.1f}  95th={p95_ay:.1f} img_frac/s²')
    print(f'    |accel_yaw| mean={_mean(ayw_l):.1f}  95th={p95_ayw:.1f} rad/s²')
    print(f'\n    → sigma_a     : {sigma_a:.1f}  (95th percentile of |accel| in x/y)')
    print(f'    → sigma_a_yaw : {sigma_a_yaw:.1f}  (95th percentile of |accel_yaw|)')

    return sigma_a, sigma_a_yaw


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    rclpy.init()
    node = _Collector()

    spin = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin.start()

    print()
    print('=' * 62)
    print('  Kalman Filter Tuner  —  /hand_pose/right')
    print('=' * 62)
    print()
    print('  Prerequisites: hand_tracker_node must be running.')
    print('  Right hand only — results apply to both arms.')

    # ── Phase 1: still noise → sigma_r ───────────────────────────────────
    print()
    print('  PHASE 1 — STILL  (→ sigma_r_pos, sigma_r_yaw)')
    print('  Hold right hand COMPLETELY STILL in front of the camera.')
    input('  Press Enter to start 5-second recording...')

    still = _collect(node, STILL_DURATION)
    if len(still) < 10:
        print(f'  WARNING: only {len(still)} samples received.')
        print('  Is /hand_pose/right publishing? (Check that hand_tracker_node')
        print('  is running and your hand is visible.)')
        sigma_r_pos, sigma_r_yaw = 0.01, 0.05
    else:
        sigma_r_pos, sigma_r_yaw = _analyse_still(still)

    # ── Phase 2: motion noise → sigma_a ──────────────────────────────────
    print()
    print('  PHASE 2 — MOTION  (→ sigma_a, sigma_a_yaw)')
    print('  Move right hand as during normal teleop: translate XY and')
    print('  rotate wrist. Cover the full motion range you typically use.')
    input('  Press Enter to start 10-second recording...')

    motion = _collect(node, MOTION_DURATION)
    if len(motion) < 10:
        print(f'  WARNING: only {len(motion)} samples received.')
        sigma_a, sigma_a_yaw = 20.0, 15.0
    else:
        sigma_a, sigma_a_yaw = _analyse_motion(motion)

    # ── Recommendations ───────────────────────────────────────────────────
    print()
    print('=' * 62)
    print('  Recommended kalman: section for teleop_params.yaml')
    print('=' * 62)
    print()
    print('  kalman:')
    print(f'    sigma_a:     {sigma_a:.1f}')
    print(f'    sigma_a_yaw: {sigma_a_yaw:.1f}')
    print(f'    sigma_r_pos: {sigma_r_pos:.4f}')
    print(f'    sigma_r_yaw: {sigma_r_yaw:.4f}')
    print()
    print('  Tuning guide:')
    print('  sigma_a  too HIGH → predicts too far between frames → jumps')
    print('  sigma_a  too LOW  → slow to follow sudden direction changes')
    print('  sigma_r  too HIGH → smooth but laggy (ignores measurements)')
    print('  sigma_r  too LOW  → twitchy (trusts every noisy reading)')
    print()

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
