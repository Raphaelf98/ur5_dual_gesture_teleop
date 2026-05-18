#!/usr/bin/env python3
import math
from dataclasses import dataclass

import rclpy
from geometry_msgs.msg import Point, TwistStamped
from visualization_msgs.msg import Marker, MarkerArray


@dataclass
class Pose2D:
    """EEF pose in the XY workspace plane."""
    x:   float = 0.0
    y:   float = 0.0
    yaw: float = 0.0


@dataclass
class Twist2D:
    """Velocity command for XY plane + wrist rotation."""
    vx:  float = 0.0
    vy:  float = 0.0
    wz:  float = 0.0


def angle_diff(target: float, current: float) -> float:
    """Shortest signed angular distance from current to target, in [-π, π]."""
    diff = target - current
    while diff >  math.pi: diff -= 2 * math.pi
    while diff < -math.pi: diff += 2 * math.pi
    return diff


def eef_pose(tf_buffer, world_frame: str, frame: str) -> Pose2D:
    """Look up EEF pose from TF. Returns a zero Pose2D on failure."""
    try:
        tf = tf_buffer.lookup_transform(world_frame, frame, rclpy.time.Time())
        q = tf.transform.rotation
        yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )
        return Pose2D(
            x=tf.transform.translation.x,
            y=tf.transform.translation.y,
            yaw=yaw,
        )
    except Exception:
        return Pose2D()


def make_twist(
    vel: Twist2D,
    world_frame: str,
    stamp,
    invert: bool,
    swap_xy: bool,
    invert_angular: bool = False,
) -> TwistStamped:
    """Pack a Twist2D into a stamped ROS message, applying axis swap and sign inversion.

    invert         — negate linear X/Y (for mirrored workspace)
    invert_angular — negate angular Z independently of invert
    """
    linear_sign  = -1.0 if invert         else 1.0
    angular_sign = -1.0 if invert_angular else 1.0
    vx, vy = (-vel.vy, vel.vx) if swap_xy else (vel.vx, vel.vy)
    msg = TwistStamped()
    msg.header.stamp    = stamp
    msg.header.frame_id = world_frame
    msg.twist.linear.x  = float(linear_sign * vx)
    msg.twist.linear.y  = float(linear_sign * vy)
    msg.twist.linear.z  = 0.0
    msg.twist.angular.z = float(angular_sign * vel.wz)
    return msg


def build_target_markers(
    left: Pose2D,
    right: Pose2D,
    world_frame: str,
    stamp,
    z_plane: float,
    arrow_length: float,
    sphere_scale: float,
    arrow_shaft_d: float,
    arrow_head_d: float,
) -> MarkerArray:
    """Build sphere + yaw-arrow RViz markers for left (green) and right (blue) targets."""
    array = MarkerArray()
    for sphere_id, arrow_id, pose, r, g, b in [
        (0, 2, left,  0.2, 1.0, 0.2),
        (1, 3, right, 0.2, 0.4, 1.0),
    ]:
        m = Marker()
        m.header.frame_id    = world_frame
        m.header.stamp       = stamp
        m.ns                 = 'teleop_targets'
        m.id                 = sphere_id
        m.type               = Marker.SPHERE
        m.action             = Marker.ADD
        m.pose.position.x    = pose.x
        m.pose.position.y    = pose.y
        m.pose.position.z    = z_plane
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = sphere_scale
        m.color.r = r; m.color.g = g; m.color.b = b; m.color.a = 0.85
        array.markers.append(m)

        a = Marker()
        a.header.frame_id = world_frame
        a.header.stamp    = stamp
        a.ns              = 'teleop_targets'
        a.id              = arrow_id
        a.type            = Marker.ARROW
        a.action          = Marker.ADD
        a.scale.x         = arrow_shaft_d
        a.scale.y         = arrow_head_d
        a.scale.z         = 0.0
        a.color.r = r; a.color.g = g; a.color.b = b; a.color.a = 1.0
        a.points = [
            Point(x=pose.x, y=pose.y, z=z_plane),
            Point(
                x=pose.x + arrow_length * math.cos(pose.yaw),
                y=pose.y + arrow_length * math.sin(pose.yaw),
                z=z_plane,
            ),
        ]
        array.markers.append(a)
    return array
