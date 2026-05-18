#!/usr/bin/env python3
from ur5_dual_robot_teleop.utils import Pose2D, Twist2D, angle_diff


class PDController:
    """
    PD controller for smooth EEF pose following.

    velocity = Kp * error + Kd * d(error)/dt
    """

    def __init__(
        self,
        kp_linear:   float = 2.0,
        kd_linear:   float = 0.3,
        kp_angular:  float = 1.5,
        kd_angular:  float = 0.2,
        max_linear:  float = 0.4,
        max_angular: float = 0.8,
    ):
        self.kp_linear   = kp_linear
        self.kd_linear   = kd_linear
        self.kp_angular  = kp_angular
        self.kd_angular  = kd_angular
        self.max_linear  = max_linear
        self.max_angular = max_angular
        self._prev_ex    = 0.0
        self._prev_ey    = 0.0
        self._prev_eyaw  = 0.0

    def reset(self):
        self._prev_ex   = 0.0
        self._prev_ey   = 0.0
        self._prev_eyaw = 0.0

    def compute_velocity(self, current: Pose2D, target: Pose2D, dt: float) -> Twist2D:
        if dt <= 0.0:
            return Twist2D()

        ex   = target.x   - current.x
        ey   = target.y   - current.y
        eyaw = angle_diff(target.yaw, current.yaw)

        dex   = (ex   - self._prev_ex)   / dt
        dey   = (ey   - self._prev_ey)   / dt
        deyaw = (eyaw - self._prev_eyaw) / dt

        vx  = self.kp_linear  * ex   + self.kd_linear  * dex
        vy  = self.kp_linear  * ey   + self.kd_linear  * dey
        wz  = self.kp_angular * eyaw + self.kd_angular * deyaw

        self._prev_ex   = ex
        self._prev_ey   = ey
        self._prev_eyaw = eyaw

        return Twist2D(
            vx=float(max(-self.max_linear,  min(self.max_linear,  vx))),
            vy=float(max(-self.max_linear,  min(self.max_linear,  vy))),
            wz=float(max(-self.max_angular, min(self.max_angular, wz))),
        )
