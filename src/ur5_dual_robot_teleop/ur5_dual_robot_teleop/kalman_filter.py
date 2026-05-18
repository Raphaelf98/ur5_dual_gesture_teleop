#!/usr/bin/env python3
"""
1D Kalman filter — constant-velocity model, position-only measurement.

State:       [position, velocity]
Measurement: position only  (H = [1, 0])
Process:     white-noise acceleration (sigma_a)

Tuning guide
------------
sigma_a  — how fast the hand can accelerate (units/s²).
           Larger → filter reacts faster to direction changes but noisier.
           Typical for normalised image coords: 3–10 img_frac/s²
           Typical for yaw: 5–15 rad/s²

sigma_r  — std dev of the position measurement noise (same units).
           Larger → filter trusts its own prediction more, smoother.
           Smaller → trusts measurements more, less smoothing.
           Position (after hand-tracker EMA): ~0.01–0.03 img_frac
           Yaw:                               ~0.05–0.15 rad
"""

import math


class KF1D:

    def __init__(self, sigma_a: float = 5.0, sigma_r: float = 0.02):
        self._qa    = sigma_a ** 2   # acceleration noise variance
        self._r     = sigma_r ** 2   # measurement noise variance
        self._x     = [0.0, 0.0]    # [position, velocity]
        self._P     = [[1.0, 0.0], [0.0, 1.0]]
        self._ready = False

    def reset(self, pos: float = 0.0) -> None:
        self._x     = [pos, 0.0]
        self._P     = [[1.0, 0.0], [0.0, 1.0]]
        self._ready = True

    @property
    def position(self) -> float:
        return self._x[0]

    @property
    def velocity(self) -> float:
        return self._x[1]

    def predict(self, dt: float) -> float:
        """Advance state by dt seconds without a measurement. Returns predicted position."""
        if not self._ready or dt <= 0.0:
            return self._x[0]

        pos, vel = self._x
        dt2 = dt * dt

        # x_pred = F @ x,  F = [[1, dt], [0, 1]]
        self._x = [pos + vel * dt, vel]

        # P_pred = F @ P @ F.T + Q
        # Q uses the discrete white-noise acceleration model:
        #   Q = sigma_a² * [[dt⁴/4, dt³/2], [dt³/2, dt²]]
        P   = self._P
        q   = self._qa
        P00 = P[0][0] + dt * (P[0][1] + P[1][0]) + dt2 * P[1][1] + q * dt2 * dt2 / 4.0
        P01 = P[0][1] + dt * P[1][1]              + q * dt2 * dt  / 2.0
        P10 = P[1][0] + dt * P[1][1]              + q * dt2 * dt  / 2.0
        P11 = P[1][1]                              + q * dt2
        self._P = [[P00, P01], [P10, P11]]

        return self._x[0]

    def update(self, z: float, wrap: bool = False) -> float:
        """Fuse measurement z into the state. Returns updated position estimate.

        wrap=True handles angular quantities by wrapping the innovation to [-π, π].
        """
        if not self._ready:
            self._x     = [z, 0.0]
            self._P     = [[1.0, 0.0], [0.0, 1.0]]
            self._ready = True
            return z

        pos, vel = self._x
        innov = z - pos
        if wrap:
            innov = (innov + math.pi) % (2.0 * math.pi) - math.pi

        # S = H P H^T + R = P[0][0] + R  (H = [1, 0])
        P  = self._P
        S  = P[0][0] + self._r
        K0 = P[0][0] / S   # Kalman gain for position
        K1 = P[1][0] / S   # Kalman gain for velocity

        self._x = [pos + K0 * innov, vel + K1 * innov]

        # P = (I - K H) P
        c = 1.0 - K0
        self._P = [
            [c * P[0][0],           c * P[0][1]          ],
            [P[1][0] - K1 * P[0][0], P[1][1] - K1 * P[0][1]],
        ]

        return self._x[0]
