import numpy as np


class FootSwingTrajectory:
    """
    Cubic Bezier swing foot trajectory, planned in world frame.

    Follows the MIT Cheetah convention: world-frame planning with
    four control points providing smooth liftoff and touchdown.

    Control points:
        P0 = (p0.x, p0.y, p0.z)           - liftoff position
        P1 = (p0.x, p0.y, p0.z + height)  - lift vertically
        P2 = (pf.x, pf.y, pf.z + height)  - stay high over target
        P3 = (pf.x, pf.y, pf.z)           - touchdown position

    XY motion is Hermite-like (zero velocity at endpoints).
    Z motion lifts to swing_height above start/end, smooth arc.
    """

    def __init__(self):
        self._p0 = np.zeros(3)
        self._pf = np.zeros(3)
        self._p = np.zeros(3)
        self._v = np.zeros(3)
        self._height = 0.0

    def set_initial_position(self, p0):
        self._p0[:] = p0

    def set_final_position(self, pf):
        self._pf[:] = pf

    def set_height(self, h):
        self._height = h

    def compute_swing_trajectory_bezier(self, phase, swing_time):
        """
        Compute desired foot position and velocity at given phase.

        Args:
            phase: Normalized swing progress in [0, 1]
            swing_time: Total swing phase duration in seconds
        """
        s = phase
        p0 = self._p0
        pf = self._pf
        h = self._height

        oms = 1.0 - s
        oms2 = oms * oms
        oms3 = oms2 * oms
        s2 = s * s
        s3 = s2 * s

        # --- Position ---
        # XY: P0=p0, P1=p0, P2=pf, P3=pf (Hermite blend)
        c_start = oms3 + 3.0 * oms2 * s
        c_end = 3.0 * oms * s2 + s3
        self._p[0] = c_start * p0[0] + c_end * pf[0]
        self._p[1] = c_start * p0[1] + c_end * pf[1]

        # Z: P0=p0z, P1=p0z+h, P2=pfz+h, P3=pfz
        self._p[2] = (oms3 * p0[2] +
                      3.0 * oms2 * s * (p0[2] + h) +
                      3.0 * oms * s2 * (pf[2] + h) +
                      s3 * pf[2])

        # --- Velocity: dB/ds / swing_time ---
        # B'(s) = 3(1-s)^2*(P1-P0) + 6(1-s)*s*(P2-P1) + 3*s^2*(P3-P2)
        inv_t = 1.0 / swing_time if swing_time > 1e-6 else 0.0

        # XY: P1-P0=0, P2-P1=pf-p0, P3-P2=0
        c_mid = 6.0 * oms * s * inv_t
        self._v[0] = c_mid * (pf[0] - p0[0])
        self._v[1] = c_mid * (pf[1] - p0[1])

        # Z: P1-P0=h, P2-P1=(pfz-p0z), P3-P2=-h
        self._v[2] = (3.0 * oms2 * h +
                      6.0 * oms * s * (pf[2] - p0[2]) +
                      3.0 * s2 * (-h)) * inv_t

    def get_position(self):
        return self._p

    def get_velocity(self):
        return self._v
