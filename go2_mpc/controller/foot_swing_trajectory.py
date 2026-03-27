"""
Cubic Bezier swing foot trajectory generator.

All positions and velocities are in **world frame (W)** — the same
frame used by the Cartesian PD controller and the world-frame
Jacobian for torque mapping.

Bezier curve
------------
A cubic Bezier curve ``B(s)`` for ``s ∈ [0, 1]`` is defined by four
control points ``P0, P1, P2, P3``::

    B(s) = (1-s)³ P0 + 3(1-s)² s P1 + 3(1-s) s² P2 + s³ P3

**XY motion** uses a Hermite-like blend (zero endpoint velocities)::

    P0 = p0,  P1 = p0,  P2 = pf,  P3 = pf

This gives ``B'(0) = 3(P1 - P0) = 0`` and ``B'(1) = 3(P3 - P2) = 0``,
so the foot starts and ends with zero horizontal velocity — smooth
liftoff and touchdown.

**Z motion** lifts the foot to ``swing_height`` above start/end::

    P0 = p0_z,  P1 = p0_z + h,  P2 = pf_z + h,  P3 = pf_z

This creates a smooth bell-shaped arc peaking near mid-swing.

Velocity is obtained analytically as ``dB/ds · (1 / T_swing)`` where
``T_swing`` is the swing phase duration in seconds.
"""

import numpy as np


class FootSwingTrajectory:
    """Single-leg cubic Bezier swing trajectory.

    Usage (called at 1 kHz by the swing controller)::

        traj.set_initial_position(p0_world)   # liftoff position (W)
        traj.set_final_position(pf_world)     # touchdown target (W)
        traj.set_height(0.10)                 # apex height above ground (m)
        traj.compute_swing_trajectory_bezier(phase, swing_time)

        p_des = traj.get_position()  # (3,) desired foot position (W)
        v_des = traj.get_velocity()  # (3,) desired foot velocity (W)

    All inputs and outputs are in **world frame**.
    """

    def __init__(self):
        self._p0 = np.zeros(3)   # liftoff position, world frame (m)
        self._pf = np.zeros(3)   # touchdown target, world frame (m)
        self._p = np.zeros(3)    # current desired position, world frame (m)
        self._v = np.zeros(3)    # current desired velocity, world frame (m/s)
        self._height = 0.0       # apex swing height above ground (m)

    def set_initial_position(self, p0):
        """Set liftoff position (world frame, m)."""
        self._p0[:] = p0

    def set_final_position(self, pf):
        """Set touchdown target position (world frame, m)."""
        self._pf[:] = pf

    def set_height(self, h):
        """Set swing apex height above the start/end z-values (m)."""
        self._height = h

    def compute_swing_trajectory_bezier(self, phase, swing_time):
        """Evaluate the Bezier curve at a given swing phase.

        Parameters
        ----------
        phase : float
            Normalised swing progress in ``[0, 1]``.
            ``0`` = liftoff, ``1`` = touchdown.
        swing_time : float
            Total swing phase duration (s).  Used to convert the
            parametric derivative ``dB/ds`` to a physical velocity
            ``dB/dt = dB/ds · (1 / T_swing)``.
        """
        s = phase
        p0 = self._p0
        pf = self._pf
        h = self._height

        # Bernstein polynomial basis terms
        oms = 1.0 - s       # (1-s)
        oms2 = oms * oms     # (1-s)²
        oms3 = oms2 * oms    # (1-s)³
        s2 = s * s           # s²
        s3 = s2 * s          # s³

        # --- Position B(s) ---
        # XY: P0=p0, P1=p0, P2=pf, P3=pf  (Hermite blend, zero endpoint vel)
        #   B_xy(s) = [(1-s)³ + 3(1-s)²s] · p0  +  [3(1-s)s² + s³] · pf
        c_start = oms3 + 3.0 * oms2 * s    # weight on p0
        c_end = 3.0 * oms * s2 + s3         # weight on pf
        self._p[0] = c_start * p0[0] + c_end * pf[0]
        self._p[1] = c_start * p0[1] + c_end * pf[1]

        # Z: P0=p0z, P1=p0z+h, P2=pfz+h, P3=pfz  (bell-shaped arc)
        self._p[2] = (oms3 * p0[2] +
                      3.0 * oms2 * s * (p0[2] + h) +
                      3.0 * oms * s2 * (pf[2] + h) +
                      s3 * pf[2])

        # --- Velocity dB/ds / swing_time ---
        # B'(s) = 3(1-s)²(P1-P0) + 6(1-s)s(P2-P1) + 3s²(P3-P2)
        inv_t = 1.0 / swing_time if swing_time > 1e-6 else 0.0

        # XY: P1-P0 = 0,  P2-P1 = pf-p0,  P3-P2 = 0
        #   → B'_xy(s) = 6(1-s)s · (pf - p0)
        c_mid = 6.0 * oms * s * inv_t
        self._v[0] = c_mid * (pf[0] - p0[0])
        self._v[1] = c_mid * (pf[1] - p0[1])

        # Z: P1-P0 = h,  P2-P1 = (pfz-p0z),  P3-P2 = -h
        self._v[2] = (3.0 * oms2 * h +
                      6.0 * oms * s * (pf[2] - p0[2]) +
                      3.0 * s2 * (-h)) * inv_t

    def get_position(self):
        """Current desired foot position (3,) in world frame (m)."""
        return self._p

    def get_velocity(self):
        """Current desired foot velocity (3,) in world frame (m/s)."""
        return self._v
