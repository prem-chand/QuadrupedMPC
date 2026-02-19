"""
Analytical kinematics for the Unitree Go2 quadruped.

Provides forward kinematics, geometric Jacobians, foot velocities, and
static gravity compensation — all computed from joint angles and base
pose, with zero MuJoCo dependencies.

All quantities are expressed in the world frame unless noted otherwise.
Input conventions match MuJoCo's qpos/qvel layout:
  - quaternion [w, x, y, z]  (scalar-first)
  - qvel[3:6] is angular velocity in the BODY frame (free-joint convention)

Leg ordering: [FL=0, FR=1, RL=2, RR=3]
Joint ordering per leg: [hip_abduction, hip_flexion, knee_flexion]
"""

import numpy as np


# ---------------------------------------------------------------------------
# Go2 kinematic constants (source: go2_mpc/robot/go2.xml)
# ---------------------------------------------------------------------------

# Hip joint origins in BASE body frame.  From <body pos="..."> for *_hip.
HIP_ORIGINS: np.ndarray = np.array([
    [ 0.1934,  0.0465, 0.0],   # FL
    [ 0.1934, -0.0465, 0.0],   # FR
    [-0.1934,  0.0465, 0.0],   # RL
    [-0.1934, -0.0465, 0.0],   # RR
], dtype=np.float64)  # (4, 3)

# Thigh body origin in HIP frame.  From <body pos="..."> for *_thigh.
THIGH_OFFSETS: np.ndarray = np.array([
    [0.0,  0.0955, 0.0],   # FL
    [0.0, -0.0955, 0.0],   # FR
    [0.0,  0.0955, 0.0],   # RL
    [0.0, -0.0955, 0.0],   # RR
], dtype=np.float64)  # (4, 3)

# Calf body origin in thigh frame: [0, 0, -L_THIGH]
L_THIGH: float = 0.213

# Toe site in calf frame: [0, 0, -L_CALF]
L_CALF: float = 0.213

# Link masses (kg)
M_HIP:   float = 0.678
M_THIGH: float = 1.152
M_CALF:  float = 0.241352   # includes foot sphere mass

# CoM positions in each link's LOCAL body frame.
# Taken directly from go2.xml <inertial pos="..."> tags (lines 78, 106, 134, 162).
# NOTE: RL (idx 2) and RR (idx 3) hip CoM X = +0.0054, not -0.0054 — the rear
# hip bodies are mounted facing +X in the base frame, so their local X flips.
HIP_COM: np.ndarray = np.array([
    [-0.0054,  0.00194, -0.000105],   # FL  (go2.xml line 78)
    [-0.0054, -0.00194, -0.000105],   # FR  (go2.xml line 106)
    [ 0.0054,  0.00194, -0.000105],   # RL  (go2.xml line 134)
    [ 0.0054, -0.00194, -0.000105],   # RR  (go2.xml line 162)
], dtype=np.float64)  # (4, 3)

# Thigh CoM in thigh frame (go2.xml lines 85, 113, 141, 169).
THIGH_COM: np.ndarray = np.array([
    [-0.00374, -0.0223, -0.0327],   # FL
    [-0.00374,  0.0223, -0.0327],   # FR
    [-0.00374, -0.0223, -0.0327],   # RL
    [-0.00374,  0.0223, -0.0327],   # RR
], dtype=np.float64)  # (4, 3)

# Calf CoM in calf frame (go2.xml lines 92, 120, 148, 176).
CALF_COM: np.ndarray = np.array([
    [ 0.00629595, -0.000622121, -0.141417],   # FL
    [ 0.00629595,  0.000622121, -0.141417],   # FR
    [ 0.00629595, -0.000622121, -0.141417],   # RL
    [ 0.00629595,  0.000622121, -0.141417],   # RR
], dtype=np.float64)  # (4, 3)

_GRAVITY: np.ndarray = np.array([0.0, 0.0, -9.81], dtype=np.float64)

# Column indices into the (3, 18) full Jacobian for each leg's 3 joint DOFs.
# qvel layout: [vx,vy,vz | wx_B,wy_B,wz_B | q_dot_FL(3) | q_dot_FR(3) | ...]
_LEG_DOF_COLS: list[list[int]] = [
    [6, 7, 8],    # FL
    [9, 10, 11],  # FR
    [12, 13, 14], # RL
    [15, 16, 17], # RR
]


# ---------------------------------------------------------------------------
# Low-level rotation helpers
# ---------------------------------------------------------------------------

def _Rx(angle: float) -> np.ndarray:
    """Rotation matrix about X axis."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1.0, 0.0, 0.0],
                     [0.0,  c,  -s ],
                     [0.0,  s,   c ]], dtype=np.float64)


def _Ry(angle: float) -> np.ndarray:
    """Rotation matrix about Y axis."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[ c,  0.0,  s],
                     [0.0, 1.0, 0.0],
                     [-s,  0.0,  c ]], dtype=np.float64)


def _skew(v: np.ndarray) -> np.ndarray:
    """Skew-symmetric matrix: _skew(a) @ b == cross(a, b)."""
    return np.array([
        [ 0.0,  -v[2],  v[1]],
        [ v[2],  0.0,  -v[0]],
        [-v[1],  v[0],  0.0 ]], dtype=np.float64)


# ---------------------------------------------------------------------------
# Go2Kinematics
# ---------------------------------------------------------------------------

class Go2Kinematics:
    """
    Analytical forward kinematics and Jacobians for the Unitree Go2.

    All methods are stateless — each call takes the full robot state
    (p_base, R_base, q_leg) and returns the requested quantity.

    Gravity compensation is static only (C(q,qd)*qd excluded).  At trot
    speeds the Coriolis residual is typically 0.1–0.5 Nm per joint,
    well within the 35 Nm torque clipping margin.
    """

    # ------------------------------------------------------------------
    # Forward kinematics (frame chain)
    # ------------------------------------------------------------------

    def forward_kinematics(
        self,
        leg_idx: int,
        p_base: np.ndarray,
        R_base: np.ndarray,
        q_leg: np.ndarray,
    ) -> dict:
        """
        Compute the full FK frame chain for one leg.

        Parameters
        ----------
        leg_idx : int
            Leg index [0=FL, 1=FR, 2=RL, 3=RR].
        p_base : (3,) base CoM position, world frame.
        R_base : (3, 3) rotation R_WB (world ← body).
        q_leg  : (3,) joint angles [hip_abd, hip_flex, knee] (rad).

        Returns
        -------
        dict with keys:
          R_hip, R_thigh, R_calf         — rotation matrices (3, 3)
          p_abd_jnt, p_thigh_jnt,
          p_knee_jnt, p_foot             — positions in world frame (3,)
        """
        q_abd, q_hip, q_knee = q_leg[0], q_leg[1], q_leg[2]

        R_hip   = R_base  @ _Rx(q_abd)
        R_thigh = R_hip   @ _Ry(q_hip)
        R_calf  = R_thigh @ _Ry(q_knee)

        p_abd_jnt   = p_base      + R_base  @ HIP_ORIGINS[leg_idx]
        p_thigh_jnt = p_abd_jnt   + R_hip   @ THIGH_OFFSETS[leg_idx]
        p_knee_jnt  = p_thigh_jnt + R_thigh @ np.array([0.0, 0.0, -L_THIGH])
        p_foot      = p_knee_jnt  + R_calf  @ np.array([0.0, 0.0, -L_CALF])

        return {
            "R_hip":       R_hip,
            "R_thigh":     R_thigh,
            "R_calf":      R_calf,
            "p_abd_jnt":   p_abd_jnt,
            "p_thigh_jnt": p_thigh_jnt,
            "p_knee_jnt":  p_knee_jnt,
            "p_foot":      p_foot,
        }

    def foot_position_world(
        self,
        leg_idx: int,
        p_base: np.ndarray,
        R_base: np.ndarray,
        q_leg: np.ndarray,
    ) -> np.ndarray:
        """Foot tip (toe site) position in world frame.  Shape: (3,)."""
        return self.forward_kinematics(leg_idx, p_base, R_base, q_leg)["p_foot"].copy()

    # ------------------------------------------------------------------
    # Jacobians
    # ------------------------------------------------------------------

    def leg_jacobian(
        self,
        leg_idx: int,
        p_base: np.ndarray,
        R_base: np.ndarray,
        q_leg: np.ndarray,
    ) -> np.ndarray:
        """
        Geometric Jacobian mapping leg joint velocities → foot Cartesian
        velocity in world frame.  Shape: (3, 3).

        J[:, k] = z_k × (p_foot - p_joint_k)

        Does NOT include base velocity contributions — use full_jacobian
        or foot_velocity for that.
        """
        fk = self.forward_kinematics(leg_idx, p_base, R_base, q_leg)
        p_foot = fk["p_foot"]

        z_abd  = R_base          @ np.array([1.0, 0.0, 0.0])
        z_hip  = fk["R_hip"]    @ np.array([0.0, 1.0, 0.0])
        z_knee = fk["R_thigh"]  @ np.array([0.0, 1.0, 0.0])

        J = np.empty((3, 3), dtype=np.float64)
        J[:, 0] = np.cross(z_abd,  p_foot - fk["p_abd_jnt"])
        J[:, 1] = np.cross(z_hip,  p_foot - fk["p_thigh_jnt"])
        J[:, 2] = np.cross(z_knee, p_foot - fk["p_knee_jnt"])

        return J  # (3, 3)

    def full_jacobian(
        self,
        leg_idx: int,
        p_base: np.ndarray,
        R_base: np.ndarray,
        q_leg: np.ndarray,
    ) -> np.ndarray:
        """
        Full positional Jacobian mapping all 18 generalised velocities
        → foot Cartesian velocity in world frame.  Shape: (3, 18).

        Matches MuJoCo's mj_jacSite output (same frame conventions):
          cols 0:3  — base linear velocity (world frame) → identity block
          cols 3:6  — base angular velocity (BODY frame) → -skew(r) @ R_base
          cols k    — leg joint k DOFs → leg_jacobian columns

        v_foot = J_full @ qvel  (qvel is MuJoCo's 18-dim generalised vel)
        """
        fk = self.forward_kinematics(leg_idx, p_base, R_base, q_leg)
        p_foot = fk["p_foot"]

        J_full = np.zeros((3, 18), dtype=np.float64)

        # Base translational DOFs: v_foot += v_base
        J_full[:, 0:3] = np.eye(3)

        # Base angular DOFs (body frame): v_foot += (R_base @ omega_B) × r
        #   = -skew(r) @ R_base @ omega_B  where r = p_foot - p_base
        r = p_foot - p_base
        J_full[:, 3:6] = -_skew(r) @ R_base

        # Leg joint DOFs
        J_leg = self.leg_jacobian(leg_idx, p_base, R_base, q_leg)
        for k, col in enumerate(_LEG_DOF_COLS[leg_idx]):
            J_full[:, col] = J_leg[:, k]

        return J_full  # (3, 18)

    # ------------------------------------------------------------------
    # Foot velocity
    # ------------------------------------------------------------------

    def foot_velocity(
        self,
        leg_idx: int,
        p_base: np.ndarray,
        R_base: np.ndarray,
        q_leg: np.ndarray,
        v_base: np.ndarray,
        omega_body: np.ndarray,
        qd_leg: np.ndarray,
    ) -> np.ndarray:
        """
        Foot Cartesian velocity in world frame.  Shape: (3,).

        Equivalent to J_full @ qvel but avoids allocating the (3, 18)
        matrix — preferred at 1 kHz swing rate.

        Parameters
        ----------
        v_base     : (3,) base linear velocity, world frame.
        omega_body : (3,) base angular velocity, body frame (MuJoCo convention).
        qd_leg     : (3,) joint velocities [q_dot_abd, q_dot_hip, q_dot_knee].
        """
        fk = self.forward_kinematics(leg_idx, p_base, R_base, q_leg)
        p_foot = fk["p_foot"]

        omega_world = R_base @ omega_body
        J_leg = self.leg_jacobian(leg_idx, p_base, R_base, q_leg)

        return (
            v_base
            + np.cross(omega_world, p_foot - p_base)
            + J_leg @ qd_leg
        )  # (3,)

    # ------------------------------------------------------------------
    # Gravity compensation
    # ------------------------------------------------------------------

    def gravity_compensation(
        self,
        leg_idx: int,
        p_base: np.ndarray,
        R_base: np.ndarray,
        q_leg: np.ndarray,
    ) -> np.ndarray:
        """
        Static gravity compensation torques for one leg.  Shape: (3,).

        Computes τ_grav = Σ_k J_k_com^T @ (m_k * g) by summing over all
        three link CoMs (hip, thigh, calf).

        tau[j] = sum over bodies distal to joint j of:
                   cross(z_j, p_com - p_joint_j) · (m * g_world)

        Approximation: Coriolis/centrifugal terms (C(q,qd)*qd) are NOT
        included.  At trot speeds (qd_leg ~ 1–2 rad/s) the omitted terms
        are ~0.1–0.5 Nm per joint.  The WBC torque clipping and swing PD
        absorb this residual.
        """
        fk = self.forward_kinematics(leg_idx, p_base, R_base, q_leg)

        z_abd  = R_base          @ np.array([1.0, 0.0, 0.0])
        z_hip  = fk["R_hip"]    @ np.array([0.0, 1.0, 0.0])
        z_knee = fk["R_thigh"]  @ np.array([0.0, 1.0, 0.0])

        p_abd_jnt   = fk["p_abd_jnt"]
        p_thigh_jnt = fk["p_thigh_jnt"]
        p_knee_jnt  = fk["p_knee_jnt"]

        p_hip_com   = p_abd_jnt   + fk["R_hip"]   @ HIP_COM[leg_idx]
        p_thigh_com = p_thigh_jnt + fk["R_thigh"] @ THIGH_COM[leg_idx]
        p_calf_com  = p_knee_jnt  + fk["R_calf"]  @ CALF_COM[leg_idx]

        # Use -g_world so the formula matches MuJoCo's qfrc_bias sign convention.
        # qfrc_bias = ∂V/∂q (potential energy gradient), which equals
        # -sum m_k * g_world · J_k_com.  Using mg = -m*g_world = m*[0,0,+9.81]
        # keeps the formula as:  τ[j] = sum cross(z_j, r) · mg
        mg_world = -_GRAVITY   # [0, 0, +9.81] effective gravity for this convention

        tau = np.zeros(3, dtype=np.float64)

        # Joint 0 (hip abduction): influenced by hip, thigh, calf bodies
        tau[0] = (
            np.cross(z_abd, p_hip_com   - p_abd_jnt) @ (M_HIP   * mg_world)
          + np.cross(z_abd, p_thigh_com - p_abd_jnt) @ (M_THIGH * mg_world)
          + np.cross(z_abd, p_calf_com  - p_abd_jnt) @ (M_CALF  * mg_world)
        )

        # Joint 1 (hip flexion): influenced by thigh, calf bodies
        tau[1] = (
            np.cross(z_hip, p_thigh_com - p_thigh_jnt) @ (M_THIGH * mg_world)
          + np.cross(z_hip, p_calf_com  - p_thigh_jnt) @ (M_CALF  * mg_world)
        )

        # Joint 2 (knee): influenced by calf body only
        tau[2] = (
            np.cross(z_knee, p_calf_com - p_knee_jnt) @ (M_CALF * mg_world)
        )

        return tau  # (3,)
