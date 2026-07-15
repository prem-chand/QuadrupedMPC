"""
URDF-driven analytical kinematics for any 3-DOF-per-leg x 4-leg quadruped.

Public method surface mirrors go2_mpc.kinematics.go2_kinematics.Go2Kinematics
so the two are interchangeable behind the Robot interface.
"""

import numpy as np

from .urdf_parser import parse_urdf
from .kinematic_chain import build_leg_chains
from .transforms import axis_angle_to_matrix, skew

# Column indices into the (3, 18) full Jacobian for each leg's joint DOFs.
# qvel layout: [vx,vy,vz | wx_B,wy_B,wz_B | leg0(3) | leg1(3) | leg2(3) | leg3(3)]
_LEG_DOF_COLS = [
    [6, 7, 8],
    [9, 10, 11],
    [12, 13, 14],
    [15, 16, 17],
]


class GenericKinematics:
    """
    Analytical FK/Jacobians/gravity-comp for a robot described by a URDF.

    All methods are stateless -- each call takes the full leg state
    (p_base, R_base, q_leg) and returns the requested quantity, exactly
    like Go2Kinematics.
    """

    def __init__(self, urdf_path, base_link, foot_links, leg_order=None):
        self._tree = parse_urdf(urdf_path)
        ordered_foot_links = (
            list(foot_links) if leg_order is None
            else [foot_links[i] for i in leg_order]
        )
        self._chains = build_leg_chains(self._tree, base_link, ordered_foot_links)
        self._gravity = np.array([0.0, 0.0, -9.81])

    # ------------------------------------------------------------------
    # Forward kinematics
    # ------------------------------------------------------------------

    def forward_kinematics(self, leg_idx, p_base, R_base, q_leg):
        """
        Compute the full FK frame chain for one leg.

        Returns a dict with:
          joint_frames -- list of {"p": (3,), "R": (3,3), "z": (3,)} per
                           actuated joint, base->foot order. "p"/"R" are the
                           joint's position/cumulative rotation BEFORE its
                           own DOF rotation is applied; "z" is its rotation
                           axis expressed in the world frame.
          p_foot -- (3,) foot position in world frame.
        """
        chain = self._chains[leg_idx]

        T_current = np.eye(4)
        T_current[:3, :3] = R_base
        T_current[:3, 3] = p_base

        joint_frames = []
        for cj, qi in zip(chain.joints, q_leg):
            T_current = T_current @ cj.offset_transform
            p_i = T_current[:3, 3].copy()
            R_i = T_current[:3, :3].copy()
            z_i = R_i @ cj.axis

            R_qi = np.eye(4)
            R_qi[:3, :3] = axis_angle_to_matrix(cj.axis, qi)
            T_current = T_current @ R_qi

            joint_frames.append({"p": p_i, "R": R_i, "z": z_i})

        T_current = T_current @ chain.foot_offset_transform

        return {"joint_frames": joint_frames, "p_foot": T_current[:3, 3].copy()}

    def foot_position_world(self, leg_idx, p_base, R_base, q_leg):
        """Foot position in world frame. Shape: (3,)."""
        return self.forward_kinematics(leg_idx, p_base, R_base, q_leg)["p_foot"].copy()

    # ------------------------------------------------------------------
    # Jacobians
    # ------------------------------------------------------------------

    def leg_jacobian(self, leg_idx, p_base, R_base, q_leg):
        """
        Geometric Jacobian mapping leg joint velocities -> foot Cartesian
        velocity in world frame. Shape: (3, n_joints_in_leg).

        J[:, k] = z_k x (p_foot - p_joint_k)
        """
        fk = self.forward_kinematics(leg_idx, p_base, R_base, q_leg)
        p_foot = fk["p_foot"]
        joint_frames = fk["joint_frames"]

        J = np.empty((3, len(joint_frames)), dtype=np.float64)
        for k, jf in enumerate(joint_frames):
            J[:, k] = np.cross(jf["z"], p_foot - jf["p"])
        return J

    def full_jacobian(self, leg_idx, p_base, R_base, q_leg):
        """
        Full positional Jacobian mapping all 18 generalised velocities
        (6 floating-base DOFs + 12 leg joint DOFs) -> foot Cartesian
        velocity in world frame. Shape: (3, 18).
        """
        fk = self.forward_kinematics(leg_idx, p_base, R_base, q_leg)
        p_foot = fk["p_foot"]

        J_full = np.zeros((3, 18), dtype=np.float64)
        J_full[:, 0:3] = np.eye(3)

        r = p_foot - p_base
        J_full[:, 3:6] = -skew(r) @ R_base

        J_leg = self.leg_jacobian(leg_idx, p_base, R_base, q_leg)
        for k, col in enumerate(_LEG_DOF_COLS[leg_idx]):
            J_full[:, col] = J_leg[:, k]

        return J_full

    # ------------------------------------------------------------------
    # Foot velocity
    # ------------------------------------------------------------------

    def foot_velocity(self, leg_idx, p_base, R_base, q_leg, v_base, omega_body, qd_leg):
        """
        Foot Cartesian velocity in world frame. Shape: (3,).

        v_base     : (3,) base linear velocity, world frame.
        omega_body : (3,) base angular velocity, body frame.
        qd_leg     : joint velocities for this leg.
        """
        fk = self.forward_kinematics(leg_idx, p_base, R_base, q_leg)
        p_foot = fk["p_foot"]

        omega_world = R_base @ omega_body
        J_leg = self.leg_jacobian(leg_idx, p_base, R_base, q_leg)

        return v_base + np.cross(omega_world, p_foot - p_base) + J_leg @ qd_leg
