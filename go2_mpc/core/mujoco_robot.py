"""
MuJoCo backend implementation of the Robot interface.

This module wraps a MuJoCo ``MjModel`` / ``MjData`` pair and exposes
all quantities through the simulator-agnostic :class:`Robot` ABC.
Controller code never imports this module directly — it only sees the
abstract interface.

MuJoCo is used exclusively for:
  - ``mj_step``  — dynamics integration
  - ``data.qpos`` / ``data.qvel``  — state readout (sensor data)
  - ``data.ctrl``  — torque command write-back

All kinematics (foot positions, Jacobians, foot velocities) and the
gravity compensation term are computed analytically by
:class:`~go2_mpc.kinematics.Go2Kinematics`, with zero dependency on
MuJoCo's forward-kinematics outputs (``site_xpos``, ``mj_jacSite``,
``qfrc_bias``).

MuJoCo frame conventions used here
-----------------------------------
- **qpos[0:3]**  — base position in world frame (m).
- **qpos[3:7]**  — base orientation as quaternion ``[w, x, y, z]``
  (scalar-first, MuJoCo default).
- **qvel[0:3]**  — base linear velocity in **world frame** (m/s).
- **qvel[3:6]**  — base angular velocity in **body frame** (rad/s).
  This is the MuJoCo convention for free joints.
- **qpos[7:]**   — joint positions (rad), ordered by XML definition.
- **qvel[6:]**   — joint velocities (rad/s), same ordering.

Leg ordering: ``[FL, FR, RL, RR]``, each with 3 DOFs
``[hip_abduction, hip_flexion, knee_flexion]``.
"""

from .robot import Robot
import numpy as np
import mujoco
from go2_mpc.kinematics import Go2Kinematics


class MujocoRobot(Robot):
    """Concrete :class:`Robot` backed by MuJoCo.

    Parameters
    ----------
    model : mujoco.MjModel
        Compiled MuJoCo model (from XML or MJCF).
    data : mujoco.MjData
        Simulation data buffer associated with *model*.
    """

    def __init__(self, model, data):
        self.model = model
        self.data = data

        # Analytical kinematics — no MuJoCo kinematics API needed.
        self._kin = Go2Kinematics()

    # ==========================
    # Simulation
    # ==========================

    def step(self):
        """Advance the simulation by one timestep (``model.opt.timestep``)."""
        mujoco.mj_step(self.model, self.data)

    def get_time(self) -> float:
        """Current simulation time (s)."""
        return self.data.time

    def set_torques(self, tau):
        """Apply joint torques to all 12 actuators.

        Parameters
        ----------
        tau : np.ndarray, shape (12,)
            Desired torques (N·m), ordered ``[FL(3), FR(3), RL(3), RR(3)]``.
        """
        self.data.ctrl[:] = tau

    # ==========================
    # Base State
    # ==========================

    def get_base_pose(self):
        """Return base pose from ``qpos``.

        Returns
        -------
        position : np.ndarray, shape (3,)
            CoM position ``[x, y, z]`` in **world frame** (m).
        quaternion : np.ndarray, shape (4,)
            Orientation ``[w, x, y, z]`` (scalar-first).
        """
        qpos = self.data.qpos
        return qpos[:3].copy(), qpos[3:7].copy()

    def get_base_velocity(self):
        """Return base velocities from ``qvel``.

        Returns
        -------
        linear_velocity : np.ndarray, shape (3,)
            Translational velocity in **world frame** (m/s).
        angular_velocity : np.ndarray, shape (3,)
            Rotational velocity in **body frame** (rad/s).
            This is the MuJoCo convention for free-joint ``qvel[3:6]``.
        """
        qvel = self.data.qvel
        return qvel[0:3].copy(), qvel[3:6].copy()

    # ==========================
    # Joints
    # ==========================

    def get_joint_state(self):
        """Return joint positions and velocities for all 12 actuated DOFs.

        Returns
        -------
        positions : np.ndarray, shape (12,)
            Joint angles (rad) from ``qpos[7:]``.
        velocities : np.ndarray, shape (12,)
            Joint velocities (rad/s) from ``qvel[6:]``.
        """
        return self.data.qpos[7:].copy(), self.data.qvel[6:].copy()

    # ==========================
    # Private helpers
    # ==========================

    def _get_base_state(self):
        """Extract (p_base, R_base, v_base, omega_body) from qpos/qvel.

        Computes the rotation matrix inline from the quaternion to avoid
        a redundant ``get_base_pose()`` call and associated copy overhead.
        """
        qpos = self.data.qpos
        qvel = self.data.qvel

        p_base = qpos[0:3]
        w, x, y, z = qpos[3:7]
        # Normalise for numerical safety
        n = np.sqrt(w*w + x*x + y*y + z*z)
        w, x, y, z = w/n, x/n, y/n, z/n
        R_base = np.array([
            [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
            [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
            [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)],
        ], dtype=np.float64)
        v_base     = qvel[0:3]
        omega_body = qvel[3:6]
        return p_base, R_base, v_base, omega_body

    # ==========================
    # Feet — analytical kinematics
    # ==========================

    def get_foot_positions_world(self):
        """Foot positions via analytical forward kinematics.

        Returns
        -------
        list of np.ndarray, each shape (3,)
            ``[FL, FR, RL, RR]`` foot positions in **world frame** (m).
        """
        p_base, R_base, _, _ = self._get_base_state()
        q_joints = self.data.qpos[7:]
        return [
            self._kin.foot_position_world(i, p_base, R_base, q_joints[3*i:3*i+3])
            for i in range(4)
        ]

    def get_foot_jacobian(self, foot_index):
        """Full positional Jacobian for a foot (analytically computed).

        Maps generalised velocities ``qvel`` (18-dim) to the foot's
        Cartesian (translational) velocity in **world frame**:
        ``v_foot_W = J @ qvel``.

        Parameters
        ----------
        foot_index : int
            Leg index in ``[0=FL, 1=FR, 2=RL, 3=RR]``.

        Returns
        -------
        np.ndarray, shape (3, 18)
            Full positional Jacobian in **world frame**.
        """
        p_base, R_base, _, _ = self._get_base_state()
        q_joints = self.data.qpos[7:]
        return self._kin.full_jacobian(
            foot_index, p_base, R_base, q_joints[3*foot_index:3*foot_index+3]
        )

    def get_leg_jacobian(self, foot_index):
        """Leg-local Jacobian: 3 joint DOFs → foot Cartesian velocity.

        Used by WBC and swing control for the torque mapping
        ``tau_leg = J_leg^T @ F_foot``.

        Parameters
        ----------
        foot_index : int
            Leg index in ``[0=FL, 1=FR, 2=RL, 3=RR]``.

        Returns
        -------
        np.ndarray, shape (3, 3)
            Jacobian in **world frame**.
        """
        p_base, R_base, _, _ = self._get_base_state()
        q_joints = self.data.qpos[7:]
        return self._kin.leg_jacobian(
            foot_index, p_base, R_base, q_joints[3*foot_index:3*foot_index+3]
        )

    def get_foot_velocity(self, foot_index):
        """Cartesian velocity of a foot in world frame (analytically computed).

        Includes base translational, base rotational, and joint velocity
        contributions:
        ``v_foot = v_base + ω_W × (p_foot − p_base) + J_leg @ q̇_leg``

        Parameters
        ----------
        foot_index : int
            Leg index in ``[0=FL, 1=FR, 2=RL, 3=RR]``.

        Returns
        -------
        np.ndarray, shape (3,)
            Translational velocity in **world frame** (m/s).
        """
        p_base, R_base, v_base, omega_body = self._get_base_state()
        q_joints  = self.data.qpos[7:]
        qd_joints = self.data.qvel[6:]
        return self._kin.foot_velocity(
            foot_index,
            p_base, R_base,
            q_joints[3*foot_index:3*foot_index+3],
            v_base,
            omega_body,
            qd_joints[3*foot_index:3*foot_index+3],
        )

    def get_gravity_compensation(self, leg_index):
        """Static gravity compensation torques for one leg.

        Computed analytically from the link CoM positions and masses
        (see :meth:`~go2_mpc.kinematics.Go2Kinematics.gravity_compensation`).

        Approximation: Coriolis/centrifugal terms excluded.  Residual
        vs. MuJoCo ``qfrc_bias`` is ~0.1–0.5 Nm at trot speeds.

        Parameters
        ----------
        leg_index : int
            Leg index in ``[0=FL, 1=FR, 2=RL, 3=RR]``.

        Returns
        -------
        np.ndarray, shape (3,)
            Bias torques (N·m) for ``[hip_abd, hip_flex, knee]``.
        """
        p_base, R_base, _, _ = self._get_base_state()
        q_joints = self.data.qpos[7:]
        return self._kin.gravity_compensation(
            leg_index, p_base, R_base, q_joints[3*leg_index:3*leg_index+3]
        )

    # ==========================
    # Legacy
    # ==========================

    def get_foot_jacobian_full(self, foot_index):
        """Deprecated: use ``get_foot_jacobian`` or ``get_leg_jacobian``."""
        return self.get_foot_jacobian(foot_index)
