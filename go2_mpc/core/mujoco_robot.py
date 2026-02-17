"""
MuJoCo backend implementation of the Robot interface.

This module wraps a MuJoCo ``MjModel`` / ``MjData`` pair and exposes
all quantities through the simulator-agnostic :class:`Robot` ABC.
Controller code never imports this module directly — it only sees the
abstract interface.

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
- **qfrc_bias**  — Coriolis + gravitational forces in generalised
  coordinates (N·m for joints, N for base translational DOFs).

Leg ordering: ``[FL, FR, RL, RR]``, each with 3 DOFs
``[hip_abduction, hip_flexion, knee_flexion]``.
"""

from .robot import Robot
import numpy as np
import mujoco


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

        # Foot contact sites defined in the XML
        self.foot_names = ["FL_toe", "FR_toe", "RL_toe", "RR_toe"]
        self.foot_ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
            for name in self.foot_names
        ]

        # Indices into qvel for each leg's 3 actuated joints.
        # The first 6 qvel entries belong to the floating base.
        self.leg_dof_indices = [
            [6, 7, 8],      # FL: hip_abd, hip_flex, knee
            [9, 10, 11],    # FR
            [12, 13, 14],   # RL
            [15, 16, 17],   # RR
        ]

        # Pre-allocated Jacobian buffers to avoid per-call allocation.
        # _J: positional (translational) Jacobian, shape (3, nv)
        # _Jr: rotational Jacobian, shape (3, nv)  (unused but required by API)
        self._J = np.zeros((3, model.nv))
        self._Jr = np.zeros((3, model.nv))

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
    # Feet
    # ==========================

    def get_foot_positions_world(self):
        """Foot positions from MuJoCo site sensors.

        Returns
        -------
        list of np.ndarray, each shape (3,)
            ``[FL, FR, RL, RR]`` foot positions in **world frame** (m).
            Each array is a copy (safe to mutate).
        """
        return [
            self.data.site_xpos[self.foot_ids[i]].copy()
            for i in range(4)
        ]

    def get_foot_jacobian(self, foot_index):
        """Full positional Jacobian for a foot site.

        Computed via ``mj_jacSite``.  Maps generalised velocities to
        the foot's Cartesian (translational) velocity in **world frame**:
        ``v_foot_W = J @ qvel``.

        Parameters
        ----------
        foot_index : int
            Leg index in ``[0=FL, 1=FR, 2=RL, 3=RR]``.

        Returns
        -------
        np.ndarray, shape (3, nv)
            Positional Jacobian in **world frame**.  Returned as a copy.
        """
        mujoco.mj_jacSite(
            self.model,
            self.data,
            self._J,
            self._Jr,
            self.foot_ids[foot_index],
        )
        return self._J.copy()

    def get_leg_jacobian(self, foot_index):
        """Leg-local Jacobian: 3 joint DOFs → foot Cartesian velocity.

        Extracts the 3 columns of the full site Jacobian corresponding
        to this leg's actuated joints.  Used by WBC and swing control
        for the torque mapping ``tau_leg = J_leg^T @ F_foot``.

        Parameters
        ----------
        foot_index : int
            Leg index in ``[0=FL, 1=FR, 2=RL, 3=RR]``.

        Returns
        -------
        np.ndarray, shape (3, 3)
            Jacobian block in **world frame**.  Returned as a copy.
        """
        mujoco.mj_jacSite(
            self.model,
            self.data,
            self._J,
            self._Jr,
            self.foot_ids[foot_index],
        )
        dofs = self.leg_dof_indices[foot_index]
        return self._J[:, dofs].copy()

    def get_foot_velocity(self, foot_index):
        """Cartesian velocity of a foot in world frame.

        Computed as ``J_full @ qvel`` (includes base and joint
        contributions).

        Parameters
        ----------
        foot_index : int
            Leg index in ``[0=FL, 1=FR, 2=RL, 3=RR]``.

        Returns
        -------
        np.ndarray, shape (3,)
            Translational velocity in **world frame** (m/s).
        """
        mujoco.mj_jacSite(
            self.model,
            self.data,
            self._J,
            self._Jr,
            self.foot_ids[foot_index],
        )
        return self._J @ self.data.qvel

    def get_gravity_compensation(self, leg_index):
        """Gravity + Coriolis bias torques for one leg.

        Extracted from MuJoCo's ``qfrc_bias`` which contains the
        full C(q,qd)*qd + g(q) vector in generalised coordinates.

        Parameters
        ----------
        leg_index : int
            Leg index in ``[0=FL, 1=FR, 2=RL, 3=RR]``.

        Returns
        -------
        np.ndarray, shape (3,)
            Bias torques (N·m) for ``[hip_abd, hip_flex, knee]``.
            Returned as a copy.
        """
        dofs = self.leg_dof_indices[leg_index]
        return self.data.qfrc_bias[dofs].copy()

    # ==========================
    # Legacy
    # ==========================

    def get_foot_jacobian_full(self, foot_index):
        """Deprecated: use ``get_foot_jacobian`` or ``get_leg_jacobian``."""
        return self.get_foot_jacobian(foot_index)
